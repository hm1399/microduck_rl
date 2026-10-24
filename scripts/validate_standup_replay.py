#!/usr/bin/env python3
"""Deterministic, headless acceptance replay for a stand-up policy.

The suite uses the same CPU BAM M6 controller and observation path as
``scripts/infer_policy.py``.  Each case has a fixed preparation interval, one
fixed velocity disturbance, a complete recovery window, and a final hold
window.  A case only passes when the robot is valid for the whole hold window.

Example::

    uv run python scripts/validate_standup_replay.py \
        --policy ../models/v6.1-standup/policy.onnx \
        --json /tmp/v61-standup-replay.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from infer_policy import (
    DEFAULT_POSE,
    MICRODUCK_XML,
    PolicyInference,
    load_bam_model,
    load_mujoco_with_bam,
)


CONTROL_DT = 0.02  # 4 x 0.005 s, matching infer_policy.py
DECIMATION = 4
DEFAULT_PREPARE_S = 1.0
DEFAULT_RECOVERY_S = 5.0
DEFAULT_HOLD_S = 5.0
DEFAULT_PUSH_SPEED = 0.25
ACCEPTANCE_PUSH_SPEEDS = (0.25, 0.40)


@dataclass(frozen=True)
class ReplayCase:
    name: str
    push_xy: tuple[float, float] | None
    push_speed: float | None = None


CASES = (
    ReplayCase("nominal_no_disturbance", None),
    ReplayCase("front_push_025", (1.0, 0.0), 0.25),
    ReplayCase("front_left_push_025", (1.0, 1.0), 0.25),
    ReplayCase("left_push_025", (0.0, 1.0), 0.25),
    ReplayCase("back_left_push_025", (-1.0, 1.0), 0.25),
    ReplayCase("back_push_025", (-1.0, 0.0), 0.25),
    ReplayCase("back_right_push_025", (-1.0, -1.0), 0.25),
    ReplayCase("right_push_025", (0.0, -1.0), 0.25),
    ReplayCase("front_right_push_025", (1.0, -1.0), 0.25),
    ReplayCase("front_push_040", (1.0, 0.0), 0.40),
    ReplayCase("front_left_push_040", (1.0, 1.0), 0.40),
    ReplayCase("left_push_040", (0.0, 1.0), 0.40),
    ReplayCase("back_left_push_040", (-1.0, 1.0), 0.40),
    ReplayCase("back_push_040", (-1.0, 0.0), 0.40),
    ReplayCase("back_right_push_040", (-1.0, -1.0), 0.40),
    ReplayCase("right_push_040", (0.0, -1.0), 0.40),
    ReplayCase("front_right_push_040", (1.0, -1.0), 0.40),
)


def validate_windows(prepare_s: float, recovery_s: float, hold_s: float) -> None:
    """Reject an acceptance setup that cannot contain all required phases."""
    if not all(math.isfinite(value) for value in (prepare_s, recovery_s, hold_s)):
        raise ValueError("prepare/recovery/hold windows must be finite")
    if min(prepare_s, recovery_s, hold_s) <= 0:
        raise ValueError("prepare/recovery/hold windows must be positive")
    for name, duration in (
        ("prepare", prepare_s),
        ("recovery", recovery_s),
        ("hold", hold_s),
    ):
        steps = int(round(duration / CONTROL_DT))
        if steps <= 0 or not math.isclose(
            steps * CONTROL_DT, duration, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{name} window must be an integer number of control steps")


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"MuJoCo scene is missing body {name!r}")
    return int(body_id)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"MuJoCo scene is missing geom {name!r}")
    return int(geom_id)


def _contacts_floor(data: mujoco.MjData, floor_id: int, geom_ids: set[int]) -> bool:
    """Return whether any geom in ``geom_ids`` touches the floor this step."""
    for i in range(data.ncon):
        contact = data.contact[i]
        if (int(contact.geom1) == floor_id and int(contact.geom2) in geom_ids) or (
            int(contact.geom2) == floor_id and int(contact.geom1) in geom_ids
        ):
            return True
    return False


def _tilt_deg(data: mujoco.MjData, trunk_id: int) -> float:
    quat = data.xquat[trunk_id]
    cos_tilt = 1.0 - 2.0 * (float(quat[1]) ** 2 + float(quat[2]) ** 2)
    return math.degrees(math.acos(float(np.clip(cos_tilt, -1.0, 1.0))))


def _make_case_result(name: str) -> dict[str, Any]:
    return {
        "case": name,
        "push_speed_mps": None,
        "pass": False,
        "first_recovery_s": None,
        "first_head_contact_s": None,
        "final_height_m": None,
        "final_tilt_deg": None,
        "final_neck_pitch_error_deg": None,
        "final_head_pitch_error_deg": None,
        "both_feet_contact_ratio_hold": 0.0,
        "head_contact_ratio_hold": 0.0,
        "failure_reasons": [],
    }


class StandupReplay:
    """One deterministic MuJoCo/BAM replay suite."""

    def __init__(
        self,
        policy_path: Path,
        scene_path: Path,
        vin: float,
        vin_drop_gain: float,
        kp_fw: float,
        action_scale: float,
        prepare_s: float,
        recovery_s: float,
        hold_s: float,
        push_speed: float,
    ) -> None:
        bam_model = load_bam_model(kp_fw, vin, None)
        self.model, self.data, self.bam_ctrl, _ = load_mujoco_with_bam(
            str(scene_path), bam_model, 0.005, vin_drop_gain, 6.0
        )
        self.policy = PolicyInference(
            self.model,
            self.data,
            standing_onnx_path=str(policy_path),
            action_scale=action_scale,
            bam_ctrl=self.bam_ctrl,
            new_cmd_obs=True,
        )
        self.policy.set_vel_cmd(0.0, 0.0, 0.0)

        self.prepare_s = prepare_s
        self.recovery_s = recovery_s
        self.hold_s = hold_s
        self.push_speed = push_speed
        self.total_s = prepare_s + recovery_s + hold_s
        validate_windows(prepare_s, recovery_s, hold_s)

        self.trunk_body_id = _body_id(self.model, "trunk_base")
        self.floor_geom_id = _geom_id(self.model, "floor")
        self.left_foot_geom_id = _geom_id(self.model, "left_foot_collision")
        self.right_foot_geom_id = _geom_id(self.model, "right_foot_collision")
        head_body_names = {"neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"}
        self.head_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            )
            in head_body_names
        }
        if not self.head_geom_ids:
            raise RuntimeError("MuJoCo scene has no collision geoms for head contact checks")

        freejoint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        if freejoint_id < 0:
            raise RuntimeError("MuJoCo scene is missing trunk_base_freejoint")
        self.trunk_qpos_adr = int(self.model.jnt_qposadr[freejoint_id])
        self.trunk_qvel_adr = int(self.model.jnt_dofadr[freejoint_id])

        self.neck_pitch_qpos = self.policy.joint_qpos_indices[5]
        self.head_pitch_qpos = self.policy.joint_qpos_indices[6]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.trunk_qpos_adr : self.trunk_qpos_adr + 3] = (0.0, 0.0, 0.125)
        self.data.qpos[self.trunk_qpos_adr + 3 : self.trunk_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        for i, qpos_idx in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qpos_idx] = DEFAULT_POSE[i]
        self.policy.last_action.fill(0.0)
        self.policy.vel_cmd[:] = 0.0
        self.policy.head_offset[:] = 0.0
        self.policy.body_cmd[:] = 0.0
        self.policy.current_policy = "standing"
        self.policy.ort_session = self.policy.standing_session
        self.policy._update_command()
        self.bam_ctrl.reset(self.data.qpos)
        self.policy.set_position_targets(self.policy.default_pose)
        mujoco.mj_forward(self.model, self.data)

    def _snapshot(self) -> dict[str, float | bool]:
        feet = {
            self.left_foot_geom_id,
            self.right_foot_geom_id,
        }
        left = _contacts_floor(self.data, self.floor_geom_id, {self.left_foot_geom_id})
        right = _contacts_floor(self.data, self.floor_geom_id, {self.right_foot_geom_id})
        head = _contacts_floor(self.data, self.floor_geom_id, self.head_geom_ids)
        height = float(self.data.xpos[self.trunk_body_id, 2])
        tilt = _tilt_deg(self.data, self.trunk_body_id)
        neck_error = abs(float(self.data.qpos[self.neck_pitch_qpos] - DEFAULT_POSE[5]))
        head_error = abs(float(self.data.qpos[self.head_pitch_qpos] - DEFAULT_POSE[6]))
        # A single scalar captures the actual acceptance contract at this step.
        valid = (
            height >= 0.10
            and tilt <= 15.0
            and left
            and right
            and not head
            and neck_error <= math.radians(30.0)
            and head_error <= math.radians(30.0)
        )
        return {
            "height_m": height,
            "tilt_deg": tilt,
            "left_foot_contact": left,
            "right_foot_contact": right,
            "both_feet_contact": left and right,
            "head_contact": head,
            "neck_pitch_error_deg": math.degrees(neck_error),
            "head_pitch_error_deg": math.degrees(head_error),
            "valid": valid,
        }

    def _step(self) -> None:
        action = self.policy.infer()
        self.policy.apply_action(action)
        for _ in range(DECIMATION):
            self.bam_ctrl.update()
            mujoco.mj_step(self.model, self.data)

    def run_case(self, case: ReplayCase) -> dict[str, Any]:
        self.reset()
        result = _make_case_result(case.name)
        n_steps = int(round(self.total_s / CONTROL_DT))
        push_step = int(round(self.prepare_s / CONTROL_DT))
        recovery_end_step = push_step + int(round(self.recovery_s / CONTROL_DT))
        hold_end_step = recovery_end_step + int(round(self.hold_s / CONTROL_DT))
        samples: list[dict[str, float | bool]] = []
        push_applied = case.push_xy is None
        case_push_speed = self.push_speed if case.push_speed is None else case.push_speed
        if case_push_speed < 0:
            raise ValueError("case push speed must be non-negative")
        result["push_speed_mps"] = None if case.push_xy is None else case_push_speed

        for step in range(n_steps + 1):
            if not push_applied and step >= push_step:
                direction = np.asarray(case.push_xy, dtype=np.float64)
                direction /= np.linalg.norm(direction)
                self.data.qvel[self.trunk_qvel_adr : self.trunk_qvel_adr + 2] = direction * case_push_speed
                push_applied = True
            sample = self._snapshot()
            samples.append(sample)
            if sample["head_contact"] and result["first_head_contact_s"] is None:
                result["first_head_contact_s"] = round(step * CONTROL_DT, 3)
            if step < n_steps:
                self._step()

        recovery_samples = samples[push_step : recovery_end_step + 1]
        hold_samples = samples[recovery_end_step : hold_end_step + 1]
        valid_recovery_offsets = [
            i for i, sample in enumerate(recovery_samples) if sample["valid"]
        ]
        if valid_recovery_offsets:
            result["first_recovery_s"] = round(
                (push_step + valid_recovery_offsets[0]) * CONTROL_DT, 3
            )
        hold_valid = [bool(sample["valid"]) for sample in hold_samples]
        result["both_feet_contact_ratio_hold"] = round(
            sum(bool(sample["both_feet_contact"]) for sample in hold_samples) / len(hold_samples),
            3,
        )
        result["head_contact_ratio_hold"] = round(
            sum(bool(sample["head_contact"]) for sample in hold_samples) / len(hold_samples),
            3,
        )
        final = samples[-1]
        result["final_height_m"] = round(float(final["height_m"]), 5)
        result["final_tilt_deg"] = round(float(final["tilt_deg"]), 3)
        result["final_neck_pitch_error_deg"] = round(
            float(final["neck_pitch_error_deg"]), 3
        )
        result["final_head_pitch_error_deg"] = round(
            float(final["head_pitch_error_deg"]), 3
        )

        reasons: list[str] = []
        if not valid_recovery_offsets:
            reasons.append("never reached standing constraints in recovery window")
        if not all(hold_valid):
            reasons.append("standing constraints were not continuous for the hold window")
        if any(bool(sample["head_contact"]) for sample in hold_samples):
            reasons.append("head contacted the floor during hold window")
        if result["both_feet_contact_ratio_hold"] < 1.0:
            reasons.append("both feet were not continuously supported during hold window")
        result["failure_reasons"] = reasons
        result["pass"] = not reasons
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="standing policy.onnx")
    parser.add_argument("--scene", type=Path, default=Path(MICRODUCK_XML))
    parser.add_argument("--json", type=Path, default=None, help="write machine-readable results")
    parser.add_argument("--prepare-s", type=float, default=DEFAULT_PREPARE_S)
    parser.add_argument("--recovery-s", type=float, default=DEFAULT_RECOVERY_S)
    parser.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S)
    parser.add_argument("--push-speed", type=float, default=DEFAULT_PUSH_SPEED)
    parser.add_argument("--vin", type=float, default=7.4)
    parser.add_argument("--vin-drop-gain", type=float, default=0.1)
    parser.add_argument("--kp-fw", type=float, default=200.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        validate_windows(args.prepare_s, args.recovery_s, args.hold_s)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.push_speed < 0:
        raise SystemExit("push speed must be non-negative")
    replay = StandupReplay(
        policy_path=args.policy,
        scene_path=args.scene,
        vin=args.vin,
        vin_drop_gain=args.vin_drop_gain,
        kp_fw=args.kp_fw,
        action_scale=args.action_scale,
        prepare_s=args.prepare_s,
        recovery_s=args.recovery_s,
        hold_s=args.hold_s,
        push_speed=args.push_speed,
    )
    print(
        f"Replay suite: {len(CASES)} cases, prepare={args.prepare_s:.1f}s, "
        f"recovery={args.recovery_s:.1f}s, hold={args.hold_s:.1f}s, "
        f"push={args.push_speed:.2f}m/s"
    )
    results = []
    for case in CASES:
        result = replay.run_case(case)
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        reasons = "; ".join(result["failure_reasons"]) or "-"
        print(
            f"{status:4} {case.name:24} "
            f"push={result['push_speed_mps'] if result['push_speed_mps'] is not None else 0.0:.2f}m/s "
            f"recover={result['first_recovery_s']!s:>6}s "
            f"head_contact={result['first_head_contact_s']!s:>6}s "
            f"final_tilt={result['final_tilt_deg']:.1f}deg "
            f"final_neck_err={result['final_neck_pitch_error_deg']:.1f}deg "
            f"feet_hold={result['both_feet_contact_ratio_hold']:.0%}"
        )
        if reasons != "-":
            print(f"     reason: {reasons}")

    payload = {
        "policy": str(args.policy),
        "scene": str(args.scene),
        "control_dt_s": CONTROL_DT,
        "prepare_s": args.prepare_s,
        "recovery_s": args.recovery_s,
        "hold_s": args.hold_s,
        "push_speed_mps": args.push_speed,
        "cases": results,
        "pass": all(bool(result["pass"]) for result in results),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
        print(f"Wrote {args.json}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
