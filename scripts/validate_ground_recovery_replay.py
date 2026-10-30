#!/usr/bin/env python3
"""Evaluate true face-down, face-up, and side-ground recovery for a policy.

Each case starts on the floor with zero velocity and HOME joints. Recovery must
be confirmed for 0.4 s within the first 5 s, followed by 5 uninterrupted seconds
of strict standing. There is no standing-start push in this suite.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from validate_standup_replay import CONTROL_DT, MICRODUCK_XML, StandupReplay


CASES = {
    "face_down": (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
    "face_up": (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
    "left_side": (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
    "right_side": (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
}


def run_case(
    replay: StandupReplay,
    quat: tuple[float, float, float, float],
    joint_angles: np.ndarray | None = None,
) -> dict:
    replay.reset()
    replay.data.qpos[replay.trunk_qpos_adr + 2] = 0.08
    replay.data.qpos[replay.trunk_qpos_adr + 3 : replay.trunk_qpos_adr + 7] = quat
    if joint_angles is not None:
        for index, qpos_idx in enumerate(replay.policy.joint_qpos_indices):
            replay.data.qpos[qpos_idx] = joint_angles[index]
    replay.data.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(replay.model, replay.data)

    steps_5s = round(5.0 / CONTROL_DT)
    samples = []
    for _ in range(2 * steps_5s + 1):
        samples.append(replay._snapshot())
        replay._step()

    streak = 0
    first_recovery_s = None
    for index, sample in enumerate(samples[: steps_5s + 1]):
        streak = streak + 1 if sample["valid"] else 0
        if streak >= round(0.4 / CONTROL_DT):
            first_recovery_s = round((index + 1) * CONTROL_DT - 0.4, 3)
            break
    hold = samples[steps_5s:]
    hold_valid = all(bool(sample["valid"]) for sample in hold)
    final = samples[-1]
    return {
        "first_recovery_s": first_recovery_s,
        "hold_valid": hold_valid,
        "pass": first_recovery_s is not None and hold_valid,
        "final_height_m": round(float(final["height_m"]), 5),
        "final_tilt_deg": round(float(final["tilt_deg"]), 3),
        "final_neck_pitch_error_deg": round(float(final["neck_pitch_error_deg"]), 3),
        "head_contact_ratio_hold": round(
            sum(bool(sample["head_contact"]) for sample in hold) / len(hold), 3
        ),
        "both_feet_ratio_hold": round(
            sum(bool(sample["both_feet_contact"]) for sample in hold) / len(hold), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path(MICRODUCK_XML))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--random-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-joints", action="store_true")
    args = parser.parse_args()
    if args.random_cases < 0:
        raise SystemExit("random-cases must be >= 0")
    replay = StandupReplay(
        policy_path=args.policy,
        scene_path=args.scene,
        vin=7.4,
        vin_drop_gain=0.1,
        kp_fw=200.0,
        action_scale=1.0,
        prepare_s=1.0,
        recovery_s=5.0,
        hold_s=5.0,
        push_speed=0.0,
    )
    cases = [(name, (quat, None)) for name, quat in CASES.items()]
    rng = np.random.default_rng(args.seed)
    for index in range(args.random_cases):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        joint_angles = None
        if args.random_joints:
            joint_angles = replay.policy.default_pose.copy()
            for joint_index, qpos_idx in enumerate(replay.policy.joint_qpos_indices):
                joint_id = int(replay.model.actuator_trnid[joint_index, 0])
                low, high = replay.model.jnt_range[joint_id]
                if np.isfinite(low) and np.isfinite(high):
                    joint_angles[joint_index] = rng.uniform(low, high)
        cases.append((f"so3_{index:03d}", (tuple(float(x) for x in quat), joint_angles)))

    results = {}
    for name, (quat, joint_angles) in cases:
        result = run_case(replay, quat, joint_angles=joint_angles)
        results[name] = result
        print(
            f"{'PASS' if result['pass'] else 'FAIL':4} {name:12} "
            f"recover={result['first_recovery_s']}s "
            f"final_tilt={result['final_tilt_deg']:.1f}deg "
            f"head_hold={result['head_contact_ratio_hold']:.0%} "
            f"feet_hold={result['both_feet_ratio_hold']:.0%}"
        )
    payload = {
        "policy": str(args.policy),
        "seed": args.seed,
        "random_cases": args.random_cases,
        "random_joints": args.random_joints,
        "cases": results,
        "passed": sum(bool(result["pass"]) for result in results.values()),
        "total": len(results),
        "pass": all(bool(result["pass"]) for result in results.values()),
    }
    print(f"TOTAL: {payload['passed']}/{payload['total']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
