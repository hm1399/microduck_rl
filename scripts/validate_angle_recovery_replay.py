#!/usr/bin/env python3
"""Regression battery for catastrophic forgetting across fall-angle bands."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from validate_ground_recovery_replay import run_case
from validate_standup_replay import MICRODUCK_XML, StandupReplay


BANDS = ((5.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 90.0))


def quat_yaw_tilt(yaw: float, tilt_deg: float, family: str, sign: float = 1.0) -> tuple[float, float, float, float]:
    ct = math.cos(math.radians(tilt_deg) / 2.0)
    st = math.sin(math.radians(tilt_deg) / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    if family == "face_down":
        return (ct * cy, -st * sy, st * cy, ct * sy)
    if family == "face_up":
        return (ct * cy, st * sy, -st * cy, ct * sy)
    return (ct * cy, sign * st * cy, sign * st * sy, ct * sy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path(MICRODUCK_XML))
    parser.add_argument("--cases-per-band", type=int, default=4)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.cases_per_band <= 0:
        raise SystemExit("cases-per-band must be positive")
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
    rng = np.random.default_rng(args.seed)
    payload_bands = {}
    for low, high in BANDS:
        results = []
        for family in ["face_down", "face_up", "side"]:
            for index in range(args.cases_per_band):
                yaw = float(rng.uniform(-math.pi, math.pi))
                tilt = float(rng.uniform(low, high))
                sign = 1.0 if index % 2 == 0 else -1.0
                result = run_case(replay, quat_yaw_tilt(yaw, tilt, family, sign))
                results.append({"family": family, "tilt_deg": tilt, **result})
        band = f"{int(low)}-{int(high)}deg"
        passed = sum(bool(result["pass"]) for result in results)
        payload_bands[band] = {
            "passed": passed,
            "total": len(results),
            "pass_rate": passed / len(results),
            "cases": results,
        }
        print(f"{band:10} {passed}/{len(results)}")
    total = sum(band["total"] for band in payload_bands.values())
    passed = sum(band["passed"] for band in payload_bands.values())
    payload = {
        "policy": str(args.policy),
        "seed": args.seed,
        "bands": payload_bands,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total,
        "pass": passed == total,
    }
    print(f"TOTAL {passed}/{total} ({passed / total:.1%})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
