#!/usr/bin/env python3
"""Deterministic zero-command standing acceptance for v4-stand policies.

A policy passes only when every post-settle sample satisfies the strict standing
predicate: height, tilt, both feet, head clear, neck error, and bounded velocity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_standup_replay import CONTROL_DT, MICRODUCK_XML, StandupReplay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path(MICRODUCK_XML))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.seconds <= 0 or args.settle_s < 0 or args.settle_s >= args.seconds:
        raise SystemExit("require seconds > 0 and 0 <= settle-s < seconds")

    replay = StandupReplay(
        policy_path=args.policy,
        scene_path=args.scene,
        vin=7.4,
        vin_drop_gain=0.1,
        kp_fw=200.0,
        action_scale=1.0,
        prepare_s=1.0,
        recovery_s=1.0,
        hold_s=1.0,
        push_speed=0.0,
    )
    replay.reset()
    steps = round(args.seconds / CONTROL_DT)
    settle_steps = round(args.settle_s / CONTROL_DT)
    samples = []
    for _ in range(steps + 1):
        samples.append(replay._snapshot())
        replay._step()
    evaluated = samples[settle_steps:]
    valid = [bool(sample["valid"]) for sample in evaluated]
    head_contacts = [bool(sample["head_contact"]) for sample in evaluated]
    both_feet = [bool(sample["both_feet_contact"]) for sample in evaluated]
    result = {
        "policy": str(args.policy),
        "seconds": args.seconds,
        "settle_s": args.settle_s,
        "valid_fraction": sum(valid) / len(valid),
        "both_feet_fraction": sum(both_feet) / len(both_feet),
        "head_contact_fraction": sum(head_contacts) / len(head_contacts),
        "min_height_m": min(float(sample["height_m"]) for sample in evaluated),
        "max_tilt_deg": max(float(sample["tilt_deg"]) for sample in evaluated),
        "max_neck_pitch_error_deg": max(float(sample["neck_pitch_error_deg"]) for sample in evaluated),
        "max_head_pitch_error_deg": max(float(sample["head_pitch_error_deg"]) for sample in evaluated),
    }
    result["pass"] = all(valid) and not any(head_contacts) and all(both_feet)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
