"""Tests for the fixed stand-up acceptance suite."""

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _module():
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "validate_standup_replay", REPO / "scripts" / "validate_standup_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_has_all_fixed_directions_and_windows():
    mod = _module()
    assert [case.name for case in mod.CASES] == [
        "nominal_no_disturbance",
        "front_push_025",
        "front_left_push_025",
        "left_push_025",
        "back_left_push_025",
        "back_push_025",
        "back_right_push_025",
        "right_push_025",
        "front_right_push_025",
        "front_push_040",
        "front_left_push_040",
        "left_push_040",
        "back_left_push_040",
        "back_push_040",
        "back_right_push_040",
        "right_push_040",
        "front_right_push_040",
    ]
    assert mod.CONTROL_DT == 0.02
    assert mod.CASES[0].push_speed is None
    assert {case.push_speed for case in mod.CASES[1:]} == {0.25, 0.40}
    assert mod.DEFAULT_RECOVERY_S == 5.0
    mod.validate_windows(1.0, 5.0, 5.0)


def test_contact_and_tilt_helpers_use_scene_geometry():
    mod = _module()
    model = mujoco.MjModel.from_xml_path(str(REPO / mod.MICRODUCK_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mod._geom_id(model, "floor")
    foot = mod._geom_id(model, "left_foot_collision")
    assert isinstance(mod._contacts_floor(data, floor, {foot}), bool)
    trunk = mod._body_id(model, "trunk_base")
    assert np.isfinite(mod._tilt_deg(data, trunk))
