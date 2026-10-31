import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    V66_GROUND_STATE_STAGES,
    MicroduckV66StandUpRlCfg,
    make_microduck_v66_standup_env_cfg,
)


def test_v66_angle_curriculum_rehearses_all_previous_ranges():
    cfg = make_microduck_v66_standup_env_cfg()
    assert make_microduck_v66_standup_env_cfg(rough=True) is not None
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert stages == V66_GROUND_STATE_STAGES
    max_angles = [stage["params"]["tilt_max_deg"] for stage in stages]
    assert max_angles == [15.0, 30.0, 60.0, 90.0, 90.0, 90.0]
    assert all(stage["params"]["tilt_min_deg"] == 5.0 for stage in stages)
    assert all(stage["params"]["tilt_aware_z"] for stage in stages)
    assert stages[-1]["params"]["so3_prob"] == 1.0
    assert stages[-1]["params"]["standing_prob"] == 0.10


def test_v66_prone_orientation_respects_custom_angle_range():
    n = 256
    env = SimpleNamespace(
        device="cpu",
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=torch.zeros(n, 21),
                qvel=torch.zeros(n, 20),
            )
        )
    )
    ids = torch.arange(n)
    mdp.set_random_prone_orientation(
        env, ids, face_down_prob=0.34, side_prob=0.33,
        tilt_min_deg=5.0, tilt_max_deg=15.0,
    )
    quat = env.sim.data.qpos[:, 3:7]
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    tilt = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
    assert torch.all(tilt >= 5.0)
    assert torch.all(tilt <= 15.0)
    assert torch.allclose(quat.norm(dim=1), torch.ones(n), atol=1e-5)
    assert torch.all(env.sim.data.qvel[:, :6] == 0.0)


def test_v66_rejects_invalid_angle_range_and_registers_task():
    from mjlab.tasks.registry import list_tasks

    env = SimpleNamespace(device="cpu", sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(1, 21), qvel=torch.zeros(1, 20))))
    try:
        mdp.set_random_prone_orientation(env, torch.tensor([0]), tilt_min_deg=30.0, tilt_max_deg=15.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid angle range must fail")
    assert "v6.6-StandUp" in list_tasks()
    assert "v6.6-StandUp-Rough" in list_tasks()
    assert MicroduckV66StandUpRlCfg.experiment_name == "v6.6-StandUp"
    assert MicroduckV66StandUpRlCfg.run_name == "v6.6-StandUp"
