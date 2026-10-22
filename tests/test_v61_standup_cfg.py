from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    make_microduck_v61_standup_env_cfg,
)


def test_v61_standup_builds_flat_and_rough():
    assert make_microduck_v61_standup_env_cfg() is not None
    assert make_microduck_v61_standup_env_cfg(rough=True) is not None


def test_v61_keeps_v6_random_reset_states():
    cfg = make_microduck_v61_standup_env_cfg()
    event = cfg.events["set_ground_state"]

    assert event.func.__name__ == "set_random_v6_standup_state"
    assert cfg.curriculum["ground_state_mix"].params["param_stages"]


def test_v61_posture_enforcement_rewards_exist():
    cfg = make_microduck_v61_standup_env_cfg()

    neck = cfg.rewards["pose_stand_neck"]
    assert neck.func.__name__ == "pose_target_match"
    assert neck.weight == 0.0  # 课程引入
    assert neck.params["joint_indices"] == [5, 6, 7, 8]
    assert neck.params["std"] == 0.1

    hold = cfg.rewards["standing_hold"]
    assert hold.func.__name__ == "standing_hold_bonus"
    assert hold.weight == 2.0
    assert hold.params["hold_s"] == 5.0

    head = cfg.rewards["head_pose_tracking"]
    assert head.params["fine_std"] == 0.1
    assert head.params["fine_weight"] == 0.5

    composite = cfg.rewards["standing_composite"]
    assert 5 in composite.params["joint_indices"]  # 头颈已纳入全身覆盖


def test_v61_posture_curricula_ramp_after_recovery_discovery():
    cfg = make_microduck_v61_standup_env_cfg()

    neck_stages = cfg.curriculum["pose_stand_neck_weight"].params["weight_stages"]
    assert neck_stages[0]["weight"] == 0.0
    assert neck_stages[-1]["step"] == 4000 * 24
    assert neck_stages[-1]["weight"] == 2.0

    bias_stages = cfg.curriculum["head_pose_bias_weight"].params["weight_stages"]
    assert bias_stages[-1]["weight"] == 2.5


def test_v61_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    assert "v6.1-StandUp" in list_tasks()
    assert "v6.1-StandUp-Rough" in list_tasks()
