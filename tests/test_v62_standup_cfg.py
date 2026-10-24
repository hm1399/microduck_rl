from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    V62_EPISODE_LENGTH_S,
    V62_HOLD_S,
    V62_PREPARE_S,
    V62_PUSH_MAGNITUDE_STAGES,
    V62_RECOVERY_CONFIRM_S,
    V62_RECOVERY_DEADLINE_S,
    make_microduck_v62_standup_env_cfg,
)


def test_v62_builds_flat_and_rough():
    assert make_microduck_v62_standup_env_cfg() is not None
    assert make_microduck_v62_standup_env_cfg(rough=True) is not None


def test_v62_has_one_budgeted_radial_push():
    cfg = make_microduck_v62_standup_env_cfg()

    assert cfg.episode_length_s == V62_EPISODE_LENGTH_S == 12.0
    assert "push_robot" not in cfg.events
    assert "push_magnitude" not in cfg.curriculum
    event = cfg.events["v62_push_once"]
    assert event.func.__name__ == "apply_v62_push_once"
    assert event.interval_range_s == (V62_PREPARE_S, V62_PREPARE_S)
    assert event.params["magnitude_range"] == (0.0, 0.0)
    assert cfg.events["set_ground_state"].params["track_v62_episode_type"] is True

    stages = cfg.curriculum["v62_push_magnitude"].params["magnitude_stages"]
    assert stages == V62_PUSH_MAGNITUDE_STAGES
    assert stages[-1]["magnitude_range"][1] == 0.40

    play = make_microduck_v62_standup_env_cfg(play=True)
    assert play.events["v62_push_once"].params["magnitude_range"] == (0.20, 0.40)
    assert "v62_push_magnitude" not in play.curriculum


def test_v62_deadline_hold_and_head_contact_are_hard_wired():
    cfg = make_microduck_v62_standup_env_cfg()
    deadline = cfg.rewards["v62_recovery_deadline"]

    assert deadline.weight == 1.0
    assert deadline.params["recovery_deadline_s"] == V62_RECOVERY_DEADLINE_S == 5.0
    assert deadline.params["recovery_confirm_s"] == V62_RECOVERY_CONFIRM_S == 0.4
    assert deadline.params["hold_s"] == V62_HOLD_S == 5.0
    assert deadline.params["recovery_success_bonus"] == 5.0
    assert deadline.params["recovery_early_bonus"] == 5.0
    assert deadline.params["recovery_timeout_penalty"] == -5.0
    assert deadline.params["hold_success_bonus"] == 8.0
    assert deadline.params["tilt_max_deg"] == 15.0
    assert deadline.params["neck_error_max_deg"] == 30.0
    assert deadline.params["lin_vel_max"] == 0.08
    assert deadline.params["ang_vel_max"] == 0.20

    assert cfg.terminations["v62_recovery_timeout"].time_out is False
    assert cfg.terminations["v62_hold_complete"].time_out is False
    assert cfg.rewards["head_contact_near_stand"].weight < 0.0
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert "head_ground_contact" in sensor_names


def test_v62_trains_only_the_nominal_idle_pose():
    cfg = make_microduck_v62_standup_env_cfg()

    assert cfg.commands["head_pose"].ranges == ((0.0, 0.0),) * 4
    assert cfg.commands["body_pose"].ranges == ((0.0, 0.0),) * 6
    assert "head_pose_range" not in cfg.curriculum
    assert "body_pose_range" not in cfg.curriculum

    neck = cfg.rewards["pose_stand_neck"]
    assert neck.params["std"] == 0.07
    assert cfg.rewards["head_pose_tracking"].params["fine_std"] == 0.07
    assert cfg.rewards["standing_composite"].params["pose_std"] == 0.25
    assert cfg.rewards["standing_composite"].params["upright_std"] == 0.25

    hold = cfg.rewards["standing_hold"].params
    assert hold["hold_s"] == 5.0
    assert hold["tilt_zero_deg"] == 15.0
    assert hold["lin_vel_max"] == 0.08
    assert hold["ang_vel_max"] == 0.20
    assert cfg.curriculum["pose_stand_neck_weight"].params["weight_stages"][-1]["weight"] == 3.0
    assert cfg.curriculum["head_pose_bias_weight"].params["weight_stages"][-1]["weight"] == 3.0
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] == -0.6


def test_v62_tasks_are_registered():
    from mjlab.tasks.registry import list_tasks

    assert "v6.2-StandUp" in list_tasks()
    assert "v6.2-StandUp-Rough" in list_tasks()
