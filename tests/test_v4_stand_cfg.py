from mjlab_microduck.tasks.microduck_stand_env_cfg import (
    MicroduckV4StandRlCfg,
    make_microduck_v4_stand_env_cfg,
)


def test_v4_stand_is_upright_only_zero_command_and_no_push():
    cfg = make_microduck_v4_stand_env_cfg()
    assert cfg.episode_length_s == 20.0
    assert "push_robot" not in cfg.events
    assert "set_ground_state" not in cfg.events
    assert "v62_recovery_deadline" not in cfg.rewards
    assert "v63_recovery" not in cfg.rewards
    assert "fell_over" in cfg.terminations

    twist = cfg.commands["twist"]
    assert twist.rel_standing_envs == 1.0
    assert twist.rel_turn_in_place_envs == 0.0
    assert twist.ranges.lin_vel_x == (0.0, 0.0)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    for name, dims in [("head_pose", 4), ("body_pose", 6)]:
        command = cfg.commands[name]
        assert command.ranges == ((0.0, 0.0),) * dims
        assert command.zero_command_prob == 1.0


def test_v4_stand_has_strict_hold_metric_and_ungated_head_cost():
    cfg = make_microduck_v4_stand_env_cfg()
    assert cfg.rewards["stand_valid"].weight == 2.0
    assert cfg.rewards["stand_valid"].params["height_min"] == 0.10
    assert cfg.rewards["stand_valid"].params["tilt_max_deg"] == 15.0
    assert cfg.rewards["stand_head_contact"].weight == -4.0
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert "head_ground_contact" in sensor_names
    for name in ["standing_envs", "head_pose_range", "body_pose_range"]:
        assert name not in cfg.curriculum


def test_v4_stand_is_registered_with_bounded_training_budget():
    from mjlab.tasks.registry import list_tasks

    assert "v4-stand" in list_tasks()
    assert MicroduckV4StandRlCfg.experiment_name == "v4-stand"
    assert MicroduckV4StandRlCfg.run_name == "v4-stand"
    assert MicroduckV4StandRlCfg.max_iterations == 2000
