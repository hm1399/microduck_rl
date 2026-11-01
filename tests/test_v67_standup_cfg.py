from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    MicroduckV67StandUpRlCfg,
    make_microduck_v67_standup_env_cfg,
)


def test_v67_locks_curriculum_to_small_angle_gate():
    cfg = make_microduck_v67_standup_env_cfg()
    assert make_microduck_v67_standup_env_cfg(rough=True) is not None
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert len(stages) == 1
    params = stages[0]["params"]
    assert params["tilt_min_deg"] == 5.0
    assert params["tilt_max_deg"] == 15.0
    assert params["standing_prob"] == 0.50
    assert params["so3_prob"] == 0.0
    assert params["joint_random_prob"] == 0.0


def test_v67_is_registered_with_distinct_wandb_name():
    from mjlab.tasks.registry import list_tasks

    assert "v6.7-StandUp" in list_tasks()
    assert "v6.7-StandUp-Rough" in list_tasks()
    assert MicroduckV67StandUpRlCfg.experiment_name == "v6.7-StandUp"
    assert MicroduckV67StandUpRlCfg.run_name == "v6.7-StandUp"
