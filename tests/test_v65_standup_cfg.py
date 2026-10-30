from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    V65_GROUND_STATE_STAGES,
    MicroduckV65StandUpRlCfg,
    make_microduck_v65_standup_env_cfg,
)


def test_v65_covers_all_orientation_families_from_start():
    cfg = make_microduck_v65_standup_env_cfg()
    assert make_microduck_v65_standup_env_cfg(rough=True) is not None
    initial = V65_GROUND_STATE_STAGES[0]["params"]
    assert initial["standing_prob"] == 0.40
    assert initial["prone_prob"] == 0.50
    assert initial["face_down_prob"] == 0.50
    assert initial["side_prob"] == 0.30
    assert initial["so3_prob"] == 0.10
    assert initial["joint_random_prob"] == 0.05
    assert cfg.events["set_ground_state"].params["so3_prob"] == 0.10


def test_v65_reaches_full_so3_and_joint_limit_coverage():
    cfg = make_microduck_v65_standup_env_cfg()
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert stages == V65_GROUND_STATE_STAGES
    final = stages[-1]["params"]
    assert final["so3_prob"] == 1.0
    assert final["joint_random_prob"] == 0.90
    assert final["joint_range_frac"] == 1.0
    assert final["standing_prob"] == 0.10


def test_v65_is_registered_with_distinct_wandb_name():
    from mjlab.tasks.registry import list_tasks

    assert "v6.5-StandUp" in list_tasks()
    assert "v6.5-StandUp-Rough" in list_tasks()
    assert MicroduckV65StandUpRlCfg.experiment_name == "v6.5-StandUp"
    assert MicroduckV65StandUpRlCfg.run_name == "v6.5-StandUp"
