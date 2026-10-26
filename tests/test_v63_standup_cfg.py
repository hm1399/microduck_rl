from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    V62_EPISODE_LENGTH_S,
    V62_HOLD_S,
    V62_PREPARE_S,
    V62_RECOVERY_DEADLINE_S,
    MicroduckV63StandUpRlCfg,
    make_microduck_v63_standup_env_cfg,
)


def test_v63_builds_and_preserves_budget_with_boundary_margin():
    cfg = make_microduck_v63_standup_env_cfg()
    assert make_microduck_v63_standup_env_cfg(rough=True) is not None
    required = V62_PREPARE_S + V62_RECOVERY_DEADLINE_S + V62_HOLD_S
    assert V62_EPISODE_LENGTH_S == 12.0
    assert required == 11.0
    assert cfg.episode_length_s >= required + 1.0


def test_v63_replaces_farmable_absolute_pose_rewards():
    cfg = make_microduck_v63_standup_env_cfg()
    for name in [
        "height_stand", "upright_linear", "standing_composite",
        "body_pose_tracking", "pose_stand_neck", "standing_hold",
        "head_contact_near_stand", "v62_recovery_deadline",
    ]:
        assert name not in cfg.rewards
    for name in ["upright_progress", "height_progress", "v63_signed_pose", "v63_recovery"]:
        assert name in cfg.rewards
    for name in ["v62_recovery_timeout", "v62_hold_complete"]:
        assert name not in cfg.terminations
    for name in ["v63_recovery_timeout", "v63_hold_complete", "v63_session_timeout"]:
        assert name in cfg.terminations


def test_v63_has_no_curriculum_for_deleted_rewards():
    cfg = make_microduck_v63_standup_env_cfg()
    reward_names = set(cfg.rewards)
    for curriculum in cfg.curriculum.values():
        reward_name = curriculum.params.get("reward_name")
        if reward_name is not None:
            assert reward_name in reward_names


def test_v63_tasks_are_registered_with_distinct_wandb_name():
    from mjlab.tasks.registry import list_tasks

    assert "v6.3-StandUp" in list_tasks()
    assert "v6.3-StandUp-Rough" in list_tasks()
    assert MicroduckV63StandUpRlCfg.experiment_name == "v6.3-StandUp"
    assert MicroduckV63StandUpRlCfg.run_name == "v6.3-StandUp"
