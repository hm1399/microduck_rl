from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    V64_GROUND_STATE_STAGES,
    MicroduckV64StandUpRlCfg,
    make_microduck_v64_standup_env_cfg,
)


def test_v64_preserves_standing_and_gradually_introduces_recovery():
    cfg = make_microduck_v64_standup_env_cfg()
    assert make_microduck_v64_standup_env_cfg(rough=True) is not None
    initial = V64_GROUND_STATE_STAGES[0]["params"]
    assert initial["standing_prob"] == 0.60
    assert initial["prone_prob"] == 0.20
    assert initial["so3_prob"] == 0.0
    assert initial["joint_random_prob"] == 0.0
    assert cfg.events["set_ground_state"].params["standing_prob"] == 0.60
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert stages == V64_GROUND_STATE_STAGES
    assert stages[-1]["params"]["standing_prob"] == 0.20
    assert stages[-1]["params"]["prone_prob"] == 0.60


def test_v64_keeps_v631_reward_fixes_and_has_distinct_registration():
    from mjlab.tasks.registry import list_tasks
    from mjlab_microduck.tasks import mdp

    cfg = make_microduck_v64_standup_env_cfg()
    assert cfg.rewards["upright_progress"].func is mdp.v63_upright_progress_reward
    assert cfg.rewards["height_progress"].func is mdp.v63_height_progress_reward
    assert cfg.rewards["v63_signed_pose"].params["tripod_height_min"] == 0.055
    assert cfg.rewards["v63_signed_pose"].params["tripod_tilt_max_deg"] == 60.0
    assert "v6.4-StandUp" in list_tasks()
    assert "v6.4-StandUp-Rough" in list_tasks()
    assert MicroduckV64StandUpRlCfg.experiment_name == "v6.4-StandUp"
    assert MicroduckV64StandUpRlCfg.run_name == "v6.4-StandUp"
