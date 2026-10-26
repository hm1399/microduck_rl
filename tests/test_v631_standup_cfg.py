from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    MicroduckV631StandUpRlCfg,
    make_microduck_v631_standup_env_cfg,
)


def test_v631_uses_dt_corrected_progress_and_observed_tripod_gate():
    cfg = make_microduck_v631_standup_env_cfg()
    assert make_microduck_v631_standup_env_cfg(rough=True) is not None
    assert cfg.rewards["upright_progress"].func is mdp.v63_upright_progress_reward
    assert cfg.rewards["height_progress"].func is mdp.v63_height_progress_reward
    for name in ["v63_signed_pose", "v63_recovery"]:
        assert cfg.rewards[name].params["tripod_height_min"] == 0.055
        assert cfg.rewards[name].params["tripod_tilt_max_deg"] == 60.0
    for name in ["v63_recovery_timeout", "v63_hold_complete", "v63_session_timeout"]:
        assert cfg.terminations[name].params["tripod_height_min"] == 0.055
        assert cfg.terminations[name].params["tripod_tilt_max_deg"] == 60.0


def test_v631_preserves_v63_timeout_weights_and_has_distinct_registration():
    from mjlab.tasks.registry import list_tasks

    cfg = make_microduck_v631_standup_env_cfg()
    assert cfg.rewards["v63_recovery"].weight == 1.0
    assert cfg.rewards["v63_recovery"].params["recovery_deadline_s"] == 5.0
    assert cfg.rewards["v63_recovery"].params["hold_s"] == 5.0
    assert "v6.3.1-StandUp" in list_tasks()
    assert "v6.3.1-StandUp-Rough" in list_tasks()
    assert MicroduckV631StandUpRlCfg.experiment_name == "v6.3.1-StandUp"
    assert MicroduckV631StandUpRlCfg.run_name == "v6.3.1-StandUp"
