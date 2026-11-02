from mjlab_microduck.tasks.microduck_roll_env_cfg import (
    MicroduckV68RollRlCfg,
    make_microduck_v68_roll_env_cfg,
)


def test_v68_trains_only_faceup_and_standing_rehearsal():
    cfg = make_microduck_v68_roll_env_cfg()
    assert make_microduck_v68_roll_env_cfg(rough=True) is not None
    params = cfg.events["set_ground_state"].params
    assert params["standing_prob"] == 0.50
    assert params["sitting_prob"] == 0.0
    assert params["prone_prob"] == 0.50
    assert params["face_down_prob"] == 0.0
    assert params["side_prob"] == 0.0
    assert params["tilt_min_deg"] == 5.0
    assert params["tilt_max_deg"] == 15.0
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert len(stages) == 1
    assert stages[0]["params"]["face_down_prob"] == 0.0
    assert stages[0]["params"]["side_prob"] == 0.0


def test_v68_keeps_strict_standing_recovery_and_registration():
    from mjlab.tasks.registry import list_tasks

    cfg = make_microduck_v68_roll_env_cfg()
    assert "v63_recovery" in cfg.rewards
    assert "v63_signed_pose" in cfg.rewards
    assert "v63_hold_complete" in cfg.terminations
    assert "v6.8-FaceUpStand" in list_tasks()
    assert MicroduckV68RollRlCfg.experiment_name == "v6.8-FaceUpStand"
    assert MicroduckV68RollRlCfg.max_iterations == 250
