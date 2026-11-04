from mjlab_microduck.tasks.microduck_roll_env_cfg import (
    MicroduckV69RollRlCfg,
    make_microduck_v69_roll_env_cfg,
)


def test_v69_locks_faceup_gate_to_three_to_five_degrees():
    cfg = make_microduck_v69_roll_env_cfg()
    assert make_microduck_v69_roll_env_cfg(rough=True) is not None
    params = cfg.events["set_ground_state"].params
    assert params["tilt_min_deg"] == 3.0
    assert params["tilt_max_deg"] == 5.0
    assert params["face_down_prob"] == 0.0
    assert params["side_prob"] == 0.0
    assert params["standing_prob"] == 0.50
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert len(stages) == 1
    assert stages[0]["params"]["tilt_max_deg"] == 5.0


def test_v69_is_registered_with_action_gate_budget():
    from mjlab.tasks.registry import list_tasks

    assert "v6.9-FaceUpStand" in list_tasks()
    assert MicroduckV69RollRlCfg.experiment_name == "v6.9-FaceUpStand"
    assert MicroduckV69RollRlCfg.max_iterations == 250
