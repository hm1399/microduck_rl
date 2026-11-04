from mjlab_microduck.tasks.microduck_roll_env_cfg import (
    MicroduckV610RollRlCfg,
    make_microduck_v610_roll_env_cfg,
)


def test_v610_limits_increment_to_five_to_fifteen_and_fifty_iterations():
    cfg = make_microduck_v610_roll_env_cfg()
    assert make_microduck_v610_roll_env_cfg(rough=True) is not None
    params = cfg.events["set_ground_state"].params
    assert params["tilt_min_deg"] == 5.0
    assert params["tilt_max_deg"] == 15.0
    assert params["standing_prob"] == 0.50
    assert len(cfg.curriculum["ground_state_mix"].params["param_stages"]) == 1
    assert MicroduckV610RollRlCfg.max_iterations == 50
    assert MicroduckV610RollRlCfg.save_interval == 50


def test_v610_is_registered_with_distinct_wandb_name():
    from mjlab.tasks.registry import list_tasks

    assert "v6.10-FaceUpStand" in list_tasks()
    assert MicroduckV610RollRlCfg.experiment_name == "v6.10-FaceUpStand"
    assert MicroduckV610RollRlCfg.run_name == "v6.10-FaceUpStand"
