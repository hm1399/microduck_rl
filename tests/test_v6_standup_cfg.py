from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    make_microduck_v6_standup_env_cfg,
)


def test_v6_standup_builds_flat_and_rough():
    assert make_microduck_v6_standup_env_cfg() is not None
    assert make_microduck_v6_standup_env_cfg(rough=True) is not None


def test_v6_reset_uses_broad_random_state_event():
    cfg = make_microduck_v6_standup_env_cfg()
    event = cfg.events["set_ground_state"]

    assert event.func.__name__ == "set_random_v6_standup_state"
    assert event.params["standing_prob"] > 0.0
    assert event.params["sitting_prob"] > 0.0
    assert event.params["prone_prob"] > 0.0
    assert event.params["side_prob"] > 0.0
    assert event.params["joint_random_prob"] == 0.0


def test_v6_randomness_curriculum_widens_over_time():
    cfg = make_microduck_v6_standup_env_cfg()
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    first = stages[0]["params"]
    last = stages[-1]["params"]

    assert len(stages) == 4
    assert first["standing_prob"] + first["sitting_prob"] + first["prone_prob"] == 1.0
    assert last["standing_prob"] + last["sitting_prob"] + last["prone_prob"] == 1.0
    assert last["side_prob"] > first["side_prob"]
    assert last["so3_prob"] > first["so3_prob"]
    assert last["joint_random_prob"] > first["joint_random_prob"]
    assert last["sitting_joint_noise_std"] > first["sitting_joint_noise_std"]
    assert last["standing_joint_noise_std"] > first["standing_joint_noise_std"]


def test_v6_task_is_registered():
    from mjlab.tasks.registry import list_tasks
    import mjlab_microduck.tasks  # noqa: F401

    assert "Mjlab-v6-StandUp-Flat-MicroDuck" in list_tasks()
    assert "Mjlab-v6-StandUp-Flat-Backlash-MicroDuck" in list_tasks()
