"""Microduck v6.8 — face-up to strict standing with standing rehearsal."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    V66_GROUND_STATE_STAGES,
    MicroduckV67StandUpRlCfg,
    make_microduck_v67_standup_env_cfg,
)


def make_microduck_v68_roll_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Train only face-up starts and standing rehearsal, not face-down or side starts."""
    cfg = make_microduck_v67_standup_env_cfg(play=play, rough=rough)
    stage = deepcopy(V66_GROUND_STATE_STAGES[:1])
    stage[0]["params"].update(
        {
            "standing_prob": 0.50,
            "sitting_prob": 0.0,
            "prone_prob": 0.50,
            "face_down_prob": 0.0,
            "side_prob": 0.0,
        }
    )
    cfg.events["set_ground_state"].params.update(deepcopy(stage[0]["params"]))
    cfg.curriculum["ground_state_mix"].params["param_stages"] = stage
    return cfg


MicroduckV68RollRlCfg = deepcopy(MicroduckV67StandUpRlCfg)
MicroduckV68RollRlCfg.experiment_name = "v6.8-FaceUpStand"
MicroduckV68RollRlCfg.run_name = "v6.8-FaceUpStand"
MicroduckV68RollRlCfg.max_iterations = 250
