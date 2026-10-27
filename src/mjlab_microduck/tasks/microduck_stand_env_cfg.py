"""Microduck v4-stand — dedicated nominal-standing base policy.

This is deliberately NOT a fall-recovery task. Every episode starts upright,
uses exact-zero twist/head/body commands, lasts 20 seconds, and has no pushes or
prone/sitting resets. A later recovery policy can warm-start from this verified
61D/14D base instead of learning standing and ground recovery simultaneously.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

V4_STAND_EPISODE_LENGTH_S = 20.0


def make_microduck_v4_stand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the zero-command, no-push, upright-only standing task."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = V4_STAND_EPISODE_LENGTH_S

    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_turn_in_place_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (
        V4_STAND_EPISODE_LENGTH_S,
        2.0 * V4_STAND_EPISODE_LENGTH_S,
    )
    for name in ["head_pose", "body_pose"]:
        pose_command = cfg.commands[name]
        pose_command.ranges = ((0.0, 0.0),) * len(pose_command.ranges)
        pose_command.zero_command_prob = 1.0
        pose_command.resampling_time_range = (
            V4_STAND_EPISODE_LENGTH_S,
            2.0 * V4_STAND_EPISODE_LENGTH_S,
        )

    # No locomotion curriculum, recovery states, or external disturbance.
    cfg.events.pop("push_robot", None)
    for name in ["standing_envs", "head_pose_range", "body_pose_range"]:
        cfg.curriculum.pop(name, None)
    for name in [
        "air_time", "foot_clearance", "foot_swing_height", "foot_slip",
    ]:
        cfg.rewards.pop(name, None)

    head_bodies = tuple(name for name in HEAD_BODY_NAMES if name != "bearing_roll")
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=rf"^({'|'.join(head_bodies)})$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_ground_cfg,)

    standing_params = {
        "height_min": 0.10,
        "tilt_max_deg": 15.0,
        "neck_error_max_deg": 30.0,
        "lin_vel_max": 0.08,
        "ang_vel_max": 0.20,
        "feet_sensor_name": "feet_ground_contact",
        "head_sensor_name": "head_ground_contact",
        "neck_joint_indices": [5, 6, 7, 8],
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
    }
    # Average Episode_Reward/stand_valid / 2 is the valid-standing fraction.
    cfg.rewards["stand_valid"] = RewardTermCfg(
        func=microduck_mdp.v4_stand_valid_reward,
        weight=2.0,
        params=deepcopy(standing_params),
    )
    # Unlike recovery, this task never needs floor contact to flip upright.
    cfg.rewards["stand_head_contact"] = RewardTermCfg(
        func=microduck_mdp.v4_stand_head_contact_cost,
        weight=-4.0,
        params={"sensor_name": "head_ground_contact"},
    )
    return cfg


MicroduckV4StandRlCfg = deepcopy(MicroduckRlCfg)
MicroduckV4StandRlCfg.experiment_name = "v4-stand"
MicroduckV4StandRlCfg.run_name = "v4-stand"
MicroduckV4StandRlCfg.max_iterations = 2_000
