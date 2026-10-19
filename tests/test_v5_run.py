import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.distill import PpoWithExpertBc, default_bc_cfg
from mjlab_microduck.tasks.microduck_run_env_cfg import (
    FELL_OVER_DISABLE_ITER,
    MicroduckRunRlCfg,
    NUM_STEPS_PER_ENV,
    VEL_X_STAGES_ABS,
    make_microduck_run_env_cfg,
)


def test_ramp_velocity_command_does_not_compound_scaled_value():
    command = object.__new__(mdp.RampVelocityCommand)
    command.cfg = SimpleNamespace(
        heading_command=False,
        stand_s=1.0,
        slow_ramp_end_s=2.0,
        full_ramp_end_s=3.0,
        slow_frac=0.4,
    )
    command._env = SimpleNamespace(
        episode_length_buf=torch.tensor([0]),
        step_dt=1.0,
    )
    command.vel_command_b = torch.tensor([[1.0, -0.5, 0.25]])
    command.vel_command_w = command.vel_command_b.clone()
    command._raw_vel_command_b = command.vel_command_b.clone()
    command._raw_vel_command_w = command.vel_command_w.clone()
    command.is_world_env = torch.tensor([False])
    command.is_standing_env = torch.tensor([False])

    command._update_command()
    assert torch.equal(command.vel_command_b, torch.zeros_like(command.vel_command_b))

    command._env.episode_length_buf[:] = 4
    command._update_command()
    assert torch.allclose(command.vel_command_b, command._raw_vel_command_b)


class _DummyPolicy(torch.nn.Module):
    obs_groups = ("actor",)
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)


def test_anchor_only_bc_skips_disabled_expert_checkpoint(tmp_path):
    actor = _DummyPolicy()
    critic = _DummyPolicy()
    anchor_path = tmp_path / "anchor.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, anchor_path)

    cfg = default_bc_cfg()
    cfg.update(
        {
            "wandb_run_path": "invalid/stand/expert",
            "coef": 0.0,
            "anchor_wandb_run_path": None,
            "anchor_checkpoint_path": str(anchor_path),
            "anchor_coef": 1.0,
        }
    )

    algorithm = PpoWithExpertBc(actor, critic, object(), bc_cfg=cfg, device="cpu")

    assert algorithm.expert is None
    assert algorithm.anchor is not None


def test_v5_recovery_curriculum_starts_safe_and_ramps():
    cfg = make_microduck_run_env_cfg()

    fell_stages = cfg.curriculum["fell_over_disable"].params["param_stages"]
    assert math.isclose(fell_stages[0]["params"]["limit_angle"], math.radians(70.0))
    assert fell_stages[-1]["step"] == FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV
    assert math.isclose(fell_stages[-1]["params"]["limit_angle"], math.pi)

    prone_stages = cfg.curriculum["prone_init_prob"].params["param_stages"]
    assert prone_stages[0]["params"]["prone_prob"] == 0.0
    assert prone_stages[-1]["params"]["prone_prob"] == 0.30

    standing_stages = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert standing_stages[0]["rel_standing_envs"] == 0.05
    assert standing_stages[-1]["rel_standing_envs"] == 0.25


def test_v5_velocity_schedule_is_below_unvalidated_run_speed():
    cfg = make_microduck_run_env_cfg()
    stages = cfg.curriculum["velocity_x_range"].params["x_stages"]

    assert [stage["lin_vel_x"][1] for stage in stages] == list(VEL_X_STAGES_ABS)
    assert max(VEL_X_STAGES_ABS) < 2.0
    assert stages[1]["step"] == 800 * NUM_STEPS_PER_ENV


def test_v5_keeps_recovery_expert_bc_enabled():
    bc_cfg = MicroduckRunRlCfg.algorithm.bc_cfg

    assert bc_cfg["coef"] > 0.0
    assert bc_cfg["gate_tilt_deg"] < 180.0
    assert bc_cfg["anchor_coef"] > 0.0
    assert bc_cfg["expert_mask_cols"] == tuple(range(6, 48))
