from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.distill import PpoWithExpertBc, default_bc_cfg


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
