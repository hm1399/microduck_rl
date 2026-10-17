"""Microduck Run environment (v5_run): v4 (velstand) + RUNNING, one policy.

Built 2026-10-17 on branch v5-run (base = develop b02fdea, the exact commit the
v4 velstand run 181y2lna was trained from). Plan: docs/v5_run_training_plan.md
in the mujoco workspace (not in this repo).

WHAT THIS ADDS ON TOP OF VELSTAND (everything else is inherited unchanged,
so the v4 checkpoint loads and the 61D obs contract holds):

  1. STAGED VELOCITY-RANGE CURRICULUM — lin_vel_x widens ±0.8 → ±1.2 → ±1.6
     → ±2.0 m/s at 600-iter steps (velocity_x_range_curriculum, mdp.py).
     Iter 1200 (±1.6) is the DECISION POINT: export + eyeball the gait before
     committing to ±2.0. lin_vel_y / ang_vel_z stay at v4 ranges.
  2. EPISODE-OPEN RAMP (RampVelocityCommand, mdp.py) — every episode commands
     0 for the first second (stand), ramps to slow_frac (≈ the v4 walk range)
     by 2 s, then to the full target by 3 s. "Stand → walk → run" is part of
     the trained behaviour, and the sustain bonus (3) has a defined start.
     Training-only: play/export uses the plain un-ramped command.
  3. SUSTAIN BONUS (run_sustain_bonus, mdp.py) — one-shot +10 + 5×(avg speed
     above 1.5, capped at +1 m/s) the first time an episode holds actual
     |vx| ≥ 1.5 for 10 continuous seconds while upright AND commanded ≥ 1.5.
     Anti-farming: command AND actual speed gated, one-shot, stumble resets
     the clock.
  4. V4 ANCHOR BC — PpoWithExpertBc with the stand expert disabled (gate
     180° never fires; v5 does not train fall recovery) and the WALK ANCHOR
     set to the v4 checkpoint, coef 0.5, gated to frames whose commanded
     |v_xy| ≤ 0.4 (the anchor's own trained range — anchoring run frames
     would drag the policy back toward walking). Standing / slow-walk /
     turn-in-place frames are pinned to v4; run frames are PPO's alone.

CURRICULA: v4 already completed every velstand phase, so on top of the
inherited velocity-collapse (velstand's WARM_START) ALL velstand curricula
are collapsed to final too — recovery economics, prone slice (45%), topple
pushes and protective costs stay exactly as v4 ended them. The only ramp
left is the velocity-range one above.

WARM START (launch, mirrors v4's mechanism):
    MICRODUCK_WARM_START=1 uv run train Mjlab-Run-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations 3000 \
        --agent.resume True \
        --wandb-run-path mandy0917-city-university-of-hong-kong/mjlab_microduck/181y2lna \
        --wandb-checkpoint-name model_1999.pt
(The BC anchor resolves the same checkpoint from the same wandb run.)
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    NUM_STEPS_PER_ENV,
    _collapse_curricula_to_final,
    make_microduck_velstand_env_cfg,
)
from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg, default_bc_cfg

# ── v4 checkpoint (wandb run 181y2lna = 2026-10-16_08-13-28_velstand) ──────
V4_WANDB_RUN = "mandy0917-city-university-of-hong-kong/mjlab_microduck/181y2lna"
V4_CHECKPOINT = "model_1999.pt"

# ── (1) staged velocity range: iter 1200 = decision point ──────────────────
VEL_X_STAGES_ITERS = (0, 600, 1200, 1800)
VEL_X_STAGES_ABS = (0.8, 1.2, 1.6, 2.0)

# ── (2) episode-open ramp ──────────────────────────────────────────────────
RAMP_STAND_S = 1.0
RAMP_SLOW_END_S = 2.0
RAMP_FULL_END_S = 3.0
RAMP_SLOW_FRAC = 0.4   # ≈ v4's ±0.4 m/s walk range, as a fraction of any stage

# ── (3) sustain bonus ──────────────────────────────────────────────────────
SUSTAIN_MIN_SPEED = 1.5
SUSTAIN_S = 10.0
SUSTAIN_BASE_BONUS = 10.0
SUSTAIN_EXCESS_COEF = 5.0
SUSTAIN_EXCESS_CAP = 1.0
SUSTAIN_UP_TILT_DEG = 25.0
SUSTAIN_UP_Z = 0.09

# ── (4) v4 anchor BC ───────────────────────────────────────────────────────
ANCHOR_COEF = 0.5
ANCHOR_TILT_DEG = 25.0
ANCHOR_MAX_LIN_VEL = 0.4   # anchor's own trained |v_xy| ceiling


def make_microduck_run_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Everything velstand has (all-collision robot, recovery layer, body
    # control, protective costs, BC-routing-ready obs) — verbatim.
    cfg = make_microduck_velstand_env_cfg(play=play, rough=rough)

    if not play:
        # v4 finished every velstand phase: pin prone slice (45%), recovery
        # economics, topple pushes and protective costs at their final state.
        # (The inherited velocity curricula were already collapsed inside
        # velstand's make, via its WARM_START.)
        _collapse_curricula_to_final(cfg)

        # (2) episode-open ramp on the twist command (training only).
        tw = cfg.commands["twist"]
        cfg.commands["twist"] = microduck_mdp.RampVelocityCommandCfg(
            **vars(tw),
            stand_s=RAMP_STAND_S,
            slow_ramp_end_s=RAMP_SLOW_END_S,
            full_ramp_end_s=RAMP_FULL_END_S,
            slow_frac=RAMP_SLOW_FRAC,
        )

        # (1) staged lin_vel_x widening — the ONLY curriculum that ramps.
        cfg.curriculum["velocity_x_range"] = CurriculumTermCfg(
            func=microduck_mdp.velocity_x_range_curriculum,
            params={
                "command_name": "twist",
                "x_stages": [
                    {"step": it * NUM_STEPS_PER_ENV, "lin_vel_x": (-v, v)}
                    for it, v in zip(VEL_X_STAGES_ITERS, VEL_X_STAGES_ABS)
                ],
            },
        )

    # (3) sustain bonus (weight 1.0 — the bonus magnitude lives in the term).
    cfg.rewards["run_sustain"] = RewardTermCfg(
        func=microduck_mdp.run_sustain_bonus,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "command_name": "twist",
            "min_speed": SUSTAIN_MIN_SPEED,
            "sustain_s": SUSTAIN_S,
            "base_bonus": SUSTAIN_BASE_BONUS,
            "excess_coef": SUSTAIN_EXCESS_COEF,
            "excess_cap": SUSTAIN_EXCESS_CAP,
            "up_tilt_deg": SUSTAIN_UP_TILT_DEG,
            "up_z": SUSTAIN_UP_Z,
        },
    )

    return cfg


MicroduckRunRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithExpertBcCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
        bc_cfg={
            **default_bc_cfg(),
            # Stand expert OFF for v5 (recovery is not being retrained; v4's
            # recovered behaviour stays via the warm start). The checkpoint
            # must still load, so it points at v4 — and gate_tilt_deg=180°
            # makes the fallen mask empty, so the expert is never queried.
            "wandb_run_path": V4_WANDB_RUN,
            "checkpoint_name": V4_CHECKPOINT,
            "coef": 0.0,
            "gate_tilt_deg": 180.0,
            "expert_mask_cols": (),
            # v4 walk anchor: standing / slow-walk / turn frames only.
            "anchor_wandb_run_path": V4_WANDB_RUN,
            "anchor_checkpoint_name": V4_CHECKPOINT,
            "anchor_coef": ANCHOR_COEF,
            "anchor_tilt_deg": ANCHOR_TILT_DEG,
            "anchor_max_lin_vel": ANCHOR_MAX_LIN_VEL,
            # The remap table was calibrated on alpha_walking (441tzs6d), not
            # on v4 — a wrong table feeds wrong yaw labels. OFF.
            "anchor_yaw_remap": None,
            "unanchor_turn_in_place": False,
            # v4 DOES body control, so body-command frames stay on the anchor.
            "body_slice": None,
        },
    ),
    wandb_project="mjlab_microduck",
    experiment_name="run",
    run_name="run",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
)
