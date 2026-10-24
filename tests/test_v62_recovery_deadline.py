import math

import torch

from mjlab_microduck.tasks import mdp


class _SensorData:
    def __init__(self, found):
        self.found = found


class _Sensor:
    def __init__(self, found):
        self.data = _SensorData(found)


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Data:
    def __init__(self, n):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0
        self.root_link_vel_w = torch.zeros(n, 6)
        self.root_link_lin_vel_w = self.root_link_vel_w[:, :3]
        self.root_link_ang_vel_w = self.root_link_vel_w[:, 3:]
        self.joint_pos = torch.zeros(n, 14)
        self.default_joint_pos = torch.zeros(n, 14)


class _Asset:
    def __init__(self, n):
        self.data = _Data(n)
        self.writes = 0

    def find_joints(self, _pattern):
        return list(range(14)), [f"joint_{i}" for i in range(14)]

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.data.root_link_vel_w[env_ids] = velocity
        self.writes += 1


class _Scene:
    def __init__(self, n):
        self.asset = _Asset(n)
        self.terrain = _Terrain(n)
        self.sensors = {
            "feet_ground_contact": _Sensor(torch.zeros(n, 2, 1)),
            "head_ground_contact": _Sensor(torch.zeros(n, 1, 1)),
        }

    def __getitem__(self, _name):
        return self.asset


class _Env:
    def __init__(self, n=2, step_dt=0.1):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = step_dt
        self.common_step_counter = 0
        self.scene = _Scene(n)


PARAMS = {
    "recovery_deadline_s": 5.0,
    "recovery_confirm_s": 0.4,
    "hold_s": 0.5,
    "recovery_success_bonus": 5.0,
    "recovery_early_bonus": 5.0,
    "recovery_timeout_penalty": -5.0,
    "hold_success_bonus": 8.0,
}


def _set_valid(env, env_id, valid=True):
    env.scene.asset.data.root_link_pos_w[env_id, 2] = 0.115 if valid else 0.05
    env.scene.sensors["feet_ground_contact"].data.found[env_id] = 1.0 if valid else 0.0
    env.scene.sensors["head_ground_contact"].data.found[env_id] = 0.0
    env.scene.asset.data.root_link_quat_w[env_id] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.scene.asset.data.root_link_vel_w[env_id] = 0.0
    env.scene.asset.data.joint_pos[env_id] = 0.0


def _step_reward(env):
    env.common_step_counter += 1
    return mdp.v62_recovery_deadline_reward(env, **PARAMS) * env.step_dt


def test_push_episode_waits_then_applies_exactly_one_radial_kick():
    env = _Env(n=2)
    ids = torch.tensor([0, 1])
    mdp._reset_v62_recovery_state(env, ids, torch.tensor([True, False]))
    _set_valid(env, 0, True)
    _set_valid(env, 1, True)

    _step_reward(env)
    assert env._v62_phase[0].item() == mdp._V62_WAIT_PUSH
    assert env._v62_phase[1].item() == mdp._V62_RECOVER

    mdp.apply_v62_push_once(env, ids, magnitude_range=(0.25, 0.25))
    speed = torch.linalg.norm(env.scene.asset.data.root_link_vel_w[0, :2]).item()
    assert math.isclose(speed, 0.25, rel_tol=1e-5)
    assert env._v62_phase[0].item() == mdp._V62_RECOVER
    assert env.scene.asset.writes == 1

    first_velocity = env.scene.asset.data.root_link_vel_w.clone()
    mdp.apply_v62_push_once(env, ids, magnitude_range=(0.40, 0.40))
    assert env.scene.asset.writes == 1
    assert torch.equal(env.scene.asset.data.root_link_vel_w, first_velocity)


def test_recovery_requires_confirmation_and_earlier_success_pays_more():
    env = _Env(n=2)
    mdp._reset_v62_recovery_state(
        env, torch.tensor([0, 1]), torch.tensor([False, False])
    )
    _set_valid(env, 0, True)
    _set_valid(env, 1, False)

    rewards = torch.zeros(2)
    for _ in range(4):
        rewards += _step_reward(env)
    assert env._v62_phase[0].item() == mdp._V62_HOLD
    assert math.isclose(rewards[0].item(), 9.6, rel_tol=1e-5)
    assert rewards[1].item() == 0.0

    for _ in range(6):
        rewards += _step_reward(env)
    _set_valid(env, 1, True)
    for _ in range(4):
        rewards += _step_reward(env)
    assert env._v62_phase[1].item() == mdp._V62_HOLD
    assert math.isclose(rewards[1].item(), 8.6, rel_tol=1e-5)
    assert rewards[0] > rewards[1]


def test_timeout_penalty_occurs_once_and_terminates():
    env = _Env(n=1)
    mdp._reset_v62_recovery_state(env, torch.tensor([0]), torch.tensor([False]))
    _set_valid(env, 0, False)

    total = 0.0
    done = torch.tensor([False])
    for _ in range(50):
        env.common_step_counter += 1
        done = mdp.v62_recovery_timeout(env, **PARAMS)
        total += (mdp.v62_recovery_deadline_reward(env, **PARAMS) * env.step_dt)[0].item()
    assert done.item() is True
    assert math.isclose(total, -5.0, rel_tol=1e-6)

    env.common_step_counter += 1
    assert mdp.v62_recovery_timeout(env, **PARAMS).item() is False
    assert (mdp.v62_recovery_deadline_reward(env, **PARAMS) * env.step_dt).item() == 0.0


def test_hold_must_be_uninterrupted_and_completion_pays_exact_bonus():
    env = _Env(n=1)
    mdp._reset_v62_recovery_state(env, torch.tensor([0]), torch.tensor([False]))
    _set_valid(env, 0, True)
    for _ in range(4):
        _step_reward(env)
    assert env._v62_phase[0].item() == mdp._V62_HOLD

    for _ in range(3):
        _step_reward(env)
    assert env._v62_hold_steps[0].item() == 3
    _set_valid(env, 0, False)
    _step_reward(env)
    assert env._v62_hold_steps[0].item() == 0

    _set_valid(env, 0, True)
    completion = 0.0
    done = torch.tensor([False])
    for _ in range(5):
        env.common_step_counter += 1
        done = mdp.v62_hold_complete(env, **PARAMS)
        completion += (mdp.v62_recovery_deadline_reward(env, **PARAMS) * env.step_dt)[0].item()
    assert done.item() is True
    assert env._v62_phase[0].item() == mdp._V62_COMPLETE
    assert math.isclose(completion, 8.0, rel_tol=1e-6)


def test_head_contact_and_excess_neck_pitch_block_success():
    env = _Env(n=1)
    _set_valid(env, 0, True)
    assert mdp._v62_standing_valid(env).item() is True

    env.scene.sensors["head_ground_contact"].data.found[0] = 1.0
    assert mdp._v62_standing_valid(env).item() is False
    env.scene.sensors["head_ground_contact"].data.found[0] = 0.0

    env.scene.asset.data.joint_pos[0, 5] = math.radians(31.0)
    assert mdp._v62_standing_valid(env).item() is False
