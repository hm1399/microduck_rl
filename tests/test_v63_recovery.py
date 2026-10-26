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

    def find_joints(self, _pattern):
        return list(range(14)), [f"joint_{i}" for i in range(14)]


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
    def __init__(self, n=1, step_dt=0.02):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = step_dt
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.scene = _Scene(n)
        mdp._reset_v62_recovery_state(
            self, torch.arange(n), torch.zeros(n, dtype=torch.bool)
        )


PARAMS = {
    "recovery_deadline_s": 5.0,
    "recovery_confirm_s": 0.4,
    "hold_s": 5.0,
    "session_deadline_s": 12.0,
}


def _set_valid(env, env_id, valid=True):
    env.scene.asset.data.root_link_pos_w[env_id, 2] = 0.115 if valid else 0.04
    env.scene.asset.data.root_link_quat_w[env_id] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.scene.asset.data.root_link_vel_w[env_id] = 0.0
    env.scene.asset.data.joint_pos[env_id] = 0.0
    env.scene.sensors["feet_ground_contact"].data.found[env_id] = float(valid)
    env.scene.sensors["head_ground_contact"].data.found[env_id] = 0.0


def _set_tilt(env, env_id, tilt_deg):
    half = math.radians(tilt_deg) / 2.0
    env.scene.asset.data.root_link_quat_w[env_id] = torch.tensor(
        [math.cos(half), 0.0, math.sin(half), 0.0]
    )


def _step_reward(env):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    return mdp.v63_recovery_reward(env, **PARAMS) * env.step_dt


def test_signed_pose_ideal_tripod_and_floor_recovery():
    env = _Env(n=3)
    for env_id in range(3):
        _set_valid(env, env_id, True)

    ideal = mdp.v63_signed_pose_reward(env)[0]
    assert ideal.item() > 0.0

    _set_tilt(env, 1, 41.0)
    env.scene.asset.data.joint_pos[1, 5] = math.radians(98.0)
    env.scene.sensors["head_ground_contact"].data.found[1] = 1.0
    tripod = mdp.v63_signed_pose_reward(env)[1]
    assert tripod.item() == -6.0

    _set_valid(env, 2, False)
    env.scene.sensors["head_ground_contact"].data.found[2] = 1.0
    floor = mdp.v63_signed_pose_reward(env)[2]
    assert floor.item() == 0.0
    assert torch.all(mdp._v62_standing_valid(env) == torch.tensor([True, False, False]))


def test_recovery_bonus_is_nonlinear_and_strictly_prefers_faster_recovery():
    env = _Env(n=2)
    _set_valid(env, 0, True)
    _set_valid(env, 1, False)
    total = torch.zeros(2)
    for _ in range(20):
        total += _step_reward(env)
    fast = total[0].item()
    assert math.isclose(fast, 4.0 + 6.0 * (1.0 - 0.4 / 5.0) ** 2, rel_tol=1e-5)
    for _ in range(30):
        total += _step_reward(env)

    _set_valid(env, 1, True)
    for _ in range(20):
        total += _step_reward(env)
    slow = total[1].item()
    assert math.isclose(slow, 4.0 + 6.0 * (1.0 - 1.4 / 5.0) ** 2, rel_tol=1e-5)
    assert fast > slow

    before = total.clone()
    total += _step_reward(env)
    assert total[0].item() > before[0].item()  # linear hold has started
    assert total[0].item() - before[0].item() < 0.04
    assert math.isclose(total[1].item() - before[1].item(), 0.032, rel_tol=1e-5)


def test_hold_is_linear_totals_eight_and_resets_on_interruption():
    env = _Env()
    _set_valid(env, 0, True)
    for _ in range(20):
        _step_reward(env)

    hold_total = 0.0
    for _ in range(50):
        hold_total += _step_reward(env)[0].item()
    assert math.isclose(hold_total, 1.6, rel_tol=1e-5)

    _set_valid(env, 0, False)
    assert _step_reward(env)[0].item() == 0.0
    assert env._v62_hold_steps[0].item() == 0
    _set_valid(env, 0, True)
    hold_total = _step_reward(env)[0].item()
    assert math.isclose(hold_total, 0.032, rel_tol=1e-5)

    for _ in range(249):
        hold_total += _step_reward(env)[0].item()
    assert math.isclose(hold_total, 8.0, rel_tol=1e-5)
    assert env._v62_phase[0].item() == mdp._V62_COMPLETE


def test_timeout_penalty_grows_with_pose_error_and_tripod_is_worst():
    env = _Env(n=3)
    for env_id in range(3):
        _set_valid(env, env_id, True)
    # Correct pose but moving too fast: valid recovery remains false while e_pose is zero.
    env.scene.asset.data.root_link_lin_vel_w[0, 0] = 1.0
    _set_tilt(env, 1, 41.0)
    env.scene.asset.data.joint_pos[1, 5] = math.radians(98.0)
    env.scene.sensors["head_ground_contact"].data.found[1] = 1.0
    _set_valid(env, 2, False)

    terminal = torch.zeros(3)
    for _ in range(250):
        terminal += _step_reward(env)
    assert env._v63_recovery_timed_out.all()
    assert terminal[0].item() == -3.0
    assert terminal[2].item() < terminal[0].item()
    assert terminal[1].item() == -18.0
    assert terminal[1].item() < terminal[2].item()


def test_hold_session_deadline_penalizes_current_tripod_not_before_twelve_seconds():
    env = _Env()
    _set_valid(env, 0, True)
    for _ in range(20):
        _step_reward(env)
    assert env._v62_phase[0].item() == mdp._V62_HOLD

    _set_tilt(env, 0, 41.0)
    env.scene.asset.data.joint_pos[0, 5] = math.radians(98.0)
    env.scene.sensors["head_ground_contact"].data.found[0] = 1.0
    terminal = 0.0
    for _ in range(579):
        terminal += _step_reward(env)[0].item()
    assert not env._v63_session_timed_out[0]
    terminal += _step_reward(env)[0].item()
    assert env._v63_session_timed_out[0]
    assert terminal == -18.0


def test_v631_observed_low_tripod_is_penalized_but_prone_contact_is_not():
    env = _Env(n=2)
    for env_id in range(2):
        _set_valid(env, env_id, True)
        env.scene.asset.data.root_link_pos_w[env_id, 2] = 0.06824
        env.scene.asset.data.joint_pos[env_id, 5] = math.radians(55.662)
        env.scene.sensors["head_ground_contact"].data.found[env_id] = 1.0
    _set_tilt(env, 0, 43.153)
    _set_tilt(env, 1, 90.0)
    params = {"tripod_height_min": 0.055, "tripod_tilt_max_deg": 60.0}

    # The old 0.08 m gate misses this exact observed failure geometry.
    assert mdp.v63_signed_pose_reward(env)[0].item() == 0.0
    score = mdp.v63_signed_pose_reward(env, **params)
    assert score[0].item() == -6.0
    assert score[1].item() == 0.0


def test_v631_potential_wrappers_cancel_reward_manager_dt():
    env = _Env()
    _set_valid(env, 0, True)
    env.episode_length_buf[0] = 2
    assert mdp.v63_upright_progress_reward(env)[0].item() == 0.0
    _set_tilt(env, 0, 90.0)
    upright = mdp.v63_upright_progress_reward(env)[0].item() * env.step_dt
    assert math.isclose(upright, -1.0, rel_tol=1e-5)

    env.scene.asset.data.root_link_pos_w[0, 2] = 0.04
    assert mdp.v63_height_progress_reward(env)[0].item() == 0.0
    env.scene.asset.data.root_link_pos_w[0, 2] = 0.115
    height = mdp.v63_height_progress_reward(env)[0].item() * env.step_dt
    assert math.isclose(height, 0.075, rel_tol=1e-5)
