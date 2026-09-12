"""Gymnasium API conformance and observation contract."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from selfdrive.config import env_config_from_dict, load_env_config
from selfdrive.envs.car_env import STEER, THROTTLE, CarEnv, EnvConfig
from selfdrive.envs.obs import ObsConfig
from selfdrive.envs.randomize import DomainRandConfig


def env(**kw) -> CarEnv:
    cfg = EnvConfig(domain_rand=DomainRandConfig(enabled=False), max_steps=kw.pop("max_steps", 200))
    return CarEnv(cfg, **kw)


def test_passes_the_gymnasium_env_checker():
    check_env(env(), skip_render_check=True)


def test_observation_size_matches_the_documented_layout():
    c = ObsConfig()
    assert c.per_frame == 17  # 8 depth + 1 confidence + 4 ultrasonic + 4 ego
    assert c.size == 17 * c.frame_stack == 68


def test_spaces():
    e = env()
    assert e.action_space.shape == (2,)
    assert e.observation_space.shape == (ObsConfig().size,)
    assert STEER == 0 and THROTTLE == 1


def test_observations_stay_inside_the_declared_box():
    e = env()
    obs, _ = e.reset(seed=3)
    assert e.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
        assert e.observation_space.contains(obs), "observation escaped its own space"
        if terminated or truncated:
            obs, _ = e.reset()


def test_out_of_range_actions_are_clipped_not_rejected():
    e = env()
    e.reset(seed=0)
    # SB3 can hand back slightly out-of-range actions from a squashed Gaussian.
    obs, r, *_ = e.step(np.array([50.0, -50.0], dtype=np.float32))
    assert np.isfinite(r)
    assert e.observation_space.contains(obs)


def test_step_info_carries_reward_terms():
    e = env()
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.array([0.0, 1.0], np.float32))
    assert set(info["reward_terms"]) == {
        "progress", "reverse", "oscillation", "proximity", "stall", "collision"
    }


def test_episode_metrics_appear_exactly_once_at_the_end():
    e = env(max_steps=60)
    e.reset(seed=0)
    seen = 0
    for _ in range(60):
        _, _, terminated, truncated, info = e.step(np.array([0.0, 1.0], np.float32))
        if "episode_metrics" in info:
            seen += 1
        if terminated or truncated:
            break
    assert seen == 1


def test_episode_metrics_separate_crashing_from_getting_stuck():
    e = env(max_steps=60)
    e.reset(seed=0)
    for _ in range(60):
        _, _, terminated, truncated, info = e.step(np.array([0.0, 1.0], np.float32))
        if terminated or truncated:
            break
    m = info["episode_metrics"]
    assert {"collided", "stuck"} <= set(m)
    assert not (m["collided"] and m["stuck"])  # they are distinct outcomes


def test_truncates_at_max_steps():
    e = env(max_steps=25)
    e.reset(seed=0)
    for _ in range(25):
        _, _, terminated, truncated, _ = e.step(np.zeros(2, np.float32))
        if terminated or truncated:
            break
    assert truncated and e.steps == 25


def test_pose_option_places_the_car_exactly():
    e = env()
    e.reset(seed=0, options={"pose": (1.25, -2.5, 0.75)})
    assert (e.car.state.x, e.car.state.y, e.car.state.theta) == (1.25, -2.5, 0.75)


def test_three_ultrasonic_build_still_fills_the_observation():
    cfg = EnvConfig(domain_rand=DomainRandConfig(enabled=False))
    cfg.ultrasonic.n_sensors = 3
    e = CarEnv(cfg)
    obs, _ = e.reset(seed=0)
    assert e.ultra.n == 3
    assert e.ultra.names == ("left", "right", "back")  # the front is what gets dropped
    assert obs.shape == (ObsConfig().size,)  # fixed width regardless of the build


# --- config plumbing ---------------------------------------------------------

def test_yaml_config_loads_and_converts_degrees():
    cfg = env_config_from_dict({"car": {"max_steer_deg": 30.0}, "dt": 0.02})
    assert cfg.car.max_steer_rad == pytest.approx(np.radians(30.0))
    assert cfg.dt == 0.02


def test_unknown_config_keys_fail_loudly():
    with pytest.raises(ValueError, match="no field"):
        env_config_from_dict({"car": {"wheelbaze": 0.3}})
    with pytest.raises(ValueError, match="unknown config sections"):
        env_config_from_dict({"cars": {}})


def test_shipped_configs_load():
    for path in ("configs/env_phase1.yaml", "configs/env_nodr.yaml"):
        cfg = load_env_config(path)
        assert cfg.obs.size > 0
        CarEnv(cfg).reset(seed=0)
