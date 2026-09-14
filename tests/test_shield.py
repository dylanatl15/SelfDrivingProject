"""The speed shield (`envs/shield.py`, `shield.enabled`).

The crash probe found the Stage 2 policies driving into obstacles their sensors had already
reported. These tests pin the rule the phone mirrors (`docs/shield.md`), that the shield stops
the car short of walls ahead and behind that it would otherwise hit, and that a shield which
never binds changes nothing, so turning it on changes only what it is meant to.
"""

import math

import numpy as np
import pytest

from selfdrive.config import env_config_from_dict
from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.envs.shield import (BRAKED, CAPPED, PASSED, ShieldConfig, allowed_speed,
                                   reported_distances, shield_throttle)
from selfdrive.world.geometry import World

S = ShieldConfig(enabled=True)  # margin 0.15 m, horizon 0.5 s, no floor, brake 0.3
V_FWD, V_REV = 1.5, 0.6
NAMES = ("front", "left", "right", "back")


def shield(throttle, speed, forward, back):
    return shield_throttle(throttle, speed, forward, back, S, V_FWD, V_REV)


def test_the_allowed_speed_closes_the_gap_beyond_the_margin_within_the_horizon():
    assert allowed_speed(0.10, S) == 0.0
    assert allowed_speed(0.15, S) == 0.0
    assert allowed_speed(0.65, S) == pytest.approx(1.0)
    floored = ShieldConfig(floor=0.1)
    assert allowed_speed(0.16, floored) == pytest.approx(0.1)
    assert allowed_speed(0.15, floored) == 0.0  # the floor never reaches inside the margin


def test_forward_throttle_is_capped_to_the_allowed_speed():
    throttle, did = shield(1.0, 0.5, 0.65, 4.0)
    assert did == CAPPED and throttle == pytest.approx(1.0 / V_FWD)
    assert shield(0.3, 0.5, 0.65, 4.0) == (0.3, PASSED)
    assert shield(1.0, 0.0, 0.10, 4.0) == (0.0, CAPPED)


def test_too_fast_for_the_gap_brakes_against_the_motion():
    # 1.0 m/s is allowed: 0.04 m/s over it is only capped, 0.2 m/s over it brakes.
    assert shield(1.0, 1.04, 0.65, 4.0)[1] == CAPPED
    assert shield(1.0, 1.2, 0.65, 4.0) == (-0.3, BRAKED)


def test_reversing_reads_the_back_ultrasonic_only():
    throttle, did = shield(-1.0, -0.2, 0.2, 0.35)
    assert did == CAPPED and throttle == pytest.approx(-0.4 / V_REV)
    assert shield(-1.0, -0.5, 0.2, 0.35) == (0.3, BRAKED)
    # A wall close ahead does not stop the car backing away from it.
    assert shield(-0.5, 0.0, 0.16, 4.0) == (-0.5, PASSED)


def test_coasting_is_never_touched():
    assert shield(0.0, 1.5, 0.05, 0.05) == (0.0, PASSED)


def test_forward_distance_is_the_nearest_depth_bucket_or_front_ultrasonic():
    depth = np.array([3.0, 2.5, 0.9, 4.0])
    ultra = np.array([0.7, 0.2, 0.3, 1.1])  # the side sensors are never read
    assert reported_distances(depth, ultra, NAMES) == (0.7, 1.1)
    assert reported_distances(depth, ultra[1:], NAMES[1:]) == (0.9, 1.1)  # no front sensor
    assert reported_distances(depth, ultra[:3], NAMES[:3]) == (0.7, math.inf)


def walled(*walls) -> World:
    box = [[0, 0, 30, 0], [30, 0, 30, 30], [30, 30, 0, 30], [0, 30, 0, 0]]
    return World(segments=box + [list(w) for w in walls], bounds=(0.0, 0.0, 30.0, 30.0))


def drive(config: ShieldConfig, world: World, throttle: float, steps: int = 600):
    env = CarEnv(EnvConfig(shield=config, domain_rand=DomainRandConfig(enabled=False),
                           max_steps=steps))
    env.reset(seed=0, options={"world": world, "pose": (15.0, 15.0, 0.0)})
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(np.array([0.0, throttle], np.float32))
        if terminated or truncated:
            break
    s, p = env.car.state, env.car.p
    return terminated, env.world.clearance(s.x, s.y, s.theta, p.length, p.width), info


@pytest.mark.parametrize("throttle, wall", [
    (1.0, (18.0, 10.0, 18.0, 20.0)),   # 2.8 m past the front bumper
    (-1.0, (12.0, 10.0, 12.0, 20.0)),  # 2.8 m past the rear bumper
])
def test_the_shield_stops_the_car_short_of_a_wall_it_would_hit(throttle, wall):
    crashed, _, _ = drive(ShieldConfig(), walled(wall), throttle)
    assert crashed
    crashed, clearance, info = drive(S, walled(wall), throttle)
    assert not crashed
    assert 0.0 < clearance < 0.4
    assert info["episode_metrics"]["shield_capped_frac"] > 0.0


def rollout(config: ShieldConfig, seed: int = 7, steps: int = 300):
    env = CarEnv(EnvConfig(shield=config, max_steps=steps))
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)  # identical action sequence for every rollout
    obs_trace, rew_trace = [obs], []
    for _ in range(steps):
        obs, r, terminated, truncated, info = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        obs_trace.append(obs)
        rew_trace.append(r)
        if terminated or truncated:
            break
    return np.array(obs_trace), np.array(rew_trace), info["episode_metrics"]


def test_a_shield_that_never_binds_changes_nothing():
    never = ShieldConfig(enabled=True, margin=0.0, horizon=1e-9)
    off_obs, off_rew, off = rollout(ShieldConfig())
    on_obs, on_rew, on = rollout(never)
    np.testing.assert_array_equal(off_obs, on_obs)
    np.testing.assert_array_equal(off_rew, on_rew)
    assert on["shield_capped_frac"] == 0.0 and on["shield_braked_frac"] == 0.0
    assert "shield_capped_frac" not in off  # unshielded metrics keep their old columns


def test_the_observation_sees_the_throttle_that_was_sent():
    ahead = walled((15.6, 10.0, 15.6, 20.0))  # 0.4 m past the front bumper
    env = CarEnv(EnvConfig(shield=S, domain_rand=DomainRandConfig(enabled=False)))
    env.reset(seed=0, options={"world": ahead, "pose": (15.0, 15.0, 0.0)})
    for _ in range(15):  # parked while the first depth frames and echoes arrive
        env.step(np.array([0.0, 0.0], np.float32))
    obs, _, _, _, info = env.step(np.array([0.0, 1.0], np.float32))
    o = env.cfg.obs
    last_throttle = (o.frame_stack - 1) * o.per_frame + o.n_depth + 1 + o.n_ultrasonic + 2
    assert info["shield"] == CAPPED and 0.0 < info["throttle_sent"] < 0.5
    assert obs[last_throttle] == pytest.approx(info["throttle_sent"])


def test_the_shield_section_loads_from_yaml_and_is_off_by_default():
    cfg = env_config_from_dict({"shield": {"enabled": True, "margin": 0.12, "floor": 0.05}})
    assert cfg.shield == ShieldConfig(enabled=True, margin=0.12, floor=0.05)
    assert EnvConfig().shield.enabled is False
