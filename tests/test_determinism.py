"""Seeding and reproducibility.

The written report will claim specific numbers from specific runs. That claim is only
meaningful if a seed reproduces an episode exactly - arena, randomized car, and every
sensor noise draw. Randomization sampled from an unseeded global RNG is the usual way
this breaks, and it is invisible until someone tries to reproduce a result.
"""

import numpy as np

from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.randomize import DomainRandConfig


def rollout(seed: int, steps: int = 120, dr: bool = True):
    e = CarEnv(EnvConfig(domain_rand=DomainRandConfig(enabled=dr), max_steps=steps))
    obs, _ = e.reset(seed=seed)
    rng = np.random.default_rng(0)  # identical action sequence for every rollout
    obs_trace, rew_trace = [obs], []
    for _ in range(steps):
        obs, r, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
        obs_trace.append(obs)
        rew_trace.append(r)
        if terminated or truncated:
            break
    params = (e.car.p.wheelbase, e.car.p.accel_tau, e.depth.p.fov_deg, e.dt)
    return np.array(obs_trace), np.array(rew_trace), params


def test_same_seed_reproduces_the_episode_exactly():
    a_obs, a_rew, a_par = rollout(7)
    b_obs, b_rew, b_par = rollout(7)
    assert a_par == b_par
    np.testing.assert_array_equal(a_obs, b_obs)
    np.testing.assert_array_equal(a_rew, b_rew)


def test_different_seeds_give_different_episodes():
    a_obs, _, a_par = rollout(1)
    b_obs, _, b_par = rollout(2)
    assert a_par != b_par  # the randomized car itself differs
    assert a_obs.shape != b_obs.shape or not np.array_equal(a_obs, b_obs)


def test_domain_randomization_is_drawn_from_the_seeded_generator():
    # If any of this came from a global RNG, two same-seeded envs would diverge.
    seen = {rollout(11)[2] for _ in range(3)}
    assert len(seen) == 1


def test_randomization_actually_varies_when_enabled():
    params = {rollout(s)[2] for s in range(6)}
    assert len(params) == 6


def test_disabling_randomization_pins_the_parameters():
    params = {rollout(s, dr=False)[2] for s in range(6)}
    assert len(params) == 1  # identical car every episode, only the arena changes


def test_reset_seed_none_does_not_reuse_the_previous_seed():
    e = CarEnv(EnvConfig(max_steps=10))
    e.reset(seed=99)
    first = e.world.segments.copy()
    e.reset()  # no seed: the generator should advance, not restart
    assert not np.array_equal(first, e.world.segments)


def test_spawn_observation_is_saturated_in_open_space():
    """Documents why the test above compares arenas rather than observations.

    A car spawned with clearance on all sides reads max range in every bucket, so two
    completely different arenas produce an identical first observation. Anything
    asserting "these episodes differ" has to look at state, not at the first frame.
    """
    e = CarEnv(EnvConfig(max_steps=10))
    obs, _ = e.reset(seed=99)
    frame = obs[: e.cfg.obs.per_frame]
    assert np.all(frame[: e.cfg.obs.n_depth] == 1.0)  # nothing within range
    assert frame[e.cfg.obs.n_depth] == -1.0  # stopped, so depth confidence is zero
