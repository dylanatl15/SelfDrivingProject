"""Reward shaping guards.

A hand-written reward is normally broken in a way a return curve cannot reveal: the
number goes up while the agent optimizes something nobody wanted. These tests pin the
ordering the reward is *supposed* to express, so a future tweak that accidentally makes
parking or spinning profitable fails here instead of after a six-hour training run.
"""

import math

import numpy as np
import pytest

from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.envs.rewards import RewardConfig, RewardFunction
from selfdrive.world.generators import ArenaParams
from selfdrive.world.geometry import World

DT = 1.0 / 30.0
OPEN_FIELD = World(segments=[[-60, -60, 60, -60], [60, -60, 60, 60],
                             [60, 60, -60, 60], [-60, 60, -60, -60]],
                   bounds=(-60.0, -60.0, 60.0, 60.0))


def open_env(max_steps: int = 400) -> CarEnv:
    """Empty arena, randomization off, so only the policy differs between runs."""
    cfg = EnvConfig(
        arena=ArenaParams(kind="outdoor"),
        domain_rand=DomainRandConfig(enabled=False),
        max_steps=max_steps,
    )
    return CarEnv(cfg)


def rollout(env: CarEnv, action, steps: int | None = None) -> float:
    obs, _ = env.reset(seed=0, options={"world": OPEN_FIELD, "pose": (0.0, 0.0, 0.0)})
    total = 0.0
    limit = steps if steps is not None else env.cfg.max_steps
    for _ in range(limit):
        _, r, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        total += r
        if terminated or truncated:
            break
    return total


# --- the anti-hacking guards -------------------------------------------------

def test_driving_forward_beats_standing_still():
    forward = rollout(open_env(), [0.0, 1.0])
    parked = rollout(open_env(), [0.0, 0.0])
    assert forward > parked
    assert parked < 0.0  # standing still is actively punished, not merely unrewarded


def test_driving_forward_beats_driving_in_circles():
    # The classic degenerate solution: full lock plus full throttle never crashes and,
    # under a plain forward-speed reward, scores exactly as well as useful driving.
    forward = rollout(open_env(), [0.0, 1.0])
    circling = rollout(open_env(), [1.0, 1.0])
    assert circling < forward
    assert circling < 0.5 * forward  # and by a wide margin, not a rounding error


def test_driving_forward_beats_shuffling_on_the_spot():
    env = open_env()
    obs, _ = env.reset(seed=0, options={"world": OPEN_FIELD, "pose": (0.0, 0.0, 0.0)})
    total = 0.0
    for i in range(400):
        throttle = 1.0 if (i // 10) % 2 == 0 else -1.0
        _, r, terminated, truncated, _ = env.step(np.array([0.0, throttle], np.float32))
        total += r
        if terminated or truncated:
            break
    assert total < rollout(open_env(), [0.0, 1.0])


def test_reversing_is_never_more_profitable_than_going_forward():
    forward = rollout(open_env(), [0.0, 1.0])
    reverse = rollout(open_env(), [0.0, -1.0])
    assert reverse < forward


def test_reversing_still_beats_sitting_wedged():
    # The behaviour the whole Phase 1 requirement rests on: when forward is impossible,
    # backing out has to be the better option, or the car will simply sit there.
    reverse = rollout(open_env(), [0.0, -1.0])
    parked = rollout(open_env(), [0.0, 0.0])
    assert reverse > parked


def test_crashing_is_catastrophic():
    wall = World(segments=[[3.0, -5.0, 3.0, 5.0]], bounds=(-10.0, -10.0, 10.0, 10.0))
    env = open_env(max_steps=400)
    env.reset(seed=0, options={"world": wall, "pose": (0.0, 0.0, 0.0)})
    total, terminated = 0.0, False
    for _ in range(400):
        _, r, terminated, truncated, _ = env.step(np.array([0.0, 1.0], np.float32))
        total += r
        if terminated or truncated:
            break
    assert terminated  # it did hit the wall
    assert total < 0.0  # and a few metres of progress does not pay for it


# --- the reward function in isolation ----------------------------------------

def straight_line(fn: RewardFunction, speed: float, steps: int, **kw):
    """Drive a reward function along +x without an environment."""
    fn.reset(0.0, 0.0)
    last = None
    for i in range(steps):
        last = fn(
            x=speed * (i + 1) * DT, y=0.0,
            throttle_cmd=kw.get("throttle", 1.0), steer_cmd=0.0, prev_steer_cmd=0.0,
            clearance=kw.get("clearance", math.inf), collided=False, dt=DT,
        )
    return last


def test_progress_term_measures_net_speed():
    fn = RewardFunction(RewardConfig())
    _, terms = straight_line(fn, speed=1.0, steps=120)
    # 1 m/s sustained for longer than the window: one metre per second of reward.
    assert terms.progress == pytest.approx(1.0 * DT, rel=1e-6)


def test_progress_term_ignores_a_closed_loop():
    """A car back where it started earns nothing, however fast it got there."""
    c = RewardConfig()
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0)
    radius, steps = 0.5, c.progress_window
    last = None
    for i in range(steps + 1):
        a = 2.0 * math.pi * i / steps
        last = fn(
            x=radius * math.sin(a), y=radius * (1.0 - math.cos(a)),
            throttle_cmd=1.0, steer_cmd=1.0, prev_steer_cmd=1.0,
            clearance=math.inf, collided=False, dt=DT,
        )
    assert last[1].progress < 0.05 * (1.0 * DT)


def test_stall_fires_on_displacement_not_on_zero_speed():
    c = RewardConfig()
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0)
    # Wheels spinning, throttle pinned, car wedged and creeping by a millimetre a step.
    for i in range(c.stall_window + 5):
        _, terms = fn(x=0.0005 * i, y=0.0, throttle_cmd=1.0, steer_cmd=0.0,
                      prev_steer_cmd=0.0, clearance=0.05, collided=False, dt=DT)
    assert terms.stall < 0.0
    assert fn.stalled_steps > 0


def test_stall_clears_once_the_car_moves_again():
    c = RewardConfig()
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0)
    for _ in range(c.stall_window + 5):
        fn(x=0.0, y=0.0, throttle_cmd=1.0, steer_cmd=0.0, prev_steer_cmd=0.0,
           clearance=0.05, collided=False, dt=DT)
    assert fn.stalled_steps > 0
    for i in range(c.stall_window + 1):
        fn(x=1.0 * i * DT * 10, y=0.0, throttle_cmd=1.0, steer_cmd=0.0,
           prev_steer_cmd=0.0, clearance=math.inf, collided=False, dt=DT)
    assert fn.stalled_steps == 0


def test_is_stalled_needs_sustained_stalling():
    c = RewardConfig(stall_limit=10)
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0)
    for _ in range(c.stall_window + 5):
        fn(x=0.0, y=0.0, throttle_cmd=1.0, steer_cmd=0.0, prev_steer_cmd=0.0,
           clearance=0.05, collided=False, dt=DT)
    assert not fn.is_stalled
    for _ in range(c.stall_limit):
        fn(x=0.0, y=0.0, throttle_cmd=1.0, steer_cmd=0.0, prev_steer_cmd=0.0,
           clearance=0.05, collided=False, dt=DT)
    assert fn.is_stalled


def test_proximity_barrier_is_smooth_and_one_sided():
    c = RewardConfig()
    fn = RewardFunction(c)
    far = straight_line(fn, 1.0, 5, clearance=c.safe_distance * 2)[1].proximity
    edge = straight_line(fn, 1.0, 5, clearance=c.safe_distance)[1].proximity
    close = straight_line(fn, 1.0, 5, clearance=c.safe_distance * 0.25)[1].proximity
    touching = straight_line(fn, 1.0, 5, clearance=0.0)[1].proximity
    assert far == 0.0 and edge == 0.0  # nothing beyond the safe distance
    assert touching < close < 0.0  # and it grows the closer you get


def test_infinite_clearance_is_handled():
    fn = RewardFunction(RewardConfig())
    _, terms = straight_line(fn, 1.0, 5, clearance=math.inf)
    assert terms.proximity == 0.0
    assert math.isfinite(terms.total)


def test_oscillation_penalty_tracks_steering_change():
    fn = RewardFunction(RewardConfig())
    fn.reset(0.0, 0.0)
    _, smooth = fn(x=0.0, y=0.0, throttle_cmd=1.0, steer_cmd=0.5, prev_steer_cmd=0.5,
                   clearance=math.inf, collided=False, dt=DT)
    fn.reset(0.0, 0.0)
    _, jerky = fn(x=0.0, y=0.0, throttle_cmd=1.0, steer_cmd=1.0, prev_steer_cmd=-1.0,
                  clearance=math.inf, collided=False, dt=DT)
    assert smooth.oscillation == 0.0
    assert jerky.oscillation < 0.0


def test_reverse_costs_only_when_reversing():
    fn = RewardFunction(RewardConfig())
    fn.reset(0.0, 0.0)
    _, fwd = fn(x=0.0, y=0.0, throttle_cmd=1.0, steer_cmd=0.0, prev_steer_cmd=0.0,
                clearance=math.inf, collided=False, dt=DT)
    fn.reset(0.0, 0.0)
    _, rev = fn(x=0.0, y=0.0, throttle_cmd=-1.0, steer_cmd=0.0, prev_steer_cmd=0.0,
                clearance=math.inf, collided=False, dt=DT)
    assert fwd.reverse == 0.0
    assert rev.reverse < 0.0


def test_terms_sum_to_the_total():
    fn = RewardFunction(RewardConfig())
    total, terms = straight_line(fn, 1.0, 80, clearance=0.3)
    assert total == pytest.approx(terms.total)
    assert total == pytest.approx(sum(terms.as_dict().values()))
