"""Reward shaping guards.

A hand-written reward is normally broken in a way a return curve cannot reveal: the
number goes up while the agent optimizes something nobody wanted. These tests pin the
ordering the reward is *supposed* to express, so a future tweak that accidentally makes
parking or spinning profitable fails here instead of after a six-hour training run.

Two of the guards exist because the hole was found the hard way. `phase1_v1` learned to
orbit open patches at full throttle under a 2 s displacement window, and a per-cell
coverage grid tried as the fix scored a slow weave above a straight line.
"""

import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import yaml

from selfdrive.config import load_env_config
from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.envs.rewards import RewardConfig, RewardFunction
from selfdrive.world.generators import ArenaParams
from selfdrive.world.geometry import World

DT = 1.0 / 30.0
BOUNDS = (-30.0, -30.0, 30.0, 30.0)
OPEN_FIELD = World(segments=[[-60, -60, 60, -60], [60, -60, 60, 60],
                             [60, 60, -60, 60], [-60, 60, -60, -60]],
                   bounds=(-60.0, -60.0, 60.0, 60.0))
# 30 s: long enough for the widest guarded loop to come round twice, short enough that
# driving straight from the centre does not reach the edge of the field.
GUARD_STEPS = 900


def open_env(max_steps: int = 400, reward: RewardConfig | None = None) -> CarEnv:
    """Empty arena, randomization off, so only the policy differs between runs."""
    cfg = EnvConfig(
        arena=ArenaParams(kind="outdoor"),
        domain_rand=DomainRandConfig(enabled=False),
        reward=reward or RewardConfig(),
        max_steps=max_steps,
    )
    return CarEnv(cfg)


def rollout(env: CarEnv, action, steps: int | None = None, heading: float = 0.0) -> float:
    """`action` is a fixed `[steer, throttle]` or a function of the step index."""
    env.reset(seed=0, options={"world": OPEN_FIELD, "pose": (0.0, 0.0, heading)})
    total = 0.0
    limit = steps if steps is not None else env.cfg.max_steps
    for i in range(limit):
        a = action(i) if callable(action) else action
        _, r, terminated, truncated, _ = env.step(np.asarray(a, dtype=np.float32))
        total += r
        if terminated or truncated:
            break
    return total


@cache
def forward_return(steps: int, heading: float = 0.0) -> float:
    return rollout(open_env(steps), [0.0, 1.0], heading=heading)


# --- the anti-hacking guards -------------------------------------------------

def test_driving_forward_beats_standing_still():
    parked = rollout(open_env(), [0.0, 0.0])
    assert forward_return(400) > parked
    assert parked < 0.0  # standing still is actively punished, not merely unrewarded


def test_driving_forward_beats_driving_in_circles():
    # The classic degenerate solution: full lock plus full throttle never crashes and,
    # under a plain forward-speed reward, scores exactly as well as useful driving.
    circling = rollout(open_env(), [1.0, 1.0])
    assert circling < 0.5 * forward_return(400)  # by a wide margin, not a rounding error


@pytest.mark.parametrize("steer", [0.15, 0.3, 0.6])
def test_driving_forward_beats_wide_fast_loops(steer):
    """What `phase1_v1` actually did: find an open patch, hold a partial lock, floor it.
    At steer 0.15 the loop is ~7 m across and takes ~16 s to close, far beyond what the
    old 2 s displacement window could see."""
    loop = rollout(open_env(GUARD_STEPS), [steer, 1.0])
    assert loop < 0.6 * forward_return(GUARD_STEPS)


@pytest.mark.parametrize("heading", [0.0, math.pi / 8, math.pi / 4])
@pytest.mark.parametrize("amplitude,period_steps", [(0.5, 60), (0.3, 90)])
def test_holding_a_line_beats_a_slow_weave(amplitude, period_steps, heading):
    """The other `phase1_v1` habit: a slow side-to-side sweep, too gradual for the
    per-step steering-change penalty to notice. Checked at several headings because a
    coverage measure that is not rotation-invariant pays the sweep more than the line."""
    weave = rollout(
        open_env(GUARD_STEPS),
        lambda i: [amplitude * math.sin(2.0 * math.pi * i / period_steps), 1.0],
        heading=heading,
    )
    assert weave < forward_return(GUARD_STEPS, heading)


def test_driving_forward_beats_shuffling_on_the_spot():
    shuffle = rollout(open_env(), lambda i: [0.0, 1.0 if (i // 10) % 2 == 0 else -1.0])
    assert shuffle < forward_return(400)


def test_reversing_is_never_more_profitable_than_going_forward():
    reverse = rollout(open_env(), [0.0, -1.0])
    assert reverse < forward_return(400)


def test_reversing_still_beats_sitting_wedged():
    # The behaviour the whole Phase 1 requirement rests on: when forward is impossible,
    # backing out has to be the better option, or the car will simply sit there.
    reverse = rollout(open_env(), [0.0, -1.0])
    parked = rollout(open_env(), [0.0, 0.0])
    assert reverse > parked


def test_braking_counts_as_a_reverse_command_but_not_as_backing():
    # phase1_mem_v2's "reverse 32 %" was mostly braking: the car rolled backwards on 5 %.
    env = open_env(max_steps=61)
    env.reset(seed=0, options={"world": OPEN_FIELD, "pose": (0.0, 0.0, 0.0)})
    for i in range(61):
        action = np.asarray([0.0, 1.0 if i < 60 else -1.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
    assert truncated and not terminated
    metrics = info["episode_metrics"]
    assert metrics["reverse_frac"] > 0.0
    assert metrics["backing_frac"] == 0.0


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

def line(speed: float, steps: int, heading: float = 0.0, start=(0.123, 0.456)):
    """Poses along a straight line. The odd start keeps it off raster pixel centres."""
    c, s = math.cos(heading), math.sin(heading)
    return [(start[0] + speed * i * DT * c, start[1] + speed * i * DT * s, heading)
            for i in range(steps + 1)]


def circle(radius: float, speed: float, laps: float):
    """Poses on a counter-clockwise circle starting at the origin heading +x."""
    omega = speed / radius
    steps = round(laps * 2.0 * math.pi * radius / speed / DT)
    return [(radius * math.sin(omega * i * DT), radius * (1.0 - math.cos(omega * i * DT)),
             omega * i * DT) for i in range(steps + 1)]


def drive(fn: RewardFunction, poses, clearance: float = math.inf, throttle: float = 1.0):
    """Feed a reward function a pose sequence; return the terms for every step."""
    x, y, theta = poses[0]
    fn.reset(x, y, theta, BOUNDS)
    return [
        fn(x=x, y=y, theta=theta, throttle_cmd=throttle, steer_cmd=0.0,
           prev_throttle_cmd=throttle, prev_steer_cmd=0.0,
           clearance=clearance, collided=False, dt=DT)[1]
        for x, y, theta in poses[1:]
    ]


@pytest.mark.parametrize("heading_deg", [0.0, 15.0, 30.0, 45.0])
def test_explore_pays_one_unit_per_metre_at_any_heading(heading_deg):
    terms = drive(RewardFunction(), line(1.0, 600, math.radians(heading_deg)))
    assert sum(t.explore for t in terms) / 20.0 == pytest.approx(1.0, abs=0.05)


def test_explore_pays_for_the_first_lap_only():
    radius = 1.5
    terms = drive(RewardFunction(), circle(radius, 1.0, laps=2.0))
    half = len(terms) // 2
    first = sum(t.explore for t in terms[:half])
    second = sum(t.explore for t in terms[half:])
    assert first == pytest.approx(2.0 * math.pi * radius, rel=0.1)
    assert second < 0.02 * first


def test_ground_pays_again_once_revisit_s_has_passed():
    out = line(1.0, 90)  # 3 m out along +x, then straight back
    path = out + [(x, y, math.pi) for x, y, _ in reversed(out[:-1])]

    def paid_on_the_way_back(revisit_s: float) -> float:
        terms = drive(RewardFunction(RewardConfig(revisit_s=revisit_s)), path)
        return sum(t.explore for t in terms[len(out):])

    assert paid_on_the_way_back(60.0) < 0.05
    assert paid_on_the_way_back(1.0) > 1.0  # ground left more than a second ago pays again


def test_coverage_is_swept_area_not_path_length():
    fn = RewardFunction()
    path_m = 20.0
    drive(fn, line(1.0, round(path_m / DT)))
    straight = fn.coverage_m2
    drive(fn, circle(1.0, 1.0, laps=path_m / (2.0 * math.pi)))
    looped = fn.coverage_m2
    swath = 2.0 * fn.c.explore_radius
    assert straight == pytest.approx(path_m * swath, rel=0.1)
    assert looped < 0.5 * straight


def test_coverage_raster_is_where_the_car_went():
    fn = RewardFunction()
    poses = line(1.0, 150)  # 5 m along +x
    drive(fn, poses)
    age, (ox, oy), res = fn.coverage_raster()

    def age_at(x, y):
        return float(age[round((x - ox) / res), round((y - oy) / res)])

    x_start, y_start, _ = poses[0]
    x_end, y_end, _ = poses[-1]
    assert age_at(x_end, y_end) <= 2 * DT + 1e-6  # just stamped
    assert age_at(x_start, y_start) > 4.0  # stamped ~5 s ago
    assert math.isinf(age_at(x_end, y_end + 1.0))  # beside the swath
    assert math.isinf(age_at(x_end + 1.0, y_end))  # not reached yet


def test_lateral_penalty_is_speed_squared_over_radius():
    c = RewardConfig()
    radius, speed = 1.0, 1.5
    terms = drive(RewardFunction(c), circle(radius, speed, laps=0.5))
    expected = -c.w_lateral * ((speed**2 / radius) / c.lateral_accel_ref) ** 2
    assert terms[-1].lateral == pytest.approx(expected, rel=0.01)
    assert all(t.lateral == 0.0 for t in drive(RewardFunction(c), line(speed, 30)))


def test_stall_fires_on_displacement_not_on_zero_speed():
    c = RewardConfig()
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    # Wheels spinning, throttle pinned, car wedged and creeping by a millimetre a step.
    for i in range(c.stall_window + 5):
        _, terms = fn(x=0.0005 * i, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.0,
                      prev_throttle_cmd=1.0, prev_steer_cmd=0.0,
                      clearance=0.05, collided=False, dt=DT)
    assert terms.stall < 0.0
    assert fn.stalled_steps > 0


def test_stall_clears_once_the_car_moves_again():
    c = RewardConfig()
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    for _ in range(c.stall_window + 5):
        fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.0,
           prev_throttle_cmd=1.0, prev_steer_cmd=0.0,
           clearance=0.05, collided=False, dt=DT)
    assert fn.stalled_steps > 0
    for i in range(c.stall_window + 1):
        fn(x=1.0 * i * DT * 10, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.0,
           prev_throttle_cmd=1.0, prev_steer_cmd=0.0, clearance=math.inf, collided=False, dt=DT)
    assert fn.stalled_steps == 0


def test_is_stalled_needs_sustained_stalling():
    c = RewardConfig(stall_limit=10)
    fn = RewardFunction(c)
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    for _ in range(c.stall_window + 5):
        fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.0,
           prev_throttle_cmd=1.0, prev_steer_cmd=0.0,
           clearance=0.05, collided=False, dt=DT)
    assert not fn.is_stalled
    for _ in range(c.stall_limit):
        fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.0,
           prev_throttle_cmd=1.0, prev_steer_cmd=0.0,
           clearance=0.05, collided=False, dt=DT)
    assert fn.is_stalled


def test_proximity_barrier_is_smooth_and_one_sided():
    c = RewardConfig()

    def proximity(clearance):
        return drive(RewardFunction(c), line(1.0, 5), clearance=clearance)[-1].proximity

    far = proximity(c.safe_distance * 2)
    edge = proximity(c.safe_distance)
    close = proximity(c.safe_distance * 0.25)
    touching = proximity(0.0)
    assert far == 0.0 and edge == 0.0  # nothing beyond the safe distance
    assert touching < close < 0.0  # and it grows the closer you get


def test_infinite_clearance_is_handled():
    terms = drive(RewardFunction(), line(1.0, 5), clearance=math.inf)[-1]
    assert terms.proximity == 0.0
    assert math.isfinite(terms.total)


def test_oscillation_penalty_tracks_steering_change():
    fn = RewardFunction(RewardConfig())
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    _, smooth = fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.5,
                   prev_throttle_cmd=1.0, prev_steer_cmd=0.5,
                   clearance=math.inf, collided=False, dt=DT)
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    _, jerky = fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=1.0,
                  prev_throttle_cmd=1.0, prev_steer_cmd=-1.0,
                  clearance=math.inf, collided=False, dt=DT)
    assert smooth.oscillation == 0.0
    assert jerky.oscillation < 0.0


def test_throttle_oscillation_is_off_unless_configured_and_tracks_throttle_change():
    def oscillation(config, throttle, prev_throttle):
        fn = RewardFunction(config)
        fn.reset(0.0, 0.0, 0.0, BOUNDS)
        _, terms = fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=throttle,
                      prev_throttle_cmd=prev_throttle, steer_cmd=0.0, prev_steer_cmd=0.0,
                      clearance=math.inf, collided=False, dt=DT)
        return terms.oscillation

    assert oscillation(RewardConfig(), 1.0, -1.0) == 0.0
    smooth = RewardConfig(w_throttle_oscillation=0.15)
    assert oscillation(smooth, 0.5, 0.5) == 0.0
    assert oscillation(smooth, 1.0, -1.0) == pytest.approx(-0.3)


def test_reverse_costs_only_when_reversing():
    fn = RewardFunction(RewardConfig())
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    _, fwd = fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=1.0, steer_cmd=0.0,
                prev_throttle_cmd=1.0, prev_steer_cmd=0.0,
                clearance=math.inf, collided=False, dt=DT)
    fn.reset(0.0, 0.0, 0.0, BOUNDS)
    _, rev = fn(x=0.0, y=0.0, theta=0.0, throttle_cmd=-1.0, steer_cmd=0.0,
                prev_throttle_cmd=-1.0, prev_steer_cmd=0.0,
                clearance=math.inf, collided=False, dt=DT)
    assert fwd.reverse == 0.0
    assert rev.reverse < 0.0


def test_terms_sum_to_the_total():
    fn = RewardFunction(RewardConfig())
    drive(fn, line(1.0, 80), clearance=0.3)
    total, terms = fn(x=3.0, y=0.456, theta=0.1, throttle_cmd=-0.5, steer_cmd=0.3,
                      prev_throttle_cmd=-0.5, prev_steer_cmd=0.0,
                      clearance=0.3, collided=True, dt=DT)
    assert total == pytest.approx(terms.total)
    assert total == pytest.approx(sum(terms.as_dict().values()))


# --- the retrace charge --------------------------------------------------------

RETRACE = RewardConfig(w_retrace=0.5)
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_retrace_is_off_unless_configured():
    assert RewardConfig().w_retrace == 0.0
    assert all(t.retrace == 0.0 for t in drive(RewardFunction(), circle(1.5, 1.0, laps=2.0)))


@pytest.mark.parametrize("speed", [0.35, 1.0])
@pytest.mark.parametrize("heading_deg", [0.0, 15.0, 30.0, 45.0])
def test_new_ground_is_never_charged_as_retraced(heading_deg, speed):
    fn = RewardFunction(RETRACE)
    terms = drive(fn, line(speed, round(6.0 / speed / DT), math.radians(heading_deg)))
    assert all(t.retrace == 0.0 for t in terms)
    assert fn.retrace_frac == 0.0


def test_a_second_lap_is_charged_what_the_first_lap_paid():
    fn = RewardFunction(RETRACE)
    terms = drive(fn, circle(1.5, 1.0, laps=2.0))
    paid = sum(t.explore for t in terms)
    charged = sum(t.retrace for t in terms)
    # The first lap is paid a little short where it closes on its own start.
    assert charged == pytest.approx(-RETRACE.w_retrace * paid, rel=0.1)
    assert fn.retrace_frac == pytest.approx(0.5, abs=0.03)


def test_driving_back_over_your_own_path_is_charged():
    out = line(1.0, 90)  # 3 m out along +x, then straight back
    path = out + [(x, y, math.pi) for x, y, _ in reversed(out[:-1])]
    terms = drive(RewardFunction(RETRACE), path)
    back = sum(t.retrace for t in terms[len(out):])
    assert back == pytest.approx(-RETRACE.w_retrace * 3.0, rel=0.15)


def test_retrace_charge_lowers_loops_and_leaves_a_straight_line_alone():
    """`phase1_v2` circled open patches, because explore-only pay makes every lap after
    the first free. With `w_retrace` set a loop must cost, while straight driving, which
    never crosses its own track, earns exactly what it did."""
    straight = rollout(open_env(GUARD_STEPS, RETRACE), [0.0, 1.0])
    assert straight == pytest.approx(forward_return(GUARD_STEPS))
    loop = [0.6, 1.0]
    charged = rollout(open_env(GUARD_STEPS, RETRACE), loop)
    assert charged < rollout(open_env(GUARD_STEPS), loop) - 5.0


def test_retrace_config_is_phase1_with_only_w_retrace_changed():
    base = yaml.safe_load((CONFIGS / "env_phase1.yaml").read_text())
    retrace = yaml.safe_load((CONFIGS / "env_phase1_retrace.yaml").read_text())
    assert base["reward"].pop("w_retrace") == 0.0
    assert retrace["reward"].pop("w_retrace") == RETRACE.w_retrace
    assert retrace == base
    loaded = load_env_config(CONFIGS / "env_phase1_retrace.yaml")
    assert loaded.reward.w_retrace == RETRACE.w_retrace


def test_crash25_config_is_phase1_with_only_collision_penalty_changed():
    base = yaml.safe_load((CONFIGS / "env_phase1.yaml").read_text())
    crash25 = yaml.safe_load((CONFIGS / "env_phase1_crash25.yaml").read_text())
    assert base["reward"].pop("collision_penalty") == 100.0
    assert crash25["reward"].pop("collision_penalty") == 25.0
    assert crash25 == base


def test_smooth_config_is_phase1_with_only_the_oscillation_weights_changed():
    base = yaml.safe_load((CONFIGS / "env_phase1.yaml").read_text())
    smooth = yaml.safe_load((CONFIGS / "env_phase1_smooth.yaml").read_text())
    for key, before, after in (("w_oscillation", 0.05, 0.15),
                               ("w_throttle_oscillation", 0.0, 0.15)):
        assert base["reward"].pop(key) == before
        assert smooth["reward"].pop(key) == after
    assert smooth == base


@pytest.mark.parametrize("config", ["env_phase1_light.yaml", "env_phase1_smooth.yaml",
                                    "env_phase1_smooth_retrace.yaml"])
def test_smooth_reward_keeps_backing_out_of_a_trap_and_holding_the_throttle(config):
    """Charging throttle changes must not make sitting wedged cheaper than reversing, and
    pumping the throttle must still lose to holding it."""
    smooth = load_env_config(CONFIGS / config).reward
    parked = rollout(open_env(reward=smooth), [0.0, 0.0])
    reverse = rollout(open_env(reward=smooth), [0.0, -1.0])
    forward = rollout(open_env(reward=smooth), [0.0, 1.0])
    pumping = rollout(open_env(reward=smooth),
                      lambda i: [0.0, 1.0 if (i // 10) % 2 == 0 else -1.0])
    assert parked < reverse < forward
    assert pumping < forward


def test_smooth_retrace_config_is_smooth_with_only_w_retrace_changed():
    smooth = yaml.safe_load((CONFIGS / "env_phase1_smooth.yaml").read_text())
    retrace = yaml.safe_load((CONFIGS / "env_phase1_smooth_retrace.yaml").read_text())
    assert smooth["reward"].pop("w_retrace") == 0.0
    assert retrace["reward"].pop("w_retrace") == RETRACE.w_retrace
    assert retrace == smooth


def test_light_config_is_phase1_with_only_throttle_oscillation_changed():
    base = yaml.safe_load((CONFIGS / "env_phase1.yaml").read_text())
    light = yaml.safe_load((CONFIGS / "env_phase1_light.yaml").read_text())
    assert base["reward"].pop("w_throttle_oscillation") == 0.0
    assert light["reward"].pop("w_throttle_oscillation") == 0.05
    assert light == base
