"""The stopping-path term (`envs/braking.py`, `w_brake`).

waypoint_pay5 crashed at 0.75-0.97 m/s on average while turning, scraping and backing up,
and outran its straight-ahead stopping distance on only about 1 % of forward steps. These
tests pin what the term has to see that a straight-ahead check and the proximity barrier
do not: the arc, the rear, and closing speed rather than distance.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from selfdrive.dynamics.actuators import Actuators
from selfdrive.dynamics.base import CarParams
from selfdrive.envs.braking import arc_pose, brake_shortfall, stopping_distance
from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.envs.rewards import RewardConfig
from selfdrive.world.generators import ArenaParams
from selfdrive.world.geometry import World

P = CarParams()
REACTION = 0.1
MARGIN = 0.05


def shortfall(world: World, pose, speed: float, steer: float = 0.0, p: CarParams = P) -> float:
    x, y, theta = pose
    clearance = world.clearance(x, y, theta, p.length, p.width)
    return brake_shortfall(world, x, y, theta, speed, steer, p, REACTION, MARGIN, 4, clearance)


def wall(*segments) -> World:
    return World(segments=[list(s) for s in segments], bounds=(-5.0, -5.0, 5.0, 5.0))


@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("speed", [0.3, 0.8, 1.5, -0.5])
def test_stopping_distance_is_what_the_drivetrain_covers_at_full_opposite_throttle(speed, tau):
    p = replace(P, accel_tau=tau)
    drive = Actuators(p)
    drive.reset(speed=speed)
    dt, travelled = 1e-4, 0.0
    sign = math.copysign(1.0, speed)
    while drive.speed * sign > 0.0:
        before = drive.speed * sign
        _, after = drive.step(0.0, -sign, dt)
        travelled += 0.5 * (before + max(after * sign, 0.0)) * dt
    assert travelled == pytest.approx(stopping_distance(speed, p, 0.0), rel=5e-3, abs=2e-4)


def test_reacting_adds_the_distance_driven_before_braking_starts():
    assert stopping_distance(1.2, P, 0.2) - stopping_distance(1.2, P, 0.0) == pytest.approx(0.24)
    assert stopping_distance(0.0, P, 0.2) == 0.0


@pytest.mark.parametrize("curvature", [0.0, 1.5, -2.0])
@pytest.mark.parametrize("s", [0.6, -0.4])
def test_arc_pose_is_the_bicycle_path_at_fixed_steering(curvature, s):
    x, y, theta = 1.0, -0.5, 0.7
    n = 20000
    ds = s / n
    for _ in range(n):
        mid = theta + 0.5 * curvature * ds
        x, y, theta = x + ds * math.cos(mid), y + ds * math.sin(mid), theta + curvature * ds
    assert arc_pose(1.0, -0.5, 0.7, curvature, s) == pytest.approx((x, y, theta), abs=1e-6)


def test_the_same_wall_costs_only_when_the_car_cannot_stop_short_of_it():
    world = wall((0.45, -2.0, 0.45, 2.0))  # 0.25 m ahead of the front bumper
    assert shortfall(world, (0.0, 0.0, 0.0), 0.3) == 0.0
    assert shortfall(world, (0.0, 0.0, 0.0), 1.5) > 0.0
    assert shortfall(world, (0.0, 0.0, 0.0), 0.0) == 0.0  # parked at it is not closing on it


def test_a_wall_alongside_costs_nothing_until_the_car_turns_into_it():
    world = wall((-2.0, 0.25, 3.0, 0.25))  # 0.15 m off the left side, parallel to it
    assert shortfall(world, (0.0, 0.0, 0.0), 1.5, steer=0.0) == 0.0
    assert shortfall(world, (0.0, 0.0, 0.0), 1.5, steer=-P.max_steer_rad) == 0.0
    assert shortfall(world, (0.0, 0.0, 0.0), 1.5, steer=P.max_steer_rad) == pytest.approx(0.5)


def test_reversing_is_checked_behind_the_car():
    world = wall((-0.28, -2.0, -0.28, 2.0))  # 0.08 m behind the rear bumper
    assert shortfall(world, (0.0, 0.0, 0.0), -0.5) > 0.0
    assert shortfall(world, (0.0, 0.0, 0.0), 0.5) == 0.0


def test_the_clearance_shortcut_never_changes_the_answer():
    rng = np.random.default_rng(3)
    segments = [[*rng.uniform(-3, 3, 2), *rng.uniform(-3, 3, 2)] for _ in range(12)]
    circles = [[*rng.uniform(-3, 3, 2), rng.uniform(0.05, 0.3)] for _ in range(6)]
    world = World(segments=segments, circles=circles, bounds=(-3.0, -3.0, 3.0, 3.0))
    checked = 0
    for _ in range(600):
        x, y, theta = *rng.uniform(-2.5, 2.5, 2), rng.uniform(-math.pi, math.pi)
        if world.collides(x, y, theta, P.length, P.width):
            continue
        speed, steer = rng.uniform(-0.6, 1.5), rng.uniform(-P.max_steer_rad, P.max_steer_rad)
        clearance = world.clearance(x, y, theta, P.length, P.width)
        fast = brake_shortfall(world, x, y, theta, speed, steer, P, REACTION, MARGIN, 4, clearance)
        full = brake_shortfall(world, x, y, theta, speed, steer, P, REACTION, MARGIN, 4, 0.0)
        assert fast == full
        checked += full > 0.0
    assert checked > 20  # the sample has to contain blocked paths to prove anything


def brake_env(w_brake: float) -> CarEnv:
    cfg = EnvConfig(arena=ArenaParams(kind="outdoor"), domain_rand=DomainRandConfig(enabled=False),
                    reward=RewardConfig(w_brake=w_brake), max_steps=300)
    return CarEnv(cfg)


def drive_at_wall(env: CarEnv) -> list[float]:
    world = World(segments=[[3.0, -5.0, 3.0, 5.0], [-5.0, -5.0, -5.0, 5.0],
                            [-5.0, -5.0, 5.0, -5.0], [-5.0, 5.0, 5.0, 5.0]],
                  bounds=(-5.0, -5.0, 5.0, 5.0))
    env.reset(seed=0, options={"world": world, "pose": (0.0, 0.0, 0.0)})
    terms = []
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 1.0], np.float32))
        terms.append(info["reward_terms"]["brake"])
        if terminated or truncated:
            break
    return terms


def test_the_env_charges_driving_at_a_wall_faster_than_the_car_can_stop():
    terms = drive_at_wall(brake_env(1.0))
    assert min(terms[:-1]) < 0.0  # charged before the crash, not only by the collision
    assert terms[0] == 0.0


def test_the_term_is_off_by_default():
    assert RewardConfig().w_brake == 0.0
    assert set(drive_at_wall(brake_env(0.0))) == {0.0}
