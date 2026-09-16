"""The showcase arenas must look like the space the car will really be driven in.

`eval/arenas.py` exists because the training generators do not: they draw 0.30 m doorways and
cone fields thin enough to ignore. These tests pin what makes an arena a fair demo floor - no
opening under 0.60 m, cones of real cone sizes, and no spawn or goal walled off from the rest.
"""

import math

import numpy as np
import pytest

from selfdrive.config import load_env_config
from selfdrive.envs.car_env import CarEnv
from selfdrive.envs.goals import GoalTracker
from selfdrive.eval.arenas import ARENAS, MIN_GAP, arena_cycle
from selfdrive.eval.evaluate import RandomPolicy, run_episodes
from selfdrive.world.geometry import point_seg_distance

SAMPLE = 0.02  # m between the points a wall is measured at
TOUCHING = 0.02  # obstacles closer than this meet; a corner is not an opening
CONE_R = (0.03, 0.50)  # 0.06 m to 1.00 m across


def points_along(segment: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = segment
    n = max(int(math.hypot(x1 - x0, y1 - y0) / SAMPLE), 2)
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)])


def openings(world) -> list[float]:
    """Every gap between two obstacles that do not meet: what the car could drive at."""
    gaps: list[float] = []
    for i, segment in enumerate(world.segments):
        pts = points_along(segment)
        others = np.delete(world.segments, i, axis=0)
        if len(others):
            d = point_seg_distance(pts, others[:, :2], others[:, 2:] - others[:, :2])
            gaps += d.min(axis=0).tolist()
        if len(world.circles):
            d = np.linalg.norm(pts[:, None, :] - world.circ_c[None], axis=-1) - world.circ_r
            gaps += d.min(axis=0).tolist()
    for i, circle in enumerate(world.circles):
        rest = np.delete(world.circles, i, axis=0)
        if len(rest):
            d = np.linalg.norm(rest[:, :2] - circle[:2], axis=1) - rest[:, 2] - circle[2]
            gaps += d.tolist()
    return [g for g in gaps if g > TOUCHING]


@pytest.fixture(scope="module")
def cfg():
    return load_env_config("configs/env_waypoint_s2_patience_shield.yaml")


@pytest.mark.parametrize("name", list(ARENAS))
def test_no_opening_is_narrower_than_the_real_test_space(name):
    gaps = openings(ARENAS[name].world)
    assert min(gaps) >= MIN_GAP - SAMPLE, f"{name}: {min(gaps):.3f} m"


@pytest.mark.parametrize("name", list(ARENAS))
def test_cones_are_between_6_cm_and_1_m_across(name):
    radii = ARENAS[name].world.circles[:, 2]
    assert all(CONE_R[0] <= r <= CONE_R[1] for r in radii), radii


@pytest.mark.parametrize("name", list(ARENAS))
def test_the_car_starts_where_it_can_drive_out_of(name, cfg):
    arena = ARENAS[name]
    x, y, theta = arena.pose
    car = cfg.car
    assert not arena.world.collides(x, y, theta, car.length, car.width)
    clear = arena.world.clearance(x, y, theta, car.length, car.width)
    assert clear >= cfg.arena.spawn_clearance, f"{name}: {clear:.3f} m"


@pytest.mark.parametrize("name", list(ARENAS))
def test_no_goal_is_walled_off_from_the_spawn(name, cfg):
    arena = ARENAS[name]
    x, y, theta = arena.pose
    goals = GoalTracker(cfg.goal, cfg.car)
    goals.prepare(arena.world)
    # The spawn is a pose the lattice can drive between; goals are drawn from that same
    # floor, so every one of them is reachable from it.
    assert goals.connected(x, y, theta)
    rng = np.random.default_rng(4)
    for _ in range(8):
        goals.reset(arena.world, x, y, rng)
        assert goals.goal is not None


def test_the_arena_flag_drives_each_one_in_turn(cfg):
    cfg.max_steps = 20
    env = CarEnv(cfg)
    result = run_episodes(RandomPolicy(seed=0), env, n_episodes=len(ARENAS), seed=5,
                          options=arena_cycle("all"))
    assert result.episodes == len(ARENAS)
    assert env.world is ARENAS["campus_path"].world  # the last of the cycle
    with pytest.raises(SystemExit):
        arena_cycle("no_such_arena")
