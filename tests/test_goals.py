"""Waypoint driving: path distance, goals, the goal block and the reward for reaching them.

The goal reward pays for closing *path* distance, so the tests that matter most are the
ones a straight-line reward would fail: a wall between the car and its goal, a U facing
the goal, and a pocket of floor that random walls have sealed off.
"""

import heapq
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from selfdrive.config import load_env_config, load_yaml
from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.goals import GoalConfig, GoalTracker
from selfdrive.envs.obs import ObsConfig, ObservationBuilder
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.envs.rewards import RewardConfig
from selfdrive.eval.evaluate import ConstantPolicy, EvalResult, run_episodes
from selfdrive.sensors.odometry import OdometryParams
from selfdrive.world.generators import ArenaParams
from selfdrive.world.geometry import World, point_seg_distance
from selfdrive.world.navigation import AXIS, DIAG, NavGrid, PathField
from selfdrive.world.reachability import PoseReach


def box(size: float, *walls) -> World:
    """A walled square floor from (0, 0) to (size, size), plus interior walls."""
    s = size
    outer = [[0, 0, s, 0], [s, 0, s, s], [s, s, 0, s], [0, s, 0, 0]]
    return World(segments=outer + [list(w) for w in walls], bounds=(0.0, 0.0, s, s))


FIELD = box(30.0)
POCKET = [(1.0, 1.0, 5.0, 1.0), (5.0, 1.0, 5.0, 5.0), (5.0, 5.0, 1.0, 5.0), (1.0, 5.0, 1.0, 1.0)]


def goal_env(max_steps: int = 600, memory: int = 0, goal: GoalConfig | None = None) -> CarEnv:
    cfg = EnvConfig(
        obs=ObsConfig(goal_block=True, memory_sectors=memory),
        goal=goal or GoalConfig(),
        reward=RewardConfig(w_explore=0.0, w_progress=1.0, goal_bonus=5.0),
        arena=ArenaParams(kind="outdoor"),
        domain_rand=DomainRandConfig(enabled=False),
        max_steps=max_steps,
    )
    return CarEnv(cfg)


# --- path distance -----------------------------------------------------------

def dijkstra(blocked: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Textbook 8-connected Dijkstra with the same move rules, as a reference."""
    nx, ny = blocked.shape
    best = {source: 0}
    heap = [(0, source)]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if d > best[(i, j)]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di, dj) == (0, 0) or not (0 <= a < nx and 0 <= b < ny) or blocked[a, b]:
                    continue
                if di and dj and (blocked[a, j] or blocked[i, b]):
                    continue  # no cutting the corner of a blocked cell
                nd = d + (DIAG if di and dj else AXIS)
                if nd < best.get((a, b), math.inf):
                    best[(a, b)] = nd
                    heapq.heappush(heap, (nd, (a, b)))
    out = np.full((nx, ny), np.inf)
    for cell, d in best.items():
        out[cell] = d / AXIS
    return out


def test_swept_path_field_matches_dijkstra_on_random_grids():
    rng = np.random.default_rng(0)
    for _ in range(150):
        nx, ny = (int(v) for v in rng.integers(1, 20, 2))
        blocked = rng.random((nx, ny)) < rng.uniform(0.0, 0.5)
        free = np.argwhere(~blocked)
        if not len(free):
            continue
        source = tuple(int(v) for v in free[rng.integers(len(free))])
        np.testing.assert_allclose(PathField(blocked).distances(source), dijkstra(blocked, source))


def test_path_distance_goes_around_a_wall_not_through_it():
    grid = NavGrid(box(10.0, (5.0, 0.0, 5.0, 8.0)), res=0.2, inflate=0.2)
    d = grid.distance(grid.field_from(3.0, 1.0), 7.0, 1.0)
    over_the_end = 2 * math.hypot(2.0, 7.0)  # the shortest route, before inflation
    assert d > 3 * 4.0  # the straight line is 4 m
    # Octile reads up to 8 % long, and stepping round the inflated end adds under a metre.
    assert over_the_end <= d <= 1.09 * over_the_end + 1.0


@pytest.mark.parametrize("angle_deg", [3.0, 22.5, 30.0, 45.0, 60.0, 87.0])
def test_a_wall_at_any_angle_is_a_solid_barrier_at_the_minimum_inflation(angle_deg):
    a = math.radians(angle_deg)
    along, normal = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    wall = (5 - 20 * along[0], 5 - 20 * along[1], 5 + 20 * along[0], 5 + 20 * along[1])
    grid = NavGrid(box(10.0, wall), res=0.2, inflate=0.2 / math.sqrt(2.0))
    here = (5 + 2 * normal[0], 5 + 2 * normal[1])
    there = (5 - 2 * normal[0], 5 - 2 * normal[1])
    field = grid.field_from(*here)
    assert math.isfinite(grid.distance(field, *here))
    assert not math.isfinite(grid.distance(field, *there))


def test_inflation_too_small_to_seal_a_diagonal_wall_is_rejected():
    with pytest.raises(ValueError, match="diagonal wall could leak"):
        NavGrid(FIELD, res=0.2, inflate=0.1)


def test_a_sealed_pocket_is_unreachable_and_off_the_connected_floor():
    tracker = GoalTracker()
    tracker.prepare(box(10.0, *POCKET))
    field = tracker.grid.field_from(8.0, 8.0)
    assert not math.isfinite(tracker.grid.distance(field, 3.0, 3.0))
    assert tracker.connected(8.0, 8.0, 0.0)
    assert not tracker.connected(3.0, 3.0, 0.0)


# --- where the car fits ------------------------------------------------------

def walled(x1: float, y1: float, *walls) -> World:
    outer = [[0, 0, x1, 0], [x1, 0, x1, y1], [x1, y1, 0, y1], [0, y1, 0, 0]]
    return World(segments=outer + [list(w) for w in walls], bounds=(0.0, 0.0, x1, y1))


def drives_to(world: World, start, end, res: float = 0.075) -> bool:
    """Whether the car can drive from pose `start` to point `end`, at any heading."""
    reach = PoseReach(NavGrid(world, res, 0.1, clearance_cap=0.5), 0.40, 0.20, 0.70)
    _, i, j = reach.index(*end, 0.0)
    return bool(reach.flood(reach.index(*start))[:, i, j].any())


@pytest.mark.parametrize("centre", [3.0, 3.0375, 3.02])  # half a cell off the lattice, on it
def test_the_car_drives_through_a_doorway_30_cm_wide_wherever_it_sits(centre):
    world = box(6.0, (2.0, 0.0, 2.0, centre - 0.15), (2.0, centre + 0.15, 2.0, 6.0))
    assert drives_to(world, (4.0, 3.0, math.pi), (1.0, 3.0))
    tracker = GoalTracker()
    tracker.prepare(world)
    assert tracker.connected(1.0, 3.0, 0.0)
    assert math.isfinite(tracker.grid.distance(tracker.grid.field_from(4.0, 3.0), 1.0, 3.0))


def test_a_corridor_the_car_can_only_drive_straight_along_leaves_no_holes_in_the_floor():
    # 0.32 m wide and 4 m long between two rooms: no room to turn, so every pose along it is
    # reached by straight runs alone, and path distance must still cross it cell by cell.
    world = box(10.0, (3.0, 0.0, 3.0, 4.84), (3.0, 5.16, 3.0, 10.0), (7.0, 0.0, 7.0, 4.84),
                (7.0, 5.16, 7.0, 10.0), (3.0, 4.84, 7.0, 4.84), (3.0, 5.16, 7.0, 5.16))
    tracker = GoalTracker()
    tracker.prepare(world)
    g = tracker.grid
    row = round((5.0 - g.origin[1]) / g.res)
    along = ~g.blocked[round((3.2 - g.origin[0]) / g.res) : round((6.8 - g.origin[0]) / g.res), row]
    assert along.all()
    d = g.distance(g.field_from(1.5, 5.0), 8.5, 5.0)
    assert 7.0 - 0.1 <= d <= 7.0 + 0.3


def test_a_doorway_narrower_than_the_car_is_a_wall():
    world = box(6.0, (2.0, 0.0, 2.0, 2.9425), (2.0, 3.1325, 2.0, 6.0))  # 0.19 m, on the lattice
    assert not drives_to(world, (4.0, 3.0, math.pi), (1.0, 3.0))
    tracker = GoalTracker()
    tracker.prepare(world)
    assert not tracker.connected(1.0, 3.0, 0.0)
    assert not math.isfinite(tracker.grid.distance(tracker.grid.field_from(4.0, 3.0), 1.0, 3.0))


@pytest.mark.parametrize(("width", "turns"), [(0.27, False), (0.8, True)])
def test_a_corridor_the_car_fits_along_may_have_a_corner_it_cannot_fit_round(width, turns):
    # Room A (x < 4) opens into an L: right along y = 2, then up into room C (y > 6).
    w = width
    world = walled(10.0, 10.0,
                   (4.0, 0.0, 4.0, 2.0), (4.0, 2.0 + w, 4.0, 10.0),  # A's wall and door
                   (4.0, 2.0, 5.0 + w, 2.0), (4.0, 2.0 + w, 5.0, 2.0 + w),  # along
                   (5.0, 2.0 + w, 5.0, 6.0), (5.0 + w, 2.0, 5.0 + w, 6.0),  # up
                   (4.0, 6.0, 5.0, 6.0), (5.0 + w, 6.0, 10.0, 6.0))  # C's wall and door
    start = (2.0, 2.0 + w / 2, 0.0)
    assert drives_to(world, start, (4.6, 2.0 + w / 2), res=0.05)
    assert drives_to(world, start, (7.0, 8.0), res=0.05) == turns
    # A floor inflated by the car's half-width alone cannot tell the difference.
    flat = NavGrid(world, 0.05, 0.1)
    assert math.isfinite(flat.distance(flat.field_from(*start[:2]), 7.0, 8.0))


@pytest.mark.parametrize(("width", "turns"), [(0.40, False), (1.0, True)])
def test_turning_round_in_a_dead_end_takes_reversing_and_room_for_the_body(width, turns):
    # Too narrow for a U-turn on the turning radius either way, so it takes a 3-point turn,
    # and under 0.45 m the body cannot swing round at all.
    world = walled(3.0, width)
    assert drives_to(world, (1.5, width / 2, 0.0), (1.5, width / 2)) is True
    reach = PoseReach(NavGrid(world, 0.05, 0.1, clearance_cap=0.5), 0.40, 0.20, 0.70)
    poses = reach.flood(reach.index(1.5, width / 2, 0.0))
    assert bool(poses[reach.index(1.5, width / 2, math.pi)]) == turns


def test_goals_plan_for_the_least_agile_car_domain_randomization_draws():
    cfg = load_env_config("configs/env_waypoint.yaml")
    dr = cfg.domain_rand
    widest = cfg.car.wheelbase * dr.wheelbase_scale[1] / math.tan(math.radians(dr.max_steer_deg[0]))
    assert cfg.goal.turn_radius >= widest


def test_goals_are_drawn_in_path_range_clear_of_obstacles_and_reproducibly():
    world = box(12.0, (6.0, 0.0, 6.0, 9.0), (2.0, 6.0, 4.5, 6.0))
    grid = NavGrid(world, res=0.2, inflate=0.2, clearance_cap=0.5)
    field = grid.field_from(3.0, 1.0)
    same = [grid.sample_goal(field, np.random.default_rng(7), (2.0, 12.0), 0.5) for _ in range(2)]
    assert same[0] == same[1]

    rng = np.random.default_rng(1)
    for _ in range(100):
        gx, gy = grid.sample_goal(field, rng, (2.0, 12.0), 0.5)
        cell = (round((gx - grid.origin[0]) / grid.res), round((gy - grid.origin[1]) / grid.res))
        assert 2.0 <= field[cell] <= 12.0
        assert point_seg_distance(np.array([[gx, gy]]), world.seg_a, world.seg_e).min() >= 0.5


def test_with_no_goal_in_range_the_farthest_reachable_cell_is_used():
    grid = NavGrid(box(4.0), res=0.2, inflate=0.2, clearance_cap=0.5)
    field = grid.field_from(1.0, 1.0)
    gx, gy = grid.sample_goal(field, np.random.default_rng(0), (50.0, 60.0), 0.5)
    assert grid.distance(field, gx, gy) == pytest.approx(
        field[np.isfinite(field) & (grid.room >= 0.5)].max())


# --- the goal tracker --------------------------------------------------------

def test_progress_along_any_closed_route_sums_to_zero():
    tracker = GoalTracker()
    rng = np.random.default_rng(0)
    tracker.reset(box(10.0, (5.0, 0.0, 5.0, 8.0)), 3.0, 1.0, rng, pin=(9.0, 1.0))
    # Out over the wall's end, down its far side, in to scrape it, and back the other way.
    corners = [(3.0, 1.0), (3.0, 9.0), (7.0, 9.0), (7.0, 4.0), (5.12, 4.0),
               (7.0, 4.5), (7.5, 9.5), (2.5, 8.5), (3.0, 1.0)]
    total, largest = 0.0, 0.0
    for (xa, ya), (xb, yb) in zip(corners, corners[1:], strict=False):
        for t in np.linspace(0.0, 1.0, 80)[1:]:
            progress, arrived = tracker.update(xa + t * (xb - xa), ya + t * (yb - ya), rng)
            assert not arrived
            total += progress
            largest = max(largest, abs(total))
    assert largest > 5.0  # the route really did move toward and away from the goal
    assert total == pytest.approx(0.0, abs=1e-9)
    assert tracker.progress_m == pytest.approx(0.0, abs=1e-9)


def test_driving_deeper_into_a_u_that_faces_the_goal_loses_progress():
    # A U open to the left with the car inside, facing its closed end; the goal is beyond.
    world = box(12.0, (3.0, 4.0, 7.0, 4.0), (3.0, 8.0, 7.0, 8.0), (7.0, 4.0, 7.0, 8.0))
    tracker = GoalTracker()
    rng = np.random.default_rng(0)
    tracker.reset(world, 5.0, 6.0, rng, pin=(9.5, 6.0))
    progress, _ = tracker.update(6.0, 6.0, rng)
    assert progress < -0.4  # a metre closer in a straight line, and further by path


def test_goal_clearance_below_the_arrival_radius_is_rejected():
    with pytest.raises(ValueError, match="clearance"):
        GoalTracker(GoalConfig(radius=0.6, clearance=0.4))


# --- the environment ---------------------------------------------------------

def test_goal_block_reads_range_and_left_positive_bearing():
    e = goal_env()
    assert e.observation_space.shape == (68 + 3,)
    obs, _ = e.reset(seed=0, options={"world": FIELD, "pose": (15.0, 15.0, 0.0),
                                      "goal": (15.0, 21.0)})
    np.testing.assert_allclose(obs[-3:], [6.0 / 15.0 * 2 - 1, 1.0, 0.0], atol=1e-6)
    obs, _ = e.reset(seed=0, options={"world": FIELD, "pose": (15.0, 15.0, 0.0),
                                      "goal": (9.0, 15.0)})
    np.testing.assert_allclose(obs[-3:], [6.0 / 15.0 * 2 - 1, 0.0, -1.0], atol=1e-6)


def test_goal_block_length_must_match_the_config():
    builder = ObservationBuilder(ObsConfig(goal_block=True))
    frame = np.zeros(17, np.float32)
    with pytest.raises(ValueError, match="goal block"):
        builder.reset(frame)
    assert builder.reset(frame, goal=builder.goal(3.0, 0.5)).shape == (71,)


def test_goal_env_passes_the_gymnasium_checker():
    check_env(goal_env(), skip_render_check=True)


def drive(env: CarEnv, action, steps: int, pose, goal) -> float:
    env.reset(seed=0, options={"world": FIELD, "pose": pose, "goal": goal})
    total = 0.0
    for _ in range(steps):
        _, r, terminated, truncated, info = env.step(np.asarray(action, np.float32))
        assert not info.get("goal_reached")
        total += r
        if terminated or truncated:
            break
    return total


def test_driving_at_the_goal_beats_parking_circling_and_driving_away():
    e = goal_env()
    toward = drive(e, [0.0, 1.0], 240, (5.0, 15.0, 0.0), (28.0, 15.0))
    parked = drive(e, [0.0, 0.0], 240, (5.0, 15.0, 0.0), (28.0, 15.0))
    circling = drive(e, [1.0, 1.0], 240, (15.0, 15.0, 0.0), (28.0, 15.0))
    away = drive(e, [0.0, 1.0], 240, (26.0, 15.0, math.pi), (28.0, 15.0))
    assert toward > 8.0
    assert parked < 0.0
    assert circling < 0.2 * toward
    assert away < 0.0


def test_arriving_pays_the_bonus_once_and_draws_the_next_goal_further_on():
    e = goal_env()
    e.reset(seed=0, options={"world": FIELD, "pose": (10.0, 15.0, 0.0), "goal": (13.0, 15.0)})
    for _ in range(150):
        _, _, terminated, truncated, info = e.step(np.array([0.0, 0.6], np.float32))
        if info.get("goal_reached"):
            break
        assert not (terminated or truncated)
    else:
        pytest.fail("never reached a goal 3 m dead ahead")
    assert info["reward_terms"]["goal"] == 5.0
    assert e.goals.reached == 1
    assert math.dist(e.goals.goal, (13.0, 15.0)) >= 1.5  # at least 2 m of path away


def test_by_default_arrival_is_judged_from_the_true_pose_whatever_the_estimate():
    rng = np.random.default_rng(0)
    t = GoalTracker()
    t.reset(FIELD, 10.0, 15.0, rng, pin=(13.0, 15.0))
    assert not t.update(11.0, 15.0, rng, estimate=(13.0, 15.0))[1]
    assert t.update(12.8, 15.0, rng, estimate=(0.0, 0.0))[1]


def test_arrival_from_odometry_is_judged_where_the_estimate_puts_the_car():
    rng = np.random.default_rng(0)
    t = GoalTracker(GoalConfig(arrival_from_odometry=True))
    t.reset(FIELD, 10.0, 15.0, rng, pin=(13.0, 15.0))
    progress, arrived = t.update(12.8, 15.0, rng, estimate=(11.0, 15.0))
    assert progress > 2.0 and not arrived  # truly there, but 2 m short by the estimate
    _, arrived = t.update(11.2, 15.0, rng, estimate=(12.7, 15.0))
    assert arrived and t.reached == 1
    assert math.dist(t.goal, (11.2, 15.0)) >= 1.5  # drawn from the car, not the old goal
    with pytest.raises(ValueError, match="estimate"):
        t.update(11.2, 15.0, rng)


def test_the_env_judges_arrival_from_the_estimate_its_goal_block_reads():
    e = goal_env(goal=GoalConfig(arrival_from_odometry=True))
    e.reset(seed=0, options={"world": FIELD, "pose": (10.0, 15.0, 0.0), "goal": (13.0, 15.0)})
    e.odometry.p = OdometryParams(pos_noise=0.0, yaw_noise=0.0)
    e.odometry.x += 1.0  # the estimate has drifted a metre ahead of the car
    for _ in range(150):
        _, _, terminated, truncated, info = e.step(np.array([0.0, 0.6], np.float32))
        if info.get("goal_reached"):
            break
        assert not (terminated or truncated)
    else:
        pytest.fail("never reached a goal the estimate put 2 m ahead")
    assert 13.0 - e.car.state.x > 1.0  # the car itself is still over a metre short
    assert 13.0 - e.odometry.x <= 0.5


def test_goals_change_nothing_else_about_an_episode():
    def trace(goal_block: bool) -> np.ndarray:
        e = CarEnv(EnvConfig(obs=ObsConfig(memory_sectors=24, goal_block=goal_block),
                             max_steps=200))
        world = box(12.0, (6.0, 0.0, 6.0, 9.0))
        obs, _ = e.reset(seed=5, options={"world": world, "pose": (3.0, 3.0, 0.3)})
        rng = np.random.default_rng(0)
        rows = [obs[:116]]
        for _ in range(150):
            obs, _, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
            rows.append(obs[:116])
            if terminated or truncated:
                break
        return np.array(rows)

    np.testing.assert_array_equal(trace(False), trace(True))


def test_waypoint_episodes_reproduce_exactly():
    def run():
        e = CarEnv(load_env_config("configs/env_waypoint.yaml"))
        obs, _ = e.reset(seed=3)
        rng = np.random.default_rng(0)
        trace, rewards, goals = [obs], [], [e.goals.goal]
        for _ in range(200):
            obs, r, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
            trace.append(obs)
            rewards.append(r)
            goals.append(e.goals.goal)
            if terminated or truncated:
                break
        return np.array(trace), np.array(rewards), goals

    a, b = run(), run()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_waypoint_arenas_never_spawn_the_car_where_its_goal_is_unreachable():
    e = CarEnv(load_env_config("configs/env_waypoint.yaml"))
    for seed in range(10):
        e.reset(seed=seed)
        s = e.car.state
        assert e.goals.connected(s.x, s.y, s.theta)
        assert 1.4 <= e.goals.remaining <= 12.6


def test_a_car_is_never_spawned_inside_a_sealed_pocket():
    # A quarter of this floor is a sealed box; unchecked, spawns would land in it often.
    world = box(10.0, *POCKET)
    e = goal_env()
    for seed in range(30):
        e.reset(seed=seed, options={"world": world})
        s = e.car.state
        assert not (1.0 < s.x < 5.0 and 1.0 < s.y < 5.0)
        assert math.isfinite(e.goals.remaining)


# --- evaluation and config ---------------------------------------------------

class ScriptedEnv:
    """Ends every episode on its first step, with the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def reset(self, seed=None):
        return np.zeros(2, np.float32), {}

    def step(self, action):
        m = self.outcomes.pop(0)
        return np.zeros(2, np.float32), 0.0, bool(m["collided"]), not m["collided"], {
            "episode_metrics": m}


def test_best_model_is_kept_by_clean_goals_when_there_are_goals():
    outcomes = [
        {"coverage_m2": 20.0, "goals_reached": 6.0, "collided": 1.0, "stuck": 0.0},
        {"coverage_m2": 10.0, "goals_reached": 3.0, "collided": 0.0, "stuck": 0.0},
    ]
    result = run_episodes(ConstantPolicy(0.0, 1.0), ScriptedEnv(outcomes), n_episodes=2)
    assert result.has_goals
    assert result.mean_goals == 4.5
    assert result.clean_goals == result.score == 1.5
    assert "clean goals" in str(result)
    assert EvalResult(clean_coverage_m2=9.0).score == 9.0


def test_waypoint_config_is_memory_light_big_plus_a_goal_and_nothing_else():
    base = load_yaml("configs/env_phase1_memory_light_big.yaml")
    ours = load_yaml("configs/env_waypoint.yaml")
    assert set(ours.pop("goal")) == {"radius", "distance_min", "distance_max", "clearance",
                                     "grid_res", "turn_radius", "headings"}
    assert {k: ours["obs"].pop(k) for k in ("goal_block", "norm_goal_max")} == {
        "goal_block": True, "norm_goal_max": 15.0}
    assert {k: ours["reward"].pop(k) for k in ("w_explore", "w_progress", "goal_bonus")} == {
        "w_explore": 0.0, "w_progress": 1.0, "goal_bonus": 5.0}
    base["reward"].pop("w_explore")
    assert ours == base
    assert load_env_config("configs/env_waypoint.yaml").obs.size == 116 + 3


def test_waypoint_pay5_config_is_the_waypoint_config_with_five_times_the_pay():
    base = load_yaml("configs/env_waypoint.yaml")
    ours = load_yaml("configs/env_waypoint_pay5.yaml")
    for key in ("w_progress", "goal_bonus"):
        assert ours["reward"].pop(key) == 5.0 * base["reward"].pop(key)
    assert ours == base


def test_waypoint_yaw25_config_is_pay5_with_the_heading_drift_range_halved():
    base = load_yaml("configs/env_waypoint_pay5.yaml")
    ours = load_yaml("configs/env_waypoint_pay5_yaw25.yaml")
    assert ours["domain_rand"].pop("odom_yaw_noise") == [0.005, 0.025]
    assert ours == base
    assert DomainRandConfig().odom_yaw_noise == (0.005, 0.05)  # what pay5 draws
    cfg = load_env_config("configs/env_waypoint_pay5_yaw25.yaml")
    assert cfg.domain_rand.odom_yaw_noise == (0.005, 0.025)


def test_waypoint_pin_config_is_pay5_with_arrival_judged_from_odometry():
    base = load_yaml("configs/env_waypoint_pay5.yaml")
    ours = load_yaml("configs/env_waypoint_pay5_pin.yaml")
    assert ours["goal"].pop("arrival_from_odometry") is True
    assert ours == base


def test_waypoint_s2_config_is_pay5_with_both_stage1_winners():
    base = load_yaml("configs/env_waypoint_pay5.yaml")
    ours = load_yaml("configs/env_waypoint_s2.yaml")
    assert ours["goal"].pop("arrival_from_odometry") is True
    assert ours["domain_rand"].pop("odom_yaw_noise") == [0.005, 0.025]
    assert ours == base
