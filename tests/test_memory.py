"""Obstacle memory ring, and the odometry it depends on.

The ring is geometry the phone has to reproduce from ARCore's pose, so the geometry is
pinned directly, down to a worked example copied from `docs/memory-ring.md`. The env tests
pin the insertion rules: only new measurements are stored, depth is placed from the pose
it was captured at, and switching the memory on changes nothing else about an episode.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from selfdrive.config import load_env_config, load_yaml
from selfdrive.dynamics.base import VehicleState
from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.memory import EgoMemory
from selfdrive.envs.obs import ObsConfig
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.sensors.depth_arc import DepthArc, DepthArcParams
from selfdrive.sensors.odometry import Odometry, OdometryParams
from selfdrive.sensors.ultrasonic import UltrasonicParams
from selfdrive.world.geometry import World

S = 24
DT = 1.0 / 30.0


def memory() -> EgoMemory:
    m = EgoMemory(sectors=S, seconds=3.0, norm_max=5.0, points_per_step=12)
    m.reset(dt=DT)
    return m


def remember(m: EgoMemory, *points, stamp: float = 0.0) -> None:
    m.insert(np.array(points, dtype=float).reshape(-1, 2), np.full(len(points), stamp))


def occupied(ring: np.ndarray) -> dict[int, tuple[float, float]]:
    """{sector: (metres, seconds)} for every sector that holds a point."""
    return {int(i): ((ring[i] + 1) / 2 * 5.0, (ring[S + i] + 1) / 2 * 3.0)
            for i in np.flatnonzero(ring[:S] < 1.0)}


# --- ring geometry -----------------------------------------------------------

def test_empty_memory_reads_all_ones():
    ring = memory().ring(0.0, 0.0, 0.0, now=0.0)
    assert ring.shape == (2 * S,) and ring.dtype == np.float32
    assert np.all(ring == 1.0)


def test_a_point_straight_ahead_is_sector_zero():
    m = memory()
    remember(m, (2.0, 0.0))
    ring = m.ring(0.0, 0.0, 0.0, now=0.0)
    assert ring[0] == pytest.approx(-0.2)  # 2 m of 5
    assert ring[S] == -1.0  # just measured
    assert occupied(ring).keys() == {0}


@pytest.mark.parametrize("point, sector", [
    ((0.0, 2.0), 6),    # left
    ((-2.0, 0.0), 12),  # behind
    ((0.0, -2.0), 18),  # right
])
def test_sectors_increase_counter_clockwise_like_steering(point, sector):
    m = memory()
    remember(m, point)
    assert occupied(m.ring(0.0, 0.0, 0.0, now=0.0)).keys() == {sector}


@pytest.mark.parametrize("bearing_deg, sector", [(7.4, 0), (7.6, 1), (-7.4, 0), (-7.6, 23)])
def test_sector_zero_is_centred_on_the_heading(bearing_deg, sector):
    m = memory()
    b = math.radians(bearing_deg)
    remember(m, (2.0 * math.cos(b), 2.0 * math.sin(b)))
    assert occupied(m.ring(0.0, 0.0, 0.0, now=0.0)).keys() == {sector}


def test_a_point_seen_ahead_stays_put_in_the_world_while_the_car_turns_and_moves():
    m = memory()
    remember(m, (2.0, 0.0), stamp=0.0)  # seen dead ahead from the origin
    # A left turn later the car is at (0.5, 0.5) facing +y, so the point is behind-right.
    ring = m.ring(0.5, 0.5, math.pi / 2, now=1.0)
    (sector, (dist, age)), = occupied(ring).items()
    assert sector == 17  # atan2(-1.5, -0.5) = -108.4 deg
    assert dist == pytest.approx(math.hypot(1.5, 0.5), abs=1e-5)
    assert age == pytest.approx(1.0, abs=1e-5)


def test_the_nearest_point_wins_and_reports_its_own_age():
    m = memory()
    remember(m, (1.0, 0.0), stamp=0.0)  # near and old
    remember(m, (3.0, 0.1), stamp=2.0)  # far and new, same sector
    assert occupied(m.ring(0.0, 0.0, 0.0, now=2.0))[0] == pytest.approx((1.0, 2.0), abs=1e-5)


def test_points_expire_after_memory_seconds():
    m = memory()
    remember(m, (2.0, 0.0), stamp=0.0)
    assert occupied(m.ring(0.0, 0.0, 0.0, now=2.9))
    assert not occupied(m.ring(0.0, 0.0, 0.0, now=3.1))


def test_the_ring_buffer_never_overwrites_a_live_point():
    m = memory()
    remember(m, (2.0, 0.0), stamp=0.0)
    for k in range(1, 90):  # 89 steps: still under 3 s
        m.insert(np.empty((0, 2)), np.empty(0))
        assert len(m.live(k * DT)[1]) == 1, f"lost at step {k}"


def test_points_beyond_norm_max_leave_the_sector_empty():
    # Otherwise a far point would read distance +1 with a real age, and "empty" would
    # stop meaning one thing.
    m = memory()
    remember(m, (6.0, 0.0))
    assert np.all(m.ring(0.0, 0.0, 0.0, now=0.0) == 1.0)


def test_clear_forgets_everything():
    m = memory()
    remember(m, (2.0, 0.0), (0.0, 2.0))
    m.clear()
    assert np.all(m.ring(0.0, 0.0, 0.0, now=0.0) == 1.0)


def test_worked_example_from_the_android_spec():
    """Copied from docs/memory-ring.md. If this changes, the phone is wrong."""
    m = memory()
    points = {  # name: (x, y, stamp)
        "A": (1.0, 3.5, 9.5),  # ahead
        "B": (0.0, 2.0, 8.0),  # left
        "C": (1.6, 2.0, 7.5),  # right
        "D": (2.5, 2.0, 9.0),  # right, farther than C
        "E": (1.0, 1.0, 6.9),  # expired
        "F": (1.0, -4.0, 9.0),  # beyond 5 m
    }
    for x, y, stamp in points.values():
        remember(m, (x, y), stamp=stamp)
    ring = m.ring(1.0, 2.0, math.radians(90.0), now=10.0)

    expected = np.ones(2 * S, dtype=np.float32)
    expected[[0, 6, 18]] = [-0.4, -0.6, -0.76]
    expected[[S + 0, S + 6, S + 18]] = [-2 / 3, 1 / 3, 2 / 3]
    np.testing.assert_allclose(ring, expected, atol=1e-6)


# --- odometry ----------------------------------------------------------------

def curvy_path(n: int = 300):
    t = np.arange(n) * DT
    return np.column_stack([np.sin(t) * 3.0, np.cos(t * 0.7) * 2.0, t * 0.9])


def test_noiseless_odometry_tracks_the_true_pose_exactly():
    path = curvy_path()
    o = Odometry(OdometryParams(pos_noise=0.0, yaw_noise=0.0))
    o.reset(*path[0])
    for x, y, th in path[1:]:
        o.update(x, y, th, DT, np.random.default_rng(0))
    x, y, th = path[-1]
    assert (o.x, o.y) == pytest.approx((x, y), abs=1e-9)
    assert math.cos(o.theta - th) == pytest.approx(1.0, abs=1e-12)


def test_scale_error_stretches_distance():
    o = Odometry(OdometryParams(scale_error=0.05, pos_noise=0.0, yaw_noise=0.0))
    o.reset(0.0, 0.0, 0.0)
    for x in np.linspace(0.0, 10.0, 101)[1:]:
        o.update(x, 0.0, 0.0, DT, np.random.default_rng(0))
    assert o.x == pytest.approx(10.5)


def lateral_drift_std(metres: float, steps: int, k: float = 0.05, trials: int = 300) -> float:
    errors = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        o = Odometry(OdometryParams(pos_noise=k, yaw_noise=0.0))
        o.reset(0.0, 0.0, 0.0)
        for x in np.linspace(0.0, metres, steps + 1)[1:]:
            o.update(x, 0.0, 0.0, DT, rng)
        errors.append(o.y)
    return float(np.std(errors))


def test_drift_grows_with_the_square_root_of_distance_whatever_the_step_size():
    one_coarse = lateral_drift_std(1.0, 20)
    one_fine = lateral_drift_std(1.0, 200)
    four = lateral_drift_std(4.0, 80)
    assert one_fine == pytest.approx(0.05, rel=0.15)  # pos_noise is per sqrt(metre)
    assert one_coarse == pytest.approx(one_fine, rel=0.15)  # independent of dt
    assert four / one_coarse == pytest.approx(2.0, rel=0.15)


def test_tracking_loss_is_reported_and_recovers():
    o = Odometry(OdometryParams(tracking_loss_per_s=1e9, tracking_recover_s=0.1))
    o.reset(0.0, 0.0, 0.0)
    assert o.update(0.01, 0.0, 0.0, DT, np.random.default_rng(0))
    assert not o.tracking
    o.p = replace(o.p, tracking_loss_per_s=0.0)
    for _ in range(4):
        o.update(0.01, 0.0, 0.0, DT, np.random.default_rng(0))
    assert o.tracking


# --- sensors feeding the memory ----------------------------------------------

WALL_AT_3 = World(
    segments=[[3.0, -12.0, 3.0, 12.0],
              [-12.0, -12.0, 12.0, -12.0], [12.0, -12.0, 12.0, 12.0],
              [12.0, 12.0, -12.0, 12.0], [-12.0, 12.0, -12.0, -12.0]],
    bounds=(-12.0, -12.0, 12.0, 12.0),
)
OPEN_ROOM = World(segments=WALL_AT_3.segments[1:], bounds=WALL_AT_3.bounds)


def test_depth_freshness_is_delayed_with_the_ranges_and_excludes_misses():
    d = DepthArc(DepthArcParams(noise_frac=0.0, dropout_prob=0.0, stationary_dropout=0.0,
                                latency_steps=2))
    state = VehicleState(speed=1.0)
    rng = np.random.default_rng(0)
    seen = [d.sample(WALL_AT_3, state, DT, rng) for _ in range(3)]
    assert np.all(seen[1] == d.p.max_range)  # still the latency prefill
    assert np.all(seen[2] < d.p.max_range) and d.fresh.all()  # the wall, arriving together

    d.sample(OPEN_ROOM, state, DT, rng)
    d.sample(OPEN_ROOM, state, DT, rng)
    d.sample(OPEN_ROOM, state, DT, rng)
    assert not d.fresh.any()  # nothing within range is not an obstacle


def test_dropped_depth_is_never_fresh():
    d = DepthArc(DepthArcParams(dropout_prob=1.0, latency_steps=0))
    rng = np.random.default_rng(0)
    for _ in range(5):
        d.sample(WALL_AT_3, VehicleState(speed=1.0), DT, rng)
        assert not d.fresh.any()


# --- environment -------------------------------------------------------------

CLEAN_DEPTH = DepthArcParams(rays_per_bucket=1, noise_frac=0.0, dropout_prob=0.0,
                             stationary_dropout=0.0, latency_steps=3)
CLEAN_ULTRA = UltrasonicParams(noise_m=0.0, dropout_prob=0.0)
PERFECT_ODOMETRY = OdometryParams(pos_noise=0.0, yaw_noise=0.0)


def clean_env(**sections) -> CarEnv:
    cfg = EnvConfig(depth=CLEAN_DEPTH, ultrasonic=CLEAN_ULTRA, odometry=PERFECT_ODOMETRY,
                    obs=ObsConfig(memory_sectors=S),
                    domain_rand=DomainRandConfig(enabled=False), max_steps=500)
    return CarEnv(replace(cfg, **sections))


def drive(e: CarEnv, world: World, steps: int, throttle: float = 1.0) -> np.ndarray:
    obs, _ = e.reset(seed=0, options={"world": world, "pose": (0.0, 0.0, 0.0)})
    for _ in range(steps):
        obs, _, terminated, truncated, _ = e.step(np.array([0.0, throttle], np.float32))
        assert not (terminated or truncated)
    return obs


def live_points(e: CarEnv) -> np.ndarray:
    return e.memory.live(e.steps * e.dt)[0]


def test_disabled_memory_keeps_the_68_float_layout():
    e = CarEnv(EnvConfig(domain_rand=DomainRandConfig(enabled=False)))
    obs, _ = e.reset(seed=0)
    assert e.memory is None and obs.shape == (68,)


def test_memory_env_passes_the_gymnasium_checker():
    check_env(CarEnv(EnvConfig(obs=ObsConfig(memory_sectors=S))), skip_render_check=True)


def test_enabled_memory_appends_two_floats_per_sector_inside_the_box():
    e = CarEnv(EnvConfig(obs=ObsConfig(memory_sectors=S), max_steps=300))
    obs, _ = e.reset(seed=4)
    assert obs.shape == (68 + 2 * S,) == e.observation_space.shape
    rng = np.random.default_rng(0)
    filled = 0
    for _ in range(600):
        obs, _, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
        assert e.observation_space.contains(obs)
        filled += int(np.any(obs[68:68 + S] < 1.0))
        if terminated or truncated:
            obs, _ = e.reset()
    assert filled > 0, "random arenas never put anything in memory"


def rollout(cfg: EnvConfig, seed: int, steps: int = 250):
    e = CarEnv(cfg)
    obs, _ = e.reset(seed=seed)
    rng = np.random.default_rng(0)
    trace, rewards = [obs], []
    for _ in range(steps):
        obs, r, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
        trace.append(obs)
        rewards.append(r)
        if terminated or truncated:
            obs, _ = e.reset()  # unseeded, so later episodes are covered too
            trace.append(obs)
    return np.array(trace), np.array(rewards)


def test_memory_changes_nothing_else_about_an_episode():
    """Same seed, same actions: every non-memory input and every reward is identical.

    This is what makes a v2-versus-memory comparison on the same eval seeds fair."""
    plain = EnvConfig(max_steps=120)
    with_memory = replace(plain, obs=ObsConfig(memory_sectors=S),
                          odometry=OdometryParams(tracking_loss_per_s=0.5))
    a_obs, a_rew = rollout(plain, seed=12)
    b_obs, b_rew = rollout(with_memory, seed=12)
    np.testing.assert_array_equal(a_obs, b_obs[:, :68])
    np.testing.assert_array_equal(a_rew, b_rew)


def test_memory_episodes_reproduce_exactly():
    cfg = EnvConfig(obs=ObsConfig(memory_sectors=S), max_steps=120,
                    odometry=OdometryParams(tracking_loss_per_s=0.5))
    a_obs, _ = rollout(cfg, seed=3)
    b_obs, _ = rollout(cfg, seed=3)
    np.testing.assert_array_equal(a_obs, b_obs)


def test_depth_is_placed_from_the_pose_it_was_captured_at():
    e = clean_env()
    drive(e, WALL_AT_3, 45)
    pts = live_points(e)
    assert len(pts) > 50
    # Three steps of latency at this speed would misplace the wall by centimetres.
    assert e.car.state.speed * 3 * e.dt > 0.05
    np.testing.assert_allclose(pts[:, 0], 3.0, atol=1e-6)


def test_perfect_odometry_in_the_env_is_the_true_pose():
    e = clean_env()
    drive(e, WALL_AT_3, 30)
    s = e.car.state
    assert (e.odometry.x, e.odometry.y, e.odometry.theta) == pytest.approx((s.x, s.y, s.theta))


def test_held_readings_are_not_stored_again():
    # Depth fully invalid leaves the front ultrasonic, which pings once per round-robin
    # cycle and repeats itself in between.
    e = clean_env(depth=replace(CLEAN_DEPTH, dropout_prob=1.0))
    e.reset(seed=0, options={"world": WALL_AT_3, "pose": (0.0, 0.0, 0.0)})
    changes, last = 0, None
    for _ in range(45):
        e.step(np.array([0.0, 1.0], np.float32))
        front = float(e.ultra._values[0])
        changes += int(front != last and front < e.ultra.p.max_range)
        last = front
    pts = live_points(e)
    assert len(pts) == changes < 45 / 3
    np.testing.assert_allclose(pts[:, 0], 3.0, atol=1e-6)


def test_nothing_in_range_stores_nothing_even_with_noise():
    # A noisy max-range reading must not become a phantom wall just inside max range.
    cfg = EnvConfig(obs=ObsConfig(memory_sectors=S),
                    domain_rand=DomainRandConfig(enabled=False), max_steps=500)
    e = CarEnv(cfg)
    obs = drive(e, OPEN_ROOM, 45)
    assert len(live_points(e)) == 0
    assert np.all(obs[68:] == 1.0)


def test_losing_tracking_clears_the_memory():
    e = clean_env(odometry=replace(PERFECT_ODOMETRY, tracking_loss_per_s=1e9))
    obs = drive(e, WALL_AT_3, 20)
    assert len(live_points(e)) == 0
    assert np.all(obs[68:] == 1.0)


# --- shipped config ----------------------------------------------------------

def test_memory_config_is_phase1_plus_memory_and_nothing_else():
    base = load_yaml("configs/env_phase1.yaml")
    mem = load_yaml("configs/env_phase1_memory.yaml")
    assert mem.pop("odometry")
    for key in ("memory_sectors", "memory_seconds", "norm_memory_max"):
        mem["obs"].pop(key)
    assert mem == base


def test_memory_config_gives_116_inputs():
    cfg = load_env_config("configs/env_phase1_memory.yaml")
    assert cfg.obs.size == 116
    obs, _ = CarEnv(cfg).reset(seed=0)
    assert obs.shape == (116,)


def test_memory_smooth_config_is_the_memory_config_with_only_oscillation_weights_changed():
    mem = load_yaml("configs/env_phase1_memory.yaml")
    smooth = load_yaml("configs/env_phase1_memory_smooth.yaml")
    for key, before, after in (("w_oscillation", 0.05, 0.15),
                               ("w_throttle_oscillation", 0.0, 0.15)):
        assert mem["reward"].pop(key) == before
        assert smooth["reward"].pop(key) == after
    assert smooth == mem
