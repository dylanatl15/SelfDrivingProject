"""Sensor model behaviour.

The corruption model is the part of this simulator most likely to be silently wrong, and
a silent error here is invisible during training - the policy simply learns a world that
does not exist. Each test pins one hardware fact.
"""

import math

import numpy as np
import pytest

from selfdrive.dynamics.base import VehicleState
from selfdrive.sensors.depth_arc import DepthArc, DepthArcParams
from selfdrive.sensors.noise import LatencyBuffer, apply_dropout
from selfdrive.sensors.ultrasonic import UltrasonicArray, UltrasonicParams
from selfdrive.world.geometry import World

DT = 1.0 / 30.0
ORIGIN = VehicleState(x=0.0, y=0.0, theta=0.0, speed=1.0)
# A box big enough that only the deliberately placed obstacles are ever hit.
ROOM = World(segments=[[-50, -50, 50, -50], [50, -50, 50, 50],
                       [50, 50, -50, 50], [-50, 50, -50, -50]],
             bounds=(-50.0, -50.0, 50.0, 50.0))


def rng():
    return np.random.default_rng(0)


# --- depth arc ---------------------------------------------------------------

def test_bucket_count_matches_the_configuration():
    d = DepthArc(DepthArcParams(n_buckets=8))
    assert d.true_ranges(ROOM, ORIGIN).shape == (8,)


def test_a_wall_dead_ahead_is_seen_at_the_right_distance():
    p = DepthArcParams(max_range=10.0, mount_forward=0.0)
    world = World(segments=[[4.0, -5.0, 4.0, 5.0]], bounds=(-10.0, -10.0, 10.0, 10.0))
    ranges = DepthArc(p).true_ranges(world, ORIGIN)
    assert ranges.min() == pytest.approx(4.0, abs=0.01)


def test_the_camera_mount_offset_is_applied():
    world = World(segments=[[4.0, -5.0, 4.0, 5.0]], bounds=(-10.0, -10.0, 10.0, 10.0))
    flush = DepthArc(DepthArcParams(max_range=10.0, mount_forward=0.0))
    ahead = DepthArc(DepthArcParams(max_range=10.0, mount_forward=0.5))
    assert flush.true_ranges(world, ORIGIN).min() - ahead.true_ranges(
        world, ORIGIN).min() == pytest.approx(0.5, abs=0.01)


def test_field_of_view_is_actually_bounded():
    """The whole point of the 69 deg default: an obstacle outside the arc is invisible.

    Simulating a wider arc than ARCore can deliver manufactures a blind spot that exists
    only on the real car.
    """
    narrow = DepthArc(DepthArcParams(fov_deg=60.0, max_range=10.0, mount_forward=0.0))
    wide = DepthArc(DepthArcParams(fov_deg=120.0, max_range=10.0, mount_forward=0.0))
    # A post out at ~50 degrees to the left: inside 120 deg, outside 60 deg.
    angle = math.radians(50.0)
    post = World(circles=[[3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.3]],
                 bounds=(-10.0, -10.0, 10.0, 10.0))
    assert wide.true_ranges(post, ORIGIN).min() < 9.0
    assert narrow.true_ranges(post, ORIGIN).min() == pytest.approx(10.0)


def test_each_bucket_reports_its_closest_hit():
    # A near post on the left and a far wall: the left buckets must report the post.
    world = World(segments=[[8.0, -9.0, 8.0, 9.0]], circles=[[1.5, 1.0, 0.2]],
                  bounds=(-10.0, -10.0, 10.0, 10.0))
    ranges = DepthArc(DepthArcParams(max_range=10.0, mount_forward=0.0)).true_ranges(
        world, ORIGIN)
    assert ranges.min() < 2.5  # the post, not the wall
    assert ranges.max() > 7.0  # buckets that miss the post still see the wall


def test_readings_are_clipped_to_the_configured_range():
    p = DepthArcParams(max_range=3.0, min_range=0.15)
    r = DepthArc(p).sample(ROOM, ORIGIN, DT, rng())
    assert r.max() <= p.max_range + 1e-9
    assert r.min() >= p.min_range - 1e-9


def test_depth_confidence_collapses_when_the_car_stops():
    """ARCore on a phone with no ToF computes depth from motion. Stopped means blind,
    which is exactly when the car is deciding whether to reverse."""
    d = DepthArc(DepthArcParams(speed_ref=0.3))
    d.sample(ROOM, VehicleState(speed=1.0), DT, rng())
    assert d.confidence == pytest.approx(1.0)
    d.sample(ROOM, VehicleState(speed=0.0), DT, rng())
    assert d.confidence == pytest.approx(0.0)
    d.sample(ROOM, VehicleState(speed=0.15), DT, rng())
    assert d.confidence == pytest.approx(0.5)


def test_stationary_dropout_actually_corrupts_more_readings():
    world = World(segments=[[2.0, -5.0, 2.0, 5.0]], bounds=(-9.0, -9.0, 9.0, 9.0))
    p = DepthArcParams(dropout_prob=0.0, stationary_dropout=1.0, latency_steps=0,
                       noise_frac=0.0, max_range=6.0, mount_forward=0.0)

    moving = DepthArc(p)
    moving.sample(world, VehicleState(speed=1.0), DT, rng())
    truth = moving.true_ranges(world, VehicleState(speed=1.0))

    stopped = DepthArc(p)
    # First sample while stopped: every bucket drops and holds the initial max-range fill.
    got = stopped.sample(world, VehicleState(speed=0.0), DT, rng())
    assert np.allclose(got, p.max_range)
    assert not np.allclose(got, truth)


def test_latency_delays_readings_by_whole_steps():
    p = DepthArcParams(latency_steps=2, dropout_prob=0.0, stationary_dropout=0.0,
                       noise_frac=0.0, max_range=6.0, mount_forward=0.0)
    d = DepthArc(p)
    world = World(segments=[[2.0, -5.0, 2.0, 5.0]], bounds=(-9.0, -9.0, 9.0, 9.0))
    state = VehicleState(speed=1.0)

    first = d.sample(world, state, DT, rng())
    assert np.allclose(first, p.max_range)  # still flushing the pre-filled buffer
    d.sample(world, state, DT, rng())
    third = d.sample(world, state, DT, rng())
    assert third.min() < p.max_range  # the real reading has now arrived


def test_latency_buffer_of_zero_is_a_passthrough():
    b = LatencyBuffer(0, (3,), 0.0)
    np.testing.assert_array_equal(b.push_pop(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])


def test_dropout_holds_the_previous_value_rather_than_faking_max_range():
    values = np.array([1.0, 1.0, 1.0])
    held = np.array([9.0, 9.0, 9.0])
    out, mask = apply_dropout(values, held, prob=1.0, rng=rng())
    np.testing.assert_array_equal(out, held)
    assert mask.all()


# --- ultrasonics -------------------------------------------------------------

def test_default_build_is_four_sensors_named_front_left_right_back():
    u = UltrasonicArray(UltrasonicParams(n_sensors=4))
    assert u.n == 4
    assert u.names == ("front", "left", "right", "back")


def test_three_sensor_build_drops_the_front():
    # The camera already covers forward, so left/right/back buy the most information.
    u = UltrasonicArray(UltrasonicParams(n_sensors=3))
    assert u.n == 3
    assert u.names == ("left", "right", "back")


def test_sensors_point_where_their_names_say():
    world = World(circles=[[0.0, 2.0, 0.3]], bounds=(-9.0, -9.0, 9.0, 9.0))  # to the LEFT
    u = UltrasonicArray(UltrasonicParams(max_range=4.0))
    r = dict(zip(u.names, u.true_ranges(world, ORIGIN), strict=True))
    assert r["left"] < 3.0
    assert r["right"] == pytest.approx(4.0)
    assert r["front"] == pytest.approx(4.0)


def test_round_robin_refreshes_one_sensor_per_slot():
    """HC-SR04s cross-talk, so the ESP32 fires them one at a time. Four sensors at
    ~20 Hz total means most policy steps see at least one stale reading."""
    p = UltrasonicParams(update_hz=20.0, noise_m=0.0, dropout_prob=0.0, max_range=4.0)
    u = UltrasonicArray(p)
    world = World(circles=[[1.0, 0.0, 0.2]], bounds=(-9.0, -9.0, 9.0, 9.0))

    # One slot is 50 ms; a single 33 ms step cannot refresh everything.
    first = u.sample(world, ORIGIN, DT, rng())
    changed = np.count_nonzero(first != p.max_range)
    assert changed <= 1, "more than one sensor refreshed in a single ping slot"


def test_every_sensor_eventually_refreshes():
    p = UltrasonicParams(update_hz=20.0, noise_m=0.0, dropout_prob=0.0, max_range=4.0)
    u = UltrasonicArray(p)
    world = World(segments=[[-1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0],
                            [1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
                  bounds=(-9.0, -9.0, 9.0, 9.0))
    for _ in range(40):
        r = u.sample(world, ORIGIN, DT, rng())
    assert np.all(r < p.max_range)


def test_a_slow_control_step_can_fit_several_ping_slots():
    p = UltrasonicParams(update_hz=100.0, noise_m=0.0, dropout_prob=0.0, max_range=4.0)
    u = UltrasonicArray(p)
    world = World(segments=[[-1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0],
                            [1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
                  bounds=(-9.0, -9.0, 9.0, 9.0))
    r = u.sample(world, ORIGIN, 0.1, rng())  # 100 ms at 100 Hz = 10 slots
    assert np.all(r < p.max_range)


def test_cone_width_lets_a_sensor_see_slightly_off_axis():
    # A post just off the front axis is inside a 15 degree cone but misses a single ray.
    angle = math.radians(5.0)
    world = World(circles=[[2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.05]],
                  bounds=(-9.0, -9.0, 9.0, 9.0))
    wide = UltrasonicArray(UltrasonicParams(cone_deg=20.0, rays_per_sensor=5, max_range=4.0))
    narrow = UltrasonicArray(UltrasonicParams(cone_deg=0.5, rays_per_sensor=1, max_range=4.0))
    assert wide.true_ranges(world, ORIGIN)[0] < 3.0
    assert narrow.true_ranges(world, ORIGIN)[0] == pytest.approx(4.0)
