"""Geometry ground truth.

Every case here is a hand-computed answer, not a regression snapshot. If the raycaster
is subtly wrong, every sensor reading and therefore every trained policy is wrong, and
the failure is invisible in a training curve - the agent just learns the wrong world.
"""

import math

import numpy as np
import pytest

from selfdrive.world.geometry import World, obb_corners, obb_sdf

WALL_X5 = [5.0, -10.0, 5.0, 10.0]  # vertical wall at x = 5


def test_empty_world_returns_max_range():
    w = World()
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=8.0)
    assert d == pytest.approx([8.0])


def test_ray_hits_wall_head_on():
    w = World(segments=[WALL_X5])
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=20.0)
    assert d == pytest.approx([5.0])


def test_ray_pointing_away_misses():
    w = World(segments=[WALL_X5])
    d = w.raycast([[0.0, 0.0]], [[-1.0, 0.0]], max_range=20.0)
    assert d == pytest.approx([20.0])


def test_ray_is_clipped_to_max_range():
    w = World(segments=[WALL_X5])
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=3.0)
    assert d == pytest.approx([3.0])


def test_diagonal_ray_hits_at_correct_distance():
    # 45 degrees up-right meets x=5 at y=5, which is still on the segment.
    w = World(segments=[WALL_X5])
    d = w.raycast([[0.0, 0.0]], [[1.0, 1.0]], max_range=20.0)
    assert d == pytest.approx([5.0 * math.sqrt(2.0)])


def test_ray_misses_past_the_end_of_a_short_wall():
    # Same wall, but only spanning y in [4, 10]; a shallow ray passes under it.
    w = World(segments=[[5.0, 4.0, 5.0, 10.0]])
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.1]], max_range=20.0)
    assert d == pytest.approx([20.0])


def test_direction_need_not_be_normalized():
    w = World(segments=[WALL_X5])
    a = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=20.0)
    b = w.raycast([[0.0, 0.0]], [[17.3, 0.0]], max_range=20.0)
    assert a == pytest.approx(b)


def test_ray_hits_near_side_of_circle():
    w = World(circles=[[3.0, 0.0, 1.0]])
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=20.0)
    assert d == pytest.approx([2.0])


def test_ray_starting_inside_circle_uses_far_root():
    # Degenerate but reachable during a collision frame; must not return a negative hit.
    w = World(circles=[[0.0, 0.0, 1.0]])
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=20.0)
    assert d == pytest.approx([1.0])


def test_nearest_primitive_wins_across_types():
    w = World(segments=[WALL_X5], circles=[[3.0, 0.0, 1.0]])
    d = w.raycast([[0.0, 0.0]], [[1.0, 0.0]], max_range=20.0)
    assert d == pytest.approx([2.0])


def test_many_rays_in_one_call_match_individual_calls():
    rng = np.random.default_rng(0)
    w = World(segments=[WALL_X5, [-4.0, -6.0, 6.0, -6.0]], circles=[[2.0, 2.0, 0.5]])
    angles = rng.uniform(-math.pi, math.pi, size=16)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    origins = np.zeros((16, 2))
    batch = w.raycast(origins, dirs, max_range=12.0)
    one_at_a_time = [w.raycast(origins[i : i + 1], dirs[i : i + 1], 12.0)[0] for i in range(16)]
    assert batch == pytest.approx(one_at_a_time)


# --- collision and clearance -------------------------------------------------

BODY = dict(length=0.40, width=0.20)  # body spans x in [-0.2, 0.2], y in [-0.1, 0.1] at theta=0


def test_obb_corners_rotate_correctly():
    c = obb_corners(0.0, 0.0, math.pi / 2, **BODY)
    # Rotated a quarter turn, the long axis now lies along y.
    assert np.abs(c[:, 0]).max() == pytest.approx(0.10)
    assert np.abs(c[:, 1]).max() == pytest.approx(0.20)


def test_sdf_sign():
    assert obb_sdf(np.array([[0.0, 0.0]]), 0, 0, 0, **BODY)[0] < 0.0  # inside
    assert obb_sdf(np.array([[0.2, 0.0]]), 0, 0, 0, **BODY)[0] == pytest.approx(0.0)  # surface
    assert obb_sdf(np.array([[0.5, 0.0]]), 0, 0, 0, **BODY)[0] == pytest.approx(0.3)  # outside


def test_wall_through_body_collides():
    w = World(segments=[[0.1, -1.0, 0.1, 1.0]])
    assert w.collides(0.0, 0.0, 0.0, **BODY)


def test_distant_wall_does_not_collide_and_clearance_is_exact():
    w = World(segments=[[0.5, -1.0, 0.5, 1.0]])
    assert not w.collides(0.0, 0.0, 0.0, **BODY)
    assert w.clearance(0.0, 0.0, 0.0, **BODY) == pytest.approx(0.3)


def test_short_wall_fully_inside_body_collides():
    # Crosses none of the four body edges, so the containment check has to catch it.
    w = World(segments=[[-0.05, 0.0, 0.05, 0.0]])
    assert w.collides(0.0, 0.0, 0.0, **BODY)


def test_rotation_changes_collision_outcome():
    w = World(segments=[[0.15, -1.0, 0.15, 1.0]])
    assert w.collides(0.0, 0.0, 0.0, **BODY)  # long axis along x, reaches 0.20
    assert not w.collides(0.0, 0.0, math.pi / 2, **BODY)  # turned, only reaches 0.10


def test_circle_collision_accounts_for_radius():
    assert not World(circles=[[0.5, 0.0, 0.10]]).collides(0.0, 0.0, 0.0, **BODY)
    assert World(circles=[[0.5, 0.0, 0.35]]).collides(0.0, 0.0, 0.0, **BODY)


def test_clearance_subtracts_circle_radius():
    w = World(circles=[[0.5, 0.0, 0.10]])
    assert w.clearance(0.0, 0.0, 0.0, **BODY) == pytest.approx(0.2)


def test_clearance_is_measured_from_the_body_not_the_centre():
    # A wall broadside to the middle of the long edge. Measuring only from the
    # corners would report 0.427 here instead of the true 0.4 gap.
    w = World(segments=[[-0.05, 0.5, 0.05, 0.5]])
    assert w.clearance(0.0, 0.0, 0.0, **BODY) == pytest.approx(0.4)


def test_clearance_is_infinite_in_an_empty_world():
    assert World().clearance(0.0, 0.0, 0.0, **BODY) == math.inf


def test_clearance_never_negative_on_overlap():
    w = World(segments=[[0.0, -1.0, 0.0, 1.0]])
    assert w.collides(0.0, 0.0, 0.0, **BODY)
    assert w.clearance(0.0, 0.0, 0.0, **BODY) >= 0.0
