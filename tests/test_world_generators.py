"""Arena generation invariants.

A generator that occasionally spawns the car inside a wall, or produces an arena with no
boundary, poisons training in a way that looks like a bad reward function.
"""

import numpy as np
import pytest

from selfdrive.world.generators import ArenaParams, make_arena, sample_spawn

BODY = dict(length=0.40, width=0.20)


@pytest.mark.parametrize("kind", ["indoor", "outdoor", "mixed"])
def test_every_arena_kind_is_bounded_and_non_empty(kind):
    rng = np.random.default_rng(0)
    w = make_arena(rng, ArenaParams(kind=kind))
    assert w.bounds is not None
    assert len(w.segments) >= 4  # at least the perimeter
    x0, y0, x1, y1 = w.bounds
    assert x1 > x0 and y1 > y0


def test_outdoor_has_cones_and_indoor_has_interior_walls():
    rng = np.random.default_rng(1)
    outdoor = make_arena(rng, ArenaParams(kind="outdoor", cone_density=(0.05, 0.10)))
    indoor = make_arena(rng, ArenaParams(kind="indoor", interior_walls=(5, 8),
                                         doorway_prob=0.0))
    assert len(outdoor.circles) > 0
    assert len(indoor.segments) > 4
    assert len(indoor.circles) == 0


def test_random_kind_produces_a_mixture_over_seeds():
    kinds = set()
    for seed in range(30):
        w = make_arena(np.random.default_rng(seed), ArenaParams(kind="random"))
        kinds.add((len(w.segments) > 4, len(w.circles) > 0))
    assert len(kinds) > 1


def test_arena_generation_is_reproducible_for_a_seed():
    a = make_arena(np.random.default_rng(5), ArenaParams(kind="mixed"))
    b = make_arena(np.random.default_rng(5), ArenaParams(kind="mixed"))
    np.testing.assert_array_equal(a.segments, b.segments)
    np.testing.assert_array_equal(a.circles, b.circles)


@pytest.mark.parametrize("seed", range(25))
def test_spawn_is_never_inside_an_obstacle(seed):
    rng = np.random.default_rng(seed)
    world = make_arena(rng, ArenaParams(kind="mixed"))
    x, y, theta = sample_spawn(world, rng, clearance=0.5, **BODY)
    assert not world.collides(x, y, theta, **BODY)


def test_spawn_respects_the_requested_clearance_when_it_can():
    rng = np.random.default_rng(3)
    world = make_arena(rng, ArenaParams(kind="outdoor", cone_density=(0.0, 0.0)))
    x, y, theta = sample_spawn(world, rng, clearance=1.0, **BODY)
    assert world.clearance(x, y, theta, **BODY) >= 1.0


def test_doorway_leaves_a_real_gap():
    # A wall with a doorway becomes two segments with clear space between them.
    from selfdrive.world.generators import _with_doorway

    parts = _with_doorway([0.0, 0.0, 6.0, 0.0], gap=1.0, rng=np.random.default_rng(0))
    assert len(parts) == 2
    # Something must be able to pass through the middle.
    gap_start = parts[0][2]
    gap_end = parts[1][0]
    assert gap_end - gap_start == pytest.approx(1.0)


def test_oversized_doorway_removes_the_wall_entirely():
    from selfdrive.world.generators import _with_doorway

    assert _with_doorway([0.0, 0.0, 0.5, 0.0], gap=2.0, rng=np.random.default_rng(0)) == []
