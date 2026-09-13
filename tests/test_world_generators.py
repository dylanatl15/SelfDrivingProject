"""Arena generation invariants.

A generator that occasionally spawns the car inside a wall, or produces an arena with no
boundary, poisons training in a way that looks like a bad reward function.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from selfdrive.config import load_env_config
from selfdrive.world.generators import ArenaParams, _interior_walls, make_arena, sample_spawn

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


# --- density as the arena grows ----------------------------------------------
# Arenas bigger than the 8-18 m default are wanted, but only if they are not easier to
# drive. Cones are already placed per square metre; walls were a fixed count with a length
# proportional to the arena, so a bigger arena silently got sparser walls.

def wall_length_per_m2(params: ArenaParams, side: float, seeds: int = 300) -> float:
    total = 0.0
    half = side / 2.0
    for seed in range(seeds):
        walls = np.asarray(_interior_walls(np.random.default_rng(seed), -half, -half, half, half,
                                           params)).reshape(-1, 4)
        total += float(np.hypot(walls[:, 2] - walls[:, 0], walls[:, 3] - walls[:, 1]).sum())
    return total / (seeds * side * side)


def test_count_based_walls_thin_out_as_the_arena_grows():
    params = ArenaParams(doorway_prob=0.0)
    assert wall_length_per_m2(params, 30.0) < 0.5 * wall_length_per_m2(params, 10.0)


def test_density_based_walls_keep_wall_length_per_area():
    params = ArenaParams(doorway_prob=0.0, walls_per_100m2=(2.0, 6.0), wall_length_m=(2.0, 8.0))
    small = wall_length_per_m2(params, 12.0)
    big = wall_length_per_m2(params, 30.0)
    assert big == pytest.approx(small, rel=0.1)
    assert small == pytest.approx(4.0 * 5.0 / 100.0, rel=0.1)  # mean count x mean length


def test_density_based_wall_lengths_stay_in_range():
    params = ArenaParams(doorway_prob=0.0, walls_per_100m2=(4.0, 4.0), wall_length_m=(2.0, 3.0))
    walls = np.asarray(_interior_walls(np.random.default_rng(0), -15, -15, 15, 15, params))
    lengths = np.hypot(walls[:, 2] - walls[:, 0], walls[:, 3] - walls[:, 1])
    assert len(walls) == 36  # 4 per 100 m2 over 900 m2
    assert lengths.min() >= 2.0 and lengths.max() <= 3.0


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_big_arena_config_is_light_with_only_arena_size_and_walls_changed():
    light = yaml.safe_load((CONFIGS / "env_phase1_light.yaml").read_text())
    big = yaml.safe_load((CONFIGS / "env_phase1_light_big.yaml").read_text())
    light_arena, big_arena = light.pop("arena"), big.pop("arena")
    assert big == light
    assert light_arena.pop("interior_walls") == [3, 10]
    assert big_arena.pop("walls_per_100m2") == [6.0, 11.0]
    assert big_arena.pop("wall_length_m") == [2.0, 8.0]
    for key, before, after in (("size_min", 8.0, 16.0), ("size_max", 18.0, 32.0)):
        assert light_arena.pop(key) == before
        assert big_arena.pop(key) == after
    assert big_arena == light_arena


def obstacles_per_m2(config, seeds=300):
    params = load_env_config(CONFIGS / config).arena
    area = wall = cones = 0.0
    for seed in range(seeds):
        w = make_arena(np.random.default_rng(seed), params)
        x0, y0, x1, y1 = w.bounds
        seg = np.asarray(w.segments)
        area += (x1 - x0) * (y1 - y0)
        wall += float(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]).sum())
        cones += len(w.circles)
    return wall / area, cones / area


def test_big_arenas_are_as_cluttered_as_phase1_arenas():
    # Perimeter included: the car cannot tell an outer wall from an inner one.
    wall, cones = obstacles_per_m2("env_phase1_light.yaml")
    big_wall, big_cones = obstacles_per_m2("env_phase1_light_big.yaml")
    assert big_wall == pytest.approx(wall, rel=0.12)
    assert big_cones == pytest.approx(cones, rel=0.12)
