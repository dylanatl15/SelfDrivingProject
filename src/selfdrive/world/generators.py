"""Randomized arena generators.

Three families, sampled per episode so the policy generalizes rather than memorizing
one map: `indoor` (walls, corridors, doorways, dead ends), `outdoor` (open ground with
scattered cones inside a soft perimeter), and `mixed`.

Interior walls are deliberately allowed to form dead ends and tight pockets. Phase 1's
hardest requirement is that the car must never stay stuck, and it cannot learn to back
out of a trap it is never placed in.

Random walls also overlap into shapes no building has: parallel walls a hand's width
apart, a wall ending 20 cm short of another, doorways offset into a slot no car fits
through. `min_passage` redraws any wall or cone that would leave a gap narrower than it
between separate obstacles; touching and crossing walls still make corners. It is off
by default, and while off the generator draws exactly what it always has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import World, point_seg_distance

PLACEMENT_TRIES = 30  # draws per wall or cone before giving up on it, with min_passage set


@dataclass
class ArenaParams:
    kind: str = "mixed"  # indoor | outdoor | mixed | random
    size_min: float = 8.0
    size_max: float = 18.0
    interior_walls: tuple[int, int] = (3, 10)
    wall_frac: tuple[float, float] = (0.25, 0.60)  # wall length as a fraction of arena size
    # Density-based layout, off by default. A wall count and a length that is a fraction of the
    # arena size thin out as the arena grows: doubling the size halves the wall length per m2.
    # With these set, the count scales with floor area and the length is in metres, so a
    # bigger arena gets more walls of the same size.
    walls_per_100m2: tuple[float, float] | None = None
    wall_length_m: tuple[float, float] | None = None
    doorway_prob: float = 0.55
    doorway_width: tuple[float, float] = (0.70, 1.20)
    cone_density: tuple[float, float] = (0.02, 0.10)  # cones per square metre
    cone_radius: tuple[float, float] = (0.08, 0.35)
    spawn_clearance: float = 0.50  # metres of free space required around the car at reset
    min_passage: float = 0.0  # narrowest gap left between separate obstacles; 0 = any


def rect_walls(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """The four boundary segments of an axis-aligned rectangle."""
    return [
        [x0, y0, x1, y0],
        [x1, y0, x1, y1],
        [x1, y1, x0, y1],
        [x0, y1, x0, y0],
    ]


def _with_doorway(
    seg: list[float], gap: float, rng: np.random.Generator
) -> list[list[float]]:
    """Punch a gap through the middle stretch of a wall, yielding up to two segments."""
    x0, y0, x1, y1 = seg
    length = math.hypot(x1 - x0, y1 - y0)
    if gap >= length:
        return []  # the doorway swallowed the whole wall
    # Keep the opening away from the very ends so it reads as a door, not a short wall.
    centre = rng.uniform(gap / 2.0 + 0.1, length - gap / 2.0 - 0.1)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    a_end = centre - gap / 2.0
    b_start = centre + gap / 2.0

    out = []
    if a_end > 1e-3:
        out.append([x0, y0, x0 + ux * a_end, y0 + uy * a_end])
    if length - b_start > 1e-3:
        out.append([x0 + ux * b_start, y0 + uy * b_start, x1, y1])
    return out


def _interior_walls(
    rng: np.random.Generator, x0: float, y0: float, x1: float, y1: float, p: ArenaParams
) -> list[list[float]]:
    size = min(x1 - x0, y1 - y0)
    margin = 1.0
    if p.walls_per_100m2 is not None:
        area = (x1 - x0) * (y1 - y0)
        n = int(round(rng.uniform(*p.walls_per_100m2) * area / 100.0))
    else:
        n = int(rng.integers(p.interior_walls[0], p.interior_walls[1] + 1))
    walls: list[list[float]] = []
    boundary = rect_walls(x0, y0, x1, y1)
    tries = PLACEMENT_TRIES if p.min_passage > 0 else 1

    for _ in range(n):
        for _ in range(tries):
            if p.wall_length_m is not None:
                length = rng.uniform(*p.wall_length_m)
            else:
                length = rng.uniform(*p.wall_frac) * size
            horizontal = rng.random() < 0.5
            if horizontal:
                length = min(length, (x1 - x0) - 2 * margin)
                sx = rng.uniform(x0 + margin, x1 - margin - length)
                sy = rng.uniform(y0 + margin, y1 - margin)
                seg = [sx, sy, sx + length, sy]
            else:
                length = min(length, (y1 - y0) - 2 * margin)
                sx = rng.uniform(x0 + margin, x1 - margin)
                sy = rng.uniform(y0 + margin, y1 - margin - length)
                seg = [sx, sy, sx, sy + length]

            if rng.random() < p.doorway_prob:
                pieces = _with_doorway(seg, rng.uniform(*p.doorway_width), rng)
            else:
                pieces = [seg]
            if p.min_passage <= 0 or _keeps_passage(pieces, boundary + walls, p.min_passage):
                walls.extend(pieces)
                break

    return walls


def _keeps_passage(pieces: list[list[float]], placed: list[list[float]], gap: float) -> bool:
    """Whether axis-aligned wall pieces stay `gap` clear of every placed wall they do not
    touch. Touching or crossing is fine: that is a corner, not a slot."""
    if not pieces or not placed:
        return True
    old = np.asarray(placed, dtype=float)
    old_lo, old_hi = np.minimum(old[:, :2], old[:, 2:]), np.maximum(old[:, :2], old[:, 2:])
    for piece in pieces:
        a = np.asarray(piece, dtype=float)
        lo, hi = np.minimum(a[:2], a[2:]), np.maximum(a[:2], a[2:])
        apart = np.maximum(0.0, np.maximum(old_lo - hi, lo - old_hi))
        d = np.hypot(apart[:, 0], apart[:, 1])
        if np.any((d > 1e-9) & (d < gap)):
            return False
    return True


def _cones(
    rng: np.random.Generator, x0: float, y0: float, x1: float, y1: float, p: ArenaParams,
    segments: list[list[float]],
) -> list[list[float]]:
    area = (x1 - x0) * (y1 - y0)
    n = int(rng.uniform(*p.cone_density) * area)
    if n <= 0:
        return []
    margin = 0.6
    if p.min_passage > 0:
        return _spaced_cones(rng, n, (x0 + margin, y0 + margin, x1 - margin, y1 - margin), p,
                             segments)
    cx = rng.uniform(x0 + margin, x1 - margin, size=n)
    cy = rng.uniform(y0 + margin, y1 - margin, size=n)
    cr = rng.uniform(p.cone_radius[0], p.cone_radius[1], size=n)
    return np.stack([cx, cy, cr], axis=1).tolist()


def _spaced_cones(rng: np.random.Generator, n: int, box, p: ArenaParams,
                  segments: list[list[float]]) -> list[list[float]]:
    """Cones one at a time, each redrawn until it is `min_passage` clear of every wall and
    every cone already placed."""
    seg = np.asarray(segments, dtype=float).reshape(-1, 4)
    a, e = seg[:, :2], seg[:, 2:] - seg[:, :2]
    cones: list[list[float]] = []
    for _ in range(n):
        for _ in range(PLACEMENT_TRIES):
            cx, cy = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
            cr = rng.uniform(*p.cone_radius)
            if len(seg):
                wall_gap = point_seg_distance(np.array([[cx, cy]]), a, e).min() - cr
                if wall_gap < p.min_passage:
                    continue
            if cones:
                c = np.asarray(cones)
                if np.any(np.hypot(c[:, 0] - cx, c[:, 1] - cy) - c[:, 2] - cr < p.min_passage):
                    continue
            cones.append([cx, cy, cr])
            break
    return cones


def make_arena(rng: np.random.Generator, params: ArenaParams | None = None) -> World:
    """Build one randomized arena."""
    p = params or ArenaParams()
    if p.min_passage > 0 and p.doorway_width[0] < p.min_passage:
        raise ValueError(f"doorways from {p.doorway_width[0]} m are narrower than "
                         f"min_passage {p.min_passage} m")
    kind = p.kind
    if kind == "random":
        kind = str(rng.choice(["indoor", "outdoor", "mixed"]))

    w = rng.uniform(p.size_min, p.size_max)
    h = rng.uniform(p.size_min, p.size_max)
    x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0

    segments = rect_walls(x0, y0, x1, y1)
    circles: list[list[float]] = []

    if kind in ("indoor", "mixed"):
        segments.extend(_interior_walls(rng, x0, y0, x1, y1, p))
    if kind in ("outdoor", "mixed"):
        circles.extend(_cones(rng, x0, y0, x1, y1, p, segments))

    return World(segments=segments, circles=circles, bounds=(x0, y0, x1, y1))


def sample_spawn(
    world: World,
    rng: np.random.Generator,
    length: float,
    width: float,
    clearance: float = 0.5,
    tries: int = 300,
) -> tuple[float, float, float]:
    """Find a pose with room around it. Falls back to the least-bad candidate."""
    x0, y0, x1, y1 = world.bounds
    margin = max(length, width)
    best, best_clear = (0.0, 0.0, 0.0), -math.inf

    for _ in range(tries):
        x = rng.uniform(x0 + margin, x1 - margin)
        y = rng.uniform(y0 + margin, y1 - margin)
        theta = rng.uniform(-math.pi, math.pi)
        if world.collides(x, y, theta, length, width):
            continue
        c = world.clearance(x, y, theta, length, width)
        if c >= clearance:
            return x, y, theta
        if c > best_clear:
            best, best_clear = (x, y, theta), c

    return best
