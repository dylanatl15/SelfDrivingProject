"""Path distance across the free floor, for the goal reward.

A reward for closing straight-line distance to a goal pays the car to press against
whatever wall stands between it and the goal, so every dead end and U facing the goal
becomes a place to get stuck. The goal reward reads path distance instead: metres along
the shortest route around walls and through gaps. Like every reward input it is ground
truth; the policy only ever sees range and bearing from its drifting odometry.

The floor is a raster of `res`-metre cells, built once per episode. A cell is blocked when
its centre lies within `inflate` of an obstacle. Moves go to the 8 neighbours at cost 1 or
sqrt(2) cells and never cut the corner of a blocked cell. `inflate` must be at least
`res / sqrt(2)`: every cell a wall passes through is then blocked, so a wall at any angle
is a solid barrier.

Octile distance reads up to 8 % long for a straight line 22.5 degrees off the grid axes,
which scales the pay for some goals a little. It cannot be farmed: progress is the change
in one fixed function of position, so any route back to where it started nets zero.

There is no scipy here, and a heap-based Dijkstra in Python is far too slow to run for
every goal. The field is computed by sweeps instead. Along every row, column and diagonal,
in both directions, one `np.minimum.accumulate` over `distance - position` carries a
distance across any number of free cells in a single call. Sweeping repeats until nothing
changes, which takes roughly one round per bend in the longest route. Distances are exact
integers (`AXIS` per cell along an axis, `DIAG` diagonally), so the sweeps reach a true
fixed point instead of chasing float rounding.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import World, point_seg_distance

AXIS = 10_000  # one cell along an axis, in integer distance units
DIAG = 14_142  # one cell diagonally
UNREACHED = np.int64(1) << 60


def _line_families(blocked: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rows, columns, diagonals and anti-diagonals of the grid, one tuple each.

    Each tuple holds the flat cell indices concatenated line by line, each cell's distance
    along its line, and a run id that increments wherever a move from the previous cell is
    not allowed: at the start of a line, into or out of a blocked cell, or diagonally past a
    blocked corner. A sweep must not carry a distance from one run into the next.
    """
    nx, ny = blocked.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    free = ~blocked
    pad = np.zeros((nx + 2, ny + 2), dtype=bool)
    pad[1:-1, 1:-1] = free

    def free_at(dx: int, dy: int) -> np.ndarray:
        """Element [i, j] is whether cell [i + dx, j + dy] is free; off-grid is not."""
        return pad[1 + dx : nx + 1 + dx, 1 + dy : ny + 1 + dy]

    lines = [
        # (sort key along which lines are concatenated, allowed move from the previous
        #  cell, distance along the line, cost per cell)
        ((iy, ix), free & free_at(-1, 0), ix, AXIS),
        ((ix, iy), free & free_at(0, -1), iy, AXIS),
        ((ix - iy, ix), free & free_at(-1, -1) & free_at(-1, 0) & free_at(0, -1),
         np.minimum(ix, iy), DIAG),
        ((ix + iy, ix), free & free_at(-1, 1) & free_at(-1, 0) & free_at(0, 1),
         ix - np.maximum(0, ix + iy - (ny - 1)), DIAG),
    ]
    out = []
    for (line_key, along_key), allowed, along, cost in lines:
        order = np.lexsort((along_key.ravel(), line_key.ravel()))
        run = np.cumsum(~allowed.ravel()[order], dtype=np.int64)
        out.append((order, along.ravel()[order].astype(np.int64) * cost, run))
    return out


class PathField:
    """Shortest 8-connected distances over one fixed blocked mask."""

    def __init__(self, blocked: np.ndarray):
        self.blocked = np.asarray(blocked, dtype=bool)
        nx, ny = self.blocked.shape
        self._lines = _line_families(self.blocked)
        # Lifts each run clear of every run before it. A distance carried in from another
        # run arrives at least `longest route + longest line` high, so it can never beat a
        # real distance, and anything above half the gap is known to be such a carry.
        self._gap = 2 * DIAG * (nx * ny + max(nx, ny)) + 2

    def distances(self, source: tuple[int, int], max_rounds: int = 1_000) -> np.ndarray:
        """Distance in cells from `source` to every cell; inf where blocked or unreachable."""
        nx, ny = self.blocked.shape
        if self.blocked[source]:
            return np.full((nx, ny), np.inf)
        d = np.full(nx * ny, UNREACHED, dtype=np.int64)
        d[source[0] * ny + source[1]] = 0
        real = self._gap // 2

        for _ in range(max_rounds):
            improved = False
            for order, along, run in self._lines:
                line = d[order]
                lift = run * self._gap
                ahead = np.minimum.accumulate(line - along - lift) + lift + along
                behind = np.minimum.accumulate((line + along + lift)[::-1])[::-1] - lift - along
                best = np.minimum(ahead, behind)
                better = (best < line) & (best < real)
                if better.any():
                    d[order[better]] = best[better]
                    improved = True
            if not improved:
                break
        else:
            raise RuntimeError(f"path field did not settle in {max_rounds} rounds")
        return np.where(d < real, d / AXIS, np.inf).reshape(nx, ny)


class NavGrid:
    """One episode's floor: obstacle clearance per cell, and path-distance fields over it."""

    def __init__(self, world: World, res: float, inflate: float, clearance_cap: float = 0.0):
        if inflate < res / math.sqrt(2.0) - 1e-9:
            raise ValueError(f"inflate {inflate} is below res / sqrt(2) = "
                             f"{res / math.sqrt(2.0):.3f}, so a diagonal wall could leak")
        self.res = float(res)
        self.inflate = float(inflate)
        x0, y0, x1, y1 = (float(b) for b in _bounds(world))
        self.nx = max(math.ceil((x1 - x0) / res), 1)
        self.ny = max(math.ceil((y1 - y0) / res), 1)
        self.origin = (x0 + res / 2.0, y0 + res / 2.0)  # centre of cell [0, 0]
        # Clearance is exact up to `cap` and reads `cap` beyond it.
        self.cap = max(inflate, clearance_cap) + res
        self.room = self._clearance(world)
        self.blocked = self.room < inflate
        self._paths = PathField(self.blocked)

    def _clearance(self, world: World) -> np.ndarray:
        """Distance from each cell centre to the nearest obstacle, capped at `cap`.

        Each primitive only touches the cells within `cap` of its bounding box, so a big
        arena costs one small window per wall rather than every cell against every wall.
        This runs once per episode, not per step.
        """
        res, cap = self.res, self.cap
        ox, oy = self.origin
        xs = ox + np.arange(self.nx) * res
        ys = oy + np.arange(self.ny) * res
        room = np.full((self.nx, self.ny), cap)

        def window(lo_x, lo_y, hi_x, hi_y):
            i0 = max(math.ceil((lo_x - cap - ox) / res), 0)
            i1 = min(math.floor((hi_x + cap - ox) / res), self.nx - 1)
            j0 = max(math.ceil((lo_y - cap - oy) / res), 0)
            j1 = min(math.floor((hi_y + cap - oy) / res), self.ny - 1)
            if i0 > i1 or j0 > j1:
                return None
            gx, gy = np.meshgrid(xs[i0 : i1 + 1], ys[j0 : j1 + 1], indexing="ij")
            return room[i0 : i1 + 1, j0 : j1 + 1], gx, gy

        for a, e in zip(world.seg_a, world.seg_e, strict=True):
            b = a + e
            win = window(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            if win is not None:
                view, gx, gy = win
                pts = np.column_stack([gx.ravel(), gy.ravel()])
                d = point_seg_distance(pts, a[None, :], e[None, :])[:, 0]
                np.minimum(view, d.reshape(gx.shape), out=view)

        for (cx, cy), r in zip(world.circ_c, world.circ_r, strict=True):
            win = window(cx - r, cy - r, cx + r, cy + r)
            if win is not None:
                view, gx, gy = win
                np.minimum(view, np.hypot(gx - cx, gy - cy) - r, out=view)
        return room

    def centre(self, ix: int, iy: int) -> tuple[float, float]:
        return self.origin[0] + ix * self.res, self.origin[1] + iy * self.res

    def free_cell_near(self, x: float, y: float, reach: int = 3) -> tuple[int, int] | None:
        """The free cell whose centre is nearest (x, y), searching `reach` cells around."""
        ix = round((x - self.origin[0]) / self.res)
        iy = round((y - self.origin[1]) / self.res)
        i0, i1 = max(ix - reach, 0), min(ix + reach, self.nx - 1)
        j0, j1 = max(iy - reach, 0), min(iy + reach, self.ny - 1)
        if i0 > i1 or j0 > j1:
            return None
        free = ~self.blocked[i0 : i1 + 1, j0 : j1 + 1]
        if not free.any():
            return None
        gx, gy = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        d2 = np.where(free, (self.origin[0] + gx * self.res - x) ** 2
                      + (self.origin[1] + gy * self.res - y) ** 2, np.inf)
        k = int(np.argmin(d2))
        return int(gx.ravel()[k]), int(gy.ravel()[k])

    def field_from(self, x: float, y: float) -> np.ndarray:
        """Path distance in metres from (x, y) to every cell centre; inf where unreachable."""
        start = self.free_cell_near(x, y)
        if start is None:
            return np.full((self.nx, self.ny), np.inf)
        return self._paths.distances(start) * self.res

    def distance(self, field: np.ndarray, x: float, y: float) -> float:
        """Path distance at any point, blended bilinearly from the reachable cell centres
        around it, so progress pays a little every step rather than once per cell.

        It is a fixed function of position, so what it pays along any closed route sums to
        zero. Next to a wall, where every surrounding centre is blocked, it falls back to the
        nearest free cell plus the straight line to it. NaN if nothing near is reachable.
        """
        fx = (x - self.origin[0]) / self.res
        fy = (y - self.origin[1]) / self.res
        i, j = math.floor(fx), math.floor(fy)
        tx, ty = fx - i, fy - j
        total = weight = 0.0
        for a, b, w in ((i, j, (1 - tx) * (1 - ty)), (i + 1, j, tx * (1 - ty)),
                        (i, j + 1, (1 - tx) * ty), (i + 1, j + 1, tx * ty)):
            if 0 <= a < self.nx and 0 <= b < self.ny and field[a, b] < math.inf:
                total += w * field[a, b]
                weight += w
        if weight > 1e-9:
            return float(total / weight)
        near = self.free_cell_near(x, y)
        if near is None or not field[near] < math.inf:
            return math.nan
        cx, cy = self.centre(*near)
        return float(field[near]) + math.hypot(x - cx, y - cy)

    def sample_goal(self, field: np.ndarray, rng: np.random.Generator,
                    distance: tuple[float, float], clearance: float) -> tuple[float, float] | None:
        """A random cell centre at a path distance within `distance` and at least `clearance`
        from every obstacle. If no cell is in range, the farthest cell that is clear enough,
        so a car boxed into a small pocket still has somewhere to go. None if there is no
        such cell at all."""
        if clearance > self.cap + 1e-9:
            raise ValueError(f"clearance {clearance} is beyond the raster cap {self.cap}")
        ok = np.isfinite(field) & (self.room >= clearance) & (field > 0.0)
        in_range = np.flatnonzero(ok & (field >= distance[0]) & (field <= distance[1]))
        if in_range.size:
            k = int(in_range[rng.integers(in_range.size)])
        else:
            reachable = np.flatnonzero(ok)
            if not reachable.size:
                return None
            k = int(reachable[np.argmax(field.ravel()[reachable])])
        return self.centre(*divmod(k, self.ny))


def _bounds(world: World) -> tuple[float, float, float, float]:
    """The world's declared bounds, or the extent of its primitives when it has none."""
    if world.bounds is not None:
        return tuple(world.bounds)
    pts = [world.seg_a, world.seg_a + world.seg_e]
    if len(world.circ_c):
        r = world.circ_r[:, None]
        pts += [world.circ_c - r, world.circ_c + r]
    pts = np.concatenate(pts)
    if not len(pts):
        raise ValueError("a world with no bounds and no obstacles has no floor to plan on")
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)
