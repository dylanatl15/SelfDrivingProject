"""Where the car can drive to: reachable poses (x, y, heading), not just reachable floor.

Whether a gap is passable is not a question of its width alone. The car is a 0.40 x 0.20 m
box that turns no tighter than its turning radius. It fits through a 0.3 m doorway it meets
straight on, and it can back and fill its way out of a pocket whose way out is a slot. A
floor raster inflated by some fixed margin gets both wrong in one direction or the other,
and random walls make plenty of both. Goals must never be impossible, and the goal reward
must not pay the car to squeeze toward a gap it cannot pass, so both are built on this.

Poses live on a lattice: the navigation raster's cells times `headings` directions. A pose
is free when the car body there clears every obstacle, judged against the raster's
clearance field with a row of discs that covers the body. The moves are the ones a car
makes, forward or reversing: an arc at the turning radius that turns one heading step
left or right, and straight runs. Snapping a move to the lattice shifts it by up to half a
cell, so the answer is approximate at that scale. At 0.075 m cells a doorway 0.30 m wide
always passes, and one narrower than the car never does.

That is reachability with manoeuvres of a sensible size. A real car shuffling back and
forth in centimetre steps can turn around in any space its body can rotate in, so a pose
this rules out is not strictly impossible. It is only somewhere no policy should be sent.

Reversing along a move undoes it, so reachability is symmetric and the free poses split
into connected components. A flood applies every move to whole lattice layers at once and
repeats until nothing changes. Straight runs come in lengths of 1, 2, 4... steps, so a
round crosses a long room in one go and the flood settles in a few rounds.
"""

from __future__ import annotations

import math

import numpy as np

from .navigation import NavGrid

STRAIGHT_DOUBLINGS = 8  # straight runs of up to 2**8 - 1 steps in one round of the flood


def _shift(a: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """out[..., i, j] = a[..., i + di, j + dj], and `fill` where that falls off the grid."""
    nx, ny = a.shape[-2:]
    out = np.full_like(a, fill)
    if abs(di) < nx and abs(dj) < ny:
        out[..., max(-di, 0) : nx - max(di, 0), max(-dj, 0) : ny - max(dj, 0)] = (
            a[..., max(di, 0) : nx - max(-di, 0), max(dj, 0) : ny - max(-dj, 0)])
    return out


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return x * c - y * s, x * s + y * c


class PoseReach:
    """The pose lattice over one episode's navigation raster, and floods across it."""

    def __init__(self, grid: NavGrid, length: float, width: float, turn_radius: float,
                 headings: int = 16):
        if headings < 4 or headings % 4:
            # Axis-aligned walls need axis-aligned headings to pass their doorways.
            raise ValueError("headings must be a positive multiple of 4")
        self.grid = grid
        self.length = length
        self.headings = headings
        self.step = 2.0 * math.pi / headings
        res = grid.res

        # Discs along the body's centre line, a quarter of its width apart, each just big
        # enough to cover its slice of the body. Clearance is blended from cell centres,
        # which reads up to res^2 / 8r long at distance r from a wall's end; the margin
        # takes that back, so a body exactly as wide as a gap never fits through it.
        n = max(math.ceil(4.0 * length / width), 1)
        spacing = length / n
        self.disc = math.hypot(spacing / 2.0, width / 2.0)
        self.margin = res * res / (8.0 * self.disc)
        if self.disc - spacing / 2.0 < res / math.sqrt(2.0):
            raise ValueError(f"cells of {res} m are too coarse for a body {width} m wide")
        if self.disc > grid.cap:
            raise ValueError("the clearance raster is capped below the body's discs")
        along = -length / 2.0 + (np.arange(n) + 0.5) * spacing

        # Free poses at every half heading step: an arc is checked at its middle too. The
        # body is a centred box, so turning it half a circle leaves it where it was.
        half = [self._body_clear(along, k * self.step / 2.0) for k in range(headings)]
        self.free = np.stack(half + half)
        self._moves = [*self._arcs(turn_radius), *self._straights()]

    def _body_clear(self, along: np.ndarray, heading: float) -> np.ndarray:
        """Whether the body clears every obstacle, per cell, at one heading."""
        room, res = self.grid.room, self.grid.res
        clear = np.ones(room.shape, dtype=bool)
        for a in along:
            fx, fy = a * math.cos(heading) / res, a * math.sin(heading) / res
            i, j = math.floor(fx), math.floor(fy)
            tx, ty = fx - i, fy - j
            at = np.zeros(room.shape)
            for di, dj, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                              (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
                if w > 1e-9:
                    at += w * _shift(room, i + di, j + dj, 0.0)
            clear &= at >= self.disc + self.margin
        return clear

    def _snap(self, x: float, y: float) -> tuple[int, int]:
        return round(x / self.grid.res), round(y / self.grid.res)

    def _arcs(self, radius: float):
        """Arcs turning one heading step at the turning radius, as (from, to, cells, ok)."""
        d, h_count = self.step, self.headings
        for h in range(h_count):
            for turn in (1, -1):
                end = self._snap(*_rotate(radius * math.sin(d),
                                          turn * radius * (1 - math.cos(d)), h * d))
                mid = self._snap(*_rotate(radius * math.sin(d / 2),
                                          turn * radius * (1 - math.cos(d / 2)), h * d))
                to = (h + turn) % h_count
                ok = (self.free[2 * h]
                      & _shift(self.free[(2 * h + turn) % (2 * h_count)], *mid, False)
                      & _shift(self.free[2 * to], *end, False))
                yield h, to, end, ok

    def _straights(self):
        """Straight runs of 1, 2, 4... steps, each clear at every step along the way."""
        for h in range(self.headings):
            v = self._straight_step(h * self.step)
            ok = self.free[2 * h] & _shift(self.free[2 * h], *v, False)
            for _ in range(STRAIGHT_DOUBLINGS):
                yield h, h, v, ok
                ok = ok & _shift(ok, *v, False)
                v = (2 * v[0], 2 * v[1])

    def _straight_step(self, heading: float) -> tuple[int, int]:
        """The lattice step nearest `heading` in direction, at most half a body long, so a
        wall of no thickness cannot slip between two poses the check looks at.

        Only steps with no common factor, (1, 0) rather than (2, 0): a longer one visits
        every other cell along its line, and a corridor the car can only drive straight
        along would reach the path-distance floor with holes in it."""
        res = self.grid.res
        reach = max(int(self.length / 2.0 / res), 1)
        best, best_key = (1, 0), (math.inf, 0.0)
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                n = math.hypot(i, j)
                if n == 0 or n * res > self.length / 2.0 + 1e-9 or math.gcd(i, j) != 1:
                    continue
                err = abs(math.remainder(math.atan2(j, i) - heading, 2.0 * math.pi))
                key = (round(err, 9), -n)
                if key < best_key:
                    best, best_key = (i, j), key
        return best

    def index(self, x: float, y: float, theta: float) -> tuple[int, int, int]:
        """The lattice pose (heading, ix, iy) nearest a pose; cells may be off the grid."""
        g = self.grid
        return (round(theta / self.step) % self.headings,
                round((x - g.origin[0]) / g.res), round((y - g.origin[1]) / g.res))

    def flood(self, seed: tuple[int, int, int]) -> np.ndarray:
        """Every pose reachable from `seed`, as a (headings, nx, ny) mask."""
        reach = np.zeros((self.headings, self.grid.nx, self.grid.ny), dtype=bool)
        if not self.free[2 * seed[0], seed[1], seed[2]]:
            return reach
        reach[seed] = True
        count = 1
        while True:
            for h, to, (di, dj), ok in self._moves:
                reach[to] |= _shift(reach[h] & ok, -di, -dj, False)  # driving the move
                reach[h] |= _shift(reach[to], di, dj, False) & ok  # and reversing along it
            now = np.count_nonzero(reach)
            if now == count:
                return reach
            count = now

    def largest_component(self) -> np.ndarray:
        """The biggest set of free poses the car can drive between."""
        unseen = self.free[::2].copy()
        best = np.zeros_like(unseen)
        while np.count_nonzero(unseen) > np.count_nonzero(best):
            # From the most open cell left, which lies in a big component if any does.
            room = np.where(unseen.any(axis=0), self.grid.room, -np.inf)
            ix, iy = np.unravel_index(int(np.argmax(room)), room.shape)
            part = self.flood((int(np.argmax(unseen[:, ix, iy])), int(ix), int(iy)))
            unseen &= ~part
            if np.count_nonzero(part) > np.count_nonzero(best):
                best = part
        return best
