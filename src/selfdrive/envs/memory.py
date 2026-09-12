"""Short-term obstacle memory, reduced to a fixed ring of sectors around the car.

The stacked frames hold about a tenth of a second of history, and the camera covers only
a 55-125 degree slice of it. A wall that left the field of view a second ago is simply
gone, which is why a memoryless policy sweeps its nose side to side to look, and why a
three-point turn in a dead end is a guess. This keeps every obstacle point measured in
the last `memory_seconds`, by the depth arc or an ultrasonic, in a world frame, and every
step re-projects them around the car's current estimated pose.

Positions, not raw frames. Stacking 90 frames would hand the policy 1,530 floats taken
from 90 different poses and leave it to learn the geometry relating them. This does that
geometry by hand and passes on 2 x `memory_sectors` floats.

Layout, with S = `memory_sectors`, every value in [-1, 1]:

    [0 : S]    distance to the nearest remembered point in each sector
               0 m -> -1, `norm_memory_max` or more -> +1
    [S : 2S]   age of that same point
               0 s -> -1, `memory_seconds` -> +1

An empty sector reads +1 in both halves. Sector 0 is centred straight ahead and indices
increase counter-clockwise, the same sign convention as steering: with 24 sectors, 6 is
the left side, 12 behind and 18 the right. Distances are measured from the pose centre,
not from a sensor.

The phone has to rebuild this from ARCore's pose using the same rules, so they are
written out for the Android side in `docs/memory-ring.md`, with a worked example that
`tests/test_memory.py` pins.
"""

from __future__ import annotations

import math

import numpy as np


class EgoMemory:
    """Ring buffer of world-frame points, one row per control step."""

    def __init__(self, sectors: int, seconds: float, norm_max: float, points_per_step: int):
        self.sectors = int(sectors)
        self.seconds = float(seconds)
        self.norm_max = float(norm_max)
        self.width = int(points_per_step)
        self._sector_rad = 2.0 * math.pi / self.sectors
        # x and y in separate arrays: masking a flat 1-D array is several times cheaper
        # than gathering rows of an (n, 2) one, and this runs every step in every worker.
        self._x = np.zeros((1, self.width))
        self._y = np.zeros((1, self.width))
        self._stamp = np.full((1, self.width), -np.inf)
        self._row = 0

    def reset(self, dt: float) -> None:
        # Stamps are never later than the step that inserts them, so a row is overwritten
        # only once everything in it is older than `seconds`.
        rows = math.ceil(self.seconds / dt) + 1
        self._x = np.zeros((rows, self.width))
        self._y = np.zeros((rows, self.width))
        self._stamp = np.full((rows, self.width), -np.inf)
        self._row = 0

    def clear(self) -> None:
        self._stamp.fill(-np.inf)

    def insert(self, points: np.ndarray, stamps: np.ndarray) -> None:
        """Store this step's world-frame points, each with the time it was measured.

        Call exactly once per step, with an empty array when there is nothing new; the
        buffer is sized on the assumption that rows advance at the control rate."""
        row = self._row
        self._row = (row + 1) % len(self._stamp)
        self._stamp[row] = -np.inf
        k = len(points)
        if k:
            self._x[row, :k] = points[:, 0]
            self._y[row, :k] = points[:, 1]
            self._stamp[row, :k] = stamps

    def live(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        """Points still remembered, in the world frame, and their ages in seconds."""
        stamp = self._stamp.ravel()
        keep = stamp >= now - self.seconds
        return (np.column_stack([self._x.ravel()[keep], self._y.ravel()[keep]]),
                now - stamp[keep])

    def ring(self, x: float, y: float, theta: float, now: float) -> np.ndarray:
        """The observation block for a car at pose (x, y, theta)."""
        out = np.ones(2 * self.sectors, dtype=np.float32)
        stamp = self._stamp.ravel()
        keep = stamp >= now - self.seconds
        if not keep.any():
            return out

        c, s = math.cos(theta), math.sin(theta)
        dx, dy = self._x.ravel()[keep] - x, self._y.ravel()[keep] - y
        fwd, left = c * dx + s * dy, c * dy - s * dx
        dist = np.hypot(fwd, left)
        near = dist < self.norm_max
        if not near.any():
            return out
        fwd, left, dist, stamp = fwd[near], left[near], dist[near], stamp[keep][near]

        # floor(bearing / width + 0.5) mod S. Shifting by a whole turn first keeps every
        # value positive, so the integer cast is the floor.
        shifted = np.arctan2(left, fwd) / self._sector_rad + (self.sectors + 0.5)
        sector = shifted.astype(np.int64) % self.sectors

        # Nearest point per sector without a Python loop: sort by sector and, within a
        # sector, by distance (dist / norm_max is in [0, 1)), then take the first of each run.
        order = np.argsort(sector + dist / self.norm_max)
        sorted_sector = sector[order]
        first = np.empty(len(order), dtype=bool)
        first[0] = True
        np.not_equal(sorted_sector[1:], sorted_sector[:-1], out=first[1:])
        pick, idx = order[first], sorted_sector[first]

        out[idx] = dist[pick] / self.norm_max * 2.0 - 1.0
        out[self.sectors + idx] = np.minimum((now - stamp[pick]) / self.seconds, 1.0) * 2.0 - 1.0
        return out
