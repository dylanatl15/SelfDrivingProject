"""Chained goals for waypoint driving.

Each episode hands the car a goal somewhere on reachable floor. Reaching it pays a bonus
and a new goal appears, so one episode chains several. The environment reward pays for
closing path distance to the current goal (`world/navigation.py`), so a 1 m gap on the
shortest route is worth driving through, where the exploration reward valued it no more
than open floor anywhere else.

Two sides of that split, as everywhere else in the environment:

*   Progress, arrival and the goals themselves use the true pose and the true floor.
*   The policy sees range and bearing to the goal from the drifting odometry estimate,
    which is what the phone will have: a pin in ARCore's frame and ARCore's own pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dynamics.base import wrap_angle
from ..world.geometry import World
from ..world.navigation import NavGrid


@dataclass
class GoalConfig:
    radius: float = 0.50  # metres from the car's centre that count as arrived
    distance_min: float = 2.0  # path metres from the car to a newly drawn goal
    distance_max: float = 12.0
    clearance: float = 0.50  # a goal sits at least this far from every obstacle
    grid_res: float = 0.20  # path-distance raster cell, metres
    inflate: float = 0.20  # cells whose centre is this close to an obstacle are blocked


class GoalTracker:
    """The current goal, the path-distance field to it, and progress along it."""

    def __init__(self, config: GoalConfig | None = None):
        self.c = config or GoalConfig()
        if self.c.clearance < self.c.radius:
            # Arrival is judged by straight-line distance, which is only a path distance
            # when no obstacle can stand inside the arrival circle.
            raise ValueError("goal clearance must be at least the arrival radius")
        self.grid: NavGrid | None = None
        self.main: np.ndarray | None = None  # cells on the largest connected stretch of floor
        self._world: World | None = None
        self.goal: tuple[float, float] | None = None
        self.field: np.ndarray | None = None
        self.remaining = math.nan  # path metres to the goal at the last update
        self.reached = 0
        self.progress_m = 0.0

    def prepare(self, world: World) -> None:
        """Build this world's raster and find its largest connected stretch of floor.

        Random walls can seal off a pocket. A car spawned in one could reach nothing outside
        it and would spend the episode crashing or circling, so the environment only spawns
        where `connected` agrees. Goals are drawn from floor the car can reach, so none is
        ever impossible.
        """
        c = self.c
        self._world = world
        self.grid = grid = NavGrid(world, c.grid_res, c.inflate, c.clearance)
        unseen = ~grid.blocked
        self.main = np.zeros_like(unseen)
        while np.count_nonzero(unseen) > np.count_nonzero(self.main):
            # From the most open cell not yet reached, which lies in a big region if any does.
            ix, iy = divmod(int(np.argmax(np.where(unseen, grid.room, -np.inf))), grid.ny)
            region = np.isfinite(grid.field_from(*grid.centre(ix, iy)))
            unseen &= ~region
            if np.count_nonzero(region) > np.count_nonzero(self.main):
                self.main = region

    def connected(self, x: float, y: float) -> bool:
        """Whether (x, y) is on the largest connected stretch of floor."""
        cell = self.grid.free_cell_near(x, y, reach=1)
        return cell is not None and bool(self.main[cell])

    def reset(self, world: World, x: float, y: float, rng: np.random.Generator,
              pin: tuple[float, float] | None = None) -> None:
        """Draw the first goal, or place it at `pin`. Prepares `world` if not yet done."""
        if world is not self._world:
            self.prepare(world)
        self.reached = 0
        self.progress_m = 0.0
        if pin is not None:
            self._set_goal((float(pin[0]), float(pin[1])), x, y)
        else:
            self._set_goal(self._draw(self.grid.field_from(x, y), rng), x, y)

    def _draw(self, field: np.ndarray, rng: np.random.Generator) -> tuple[float, float] | None:
        c = self.c
        return self.grid.sample_goal(field, rng, (c.distance_min, c.distance_max), c.clearance)

    def _set_goal(self, goal: tuple[float, float] | None, x: float, y: float) -> None:
        self.goal = goal
        if goal is None:
            self.field, self.remaining = None, math.nan
            return
        self.field = self.grid.field_from(*goal)
        self.remaining = self.grid.distance(self.field, x, y)

    def update(self, x: float, y: float, rng: np.random.Generator) -> tuple[float, bool]:
        """Path metres closed since the last update, and whether the goal was reached.

        On arrival the next goal is drawn from the old goal's field. The car is within
        `radius` of the old goal, so that field measures distance from the car too, and a
        new goal costs one field rather than two.
        """
        if self.goal is None:
            return 0.0, False
        now = self.grid.distance(self.field, x, y)
        progress = 0.0
        if math.isfinite(now) and math.isfinite(self.remaining):
            progress = self.remaining - now
        if math.isfinite(now) or not math.isfinite(self.remaining):
            self.remaining = now  # scraping a wall reads NaN; keep the last good distance
        self.progress_m += progress

        arrived = math.hypot(x - self.goal[0], y - self.goal[1]) <= self.c.radius
        if arrived:
            self.reached += 1
            self._set_goal(self._draw(self.field, rng), x, y)
        return progress, arrived

    def vector(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Range and left-positive bearing from a pose to the goal; (inf, 0) without one."""
        if self.goal is None:
            return math.inf, 0.0
        dx, dy = self.goal[0] - x, self.goal[1] - y
        return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - theta)
