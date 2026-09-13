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

from ..dynamics.base import CarParams, wrap_angle
from ..world.geometry import World
from ..world.navigation import NavGrid
from ..world.reachability import PoseReach


@dataclass
class GoalConfig:
    radius: float = 0.50  # metres from the car's centre that count as arrived
    distance_min: float = 2.0  # path metres from the car to a newly drawn goal
    distance_max: float = 12.0
    clearance: float = 0.50  # a goal sits at least this far from every obstacle
    grid_res: float = 0.075  # navigation raster cell, metres
    # Plan with the least agile car domain randomization draws, so no goal is out of reach
    # for any of them. The defaults draw a 0.25 m x 1.1 wheelbase at 22 degrees: 0.68 m.
    turn_radius: float = 0.70
    headings: int = 16  # heading steps in the pose lattice


class GoalTracker:
    """The current goal, the path-distance field to it, and progress along it."""

    def __init__(self, config: GoalConfig | None = None, car: CarParams | None = None):
        self.c = config or GoalConfig()
        self.car = car or CarParams()
        if self.c.clearance < self.c.radius:
            # Arrival is judged by straight-line distance, which is only a path distance
            # when no obstacle can stand inside the arrival circle.
            raise ValueError("goal clearance must be at least the arrival radius")
        self.grid: NavGrid | None = None
        self.poses: np.ndarray | None = None  # (headings, nx, ny): where the car can drive
        self._world: World | None = None
        self.goal: tuple[float, float] | None = None
        self.field: np.ndarray | None = None
        self.remaining = math.nan  # path metres to the goal at the last update
        self.reached = 0
        self.progress_m = 0.0

    def prepare(self, world: World) -> None:
        """Build this world's raster and find the poses the car can drive between.

        Random walls can seal off a pocket of floor. A car spawned in one could reach
        nothing outside it, so the environment only spawns where `connected` agrees. Goals
        are drawn from floor the car can drive to, and path distance runs over that floor
        only, so a gap too tight for the car is a wall to the reward as well.
        """
        c, car = self.c, self.car
        self._world = world
        self.grid = grid = NavGrid(world, c.grid_res, car.width / 2.0, c.clearance)
        reach = PoseReach(grid, car.length, car.width, c.turn_radius, c.headings)
        self.poses = reach.largest_component()
        self._step = reach.step
        grid.restrict(self.poses.any(axis=0))

    def connected(self, x: float, y: float, theta: float) -> bool:
        """Whether a pose is one the car can drive between, give or take a lattice step."""
        g, h_count = self.grid, self.c.headings
        h = round(theta / self._step)
        ix, iy = round((x - g.origin[0]) / g.res), round((y - g.origin[1]) / g.res)
        near = self.poses[[(h + k) % h_count for k in (-1, 0, 1)],
                          max(ix - 1, 0) : ix + 2, max(iy - 1, 0) : iy + 2]
        return bool(near.any())

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
