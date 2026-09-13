"""Pose estimate from ARCore's visual-inertial odometry.

The obstacle memory stores points in a world frame and re-projects them around the car
every step, so it depends on knowing how far the car has moved since it saw each one. On
the phone that comes from ARCore motion tracking, which is good and not perfect:

*   It drifts. Error grows with distance travelled and angle turned, as a random walk, so
    each step's noise has a standard deviation proportional to the square root of that
    step's motion. That makes the accumulated error independent of `dt`.
*   Its scale can be slightly off, a monocular-camera artifact the IMU mostly corrects.
*   It loses tracking: fast rotation, blur on a vibrating chassis, a blank wall filling the
    view. Old points cannot be related to the new pose afterwards, so the memory is
    cleared, and nothing is stored until tracking returns.

Over the three seconds the memory spans, that is centimetres. It is modelled anyway,
because a policy trained on perfect odometry learns to trust remembered points to the
millimetre, and that trust does not transfer.

The memory and the goal block read this estimate. The goal block measures range and bearing
from it for a whole episode rather than three seconds, so its error grows to tens of
centimetres, and past a metre on the worst heading-drift draws. The rest of the environment
uses the true pose, and rewards always do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dynamics.base import wrap_angle


@dataclass
class OdometryParams:
    scale_error: float = 0.0  # fractional translation error; +0.02 reads 2 % long
    pos_noise: float = 0.03  # metres of drift per sqrt(metre) travelled
    yaw_noise: float = 0.02  # radians of drift per sqrt(radian) turned
    tracking_loss_per_s: float = 0.0  # expected tracking failures per second
    tracking_recover_s: float = 0.5  # seconds without a usable pose after a failure


class Odometry:
    """Dead-reckons the true motion into a corrupted estimate, one step at a time."""

    def __init__(self, params: OdometryParams | None = None):
        self.p = params or OdometryParams()
        self.x = self.y = self.theta = 0.0
        self._true = (0.0, 0.0, 0.0)
        self._lost_s = 0.0

    def reset(self, x: float, y: float, theta: float) -> None:
        # ARCore's origin is wherever tracking started. Only relative motion matters to the
        # memory, so the estimate simply starts at the true pose.
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self._true = (self.x, self.y, self.theta)
        self._lost_s = 0.0

    @property
    def tracking(self) -> bool:
        return self._lost_s <= 0.0

    def update(self, x: float, y: float, theta: float, dt: float,
               rng: np.random.Generator) -> bool:
        """Advance by the true motion since the last call. True on the step tracking is
        lost, which is the caller's cue to clear anything stored against the old pose."""
        p = self.p
        px, py, pt = self._true
        self._true = (float(x), float(y), float(theta))

        # The true step, expressed in the car's own frame at the previous pose.
        dx, dy = x - px, y - py
        c, s = math.cos(pt), math.sin(pt)
        fwd = (c * dx + s * dy) * (1.0 + p.scale_error)
        lat = (-s * dx + c * dy) * (1.0 + p.scale_error)
        turn = wrap_angle(theta - pt)

        dist = math.hypot(dx, dy)
        if p.pos_noise > 0.0 and dist > 0.0:
            sigma = p.pos_noise * math.sqrt(dist)
            fwd += rng.normal(0.0, sigma)
            lat += rng.normal(0.0, sigma)
        if p.yaw_noise > 0.0 and turn != 0.0:
            turn += rng.normal(0.0, p.yaw_noise * math.sqrt(abs(turn)))

        # Applied along the estimated heading, so a heading error bends every later step.
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x += c * fwd - s * lat
        self.y += s * fwd + c * lat
        self.theta = wrap_angle(self.theta + turn)

        if self._lost_s > 0.0:
            self._lost_s -= dt
        if p.tracking_loss_per_s > 0.0 and rng.random() < p.tracking_loss_per_s * dt:
            self._lost_s = p.tracking_recover_s
            return True
        return False
