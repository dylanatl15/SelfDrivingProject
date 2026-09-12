"""Vehicle state, chassis parameters, and the model interface.

`VehicleModel` is a Protocol so a differential-drive model can be dropped in later
without the environment knowing. Phase 1 ships one implementation, the kinematic
bicycle, because the chassis is Ackermann (steering servo up front, rear drive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # heading, rad
    speed: float = 0.0  # signed body-forward speed, m/s
    steer: float = 0.0  # actual front wheel angle, rad

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.speed, self.steer], dtype=np.float64)


@dataclass
class CarParams:
    """Chassis constants, in SI units.

    Defaults describe a generic ~1/10 scale RC platform large enough to carry a phone,
    an ESP32 and a 2S LiPo. They are placeholders until the team measures the real car;
    domain randomization spans a wide enough band that the policy should not care.
    """

    # geometry
    wheelbase: float = 0.25
    length: float = 0.40
    width: float = 0.20

    # steering servo
    max_steer_rad: float = math.radians(28.0)
    steer_rate_rad_s: float = math.radians(200.0)  # servos slew, they do not teleport
    steer_trim_rad: float = 0.0  # mechanical misalignment of the linkage

    # drivetrain
    max_speed_fwd: float = 1.5
    max_speed_rev: float = 0.6
    accel_tau: float = 0.25  # first-order lag: acceleration AND stopping inertia
    throttle_deadband: float = 0.05  # small PWM does not overcome static friction
    throttle_gain: float = 1.0  # open-loop command-to-speed error; ~1.0 with encoders
    understeer: float = 0.05  # the same wheel angle bends the path less at speed
    stop_eps: float = 0.02  # below this, a zero-throttle car is treated as stopped

    @property
    def min_turn_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steer_rad)


class VehicleModel(Protocol):
    """Anything the environment can drive."""

    state: VehicleState

    def reset(self, x: float, y: float, theta: float) -> VehicleState: ...

    def step(self, steer_cmd: float, throttle_cmd: float, dt: float) -> VehicleState: ...
