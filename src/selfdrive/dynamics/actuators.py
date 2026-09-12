"""Command-to-actuator response.

This layer, not the tire model, is where the sim-to-real gap is won. A policy trained
against instantaneous, perfect actuators learns maneuvers a real servo and brushed motor
cannot execute, and it falls apart the first time it runs on hardware. Everything here
is cheap and every term maps to something measurable on the real car.
"""

from __future__ import annotations

import math

import numpy as np

from .base import CarParams


class Actuators:
    """Holds the physical actuator state between steps."""

    def __init__(self, params: CarParams):
        self.p = params
        self.steer = 0.0
        self.speed = 0.0

    def reset(self, steer: float = 0.0, speed: float = 0.0) -> None:
        self.steer = steer
        self.speed = speed

    def target_speed(self, throttle_cmd: float) -> float:
        """Map a throttle command in [-1, 1] to the speed the drivetrain is aiming for."""
        p = self.p
        t = float(np.clip(throttle_cmd, -1.0, 1.0))
        mag = abs(t)
        if mag <= p.throttle_deadband:
            return 0.0
        # Rescale past the deadband so full stick still means full speed.
        mag = (mag - p.throttle_deadband) / (1.0 - p.throttle_deadband)
        mag = min(mag * p.throttle_gain, 1.0)
        v_max = p.max_speed_fwd if t > 0.0 else p.max_speed_rev
        return math.copysign(mag * v_max, t)

    def step(self, steer_cmd: float, throttle_cmd: float, dt: float) -> tuple[float, float]:
        """Advance the actuators one control period. Returns (steer_rad, speed_mps)."""
        p = self.p

        # Steering: slew-rate limited travel toward the command, offset by linkage trim.
        target = float(np.clip(steer_cmd, -1.0, 1.0)) * p.max_steer_rad + p.steer_trim_rad
        target = float(np.clip(target, -p.max_steer_rad, p.max_steer_rad))
        max_delta = p.steer_rate_rad_s * dt
        self.steer += float(np.clip(target - self.steer, -max_delta, max_delta))

        # Drivetrain: one first-order lag gives both acceleration lag and coast-down.
        v_target = self.target_speed(throttle_cmd)
        alpha = 1.0 - math.exp(-dt / max(p.accel_tau, 1e-3))
        self.speed += alpha * (v_target - self.speed)
        # An exponential never reaches zero, so a coasting car would creep forever.
        if v_target == 0.0 and abs(self.speed) < p.stop_eps:
            self.speed = 0.0

        return self.steer, self.speed
