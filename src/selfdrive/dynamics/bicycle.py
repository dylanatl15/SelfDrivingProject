"""Kinematic bicycle model.

Deliberately no tire friction, load transfer or voltage sag. At toy-car speeds the
kinematic model is accurate enough, and the realism that actually matters for transfer
lives in `actuators.py`. What this model does enforce is the constraint a point-mass
sim would let the policy cheat: the car cannot turn tighter than its minimum radius,
and it cannot move sideways.
"""

from __future__ import annotations

import math

from .actuators import Actuators
from .base import CarParams, VehicleState, wrap_angle


class KinematicBicycle:
    """Rear-axle-referenced kinematic bicycle."""

    def __init__(self, params: CarParams | None = None):
        self.p = params or CarParams()
        self.actuators = Actuators(self.p)
        self.state = VehicleState()

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> VehicleState:
        self.state = VehicleState(x=x, y=y, theta=theta)
        self.actuators.reset()
        return self.state

    def step(self, steer_cmd: float, throttle_cmd: float, dt: float) -> VehicleState:
        p = self.p
        steer, speed = self.actuators.step(steer_cmd, throttle_cmd, dt)

        # Understeer gradient: at speed the same wheel angle produces a wider arc.
        effective = steer / (1.0 + p.understeer * speed * speed)
        yaw_rate = (speed / p.wheelbase) * math.tan(effective)

        # Midpoint heading keeps the traced radius honest at a 30 Hz control period;
        # plain forward Euler visibly widens every turn.
        s = self.state
        theta_mid = s.theta + 0.5 * yaw_rate * dt
        s.x += speed * math.cos(theta_mid) * dt
        s.y += speed * math.sin(theta_mid) * dt
        s.theta = wrap_angle(s.theta + yaw_rate * dt)
        s.speed = speed
        s.steer = steer
        return s
