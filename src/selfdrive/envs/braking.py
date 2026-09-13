"""Whether the car can stop before it hits something, on the arc it is driving now.

The proximity barrier charges distance to the nearest obstacle whatever the speed, so it
cannot tell parking beside a wall from arriving at one. waypoint_pay5 crashed at 0.75 to
0.97 m/s on average, much of it while turning or backing up, yet outran its straight-ahead
stopping distance on only about 1 % of forward steps: the danger was the arc, the sides and
the rear, not the ray straight ahead. This measures what the barrier misses. It takes the
distance the car needs to stop from its current speed, lays that distance along its current
arc, forwards or reversing, and reports how much of it the body cannot drive.

Every input is ground truth, like every other reward input: the true pose, speed and wheel
angle, this episode's drawn actuator lag and sensor latency, and the true walls. It assumes
the steering holds while the car brakes. Steering away can still save a car this calls
blocked, so the charge is conservative, like the barrier it complements.
"""

from __future__ import annotations

import math

from ..dynamics.base import CarParams
from ..world.geometry import World


def stopping_distance(speed: float, p: CarParams, reaction_s: float) -> float:
    """Metres from `speed` to rest: `reaction_s` at speed, then full opposite throttle
    through the drivetrain's first-order lag (`dynamics/actuators.py`)."""
    v = abs(speed)
    if v == 0.0:
        return 0.0
    reverse_limit = p.max_speed_rev if speed > 0 else p.max_speed_fwd
    v_brake = max(min(p.throttle_gain, 1.0) * reverse_limit, 1e-6)
    tau = max(p.accel_tau, 1e-3)
    # Speed runs as v(t) = -v_brake + (v + v_brake) exp(-t / tau). It reaches zero after
    # tau ln(1 + v / v_brake), having covered tau v - tau v_brake ln(1 + v / v_brake).
    return v * reaction_s + tau * (v - v_brake * math.log1p(v / v_brake))


def arc_pose(x: float, y: float, theta: float, curvature: float,
             s: float) -> tuple[float, float, float]:
    """The pose `s` signed metres along an arc of `curvature` (1/m, left-positive)."""
    if abs(curvature * s) < 1e-9:
        return x + s * math.cos(theta), y + s * math.sin(theta), theta
    end = theta + curvature * s
    return (x + (math.sin(end) - math.sin(theta)) / curvature,
            y - (math.cos(end) - math.cos(theta)) / curvature, end)


def brake_shortfall(world: World, x: float, y: float, theta: float, speed: float,
                    steer: float, p: CarParams, reaction_s: float, margin: float,
                    samples: int, clearance: float) -> float:
    """The share of the stopping path, plus `margin`, that the body cannot drive.

    0 when the car can stop with `margin` to spare, 1 when the first stretch is already
    blocked. Poses are checked at `samples` even steps along the arc, nearest first, so
    the answer comes in steps of 1 / `samples`.

    `clearance` is the body's distance to the nearest obstacle now. Along an arc of length
    `s` no point of the body moves more than `s` plus the turn times the half-diagonal, so a
    clearance beyond that proves the whole path clear and nothing is checked. Most steps
    take that exit.
    """
    if abs(speed) < 1e-3:
        return 0.0
    reach = stopping_distance(speed, p, reaction_s) + margin
    curvature = math.tan(steer / (1.0 + p.understeer * speed * speed)) / p.wheelbase
    if clearance > reach * (1.0 + abs(curvature) * 0.5 * math.hypot(p.length, p.width)):
        return 0.0
    direction = 1.0 if speed > 0 else -1.0
    for k in range(1, samples + 1):
        px, py, pt = arc_pose(x, y, theta, curvature, direction * reach * k / samples)
        if world.collides(px, py, pt, p.length, p.width):
            return 1.0 - (k - 1) / samples
    return 0.0
