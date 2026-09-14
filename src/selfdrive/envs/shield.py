"""A speed shield between the policy and the car (`shield.enabled`).

The crash probe on the Stage 2 models found that crashes were decisions, not blindness. The
car drove into obstacles its sensors had already reported: forward at about 1 m/s at full
lock, or reversing at about 0.5 m/s, and it almost never braked. The shield caps the throttle
so the car can still stop short of the nearest obstacle reported in the direction it is
driving:

    allowed speed = (reported distance - margin) / horizon, never below `floor` while the
    distance exceeds the margin, and 0 inside it.

If the car is faster than that by more than `slack`, the shield commands `brake` against the
motion instead.

It reads only what the phone has: the depth buckets and ultrasonic ranges the last
observation was built from, and the speed telemetry. Forward distance is the nearest depth
bucket or the front ultrasonic. Reverse distance is the back ultrasonic. Throttle is scaled
by the config's nominal top speeds, not this episode's randomized ones, because the nominal
values are all the phone knows. It is a controller, not a reward, so reading sensors is
right here.

The car, the reward, the observation's last-throttle input and the episode metrics all see
the throttle the shield sent, because that is the command the ESP32 receives. The phone
mirrors all of this exactly (`docs/shield.md`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PASSED, CAPPED, BRAKED, UNSTUCK = 0, 1, 2, 3


@dataclass
class ShieldConfig:
    enabled: bool = False
    margin: float = 0.15  # m of reported distance inside which the allowed speed is 0
    horizon: float = 0.5  # s: beyond the margin, the allowed speed closes the gap in this long
    floor: float = 0.0  # m/s allowed whatever the distance, while it exceeds the margin
    brake: float = 0.3  # throttle commanded against the motion when the car is too fast
    slack: float = 0.05  # m/s over the allowed speed before the shield brakes
    # Backing out of a spot the cap holds the car in (`Unstick`). 0 s leaves it off.
    unstick_after: float = 0.0  # s of asking forward while held before backing out
    unstick_for: float = 1.0  # s of reversing, cut short at the back margin
    unstick_throttle: float = 0.5  # reverse throttle while backing out, capped as usual
    unstick_speed: float = 0.1  # m/s: slower than this while capped counts as held


def allowed_speed(distance: float, s: ShieldConfig) -> float:
    """The fastest the car may drive toward an obstacle reported `distance` metres away."""
    if distance <= s.margin:
        return 0.0
    return max(s.floor, (distance - s.margin) / s.horizon)


def reported_distances(depth: np.ndarray, ultra: np.ndarray,
                       names: Sequence[str]) -> tuple[float, float]:
    """Forward and reverse distances, in metres. A build without the sensor has no reading
    in that direction: no front ultrasonic leaves depth alone, and no back one leaves
    reversing uncapped."""
    forward = float(np.min(depth))
    if "front" in names:
        forward = min(forward, float(ultra[list(names).index("front")]))
    back = float(ultra[list(names).index("back")]) if "back" in names else math.inf
    return forward, back


def shield_throttle(throttle: float, speed: float, forward_m: float, back_m: float,
                    s: ShieldConfig, max_speed_fwd: float,
                    max_speed_rev: float) -> tuple[float, int]:
    """Return the throttle to send and what the shield did: PASSED, CAPPED or BRAKED."""
    if throttle > 0.0:
        allowed = allowed_speed(forward_m, s)
        if speed > allowed + s.slack:
            return -s.brake, BRAKED
        out = min(throttle, allowed / max_speed_fwd)
    elif throttle < 0.0:
        allowed = allowed_speed(back_m, s)
        if -speed > allowed + s.slack:
            return s.brake, BRAKED
        out = max(throttle, -allowed / max_speed_rev)
    else:
        return throttle, PASSED
    return out, (CAPPED if abs(out - throttle) > 1e-6 else PASSED)


class Unstick:
    """The shield, plus backing out of a spot its cap holds the car in (`unstick_after` > 0).

    A model trained without the shield never learned that pushing into its cap gets nowhere.
    With the shield bolted on, the 9.5M patience model ended 17 % of fresh exam episodes stuck.
    In their last 5 s it sat nose-in 0.17 m from an obstacle at full lock, asking for forward
    throttle the shield cut to a crawl, with clear floor behind it that it never used.

    After `unstick_after` seconds of asking forward while capped below `unstick_speed`, the car
    reverses at `unstick_throttle` for `unstick_for` seconds on the mirrored steering. Reversing
    on the opposite lock turns the heading the way the policy was steering, so the nose swings
    off the obstacle, as in a three-point turn. The reverse passes through the same cap as any
    other and stops when the back reading reaches the margin. It only starts with room behind.
    With `unstick_after` 0 this is exactly `shield_throttle`, steering untouched.
    """

    def __init__(self, s: ShieldConfig, dt: float):
        self.s = s
        self.after = max(1, round(s.unstick_after / dt))
        self.length = max(1, round(s.unstick_for / dt))
        self.held = 0  # consecutive steps held against the cap
        self.left = 0  # steps of backing out still to go
        self.steer = 0.0  # the mirrored steering held while backing out

    def step(self, steer: float, throttle: float, speed: float, forward_m: float, back_m: float,
             max_speed_fwd: float, max_speed_rev: float) -> tuple[float, float, int]:
        """Return the steering and throttle to send, and what the shield did."""
        s = self.s
        if self.left > 0:
            self.left -= 1
            out, _ = shield_throttle(-s.unstick_throttle, speed, forward_m, back_m, s,
                                     max_speed_fwd, max_speed_rev)
            if out == 0.0:
                self.left = 0  # nothing more behind
            return self.steer, out, UNSTUCK
        out, did = shield_throttle(throttle, speed, forward_m, back_m, s, max_speed_fwd,
                                   max_speed_rev)
        if s.unstick_after > 0.0:
            held = did == CAPPED and throttle > 0.0 and abs(speed) < s.unstick_speed
            self.held = self.held + 1 if held else 0
            if self.held >= self.after and allowed_speed(back_m, s) > 0.0:
                self.held, self.left, self.steer = 0, self.length, -steer
        return steer, out, did
