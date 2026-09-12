"""HC-SR04 style ultrasonic bumpers.

These cover the camera's blind spots - the sides and the rear, which the depth arc
never sees. Three physical facts shape the model:

*   A ping takes time. At a 4 m range the round trip plus settling is roughly 20-60 ms.
*   They cross-talk, so the ESP32 has to fire them one at a time. Four sensors
    round-robin therefore refresh at perhaps 15-25 Hz in total against a 30 Hz policy,
    meaning most readings the policy sees are stale. `update_hz` plus per-sensor hold
    models this, and getting it wrong teaches the policy to react to side obstacles
    faster than the hardware physically can.
*   The beam is a cone of roughly 15 degrees, not a line, and it reports the closest
    thing anywhere inside that cone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..dynamics.base import VehicleState
from ..world.geometry import World
from .noise import apply_gaussian

# Mounting angle relative to the car's heading, in degrees: front, left, right, back.
DEFAULT_ANGLES_DEG = (0.0, 90.0, -90.0, 180.0)
DEFAULT_NAMES = ("front", "left", "right", "back")


@dataclass
class UltrasonicParams:
    n_sensors: int = 4  # 4 = front/left/right/back; 3 drops the front
    cone_deg: float = 15.0
    rays_per_sensor: int = 3
    max_range: float = 4.00  # HC-SR04 spec ceiling
    min_range: float = 0.02
    noise_m: float = 0.015
    dropout_prob: float = 0.03  # soft or angled surfaces scatter the echo away
    update_hz: float = 20.0  # total across all sensors, not each
    mount_forward: float = 0.18
    mount_side: float = 0.10
    mount_back: float = -0.18
    angles_deg: tuple[float, ...] = field(default_factory=lambda: DEFAULT_ANGLES_DEG)


class UltrasonicArray:
    """Round-robin ring of ultrasonic sensors with per-sensor staleness."""

    def __init__(self, params: UltrasonicParams | None = None):
        self.p = params or UltrasonicParams()
        if self.p.n_sensors == 3:
            # Dropping the front is the documented 3-sensor layout: the camera already
            # covers forward, so left/right/back buy the most new information.
            keep = [1, 2, 3]
        else:
            keep = list(range(self.p.n_sensors))
        self._angles = np.radians([self.p.angles_deg[i] for i in keep])
        self._names = tuple(DEFAULT_NAMES[i] for i in keep)
        self.n = len(keep)
        self._values = np.full(self.n, self.p.max_range)
        self._next = 0
        self._clock = 0.0

        # Precomputed like the depth arc, and for the same reason: rebuilding ray fans
        # inside the step loop dominated the profile.
        p = self.p
        cone = math.radians(p.cone_deg)
        spread = (np.zeros(1) if p.rays_per_sensor == 1
                  else np.linspace(-cone / 2.0, cone / 2.0, p.rays_per_sensor))
        self._ray_offsets = self._angles[:, None] + spread[None, :]  # (n, rays)
        self._mounts = np.array([self._mount_for(a) for a in self._angles])  # (n, 2)
        self._scratch = np.empty((p.rays_per_sensor, 2))

    def _mount_for(self, angle: float) -> tuple[float, float]:
        p = self.p
        if abs(angle) < math.radians(45.0):
            return (p.mount_forward, 0.0)
        if abs(angle) > math.radians(135.0):
            return (p.mount_back, 0.0)
        return (0.0, math.copysign(p.mount_side, angle))

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def angles(self) -> np.ndarray:
        """Mounting angles relative to the car's heading, in radians."""
        return self._angles.copy()

    def mount(self, idx: int) -> tuple[float, float]:
        """Mounting point of sensor `idx` in the car's frame."""
        return self._mount(idx)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._values = np.full(self.n, self.p.max_range)
        self._next = 0
        self._clock = 0.0

    def _mount(self, idx: int) -> tuple[float, float]:
        return (float(self._mounts[idx, 0]), float(self._mounts[idx, 1]))

    def _ray_geometry(self, state: VehicleState, idx: int) -> tuple[np.ndarray, np.ndarray]:
        angles = state.theta + self._ray_offsets[idx]
        dirs = self._scratch
        np.cos(angles, out=dirs[:, 0])
        np.sin(angles, out=dirs[:, 1])
        ct, st = math.cos(state.theta), math.sin(state.theta)
        mx, my = self._mounts[idx]
        origin = np.array([state.x + mx * ct - my * st, state.y + mx * st + my * ct])
        return np.broadcast_to(origin, dirs.shape), dirs

    def _ping(self, world: World, state: VehicleState, idx: int,
              rng: np.random.Generator) -> float:
        p = self.p
        origins, dirs = self._ray_geometry(state, idx)
        hit = float(world.raycast(origins, dirs, p.max_range, normalize=False).min())
        if rng.random() < p.dropout_prob:
            return float(self._values[idx])  # echo lost, the ESP32 reports the last value
        hit = float(apply_gaussian(np.array([hit]), p.noise_m, rng)[0])
        return float(np.clip(hit, p.min_range, p.max_range))

    def sample(self, world: World, state: VehicleState, dt: float,
               rng: np.random.Generator) -> np.ndarray:
        """Advance the round-robin by `dt` and return every sensor's latest held value."""
        slot = 1.0 / max(self.p.update_hz, 1e-6)
        self._clock += dt
        # A slow policy step may fit several ping slots; a fast one may fit none.
        while self._clock >= slot:
            self._clock -= slot
            self._values[self._next] = self._ping(world, state, self._next, rng)
            self._next = (self._next + 1) % self.n
        return self._values.copy()

    def true_ranges(self, world: World, state: VehicleState) -> np.ndarray:
        """Noise-free, all sensors fresh. For the renderer and for tests."""
        p = self.p
        out = np.empty(self.n)
        for i in range(self.n):
            origins, dirs = self._ray_geometry(state, i)
            out[i] = world.raycast(origins, dirs, p.max_range, normalize=False).min()
        return np.clip(out, p.min_range, p.max_range)
