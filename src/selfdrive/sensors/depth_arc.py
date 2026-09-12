"""Camera depth compressed into a 1D virtual laser sweep.

The phone runs ARCore and produces a depth map. Feeding that raw into an MLP is both
wasteful and fragile, so it is collapsed to one closest-distance value per angular
bucket across the camera's horizontal field of view.

Two hardware facts drive this model, and both are easy to get wrong:

1.  ARCore does not support ultra-wide cameras - it binds the main lens. The S21 FE's
    main camera is 26 mm equivalent, so the usable arc is about 69 degrees held in
    landscape and only about 50 degrees in portrait. Simulating a wider arc than the
    phone can see manufactures a blind spot that exists only on the real car.

2.  The S21 FE has no time-of-flight sensor, so ARCore's depth is motion stereo. It
    needs the camera to translate. When the car is stopped or turning in place the
    depth map degrades badly - which is exactly the situation the unstick behaviour
    has to handle. `stationary_dropout` models that, and without it the policy learns
    to reverse using depth it will not actually have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dynamics.base import VehicleState
from ..world.geometry import World
from .noise import LatencyBuffer, apply_dropout, apply_relative_gaussian


@dataclass
class DepthArcParams:
    fov_deg: float = 69.0  # S21 FE main lens, landscape
    n_buckets: int = 8
    rays_per_bucket: int = 5  # one ray per bucket would miss thin obstacles
    max_range: float = 5.0
    min_range: float = 0.15  # ARCore near plane
    mount_forward: float = 0.18  # phone sits ahead of the car's centre, metres
    noise_frac: float = 0.02  # stereo error scales with distance
    dropout_prob: float = 0.02  # baseline invalid-pixel rate while moving well
    stationary_dropout: float = 0.60  # extra dropout as the car stops translating
    speed_ref: float = 0.30  # speed at which motion stereo is considered healthy
    latency_steps: int = 1


class DepthArc:
    """Virtual laser sweep over the camera's field of view."""

    def __init__(self, params: DepthArcParams | None = None):
        self.p = params or DepthArcParams()
        self._held = np.full(self.p.n_buckets, self.p.max_range)
        self._buffer = LatencyBuffer(self.p.latency_steps, (self.p.n_buckets,), self.p.max_range)
        self.confidence = 1.0

        # Ray angles are fixed for the life of this sensor (a new DepthArc is built each
        # episode when randomization redraws the FOV), so they are computed once. Doing
        # it per step cost 45% of total runtime: eight Python-level bucket iterations,
        # each allocating through linspace and a rotation matrix, ~80 tiny numpy calls a
        # step where five will do.
        p = self.p
        fov = math.radians(p.fov_deg)
        bucket = fov / p.n_buckets
        centres = -fov / 2.0 + bucket * (np.arange(p.n_buckets) + 0.5)
        spread = (np.zeros(1) if p.rays_per_bucket == 1
                  else np.linspace(-bucket / 2.0, bucket / 2.0, p.rays_per_bucket))
        self._ray_offsets = (centres[:, None] + spread[None, :]).ravel()
        self._n_rays = self._ray_offsets.size
        self._dirs = np.empty((self._n_rays, 2))
        self._origins = np.empty((self._n_rays, 2))

    def reset(self, rng: np.random.Generator | None = None) -> None:
        p = self.p
        self._held = np.full(p.n_buckets, p.max_range)
        self._buffer = LatencyBuffer(p.latency_steps, (p.n_buckets,), p.max_range)
        self.confidence = 1.0

    def true_ranges(self, world: World, state: VehicleState) -> np.ndarray:
        """Noise-free closest distance per bucket. Used by the renderer and by tests."""
        p = self.p
        angles = state.theta + self._ray_offsets
        np.cos(angles, out=self._dirs[:, 0])
        np.sin(angles, out=self._dirs[:, 1])

        # The camera sits ahead of the car's centre, so every ray starts from there.
        ct, st = math.cos(state.theta), math.sin(state.theta)
        self._origins[:, 0] = state.x + p.mount_forward * ct
        self._origins[:, 1] = state.y + p.mount_forward * st

        hits = world.raycast(self._origins, self._dirs, p.max_range, normalize=False)
        # Closest hit anywhere in the slice - the safety-relevant reading.
        return np.clip(hits.reshape(p.n_buckets, p.rays_per_bucket).min(axis=1),
                       p.min_range, p.max_range)

    def sample(self, world: World, state: VehicleState, dt: float,
               rng: np.random.Generator) -> np.ndarray:
        p = self.p
        ranges = self.true_ranges(world, state)

        # Motion stereo needs translation; confidence collapses as the car stops.
        self.confidence = float(np.clip(abs(state.speed) / max(p.speed_ref, 1e-6), 0.0, 1.0))
        drop_prob = p.dropout_prob + (1.0 - self.confidence) * p.stationary_dropout

        ranges = apply_relative_gaussian(ranges, p.noise_frac, rng)
        ranges, _ = apply_dropout(ranges, self._held, drop_prob, rng)
        ranges = np.clip(ranges, p.min_range, p.max_range)

        self._held = ranges.copy()
        return self._buffer.push_pop(ranges)
