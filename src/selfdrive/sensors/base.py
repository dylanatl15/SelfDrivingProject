"""Shared sensor helpers: mounting points and ray fans in world coordinates."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..dynamics.base import VehicleState
from ..world.geometry import World


def body_to_world(state: VehicleState, local: np.ndarray) -> np.ndarray:
    """Map points from the car's frame (x forward, y left) into world coordinates."""
    c, s = np.cos(state.theta), np.sin(state.theta)
    r = np.array([[c, -s], [s, c]])
    return np.asarray(local, dtype=float).reshape(-1, 2) @ r.T + np.array([state.x, state.y])


def fan(
    state: VehicleState,
    mount_local: tuple[float, float],
    centre_angle: float,
    spread: float,
    n_rays: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `n_rays` rays from one mounting point, evenly spread about `centre_angle`.

    Angles are relative to the car's heading. Returns (origins, directions), both (n, 2).
    """
    if n_rays == 1:
        offsets = np.zeros(1)
    else:
        offsets = np.linspace(-spread / 2.0, spread / 2.0, n_rays)
    angles = state.theta + centre_angle + offsets
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    origin = body_to_world(state, np.array([mount_local]))[0]
    return np.repeat(origin[None, :], n_rays, axis=0), dirs


class Sensor(Protocol):
    """Anything the observation builder can poll once per control step."""

    def reset(self, rng: np.random.Generator) -> None: ...

    def sample(self, world: World, state: VehicleState, dt: float) -> np.ndarray: ...
