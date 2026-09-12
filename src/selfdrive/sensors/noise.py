"""Sensor corruption primitives shared by the depth arc and the ultrasonics.

None of this is decoration. A policy trained on clean, instantaneous, always-valid
sensors learns to trust readings that the real car will not have, and the failure mode
on hardware is confident driving into things the model believes are not there.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class LatencyBuffer:
    """Delays readings by a whole number of control steps.

    Real perception is never current: ARCore has to run depth-from-motion, the result
    crosses a USB serial link, and the policy acts on it a frame or more later.
    """

    def __init__(self, steps: int, shape: tuple[int, ...], fill: float):
        self.steps = max(int(steps), 0)
        self._q: deque[np.ndarray] = deque(
            (np.full(shape, float(fill)) for _ in range(self.steps + 1)),
            maxlen=self.steps + 1,
        )

    def push_pop(self, value: np.ndarray) -> np.ndarray:
        """Record the current reading and return the one from `steps` steps ago."""
        self._q.append(np.asarray(value, dtype=float).copy())
        return self._q[0].copy()


def apply_gaussian(values: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive noise with a fixed standard deviation, in metres."""
    if sigma <= 0.0:
        return values
    return values + rng.normal(0.0, sigma, size=values.shape)


def apply_relative_gaussian(
    values: np.ndarray, sigma_frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative noise. Stereo depth error grows with distance, so this fits it."""
    if sigma_frac <= 0.0:
        return values
    return values * (1.0 + rng.normal(0.0, sigma_frac, size=values.shape))


def apply_dropout(
    values: np.ndarray,
    held: np.ndarray,
    prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace dropped readings with the last good value.

    Holding the stale value rather than substituting max range is the honest model:
    that is what both ARCore (invalid depth pixels) and an ultrasonic round-robin
    (a sensor that has not had its turn yet) actually hand back.

    Returns the corrupted values and the boolean mask of which entries were dropped.
    """
    if prob <= 0.0:
        return values, np.zeros(values.shape, dtype=bool)
    mask = rng.random(values.shape) < prob
    return np.where(mask, held, values), mask
