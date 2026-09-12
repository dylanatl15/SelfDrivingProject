"""Observation assembly.

One flat `Box` built from fixed-size blocks, frame-stacked. The block layout is part of
the contract with the Android app, so it is spelled out here and must stay in sync with
`docs/protocol.md`.

Per frame (17 floats, all in [-1, 1]):

    [0 : 8]   depth arc buckets, left to right across the camera's field of view
    [8]       depth confidence - collapses as the car stops, because ARCore on a
              phone with no ToF sensor computes depth from motion
    [9 : 13]  ultrasonics: front, left, right, back (3-sensor builds drop front)
    [13]      signed speed
    [14]      actual steering angle
    [15]      last throttle command
    [16]      last steering command

    Phase 2 appends a two-float goal block (range, bearing to the GPS pin) here.

Two rules that exist for deployment rather than for training:

*   The normalization constants are **fixed**, never the per-episode randomized values.
    If an episode randomizes the depth range to 8 m and the observation divides by 8,
    the policy sees a different scale every episode and nothing transfers. The car also
    has no idea what its randomized parameters are at run time.
*   Normalization is analytic, so `VecNormalize` is not used on observations. That keeps
    the exported ONNX model self-contained - there is no running-statistics file that
    could drift out of sync with the phone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ObsConfig:
    n_depth: int = 8
    n_ultrasonic: int = 4
    frame_stack: int = 4

    # Fixed normalization references. These are a published interface, not tunables:
    # changing one invalidates every trained checkpoint and must be mirrored on the phone.
    norm_depth_max: float = 5.00  # metres
    norm_ultra_max: float = 4.00  # metres, HC-SR04 ceiling
    norm_speed_max: float = 1.50  # metres per second
    norm_steer_max: float = 0.4887  # radians, 28 degrees

    @property
    def per_frame(self) -> int:
        return self.n_depth + 1 + self.n_ultrasonic + 4

    @property
    def size(self) -> int:
        return self.per_frame * self.frame_stack


def _unit_to_pm1(values: np.ndarray, scale: float) -> np.ndarray:
    """Map a non-negative distance in [0, scale] onto [-1, 1]."""
    return np.clip(values / scale, 0.0, 1.0) * 2.0 - 1.0


class ObservationBuilder:
    """Turns raw sensor readings into the stacked policy input."""

    def __init__(self, config: ObsConfig | None = None):
        self.c = config or ObsConfig()
        self._stack: deque[np.ndarray] = deque(maxlen=self.c.frame_stack)

    @property
    def size(self) -> int:
        return self.c.size

    def frame(
        self,
        depth: np.ndarray,
        depth_confidence: float,
        ultrasonic: np.ndarray,
        speed: float,
        steer: float,
        last_throttle: float,
        last_steer: float,
    ) -> np.ndarray:
        c = self.c
        out = np.empty(c.per_frame, dtype=np.float32)
        i = 0

        out[i : i + c.n_depth] = _unit_to_pm1(np.asarray(depth, dtype=float), c.norm_depth_max)
        i += c.n_depth

        out[i] = np.clip(depth_confidence, 0.0, 1.0) * 2.0 - 1.0
        i += 1

        out[i : i + c.n_ultrasonic] = _unit_to_pm1(
            np.asarray(ultrasonic, dtype=float), c.norm_ultra_max
        )
        i += c.n_ultrasonic

        # Ego state. Speed is signed, so reverse reads negative and the policy can tell
        # which way it is going - a single depth frame cannot.
        out[i] = np.clip(speed / c.norm_speed_max, -1.0, 1.0)
        out[i + 1] = np.clip(steer / c.norm_steer_max, -1.0, 1.0)
        out[i + 2] = np.clip(last_throttle, -1.0, 1.0)
        out[i + 3] = np.clip(last_steer, -1.0, 1.0)
        return out

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        """Prime the stack by repeating the first frame.

        Zero-filling would present a fabricated history - an apparent jump from
        mid-range readings to whatever is really there - on the first few steps of
        every episode.
        """
        self._stack.clear()
        for _ in range(self.c.frame_stack):
            self._stack.append(first_frame.copy())
        return self.stacked()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._stack.append(frame.copy())
        return self.stacked()

    def stacked(self) -> np.ndarray:
        """Oldest frame first, newest last."""
        return np.concatenate(list(self._stack)).astype(np.float32)

    def latest_frame(self) -> np.ndarray | None:
        """The most recent per-frame block, or None before the first reset.

        The renderer uses this to draw what the policy actually received alongside the
        true sensor sweep, which is how dropout and latency are made visible.
        """
        return self._stack[-1].copy() if self._stack else None

    def depth_slice(self, frame: np.ndarray) -> np.ndarray:
        return frame[: self.c.n_depth]
