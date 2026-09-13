"""Observation assembly.

One flat `Box` built from fixed-size blocks, frame-stacked. The block layout is part of
the contract with the Android app, so it is spelled out here and must stay in sync with
`docs/protocol.md`.

Per frame (17 floats, all in [-1, 1]):

    [0 : 8]   depth arc buckets, RIGHT to left across the camera's field of view:
              bucket 0 is the most clockwise, since angles are left-positive
    [8]       depth confidence - collapses as the car stops, because ARCore on a
              phone with no ToF sensor computes depth from motion
    [9 : 13]  ultrasonics: front, left, right, back (3-sensor builds drop front)
    [13]      signed speed
    [14]      actual steering angle
    [15]      last throttle command
    [16]      last steering command

With `memory_sectors` S > 0, one obstacle-memory block of 2S floats follows the stacked
frames. It is not stacked itself, since it already spans `memory_seconds`:

    [0 : S]   distance to the nearest remembered obstacle in each sector
    [S : 2S]  age of that obstacle

Layout and rules are in `envs/memory.py`. S defaults to 0, which keeps the 68-float input.

With `goal_block` on, three floats follow everything else. They are not stacked either:

    [0]       straight-line range to the goal, over `norm_goal_max`
    [1]       sine of the bearing to the goal, left-positive like steering
    [2]       cosine of that bearing

    [3]       with `goal_patience`: seconds since that range last fell to a new low,
              over `norm_patience_max`

The bearing is split into sine and cosine because a single angle jumps from +pi to -pi as
the goal passes behind the car. All of it comes from the pose estimate, not the true pose;
the phone's rules are in `docs/goal-block.md`, and the clock's in `envs/goals.py`.

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

    # Obstacle memory, appended once after the stack. 0 disables it; 24 gives 15-degree
    # sectors. Part of the same published interface as the constants above.
    memory_sectors: int = 0
    memory_seconds: float = 3.0
    norm_memory_max: float = 5.00  # metres

    # Range and bearing to the current goal, appended last. Same published interface.
    goal_block: bool = False
    norm_goal_max: float = 15.00  # metres
    # A fourth goal float: how long the car has gone without getting closer (GoalPatience).
    goal_patience: bool = False
    norm_patience_max: float = 10.0  # seconds
    patience_gain: float = 0.25  # metres the range must fall below its best to restart the clock

    @property
    def per_frame(self) -> int:
        return self.n_depth + 1 + self.n_ultrasonic + 4

    @property
    def goal_size(self) -> int:
        if not self.goal_block:
            return 0
        return 4 if self.goal_patience else 3

    @property
    def size(self) -> int:
        return self.per_frame * self.frame_stack + 2 * self.memory_sectors + self.goal_size


def _unit_to_pm1(values: np.ndarray, scale: float) -> np.ndarray:
    """Map a non-negative distance in [0, scale] onto [-1, 1]."""
    return np.clip(values / scale, 0.0, 1.0) * 2.0 - 1.0


class ObservationBuilder:
    """Turns raw sensor readings into the stacked policy input."""

    def __init__(self, config: ObsConfig | None = None):
        self.c = config or ObsConfig()
        self._stack: deque[np.ndarray] = deque(maxlen=self.c.frame_stack)
        self._ring: np.ndarray | None = None
        self._goal: np.ndarray | None = None

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

    def goal(self, range_m: float, bearing: float, waited_s: float | None = None) -> np.ndarray:
        """The goal block for a range in metres, a left-positive bearing in radians and, with
        `goal_patience`, the seconds waited for a new low range. An infinite range (no goal)
        reads as the far end of the scale, dead ahead."""
        block = [_unit_to_pm1(np.asarray(range_m, dtype=float), self.c.norm_goal_max),
                 np.sin(bearing), np.cos(bearing)]
        if self.c.goal_patience:
            if waited_s is None:
                raise ValueError("goal_patience needs the seconds waited")
            block.append(_unit_to_pm1(np.asarray(waited_s, dtype=float), self.c.norm_patience_max))
        return np.array(block, dtype=np.float32)

    def reset(self, first_frame: np.ndarray, ring: np.ndarray | None = None,
              goal: np.ndarray | None = None) -> np.ndarray:
        """Prime the stack by repeating the first frame.

        Zero-filling would present a fabricated history - an apparent jump from
        mid-range readings to whatever is really there - on the first few steps of
        every episode.
        """
        self._stack.clear()
        for _ in range(self.c.frame_stack):
            self._stack.append(first_frame.copy())
        self._set_ring(ring)
        self._set_goal(goal)
        return self.stacked()

    def push(self, frame: np.ndarray, ring: np.ndarray | None = None,
             goal: np.ndarray | None = None) -> np.ndarray:
        self._stack.append(frame.copy())
        self._set_ring(ring)
        self._set_goal(goal)
        return self.stacked()

    def _set_goal(self, goal: np.ndarray | None) -> None:
        if (0 if goal is None else len(goal)) != self.c.goal_size:
            raise ValueError(f"goal block must have {self.c.goal_size} floats, got "
                             f"{None if goal is None else len(goal)}")
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float32)

    def _set_ring(self, ring: np.ndarray | None) -> None:
        expected = 2 * self.c.memory_sectors
        if (0 if ring is None else len(ring)) != expected:
            raise ValueError(f"memory ring must have {expected} floats, got "
                             f"{None if ring is None else len(ring)}")
        self._ring = None if ring is None else np.asarray(ring, dtype=np.float32)

    def stacked(self) -> np.ndarray:
        """Oldest frame first, newest last, then the memory ring and the goal block."""
        parts = list(self._stack)
        if self._ring is not None:
            parts.append(self._ring)
        if self._goal is not None:
            parts.append(self._goal)
        return np.concatenate(parts).astype(np.float32)

    def latest_ring(self) -> np.ndarray | None:
        """The memory block the policy last received, or None without memory."""
        return None if self._ring is None else self._ring.copy()

    def latest_frame(self) -> np.ndarray | None:
        """The most recent per-frame block, or None before the first reset.

        The renderer uses this to draw what the policy actually received alongside the
        true sensor sweep, which is how dropout and latency are made visible.
        """
        return self._stack[-1].copy() if self._stack else None

    def depth_slice(self, frame: np.ndarray) -> np.ndarray:
        return frame[: self.c.n_depth]
