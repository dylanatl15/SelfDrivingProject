"""Reward terms for Phase 1: drive, do not crash, never stay stuck.

Every term is returned separately so TensorBoard shows which one the policy is actually
optimizing. Reward hacking is the normal failure mode of a hand-written reward, and it
is essentially undetectable from a single scalar curve.

Two deliberate departures from the obvious design:

*   Progress is scored as **net displacement over a sliding window**, not instantaneous
    forward speed. A speed reward pays a car that drives in tight circles forever, which
    satisfies "never crashes" and is useless as a demo. Over a 2 s window straight
    driving covers ~3 m while the tightest possible circle covers at most one diameter,
    about 0.9 m, so circling earns roughly a third as much. The same term also kills
    shuffling back and forth on the spot, which nets out to zero.

*   Reversing is a small **cost**, never a bonus. The tempting design - pay the agent for
    reversing when the front is blocked - trains it to drive at a wall, back up, and
    repeat forever, farming the bonus. Here the arithmetic simply makes reversing the
    cheaper option when stuck: at full reverse, progress roughly cancels the reverse cost
    for a net near zero, while staying wedged costs `w_stall` every step. Backing out of
    a trap emerges because it beats the alternative, not because it is paid for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RewardConfig:
    w_progress: float = 1.0  # per metre of net travel
    w_reverse: float = 0.02  # per step at full reverse
    w_oscillation: float = 0.05  # per unit change in the steering command
    w_proximity: float = 0.05  # per step at zero clearance
    w_stall: float = 0.50  # per step while wedged
    collision_penalty: float = 100.0

    safe_distance: float = 0.60  # clearance below which the proximity barrier starts
    progress_window: int = 60  # steps (~2 s at 30 Hz) used for net displacement
    stall_window: int = 45  # steps (~1.5 s) used for the stall check
    stall_eps: float = 0.08  # metres of net travel required within that window
    stall_limit: int = 90  # consecutive stalled steps before the episode is truncated


@dataclass
class RewardTerms:
    progress: float = 0.0
    reverse: float = 0.0
    oscillation: float = 0.0
    proximity: float = 0.0
    stall: float = 0.0
    collision: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.reverse
            + self.oscillation
            + self.proximity
            + self.stall
            + self.collision
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class RewardFunction:
    """Stateful because progress and stall are both judged over a window of past poses."""

    def __init__(self, config: RewardConfig | None = None):
        self.c = config or RewardConfig()
        maxlen = max(self.c.progress_window, self.c.stall_window) + 1
        self._history: deque[np.ndarray] = deque(maxlen=maxlen)
        self.stalled_steps = 0

    def reset(self, x: float, y: float) -> None:
        self._history.clear()
        self._history.append(np.array([x, y], dtype=float))
        self.stalled_steps = 0

    def _net_displacement(self, window: int) -> tuple[float, int]:
        """Straight-line distance covered over the last `window` steps, and how many
        steps were actually available (the window is short at the start of an episode)."""
        if len(self._history) < 2:
            return 0.0, 0
        k = min(window, len(self._history) - 1)
        return float(np.linalg.norm(self._history[-1] - self._history[-1 - k])), k

    @property
    def is_stalled(self) -> bool:
        return self.stalled_steps >= self.c.stall_limit

    def __call__(
        self,
        x: float,
        y: float,
        throttle_cmd: float,
        steer_cmd: float,
        prev_steer_cmd: float,
        clearance: float,
        collided: bool,
        dt: float,
    ) -> tuple[float, RewardTerms]:
        c = self.c
        self._history.append(np.array([x, y], dtype=float))
        t = RewardTerms()

        # Progress: average speed actually achieved across the window, in metres.
        net, k = self._net_displacement(c.progress_window)
        if k > 0:
            t.progress = c.w_progress * (net / (k * dt)) * dt

        # Reversing is allowed and sometimes necessary, but it is never free.
        t.reverse = -c.w_reverse * max(0.0, -float(throttle_cmd))

        # Jitter wrecks a real steering servo and looks terrible on a demo table.
        t.oscillation = -c.w_oscillation * abs(float(steer_cmd) - float(prev_steer_cmd))

        # Smooth barrier on ground-truth clearance. Using the noisy sensor reading here
        # would teach the policy to chase its own sensor artifacts.
        if np.isfinite(clearance) and clearance < c.safe_distance:
            deficit = np.clip(1.0 - clearance / c.safe_distance, 0.0, 1.0)
            t.proximity = -c.w_proximity * float(deficit) ** 2

        # Stall: judged on displacement, never on `speed == 0`, which a float never hits
        # and which would miss a wedged car whose wheels are still turning.
        stall_net, stall_k = self._net_displacement(c.stall_window)
        if stall_k >= c.stall_window and stall_net < c.stall_eps:
            self.stalled_steps += 1
            t.stall = -c.w_stall
        else:
            self.stalled_steps = 0

        if collided:
            t.collision = -c.collision_penalty

        return t.total, t
