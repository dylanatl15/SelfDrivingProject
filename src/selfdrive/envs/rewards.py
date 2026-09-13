"""Reward terms for Phase 1: drive, do not crash, never stay stuck.

Every term is returned separately so TensorBoard shows which one the policy is actually
optimizing. Reward hacking is the normal failure mode of a hand-written reward, and it
is essentially undetectable from a single scalar curve.

Three deliberate departures from the obvious design. Each closes a hack that a policy
either found here or reliably finds elsewhere.

*   The drive signal pays for **new ground**, not speed and not displacement. A speed
    reward pays a car to drive in tight circles forever. The first version of this file
    paid net displacement over a 2 s window instead, which beats tight circles but not
    wide ones: a loop of 1-2 m radius at 1.3 m/s takes longer than 2 s to close, so the
    window still credited it with 74-93 % of straight-line progress, at no risk of
    touching a wall. `phase1_v1` duly learned to find an open patch and orbit it at full
    throttle.

    Now the car stamps a disk of `explore_radius` onto a fine raster of the floor every
    step, and is paid for the freshly stamped area divided by the disk's diameter: one
    unit per metre of ground it has not covered within `revisit_s`. A loop of any size
    pays for its first lap only, shuffling on the spot pays nothing, and a tight turn
    pays less than a straight because the swath overlaps itself on the inside.

    A coarse grid paying per cell entered looks equivalent and is not. It pays ~40 % more
    per metre for crossing the grid diagonally than for running along it, arena walls are
    axis-aligned and visible to the policy, and so it pays for zig-zagging down corridors
    - measured, it scored a slow weave above a straight line. A disk is rotation-invariant,
    so no heading earns more than another.

    The policy cannot see the raster and has no memory of it. What it can learn is the
    reactive habit that keeps finding new ground: hold a line, take the open side, do not
    retrace.

*   Driving over ground already covered can **cost**, per metre, through `w_retrace`.
    Paying only for new ground makes a second lap free rather than unprofitable, and
    `phase1_v2` settled for free. A 69° camera with no memory sees what is around it by
    turning, and one collision costs as much as a hundred metres of new ground, so from
    about 1M steps it drove at near-full steering lock for 62-76 % of its evaluation
    steps, circling open patches slowly. The charge is counted on the same raster pixels
    as the pay: a disk sliding over covered ground is charged exactly what it earned the
    first time across, and driving onto new ground is never charged.

*   **Lateral acceleration** costs, quadratically. The kinematic bicycle model has no
    tires and will corner a toy car at 0.4 g; the real car slides or tips over, so a
    policy that depends on fast full-lock turns does not transfer. The same term makes
    the slow side-to-side weave cost something - it is too gradual for the per-step
    steering-change penalty to register.

*   Reversing is a small **cost**, never a bonus. The tempting design - pay the agent for
    reversing when the front is blocked - trains it to drive at a wall, back up, and
    repeat forever, farming the bonus. Here backing out of a trap emerges because it is
    cheaper than the alternative: reversing costs `w_reverse` per step, staying wedged
    costs `w_stall`, twenty-five times as much.

*   With a goal (`obs.goal_block`), the drive signal can be **progress** instead:
    `w_progress` per metre of path distance closed toward the goal, plus `goal_bonus` on
    arrival. Path distance goes around walls (`world/navigation.py`), so no dead end facing
    the goal pays anything to sit in, and it is a fixed function of position, so no loop
    pays. The exploration terms stay computed for the coverage metric; set `w_explore` 0.

This runs once per env step in every training worker, so it sticks to scalar math and a
handful of numpy calls on flat arrays. Small-array `np.clip` and 2-D fancy indexing made
an earlier draft cost 83 us a step.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from ..dynamics.base import wrap_angle

# How many of the latest stamps still count as the disk's own footprint; see `_stamp`.
RECENT_STAMPS = 4  # 3 still charged 4 % of a diagonal line on new ground; 4 charges none


@dataclass
class RewardConfig:
    w_explore: float = 1.0  # per metre of fresh ground
    w_retrace: float = 0.0  # per metre of ground covered again within revisit_s; 0 = off
    w_progress: float = 0.0  # per metre of path distance closed toward the goal
    goal_bonus: float = 0.0  # per goal reached
    w_reverse: float = 0.02  # per step at full reverse
    w_oscillation: float = 0.05  # per unit change in the steering command
    w_throttle_oscillation: float = 0.0  # per unit change in the throttle command; 0 = off
    w_lateral: float = 0.05  # per step at lateral_accel_ref; quadratic
    w_proximity: float = 0.05  # per step at zero clearance
    w_stall: float = 0.50  # per step while wedged
    collision_penalty: float = 100.0

    explore_radius: float = 0.25  # metres; the swath the car claims as it drives
    explore_res: float = 0.05  # coverage raster pixel, metres
    revisit_s: float = 60.0  # longer than a 50 s training episode: ground pays once
    lateral_accel_ref: float = 3.0  # m/s^2, ~0.3 g
    safe_distance: float = 0.60  # clearance below which the proximity barrier starts
    stall_window: int = 45  # steps (~1.5 s) used for the stall check
    stall_eps: float = 0.08  # metres of net travel required within that window
    stall_limit: int = 90  # consecutive stalled steps before the episode is truncated


@dataclass
class RewardTerms:
    explore: float = 0.0
    retrace: float = 0.0
    progress: float = 0.0
    goal: float = 0.0
    reverse: float = 0.0
    oscillation: float = 0.0
    lateral: float = 0.0
    proximity: float = 0.0
    stall: float = 0.0
    collision: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.explore
            + self.retrace
            + self.progress
            + self.goal
            + self.reverse
            + self.oscillation
            + self.lateral
            + self.proximity
            + self.stall
            + self.collision
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class RewardFunction:
    """Stateful: exploration keeps a coverage raster, lateral acceleration needs the
    previous pose, and stall is judged over a window of past positions."""

    def __init__(self, config: RewardConfig | None = None):
        self.c = config or RewardConfig()
        self._history: deque[tuple[float, float]] = deque(maxlen=self.c.stall_window + 1)

        r = self.c.explore_radius / self.c.explore_res
        self._disk_n = n = math.ceil(r)
        gx, gy = np.mgrid[-n : n + 1, -n : n + 1]
        inside = gx**2 + gy**2 <= r * r
        self._disk_dx, self._disk_dy = gx[inside], gy[inside]
        # Sliding the stamp one pixel along an axis freshens one pixel per row, so this
        # converts fresh pixels to metres of swath exactly for axis moves and to within a
        # few percent at any other heading.
        self._metres_per_pixel = self.c.explore_res / (2 * n + 1)

        # The raster is stored flat: one gather and one scatter per step.
        self._seen = np.full(1, -np.inf, dtype=np.float32)
        self._disk_flat = self._disk_dx
        self._shape = (1, 1)
        self._origin = (0.0, 0.0)
        self._pixel: tuple[int, int] | None = None
        self._painted = 0
        self._fresh_px = 0  # swept onto fresh ground by steps, not counting the spawn stamp
        self._retraced_px = 0  # swept over ground covered within revisit_s
        self._recent: deque[np.float32] = deque(maxlen=RECENT_STAMPS)
        self._clock = 0.0
        self._heading = 0.0
        self.stalled_steps = 0

    def reset(self, x: float, y: float, theta: float, bounds) -> None:
        """`bounds` is the arena's `(x0, y0, x1, y1)`; the coverage raster is sized to it."""
        c = self.c
        margin = self._disk_n + 1  # pixels; a stamp centred anywhere in bounds fits
        x0, y0, x1, y1 = (float(b) for b in bounds)
        self._origin = (x0 - margin * c.explore_res, y0 - margin * c.explore_res)
        w = math.ceil((x1 - x0) / c.explore_res) + 2 * margin + 1
        h = math.ceil((y1 - y0) / c.explore_res) + 2 * margin + 1
        self._shape = (w, h)
        self._seen = np.full(w * h, -np.inf, dtype=np.float32)
        self._disk_flat = self._disk_dx * h + self._disk_dy
        self._pixel = None
        self._painted = 0
        self._fresh_px = 0
        self._retraced_px = 0
        self._recent.clear()
        self._clock = 0.0
        self._heading = float(theta)
        self._history.clear()
        self._history.append((float(x), float(y)))
        self.stalled_steps = 0
        self._stamp(x, y)  # the spawn footprint is not new ground

    def _stamp(self, x: float, y: float) -> tuple[int, int]:
        """Mark the disk at (x, y) as seen now. Return how many pixels the move swept onto
        fresh ground, and how many it swept over ground covered within `revisit_s`."""
        c = self.c
        n = self._disk_n
        w, h = self._shape
        # Clamped as scalars, which is far cheaper than clipping every disk index. It only
        # bites on the rare step where a collision puts the car centre past a wall.
        px = min(max(round((x - self._origin[0]) / c.explore_res), n), w - 1 - n)
        py = min(max(round((y - self._origin[1]) / c.explore_res), n), h - 1 - n)
        if (px, py) == self._pixel:
            # Every pixel under the disk was stamped a step or more ago. Their timestamps
            # go stale while parked, which could only matter after revisit_s of stalling.
            return 0, 0
        self._pixel = (px, py)

        idx = self._disk_flat + (px * h + py)
        last = self._seen[idx]
        never = int(np.count_nonzero(last == -np.inf))
        self._painted += never
        if self._clock < c.revisit_s:
            fresh = never  # nothing stamped this episode can have expired yet
        else:
            fresh = int(np.count_nonzero(last <= self._clock - c.revisit_s))
        # Swept by this move: every pixel under the disk that none of the latest stamps
        # covered. The previous stamp alone is not enough, because a disk moving diagonally
        # steps in a staircase and re-enters a few rim pixels of the stamp before it.
        # Whatever was swept and is not fresh is ground covered again.
        swept = int(np.count_nonzero(last < self._recent[0])) if self._recent else fresh
        now = np.float32(self._clock)
        self._seen[idx] = now
        self._recent.append(now)
        return fresh, max(swept - fresh, 0)

    @property
    def coverage_m2(self) -> float:
        """Floor area swept this episode. Unlike path length, circling cannot inflate it.

        Metres of swath times the true swath width, not pixel count times pixel area: a
        rasterized disk is 2n+1 pixels across, ~10 % wider than the disk it stands for."""
        return self._painted * self._metres_per_pixel * 2.0 * self.c.explore_radius

    @property
    def retrace_frac(self) -> float:
        """Share of the swath driven this episode that went over ground already covered:
        zero for a car that never crosses its own track, a half after two laps of a loop."""
        swept = self._fresh_px + self._retraced_px
        return self._retraced_px / swept if swept else 0.0

    def coverage_raster(self) -> tuple[np.ndarray, tuple[float, float], float]:
        """For the viewport; nothing in training reads it.

        Returns seconds since each pixel was last covered (inf where never) as a `(w, h)`
        array indexed `[x, y]`, the world position of pixel `[0, 0]`'s centre, and the
        pixel size. A pixel counts as covered while its age is below `revisit_s`.
        """
        return self._clock - self._seen.reshape(self._shape), self._origin, self.c.explore_res

    def _net_displacement(self, window: int) -> tuple[float, int]:
        """Straight-line distance covered over the last `window` steps, and how many
        steps were actually available (the window is short at the start of an episode)."""
        if len(self._history) < 2:
            return 0.0, 0
        k = min(window, len(self._history) - 1)
        (xa, ya), (xb, yb) = self._history[-1 - k], self._history[-1]
        return math.hypot(xb - xa, yb - ya), k

    @property
    def is_stalled(self) -> bool:
        return self.stalled_steps >= self.c.stall_limit

    def __call__(
        self,
        x: float,
        y: float,
        theta: float,
        throttle_cmd: float,
        prev_throttle_cmd: float,
        steer_cmd: float,
        prev_steer_cmd: float,
        clearance: float,
        collided: bool,
        dt: float,
        progress_m: float = 0.0,
        reached: bool = False,
    ) -> tuple[float, RewardTerms]:
        """`progress_m` and `reached` come from the environment's goal tracker, which owns
        the path-distance field; both stay 0 without a goal."""
        c = self.c
        x, y, theta = float(x), float(y), float(theta)
        x_prev, y_prev = self._history[-1]
        step_len = math.hypot(x - x_prev, y - y_prev)
        self._history.append((x, y))
        self._clock += dt
        t = RewardTerms()

        fresh, retraced = self._stamp(x, y)
        self._fresh_px += fresh
        self._retraced_px += retraced
        t.explore = c.w_explore * fresh * self._metres_per_pixel
        t.retrace = -c.w_retrace * retraced * self._metres_per_pixel
        t.progress = c.w_progress * float(progress_m)
        t.goal = c.goal_bonus * float(reached)

        # Reversing is allowed and sometimes necessary, but it is never free.
        t.reverse = -c.w_reverse * max(0.0, -float(throttle_cmd))

        # Jitter wrecks a real steering servo and looks terrible on a demo table. Throttle
        # jitter is the same habit on the other actuator: phase1_mem_v2 flipped the sign of
        # its throttle 4.5 times a second to hold ~0.7 m/s, which the sim's motor lag smooths
        # over and a real ESC may not.
        t.oscillation = (
            -c.w_oscillation * abs(float(steer_cmd) - float(prev_steer_cmd))
            - c.w_throttle_oscillation * abs(float(throttle_cmd) - float(prev_throttle_cmd))
        )

        # Lateral acceleration = speed x yaw rate, measured from consecutive poses rather
        # than read out of the vehicle model, so it survives swapping the model.
        yaw = abs(wrap_angle(theta - self._heading))
        self._heading = theta
        a_lat = step_len * yaw / (dt * dt)
        t.lateral = -c.w_lateral * (a_lat / c.lateral_accel_ref) ** 2

        # Smooth barrier on ground-truth clearance. Using the noisy sensor reading here
        # would teach the policy to chase its own sensor artifacts.
        if math.isfinite(clearance) and clearance < c.safe_distance:
            deficit = min(max(1.0 - clearance / c.safe_distance, 0.0), 1.0)
            t.proximity = -c.w_proximity * deficit**2

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
