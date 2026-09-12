"""Domain randomization.

Resampled on every `reset()` so the policy learns a band of plausible cars rather than
one exact mathematical sandbox. The axes here were chosen for whether they actually move
the sim-to-real needle. Camera height, for instance, is a popular thing to randomize and
is a complete no-op against a 1D depth-bucket observation in a 2D simulator - it is not
here. Actuator lag, sensor latency and speed-dependent depth dropout are, because each
one changes what the policy is able to react to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Range = tuple[float, float]


def _u(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


@dataclass
class DomainRandConfig:
    enabled: bool = True

    # --- chassis and actuators ---
    wheelbase_scale: Range = (0.90, 1.10)
    accel_tau: Range = (0.10, 0.50)
    steer_rate_deg_s: Range = (90.0, 360.0)
    steer_trim_deg: Range = (-3.0, 3.0)
    max_steer_deg: Range = (22.0, 32.0)
    throttle_deadband: Range = (0.00, 0.08)
    throttle_gain: Range = (0.80, 1.10)  # open-loop command error; ~1.0 with encoders
    understeer: Range = (0.00, 0.15)
    max_speed_fwd: Range = (1.00, 1.80)

    # --- camera depth ---
    # 55 covers a phone mounted in portrait (~50 deg on the S21 FE main lens), 69 is
    # landscape, and the top of the band leaves room for a wider custom depth pipeline.
    depth_fov_deg: Range = (55.0, 125.0)
    depth_max_range: Range = (3.00, 8.00)
    depth_noise_frac: Range = (0.01, 0.06)
    depth_dropout: Range = (0.00, 0.08)
    depth_stationary_dropout: Range = (0.30, 0.80)
    depth_latency_steps: Range = (1, 4)

    # --- ultrasonics ---
    ultra_noise_m: Range = (0.010, 0.030)
    ultra_dropout: Range = (0.00, 0.10)
    ultra_update_hz: Range = (10.0, 30.0)

    # --- control loop ---
    dt: Range = (0.025, 0.040)  # 25-40 ms, i.e. 25-40 Hz with jitter

    # --- visual-inertial odometry, read only when the observation has a memory ring ---
    odom_scale_error: Range = (-0.03, 0.03)
    odom_pos_noise: Range = (0.01, 0.08)  # m per sqrt(m)
    odom_yaw_noise: Range = (0.005, 0.05)  # rad per sqrt(rad)
    odom_tracking_loss_per_s: Range = (0.0, 0.05)

    def sample(self, rng: np.random.Generator) -> dict[str, float] | None:
        """Draw one episode's parameters, or None when randomization is off.

        None rather than range midpoints on purpose. A midpoint is not the car: the
        configured depth FOV is 69 degrees (the S21 FE main lens in landscape) while the
        midpoint of the randomization band is 90. Returning None makes the disabled case
        mean "use exactly what the config says", which is what a baseline run and the
        scenario suite both need.
        """
        if not self.enabled:
            return None
        pick = lambda r: _u(rng, r)  # noqa: E731
        return {
            "wheelbase_scale": pick(self.wheelbase_scale),
            "accel_tau": pick(self.accel_tau),
            "steer_rate_rad_s": math.radians(pick(self.steer_rate_deg_s)),
            "steer_trim_rad": math.radians(pick(self.steer_trim_deg)),
            "max_steer_rad": math.radians(pick(self.max_steer_deg)),
            "throttle_deadband": pick(self.throttle_deadband),
            "throttle_gain": pick(self.throttle_gain),
            "understeer": pick(self.understeer),
            "max_speed_fwd": pick(self.max_speed_fwd),
            "depth_fov_deg": pick(self.depth_fov_deg),
            "depth_max_range": pick(self.depth_max_range),
            "depth_noise_frac": pick(self.depth_noise_frac),
            "depth_dropout": pick(self.depth_dropout),
            "depth_stationary_dropout": pick(self.depth_stationary_dropout),
            "depth_latency_steps": int(round(pick(self.depth_latency_steps))),
            "ultra_noise_m": pick(self.ultra_noise_m),
            "ultra_dropout": pick(self.ultra_dropout),
            "ultra_update_hz": pick(self.ultra_update_hz),
            "dt": pick(self.dt),
        }

    def sample_odometry(self, rng: np.random.Generator) -> dict[str, float] | None:
        """Draw the odometry parameters, keyed like `OdometryParams`, or None when off.

        Separate from `sample` so that adding these axes changed none of its draws: a seed
        still produces the same car and arena it did before the memory existed."""
        if not self.enabled:
            return None
        pick = lambda r: _u(rng, r)  # noqa: E731
        return {
            "scale_error": pick(self.odom_scale_error),
            "pos_noise": pick(self.odom_pos_noise),
            "yaw_noise": pick(self.odom_yaw_noise),
            "tracking_loss_per_s": pick(self.odom_tracking_loss_per_s),
        }
