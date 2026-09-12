"""Drive the simulator through the real serial protocol.

The point is that the policy's path to the car is *identical* in simulation and on
hardware: build a `$C` frame, receive a `$T` frame, rebuild the observation from
telemetry. Any bug in the codec, the field order, the units or the sign conventions
shows up here, on a laptop, instead of on a moving vehicle.

It also provides the midterm fallback. A hardware-in-the-loop demo - real phone, real
ARCore, real policy, real serial frames, simulated chassis - needs nothing from the
mechanical build, so a chassis that slips a week does not cost the demo.

`SimEsp32` implements the same `write()`/`readline()` surface as a serial port, so
swapping in `pyserial` later changes one line at the call site.
"""

from __future__ import annotations

import math

import numpy as np

from ..envs.car_env import STEER, THROTTLE, CarEnv
from .protocol import (
    FAILSAFE_TIMEOUT_MS,
    FLAG_FAILSAFE,
    FLAG_NO_ENCODER,
    FLAG_US_TIMEOUT,
    NO_READING,
    Command,
    ProtocolError,
    Telemetry,
    decode_command,
    encode_telemetry,
)


class SimEsp32:
    """Stands in for the ESP32: consumes `$C` frames, steps the sim, emits `$T` frames.

    The 200 ms failsafe is implemented here too, so the behaviour the firmware is
    required to have can be tested before the firmware exists.
    """

    def __init__(self, env: CarEnv, has_encoder: bool = False,
                 failsafe_ms: int = FAILSAFE_TIMEOUT_MS):
        self.env = env
        self.has_encoder = has_encoder
        self.failsafe_ms = failsafe_ms
        self._command = Command(seq=0, steer_deg=0.0, throttle=0.0)
        self._since_command_ms = 0.0
        self._failsafe = True  # nothing has been commanded yet
        self._last_info: dict = {}
        self._terminated = False
        self._truncated = False

    # --- serial-port-shaped surface -----------------------------------------

    def write(self, line: bytes | str) -> None:
        """Accept a command frame. Malformed frames are dropped, exactly as the
        firmware is required to do - never act on a partially parsed frame."""
        try:
            self._command = decode_command(line)
        except ProtocolError:
            return
        self._since_command_ms = 0.0
        self._failsafe = False

    def readline(self) -> bytes:
        return encode_telemetry(self.telemetry())

    # --- simulation ----------------------------------------------------------

    def step(self) -> tuple[bool, bool]:
        """Advance one control period. Returns (terminated, truncated)."""
        dt = self.env.dt
        self._since_command_ms += dt * 1000.0
        if self._since_command_ms > self.failsafe_ms:
            self._failsafe = True

        if self._failsafe:
            # Drive to zero, steering holds - the documented failsafe behaviour.
            throttle = 0.0
            steer_deg = self._command.steer_deg
        else:
            throttle = self._command.throttle
            steer_deg = self._command.steer_deg

        max_steer_deg = math.degrees(self.env.car.p.max_steer_rad)
        steer_norm = float(np.clip(steer_deg / max_steer_deg, -1.0, 1.0))

        action = np.zeros(2, dtype=np.float32)
        action[STEER] = steer_norm
        action[THROTTLE] = float(np.clip(throttle, -1.0, 1.0))

        _, _, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        self._terminated, self._truncated = terminated, truncated
        return terminated, truncated

    def telemetry(self) -> Telemetry:
        s = self.env.car.state
        ultra = self.env.ultra.true_ranges(self.env.world, s)
        names = self.env.ultra.names
        by_name = dict(zip(names, ultra, strict=True))
        # A 3-sensor build genuinely has no front reading; report the sentinel rather
        # than inventing a number the real car cannot produce.
        readings = [by_name.get(n, NO_READING) for n in ("front", "left", "right", "back")]

        flags = 0
        if self._failsafe:
            flags |= FLAG_FAILSAFE
        if not self.has_encoder:
            flags |= FLAG_NO_ENCODER
        if any(r == NO_READING for r in readings):
            flags |= FLAG_US_TIMEOUT

        return Telemetry(
            seq=self._command.seq,
            speed_mps=s.speed if self.has_encoder else 0.0,
            steer_deg=math.degrees(s.steer),
            us_front_m=float(readings[0]),
            us_left_m=float(readings[1]),
            us_right_m=float(readings[2]),
            us_back_m=float(readings[3]),
            flags=flags,
        )

    @property
    def failsafe_active(self) -> bool:
        return self._failsafe
