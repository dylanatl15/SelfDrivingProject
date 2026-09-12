# Serial Protocol v1 — Phone ⇄ ESP32

**Status: frozen.** The Android app, the ESP32 firmware and the simulator are all built
against this document. Propose changes as v2 rather than editing v1 in place — a silent
change here breaks two other people's work without any compile error.

Reference implementation: [`src/selfdrive/link/protocol.py`](../src/selfdrive/link/protocol.py).
Conformance tests: [`tests/test_protocol.py`](../tests/test_protocol.py).

## Transport

| | |
|---|---|
| Physical | USB-OTG, phone (host) → ESP32 (device) |
| Baud | **115200**, 8N1, no flow control |
| Framing | ASCII, one frame per line, `\n` terminated |
| Command rate | 30 Hz nominal (policy rate) |
| Telemetry rate | 30 Hz nominal, free-running |

ASCII costs bandwidth but can be read with any serial monitor while the car is
misbehaving on a bench. At 30 Hz a ~60 byte frame uses well under 2% of the link.

## Frame format

```
$<TYPE>,<field>,...,<field>*<CRC8>\n
```

`CRC8` is two uppercase hex digits over the payload between `$` and `*`, exclusive.
Polynomial `0x07` (CRC-8/ATM), init `0x00`, no reflection, no final XOR.

Any frame that fails CRC, has the wrong field count, or fails to parse **must be
dropped silently**. Never act on a partially parsed frame, and never block waiting for a
good one.

## Sign conventions

These must match the simulator exactly or the trained policy will steer backwards.

| Quantity | Positive means |
|---|---|
| `steer_deg` | **left** (counter-clockwise, increasing heading) |
| `throttle` | **forward** |

## `$C` — Command (phone → ESP32)

```
$C,<seq>,<steer_deg>,<throttle>*<crc>
```

| Field | Type | Range | Notes |
|---|---|---|---|
| `seq` | uint16 | 0–65535 | monotonic, wraps; echoed back in telemetry |
| `steer_deg` | float, 2 dp | ±`MAX_STEER_DEG` | target angle, **not** a servo microsecond value |
| `throttle` | float, 3 dp | −1.000 … +1.000 | −1 full reverse, +1 full forward |

Example: `$C,417,-12.50,0.800*DB`

The phone sends geometric intent. Translating an angle into servo PWM and a throttle
fraction into motor PWM is the ESP32's job, and keeping that split is what lets the
chassis be retuned without retraining the policy.

## `$T` — Telemetry (ESP32 → phone)

```
$T,<seq>,<v_mps>,<steer_deg>,<us_f>,<us_l>,<us_r>,<us_b>,<flags>*<crc>
```

| Field | Type | Units | Notes |
|---|---|---|---|
| `seq` | uint16 | — | echo of the last accepted `$C` |
| `v_mps` | float, 3 dp | m/s | signed; `0.0` with `NO_ENCODER` set if unmeasurable |
| `steer_deg` | float, 2 dp | deg | measured or estimated actual angle |
| `us_f/l/r/b` | float, 3 dp | m | `-1.000` means no echo this cycle |
| `flags` | uint8 hex | — | see below |

Example: `$T,417,0.842,-11.75,1.230,0.450,4.000,-1.000,02*DD`

### Flags

| Bit | Name | Meaning |
|---|---|---|
| `0x01` | `FAILSAFE` | motors cut, no valid command in time |
| `0x02` | `NO_ENCODER` | `v_mps` is an estimate, not a measurement |
| `0x04` | `LOW_BATTERY` | pack voltage below threshold |
| `0x08` | `US_TIMEOUT` | at least one ultrasonic returned no echo |
| `0x10` | `OVERCURRENT` | driver current limit tripped |

## Failsafe — required

**If no valid `$C` frame arrives within 200 ms, the ESP32 must cut motor output and set
`FLAG_FAILSAFE`.** Steering may hold its last position; drive must go to zero.

This is a safety requirement, not a tuning parameter. A crashed Android app, a locked
phone screen, or a USB cable that shakes loose over a bump must not leave a
LiPo-powered car driving at a wall — or at a person. `$C` at 30 Hz *is* the heartbeat;
no separate keepalive frame exists.

Recovery: clear the failsafe on the next valid `$C`, but ramp throttle from zero rather
than jumping to the commanded value.

## Ultrasonics

Fire them **one at a time**, round-robin. HC-SR04s cross-talk if pinged simultaneously —
one sensor hears another's burst and reports a phantom obstacle. A full cycle of four at
a ~20–60 ms settling time each lands around 15–25 Hz total, so most `$T` frames repeat at
least one stale reading. That is expected and the simulator models it; do not try to hide
it by interpolating.

A 3-sensor build drops the **front** sensor: the camera already covers forward, so
left/right/back buy the most new information.

## Notes for the Android side

- The policy consumes normalized observations; scaling constants are in
  [`src/selfdrive/envs/obs.py`](../src/selfdrive/envs/obs.py) (`ObsConfig`) and **must**
  be mirrored exactly. Changing one there invalidates every trained checkpoint.
- Observation block layout, frame stacking and ego-state fields are documented in the
  same file. The phone builds the identical 17-float frame and stacks 4 of them.
- Mount the phone in **landscape**. ARCore binds the main lens (it does not support
  ultra-wide), which is ~69° horizontally in landscape but only ~50° in portrait. That
  is 20° of free peripheral vision for the cost of rotating a bracket.
