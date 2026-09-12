"""Serial wire protocol v1 between the phone and the ESP32.

Frozen so the Android app, the ESP32 firmware and the simulator can be built against it
independently. The full specification, including the parts that are firmware-side only,
lives in `docs/protocol.md`; this module is the reference codec and the thing the tests
check firmware behaviour against.

Frame format, NMEA-style ASCII, one frame per line:

    $<TYPE>,<field>,...,<field>*<CRC8><LF>

CRC8 covers the payload between `$` and `*`, exclusive, as two uppercase hex digits.
ASCII is not the most compact choice, but it can be read with a serial monitor when the
car is misbehaving on a table at midnight, which is worth more than the bytes.

Sign conventions, which must match the simulator or nothing transfers:
    steering  positive = left  (counter-clockwise, same as increasing heading)
    throttle  positive = forward, range [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

CRC8_POLY = 0x07  # CRC-8/ATM; a five-line loop on an ESP32
BAUD_DEFAULT = 115200

# Telemetry status bits.
FLAG_FAILSAFE = 0x01  # motors cut because no command arrived in time
FLAG_NO_ENCODER = 0x02  # v_mps is an estimate, not a measurement
FLAG_LOW_BATTERY = 0x04
FLAG_US_TIMEOUT = 0x08  # at least one ultrasonic returned no echo
FLAG_OVERCURRENT = 0x10

# The ESP32 must cut motor output if no valid command frame arrives within this window.
# This is a safety requirement, not a tuning parameter: a crashed phone app or an
# unplugged USB cable must not leave a battery-powered car driving at a wall or a person.
FAILSAFE_TIMEOUT_MS = 200

NO_READING = -1.0  # ultrasonic sentinel for "no echo"


class ProtocolError(ValueError):
    """Raised on a malformed or corrupted frame. Callers should drop it and continue."""


def crc8(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ CRC8_POLY) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _wrap(payload: str) -> bytes:
    return f"${payload}*{crc8(payload.encode('ascii')):02X}\n".encode("ascii")


def _unwrap(line: bytes | str, expect: str) -> list[str]:
    text = line.decode("ascii", errors="replace") if isinstance(line, bytes) else line
    text = text.strip()
    if not text.startswith("$") or "*" not in text:
        raise ProtocolError(f"not a frame: {text!r}")
    payload, _, checksum = text[1:].rpartition("*")
    try:
        given = int(checksum, 16)
    except ValueError as exc:
        raise ProtocolError(f"bad checksum field: {checksum!r}") from exc
    if given != crc8(payload.encode("ascii")):
        raise ProtocolError(f"CRC mismatch on {text!r}")
    fields = payload.split(",")
    if fields[0] != expect:
        raise ProtocolError(f"expected a {expect!r} frame, got {fields[0]!r}")
    return fields


@dataclass
class Command:
    """Phone to ESP32: what the policy wants the car to do."""

    seq: int  # uint16, wraps
    steer_deg: float  # positive = left
    throttle: float  # [-1, 1], negative = reverse


@dataclass
class Telemetry:
    """ESP32 to phone: what the car is actually doing."""

    seq: int  # echo of the last command accepted
    speed_mps: float  # 0.0 and FLAG_NO_ENCODER if the car has no encoders
    steer_deg: float  # measured or estimated servo position
    us_front_m: float
    us_left_m: float
    us_right_m: float
    us_back_m: float
    flags: int = 0

    @property
    def failsafe(self) -> bool:
        return bool(self.flags & FLAG_FAILSAFE)

    def ultrasonics(self) -> list[float]:
        return [self.us_front_m, self.us_left_m, self.us_right_m, self.us_back_m]


def encode_command(cmd: Command) -> bytes:
    return _wrap(f"C,{cmd.seq & 0xFFFF},{cmd.steer_deg:.2f},{cmd.throttle:.3f}")


def decode_command(line: bytes | str) -> Command:
    f = _unwrap(line, "C")
    if len(f) != 4:
        raise ProtocolError(f"C frame needs 4 fields, got {len(f)}")
    try:
        return Command(seq=int(f[1]), steer_deg=float(f[2]), throttle=float(f[3]))
    except ValueError as exc:
        raise ProtocolError(f"unparseable C frame: {line!r}") from exc


def encode_telemetry(t: Telemetry) -> bytes:
    return _wrap(
        f"T,{t.seq & 0xFFFF},{t.speed_mps:.3f},{t.steer_deg:.2f},"
        f"{t.us_front_m:.3f},{t.us_left_m:.3f},{t.us_right_m:.3f},{t.us_back_m:.3f},"
        f"{t.flags:02X}"
    )


def decode_telemetry(line: bytes | str) -> Telemetry:
    f = _unwrap(line, "T")
    if len(f) != 9:
        raise ProtocolError(f"T frame needs 9 fields, got {len(f)}")
    try:
        return Telemetry(
            seq=int(f[1]),
            speed_mps=float(f[2]),
            steer_deg=float(f[3]),
            us_front_m=float(f[4]),
            us_left_m=float(f[5]),
            us_right_m=float(f[6]),
            us_back_m=float(f[7]),
            flags=int(f[8], 16),
        )
    except ValueError as exc:
        raise ProtocolError(f"unparseable T frame: {line!r}") from exc
