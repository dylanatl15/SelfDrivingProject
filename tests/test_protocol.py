"""Serial protocol conformance.

These tests are the contract. The ESP32 firmware and the Android app are written against
`docs/protocol.md` by people who are not running this code, so the two example frames in
that document are pinned here byte for byte - if someone changes the codec, the doc that
two teammates are building against must fail loudly rather than quietly become a lie.
"""

import pytest

from selfdrive.link.protocol import (
    FLAG_FAILSAFE,
    FLAG_NO_ENCODER,
    Command,
    ProtocolError,
    Telemetry,
    crc8,
    decode_command,
    decode_telemetry,
    encode_command,
    encode_telemetry,
)

# Verbatim from docs/protocol.md.
DOC_COMMAND = b"$C,417,-12.50,0.800*DB\n"
DOC_TELEMETRY = b"$T,417,0.842,-11.75,1.230,0.450,4.000,-1.000,02*DD\n"


def test_documented_command_example_is_exact():
    assert encode_command(Command(seq=417, steer_deg=-12.5, throttle=0.8)) == DOC_COMMAND


def test_documented_telemetry_example_is_exact():
    t = Telemetry(
        seq=417, speed_mps=0.842, steer_deg=-11.75,
        us_front_m=1.23, us_left_m=0.45, us_right_m=4.0, us_back_m=-1.0,
        flags=FLAG_NO_ENCODER,
    )
    assert encode_telemetry(t) == DOC_TELEMETRY


def test_crc8_is_crc8_atm():
    # Independently known value for the standard check string.
    assert crc8(b"123456789") == 0xF4


def test_command_roundtrip():
    original = Command(seq=65535, steer_deg=27.25, throttle=-0.375)
    back = decode_command(encode_command(original))
    assert back.seq == original.seq
    assert back.steer_deg == pytest.approx(original.steer_deg)
    assert back.throttle == pytest.approx(original.throttle)


def test_telemetry_roundtrip():
    original = Telemetry(
        seq=1, speed_mps=-0.512, steer_deg=3.0,
        us_front_m=0.2, us_left_m=-1.0, us_right_m=2.5, us_back_m=4.0,
        flags=FLAG_FAILSAFE | FLAG_NO_ENCODER,
    )
    back = decode_telemetry(encode_telemetry(original))
    assert back == original
    assert back.failsafe
    assert back.ultrasonics() == [0.2, -1.0, 2.5, 4.0]


def test_sequence_number_wraps_at_uint16():
    assert decode_command(encode_command(Command(65536 + 7, 0.0, 0.0))).seq == 7


def test_corrupted_payload_is_rejected():
    bad = DOC_COMMAND.replace(b"0.800", b"0.900")  # CRC no longer matches
    with pytest.raises(ProtocolError):
        decode_command(bad)


def test_wrong_frame_type_is_rejected():
    with pytest.raises(ProtocolError):
        decode_telemetry(DOC_COMMAND)


@pytest.mark.parametrize(
    "line",
    [
        b"",
        b"\n",
        b"C,417,-12.50,0.800*DB\n",  # no leading $
        b"$C,417,-12.50,0.800\n",  # no checksum
        b"$C,417,-12.50*7B\n",  # too few fields
        b"$C,417,-12.50,0.800,9*DB\n",  # too many fields
        b"$C,nope,-12.50,0.800*DB\n",  # unparseable field
        b"$C,417,-12.50,0.800*ZZ\n",  # non-hex checksum
    ],
)
def test_malformed_frames_raise_rather_than_return_garbage(line):
    # Firmware is told to drop these silently; the codec must make that possible by
    # failing loudly instead of handing back a half-parsed Command.
    with pytest.raises(ProtocolError):
        decode_command(line)


def test_decode_accepts_str_as_well_as_bytes():
    assert decode_command(DOC_COMMAND.decode()).seq == 417


def test_frames_are_newline_terminated_and_ascii():
    frame = encode_command(Command(1, 0.0, 0.0))
    assert frame.endswith(b"\n")
    assert frame.decode("ascii").isprintable() is False  # trailing \n
    assert frame[:-1].decode("ascii").isprintable()
