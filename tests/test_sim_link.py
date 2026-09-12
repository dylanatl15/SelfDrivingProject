"""Simulated ESP32 behaviour, including the failsafe.

The 200 ms failsafe is a safety requirement, not a tuning parameter: a crashed phone app
or a USB cable shaken loose over a bump must not leave a LiPo-powered car driving at a
wall or a person. Testing it here means the required behaviour is pinned before the
firmware that has to implement it exists.
"""

import math

import pytest

from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.link.protocol import (
    FLAG_NO_ENCODER,
    FLAG_US_TIMEOUT,
    NO_READING,
    Command,
    Telemetry,
    decode_telemetry,
    encode_command,
)
from selfdrive.link.sim_link import SimEsp32


def make_link(has_encoder: bool = False, n_ultrasonic: int = 4) -> SimEsp32:
    cfg = EnvConfig(domain_rand=DomainRandConfig(enabled=False), max_steps=2000)
    cfg.ultrasonic.n_sensors = n_ultrasonic
    env = CarEnv(cfg)
    env.reset(seed=0, options={"pose": (0.0, 0.0, 0.0)})
    return SimEsp32(env, has_encoder=has_encoder)


def drive(link: SimEsp32, seq_start: int, steer_deg: float, throttle: float, steps: int):
    for i in range(steps):
        link.write(encode_command(Command(seq_start + i, steer_deg, throttle)))
        link.step()


def test_a_command_frame_moves_the_car():
    link = make_link()
    drive(link, 0, 0.0, 1.0, 60)
    assert link.env.car.state.x > 0.5
    assert not link.failsafe_active


def test_telemetry_is_a_valid_frame_and_echoes_the_sequence_number():
    link = make_link()
    drive(link, 500, 5.0, 0.5, 5)
    t = decode_telemetry(link.readline())
    assert isinstance(t, Telemetry)
    assert t.seq == 504


def test_failsafe_engages_after_the_timeout():
    link = make_link()
    drive(link, 0, 0.0, 1.0, 60)
    moving = link.env.car.state.speed
    assert moving > 0.1

    # Phone goes silent. 200 ms at ~33 ms per step is about 7 steps.
    for _ in range(10):
        link.step()
    assert link.failsafe_active
    assert decode_telemetry(link.readline()).failsafe


def test_failsafe_actually_stops_the_car():
    link = make_link()
    drive(link, 0, 0.0, 1.0, 60)
    for _ in range(120):
        link.step()  # no commands at all
    assert link.env.car.state.speed == pytest.approx(0.0, abs=1e-6)


def test_failsafe_clears_on_the_next_valid_command():
    link = make_link()
    for _ in range(30):
        link.step()
    assert link.failsafe_active

    link.write(encode_command(Command(1, 0.0, 1.0)))
    assert not link.failsafe_active
    link.step()
    assert not decode_telemetry(link.readline()).failsafe


def test_a_corrupted_frame_does_not_clear_the_failsafe():
    link = make_link()
    for _ in range(30):
        link.step()
    assert link.failsafe_active

    corrupted = encode_command(Command(1, 0.0, 1.0)).replace(b"1.000", b"1.500")
    link.write(corrupted)
    assert link.failsafe_active, "a bad CRC must never be treated as a live command"


def test_a_corrupted_frame_does_not_change_the_held_command():
    link = make_link()
    good = encode_command(Command(1, 10.0, 0.5))
    link.write(good)
    link.write(b"$C,2,-99.00,-1.000*00\n")  # wrong CRC
    link.step()
    assert link.telemetry().seq == 1  # still the last frame that actually parsed


def test_no_encoder_flag_and_zero_speed_when_the_car_has_none():
    link = make_link(has_encoder=False)
    drive(link, 0, 0.0, 1.0, 60)
    t = decode_telemetry(link.readline())
    assert t.flags & FLAG_NO_ENCODER
    assert t.speed_mps == 0.0
    assert link.env.car.state.speed > 0.1  # it is moving, it just cannot measure it


def test_encoder_equipped_car_reports_real_speed():
    link = make_link(has_encoder=True)
    drive(link, 0, 0.0, 1.0, 60)
    t = decode_telemetry(link.readline())
    assert not t.flags & FLAG_NO_ENCODER
    assert t.speed_mps == pytest.approx(link.env.car.state.speed, abs=1e-3)


def test_three_sensor_build_reports_a_sentinel_for_the_missing_front():
    link = make_link(n_ultrasonic=3)
    drive(link, 0, 0.0, 0.5, 5)
    t = decode_telemetry(link.readline())
    assert t.us_front_m == NO_READING
    assert t.flags & FLAG_US_TIMEOUT
    assert t.us_left_m >= 0.0 and t.us_right_m >= 0.0 and t.us_back_m >= 0.0


def test_steering_degrees_are_clamped_to_the_mechanical_limit():
    link = make_link()
    drive(link, 0, 999.0, 0.5, 90)
    assert abs(link.env.car.state.steer) <= link.env.car.p.max_steer_rad + 1e-9


def test_steering_sign_convention_matches_the_protocol():
    """docs/protocol.md: positive steer_deg means LEFT, i.e. increasing heading.

    Checked over a short drive on purpose. At near-full lock this car completes a
    circle in well under a second of simulated time, and a wrapped heading would make
    a correct left turn look like a right one.
    """
    link = make_link()
    drive(link, 0, 20.0, 1.0, 20)
    assert 0.0 < link.env.car.state.theta < math.pi
    assert link.env.car.state.y > 0.0  # and it moved to the left of where it started


def test_negative_steering_turns_the_other_way():
    link = make_link()
    drive(link, 0, -20.0, 1.0, 20)
    assert -math.pi < link.env.car.state.theta < 0.0
    assert link.env.car.state.y < 0.0
