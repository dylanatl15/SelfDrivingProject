"""Dynamics ground truth.

The point of the kinematic bicycle is to forbid maneuvers the real car cannot perform.
These tests assert exactly those constraints - minimum turning radius, no sideways
motion, actuator lag - because a policy trained against a model that quietly allows
them will look excellent in the sim and fail on the carpet.
"""

import math

import numpy as np
import pytest

from selfdrive.dynamics import Actuators, CarParams, KinematicBicycle, wrap_angle

DT = 1.0 / 30.0


def ideal_params(**kw) -> CarParams:
    """No understeer, no deadband, no trim - isolates whatever a test is measuring."""
    base = dict(understeer=0.0, throttle_deadband=0.0, steer_trim_rad=0.0)
    base.update(kw)
    return CarParams(**base)


def settle(car: KinematicBicycle, steer: float, throttle: float, seconds: float = 3.0):
    for _ in range(int(seconds / DT)):
        car.step(steer, throttle, DT)


# --- straight line -----------------------------------------------------------

def test_zero_steer_drives_straight():
    car = KinematicBicycle(ideal_params())
    car.reset(0.0, 0.0, 0.0)
    settle(car, steer=0.0, throttle=1.0)
    assert car.state.y == pytest.approx(0.0, abs=1e-9)
    assert car.state.x > 1.0


def test_speed_converges_to_commanded_maximum():
    p = ideal_params(accel_tau=0.1)
    car = KinematicBicycle(p)
    car.reset()
    settle(car, steer=0.0, throttle=1.0, seconds=5.0)
    assert car.state.speed == pytest.approx(p.max_speed_fwd, rel=1e-3)


def test_reverse_uses_the_lower_reverse_speed_limit():
    p = ideal_params(accel_tau=0.1)
    car = KinematicBicycle(p)
    car.reset()
    settle(car, steer=0.0, throttle=-1.0, seconds=5.0)
    assert car.state.speed == pytest.approx(-p.max_speed_rev, rel=1e-3)
    assert car.state.x < 0.0


# --- turning -----------------------------------------------------------------

def test_turning_radius_matches_ackermann_geometry():
    p = ideal_params(accel_tau=0.05)
    car = KinematicBicycle(p)
    car.reset()
    settle(car, steer=1.0, throttle=1.0, seconds=2.0)  # let speed and servo settle

    before = car.state.theta
    car.step(1.0, 1.0, DT)
    yaw_rate = (car.state.theta - before) / DT
    measured = abs(car.state.speed / yaw_rate)
    assert measured == pytest.approx(p.min_turn_radius, rel=1e-3)


def test_full_circle_closes_on_itself():
    p = ideal_params(accel_tau=0.01)
    car = KinematicBicycle(p)
    car.reset()
    settle(car, steer=1.0, throttle=1.0, seconds=1.0)

    start = np.array([car.state.x, car.state.y])
    period = 2.0 * math.pi * p.min_turn_radius / car.state.speed
    for _ in range(int(round(period / DT))):
        car.step(1.0, 1.0, DT)
    end = np.array([car.state.x, car.state.y])
    # Midpoint integration at 30 Hz should close a ~0.5 m circle to within a centimetre.
    assert np.linalg.norm(end - start) < 0.01


def test_car_cannot_move_sideways():
    """No lateral slip: every step is pure travel along the arc the wheels describe.

    Measured against the *midpoint* heading, not the heading at the start of the step.
    A car on an arc necessarily ends up offset from its initial heading - the chord of
    an arc is not parallel to its starting tangent, which at full lock is about 2.7 mm
    per 33 ms step here. That is geometry, not slip. The midpoint heading is the one a
    slip-free kinematic step must be exactly parallel to.
    """
    p = ideal_params(accel_tau=0.05)
    car = KinematicBicycle(p)
    car.reset()
    settle(car, steer=1.0, throttle=1.0, seconds=1.0)

    prev = np.array([car.state.x, car.state.y])
    theta_before = car.state.theta
    car.step(1.0, 1.0, DT)
    step_vec = np.array([car.state.x, car.state.y]) - prev

    d_theta = wrap_angle(car.state.theta - theta_before)
    mid = theta_before + 0.5 * d_theta
    heading = np.array([math.cos(mid), math.sin(mid)])
    # numpy 2 removed np.cross for 2-D inputs; this is the scalar z component.
    lateral = abs(float(heading[0] * step_vec[1] - heading[1] * step_vec[0]))
    assert lateral < 1e-12

    # And the distance covered is exactly speed * dt - nothing sideways was added.
    assert float(np.linalg.norm(step_vec)) == pytest.approx(abs(car.state.speed) * DT,
                                                            rel=1e-9)


def test_understeer_widens_the_turn():
    tight = KinematicBicycle(ideal_params(accel_tau=0.05))
    wide = KinematicBicycle(ideal_params(accel_tau=0.05, understeer=0.5))
    for car in (tight, wide):
        car.reset()
        settle(car, steer=1.0, throttle=1.0, seconds=2.0)
    def yaw_rate(car):
        before = car.state.theta
        car.step(1.0, 1.0, DT)
        return abs(car.state.theta - before) / DT

    # Same wheel angle, same speed: the understeering car must rotate more slowly.
    assert yaw_rate(wide) < yaw_rate(tight)


# --- actuators ---------------------------------------------------------------

def test_acceleration_lag_follows_first_order_response():
    p = ideal_params(accel_tau=0.5)
    a = Actuators(p)
    for _ in range(int(p.accel_tau / DT)):
        a.step(0.0, 1.0, DT)
    # One time constant in, a first-order system reaches ~63.2% of its target.
    assert a.speed == pytest.approx(0.632 * p.max_speed_fwd, rel=0.02)


def test_car_coasts_to_a_true_stop():
    p = ideal_params(accel_tau=0.2)
    a = Actuators(p)
    a.step(0.0, 1.0, 5.0)
    a.step(0.0, 0.0, 5.0)
    # An exponential never reaches zero; without the stop threshold the car creeps forever
    # and every displacement-based stall check silently stops firing.
    assert a.speed == 0.0


def test_throttle_deadband_blocks_small_commands():
    a = Actuators(CarParams(throttle_deadband=0.2))
    assert a.target_speed(0.15) == 0.0
    assert a.target_speed(-0.15) == 0.0
    assert a.target_speed(0.5) > 0.0


def test_full_throttle_still_reaches_full_speed_through_a_deadband():
    p = CarParams(throttle_deadband=0.2)
    assert Actuators(p).target_speed(1.0) == pytest.approx(p.max_speed_fwd)


def test_steering_is_slew_rate_limited():
    p = ideal_params(steer_rate_rad_s=math.radians(90.0))
    a = Actuators(p)
    a.step(1.0, 0.0, DT)
    assert a.steer == pytest.approx(math.radians(90.0) * DT)
    assert a.steer < p.max_steer_rad  # cannot reach the stop in a single control period


def test_steering_saturates_at_the_mechanical_limit():
    p = ideal_params()
    a = Actuators(p)
    for _ in range(200):
        a.step(5.0, 0.0, DT)  # deliberately out of range
    assert a.steer == pytest.approx(p.max_steer_rad)


def test_steer_trim_biases_the_neutral_point():
    p = CarParams(steer_trim_rad=math.radians(3.0))
    a = Actuators(p)
    for _ in range(200):
        a.step(0.0, 0.0, DT)
    assert a.steer == pytest.approx(math.radians(3.0))


def test_trim_cannot_push_past_the_mechanical_limit():
    p = CarParams(steer_trim_rad=math.radians(20.0))
    a = Actuators(p)
    for _ in range(400):
        a.step(1.0, 0.0, DT)
    assert a.steer == pytest.approx(p.max_steer_rad)


def test_open_loop_gain_error_scales_achieved_speed():
    slow = Actuators(ideal_params(throttle_gain=0.7))
    assert slow.target_speed(1.0) == pytest.approx(0.7 * CarParams().max_speed_fwd)
