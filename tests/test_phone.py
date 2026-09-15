"""The phone's reference controller must build the policy's input from what a phone has.

`link/phone.py` is what the Android app copies. These run it beside CarEnv on the same episodes
(`link/phone_check.py` shadow mode) and hold each input to the difference that the serial frames'
rounding explains, so a rule the phone gets wrong, or a doc leaves out, fails here.
"""

import math

import numpy as np
import pytest

from selfdrive.config import load_env_config
from selfdrive.envs.car_env import CarEnv
from selfdrive.link.phone import ArPose, DepthImage, PhoneController
from selfdrive.link.phone_check import ar_pose, input_groups, shadow
from selfdrive.link.protocol import FLAG_NO_ENCODER, Telemetry

# The most each input may differ by: $T rounds speed to 1 mm/s and steering to 0.01 degrees, and
# confidence divides speed by 0.3 m/s. Depth and the goal reach the phone unrounded.
ROUNDING = {"depth": 1e-6, "confidence": 4e-3, "speed": 4e-4, "steer": 2e-4,
            "goal_range": 1e-5, "goal_bearing": 1e-5, "patience": 1e-5}


def steer_to_goal(obs):
    """Head for the goal at a steady throttle, which runs into enough walls to wake the shield."""
    return np.array([np.clip(3.0 * obs[117], -1.0, 1.0), 0.7], dtype=np.float32)


@pytest.fixture(scope="module")
def cfg():
    c = load_env_config("configs/env_waypoint_s2_patience_shield.yaml")
    c.shield.unstick_after = 1.0
    c.max_steps = 400
    # The goal block holds while tracking is lost, where CarEnv's does not; tested on its own below.
    c.domain_rand.odom_tracking_loss_per_s = (0.0, 0.0)
    return c


@pytest.mark.parametrize("seed", [4, 11])
def test_phone_builds_the_input_the_policy_trained_on(cfg, seed):
    row = shadow(CarEnv(cfg), PhoneController(cfg, steer_to_goal), steer_to_goal, seed)
    largest = {name: gap[0] for name, gap in row["gaps"].items()}
    for name, bound in ROUNDING.items():
        assert largest[name] <= bound, (name, largest[name])
    # A ping that heard nothing arrives as -1.000 and reads 4.0 m. CarEnv keeps the noisy value
    # that ping clipped to 4.0 m.
    assert largest["ultrasonic"] <= 0.1
    # The phone skips a new ping that repeats the last frame's millimetre (docs/memory-ring.md),
    # and rounding moves points across sector edges, so the ring differs now and then.
    assert row["gaps"]["memory"][1] <= 0.03 * row["steps"]
    assert row["shield_disagree"] <= 0.01 * row["steps"]
    assert row["shield_capped_frac"] > 0.0  # the shield had something to decide
    assert row["arrival_only_steps"] == 0
    assert row["arrivals_agreed"] == row["goals_changed"]


def test_floor_pose_reads_arcore_axes_as_docs_goal_block_describe():
    """ARCore's camera looks along its local -Z. With the identity rotation that is world -Z, which
    is +y on the floor, so the car faces 90 degrees; the car's centre sits behind the camera."""
    x, y, theta = ArPose(1.0, 0.1, -2.0, 0.0, 0.0, 0.0, 1.0).floor_pose(0.18)
    assert (x, y, theta) == pytest.approx((1.0, 2.0 - 0.18, math.pi / 2))
    # A quarter turn clockwise about +Y looks along world +X: heading 0.
    q = math.sin(-math.pi / 4), math.cos(-math.pi / 4)
    assert ArPose(0.0, 0.0, 0.0, 0.0, q[0], 0.0, q[1]).floor_pose(0.0)[2] == pytest.approx(0.0)
    for pose in [(0.0, 0.0, 0.0), (1.5, -2.0, 2.5), (-3.0, 4.0, -1.2)]:
        assert ar_pose(*pose, 0.18).floor_pose(0.18) == pytest.approx(pose, abs=1e-12)


def _drive_straight(phone, poses, flags=0, pin=None):
    """Feed the phone a car at each (x, y, theta, tracking) at 30 Hz; return every input."""
    cfg = phone.cfg
    image_fov = math.radians(cfg.depth.fov_deg)
    out = []
    for k, (x, y, theta, tracking) in enumerate(poses):
        pose = ar_pose(x, y, theta, cfg.depth.mount_forward, tracking)
        image = DepthImage(np.full(cfg.depth.n_buckets, 5.0), np.zeros(cfg.depth.n_buckets, bool),
                           image_fov, k / 30.0, pose)
        t = Telemetry(k, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, flags)
        out.append(phone.observe(t, image, pose, pin, k / 30.0, 1.0 / 30.0, start=k == 0))
        phone.act(out[-1])
    return out


def test_without_an_encoder_speed_comes_from_the_pose(cfg):
    phone = PhoneController(cfg, lambda obs: np.zeros(2), speed_window=0.2)
    speed = input_groups(cfg.obs)["speed"]
    for v in (0.6, -0.4):
        heading = 0.7
        poses = [(v * k / 30.0 * math.cos(heading), v * k / 30.0 * math.sin(heading), heading, True)
                 for k in range(20)]
        obs = _drive_straight(phone, poses, flags=FLAG_NO_ENCODER)
        assert obs[-1][speed][0] == pytest.approx(v / cfg.obs.norm_speed_max, abs=1e-6)


def test_goal_block_holds_while_tracking_is_lost(cfg):
    phone = PhoneController(cfg, lambda obs: np.zeros(2))
    goal = input_groups(cfg.obs)
    block = slice(goal["goal_range"].start, goal["patience"].stop)
    poses = [(0.02 * k, 0.0, 0.0, not 10 <= k < 15) for k in range(20)]
    poses[12] = (5.0, 3.0, 1.0, False)  # a lost pose may be anywhere
    obs = _drive_straight(phone, poses, pin=(4.0, 0.0, -1.0))
    held = obs[9][block]
    for k in range(10, 15):
        assert obs[k][block][:3] == pytest.approx(held[:3])  # range and bearing
        assert obs[k][block][3] > obs[k - 1][block][3]  # the clock keeps counting
    assert obs[15][block][0] < held[0]  # tracking back: the car has closed in
