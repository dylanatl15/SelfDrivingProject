"""Run the phone's reference controller against the simulator and measure how far its inputs
stray from the ones the policy trained on.

    python -m selfdrive.link.phone_check --onnx runs/exports/<model>.onnx --env <env.yaml> \\
        --mode drive --seed-start 4000000 --n 25 --out rows.jsonl

`shadow` lets `CarEnv` drive its own episode with its own shield, as in an exam. Beside it the
phone gets the `$T` frames, depth images and poses that episode produces and builds its own input
and command, and the two are compared at every step. The phone never steers there, so a small
difference cannot grow into a different episode: anything beyond rounding is a rule the phone
is missing or gets wrong.

`drive` hands the car to the phone, through the serial codec and `SimEsp32`, with the
environment's shield off because the phone runs it. That scores the whole loop, like an exam.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from ..config import load_env_config
from ..envs.car_env import STEER, THROTTLE, CarEnv
from ..envs.obs import ObsConfig
from ..envs.shield import BRAKED, CAPPED, UNSTUCK
from .phone import ArPose, DepthImage, OnnxPolicy, PhoneController
from .protocol import decode_telemetry
from .sim_link import SimEsp32

KEYS = ("goals_reached", "collided", "stuck", "mean_speed_mps", "backing_frac", "lock_frac",
        "min_clearance_m", "goal_progress_m", "distance_m")
GAP = 0.01  # an input off by more than this counts as a step that differs


def ar_pose(x: float, y: float, theta: float, mount_forward: float, tracking: bool = True,
            height: float = 0.12) -> ArPose:
    """The ARCore pose of a phone whose car sits at (x, y, theta): the inverse of
    `ArPose.floor_pose`. The lens looks along the heading, a rotation of theta - 90 degrees
    about +Y."""
    half = (theta - math.pi / 2.0) / 2.0
    return ArPose(x + mount_forward * math.cos(theta), height,
                  -(y + mount_forward * math.sin(theta)),
                  0.0, math.sin(half), 0.0, math.cos(half), tracking)


class SimPhone:
    """What the phone's own sensors hand it in the simulator. ARCore's pose is the environment's
    drifting odometry estimate, the depth image the buckets the observation was built from, and
    the pin the goal the environment is paying for."""

    def __init__(self, env: CarEnv):
        self.env = env
        self._poses: list[tuple[float, ArPose]] = []

    @property
    def now(self) -> float:
        return self.env.steps * self.env.dt

    def _pose(self) -> tuple[float, ArPose]:
        o = self.env.odometry
        return self.now, ar_pose(o.x, o.y, o.theta, self.env.cfg.depth.mount_forward, o.tracking)

    def reset(self) -> None:
        self._poses = [self._pose()]

    def advance(self) -> None:
        """Record the pose after an environment step."""
        self._poses.append(self._pose())

    def pose(self) -> ArPose:
        return self._poses[-1][1]

    def image(self) -> DepthImage:
        env = self.env
        # Depth arrives `latency_steps` late; the image carries the pose it was captured at.
        stamp, pose = self._poses[max(0, len(self._poses) - 1 - env.depth.p.latency_steps)]
        depth, _ = env.readings
        return DepthImage(depth, env.depth.fresh.copy(), math.radians(env.depth.p.fov_deg),
                          stamp, pose)

    def pin(self) -> tuple[float, float, float] | None:
        goal = self.env.goals.goal if self.env.goals is not None else None
        return None if goal is None else (goal[0], 0.0, -goal[1])


def input_groups(o: ObsConfig) -> dict[str, slice]:
    """Index ranges of the newest frame's fields and the unstacked blocks. Older frames are
    earlier steps' newest frames, so comparing them again adds nothing."""
    base = o.per_frame * (o.frame_stack - 1)
    groups, i = {}, base
    for name, size in (("depth", o.n_depth), ("confidence", 1), ("ultrasonic", o.n_ultrasonic),
                       ("speed", 1), ("steer", 1), ("last_throttle", 1), ("last_steer", 1)):
        groups[name] = slice(i, i + size)
        i += size
    i = o.per_frame * o.frame_stack
    if o.memory_sectors:
        groups["memory"] = slice(i, i + 2 * o.memory_sectors)
        i += 2 * o.memory_sectors
    if o.goal_block:
        groups["goal_range"] = slice(i, i + 1)
        groups["goal_bearing"] = slice(i + 1, i + 3)
        if o.goal_patience:
            groups["patience"] = slice(i + 3, i + 4)
    return groups


class Gaps:
    """Per input group: the largest difference seen and the steps it exceeded `GAP`."""

    def __init__(self, o: ObsConfig):
        self.groups = input_groups(o)
        self.largest = dict.fromkeys(self.groups, 0.0)
        self.steps_over = dict.fromkeys(self.groups, 0)

    def add(self, mine: np.ndarray, theirs: np.ndarray) -> None:
        d = np.abs(np.asarray(mine, dtype=np.float64) - np.asarray(theirs, dtype=np.float64))
        for name, sl in self.groups.items():
            m = float(d[sl].max())
            self.largest[name] = max(self.largest[name], m)
            self.steps_over[name] += m > GAP

    def summary(self) -> dict:
        return {name: [self.largest[name], self.steps_over[name]] for name in self.groups}


def _row(env: CarEnv, phone: PhoneController, seed: int, info: dict, gaps: Gaps,
         extra: dict) -> dict:
    m = info["episode_metrics"]
    row = dict(seed=seed, steps=env.steps, minutes=env.steps * env.dt / 60.0,
               yaw_noise=env.odometry.p.yaw_noise,
               max_steer_deg=math.degrees(env.car.p.max_steer_rad),
               fov_deg=env.depth.p.fov_deg, latency_steps=env.depth.p.latency_steps,
               **{k: float(m.get(k, float("nan"))) for k in KEYS})
    row["clean"] = row["goals_reached"] * (row["collided"] == 0 and row["stuck"] == 0)
    row.update(gaps=gaps.summary(), goals_changed=phone.goals_changed,
               arrivals_agreed=phone.arrivals_agreed, arrival_only_steps=phone.arrival_only_steps,
               **extra)
    return row


def shadow(env: CarEnv, phone: PhoneController, policy, seed: int, has_encoder: bool = True,
           measured_steer: bool = True) -> dict:
    """The environment drives; the phone builds its input and command beside it."""
    obs, _ = env.reset(seed=seed)
    sim, link = SimPhone(env), SimEsp32(env, has_encoder=has_encoder)
    sim.reset()
    if measured_steer:
        phone.max_steer_deg = math.degrees(env.car.p.max_steer_rad)
    gaps = Gaps(env.cfg.obs)
    steer_gap = throttle_gap = 0.0
    shield_disagree = lost = 0
    start = True
    while True:
        mine = phone.observe(decode_telemetry(link.readline()), sim.image(), sim.pose(),
                             sim.pin(), sim.now, env.dt, start)
        start = False
        gaps.add(mine, obs)
        link.write(phone.act(mine))
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if "throttle_sent" in info:
            steer_gap = max(steer_gap, abs(phone.sent[STEER] - info["steer_sent"]))
            throttle_gap = max(throttle_gap, abs(phone.sent[THROTTLE] - info["throttle_sent"]))
            shield_disagree += phone.did != info["shield"]
        sim.advance()
        lost += not env.odometry.tracking
        if terminated or truncated:
            m = info["episode_metrics"]
            shielded = {k: m[k] for k in ("shield_capped_frac", "shield_braked_frac",
                                          "shield_unstuck_frac") if k in m}
            return _row(env, phone, seed, info, gaps, dict(
                steer_gap=steer_gap, throttle_gap=throttle_gap, shield_disagree=shield_disagree,
                lost_steps=lost, **shielded))


def drive(env: CarEnv, phone: PhoneController, seed: int, has_encoder: bool = True,
          measured_steer: bool = True) -> dict:
    """The phone drives the car through the serial protocol."""
    obs, _ = env.reset(seed=seed)
    sim, link = SimPhone(env), SimEsp32(env, has_encoder=has_encoder)
    sim.reset()
    if measured_steer:
        phone.max_steer_deg = math.degrees(env.car.p.max_steer_rad)
    gaps = Gaps(env.cfg.obs)
    did = {CAPPED: 0, BRAKED: 0, UNSTUCK: 0}
    lost = 0
    start = True
    while True:
        mine = phone.observe(decode_telemetry(link.readline()), sim.image(), sim.pose(),
                             sim.pin(), sim.now, env.dt, start)
        start = False
        gaps.add(mine, obs)
        link.write(phone.act(mine))
        if phone.did in did:
            did[phone.did] += 1
        terminated, truncated = link.step()
        obs = link.obs
        sim.advance()
        lost += not env.odometry.tracking
        if terminated or truncated:
            n = max(env.steps, 1)
            return _row(env, phone, seed, link.last_info, gaps, dict(
                shield_capped_frac=did[CAPPED] / n, shield_braked_frac=did[BRAKED] / n,
                shield_unstuck_frac=did[UNSTUCK] / n, lost_steps=lost))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--onnx", required=True)
    p.add_argument("--env", required=True, help="the model's env config, shield section included")
    p.add_argument("--mode", choices=("shadow", "drive"), required=True)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-encoder", action="store_true", help="speed from ARCore's pose")
    p.add_argument("--nominal-steer", action="store_true",
                   help="map full lock onto the config's steering limit, not the car's own")
    p.add_argument("--speed-window", type=float, default=0.2)
    args = p.parse_args(argv)

    cfg = load_env_config(args.env)
    cfg.goal.arrival_from_odometry = True  # the phone judges arrival from its own pose
    env_cfg = copy.deepcopy(cfg)
    env_cfg.shield.enabled = cfg.shield.enabled and args.mode == "shadow"
    env = CarEnv(env_cfg)
    policy = OnnxPolicy(args.onnx)
    phone = PhoneController(cfg, policy, speed_window=args.speed_window)
    kwargs = dict(has_encoder=not args.no_encoder, measured_steer=not args.nominal_steer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for seed in range(args.seed_start, args.seed_start + args.n):
            if args.mode == "shadow":
                row = shadow(env, phone, policy, seed, **kwargs)
            else:
                row = drive(env, phone, seed, **kwargs)
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
