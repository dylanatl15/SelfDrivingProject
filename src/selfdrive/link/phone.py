"""The phone's half of the control loop, written the way the Android app has to run it.

Each control step the phone reads a `$T` frame, the newest depth image and ARCore's pose, builds
the policy input, runs the policy, shields the throttle and writes a `$C` frame. This class does
exactly that and reads nothing else: no true pose, no true speed, no randomized car parameters.
`CarEnv` builds the same input with the whole simulator at hand, so an input the phone cannot
build the same way shows up as a difference. `link/phone_check.py` measures those differences
step by step.

Where each input comes from on the phone:

    depth buckets       the depth image, metres, bucket 0 rightmost   envs/obs.py
    depth confidence    speed: clip(|v| / depth.speed_ref, 0, 1)      sensors/depth_arc.py
    ultrasonics         $T us_*, the -1.000 sentinel read as 4.0 m    docs/shield.md
    speed               $T v_mps, or ARCore's pose without an encoder
    steering angle      $T steer_deg
    last commands       what the previous $C sent, after the shield   docs/shield.md
    obstacle memory     depth image, $T and ARCore's pose             docs/memory-ring.md
    goal block, clock   ARCore's pose and the pin                     docs/goal-block.md

Every constant comes from the model's environment config, never from a randomized episode,
because the phone does not know those either. The steering command is the one exception: the
angle `$C` carries is the policy's fraction of the car's real steering limit, which the team has
to measure (`max_steer_deg`).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..dynamics.base import wrap_angle
from ..envs.car_env import STEER, THROTTLE, EnvConfig
from ..envs.goals import GoalPatience
from ..envs.memory import EgoMemory
from ..envs.obs import ObservationBuilder
from ..envs.shield import PASSED, Unstick, reported_distances
from ..sensors.ultrasonic import UltrasonicArray
from .protocol import FLAG_NO_ENCODER, Command, Telemetry, encode_command


@dataclass
class ArPose:
    """An ARCore camera pose: metres and a unit quaternion in ARCore's right-handed, +Y-up world
    frame, as `Pose.tx()` ... `Pose.qw()` return them, and whether ARCore is tracking."""

    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float
    tracking: bool = True

    def floor_pose(self, mount_forward: float) -> tuple[float, float, float]:
        """The car's pose on the floor, per docs/goal-block.md: x = X, y = -Z, heading
        left-positive, at the car's centre `mount_forward` metres behind the camera."""
        x, y, z, w = self.qx, self.qy, self.qz, self.qw
        # The lens looks along local -Z. Rotated by the pose, that is minus R's third column.
        fx = -2.0 * (x * z + w * y)
        fz = -(1.0 - 2.0 * (x * x + y * y))
        theta = math.atan2(-fz, fx)
        return (self.tx - mount_forward * math.cos(theta),
                -self.tz - mount_forward * math.sin(theta), theta)


@dataclass
class DepthImage:
    """One depth image as the phone's depth pipeline hands it over, already cut into buckets."""

    ranges: np.ndarray  # metres per bucket, bucket 0 the rightmost
    fresh: np.ndarray  # per bucket: new since the last step, with at least one valid pixel
    fov: float  # horizontal field of view, radians
    stamp: float  # capture time, seconds on the control clock
    pose: ArPose  # the camera pose at capture


class OnnxPolicy:
    """The exported policy (`export/to_onnx.py`): one observation in, [steer, throttle] out."""

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"observation": np.asarray(obs, np.float32)[None]})[0][0]


def _to_floor(points: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    """Car-frame points (rows of forward, left) to the floor frame of the pose (x, y, theta)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.column_stack([x + c * points[:, 0] - s * points[:, 1],
                            y + s * points[:, 0] + c * points[:, 1]])


class PhoneController:
    """Telemetry, depth and pose in; a `$C` frame out. Call `observe`, then `act`, every step."""

    def __init__(self, cfg: EnvConfig, policy: Callable[[np.ndarray], np.ndarray],
                 max_steer_deg: float | None = None, speed_window: float = 0.2):
        self.cfg = cfg
        self.policy = policy
        # The steering limit the ESP32 maps a full-lock command onto. Until the car is measured,
        # the config's nominal value.
        self.max_steer_deg = (math.degrees(cfg.car.max_steer_rad) if max_steer_deg is None
                              else max_steer_deg)
        self.speed_window = speed_window  # s of pose history a speed estimate spans
        o = cfg.obs
        self.builder = ObservationBuilder(o)
        sensors = UltrasonicArray(cfg.ultrasonic)  # nominal mounts and names; draws nothing
        self.names = sensors.names
        self._mounts = sensors.mounts
        a = sensors.angles
        self._dirs = np.column_stack([np.cos(a), np.sin(a)])
        self.memory = (EgoMemory(o.memory_sectors, o.memory_seconds, o.norm_memory_max,
                                 points_per_step=cfg.depth.n_buckets + sensors.n)
                       if o.memory_sectors else None)
        self.patience = GoalPatience(o.patience_gain) if o.goal_patience else None
        self.seq = 0
        self.sent = np.zeros(2)
        self.did = PASSED

    # --- building the input --------------------------------------------------

    def observe(self, t: Telemetry, image: DepthImage, pose: ArPose,
                pin: tuple[float, float, float] | None, now: float, dt: float,
                start: bool = False) -> np.ndarray:
        """The policy input for this step. `pin` is the goal in ARCore world coordinates, `now`
        the control clock in seconds and `dt` the control period. `start` begins a drive."""
        c, o = self.cfg, self.cfg.obs
        x, y, theta = pose.floor_pose(c.depth.mount_forward)
        if start:
            self._begin(dt, (x, y, theta))
        if pose.tracking:
            self._tracked = (x, y, theta)

        raw = np.array([getattr(t, f"us_{name}_m") for name in self.names])
        ultra = np.where(raw < 0.0, c.ultrasonic.max_range, raw)
        speed = self._speed(t, x, y, theta, now, pose.tracking)
        depth = np.asarray(image.ranges, dtype=float)
        self._depth, self._ultra, self._v = depth, ultra, speed  # what the shield reads

        # A 3-sensor build pads the slot after its three readings, as CarEnv does.
        slots = np.concatenate([ultra, np.full(o.n_ultrasonic - len(ultra),
                                               c.ultrasonic.max_range)])
        frame = self.builder.frame(
            depth=depth,
            depth_confidence=min(abs(speed) / c.depth.speed_ref, 1.0),
            ultrasonic=slots,
            speed=speed,
            steer=math.radians(t.steer_deg),
            last_throttle=float(self.sent[THROTTLE]),
            last_steer=float(self.sent[STEER]),
        )
        ring = (self._remember(image, raw, x, y, theta, now, pose.tracking)
                if self.memory is not None else None)
        goal = self._goal(pin, dt, start) if o.goal_block else None
        if start:
            return self.builder.reset(frame, ring, goal)
        return self.builder.push(frame, ring, goal)

    def _begin(self, dt: float, pose: tuple[float, float, float]) -> None:
        if self.memory is not None:
            self.memory.reset(dt)
        self._unstick = Unstick(self.cfg.shield, dt)
        self.sent = np.zeros(2)
        self.did = PASSED
        self._tracked = pose
        self._prev_raw: np.ndarray | None = None
        self._poses: deque[tuple[float, float, float]] = deque()
        self._v_estimate = 0.0
        self._pin: tuple[float, float] | None = None
        self.goals_changed = 0  # pins the app moved on from
        self.arrivals_agreed = 0  # ...of which the phone's own 0.5 m rule had judged arrived
        self.arrival_only_steps = 0  # steps the rule judged arrived with no new pin

    def _speed(self, t: Telemetry, x: float, y: float, theta: float, now: float,
               tracking: bool) -> float:
        """The telemetry speed, or without an encoder the pose's motion along the heading over
        the last `speed_window` seconds. While tracking is lost the last estimate holds."""
        if not t.flags & FLAG_NO_ENCODER:
            return t.speed_mps
        if not tracking:
            self._poses.clear()
            return self._v_estimate
        self._poses.append((now, x, y))
        while len(self._poses) > 1 and now - self._poses[1][0] >= self.speed_window:
            self._poses.popleft()
        t0, x0, y0 = self._poses[0]
        if now > t0:
            along = (x - x0) * math.cos(theta) + (y - y0) * math.sin(theta)
            self._v_estimate = along / (now - t0)
        return self._v_estimate

    def _remember(self, image: DepthImage, raw: np.ndarray, x: float, y: float, theta: float,
                  now: float, tracking: bool) -> np.ndarray:
        """Store this step's new obstacle points (docs/memory-ring.md) and return the ring."""
        c, o, m = self.cfg, self.cfg.obs, self.memory
        changed = (np.ones(len(raw), dtype=bool) if self._prev_raw is None
                   else raw != self._prev_raw)
        self._prev_raw = raw
        if not tracking:
            m.clear()  # nothing stored relates to the pose tracking comes back with
            m.insert(np.empty((0, 2)), np.empty(0))
            return m.ring(x, y, theta, now)

        d = np.asarray(image.ranges, dtype=float)
        n = len(d)
        a = -image.fov / 2.0 + image.fov / n * (np.arange(n) + 0.5)
        keep = np.asarray(image.fresh, dtype=bool) & (d < o.norm_depth_max)
        seen = np.column_stack([c.depth.mount_forward + d[keep] * np.cos(a[keep]),
                                d[keep] * np.sin(a[keep])])
        heard = changed & (raw >= 0.0) & (raw < o.norm_ultra_max)
        felt = self._mounts[heard] + raw[heard, None] * self._dirs[heard]
        # Depth is placed from the pose its image was captured at, ultrasonics from the current one.
        cx, cy, ctheta = image.pose.floor_pose(c.depth.mount_forward)
        m.insert(np.concatenate([_to_floor(seen, cx, cy, ctheta), _to_floor(felt, x, y, theta)]),
                 np.concatenate([np.full(len(seen), image.stamp), np.full(len(felt), now)]))
        return m.ring(x, y, theta, now)

    def _goal(self, pin: tuple[float, float, float] | None, dt: float,
              start: bool) -> np.ndarray:
        """The goal block (docs/goal-block.md). While tracking is lost it is computed from the
        last tracked pose, so it holds, and the clock keeps counting against that range."""
        x, y, theta = self._tracked
        radius = self.cfg.goal.radius
        floor = None if pin is None else (float(pin[0]), -float(pin[2]))

        arrived = self._pin is not None and math.hypot(self._pin[0] - x, self._pin[1] - y) <= radius
        new = start or floor != self._pin
        if new and not start:
            self.goals_changed += 1
            self.arrivals_agreed += arrived
        elif arrived:
            self.arrival_only_steps += 1
        self._pin = floor

        if floor is None:
            r, b = math.inf, 0.0
        else:
            dx, dy = floor[0] - x, floor[1] - y
            r, b = math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - theta)
        waited = None
        if self.patience is not None:
            if new:
                self.patience.reset(r)
            else:
                self.patience.update(r, dt)
            waited = self.patience.seconds
        return self.builder.goal(r, b, waited)

    # --- acting ----------------------------------------------------------------

    def act(self, obs: np.ndarray) -> bytes:
        """Run the policy on `obs`, shield its command (docs/shield.md) and encode the `$C` frame.
        The values sent become the next input's last commands."""
        action = np.clip(np.asarray(self.policy(obs), dtype=np.float32).reshape(2), -1.0, 1.0)
        steer, throttle = float(action[STEER]), float(action[THROTTLE])
        self.did = PASSED
        if self.cfg.shield.enabled:
            forward, back = reported_distances(self._depth, self._ultra, self.names)
            car = self.cfg.car
            steer, throttle, self.did = self._unstick.step(
                steer, throttle, self._v, forward, back, car.max_speed_fwd, car.max_speed_rev)
        self.sent = np.zeros(2)
        self.sent[STEER], self.sent[THROTTLE] = steer, throttle
        frame = encode_command(Command(self.seq, steer * self.max_steer_deg, throttle))
        self.seq = (self.seq + 1) & 0xFFFF
        return frame
