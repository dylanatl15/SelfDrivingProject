"""Phase 1 Gymnasium environment: drive forward, do not crash, never stay stuck.

Design notes worth knowing before changing anything here:

*   `render_mode` defaults to `None` and pygame is imported lazily inside `render()`, so
    the 20 training workers never touch it. Visualisation is an evaluation activity.
*   Rewards read ground-truth geometry while observations read noisy sensors. That split
    is intentional: reward computed from corrupted sensors trains the policy to chase its
    own sensor artifacts, and privileged information at training time is free.
    The obstacle memory, when enabled, is an observation too: it places points using an
    odometry estimate that drifts, never the true pose. So is the goal block, when enabled:
    range and bearing from that same estimate. Goal progress and arrival, being reward,
    use the true pose.
*   Every episode resamples domain randomization from `self.np_random`, so a given seed
    reproduces the whole episode - arena, car, noise - exactly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..dynamics.base import CarParams
from ..dynamics.bicycle import KinematicBicycle
from ..sensors.depth_arc import DepthArc, DepthArcParams
from ..sensors.odometry import Odometry, OdometryParams
from ..sensors.ultrasonic import UltrasonicArray, UltrasonicParams
from ..world.generators import ArenaParams, make_arena, sample_spawn
from .goals import GoalConfig, GoalTracker
from .memory import EgoMemory
from .obs import ObsConfig, ObservationBuilder
from .randomize import DomainRandConfig
from .rewards import RewardConfig, RewardFunction

# Action layout. Kept as constants because the Android app indexes the same order.
STEER = 0
THROTTLE = 1

# With goals, spawns drawn per arena before the arena itself is drawn again.
SPAWN_TRIES = 20
ARENA_TRIES = 5


@dataclass
class EnvConfig:
    car: CarParams = field(default_factory=CarParams)
    depth: DepthArcParams = field(default_factory=DepthArcParams)
    ultrasonic: UltrasonicParams = field(default_factory=UltrasonicParams)
    odometry: OdometryParams = field(default_factory=OdometryParams)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    arena: ArenaParams = field(default_factory=lambda: ArenaParams(kind="random"))
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)  # read only with obs.goal_block
    dt: float = 1.0 / 30.0
    max_steps: int = 1500  # ~50 s at 30 Hz


def _to_world(body: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    """Car-frame points (rows of forward, left) to the frame the pose is expressed in."""
    c, s = math.cos(theta), math.sin(theta)
    out = np.empty_like(body)
    out[:, 0] = x + c * body[:, 0] - s * body[:, 1]
    out[:, 1] = y + s * body[:, 0] + c * body[:, 1]
    return out


class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self._viewer = None  # built on first render(), never in a training worker
        self.hud_overlay: list[str] = []  # extra viewport lines, set by eval/watch.py

        self.obs_builder = ObservationBuilder(self.cfg.obs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32
        )

        # Populated by reset(); declared here so attribute errors surface early.
        self.world = None
        self.car: KinematicBicycle | None = None
        self.depth: DepthArc | None = None
        self.ultra: UltrasonicArray | None = None
        self.reward_fn = RewardFunction(self.cfg.reward)
        self.dt = self.cfg.dt
        self.steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._episode: dict[str, float] = {}

        # Obstacle memory exists only when the observation has a slot for it.
        o = self.cfg.obs
        self.memory: EgoMemory | None = None
        self.odometry: Odometry | None = None
        if o.memory_sectors:
            self.memory = EgoMemory(
                o.memory_sectors, o.memory_seconds, o.norm_memory_max,
                points_per_step=self.cfg.depth.n_buckets + self.cfg.ultrasonic.n_sensors,
            )
        # Likewise goals, which also need the odometry estimate the observation reads.
        self.goals: GoalTracker | None = GoalTracker(self.cfg.goal) if o.goal_block else None

    # --- episode setup -------------------------------------------------------

    def _apply_randomization(self) -> dict[str, float] | None:
        c = self.cfg
        d = c.domain_rand.sample(self.np_random)

        if d is None:
            # Randomization off: run exactly the car the config describes.
            self.dt = c.dt
            self.car = KinematicBicycle(c.car)
            self.depth = DepthArc(c.depth)
            self.ultra = UltrasonicArray(c.ultrasonic)
            return None

        self.dt = d["dt"]
        car = replace(
            c.car,
            wheelbase=c.car.wheelbase * d["wheelbase_scale"],
            accel_tau=d["accel_tau"],
            steer_rate_rad_s=d["steer_rate_rad_s"],
            steer_trim_rad=d["steer_trim_rad"],
            max_steer_rad=d["max_steer_rad"],
            throttle_deadband=d["throttle_deadband"],
            throttle_gain=d["throttle_gain"],
            understeer=d["understeer"],
            max_speed_fwd=d["max_speed_fwd"],
        )
        depth = replace(
            c.depth,
            fov_deg=d["depth_fov_deg"],
            max_range=d["depth_max_range"],
            noise_frac=d["depth_noise_frac"],
            dropout_prob=d["depth_dropout"],
            stationary_dropout=d["depth_stationary_dropout"],
            latency_steps=d["depth_latency_steps"],
        )
        ultra = replace(
            c.ultrasonic,
            noise_m=d["ultra_noise_m"],
            dropout_prob=d["ultra_dropout"],
            update_hz=d["ultra_update_hz"],
        )

        self.car = KinematicBicycle(car)
        self.depth = DepthArc(depth)
        self.ultra = UltrasonicArray(ultra)
        return d

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Options: `world` pins the arena, `pose` the start pose and `goal` the first goal.

        They exist for the hand-authored evaluation scenarios (dead-end corridor, U-trap
        and friends), which have to place the car in an exact spot to test anything, and
        for tests that need a known starting condition. Goals after the first are drawn.
        """
        super().reset(seed=seed)
        self._apply_randomization()
        options = options or {}

        self.world, (x, y, theta) = self._place(options)

        self.car.reset(x, y, theta)
        self.depth.reset(self.np_random)
        self.ultra.reset(self.np_random)
        self.reward_fn.reset(x, y, theta, self.world.bounds)
        if self.memory is not None or self.goals is not None:
            self._reset_odometry(x, y, theta)
        if self.memory is not None:
            self._reset_memory(x, y, theta)
        if self.goals is not None:
            # A child generator of its own, spawned after odometry's for the same reason.
            self._goal_rng = self.np_random.spawn(1)[0]
            self.goals.reset(self.world, x, y, self._goal_rng, pin=options.get("goal"))
        self.steps = 0
        self._last_action[:] = 0.0
        self._episode = {
            "distance": 0.0,
            "speed_sum": 0.0,
            "min_clearance": math.inf,
            "stall_steps": 0.0,
            "reverse_steps": 0.0,
            "backing_steps": 0.0,
            "lock_steps": 0.0,
            "collided": 0.0,
        }

        obs = self.obs_builder.reset(*self._frame())
        return obs, {"arena_primitives": self.world.n_primitives}

    def _place(self, options: dict):
        """The arena and the start pose.

        With goals, the car never starts in a sealed pocket of floor: a spawn off the
        largest connected stretch is drawn again, and an arena that offers none is replaced.
        Without goals this draws exactly one arena and one spawn, as it always has, so
        Phase 1 episodes are unchanged.
        """
        pose = options.get("pose")
        tries = SPAWN_TRIES if self.goals is not None else 1
        for _ in range(ARENA_TRIES):
            world = options.get("world") or make_arena(self.np_random, self.cfg.arena)
            if self.goals is not None:
                self.goals.prepare(world)
            if pose is not None:
                return world, tuple(float(v) for v in pose)
            for _ in range(tries):
                spawn = sample_spawn(world, self.np_random, self.car.p.length,
                                     self.car.p.width, self.cfg.arena.spawn_clearance)
                if self.goals is None or self.goals.connected(spawn[0], spawn[1]):
                    return world, spawn
        return world, spawn  # every arena drawn was badly fragmented: take the last spawn

    def _reset_odometry(self, x: float, y: float, theta: float) -> None:
        # A child generator for everything odometry draws. Spawning one leaves `np_random`
        # untouched, so a seed produces the same arena, car and sensor noise with the memory
        # on as with it off, and the two policies can be compared on identical episodes.
        self._odometry_rng = self.np_random.spawn(1)[0]
        drawn = self.cfg.domain_rand.sample_odometry(self._odometry_rng)
        params = self.cfg.odometry if drawn is None else replace(self.cfg.odometry, **drawn)
        self.odometry = Odometry(params)
        self.odometry.reset(x, y, theta)

    def _reset_memory(self, x: float, y: float, theta: float) -> None:
        self.memory.reset(self.dt)

        # Sensor geometry in the car's frame, fixed once randomization has drawn the FOV.
        a = self.depth.bucket_angles
        self._depth_dirs = np.column_stack([np.cos(a), np.sin(a)])
        self._depth_mount = np.array([self.depth.p.mount_forward, 0.0])
        a = self.ultra.angles
        self._ultra_dirs = np.column_stack([np.cos(a), np.sin(a)])
        self._ultra_mounts = self.ultra.mounts
        self._ultra_last = np.full(self.ultra.n, np.nan)
        lag = self.depth.p.latency_steps + 1
        self._capture_poses = deque([(float(x), float(y), float(theta), 0.0)] * lag, maxlen=lag)

    # --- stepping ------------------------------------------------------------

    def _frame(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Sample the sensors once; return this step's frame, memory ring and goal block."""
        s = self.car.state
        depth = self.depth.sample(self.world, s, self.dt, self.np_random)
        ultra = self.ultra.sample(self.world, s, self.dt, self.np_random)
        ring = self._remember(depth, ultra) if self.memory is not None else None
        if self.ultra.n < self.cfg.obs.n_ultrasonic:
            # A 3-sensor build still has to fill a fixed-width observation slot.
            ultra = np.concatenate([ultra, np.full(self.cfg.obs.n_ultrasonic - self.ultra.n,
                                                   self.ultra.p.max_range)])
        frame = self.obs_builder.frame(
            depth=depth,
            depth_confidence=self.depth.confidence,
            ultrasonic=ultra,
            speed=s.speed,
            steer=s.steer,
            last_throttle=float(self._last_action[THROTTLE]),
            last_steer=float(self._last_action[STEER]),
        )
        goal = None
        if self.goals is not None:
            odo = self.odometry
            goal = self.obs_builder.goal(*self.goals.vector(odo.x, odo.y, odo.theta))
        return frame, ring, goal

    def _remember(self, depth: np.ndarray, ultra: np.ndarray) -> np.ndarray:
        """Store this step's new obstacle readings and return the ring the policy sees."""
        odo, now = self.odometry, self.steps * self.dt
        # Depth arrives `latency_steps` late, so it is placed from the pose it was captured at.
        self._capture_poses.append((odo.x, odo.y, odo.theta, now))
        cx, cy, ctheta, captured = self._capture_poses[0]

        # Only new measurements are stored: a held reading was taken from an earlier pose,
        # and storing it again at this one smears the obstacle along the car's path. For an
        # ultrasonic, "new" means "changed" - the one test the phone can apply to $T
        # telemetry, where a sensor waiting for its round-robin turn repeats its last value.
        changed = ultra != self._ultra_last
        self._ultra_last = ultra.copy()

        if odo.tracking:
            # Cut at the observation's ranges, which are the only ranges the phone knows.
            o = self.cfg.obs
            d_ok = self.depth.fresh & (depth < o.norm_depth_max)
            u_ok = changed & self.ultra.echo & (ultra < o.norm_ultra_max)
            seen = self._depth_mount + depth[d_ok, None] * self._depth_dirs[d_ok]
            felt = self._ultra_mounts[u_ok] + ultra[u_ok, None] * self._ultra_dirs[u_ok]
            points = np.concatenate([_to_world(seen, cx, cy, ctheta),
                                     _to_world(felt, odo.x, odo.y, odo.theta)])
            stamps = np.concatenate([np.full(len(seen), captured), np.full(len(felt), now)])
        else:
            points, stamps = np.empty((0, 2)), np.empty(0)
        self.memory.insert(points, stamps)
        return self.memory.ring(odo.x, odo.y, odo.theta, now)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_steer_cmd = float(self._last_action[STEER])
        prev_throttle_cmd = float(self._last_action[THROTTLE])
        p = self.car.p

        # Worlds are static unless a scenario says otherwise (the moving-obstacle test
        # defines update(); everything else leaves the geometry fast path untouched).
        world_update = getattr(self.world, "update", None)
        if world_update is not None:
            world_update(self.steps * self.dt)

        before = np.array([self.car.state.x, self.car.state.y])
        state = self.car.step(float(action[STEER]), float(action[THROTTLE]), self.dt)
        after = np.array([state.x, state.y])

        collided = self.world.collides(state.x, state.y, state.theta, p.length, p.width)
        clearance = self.world.clearance(state.x, state.y, state.theta, p.length, p.width)
        progress, reached = 0.0, False
        if self.goals is not None:
            progress, reached = self.goals.update(state.x, state.y, self._goal_rng)

        reward, terms = self.reward_fn(
            x=state.x,
            y=state.y,
            theta=state.theta,
            throttle_cmd=float(action[THROTTLE]),
            prev_throttle_cmd=prev_throttle_cmd,
            steer_cmd=float(action[STEER]),
            prev_steer_cmd=prev_steer_cmd,
            clearance=clearance,
            collided=collided,
            dt=self.dt,
            progress_m=progress,
            reached=reached,
        )

        self._last_action = action
        self.steps += 1
        self._episode["distance"] += float(np.linalg.norm(after - before))
        self._episode["speed_sum"] += abs(state.speed)
        self._episode["min_clearance"] = min(self._episode["min_clearance"], clearance)
        self._episode["stall_steps"] += float(terms.stall != 0.0)
        # A negative throttle command is mostly braking; backing is the car rolling backwards.
        self._episode["reverse_steps"] += float(action[THROTTLE] < 0.0)
        self._episode["backing_steps"] += float(state.speed < -0.05)
        # Forward at near-full steering lock: how phase1_v2 circled open patches.
        self._episode["lock_steps"] += float(abs(action[STEER]) > 0.8 and action[THROTTLE] > 0)
        self._episode["collided"] = float(collided)

        if self.odometry is not None:
            lost = self.odometry.update(state.x, state.y, state.theta, self.dt, self._odometry_rng)
            if lost and self.memory is not None:
                self.memory.clear()  # tracking lost: nothing stored relates to the new pose
        obs = self.obs_builder.push(*self._frame())
        terminated = bool(collided)
        truncated = bool(self.steps >= self.cfg.max_steps or self.reward_fn.is_stalled)

        info: dict = {"reward_terms": terms.as_dict(), "clearance": clearance}
        if reached:
            info["goal_reached"] = True
        if terminated or truncated:
            info["episode_metrics"] = self._summary(truncated)
        return obs, float(reward), terminated, truncated, info

    def _summary(self, truncated: bool) -> dict[str, float]:
        n = max(self.steps, 1)
        out = {
            "distance_m": self._episode["distance"],
            # Path length rewards orbiting an open patch; covered floor area does not.
            "coverage_m2": self.reward_fn.coverage_m2,
            "mean_speed_mps": self._episode["speed_sum"] / n,
            "min_clearance_m": self._episode["min_clearance"],
            "stall_frac": self._episode["stall_steps"] / n,
            "reverse_frac": self._episode["reverse_steps"] / n,
            "backing_frac": self._episode["backing_steps"] / n,
            "lock_frac": self._episode["lock_steps"] / n,
            "retrace_frac": self.reward_fn.retrace_frac,
            "collided": self._episode["collided"],
            # The two ways Phase 1 can fail, separated so evaluation can report them apart.
            "stuck": float(truncated and self.reward_fn.is_stalled),
            "steps": float(self.steps),
        }
        if self.goals is not None:
            out["goals_reached"] = float(self.goals.reached)
            # Net path metres closed across every goal, the one being driven to included.
            out["goal_progress_m"] = self.goals.progress_m
        return out

    # --- rendering -----------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            # Imported here, not at module scope: training workers must never load pygame.
            from ..render.pygame_view import PygameView

            self._viewer = PygameView(self.render_mode, self.metadata["render_fps"])
        return self._viewer.draw(self)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
