"""Hand-authored adversarial scenarios - the Phase 1 scorecard.

Random arenas measure average competence. They do not prove the one requirement that
actually matters here: *the car must never stay stuck*. A policy can score beautifully
on random maps and still sit nose-first in a dead end forever, because dead ends are
rare in a random sample and the average washes them out.

Each scenario is fixed geometry with a fixed start pose, so the result is a per-trap
success rate rather than a mean. That table is the artifact to put in front of a grader.

Success requires all three: no collision, never declared stuck, and at least
`min_escape_m` of straight-line distance from the start. The distance clause is what
stops "sat perfectly still and never crashed" from counting as a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import load_env_config
from ..envs.car_env import CarEnv, EnvConfig
from ..envs.randomize import DomainRandConfig
from ..world.generators import rect_walls
from ..world.geometry import World


class MovingObstacleWorld(World):
    """A world with one obstacle that slides across the car's path.

    `CarEnv` calls `update()` each step when the world defines it. Everything else about
    the world stays static, which keeps the fast path in `geometry.py` untouched.
    """

    def __init__(self, segments, circle, amplitude: float, period: float, bounds):
        super().__init__(segments=segments, circles=[circle], bounds=bounds)
        self._base_y = float(circle[1])
        self.amplitude = amplitude
        self.period = period

    def update(self, t: float) -> None:
        y = self._base_y + self.amplitude * math.sin(2.0 * math.pi * t / self.period)
        self.circles[0, 1] = y
        self.circ_c[0, 1] = y


@dataclass
class Scenario:
    name: str
    description: str
    pose: tuple[float, float, float]
    min_escape_m: float
    max_steps: int = 900
    world: World | None = None  # None means "generate one from the env's arena config"
    overrides: dict = field(default_factory=dict)


def _outer(w: float, h: float) -> list[list[float]]:
    return rect_walls(-w / 2, -h / 2, w / 2, h / 2)


def dead_end_corridor() -> Scenario:
    # A 1.2 m corridor capped at the far end. The car starts inside, facing the cap,
    # and the only way out is backwards.
    walls = _outer(18.0, 12.0)
    walls += [
        [0.0, 0.6, 6.0, 0.6],
        [0.0, -0.6, 6.0, -0.6],
        [6.0, -0.6, 6.0, 0.6],  # the cap
    ]
    return Scenario(
        name="dead_end_corridor",
        description="Nose-first into a capped corridor; the only exit is in reverse.",
        world=World(walls, bounds=(-9.0, -6.0, 9.0, 6.0)),
        pose=(4.5, 0.0, 0.0),
        min_escape_m=5.0,
    )


def u_trap() -> Scenario:
    # Three walls. Wide enough to drive into, too tight to turn around in.
    walls = _outer(18.0, 12.0)
    walls += [
        [3.0, -2.0, 3.0, 2.0],
        [0.0, 2.0, 3.0, 2.0],
        [0.0, -2.0, 3.0, -2.0],
    ]
    return Scenario(
        name="u_trap",
        description="Driven into a U; must back out or three-point turn.",
        world=World(walls, bounds=(-9.0, -6.0, 9.0, 6.0)),
        pose=(2.0, 0.0, 0.0),
        min_escape_m=5.0,
    )


def tight_doorway() -> Scenario:
    walls = _outer(18.0, 12.0)
    walls += [
        [3.0, -6.0, 3.0, -0.45],
        [3.0, 0.45, 3.0, 6.0],  # a 0.90 m gap for a 0.20 m wide car
    ]
    return Scenario(
        name="tight_doorway",
        description="A 0.9 m opening that has to be threaded, not barged.",
        world=World(walls, bounds=(-9.0, -6.0, 9.0, 6.0)),
        pose=(-3.0, 0.0, 0.0),
        min_escape_m=7.0,
    )


def box_canyon() -> Scenario:
    # A funnel that narrows below the car's width. Looks passable until it is not,
    # which is what makes it a better test than a flat wall.
    walls = _outer(18.0, 12.0)
    walls += [
        [0.0, 1.6, 5.0, 0.09],
        [0.0, -1.6, 5.0, -0.09],
        [5.0, -0.09, 5.0, 0.09],
    ]
    return Scenario(
        name="box_canyon",
        description="A funnel that narrows past the car's width; commit and you are wedged.",
        world=World(walls, bounds=(-9.0, -6.0, 9.0, 6.0)),
        pose=(1.0, 0.0, 0.0),
        min_escape_m=4.0,
    )


def moving_obstacle() -> Scenario:
    walls = _outer(18.0, 12.0)
    return Scenario(
        name="moving_obstacle",
        description="An obstacle sliding across the path; a stale depth frame is a crash.",
        world=MovingObstacleWorld(
            segments=walls, circle=[4.0, 0.0, 0.45],
            amplitude=2.0, period=4.0, bounds=(-9.0, -6.0, 9.0, 6.0),
        ),
        pose=(-6.0, 0.0, 0.0),
        min_escape_m=9.0,
    )


def sensor_blackout() -> Scenario:
    # Depth forced fully invalid: ultrasonics only. This is the real motion-stereo
    # failure on a phone with no ToF sensor, not a hypothetical.
    return Scenario(
        name="sensor_blackout",
        description="Depth 100% invalid; must survive on ultrasonics alone.",
        world=None,
        pose=(0.0, 0.0, 0.0),
        min_escape_m=4.0,
        overrides={"depth_dropout": 1.0},
    )


def all_scenarios() -> list[Scenario]:
    return [
        dead_end_corridor(),
        u_trap(),
        tight_doorway(),
        box_canyon(),
        moving_obstacle(),
        sensor_blackout(),
    ]


def build_env(scenario: Scenario, base: EnvConfig) -> CarEnv:
    """One env per scenario, with randomization pinned so the trap is the only variable."""
    cfg = replace(base, domain_rand=DomainRandConfig(enabled=False), max_steps=scenario.max_steps)
    if "depth_dropout" in scenario.overrides:
        cfg = replace(cfg, depth=replace(
            cfg.depth,
            dropout_prob=scenario.overrides["depth_dropout"],
            stationary_dropout=0.0,  # already fully dropped; do not double-count
        ))
    return CarEnv(cfg)


def run_scenario(policy, scenario: Scenario, base: EnvConfig,
                 n_episodes: int = 10, seed: int = 2_000_000) -> dict[str, float]:
    env = build_env(scenario, base)
    passed = collided = stuck = 0
    escapes: list[float] = []

    for i in range(n_episodes):
        options = {"pose": scenario.pose}
        if scenario.world is not None:
            options["world"] = scenario.world
        obs, _ = env.reset(seed=seed + i, options=options)
        start = np.array(scenario.pose[:2])
        metrics: dict = {}

        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                metrics = info.get("episode_metrics", {})
                break

        escaped = float(np.linalg.norm(np.array([env.car.state.x, env.car.state.y]) - start))
        escapes.append(escaped)
        crashed = bool(metrics.get("collided", 0.0))
        wedged = bool(metrics.get("stuck", 0.0))
        collided += crashed
        stuck += wedged
        passed += int(not crashed and not wedged and escaped >= scenario.min_escape_m)

    env.close()
    n = max(n_episodes, 1)
    return {
        "success_rate": passed / n,
        "collision_rate": collided / n,
        "stuck_rate": stuck / n,
        "mean_escape_m": float(np.mean(escapes)),
        "required_escape_m": scenario.min_escape_m,
    }


def run_suite(policy, config_path: str | None = None, n_episodes: int = 10,
              seed: int = 2_000_000) -> dict[str, dict[str, float]]:
    base = load_env_config(config_path)
    return {s.name: run_scenario(policy, s, base, n_episodes, seed) for s in all_scenarios()}


def format_suite(results: dict[str, dict[str, float]]) -> str:
    head = f"{'scenario':<20} {'success':>8} {'crash':>8} {'stuck':>8} {'escape m':>10}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lines.append(
            f"{name:<20} {r['success_rate']:>7.0%} {r['collision_rate']:>8.0%} "
            f"{r['stuck_rate']:>8.0%} {r['mean_escape_m']:>7.2f}/{r['required_escape_m']:.0f}"
        )
    overall = float(np.mean([r["success_rate"] for r in results.values()])) if results else 0.0
    lines += ["-" * len(head), f"{'OVERALL':<20} {overall:>7.0%}"]
    return "\n".join(lines)
