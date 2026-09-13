"""Episode rollouts and the Phase 1 scorecard.

Phase 1 has exactly two failure modes and they need separate numbers. A policy that
crawls into a corner and parks has a perfect collision rate, and a policy that sprints
into walls never gets stuck. `success_rate` counts only episodes that did neither.

Neither failure catches a policy that circles an open patch for the whole episode, which
never crashes and never sticks: `phase1_v2` scored 90 % that way at 11M steps.
`clean_coverage_m2` does, because it counts only floor covered in episodes that did not fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


class ConstantPolicy:
    """Fixed action every step. Used as a baseline and by the reward-hacking guards."""

    def __init__(self, steer: float, throttle: float):
        self.action = np.array([steer, throttle], dtype=np.float32)

    def predict(self, obs, deterministic: bool = True):
        return self.action, None


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32), None


@dataclass
class EvalResult:
    episodes: int = 0
    success_rate: float = 0.0  # neither crashed nor got stuck
    collision_rate: float = 0.0
    stuck_rate: float = 0.0
    mean_return: float = 0.0
    mean_distance_m: float = 0.0
    mean_coverage_m2: float = 0.0  # path length can be inflated by circling; this cannot
    clean_coverage_m2: float = 0.0  # the same, counting crashed and stuck episodes as zero
    mean_speed_mps: float = 0.0
    mean_min_clearance_m: float = 0.0
    mean_reverse_frac: float = 0.0
    mean_retrace_frac: float = 0.0  # share of the swath driven over ground already covered
    mean_lock_frac: float = 0.0  # share of steps driving forward at near-full steering lock
    mean_steps: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"episodes {self.episodes:4d}  success {self.success_rate:6.1%}  "
            f"collision {self.collision_rate:6.1%}  stuck {self.stuck_rate:6.1%}  "
            f"dist {self.mean_distance_m:6.2f} m  cover {self.mean_coverage_m2:6.1f} m2  "
            f"clean {self.clean_coverage_m2:5.1f} m2  speed {self.mean_speed_mps:5.2f} m/s  "
            f"reverse {self.mean_reverse_frac:5.1%}  retrace {self.mean_retrace_frac:5.1%}  "
            f"lock {self.mean_lock_frac:5.1%}"
        )


def run_episodes(
    policy: Policy,
    env,
    n_episodes: int = 20,
    seed: int = 0,
    deterministic: bool = True,
    render: bool = False,
    frame_sink: list | None = None,
) -> EvalResult:
    """Roll out `n_episodes` and summarize. Seeds are consecutive from `seed`, so the
    same call reproduces the same arenas - which is what makes two checkpoints
    comparable."""
    rows: list[dict[str, float]] = []
    returns: list[float] = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        metrics: dict[str, float] = {}
        while True:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            if render:
                frame = env.render()
                if frame_sink is not None and frame is not None:
                    frame_sink.append(frame)
            if terminated or truncated:
                metrics = info.get("episode_metrics", {})
                break
        rows.append(metrics)
        returns.append(total)

    if not rows:
        return EvalResult()

    def mean(key: str) -> float:
        vals = [r.get(key, 0.0) for r in rows]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else 0.0

    collided = np.array([r.get("collided", 0.0) for r in rows])
    stuck = np.array([r.get("stuck", 0.0) for r in rows])
    clean = (collided == 0.0) & (stuck == 0.0)
    coverage = np.array([r.get("coverage_m2", 0.0) for r in rows])

    return EvalResult(
        episodes=len(rows),
        success_rate=float(np.mean(clean)),
        collision_rate=float(np.mean(collided)),
        stuck_rate=float(np.mean(stuck)),
        mean_return=float(np.mean(returns)),
        mean_distance_m=mean("distance_m"),
        mean_coverage_m2=mean("coverage_m2"),
        clean_coverage_m2=float(np.mean(np.where(clean, coverage, 0.0))),
        mean_speed_mps=mean("mean_speed_mps"),
        mean_min_clearance_m=mean("min_clearance_m"),
        mean_reverse_frac=mean("reverse_frac"),
        mean_retrace_frac=mean("retrace_frac"),
        mean_lock_frac=mean("lock_frac"),
        mean_steps=mean("steps"),
    )


# --- CLI ---------------------------------------------------------------------

def _load_policy(path: str | None, seed: int = 0):
    if path is None:
        return RandomPolicy(seed)
    from stable_baselines3 import PPO

    return PPO.load(path, device="cpu")


def main(argv=None) -> None:
    import argparse

    from ..config import load_env_config
    from ..envs.car_env import CarEnv

    p = argparse.ArgumentParser(description="Evaluate a Phase 1 policy.")
    p.add_argument("--model", default=None, help="path to a saved PPO model; omit for random")
    p.add_argument("--config", default="configs/env_phase1.yaml")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1_000_000)
    p.add_argument("--render", choices=["human", "rgb_array"], default=None)
    p.add_argument("--scenarios", action="store_true",
                   help="run the adversarial suite instead of random arenas")
    p.add_argument("--video", default=None, help="write an mp4 here (implies rgb_array)")
    args = p.parse_args(argv)

    policy = _load_policy(args.model, args.seed)

    if args.scenarios:
        from .scenarios import format_suite, run_suite

        print(format_suite(run_suite(policy, args.config, args.episodes, args.seed)))
        return

    mode = "rgb_array" if args.video else args.render
    env = CarEnv(load_env_config(args.config), render_mode=mode)
    frames: list = [] if args.video else None
    result = run_episodes(
        policy, env, args.episodes, seed=args.seed, deterministic=True,
        render=mode is not None, frame_sink=frames,
    )
    print(result)

    if args.video and frames:
        import imageio.v2 as imageio

        imageio.mimsave(args.video, frames, fps=30, macro_block_size=1)
        print(f"wrote {args.video} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
