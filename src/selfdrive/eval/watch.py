"""Live viewport onto a training run.

    uv run python -m selfdrive.eval.watch                        # newest run under runs/
    uv run python -m selfdrive.eval.watch --run runs/ppo_20260912_163000 --fps 90

Training workers never render (see render/pygame_view.py), so there is no window to
attach to. This runs its own single env in a separate process, drives it with the newest
checkpoint the trainer has written, and swaps to each newer one between episodes. What
you see lags training by at most one checkpoint interval (`checkpoint_every_steps`).

Before the first checkpoint lands it drives a random policy, which is also a useful
baseline to have watched once.

Arenas use seeds from 2_000_000, clear of the training seeds and of PeriodicEval's range,
so the window never shows a map the scorecard is computed on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

CHECKPOINT_RE = re.compile(r"_(\d+)_steps\.zip$")
FINAL = sys.maxsize  # a current final_model.zip outranks every numbered checkpoint
WATCH_SEED_BASE = 2_000_000
SETTLE_S = 2.0  # ignore files modified more recently than this; the trainer may be mid-write


def newest_run(root: Path) -> Path | None:
    """Most recently started run directory, judged by when train_config.json was written."""
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if (p / "train_config.json").is_file()]
    return max(runs, key=lambda p: (p / "train_config.json").stat().st_mtime, default=None)


def newest_checkpoint(run_dir: Path, now: float | None = None) -> tuple[int, Path] | None:
    """`(timesteps, path)` of the latest fully written checkpoint, or None.

    Ranked by the step count in the filename, not by name or mtime: `ppo_10000000_steps`
    sorts before `ppo_2000000_steps` as a string.

    final_model.zip wins only if no checkpoint is newer. A run that was stopped and
    resumed keeps its old final model until it finishes again, and phase1_v5's viewer sat
    on the 1M one for three million steps.
    """
    now = time.time() if now is None else now
    found: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("*_steps.zip"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))

    settled = [(steps, p) for steps, p in found if now - p.stat().st_mtime >= SETTLE_S]
    best = max(settled, key=lambda item: item[0], default=None)
    final = run_dir / "final_model.zip"
    if final.is_file() and now - final.stat().st_mtime >= SETTLE_S:
        if best is None or final.stat().st_mtime >= best[1].stat().st_mtime:
            return FINAL, final
    return best


def describe_checkpoint(steps: int) -> str:
    return "final model" if steps == FINAL else f"checkpoint {steps / 1e6:.1f}M steps"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Watch a training run's newest checkpoint drive.")
    p.add_argument("--run", default=None, help="run directory; default: newest under --runs-root")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--config", default=None, help="env YAML; default: the one the run trained on")
    p.add_argument("--fps", type=int, default=30, help="30 is real time; raise to fast-forward")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions as training does; default is the deployed mean action")
    args = p.parse_args(argv)

    run_dir = Path(args.run) if args.run else newest_run(Path(args.runs_root))
    if run_dir is None or not run_dir.is_dir():
        raise SystemExit(f"no training run found under {args.runs_root}/ - start one first")
    config = args.config or json.loads((run_dir / "train_config.json").read_text())["env_config"]

    import torch

    torch.set_num_threads(1)  # stay off the cores the training workers are using

    from stable_baselines3 import PPO

    from ..config import load_env_config
    from ..envs.car_env import CarEnv
    from .evaluate import RandomPolicy

    env = CarEnv(load_env_config(config), render_mode="human")
    env.metadata = {**env.metadata, "render_fps": args.fps}

    policy = RandomPolicy(seed=0)
    label = "random policy (no checkpoint yet)"
    loaded: Path | None = None
    episode = 0
    print(f"watching {run_dir}  env {config}  - close the window or Ctrl+C to stop")

    try:
        while True:
            latest = newest_checkpoint(run_dir)
            if latest is not None and latest[1] != loaded:
                try:
                    policy = PPO.load(latest[1], device="cpu")
                    loaded = latest[1]
                    label = describe_checkpoint(latest[0])
                    print(f"loaded {label}")
                except Exception as exc:  # a torn write; the next episode retries it
                    print(f"could not load {latest[1].name} yet: {exc}")

            seed = WATCH_SEED_BASE + episode
            obs, _ = env.reset(seed=seed)
            env.hud_overlay = [f"policy  {label}", f"episode {episode}   seed {seed}"]
            total, info = 0.0, {}
            while True:
                action, _ = policy.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                env.render()
                if terminated or truncated:
                    break

            m = info.get("episode_metrics", {})
            outcome = "CRASH" if m.get("collided") else "STUCK" if m.get("stuck") else "ok"
            print(
                f"ep {episode:4d}  {label:<34} {outcome:<5}  "
                f"dist {m.get('distance_m', 0.0):6.2f} m  "
                f"cover {m.get('coverage_m2', 0.0):5.1f} m2  "
                f"speed {m.get('mean_speed_mps', 0.0):4.2f} m/s  "
                f"reverse {m.get('reverse_frac', 0.0):5.1%}  return {total:8.1f}",
                flush=True,
            )
            episode += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
