"""Bundle a run into one markdown report.

    uv run python -m selfdrive.eval.report --run runs/<run> --model runs/<run>/best_model.zip

The senior project needs a written deliverable, and scraping numbers out of TensorBoard
event files at the end of a semester is worse than writing them down as they are
produced. This reads the per-episode CSV the training callback wrote, runs the
adversarial scenario suite, and emits a single markdown file with the tables already
formatted, plus a tidy CSV for whatever plotting tool you prefer.

Deliberately no plotting dependency. The tidy CSV has one row per metric per bucket;
feed it to matplotlib, a spreadsheet, or pandas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_episodes(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "episodes.csv"
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(path) as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = math.nan
            rows.append(row)
    return rows


def bucket_training_curve(rows: list[dict[str, float]], n_buckets: int = 20
                          ) -> list[dict[str, float]]:
    """Collapse episodes into equal-width timestep buckets.

    Raw per-episode rows are far too noisy to read; a training run produces tens of
    thousands of them.
    """
    if not rows:
        return []
    steps = np.array([r.get("timesteps", math.nan) for r in rows])
    finite = steps[np.isfinite(steps)]
    if finite.size == 0:
        return []

    edges = np.linspace(finite.min(), finite.max() + 1, n_buckets + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, n_buckets - 1)
    out = []
    for b in range(n_buckets):
        sel = [r for r, i in zip(rows, idx, strict=True) if i == b]
        if not sel:
            continue

        def mean(key: str, sel=sel) -> float:
            vals = [r.get(key, math.nan) for r in sel]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else math.nan

        out.append({
            "timesteps": float(edges[b]),
            "episodes": float(len(sel)),
            "collision_rate": mean("collided"),
            "stuck_rate": mean("stuck"),
            "distance_m": mean("distance_m"),
            "coverage_m2": mean("coverage_m2"),
            "mean_speed_mps": mean("mean_speed_mps"),
            "min_clearance_m": mean("min_clearance_m"),
            "reverse_frac": mean("reverse_frac"),
        })
    return out


def _table(rows: list[dict[str, float]], columns: list[str], fmt: dict[str, str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(format(r.get(c, math.nan), fmt.get(c, ".2f")) for c in columns) + " |"
        for r in rows
    ]
    return "\n".join([head, rule, *body])


def build_report(run_dir: Path, model: str | None, config: str, n_episodes: int) -> str:
    rows = read_episodes(run_dir)
    curve = bucket_training_curve(rows)

    parts = [f"# Training report - `{run_dir.name}`", ""]

    cfg_path = run_dir / "train_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        parts += [
            "## Run configuration",
            "",
            f"- workers: {cfg.get('n_envs')}",
            f"- budget: {cfg.get('total_timesteps'):,} steps" if cfg.get("total_timesteps")
            else "",
            f"- seed: {cfg.get('seed')}",
            f"- net_arch: {cfg.get('net_arch')}",
            f"- env config: `{cfg.get('env_config')}`",
            "",
        ]

    if curve:
        parts += [
            "## Training curve",
            "",
            "Episodes bucketed by timestep. `collision_rate` and `stuck_rate` are the two",
            "Phase 1 failure modes and should both fall; `coverage_m2` should rise.",
            "`distance_m` alone proves little: a car orbiting an open patch racks it up.",
            "",
            _table(
                curve,
                ["timesteps", "episodes", "collision_rate", "stuck_rate", "distance_m",
                 "coverage_m2", "mean_speed_mps", "reverse_frac"],
                {"timesteps": ".0f", "episodes": ".0f", "collision_rate": ".3f",
                 "stuck_rate": ".3f", "distance_m": ".2f", "coverage_m2": ".1f",
                 "mean_speed_mps": ".2f", "reverse_frac": ".3f"},
            ),
            "",
            f"Total episodes recorded: {len(rows):,}",
            "",
        ]
    else:
        parts += ["## Training curve", "", "_No `episodes.csv` found in the run directory._", ""]

    if model:
        from stable_baselines3 import PPO

        from ..config import load_env_config
        from ..envs.car_env import CarEnv
        from .evaluate import RandomPolicy, run_episodes
        from .scenarios import format_suite, run_suite

        policy = PPO.load(model, device="cpu")
        env = CarEnv(load_env_config(config))
        trained = run_episodes(policy, env, n_episodes, seed=1_000_000)
        baseline = run_episodes(RandomPolicy(0), env, n_episodes, seed=1_000_000)
        env.close()

        parts += [
            "## Random arenas",
            "",
            "Held-out seeds, identical arenas for both rows.",
            "",
            "| policy | success | collision | stuck | distance m | coverage m2 | speed m/s "
            "| reverse |",
            "|---|---|---|---|---|---|---|---|",
            f"| trained | {trained.success_rate:.1%} | {trained.collision_rate:.1%} | "
            f"{trained.stuck_rate:.1%} | {trained.mean_distance_m:.2f} | "
            f"{trained.mean_coverage_m2:.1f} | "
            f"{trained.mean_speed_mps:.2f} | {trained.mean_reverse_frac:.1%} |",
            f"| random | {baseline.success_rate:.1%} | {baseline.collision_rate:.1%} | "
            f"{baseline.stuck_rate:.1%} | {baseline.mean_distance_m:.2f} | "
            f"{baseline.mean_coverage_m2:.1f} | "
            f"{baseline.mean_speed_mps:.2f} | {baseline.mean_reverse_frac:.1%} |",
            "",
            "## Adversarial scenarios",
            "",
            "Hand-authored traps, fixed geometry and start pose. This is the table that",
            "actually demonstrates the Phase 1 requirement: a policy can score well on",
            "random arenas and still never escape a dead end.",
            "",
            "```",
            format_suite(run_suite(policy, config, n_episodes)),
            "```",
            "",
        ]

    return "\n".join(p for p in parts if p != "" or True)


def write_tidy_csv(run_dir: Path, out: Path) -> None:
    """One row per (timestep bucket, metric). Convenient for any plotting tool."""
    curve = bucket_training_curve(read_episodes(run_dir))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timesteps", "metric", "value"])
        for row in curve:
            for metric, value in row.items():
                if metric != "timesteps":
                    w.writerow([int(row["timesteps"]), metric, value])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate a markdown report for a training run.")
    p.add_argument("--run", required=True, help="run directory, e.g. runs/ppo_20260912_0100")
    p.add_argument("--model", default=None, help="model to evaluate; omit for curve only")
    p.add_argument("--config", default="configs/env_phase1.yaml")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default=None, help="defaults to <run>/report.md")
    args = p.parse_args(argv)

    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "report.md"
    out.write_text(build_report(run_dir, args.model, args.config, args.episodes))
    write_tidy_csv(run_dir, run_dir / "curve_tidy.csv")
    print(f"wrote {out}")
    print(f"wrote {run_dir / 'curve_tidy.csv'}")


if __name__ == "__main__":
    main()
