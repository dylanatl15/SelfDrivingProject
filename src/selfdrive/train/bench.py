"""Environment throughput benchmark.

    uv run python -m selfdrive.train.bench --n-envs 20

Training time here is env stepping, not gradient descent, so this number sets the whole
schedule.

Measured on this machine (Xeon E5-2680 v4, 14C/28T, WSL2): ~900 steps/s in a single
process, ~4,800 steps/s across 20 subprocess workers. That is only ~26% per-worker
efficiency, and it is *not* a CPU limit - it is per-vec-step pipe latency in the parent
process, which SubprocVecEnv pays on every step. Evidence: efficiency already drops to
56% at two workers, 28 workers performs no better than 20, pinning BLAS threads to 1
changes nothing, and fork/forkserver/spawn are all within 3% of each other. WSL2 syscall
overhead is the usual culprit.

Practical consequence: ~4,800 steps/s means a 20M-step budget takes about 70 minutes,
which is fine. Do not spend time optimizing the env further to chase throughput - with
zero compute per step the ceiling would still be about 5,700 steps/s. If this ever needs
to be genuinely faster, the fix is an env that batches many steps per IPC round trip,
not faster geometry.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..config import load_env_config
from ..envs.car_env import CarEnv
from .vec import make_vec_env

# Measured ceiling on this box, not an aspiration. See the module docstring.
TARGET_STEPS_PER_S = 4_500


def bench_single(config_path: str | None, steps: int = 20_000) -> float:
    env = CarEnv(load_env_config(config_path))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, size=2))
        if terminated or truncated:
            env.reset()
    return steps / (time.perf_counter() - t0)


def bench_vec(config_path: str | None, n_envs: int, steps: int = 2_000) -> float:
    venv = make_vec_env(config_path, n_envs=n_envs, seed=0, normalize_reward=False)
    venv.reset()
    actions = np.zeros((n_envs, 2), dtype=np.float32)
    rng = np.random.default_rng(0)
    venv.step(actions)  # warm the workers so process start-up is not timed
    t0 = time.perf_counter()
    for _ in range(steps):
        actions[:] = rng.uniform(-1, 1, size=(n_envs, 2))
        venv.step(actions)
    elapsed = time.perf_counter() - t0
    venv.close()
    return (steps * n_envs) / elapsed


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/env_phase1.yaml")
    p.add_argument("--n-envs", type=int, default=20)
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args(argv)

    single = bench_single(args.config, steps=20_000)
    print(f"single env      {single:10,.0f} steps/s")

    vec = bench_vec(args.config, args.n_envs, steps=args.steps)
    print(f"{args.n_envs} workers      {vec:10,.0f} steps/s   "
          f"({vec / single:.1f}x, {vec / args.n_envs / single:.0%} per-worker efficiency)")

    budget = 20_000_000
    print(f"\n{budget:,} steps at this rate: {budget / vec / 60:.1f} min")
    status = "OK" if vec >= TARGET_STEPS_PER_S else "REGRESSION - was ~4,800"
    print(f"baseline for this machine: >={TARGET_STEPS_PER_S:,} steps/s aggregate  {status}")


if __name__ == "__main__":
    main()
