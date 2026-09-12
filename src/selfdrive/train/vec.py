"""Vectorized environment construction.

Twenty subprocess workers on a 14C/28T Xeon. The policy is tiny, so wall-clock training
time is almost entirely env stepping, and worker count is the main throughput knob.

`render_mode` is never passed to a training worker. Evaluation builds its own single
environment when it wants pixels.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ..config import load_env_config
from ..envs.car_env import CarEnv


def make_env(config_path: str | Path | None, seed: int, rank: int, render_mode=None):
    def _init():
        env = CarEnv(load_env_config(config_path), render_mode=render_mode)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def make_vec_env(
    config_path: str | Path | None,
    n_envs: int = 20,
    seed: int = 0,
    normalize_reward: bool = True,
):
    fns = [make_env(config_path, seed, i) for i in range(n_envs)]
    # A single env is not worth the IPC cost, and DummyVecEnv is far easier to debug.
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    if normalize_reward:
        # Observations are already analytically normalized inside the env; normalizing
        # them again here would put a running-stats file between training and the phone.
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=100.0)
    return venv
