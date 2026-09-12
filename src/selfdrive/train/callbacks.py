"""Training callbacks: reward-term logging, episode CSV, periodic eval, video.

The reward-term logger is the important one. A hand-written reward is normally broken in
a way that a single scalar return curve cannot show - the return rises while the agent
quietly optimizes the wrong term. Logging each term separately makes that visible on the
first TensorBoard glance.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..config import load_env_config
from ..envs.car_env import CarEnv
from ..eval.evaluate import run_episodes


class RewardTermLogger(BaseCallback):
    """Mean contribution of each reward term, so hacking shows up as a shape change."""

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._sums: dict[str, float] = defaultdict(float)
        self._n = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("reward_terms")
            if terms:
                for k, v in terms.items():
                    self._sums[k] += float(v)
                self._n += 1

        if self._n and self.n_calls % self.log_freq == 0:
            for k, v in self._sums.items():
                self.logger.record(f"reward/{k}", v / self._n)
            self._sums.clear()
            self._n = 0
        return True


class EpisodeCsvLogger(BaseCallback):
    """Append per-episode metrics to CSV.

    The written report needs plots. Scraping them out of TensorBoard event files at the
    end of the semester is worse than writing the rows as they happen.
    """

    FIELDS = [
        "timesteps", "wall_s", "distance_m", "mean_speed_mps", "min_clearance_m",
        "stall_frac", "reverse_frac", "collided", "stuck", "steps",
    ]

    def __init__(self, path: str | Path, verbose: int = 0):
        super().__init__(verbose)
        self.path = Path(path)
        self._start = time.time()
        self._fh = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("episode_metrics")
            if not m:
                continue
            row = {k: m.get(k, "") for k in self.FIELDS}
            row["timesteps"] = self.num_timesteps
            row["wall_s"] = round(time.time() - self._start, 1)
            self._writer.writerow(row)
        return True

    def _on_rollout_end(self) -> None:
        if self._fh:
            self._fh.flush()

    def _on_training_end(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class PeriodicEval(BaseCallback):
    """Deterministic evaluation on a held-out seed range, plus optional video.

    Evaluation seeds start at 1_000_000 so they can never collide with training seeds -
    otherwise the scorecard silently reports training-set performance.
    """

    EVAL_SEED_BASE = 1_000_000

    def __init__(
        self,
        config_path: str | None,
        every_steps: int = 500_000,
        n_episodes: int = 20,
        video_every_steps: int = 0,
        video_episodes: int = 2,
        video_dir: str | Path = "videos",
        best_model_path: str | Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.config_path = config_path
        self.every_steps = every_steps
        self.n_episodes = n_episodes
        self.video_every_steps = video_every_steps
        self.video_episodes = video_episodes
        self.video_dir = Path(video_dir)
        self.best_model_path = Path(best_model_path) if best_model_path else None
        self._next_eval = every_steps
        self._next_video = video_every_steps if video_every_steps else None
        self._best = -np.inf
        self._env = None

    def _get_env(self, render: bool):
        # One env for scoring, rebuilt when pixels are needed. Never a training worker.
        mode = "rgb_array" if render else None
        if self._env is None or self._env.render_mode != mode:
            if self._env is not None:
                self._env.close()
            self._env = CarEnv(load_env_config(self.config_path), render_mode=mode)
        return self._env

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.every_steps

        result = run_episodes(
            self.model, self._get_env(render=False), self.n_episodes,
            seed=self.EVAL_SEED_BASE, deterministic=True,
        )
        for k, v in result.as_dict().items():
            self.logger.record(f"eval/{k}", v)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps:>10,}] {result}")

        # Success rate, not return: return is normalized and drifts with VecNormalize.
        if self.best_model_path and result.success_rate > self._best:
            self._best = result.success_rate
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_path)

        if self._next_video is not None and self.num_timesteps >= self._next_video:
            self._next_video += self.video_every_steps
            self._record_video()
        return True

    def _record_video(self) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError:
            if self.verbose:
                print("[eval] imageio missing, skipping video")
            return

        frames: list = []
        run_episodes(
            self.model, self._get_env(render=True), self.video_episodes,
            seed=self.EVAL_SEED_BASE, deterministic=True, render=True, frame_sink=frames,
        )
        if not frames:
            return
        self.video_dir.mkdir(parents=True, exist_ok=True)
        path = self.video_dir / f"step_{self.num_timesteps:010d}.mp4"
        imageio.mimsave(path, frames, fps=30, macro_block_size=1)
        if self.verbose:
            print(f"[eval] wrote {path}")

    def _on_training_end(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
