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
from stable_baselines3 import PPO
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
        "timesteps", "wall_s", "distance_m", "coverage_m2", "mean_speed_mps", "min_clearance_m",
        "stall_frac", "reverse_frac", "lock_frac", "retrace_frac", "collided", "stuck", "steps",
    ]

    def __init__(self, path: str | Path, start_timesteps: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.path = Path(path)
        self.start_timesteps = start_timesteps
        self._start = time.time()
        self._fh = None
        self._writer = None

    def rows_to_keep(self) -> list[dict]:
        """On a resume, the rows logged up to the checkpoint. Anything later came from the
        stretch of training that was lost, which is about to be run again."""
        if not self.start_timesteps or not self.path.exists():
            return []
        with open(self.path, newline="") as fh:
            return [row for row in csv.DictReader(fh)
                    if int(row["timesteps"]) <= self.start_timesteps]

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        kept = self.rows_to_keep()  # read in full before the file is reopened for writing
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self._writer.writerows(kept)
        if kept:
            self._start = time.time() - float(kept[-1]["wall_s"] or 0.0)

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

    The best model is the one with the most `clean_coverage_m2`: floor covered in episodes
    that neither crashed nor got stuck. Keeping it by success rate crowned `phase1_v2` at
    11M steps, which circled open patches and so never failed.
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
        start_steps: int = 0,
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
        self.start_steps = start_steps
        # The next multiple after where training starts, so a resumed run neither repeats
        # an evaluation it already logged nor fires one per step until it catches up.
        self._next_eval = (start_steps // every_steps + 1) * every_steps
        self._next_video = ((start_steps // video_every_steps + 1) * video_every_steps
                            if video_every_steps else None)
        self._best = -np.inf
        self._env = None

    def _on_training_start(self) -> None:
        # A resumed run must not replace its best model with whatever it evaluates first,
        # so the saved one is scored again on the same seeds. Evaluation is deterministic,
        # so this reproduces the score that earned it the file.
        best = self.best_model_path.with_suffix(".zip") if self.best_model_path else None
        if not (self.start_steps and best is not None and best.exists()):
            return
        result = run_episodes(
            PPO.load(best, device="cpu"), self._get_env(render=False), self.n_episodes,
            seed=self.EVAL_SEED_BASE, deterministic=True,
        )
        self._best = result.clean_coverage_m2
        if self.verbose:
            print(f"[eval] resuming; the saved best model covers "
                  f"{result.clean_coverage_m2:.1f} m2 clean")

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

        # Clean coverage, not success rate (see the class docstring) and not return,
        # which is normalized and drifts with VecNormalize.
        if self.best_model_path and result.clean_coverage_m2 > self._best:
            self._best = result.clean_coverage_m2
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_path)

        if self._next_video is not None and self.num_timesteps >= self._next_video:
            self._next_video += self.video_every_steps
            try:
                self._record_video()
            except Exception as exc:  # a broken video must never end a multi-hour run
                print(f"[eval] video failed, training continues: {exc!r}")
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
