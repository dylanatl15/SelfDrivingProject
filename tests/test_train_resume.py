"""Resuming a training run from a checkpoint.

A run is hours long and dies with whatever launched it: `phase1_v2` was killed at 4.5M
steps by a restart of the session that started it. Resuming has to carry on the step
count and restore the reward statistics, and must not double-log the stretch of training
that was lost.
"""

import csv
import json

import pytest
import torch
import yaml
from stable_baselines3 import PPO

from selfdrive.train import train_ppo
from selfdrive.train.callbacks import EpisodeCsvLogger, PeriodicEval
from selfdrive.train.policies import MeanPenaltyPolicy


def tiny_run_config(tmp_path, total: int, **extra):
    cfg = {
        "env_config": "configs/env_nodr.yaml",
        "n_envs": 1, "total_timesteps": total,
        "n_steps": 64, "batch_size": 64, "n_epochs": 1, "net_arch": [16],
        "eval_every_steps": 10**9, "video_every_steps": 0,
        "checkpoint_every_steps": 128, "run_dir": str(tmp_path), "target_kl": 0.03,
        **extra,
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_resume_carries_on_the_step_count_and_restores_reward_statistics(tmp_path):
    train_ppo.main(["--config", str(tiny_run_config(tmp_path, 256)), "--name", "run"])
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "ppo_128_steps.zip"
    assert checkpoint.exists()
    assert (run / "checkpoints" / "ppo_vecnormalize_128_steps.pkl").exists()

    train_ppo.main(["--resume", str(checkpoint), "--total-timesteps", "384"])

    model = PPO.load(run / "final_model.zip")
    assert model.num_timesteps == 384
    assert model.target_kl == 0.03  # settings from the run's saved config survive a resume
    assert (run / "checkpoints" / "ppo_384_steps.zip").exists()
    (record,) = json.loads((run / "train_config.json").read_text())["resumed"]
    assert record["steps"] == 128 and record["stats_restored"]


def test_squashed_gsde_settings_reach_the_model_and_survive_a_resume(tmp_path):
    config = tiny_run_config(tmp_path, 128, use_sde=True, sde_sample_freq=4,
                             squash_output=True, log_std_init=-2.0)
    train_ppo.main(["--config", str(config), "--name", "run"])
    checkpoint = tmp_path / "run" / "checkpoints" / "ppo_128_steps.zip"
    train_ppo.main(["--resume", str(checkpoint), "--total-timesteps", "256"])

    model = PPO.load(tmp_path / "run" / "final_model.zip")
    assert model.num_timesteps == 256
    assert model.use_sde and model.sde_sample_freq == 4 and model.policy.squash_output


def test_mean_penalty_policy_trains_saves_and_resumes(tmp_path):
    config = tiny_run_config(tmp_path, 128, use_sde=True, sde_sample_freq=4, squash_output=True,
                             log_std_init=-2.0, ent_coef=0.005, mean_penalty=0.05, mean_margin=1.5)
    train_ppo.main(["--config", str(config), "--name", "run"])
    checkpoint = tmp_path / "run" / "checkpoints" / "ppo_128_steps.zip"
    train_ppo.main(["--resume", str(checkpoint), "--total-timesteps", "256"])

    model = PPO.load(tmp_path / "run" / "final_model.zip")
    assert model.num_timesteps == 256
    assert isinstance(model.policy, MeanPenaltyPolicy)
    assert (model.policy.mean_penalty, model.policy.mean_margin) == (0.05, 1.5)


def test_trainer_pins_torch_threads(tmp_path):
    # Unpinned trainers running side by side starved each other's workers of CPU.
    before = torch.get_num_threads()
    try:
        config = tiny_run_config(tmp_path, 64, torch_threads=2)
        train_ppo.main(["--config", str(config), "--name", "run"])
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_resume_refuses_a_checkpoint_already_past_the_budget(tmp_path):
    train_ppo.main(["--config", str(tiny_run_config(tmp_path, 128)), "--name", "run"])
    with pytest.raises(SystemExit, match="raise --total-timesteps"):
        train_ppo.main(["--resume", str(tmp_path / "run" / "checkpoints" / "ppo_128_steps.zip")])


def test_resumed_episode_log_drops_rows_from_the_lost_stretch(tmp_path):
    path = tmp_path / "episodes.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EpisodeCsvLogger.FIELDS)
        w.writeheader()
        for steps in (100, 400, 500, 900):  # checkpoint at 500; 900 was never saved
            w.writerow({"timesteps": steps, "wall_s": steps / 10})
    kept = EpisodeCsvLogger(path, start_timesteps=500).rows_to_keep()
    assert [int(r["timesteps"]) for r in kept] == [100, 400, 500]
    assert EpisodeCsvLogger(path).rows_to_keep() == []  # a fresh run starts the file over


@pytest.mark.parametrize("start, next_eval, next_video", [
    (0, 500_000, 2_000_000),
    (4_000_000, 4_500_000, 6_000_000),  # 4M was already evaluated before the crash
    (4_200_000, 4_500_000, 6_000_000),
])
def test_resumed_evaluation_schedule_continues_from_the_checkpoint(start, next_eval, next_video):
    cb = PeriodicEval(None, every_steps=500_000, video_every_steps=2_000_000, start_steps=start)
    assert (cb._next_eval, cb._next_video) == (next_eval, next_video)
