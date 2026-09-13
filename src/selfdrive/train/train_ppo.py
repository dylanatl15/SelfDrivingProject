"""PPO training entry point.

    uv run python -m selfdrive.train.train_ppo --config configs/train_ppo.yaml
    uv run python -m selfdrive.train.train_ppo --resume runs/<name>/checkpoints/ppo_<N>_steps.zip

Everything that defines a run is written into the run directory alongside the model, so
a checkpoint six weeks from now can still be explained in the written report.

A run takes hours and dies with whatever launched it, so it can be resumed from any
checkpoint. The resumed stretch uses the run's saved config, fresh worker seeds rather
than a replay of the arenas the run began with, and the reward-normalization statistics
saved beside the checkpoint. A checkpoint written before those statistics were saved
still resumes; they restart from zero and settle within a few rollouts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from ..config import describe, load_env_config, load_yaml
from .callbacks import EpisodeCsvLogger, PeriodicEval, RewardTermLogger
from .vec import make_vec_env

DEFAULTS: dict = {
    "env_config": "configs/env_phase1.yaml",
    "n_envs": 20,
    "total_timesteps": 20_000_000,
    "seed": 0,
    "device": "cpu",
    "policy": "MlpPolicy",
    "net_arch": [256, 256],
    "n_steps": 512,
    "batch_size": 512,
    "n_epochs": 10,
    "learning_rate": 3.0e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.004,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Squashed gSDE bounds every sampled action with tanh. Gaussian actions clipped to
    # [-1, 1] make extra std free, and phase1_v3's entropy bonus ran it past 3.
    "use_sde": False,
    "sde_sample_freq": -1,
    "squash_output": False,
    "log_std_init": 0.0,
    "normalize_reward": True,
    "eval_every_steps": 500_000,
    "eval_episodes": 20,
    "video_every_steps": 2_000_000,
    "video_episodes": 2,
    "checkpoint_every_steps": 500_000,
    "run_dir": "runs",
}

CHECKPOINT_RE = re.compile(r"^ppo_(\d+)_steps\.zip$")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Phase 1 driving policy.")
    p.add_argument("--config", default=None, help="training YAML; CLI flags win over it")
    p.add_argument("--env-config", default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None, help="run directory name; defaults to a timestamp")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT",
                   help="continue the run that wrote this checkpoint, with its saved config; "
                        "only --total-timesteps may change")
    return p.parse_args(argv)


def resolve(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(load_yaml(args.config))
    for key in ("env_config", "n_envs", "total_timesteps", "seed", "device"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def start_run(args: argparse.Namespace):
    cfg = resolve(args)

    name = args.name or time.strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = Path(cfg["run_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the env YAML and point everything at the copy. PeriodicEval re-reads its
    # config at every evaluation, so editing the shared YAML mid-run used to crash the run
    # (phase1_v1 died at 4M steps when the reward keys changed under it).
    snapshot = run_dir / "env_config.yaml"
    shutil.copyfile(cfg["env_config"], snapshot)
    cfg["env_config_source"] = cfg["env_config"]
    cfg["env_config"] = str(snapshot)

    env_cfg = load_env_config(cfg["env_config"])
    (run_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (run_dir / "env_config.txt").write_text(describe(env_cfg))

    print(f"run dir      {run_dir}")
    print(f"workers      {cfg['n_envs']}   device {cfg['device']}")
    print(f"observation  {env_cfg.obs.size} floats "
          f"({env_cfg.obs.per_frame} per frame x {env_cfg.obs.frame_stack})")
    squashed = " squashed" if cfg["squash_output"] else ""
    print(f"actions      {f'gSDE{squashed}' if cfg['use_sde'] else 'Gaussian, clipped by the env'}")
    print(f"budget       {cfg['total_timesteps']:,} steps")

    venv = make_vec_env(
        cfg["env_config"], n_envs=cfg["n_envs"], seed=cfg["seed"],
        normalize_reward=cfg["normalize_reward"],
    )

    model = PPO(
        cfg["policy"],
        venv,
        policy_kwargs={
            "net_arch": list(cfg["net_arch"]),
            "squash_output": cfg["squash_output"],
            "log_std_init": cfg["log_std_init"],
        },
        use_sde=cfg["use_sde"],
        sde_sample_freq=cfg["sde_sample_freq"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg["target_kl"],
        seed=cfg["seed"],
        device=cfg["device"],
        tensorboard_log=str(run_dir / "tb"),
        verbose=1,
    )
    return run_dir, cfg, venv, model


def resume_run(args: argparse.Namespace):
    checkpoint = Path(args.resume)
    match = CHECKPOINT_RE.match(checkpoint.name)
    if match is None:
        raise SystemExit(f"not a checkpoint written by this trainer: {checkpoint}")
    steps = int(match.group(1))
    run_dir = checkpoint.parent.parent
    cfg = {**DEFAULTS, **json.loads((run_dir / "train_config.json").read_text())}
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps

    seed = cfg["seed"] + steps  # new arenas, not the ones the run started on
    venv = make_vec_env(
        cfg["env_config"], n_envs=cfg["n_envs"], seed=seed,
        normalize_reward=cfg["normalize_reward"],
    )
    stats = checkpoint.with_name(f"ppo_vecnormalize_{steps}_steps.pkl")
    if cfg["normalize_reward"]:
        if stats.exists():
            venv = VecNormalize.load(str(stats), venv.venv)
        else:
            print(f"resume       no {stats.name}: reward-normalization statistics restart")

    model = PPO.load(checkpoint, env=venv, device=cfg["device"])
    model.set_random_seed(seed)
    if model.num_timesteps >= cfg["total_timesteps"]:
        raise SystemExit(f"checkpoint is already at {model.num_timesteps:,} of "
                         f"{cfg['total_timesteps']:,} steps; raise --total-timesteps")

    cfg.setdefault("resumed", []).append({
        "checkpoint": str(checkpoint), "steps": model.num_timesteps, "seed": seed,
        "stats_restored": bool(cfg["normalize_reward"] and stats.exists()),
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    (run_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))

    print(f"run dir      {run_dir}")
    print(f"resume       {checkpoint.name} at {model.num_timesteps:,} steps, worker seed {seed}")
    print(f"budget       {cfg['total_timesteps']:,} steps")
    return run_dir, cfg, venv, model


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir, cfg, venv, model = resume_run(args) if args.resume else start_run(args)
    start = model.num_timesteps

    callbacks = CallbackList([
        RewardTermLogger(log_freq=2000),
        EpisodeCsvLogger(run_dir / "episodes.csv", start_timesteps=start),
        PeriodicEval(
            config_path=cfg["env_config"],
            every_steps=cfg["eval_every_steps"],
            n_episodes=cfg["eval_episodes"],
            video_every_steps=cfg["video_every_steps"],
            video_episodes=cfg["video_episodes"],
            video_dir=run_dir / "videos",
            best_model_path=run_dir / "best_model",
            start_steps=start,
        ),
        CheckpointCallback(
            save_freq=max(cfg["checkpoint_every_steps"] // cfg["n_envs"], 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,  # what makes a resume exact
        ),
    ])

    try:
        # Continuing a run keeps its step count, so TensorBoard and the checkpoint names
        # carry on from the checkpoint rather than starting again at zero.
        model.learn(total_timesteps=cfg["total_timesteps"] - start, callback=callbacks,
                    reset_num_timesteps=start == 0, progress_bar=False)
    except KeyboardInterrupt:
        print("\ninterrupted - saving before exit")
    finally:
        model.save(run_dir / "final_model")
        if cfg["normalize_reward"]:
            venv.save(str(run_dir / "vecnormalize.pkl"))
        venv.close()
        print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
