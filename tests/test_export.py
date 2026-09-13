"""The exported graph must pick the action SB3 picks, however the policy bounds it."""

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from selfdrive.config import load_env_config
from selfdrive.envs.car_env import CarEnv
from selfdrive.export.to_onnx import DeterministicPolicy


@pytest.mark.parametrize("kwargs", [
    {},  # Gaussian clipped by the env: phase1_v1 to v3
    {"use_sde": True, "squash_output": True, "log_std_init": -2.0},  # train_ppo_v4.yaml
])
def test_export_graph_matches_the_sb3_deterministic_action(kwargs):
    use_sde = kwargs.pop("use_sde", False)
    model = PPO("MlpPolicy", CarEnv(load_env_config("configs/env_nodr.yaml")), use_sde=use_sde,
                policy_kwargs={"net_arch": [16], **kwargs}, device="cpu", seed=0)
    with th.no_grad():
        model.policy.action_net.bias.fill_(3.0)  # a mean past the bounds: clip and tanh differ
    obs = np.random.default_rng(0).uniform(-1, 1, (32, model.observation_space.shape[0]))
    obs = obs.astype(np.float32)

    expected, _ = model.predict(obs, deterministic=True)
    with th.no_grad():
        exported = DeterministicPolicy(model.policy.eval())(th.as_tensor(obs)).numpy()
    np.testing.assert_allclose(exported, expected, atol=1e-5)
