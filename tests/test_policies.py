"""The pre-tanh mean penalty.

Squashed gSDE policies saturate: phase1_v7 held its throttle mean at +2 to +3.5 before tanh
with a wall under 1 m ahead, where no noise sample brakes. The entropy bonus cannot pull a
mean back, so `MeanPenaltyPolicy` adds a penalty on it through PPO's entropy term. It also
floors the gSDE noise scale, which PPO's KL cap otherwise narrows until updates stall.
"""

import math

import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from selfdrive.config import load_env_config, load_yaml
from selfdrive.envs.car_env import CarEnv
from selfdrive.train.policies import MeanPenaltyPolicy

ENT_COEF = 0.005


def make_policy(mean_penalty=0.05, mean_margin=1.5, bias=None, log_std_min=None):
    env = CarEnv(load_env_config("configs/env_nodr.yaml"))
    model = PPO(MeanPenaltyPolicy, env, use_sde=True, ent_coef=ENT_COEF, device="cpu", seed=0,
                policy_kwargs={"net_arch": [16], "squash_output": True, "log_std_init": -2.0,
                               "mean_penalty": mean_penalty, "mean_margin": mean_margin,
                               "ent_coef": ENT_COEF, "log_std_min": log_std_min})
    policy = model.policy
    if bias is not None:
        with th.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.fill_(bias)
    obs, _ = env.reset(seed=0)
    obs_t, _ = policy.obs_to_tensor(obs)
    return policy, obs_t, th.zeros((1, 2))


def extra_loss(policy, obs, actions):
    """What the penalty adds to PPO's loss: ent_coef * entropy_loss minus the stock estimate."""
    _, log_prob, entropy = policy.evaluate_actions(obs, actions)
    return ENT_COEF * -entropy.mean() - ENT_COEF * log_prob.mean()


def test_penalty_is_zero_inside_the_margin_and_quadratic_past_it():
    policy, _, _ = make_policy()
    mean = th.tensor([[1.4, -1.5], [2.5, 0.0], [-3.5, 3.5]])
    assert policy.mean_penalty_of(mean).tolist() == pytest.approx([0.0, 1.0, 8.0])


def test_without_a_penalty_it_is_the_stock_policy():
    policy, obs, actions = make_policy(mean_penalty=0.0, bias=3.0)
    ours = policy.evaluate_actions(obs, actions)
    stock = ActorCriticPolicy.evaluate_actions(policy, obs, actions)
    assert ours[2] is None and stock[2] is None  # squashed gSDE: PPO falls back to -log_prob
    assert th.equal(ours[0], stock[0]) and th.equal(ours[1], stock[1])


def test_a_penalty_needs_an_entropy_coefficient():
    env = CarEnv(load_env_config("configs/env_nodr.yaml"))
    with pytest.raises(ValueError, match="ent_coef"):
        PPO(MeanPenaltyPolicy, env, use_sde=True, device="cpu",
            policy_kwargs={"net_arch": [16], "squash_output": True,
                           "mean_penalty": 0.05, "ent_coef": 0.0})


def test_saturated_mean_adds_the_penalty_to_the_loss_and_its_gradient_pulls_the_mean_back():
    policy, obs, actions = make_policy(mean_penalty=0.05, mean_margin=1.5, bias=3.0)
    extra = extra_loss(policy, obs, actions)
    assert extra.item() == pytest.approx(0.05 * 2 * 1.5 ** 2, rel=1e-4)  # two actions at 3.0
    extra.backward()
    assert (policy.action_net.bias.grad > 0).all()  # gradient descent lowers the mean
    assert policy.last_mean_abs == pytest.approx(3.0)


def test_mean_inside_the_margin_adds_nothing():
    policy, obs, actions = make_policy(mean_penalty=0.05, mean_margin=1.5, bias=1.0)
    assert extra_loss(policy, obs, actions).item() == pytest.approx(0.0, abs=1e-6)


def test_the_noise_floor_holds_the_noise_sampled_and_trained_at_or_above_it():
    policy, obs, actions = make_policy(log_std_min=-3.0)
    with th.no_grad():
        policy.log_std.fill_(-5.0)
    policy.reset_noise(4)
    assert policy.action_dist.weights_dist.scale.min().item() == pytest.approx(math.exp(-3.0))
    with th.no_grad():
        policy.log_std.fill_(-5.0)
    policy.evaluate_actions(obs, actions)
    assert policy.log_std.min().item() == pytest.approx(-3.0)


def test_without_a_floor_the_noise_scale_is_left_alone():
    policy, obs, actions = make_policy()
    with th.no_grad():
        policy.log_std.fill_(-5.0)
    policy.reset_noise(4)
    policy.evaluate_actions(obs, actions)
    assert policy.log_std.max().item() == pytest.approx(-5.0)


def test_a_run_with_a_noise_floor_trains_and_keeps_its_noise_there():
    env = CarEnv(load_env_config("configs/env_nodr.yaml"))
    model = PPO(MeanPenaltyPolicy, env, use_sde=True, n_steps=64, batch_size=32, n_epochs=2,
                ent_coef=ENT_COEF, device="cpu", seed=0,
                policy_kwargs={"net_arch": [16], "squash_output": True, "log_std_init": -4.0,
                               "log_std_min": -3.0, "ent_coef": ENT_COEF})
    model.learn(128)
    assert model.policy.log_std.min().item() >= -3.0 - 1e-3  # one Adam step past it at most


def test_train_ppo_v8_is_v7_with_a_noise_floor():
    v7 = load_yaml("configs/train_ppo_v7.yaml")
    v8 = load_yaml("configs/train_ppo_v8.yaml")
    assert v8.pop("log_std_min") == -3.0
    assert v8 == v7
