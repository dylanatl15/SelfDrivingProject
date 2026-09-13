"""A policy that keeps its pre-tanh action means where tanh still responds.

Squashed gSDE samples u ~ N(mu, sigma) and drives with tanh(u). Once |mu| is well past 2,
nearly every sample squashes to the same command, so exploration stops trying different
actions and PPO has no advantage signal to bring mu back. phase1_v7 held its throttle mean
at +2.1 to +3.5 with a wall under 1 m ahead, and its noise tried braking on under 8 % of
those steps; the crashes in its probes came at full steering lock and near-full throttle.

The entropy bonus cannot fix that. With no analytic entropy SB3 uses -log_prob of actions
from the rollout buffer, whose tanh correction is constant in the parameters, so the bonus
only grows sigma. This policy adds `mean_penalty * mean(sum(relu(|mu| - mean_margin)^2))`
to the PPO loss instead.

PPO has no hook for an extra loss term, so the penalty rides in the entropy slot: the loss
is `ent_coef * -mean(entropy)`, and `evaluate_actions` returns the usual entropy estimate
minus `penalty / ent_coef`. That needs `ent_coef > 0`, and it means `train/entropy_loss`
includes the penalty; `train/mean_penalty` logs it on its own.

`log_std_min` floors the gSDE noise scale. Under `target_kl` that scale matters twice: the
KL between two Gaussians of one std grows as (change in mean / std)^2, so narrower noise lets
each update move the policy less. PPO narrowed it anyway. waypoint_pay5's `train/std` fell
from 0.082 at 500k steps to 0.030 at 3M, most updates stopped early on KL, and its evals fell
with it. The floor is applied to the parameter before each use, so the noise sampled, the
distribution trained and `train/std` agree; `train/std_at_floor` logs how much of it binds.
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class MeanPenaltyPolicy(ActorCriticPolicy):
    def __init__(self, *args, mean_margin: float = 1.5, mean_penalty: float = 0.0,
                 ent_coef: float = 0.0, log_std_min: float | None = None, **kwargs):
        if mean_penalty > 0.0 and ent_coef <= 0.0:
            raise ValueError("mean_penalty rides in the entropy term, so it needs ent_coef > 0")
        self.mean_margin = float(mean_margin)
        self.mean_penalty = float(mean_penalty)
        self.ent_coef = float(ent_coef)
        self.last_mean_penalty = 0.0
        self.last_mean_abs = 0.0
        self.log_std_min = None if log_std_min is None else float(log_std_min)
        super().__init__(*args, **kwargs)

    def _floor_log_std(self) -> None:
        if self.log_std_min is not None:
            with th.no_grad():
                self.log_std.clamp_(min=self.log_std_min)

    def reset_noise(self, n_envs: int = 1) -> None:
        self._floor_log_std()
        super().reset_noise(n_envs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        self._floor_log_std()
        return super()._get_action_dist_from_latent(latent_pi)

    def mean_penalty_of(self, mean_actions: th.Tensor) -> th.Tensor:
        """Per-sample penalty on pre-tanh means beyond the margin, summed over action dims."""
        excess = th.relu(mean_actions.abs() - self.mean_margin)
        return (excess ** 2).sum(dim=1)

    def evaluate_actions(self, obs, actions: th.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        if self.mean_penalty <= 0.0:
            return values, log_prob, entropy

        mean_actions = distribution.distribution.mean  # before tanh
        penalty = self.mean_penalty_of(mean_actions)
        self.last_mean_penalty = float(penalty.mean().detach())
        self.last_mean_abs = float(mean_actions.abs().mean().detach())
        base = -log_prob if entropy is None else entropy  # what PPO would have used
        return values, log_prob, base - self.mean_penalty * penalty / self.ent_coef
