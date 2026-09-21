"""SB3-compatible PPO with tensor collection and batched optimizer diagnostics.

The PPO ratios and minibatch order follow SB3/SB3-Contrib 2.7.1. Storage,
transport and metric transfers stay on-device where possible. An explicitly
configured exploration bonus can replace joint-entropy regularization.
"""

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from torch.nn import functional as F

from .cuda_buffer import TensorRolloutBuffer
from .exploration import exploration_loss
from .optimization import actor_weights, policy_advantages, weighted_mean


class TensorPPO:
    def _maybe_recommend_cpu(self, mlp_policy_class_name="ActorCriticPolicy"):
        """The upstream warning assumes CPU observations; this path requires CUDA."""

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        masked = isinstance(self, MaskablePPO)
        self._last_obs = torch.as_tensor(self._last_obs, device=self.device)
        self._last_episode_starts = torch.as_tensor(self._last_episode_starts, device=self.device)
        with env.device_context():
            for _ in range(n_rollout_steps):
                # The environment exposes reusable zero-copy views; save the
                # prior observation/mask before its next in-place device step.
                obs = self._last_obs.clone()
                masks = env.action_masks().clone() if masked else None
                with torch.no_grad(), env.features.profiler.track("inference"):
                    kwargs = {"action_masks": masks} if masked else {}
                    actions, values, log_probs = self.policy(obs, **kwargs)
                new_obs, rewards, dones, timeouts, terminal, infos = env.step_tensors(actions)
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                self._update_info_buffer(infos)
                if terminal is not None and any(
                    info.get("TimeLimit.truncated", False) for info in infos
                ):
                    with torch.no_grad():
                        terminal_values = self.policy.predict_values(terminal).flatten()
                    rewards = rewards + self.gamma * terminal_values * timeouts.float()
                rollout_buffer.add(
                    obs, actions, rewards, self._last_episode_starts, values, log_probs, masks
                )
                self._last_obs, self._last_episode_starts = new_obs, dones
            with torch.no_grad():
                values = self.policy.predict_values(self._last_obs)
            with env.features.profiler.track("gae"):
                rollout_buffer.compute_returns_and_advantage(values, dones)
        callback.on_rollout_end()
        env.features.profiler.flush()
        return True

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )
        masked = isinstance(self, MaskablePPO)
        metrics, last_kls = [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            last_kls = []
            for data in self.rollout_buffer.get(self.batch_size):
                kwargs = {"action_masks": data.action_masks} if masked else {}
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations, data.actions.long().flatten(), **kwargs
                )
                values = values.flatten()
                advantages = data.advantages
                objective = getattr(self, "actor_objective", "all_steps")
                weights = actor_weights(
                    data.action_masks if masked else None, advantages, objective
                )
                if objective == "choice_points_v1":
                    advantages = policy_advantages(advantages, weights, self.normalize_advantage)
                elif self.normalize_advantage and (masked or len(advantages) > 1):
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - data.old_log_prob)
                policy_loss = -weighted_mean(
                    torch.min(
                        advantages * ratio,
                        advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                    ),
                    weights,
                )
                clipped_fraction = ((ratio - 1).abs() > clip_range).float().mean()
                predicted = (
                    values
                    if clip_vf is None
                    else data.old_values + torch.clamp(values - data.old_values, -clip_vf, clip_vf)
                )
                value_loss = F.mse_loss(data.returns, predicted)
                entropy_loss = -(-log_prob).mean() if entropy is None else -entropy.mean()
                regularizer, exploration_metrics = exploration_loss(
                    self.policy,
                    entropy,
                    log_prob,
                    self.ent_coef,
                    weights if objective == "choice_points_v1" else None,
                )
                loss = policy_loss + regularizer + self.vf_coef * value_loss
                with torch.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = weighted_mean((torch.exp(log_ratio) - 1) - log_ratio, weights)
                    last_kls.append(kl)
                    metrics.append(
                        torch.stack(
                            (
                                policy_loss.detach(),
                                value_loss.detach(),
                                entropy_loss.detach(),
                                clipped_fraction,
                                exploration_metrics["joint_entropy"],
                                exploration_metrics["exploration_bonus"],
                                weights.mean(),
                            )
                        )
                    )
                if self.target_kl is not None and float(kl) > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            if not masked:
                self._n_updates += 1
            if not continue_training:
                break
        if masked:
            self._n_updates += self.n_epochs
        returns = torch.as_tensor(self.rollout_buffer.returns, device=self.device)
        old_values = torch.as_tensor(self.rollout_buffer.values, device=self.device)
        var_y = returns.flatten().var(unbiased=False)
        explained = 1 - (returns - old_values).flatten().var(unbiased=False) / var_y
        explained = torch.where(var_y == 0, torch.full_like(explained, float("nan")), explained)
        # One transfer after all optimizer updates, including the final update.
        numbers = (
            torch.cat(
                (
                    torch.stack(metrics).mean(0),
                    torch.stack(last_kls).mean()[None],
                    loss.detach()[None],
                    explained[None],
                )
            )
            .cpu()
            .tolist()
        )
        keys = (
            "policy_gradient_loss",
            "value_loss",
            "entropy_loss",
            "clip_fraction",
            "joint_entropy",
            "exploration_bonus",
            "actor_sample_fraction",
            "approx_kl",
            "loss",
            "explained_variance",
        )
        for key, value in zip(keys, numbers):
            self.logger.record(f"train/{key}", value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_vf is not None:
            self.logger.record("train/clip_range_vf", clip_vf)

    def _excluded_save_params(self):
        return super()._excluded_save_params()


class ResearchMaskablePPO(MaskablePPO):
    """CPU collector with the same tested optimization objective as CUDA."""

    train = TensorPPO.train


class CudaMaskablePPO(TensorPPO, MaskablePPO):
    pass


class CudaPPO(TensorPPO, PPO):
    pass


def configure_tensor_buffer(model, masked):
    model.rollout_buffer_class = TensorRolloutBuffer
    model.rollout_buffer_kwargs = {"masked": masked}
    model.rollout_buffer = TensorRolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        masked=masked,
    )
