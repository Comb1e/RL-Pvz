"""SB3-compatible PPO with tensor collection and batched optimizer diagnostics.

The PPO ratios and minibatch order follow SB3/SB3-Contrib 2.7.1. Storage,
transport and metric transfers stay on-device where possible. An explicitly
configured exploration bonus can replace joint-entropy regularization.
"""

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import update_learning_rate
from torch.nn import functional as F

from .cuda_buffer import TensorRolloutBuffer
from .exploration import exploration_loss


class TensorPPO:
    def _maybe_recommend_cpu(self, mlp_policy_class_name="ActorCriticPolicy"):
        """The upstream warning assumes CPU observations; this path requires CUDA."""

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        self._last_obs = torch.as_tensor(self._last_obs, device=self.device)
        self._last_episode_starts = torch.as_tensor(self._last_episode_starts, device=self.device)
        with env.device_context():
            for _ in range(n_rollout_steps):
                # The environment exposes reusable zero-copy views; save the
                # prior observation/mask before its next in-place device step.
                obs = self._last_obs.clone()
                masks = env.action_masks().clone()
                with torch.no_grad(), env.features.profiler.track("inference"):
                    actions, log_probs = self.policy.sample_actions(obs, masks)
                new_obs, rewards, dones, timeouts, terminal, infos = env.step_tensors(actions)
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                self._update_info_buffer(infos)
                if terminal is not None and any(
                    info.get("TimeLimit.truncated", False) for info in infos
                ):
                    indices = torch.nonzero(timeouts, as_tuple=True)[0]
                    rollout_buffer.add_timeouts(indices, terminal[indices])
                rollout_buffer.add(
                    obs, actions, rewards, self._last_episode_starts, None, log_probs, masks
                )
                self._last_obs, self._last_episode_starts = new_obs, dones
            with env.features.profiler.track("critic_inference"):
                values = rollout_buffer.evaluate_values(
                    self.policy, self._last_obs, getattr(self, "value_batch_size", 1024)
                )
            with env.features.profiler.track("gae"):
                rollout_buffer.compute_returns_and_advantage(values, dones)
        callback.on_rollout_end()
        env.features.profiler.flush()
        return True

    @torch.no_grad()
    def _type_probabilities(self):
        buffer = self.rollout_buffer
        observations = buffer.observations.flatten(0, 1)
        masks = buffer.action_masks.flatten(0, 1)
        size = getattr(self, "value_batch_size", 1024)
        return torch.cat(
            [
                self.policy.get_distribution(
                    observations[i : i + size], masks[i : i + size]
                ).types.probs
                for i in range(0, len(observations), size)
            ]
        )

    @torch.no_grad()
    def _post_update_metrics(self, old_types):
        buffer = self.rollout_buffer
        observations = buffer.observations.flatten(0, 1)
        masks = buffer.action_masks.flatten(0, 1)
        actions = buffer.actions.flatten().long()
        old_logs = buffer.log_probs.flatten()
        totals = torch.zeros(4, device=self.device)
        size = getattr(self, "value_batch_size", 1024)
        for i in range(0, len(observations), size):
            ix = slice(i, i + size)
            distribution = self.policy.get_distribution(observations[ix], masks[ix])
            log_ratio = distribution.log_prob(actions[ix]) - old_logs[ix]
            previous, current = old_types[ix], distribution.types.probs
            kl = (
                previous * (previous.clamp_min(1e-30).log() - current.clamp_min(1e-30).log())
            ).sum(-1)
            legal_dig = masks[ix, 361:].any(-1)
            totals += torch.stack(
                (
                    (log_ratio.exp() - 1 - log_ratio).sum(),
                    kl.sum(),
                    (current[:, -1] * legal_dig).sum(),
                    legal_dig.sum(),
                )
            )
        approx, types, dig, count = totals.cpu().tolist()
        return {
            "post_update_approx_kl": approx / len(observations),
            "post_update_type_kl": types / len(observations),
            "dig_probability_when_legal": dig / count if count else float("nan"),
            "dig_legal_observations": count,
        }

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        critic_lr = self.policy.critic_learning_rate
        if critic_lr is None:
            critic_lr = self.lr_schedule(self._current_progress_remaining)
        update_learning_rate(self.policy.critic_optimizer, critic_lr)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = (
            self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        )
        old_types = self._type_probabilities()
        actor_metrics, value_losses, actor_norms, critic_norms = [], [], [], []
        actor_active, actor_steps, critic_steps = True, 0, 0
        actor_epochs = 0
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                if actor_active:
                    log_prob, entropy = self.policy.evaluate_actor(
                        data.observations, data.actions.long().flatten(), data.action_masks
                    )
                    advantages = data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = torch.exp(log_prob - data.old_log_prob)
                    policy_loss = -torch.minimum(
                        advantages * ratio,
                        advantages * ratio.clamp(1 - clip_range, 1 + clip_range),
                    ).mean()
                    regularizer, exploration_metrics = exploration_loss(
                        self.policy, entropy, log_prob
                    )
                    actor_loss = policy_loss + regularizer
                    with torch.no_grad():
                        log_ratio = log_prob - data.old_log_prob
                        kl = (log_ratio.exp() - 1 - log_ratio).mean()
                        actor_metrics.append(
                            torch.stack(
                                (
                                    policy_loss.detach(),
                                    -entropy.mean().detach(),
                                    ((ratio - 1).abs() > clip_range).float().mean(),
                                    exploration_metrics["joint_entropy"],
                                    exploration_metrics["exploration_bonus"],
                                    (data.action_masks.sum(-1) > 1).float().mean(),
                                    kl,
                                    actor_loss.detach(),
                                )
                            )
                        )
                    if self.target_kl is not None and float(kl) > 1.5 * self.target_kl:
                        actor_active = False
                    else:
                        self.policy.optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        actor_norms.append(
                            torch.nn.utils.clip_grad_norm_(
                                self.policy.actor_parameters(), self.max_grad_norm
                            ).detach()
                        )
                        self.policy.optimizer.step()
                        actor_steps += 1
                # Critic updates cannot change the actor and continue after actor KL stopping.
                values = self.policy.predict_values(data.observations).flatten()
                predicted = (
                    values
                    if clip_vf is None
                    else data.old_values + (values - data.old_values).clamp(-clip_vf, clip_vf)
                )
                value_loss = F.mse_loss(data.returns, predicted)
                value_losses.append(value_loss.detach())
                self.policy.critic_optimizer.zero_grad(set_to_none=True)
                (self.vf_coef * value_loss).backward()
                critic_norms.append(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.critic_parameters(), self.max_grad_norm
                    ).detach()
                )
                self.policy.critic_optimizer.step()
                critic_steps += 1
            actor_epochs += int(actor_active)
        self._n_updates += self.n_epochs
        returns, old_values = self.rollout_buffer.returns, self.rollout_buffer.values
        variance = returns.flatten().var(unbiased=False)
        explained = 1 - (returns - old_values).flatten().var(unbiased=False) / variance
        explained = torch.where(variance == 0, torch.full_like(explained, float("nan")), explained)
        actor = torch.stack(actor_metrics).mean(0)
        value_mean = torch.stack(value_losses).mean()

        def mean_norm(norms):
            return torch.stack(norms).mean() if norms else torch.zeros((), device=self.device)

        numbers = (
            torch.cat(
                (
                    actor,
                    torch.stack(
                        (
                            value_mean,
                            explained,
                            mean_norm(actor_norms),
                            mean_norm(critic_norms),
                        )
                    ),
                )
            )
            .cpu()
            .tolist()
        )
        keys = (
            "policy_gradient_loss",
            "entropy_loss",
            "clip_fraction",
            "joint_entropy",
            "exploration_bonus",
            "choice_fraction",
            "approx_kl",
            "actor_loss",
            "value_loss",
            "explained_variance",
            "actor_grad_norm",
            "critic_grad_norm",
        )
        metrics = dict(zip(keys, numbers))
        metrics.update(
            loss=metrics["actor_loss"] + self.vf_coef * metrics["value_loss"],
            optimizer_steps=actor_steps,
            actor_optimizer_steps=actor_steps,
            critic_optimizer_steps=critic_steps,
            epochs_completed=actor_epochs,
            critic_epochs_completed=self.n_epochs,
            kl_stopped=float(not actor_active),
            critic_learning_rate=critic_lr,
            **self._post_update_metrics(old_types),
        )
        for key, value in metrics.items():
            self.logger.record(f"train/{key}", value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_vf is not None:
            self.logger.record("train/clip_range_vf", clip_vf)

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return [*state_dicts, "policy.critic_optimizer"], variables


class CudaMaskablePPO(TensorPPO, MaskablePPO):
    pass


def configure_tensor_buffer(model):
    model.rollout_buffer_class = TensorRolloutBuffer
    model.rollout_buffer_kwargs = {}
    model.rollout_buffer = TensorRolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )
