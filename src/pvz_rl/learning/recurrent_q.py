"""Chronological complete-episode fitting on the shared CUDA cohort lifecycle."""

import numpy as np
import torch

from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.cuda_q import CudaCohortLifecycle
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.sequential_q import action_parts, balanced_q_loss


def sequence_batches(buffer, chunk_length, decision_budget):
    """Bounded chronological chunks, padded only after an episode's final decision."""
    if decision_budget < 1 or decision_budget % chunk_length:
        raise ValueError("Recurrent batch_size must be a positive multiple of chunk_length")
    width = decision_budget // chunk_length
    steps = buffer.size // buffer.n_envs
    for first in range(0, buffer.n_envs, width):
        slots = np.arange(first, min(first + width, buffer.n_envs))
        for start in range(0, steps, chunk_length):
            times = np.arange(start, min(start + chunk_length, steps))
            rows = buffer.take(slots[:, None] + times[None] * buffer.n_envs)
            if not rows["active"].any():
                break
            yield start == 0, rows, buffer.observations(rows)


def sequence_loss(policy, rows, state, counts, observations):
    """The contribution to whole-cohort equal-group regression, excluding padding."""
    device = next(policy.parameters()).device

    def tensor(name, dtype):
        return torch.as_tensor(np.array(rows[name]), device=device, dtype=dtype)

    active = tensor("active", torch.bool)
    actions = tensor("action", torch.long)[active]
    q, tiles, context, state = policy.forward_sequence(
        observations.to(device),
        state,
        previous_actions=tensor("previous", torch.long),
        execution_outcomes=tensor("previous_outcome", torch.float32),
        return_context=True,
    )
    branches, locations = action_parts(actions)
    first = q[active].gather(1, branches[:, None]).flatten()
    second = torch.zeros_like(first)
    nonwait = branches != 0
    if nonwait.any():
        second[nonwait] = (
            policy.tile_values(tiles[active][nonwait], context[active][nonwait], branches[nonwait])
            .gather(1, locations[nonwait, None])
            .flatten()
        )
    loss, branch_error, tile_error = balanced_q_loss(
        first, second, tensor("target", torch.float32)[active], actions, counts, int(sum(counts))
    )
    return loss, state.detach(), branch_error, tile_error[nonwait]


class CudaRecurrentQ(CudaCohortLifecycle):
    policy_protocol = "transformer_lstm_q_v2"
    optimizer_version = "complete_return_lstm_v1"

    def _new_memory(self):
        return PolicyRunner(self.policy, self.cfg, self.env.batch.rules, self.n_envs, self.device)

    def _begin(self, callback):
        super()._begin(callback)
        self._sequence_iterator = self._sequence_state = None
        self._pass_loss = 0.0

    def _collect_step(self, callback):
        env = self.env
        with torch.no_grad(), env.device_context():
            active = env.enabled_envs.copy()
            enabled = torch.as_tensor(active, device=self.device)
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            # Copy public inputs before the simulator overwrites its observation tensor.
            observations = self._last_obs.cpu()
            rows["entity_count"] = observations.entity_mask.sum(-1).numpy()
            rows["entity_omitted"] = env.features.truncation_counts.get()
            rows["previous"] = self._memory.previous_actions.cpu().numpy()
            rows["previous_outcome"] = self._memory.outcomes.cpu().numpy()
            rows["reset"] = self._first
            rows["active"], rows["env"] = active, np.arange(self.n_envs)
            rows["tick"] = env.header_tensor[:, 0].cpu().numpy()
            masks = env.action_masks().clone()
            result = self._memory.decide(
                self._last_obs,
                masks,
                env.header_tensor[:, 0],
                active=enabled,
                deterministic=False,
            )
            actions, values, tile_values, details = result
            rows["mask"] = np.packbits(masks.cpu().numpy(), axis=-1, bitorder="little")
            rows["action"] = actions.cpu().numpy()
            rows["branch_value"] = values.cpu().numpy()
            rows["tile_value"] = tile_values.cpu().numpy()
            rows["greedy_action"] = details["greedy_actions"].cpu().numpy()
            coins = details["coins"].cpu().numpy()
            rows["species_coin"], rows["tile_coin"] = coins[:, 0], coins[:, 1]
            next_obs, _, dones, _, _, infos = env.step_tensors(actions, autoreset=False)
            rows["reward"] = env.last_transition_host[:, 3]
            rows["duration"] = env.last_transition_host[:, 2]
            rows["done"] = env.last_transition_host[:, 0].astype(bool)
            rows["accepted"] = env.last_action_result_host[:, 0].astype(bool)
            self._memory.observe_result(
                env.proposed_actions,
                rows["accepted"].copy(),
                rows["duration"].copy(),
                active=enabled,
            )
            env.action_journal.record_batch_safe(
                rows,
                details["branch_q"].cpu().numpy(),
                env.last_action_result_host,
                env._episode_serial,
            )
            rows["executed_action"] = env.executed_actions.cpu().numpy()
            self._buffer.append(rows, observations)
            self._planting_samples += int(
                np.count_nonzero(active & (rows["action"] > 0) & (rows["action"] < 361))
            )
            self._stats["species_exploration_coins"] += int(rows["species_coin"][active].sum())
            self._stats["tile_exploration_coins"] += int(rows["tile_coin"][active].sum())
            self._stats["exploratory_changes"] += int(
                np.count_nonzero(active & (rows["action"] != rows["greedy_action"]))
            )
            self._simulation_ticks += int(rows["duration"][active].sum())
            self._first, self._last_obs = False, next_obs
            self.num_timesteps += int(active.sum())
            active_infos = [info for info, on in zip(infos, active) if on]
            if not hasattr(callback, "cfg"):
                self.training_games += sum("episode_metrics" in info for info in active_infos)
            callback.update_locals({"infos": active_infos, "dones": dones})
            # Publish the phase before callbacks may request an interruption.
            if not env.enabled_envs.any():
                self._phase(CohortPhase.RETURNS, callback)
            if not callback.on_step():
                raise KeyboardInterrupt

    def _fit_step(self, callback):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            self._fit_chunk(callback)
        finally:
            end.record()
            end.synchronize()
            self._device_seconds = (
                getattr(self, "_device_seconds", 0.0) + start.elapsed_time(end) / 1000
            )

    def _fit_chunk(self, callback):
        if self._fit_epoch >= self.n_epochs:
            self._phase(CohortPhase.SYNCHRONIZE, callback)
            return
        if getattr(self, "_sequence_iterator", None) is None:
            self.policy.optimizer.zero_grad(set_to_none=True)
            self._sequence_state = None
            self._pass_loss = 0.0
            self._sequence_iterator = iter(
                sequence_batches(self._buffer, self.cfg["policy"]["chunk_length"], self.batch_size)
            )
        item = next(self._sequence_iterator, None)
        if item is None:
            norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True
            )
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad(set_to_none=True)
            self._stats["q_optimizer_steps"] += 1
            self._fit_epoch += 1
            self._sequence_iterator = self._sequence_state = None
            self.logger.record("train/q_loss", self._pass_loss)
            self.logger.record("train/q_grad_norm", float(norm))
        else:
            reset, rows, observations = item
            if reset:
                self._sequence_state = None
            self.policy.set_training_mode(True)
            loss, self._sequence_state, branch_error, tile_error = sequence_loss(
                self.policy, rows, self._sequence_state, self._buffer.group_counts, observations
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite recurrent Q loss")
            loss.backward()
            self._pass_loss += float(loss.detach())
            self.logger.record("train/branch_loss", float(branch_error.detach().mean()))
            self.logger.record(
                "train/tile_loss", float(tile_error.detach().mean()) if len(tile_error) else None
            )
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _restore_runtime(self):
        super()._restore_runtime()
        # Only complete passes commit an optimizer update. Recompute the current
        # pass from episode starts; gradients and carried fitting states are transient.
        self.policy.optimizer.zero_grad(set_to_none=True)
        self._sequence_iterator = self._sequence_state = None

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "_sequence_iterator", "_sequence_state"]
