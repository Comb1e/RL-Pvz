"""Chronological complete-episode fitting on the shared CUDA cohort lifecycle."""

from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.encoding import EntityBatch
from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.cuda_q import CudaCohortLifecycle
from pvz_rl.learning.host_transfer import HostTransfer
from pvz_rl.learning.sequence_transport import SequencePrefetch, sequence_rows
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.sequential_q import action_parts, balanced_q_loss


def sequence_batches(buffer, chunk_length, decision_budget):
    """Chronological reference path, also used when staging cannot fit in RAM."""
    for reset, rows in sequence_rows(buffer, chunk_length, decision_budget):
        yield reset, rows, buffer.observations(rows)


def sequence_loss(policy, rows, state, counts, observations, fields=None):
    """The contribution to whole-cohort equal-group regression, excluding padding."""
    device = next(policy.parameters()).device

    def tensor(name, dtype):
        return (
            fields[name]
            if fields is not None
            else torch.as_tensor(np.array(rows[name]), device=device, dtype=dtype)
        )

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
    if np.any((rows["action"] != 0) & rows["active"]):
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

    def _setup_model(self):
        super()._setup_model()
        # SB3 restores ordinary attributes before rebuilding the policy. Keep a
        # saved cohort's FP32/memory fallback instead of overwriting it here.
        if not getattr(self, "execution_state", None):
            self._reset_execution()

    def _new_memory(self):
        return PolicyRunner(self.policy, self.cfg, self.env.batch.rules, self.n_envs, self.device)

    def _begin(self, callback):
        super()._begin(callback)
        self._sequence_iterator = self._sequence_state = None
        self._reset_execution()

    def _collect_step(self, callback):
        env = self.env
        with torch.no_grad(), env.device_context():
            active = env.enabled_envs.copy()
            enabled = torch.as_tensor(active, device=self.device)
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            rows["reset"] = self._first
            rows["active"], rows["env"] = active, np.arange(self.n_envs)
            masks = env.action_masks().clone()
            result = self._memory.decide(
                self._last_obs,
                masks,
                env.header_tensor[:, 0],
                active=enabled,
                deterministic=False,
            )
            actions, values, tile_values, details = result
            if not hasattr(self, "_collection_host"):
                self._collection_host = HostTransfer()
            # Queue immutable public inputs and decision evidence before simulation.
            columns = torch.cat(
                (
                    self._memory.previous_actions[:, None].float(),
                    self._memory.outcomes,
                    env.header_tensor[:, :1].float(),
                    actions[:, None].float(),
                    values[:, None],
                    tile_values[:, None],
                    details["greedy_actions"][:, None].float(),
                    details["coins"].float(),
                    details["branch_q"],
                    torch.from_dlpack(env.features.truncation_counts).float(),
                ),
                dim=1,
            )
            host = self._collection_host.enqueue(
                dict(
                    entities=self._last_obs.entities,
                    mask=self._last_obs.entity_mask,
                    globals=self._last_obs.globals,
                    action_masks=masks,
                    columns=columns,
                )
            )
            next_obs, _, dones, _, _, infos = env.step_tensors(actions, autoreset=False)
            # step_tensors completes these queued copies at its existing host boundary.
            observations = EntityBatch(host["entities"], host["mask"], host["globals"])
            packed = host["columns"].numpy()
            rows["entity_count"] = host["mask"].numpy().sum(-1)
            rows["previous"], rows["previous_outcome"] = packed[:, 0], packed[:, 1:3]
            rows["tick"], rows["action"] = packed[:, 3], packed[:, 4]
            rows["branch_value"], rows["tile_value"] = packed[:, 5], packed[:, 6]
            rows["greedy_action"] = packed[:, 7]
            rows["species_coin"], rows["tile_coin"] = packed[:, 8], packed[:, 9]
            rows["entity_omitted"] = packed[:, 20:23]
            rows["mask"] = np.packbits(host["action_masks"].numpy(), axis=-1, bitorder="little")
            rows["reward"] = env.last_transition_host[:, 3]
            rows["duration"] = env.last_transition_host[:, 2]
            rows["done"] = env.last_transition_host[:, 0].astype(bool)
            rows["accepted"] = env.last_action_result_host[:, 0].astype(bool)
            self._memory.observe_result(
                env.proposed_actions,
                env.header_tensor[:, 12],
                env.transition_ticks,
                active=enabled,
            )
            env.action_journal.record_batch_safe(
                rows,
                packed[:, 10:20],
                env.last_action_result_host,
                env._episode_serial,
            )
            rows["executed_action"] = np.where(rows["accepted"], rows["action"], 0)
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

    def _reset_execution(self):
        precision = self.cfg["training"].get("performance", {}).get("fit_precision", "fp32")
        self.execution_state = dict(
            requested_precision=precision,
            precision=precision,
            microbatch=self.cfg["policy"]["encoder_microbatch"],
            checkpointing=False,
            fallbacks=[],
        )
        self._device_events = []
        self._optimizer_events = []
        self._device_seconds = getattr(self, "_device_seconds", 0.0)
        self._optimizer_seconds = 0.0
        self._last_metrics_at = 0.0

    def configure_execution(self, cfg):
        """Apply only approved execution settings to a resumed model."""
        from pvz_rl.learning.performance import refresh_performance

        previous = self.cfg
        self.policy.cfg = refresh_performance(previous, cfg)
        self.policy.features_extractor.cfg = self.policy.cfg
        self.policy_kwargs["features_extractor_kwargs"]["layout_cfg"] = self.policy.cfg
        if (
            getattr(self, "execution_state", {}).get("requested_precision")
            != cfg["training"]["performance"].get("fit_precision", "fp32")
            or previous["policy"]["encoder_microbatch"] != cfg["policy"]["encoder_microbatch"]
        ):
            self._reset_execution()
        self.policy.entity.token_budget = cfg["policy"]["encoder_token_budget"]
        for layer in self.policy.entity.layers:
            layer.query_chunk = cfg["policy"]["attention_query_chunk"]

    def _drain(self):
        iterator = getattr(self, "_sequence_iterator", None)
        if isinstance(iterator, SequencePrefetch):
            for future in iterator.futures:
                future.result()
            iterator.stream.synchronize()
        for start, end in getattr(self, "_device_events", []):
            end.synchronize()
            self._device_seconds += start.elapsed_time(end) / 1000
        self._device_events = []
        for start, end in getattr(self, "_optimizer_events", []):
            end.synchronize()
            self._optimizer_seconds += start.elapsed_time(end) / 1000
        self._optimizer_events = []

    def _close_sequences(self):
        iterator = getattr(self, "_sequence_iterator", None)
        if isinstance(iterator, SequencePrefetch):
            iterator.close()
        self._sequence_iterator = self._sequence_state = None

    def _restart_pass(self, reason):
        self._close_sequences()
        self.policy.optimizer.zero_grad(set_to_none=True)
        self.execution_state["fallbacks"].append(reason)
        self._drain()

    def _fit_step(self, callback):
        self._optimizer_committing = False
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        retry = None
        try:
            self._fit_chunk(callback)
        except torch.cuda.OutOfMemoryError:
            if self._optimizer_committing:
                raise
            retry = "memory"
        except FloatingPointError:
            if self.execution_state["precision"] == "fp32":
                raise
            retry = "nonfinite_bf16"
        finally:
            end.record()
            self._device_events.append((start, end))
        # Retry outside the exception frame, after failed activation graphs are released.
        if retry:
            if retry == "nonfinite_bf16":
                self.execution_state["precision"] = "fp32"
            elif self.execution_state["microbatch"] > 16:
                self.execution_state["microbatch"] = max(
                    16, self.execution_state["microbatch"] // 2
                )
            elif not self.execution_state["checkpointing"]:
                self.execution_state["checkpointing"] = True
            else:
                self._restart_pass("memory_exhausted")
                raise RuntimeError(
                    "Encoder allocation failed after bounded memory fallbacks; no entities removed"
                )
            self._restart_pass(retry)
            if retry == "memory":
                torch.cuda.empty_cache()

    def _start_pass(self):
        if (
            self.execution_state["precision"] == "features_bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("BF16 fitting is unavailable; choose fit_precision=fp32")
        self.policy.optimizer.zero_grad(set_to_none=True)
        self._sequence_state = None
        self._pass_metrics = torch.zeros(5, device=self.device, dtype=torch.float64)
        self.policy.entity.microbatch = self.execution_state["microbatch"]
        self.policy.entity.activation_checkpointing = self.execution_state["checkpointing"]
        args = self._buffer, self.cfg["policy"]["chunk_length"], self.batch_size
        if self.cfg["training"].get("performance", {}).get("prefetch", False):
            try:
                self._sequence_iterator = SequencePrefetch(
                    *args, self.device, self.cfg["encoding"]["max_entities"]
                )
            except MemoryError:
                self._sequence_iterator = iter(sequence_batches(*args))
        else:
            self._sequence_iterator = iter(sequence_batches(*args))

    def _publish_fit_metrics(self):
        self._drain()
        loss, branch_sum, branch_n, tile_sum, tile_n = self._pass_metrics.cpu().tolist()
        self.logger.record("train/q_loss", loss)
        self.logger.record("train/branch_loss", branch_sum / branch_n if branch_n else None)
        self.logger.record("train/tile_loss", tile_sum / tile_n if tile_n else None)
        self._last_metrics_at = perf_counter()

    def _fit_chunk(self, callback):
        if self._fit_epoch >= self.n_epochs:
            self._phase(CohortPhase.SYNCHRONIZE, callback)
            return
        if getattr(self, "_sequence_iterator", None) is None:
            self._start_pass()
        item = next(self._sequence_iterator, None)
        if item is None:
            self._publish_fit_metrics()
            if not torch.isfinite(self._pass_metrics).all():
                raise FloatingPointError("Non-finite recurrent Q loss")
            try:
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                raise FloatingPointError("Non-finite recurrent gradients") from exc
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self._optimizer_committing = True
            self.policy.optimizer.step()
            self._optimizer_committing = False
            end.record()
            self._optimizer_events.append((start, end))
            self.policy.optimizer.zero_grad(set_to_none=True)
            self._stats["q_optimizer_steps"] += 1
            self._fit_epoch += 1
            self._close_sequences()
            self.logger.record("train/q_grad_norm", float(norm))
        else:
            reset, rows, observations, *metadata = item
            if reset:
                self._sequence_state = None
            self.policy.set_training_mode(True)
            with self.policy.fitting_precision(self.execution_state["precision"]):
                loss, state, branch_error, tile_error = sequence_loss(
                    self.policy,
                    rows,
                    self._sequence_state,
                    self._buffer.group_counts,
                    observations,
                    metadata[0] if metadata else None,
                )
                loss.backward()
            self._sequence_state = state
            self._pass_metrics += torch.stack(
                (
                    loss.detach().double(),
                    branch_error.detach().double().sum(),
                    loss.new_tensor(branch_error.numel()).double(),
                    tile_error.detach().double().sum(),
                    loss.new_tensor(tile_error.numel()).double(),
                )
            )
            if perf_counter() - self._last_metrics_at >= self.cfg["logging"]["progress_seconds"]:
                self._publish_fit_metrics()
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _synchronize(self, callback):
        self._drain()
        self._stats["optimizer_seconds"] = self._optimizer_seconds
        self._stats["fit_precision"] = self.execution_state["precision"]
        self._stats["execution_fallbacks"] = list(self.execution_state["fallbacks"])
        super()._synchronize(callback)

    def save(self, *args, **kwargs):
        self._drain()
        return super().save(*args, **kwargs)

    def learn(self, *args, **kwargs):
        try:
            return super().learn(*args, **kwargs)
        finally:
            self._drain()
            self._close_sequences()

    def _restore_runtime(self):
        super()._restore_runtime()
        self.policy.optimizer.zero_grad(set_to_none=True)
        self._sequence_iterator = self._sequence_state = None
        if not getattr(self, "execution_state", None):
            self._reset_execution()
        self._device_events = []
        self._optimizer_events = []
        self._last_metrics_at = 0.0
        self._optimizer_seconds = getattr(self, "_optimizer_seconds", 0.0)

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "_sequence_iterator",
            "_sequence_state",
            "_pass_metrics",
            "_device_events",
            "_optimizer_events",
            "_collection_host",
        ]
