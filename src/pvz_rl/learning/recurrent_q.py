"""Chronological complete-episode fitting on the shared CUDA cohort lifecycle."""

from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.cuda_features import REWARD_FIELDS
from pvz_rl.envs.encoding import ENTITY_WIDTH, PackedEntityBatch
from pvz_rl.envs.probes import CounterfactualCollector
from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.cuda_q import CudaCohortLifecycle
from pvz_rl.learning.exploration import CommittedPlantExploration
from pvz_rl.learning.objective import COMPONENTS, finalize_episode_metrics, probe_loss
from pvz_rl.learning.sequence_transport import SequencePrefetch, sequence_rows
from pvz_rl.policy.active_batch import ActiveBatchPlan
from pvz_rl.policy.cudagraph_backend import EncoderCompilationError
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.sequential_q import action_parts, balanced_q_loss
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy


def sequence_batches(buffer, chunk_length, decision_budget):
    """Chronological reference path, also used when staging cannot fit in RAM."""
    for reset, rows in sequence_rows(buffer, chunk_length, decision_budget):
        yield reset, rows, buffer.observations(rows)


def sequence_loss(
    policy,
    rows,
    state,
    counts,
    observations,
    fields=None,
    *,
    probe_counts=None,
    capture_diagnostics=False,
):
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
        events=tensor("events", torch.float32),
        memory_write=tensor("memory_write", torch.bool) & active,
        write_plan=rows["memory_write"] & rows["active"],
        active_plan=rows["active"],
        return_context=True,
    )
    branches, locations = action_parts(actions)
    if capture_diagnostics:
        selected = torch.zeros_like(q, dtype=torch.bool)
        selected[active] = torch.nn.functional.one_hot(branches, 10).bool()
        live = active[..., None].expand_as(selected)

        def capture(gradient):
            policy._output_gradient_norms = torch.stack(
                (gradient[selected].norm(), gradient[live & ~selected].norm())
            )

        q.register_hook(capture)
        live_sequences = active[:, -1]
        cell = state.cell[:, live_sequences].detach().abs().flatten()
        policy._cell_diagnostics = (
            torch.cat(
                (
                    torch.quantile(cell, cell.new_tensor([0.5, 0.9, 1.0])),
                    (cell > 5).float().mean()[None],
                )
            )
            if cell.numel()
            else q.new_zeros(4)
        )
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
        first,
        second,
        tensor("target", torch.float32)[active],
        actions,
        counts,
        accepted=tensor("accepted", torch.bool)[active],
        accepted_share=policy.cfg["training"]["objective"]["accepted_outcome_share"],
    )
    auxiliary = loss.new_zeros(())
    objective = policy.cfg["training"].get("objective", {})
    if probe_counts is not None and objective.get("probe_loss_weight", 0):
        if fields is not None:
            probes = fields.probes
        else:
            probe_rows = rows["probes"]
            probes = torch.as_tensor(
                np.stack(
                    [
                        probe_rows[k]
                        for k in ("valid", "branch_role", "action", "target", "accepted")
                    ],
                    -1,
                ),
                device=device,
                dtype=torch.float32,
            )
        auxiliary = probe_loss(
            policy,
            q[active],
            tiles[active],
            context[active],
            probes[active],
            probe_counts,
            objective["probe_huber_delta"],
        )
        loss = loss + objective["probe_loss_weight"] * auxiliary
    policy._last_probe_loss = auxiliary.detach()
    return loss, state.detach(), branch_error, tile_error[nonwait]


class CudaRecurrentQ(CudaCohortLifecycle):
    policy_protocol = "transformer_lstm_q_v3"
    optimizer_version = "complete_return_event_lstm_v1"

    def _setup_model(self):
        super()._setup_model()
        # SB3 restores ordinary attributes before rebuilding the policy. Keep a
        # saved cohort's FP32/memory fallback instead of overwriting it here.
        if not getattr(self, "execution_state", None):
            self._reset_execution()

    def _new_memory(self):
        return PolicyRunner(self.policy, self.cfg, self.env.batch.rules, self.n_envs, self.device)

    def _begin(self, callback):
        self._drain()
        self._close_sequences()
        boundary = self._phase_memory("before_collection_release")
        self.policy.entity.release_fitting()
        if getattr(self, "_teacher", None) is not None:
            self._teacher.entity.release_fitting()
        self.policy.entity.microbatch = self.cfg["policy"]["encoder_microbatch"]
        self.policy.entity.token_budget = self.cfg["policy"]["encoder_token_budget"]
        self.policy.entity.activation_checkpointing = False
        torch.cuda.empty_cache()
        self.env.cp.get_default_memory_pool().free_all_blocks()
        released = self._phase_memory("after_collection_release")
        self.collection_execution = {}
        super()._begin(callback)
        self._stats["phase_memory"] = [boundary, released]
        self._stats["collection_logical_rows"] = 0
        self._stats["collection_padded_rows"] = 0
        self._stats["collection_entity_widths"] = {}
        self._plant_exploration = CommittedPlantExploration(
            self.n_envs, self.device, self.plant_exploration_rate, self.exploration_rate
        )
        self._buffer.exploration = dict(
            plant_epsilon=self.plant_exploration_rate, tile_epsilon=self.exploration_rate
        )
        self._plant_costs = torch.tensor(
            [
                self.env.batch.rules.plants[kind]["cost"]
                for kind in self.cfg["environment"]["plants"]
            ],
            device=self.device,
        )
        self._ensure_teacher()
        self._teacher_memory = PolicyRunner(
            self._teacher, self.cfg, self.env.batch.rules, self.n_envs, self.device
        )
        if getattr(self, "_probes", None) is None:
            self._probes = CounterfactualCollector(self.env)
        self._probes.branch_cursor.zero_()
        self._probes.tile_cursor.zero_()
        self._probes.decision_cursor.zero_()
        self._probes.profiler.flush()
        self._probes.profiler.seconds.clear()
        if self.cfg["training"]["performance"]["compile_kernels"]:
            if self._teacher.entity.compilation_status == "disabled":
                self._teacher.entity.enable_compilation()
        self._sequence_iterator = self._sequence_state = None
        self._reset_execution()

    def _ensure_teacher(self):
        if getattr(self, "_teacher", None) is None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self._teacher = (
                    TransformerLSTMPolicy(self.cfg).to(self.device).eval().requires_grad_(False)
                )
            self._teacher.load_state_dict(self.policy.state_dict())
            self._teacher_version = getattr(self, "_teacher_version", 0)

    @torch.no_grad()
    def _update_teacher(self):
        self._ensure_teacher()
        decay = self.cfg["training"]["objective"]["ema_decay"]
        for target, source in zip(
            self._teacher.parameters(), self.policy.parameters(), strict=True
        ):
            target.lerp_(source, 1 - decay)
        self._teacher_version += 1

    def _finalize_rewards(self, callback):
        self._flush_probe_bootstrap(wait=True)
        self.env.handoff.clear()
        self._buffer.reserve_staging(0)
        summary = self._buffer.finalize_rewards()
        pending = getattr(self, "_pending_episodes", {})
        finalize_episode_metrics(
            [p["metrics"] for p in pending.values()],
            self.cfg,
            median=summary["duration_median_seconds"],
        )
        if hasattr(callback, "finalize_cohort_rewards"):
            callback.finalize_cohort_rewards(pending)
        self._phase(CohortPhase.RETURNS, callback)

    def _phase_memory(self, boundary):
        from pvz_rl.learning.performance import phase_memory

        memory = phase_memory(self.device)
        return dict(
            boundary=boundary,
            torch_allocated_mib=memory["allocated_mib"],
            torch_reserved_mib=memory["reserved_mib"],
            cupy_reserved_mib=self.env.cp.get_default_memory_pool().total_bytes() / 2**20,
            free_mib=memory["free_mib"],
            total_mib=memory["total_mib"],
        )

    def _phase(self, phase, callback):
        if phase == CohortPhase.FIT:
            self._flush_probe_bootstrap(wait=True)
            self.policy.entity.release_collection()
            if getattr(self, "_teacher", None) is not None:
                self._teacher.entity.release_collection()
            if hasattr(self, "_collection_packed"):
                del self._collection_packed
            if getattr(self, "_probes", None) is not None:
                self._probes.release_workspaces()
            torch.cuda.empty_cache()
            self.env.cp.get_default_memory_pool().free_all_blocks()
            self._stats.setdefault("phase_memory", []).append(self._phase_memory("fit_entry"))
        super()._phase(phase, callback)

    def _flush_probe_bootstrap(self, *, wait=False):
        pending = getattr(self, "_pending_probe_bootstrap", None)
        if pending is None:
            return
        if wait:
            self.env.handoff.wait()
        indices, host = pending
        self._buffer.update_probe_bootstrap(indices, host.numpy().reshape(-1, self.n_envs).T)
        self._pending_probe_bootstrap = None

    def _collect_step(self, callback):
        env = self.env
        with torch.no_grad(), env.device_context():
            active = env.enabled_envs.copy()
            enabled = torch.as_tensor(active, device=self.device)
            plan = ActiveBatchPlan.from_counts(
                active,
                env.features.summary_host[:, 0],
                env.features.limit,
                self.policy.entity.microbatch,
                self.policy.entity.token_budget,
            )
            live_slots = torch.as_tensor(plan.original_slot_ids, device=self.device)
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            rows["reset"] = self._first
            rows["active"], rows["env"] = active, np.arange(self.n_envs)
            masks = env.action_masks().clone()
            with env.profiler.track("online_inference"):
                result = self._memory.decide(
                    self._last_obs,
                    masks,
                    env.header_tensor[:, 0],
                    active=enabled,
                    deterministic=False,
                    defer_check=True,
                    plan=plan,
                )
            actions, values, tile_values, details = result
            actions, values, tile_values, details = self._plant_exploration.apply(
                self.policy,
                actions,
                details,
                masks,
                env.header_tensor[:, 2],
                torch.from_dlpack(env.batch.cooldowns),
                self._plant_costs,
                enabled,
                plan=plan,
            )
            self._memory.valid &= details["valid"]
            probe_host = self._probes.collect(
                self.policy,
                self._teacher_memory,
                self._last_obs,
                actions,
                details,
                masks,
                enabled,
                plan=plan,
            )
            if not hasattr(self, "_collection_packed"):
                self._collection_packed = torch.empty(
                    self.n_envs * env.features.limit,
                    ENTITY_WIDTH,
                    dtype=torch.int32,
                    device=self.device,
                )
            # Queue immutable public inputs and decision evidence before simulation.
            columns = torch.cat(
                (
                    details["events"].float(),
                    details["memory_write"][:, None].float(),
                    env.header_tensor[:, :1].float(),
                    actions[:, None].float(),
                    values[:, None],
                    tile_values[:, None],
                    details["greedy_actions"][:, None].float(),
                    details["coins"].float(),
                    details["branch_q"],
                    torch.from_dlpack(env.features.truncation_counts).float(),
                    details["policy_action"][:, None].float(),
                    details["exploration_mode"][:, None].float(),
                    details["pending_species"][:, None].float(),
                ),
                dim=1,
            )
            with env.profiler.track("encoding"):
                counts = torch.where(enabled, env.features.summary_tensor[: self.n_envs, 0], 0)
                offsets = counts.cumsum(0) - counts
                env.features.pack(enabled, offsets, self._collection_packed)
            with env.profiler.track("transfer"):
                host = env.handoff.enqueue_fields(
                    "behavior",
                    dict(
                        entities=self._collection_packed[
                            : int(env.features.summary_host[active, 0].sum())
                        ],
                        counts=counts,
                        offsets=offsets,
                        globals=self._last_obs.globals.index_select(0, live_slots),
                        action_masks=masks & enabled[:, None],
                        columns=columns,
                        valid=self._memory.valid,
                    ),
                )
            env.step_device(actions)
            species_coin = self._plant_exploration.observe(
                actions,
                env.header_tensor[:, 12].bool(),
                details["exploration_mode"],
                enabled,
                env._step_device_result[1],
            )
            with env.profiler.track("transfer"):
                exploration_host = env.handoff.enqueue_fields(
                    "exploration", dict(coins=species_coin, pending=self._plant_exploration.pending)
                )
            previous_wait = env.handoff.wait_seconds
            env.handoff.wait()
            self._flush_probe_bootstrap()
            self.policy.entity.release_retired()
            self._teacher.entity.release_retired()
            env.profiler.host("host_synchronization", env.handoff.wait_seconds - previous_wait)
            next_obs, _, dones, _, _, infos = env.step_host(autoreset=False)
            if not bool(host["valid"]):
                raise ValueError("Q values must be finite and every decision needs a legal action")
            globals_host = np.zeros((self.n_envs, self._last_obs.globals.shape[-1]), np.float32)
            globals_host[active] = host["globals"].numpy()
            observations = PackedEntityBatch(
                host["entities"].numpy()[: int(host["counts"].sum())],
                host["offsets"].numpy(),
                host["counts"].numpy(),
                globals_host,
            )
            packed = host["columns"].numpy()
            rows["entity_count"] = host["counts"].numpy()
            rows["events"], rows["memory_write"] = packed[:, :8], packed[:, 8]
            rows["tick"], rows["action"] = packed[:, 9], packed[:, 10]
            rows["branch_value"], rows["tile_value"] = packed[:, 11], packed[:, 12]
            rows["greedy_action"] = packed[:, 13]
            rows["species_coin"] = exploration_host["coins"].numpy()
            rows["tile_coin"] = packed[:, 15]
            rows["policy_action"] = packed[:, 29]
            rows["exploration_mode"] = packed[:, 30]
            rows["pending_species"] = packed[:, 31]
            rows["next_species"] = exploration_host["pending"].numpy()
            rows["entity_omitted"] = packed[:, 26:29]
            rows["mask"] = np.packbits(host["action_masks"].numpy(), axis=-1, bitorder="little")
            rows["reward"] = env.last_transition_host[:, 3]
            rows["duration"] = env.last_transition_host[:, 2]
            rows["done"] = env.last_transition_host[:, 0].astype(bool)
            rows["accepted"] = env.last_action_result_host[:, 0].astype(bool)
            rows["components"] = env.last_reward_parts_host[
                :, [REWARD_FIELDS.index(k) for k in COMPONENTS]
            ]
            rows["components_valid"] = active
            rows["won"] = rows["components"][:, 0] > 0
            rows["home_entries"] = env.last_reward_parts_host[
                :, [REWARD_FIELDS.index(k) for k in ("home_outer_entries", "home_inner_entries")]
            ]
            rows["probes"], probe_observations = self._probes.materialize(probe_host)
            if len(self._probes.profiler.pending) >= 128:
                self._probes.profiler.flush()
            staging = env.handoff.nbytes
            if not self._buffer.reserve_staging(staging):
                raise MemoryError("RAM budget cannot hold counterfactual collection staging")
            self._memory.observe_result(
                env.features.history_tensor,
                env.transition_ticks,
                active=enabled,
            )
            self._teacher_memory.observe_result(
                env.features.history_tensor, env.transition_ticks, active=enabled
            )
            presentation_started = perf_counter()
            env.action_journal.record_batch_safe(
                rows,
                packed[:, 16:26],
                env.last_action_result_host,
                env._episode_serial,
            )
            env.profiler.host("presentation", perf_counter() - presentation_started)
            rows["executed_action"] = np.where(rows["accepted"], rows["action"], 0)
            rows[~active] = np.zeros((), dtype=rows.dtype)
            rows["env"] = np.arange(self.n_envs)
            storage_started = perf_counter()
            start_row = self._buffer.size
            self._buffer.append(rows, observations, probe_observations)
            self._pending_probe_bootstrap = (
                start_row + np.arange(self.n_envs),
                self._probes.bootstrap(probe_host),
            )
            if not self._buffer.reserve_staging(env.handoff.nbytes):
                raise MemoryError("RAM budget cannot hold compact probe bootstrap staging")
            env.profiler.host("storage", perf_counter() - storage_started)
            self._stats["collection_logical_rows"] = (
                self._stats.get("collection_logical_rows", 0) + plan.logical_rows
            )
            self._stats["collection_padded_rows"] = (
                self._stats.get("collection_padded_rows", 0) + plan.padded_rows
            )
            widths = self._stats.setdefault("collection_entity_widths", {})
            for bucket in plan.buckets:
                key = str(bucket.entity_width)
                widths[key] = widths.get(key, 0) + len(bucket.slot_ids)
            probe_totals = self._stats.setdefault(
                "probe_inference_rows", dict(logical_rows=0, unique_rows=0, padded_rows=0)
            )
            for key in probe_totals:
                probe_totals[key] += self._probes.inference_metrics[key]
            self.collection_execution = dict(
                scope="current_cohort",
                active_games=int(env.enabled_envs.sum()),
                logical_rows=self._stats["collection_logical_rows"],
                padded_rows=self._stats["collection_padded_rows"],
                entity_widths=dict(widths),
                current_logical_rows=plan.logical_rows,
                current_padded_rows=plan.padded_rows,
                online_rows=dict(logical=plan.logical_rows, padded=plan.padded_rows),
                actual_ema_rows=dict(logical=plan.logical_rows, padded=plan.padded_rows),
                graph_cache={
                    name: dict(network.entity.collection_metrics)
                    for name, network in (("online", self.policy), ("ema", self._teacher))
                },
                probe_inference=dict(self._probes.inference_metrics),
                probe_inference_totals=dict(probe_totals),
                phase_memory=list(self._stats.get("phase_memory", [])),
                timings=env.profiler.snapshot(),
            )
            self._planting_samples += int(
                np.count_nonzero(active & (rows["action"] > 0) & (rows["action"] < 361))
            )
            self._stats["species_exploration_coins"] += int(rows["species_coin"][active].sum())
            self._stats["tile_exploration_coins"] += int(rows["tile_coin"][active].sum())
            for name, selected in (
                ("committed_wait_decisions", rows["exploration_mode"] == 1),
                ("committed_plant_attempts", rows["exploration_mode"] == 2),
                ("committed_plant_successes", (rows["exploration_mode"] == 2) & rows["accepted"]),
            ):
                self._stats[name] = self._stats.get(name, 0) + int(
                    np.count_nonzero(active & selected)
                )
            self._stats["exploratory_changes"] += int(
                np.count_nonzero(active & (rows["action"] != rows["greedy_action"]))
            )
            self._simulation_ticks += int(rows["duration"][active].sum())
            self._first, self._last_obs = False, next_obs
            self.num_timesteps += int(active.sum())
            active_infos = [info for info, on in zip(infos, active) if on]
            for index, info in enumerate(infos):
                if "episode_metrics" in info:
                    info["episode_metrics"]["time_reward_pending"] = True
                    info["episode_slot"] = index
                    self._pending_episodes[index] = dict(
                        metrics=info["episode_metrics"],
                        record_id=f"cohort:{self._n_updates}:slot:{index}",
                    )
            if not hasattr(callback, "cfg"):
                self.training_games += sum("episode_metrics" in info for info in active_infos)
            callback.update_locals({"infos": active_infos, "dones": dones})
            # Publish the phase before callbacks may request an interruption.
            if not env.enabled_envs.any():
                self._phase(CohortPhase.FINALIZE_REWARDS, callback)
            if not callback.on_step():
                raise KeyboardInterrupt

    def _reset_execution(self):
        precision = self.cfg["training"].get("performance", {}).get("fit_precision", "fp32")
        performance = self.cfg["training"].get("performance", {})
        self.execution_state = dict(
            requested_precision=precision,
            precision=precision,
            microbatch=self.cfg["policy"]["encoder_microbatch"],
            token_budget=self.cfg["policy"]["encoder_token_budget"],
            fit_sequence_groups=performance.get("fit_sequence_groups", 1),
            checkpointing=False,
            fallbacks=[],
        )
        self._device_events = []
        self._optimizer_events = []
        self._device_seconds = getattr(self, "_device_seconds", 0.0)
        self._optimizer_seconds = 0.0
        self._last_metrics_at = 0.0
        self._cached_metrics = (None, None, None)
        self._fit_pass_start = None

    def configure_execution(self, cfg):
        """Apply only approved execution settings to a resumed model."""
        from pvz_rl.learning.performance import refresh_performance

        previous = self.cfg
        self.policy.cfg = refresh_performance(previous, cfg)
        self.policy.cfg["logging"] = dict(cfg["logging"])
        self.policy.features_extractor.cfg = self.policy.cfg
        self.policy_kwargs["features_extractor_kwargs"]["layout_cfg"] = self.policy.cfg
        if (
            getattr(self, "execution_state", {}).get("requested_precision")
            != cfg["training"]["performance"].get("fit_precision", "fp32")
            or previous["policy"]["encoder_microbatch"] != cfg["policy"]["encoder_microbatch"]
            or previous["policy"]["encoder_token_budget"] != cfg["policy"]["encoder_token_budget"]
            or previous["training"].get("performance", {}).get("fit_sequence_groups", 1)
            != cfg["training"].get("performance", {}).get("fit_sequence_groups", 1)
        ):
            self._reset_execution()
        self.policy.entity.token_budget = cfg["policy"]["encoder_token_budget"]
        if self.phase != CohortPhase.FIT:
            self.policy.entity.microbatch = cfg["policy"]["encoder_microbatch"]
        self.policy.entity.configure_collection(self.cfg["training"]["performance"])
        if getattr(self, "_teacher", None) is not None:
            self._teacher.entity.configure_collection(self.cfg["training"]["performance"])
        for layer in self.policy.entity.layers:
            layer.query_chunk = cfg["policy"]["attention_query_chunk"]

    def _drain(self):
        self._flush_probe_bootstrap(wait=True)
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
        self._fit_pass_start = None
        self._drain()

    def _fit_step(self, callback):
        self._optimizer_committing = False
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
        except EncoderCompilationError as exc:
            if (
                self._optimizer_committing
                or "encoder_compilation" in self.execution_state["fallbacks"]
            ):
                raise
            self.policy.entity.disable_compilation(exc)
            retry = "encoder_compilation"
        # Retry outside the exception frame, after failed activation graphs are released.
        if retry:
            if retry == "nonfinite_bf16":
                self.execution_state["precision"] = "fp32"
            elif retry == "encoder_compilation":
                pass  # Replay the uncommitted pass eagerly with the same precision.
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
        self._ensure_teacher()
        if (
            self.execution_state["precision"] == "features_bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("BF16 fitting is unavailable; choose fit_precision=fp32")
        self.policy.optimizer.zero_grad(set_to_none=True)
        self._sequence_state = None
        self._diagnostics_sampled = False
        self._pass_metrics = torch.zeros(6, device=self.device, dtype=torch.float64)
        self.policy.entity.microbatch = self.execution_state["microbatch"]
        self.policy.entity.token_budget = self.execution_state["token_budget"]
        self.policy.entity.activation_checkpointing = self.execution_state["checkpointing"]
        chunk_length = self.cfg["policy"]["chunk_length"]
        groups = self.execution_state["fit_sequence_groups"]
        fit_batch_size = min(self.batch_size * groups, self.n_envs * chunk_length)
        args = self._buffer, chunk_length, fit_batch_size
        self._fit_pass_start = torch.cuda.Event(enable_timing=True)
        self._fit_pass_start.record()
        if self.cfg["training"].get("performance", {}).get("prefetch", False):
            try:
                self._sequence_iterator = SequencePrefetch(
                    *args, self.device, self.cfg["encoding"]["max_entities"]
                )
            except MemoryError:
                self._sequence_iterator = iter(sequence_batches(*args))
        else:
            self._sequence_iterator = iter(sequence_batches(*args))

    def _publish_fit_metrics(self, *, final=False):
        if final:
            loss, branch_sum, branch_n, tile_sum, tile_n, probe = self._pass_metrics.cpu().tolist()
            self.logger.record("train/probe_loss", probe)
            if getattr(self.policy, "_output_gradient_norms", None) is not None:
                diagnostic = (
                    torch.cat((self.policy._output_gradient_norms, self.policy._cell_diagnostics))
                    .cpu()
                    .tolist()
                )
                for key, value in zip(
                    (
                        "selected_output_gradient_norm",
                        "unselected_output_gradient_norm",
                        "cell_abs_median",
                        "cell_abs_p90",
                        "cell_abs_max",
                        "cell_saturation_fraction",
                    ),
                    diagnostic,
                ):
                    self.logger.record("train/" + key, value)
            self._cached_metrics = (
                loss,
                branch_sum / branch_n if branch_n else None,
                tile_sum / tile_n if tile_n else None,
            )
            _, branch_loss, tile_loss = self._cached_metrics
        else:
            loss, branch_loss, tile_loss = self._cached_metrics
        self.logger.record("train/q_loss", loss)
        self.logger.record("train/branch_loss", branch_loss)
        self.logger.record("train/tile_loss", tile_loss)
        self._last_metrics_at = perf_counter()

    def _fit_chunk(self, callback):
        if self._fit_epoch >= self.n_epochs:
            self._phase(CohortPhase.SYNCHRONIZE, callback)
            return
        if getattr(self, "_sequence_iterator", None) is None:
            self._start_pass()
        item = next(self._sequence_iterator, None)
        if item is None:
            if self._fit_pass_start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._device_events.append((self._fit_pass_start, end))
                self._fit_pass_start = None
            self._publish_fit_metrics(final=True)
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
            self._update_teacher()
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
                    self._buffer.outcome_counts,
                    observations,
                    metadata[0] if metadata else None,
                    probe_counts=self._buffer.probe_counts,
                    capture_diagnostics=not self._diagnostics_sampled,
                )
                loss.backward()
            self._diagnostics_sampled = True
            self._sequence_state = state
            self._pass_metrics += torch.stack(
                (
                    loss.detach().double(),
                    branch_error.detach().double().sum(),
                    loss.new_tensor(branch_error.numel()).double(),
                    tile_error.detach().double().sum(),
                    loss.new_tensor(tile_error.numel()).double(),
                    self.policy._last_probe_loss.double(),
                )
            )
            if perf_counter() - self._last_metrics_at >= self.cfg["logging"]["progress_seconds"]:
                self._publish_fit_metrics()
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _synchronize(self, callback):
        self._drain()
        self._stats.update(getattr(self, "_probes").profiler.flush())
        self._stats["trajectory_bytes"] = (
            self._buffer.size * self._buffer.dtype.itemsize + self._buffer.entity_size * 44
        )
        self._stats["trajectory_ram_bytes"] = self._buffer.ram_used
        self._stats["ema_version"] = self._teacher_version
        self._stats["probe_scratch_lanes"] = self._probes.lanes
        self._stats["ema_compilation_status"] = self._teacher.entity.compilation_status
        self._stats["optimizer_seconds"] = self._optimizer_seconds
        self._stats["fit_precision"] = self.execution_state["precision"]
        self._stats["fit_sequence_groups"] = self.execution_state["fit_sequence_groups"]
        self._stats["encoder_microbatch"] = self.execution_state["microbatch"]
        self._stats["encoder_token_budget"] = self.execution_state["token_budget"]
        self._stats["execution_fallbacks"] = list(self.execution_state["fallbacks"])
        self._stats["compilation_status"] = getattr(self.policy, "compilation_status", "disabled")
        self._stats["compilation_backend"] = getattr(self.policy, "compilation_backend", None)
        self._stats["collection_graphs"] = {
            name: dict(network.entity.collection_metrics)
            for name, network in (("online", self.policy), ("ema", self._teacher))
        }
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
        extra = self.runtime_state
        if extra and extra.get("environment") is not None:
            if extra.get("objective") != self.cfg["training"].get("objective"):
                raise ValueError("Checkpoint objective differs; use weights-only initialization")
            if self.phase != CohortPhase.IDLE and any(
                extra.get(key) is None
                for key in (
                    "teacher",
                    "teacher_memory",
                    "probe_schedule",
                    "pending_episodes",
                    "compilation",
                    "plant_exploration",
                )
            ):
                raise ValueError("Incomplete objective recovery state")
        super()._restore_runtime()
        if extra and extra.get("plant_exploration") is not None:
            self._plant_exploration = CommittedPlantExploration(
                self.n_envs, self.device, self.plant_exploration_rate, self.exploration_rate
            )
            self._plant_exploration.restore(extra["plant_exploration"])
            if self._buffer is not None and self._buffer.exploration != dict(
                plant_epsilon=self.plant_exploration_rate, tile_epsilon=self.exploration_rate
            ):
                raise ValueError("Trajectory exploration probabilities differ from recovery state")
            self._plant_costs = torch.tensor(
                [
                    self.env.batch.rules.plants[kind]["cost"]
                    for kind in self.cfg["environment"]["plants"]
                ],
                device=self.device,
            )
        if self._buffer is not None and self._buffer.cfg != self.cfg:
            from pvz_rl.learning.training_requirements import resume_protocol

            if resume_protocol(self._buffer.cfg, "masked") != resume_protocol(self.cfg, "masked"):
                raise ValueError("Trajectory objective/configuration differs from policy")
        if extra and "pending_episodes" in extra:
            self._pending_episodes = extra["pending_episodes"]
        if extra and "teacher" in extra:
            self._ensure_teacher()
            self._teacher.load_state_dict(extra["teacher"])
            self._teacher_version = extra["teacher_version"]
            if extra.get("teacher_memory") is not None:
                self._teacher_memory = PolicyRunner(
                    self._teacher, self.cfg, self.env.batch.rules, self.n_envs, self.device
                )
                self._teacher_memory.restore(extra["teacher_memory"])
            if extra.get("probe_schedule") is not None:
                self._probes = CounterfactualCollector(self.env)
                self._probes.restore(extra["probe_schedule"])
            for name, network in (("policy", self.policy), ("teacher", self._teacher)):
                compilation = extra["compilation"][name]
                shapes = compilation.get("shapes")
                if (
                    compilation.get("status")
                    not in ("disabled", "requested", "compiled", "fallback", "unavailable")
                    or not isinstance(shapes, list)
                    or len(shapes) > 4
                    or any(
                        not isinstance(shape, (list, tuple))
                        or len(shape) != 2
                        or any(type(size) is not int or size < 0 for size in shape)
                        or shape[0] < 1
                        for shape in shapes
                    )
                ):
                    raise ValueError("Invalid event-memory encoder execution recovery state")
                if (
                    compilation["status"] != "fallback"
                    and self.cfg["training"]["performance"]["compile_kernels"]
                ):
                    network.entity.enable_compilation()
                elif compilation["status"] == "fallback":
                    network.entity.disable_compilation(compilation["error"])
        self.policy.optimizer.zero_grad(set_to_none=True)
        self._sequence_iterator = self._sequence_state = None
        if not getattr(self, "execution_state", None):
            self._reset_execution()
        self._device_events = []
        self._optimizer_events = []
        self._last_metrics_at = 0.0
        self._optimizer_seconds = getattr(self, "_optimizer_seconds", 0.0)
        self._fit_pass_start = None
        self._pending_probe_bootstrap = None
        self.collection_execution = {}
        if self.phase == CohortPhase.COLLECT:
            for network in (self.policy, getattr(self, "_teacher", None)):
                if network is not None:
                    network.entity.microbatch = self.cfg["policy"]["encoder_microbatch"]
                    network.entity.token_budget = self.cfg["policy"]["encoder_token_budget"]
                    network.entity.activation_checkpointing = False
        if self.phase == CohortPhase.FIT and getattr(self, "_probes", None) is not None:
            self._probes.release_workspaces()

    def _extra_runtime(self):
        if getattr(self, "_teacher", None) is None:
            return {}
        return dict(
            plant_exploration=self._plant_exploration.snapshot()
            if getattr(self, "_plant_exploration", None) is not None
            else None,
            teacher={k: v.cpu() for k, v in self._teacher.state_dict().items()},
            teacher_version=self._teacher_version,
            teacher_memory=self._teacher_memory.snapshot()
            if getattr(self, "_teacher_memory", None)
            else None,
            probe_schedule=self._probes.snapshot() if getattr(self, "_probes", None) else None,
            pending_episodes=getattr(self, "_pending_episodes", {}),
            objective=dict(self.cfg["training"]["objective"]),
            compilation={
                name: dict(
                    status=network.entity.compilation_status,
                    error=network.entity.compilation_error,
                    shapes=[],
                )
                for name, network in (("policy", self.policy), ("teacher", self._teacher))
            },
        )

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "_sequence_iterator",
            "_sequence_state",
            "_pass_metrics",
            "_device_events",
            "_optimizer_events",
            "_fit_pass_start",
            "_collection_packed",
            "_pending_probe_bootstrap",
            "_teacher",
            "_teacher_memory",
            "_plant_exploration",
            "_plant_costs",
            "_probes",
            "_pending_episodes",
        ]
