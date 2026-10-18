"""Stateless two-stage fitting on the shared CUDA cohort lifecycle."""

from contextlib import closing
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.cuda_features import REWARD_FIELDS
from pvz_rl.envs.encoding import ENTITY_WIDTH, PackedEntityBatch
from pvz_rl.learning.action_routing import classify_actions
from pvz_rl.learning.batch_transport import (
    BatchPrefetch,
    FittingSpool,
    batch_rows,
    fitting_metadata,
)
from pvz_rl.learning.cohort import CohortPhase, atomic_transition, check_transition_interruption
from pvz_rl.learning.collection_scheduler import CollectionScheduler
from pvz_rl.learning.cuda_q import CudaCohortLifecycle
from pvz_rl.learning.exploration import CommittedPlantExploration
from pvz_rl.learning.flat_q import batch_loss
from pvz_rl.learning.objective import (
    COMPONENTS,
    OBJECTIVE_PROTOCOL,
    consistency_diagnostics,
    finalize_episode_metrics,
)
from pvz_rl.policy.active_batch import ActiveBatchPlan
from pvz_rl.policy.cudagraph_backend import EncoderCompilationError
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy


def flat_batches(buffer, decision_budget, spool=None):
    if spool is not None:
        for rows, packed in spool.iter_batches(decision_budget):
            width = spool.buckets[np.searchsorted(spool.buckets, int(packed.counts.max(initial=0)))]
            yield (
                rows,
                buffer.observations_from_packed(packed, validated=True, width=width),
                fitting_metadata(rows, "cpu"),
            )
        return
    with closing(batch_rows(buffer, decision_budget, spool=spool)) as source:
        for rows in source:
            yield rows, buffer.observations(rows)


class CudaSequentialQ(CudaCohortLifecycle):
    policy_protocol = "two_stage_transformer_q_v1"
    optimizer_version = OBJECTIVE_PROTOCOL

    def _setup_model(self):
        super()._setup_model()
        # SB3 restores ordinary attributes before rebuilding the policy. Keep a
        # saved cohort's FP32/memory fallback instead of overwriting it here.
        if not getattr(self, "execution_state", None):
            self._reset_execution()

    def _new_runner(self):
        return PolicyRunner(self.policy)

    def _begin(self, callback):
        self._drain()
        self._close_batches()
        boundary = self._phase_memory("before_collection_release")
        for encoder in self.policy.encoders:
            encoder.release_fitting()
        if getattr(self, "_teacher", None) is not None:
            for encoder in self._teacher.encoders:
                encoder.release_fitting()
        for encoder in self.policy.encoders:
            encoder.microbatch = self.cfg["policy"]["encoder_microbatch"]
            encoder.token_budget = self.cfg["policy"]["encoder_token_budget"]
            encoder.activation_checkpointing = False
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
        self._scheduler = CollectionScheduler(self.n_envs, self.device)
        if self.cfg["training"]["performance"]["compile_kernels"]:
            for encoder in self._teacher.encoders:
                if encoder.compilation_status == "disabled":
                    encoder.enable_compilation()
        self._batch_iterator = None
        self._fitting_spool = None
        self._reset_execution()

    def _ensure_teacher(self):
        if getattr(self, "_teacher", None) is None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self._teacher = (
                    TwoStageTransformerPolicy(self.cfg).to(self.device).eval().requires_grad_(False)
                )
            self._teacher.load_state_dict(self.policy.runtime_state_dict())
            self._teacher_version = getattr(self, "_teacher_version", 0)

    @torch.no_grad()
    def _update_teacher(self):
        self._ensure_teacher()
        decay = self.cfg["training"]["objective"]["ema_decay"]
        for target, source in zip(
            self._teacher.runtime_parameters(), self.policy.runtime_parameters(), strict=True
        ):
            target.lerp_(source, 1 - decay)
        self._teacher_version += 1

    def _finalize_rewards(self, callback):
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
            for encoder in self.policy.encoders:
                encoder.release_collection()
            if getattr(self, "_teacher", None) is not None:
                for encoder in self._teacher.encoders:
                    encoder.release_collection()
            if hasattr(self, "_collection_packed"):
                del self._collection_packed
            torch.cuda.empty_cache()
            self.env.cp.get_default_memory_pool().free_all_blocks()
            self._stats.setdefault("phase_memory", []).append(self._phase_memory("fit_entry"))
        super()._phase(phase, callback)

    def _collect_step(self, callback, *, draining=False):
        if not draining:
            check_transition_interruption()
        self._collection_callback = callback
        env = self.env
        with torch.no_grad(), env.device_context():
            unfinished = env.enabled_envs.copy()
            new = self._scheduler.ready(unfinished)
            issued_host = new | self._scheduler.pending
            enabled = torch.as_tensor(new, device=self.device)
            plan = ActiveBatchPlan.from_counts(
                new,
                env.features.summary_host[:, 0],
                env.features.limit,
                self.policy.species_encoder.microbatch,
                self.policy.species_encoder.token_budget,
            )
            live_slots = torch.as_tensor(np.flatnonzero(issued_host), device=self.device)
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            rows["reset"] = self._scheduler.resets
            rows["env"] = np.arange(self.n_envs)
            masks = env.action_masks().clone()
            if new.any():
                with env.profiler.track("online_inference"):
                    result = self._runner.decide(
                        self._last_obs,
                        masks,
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
                self._runner.valid &= details["valid"]
            if not hasattr(self, "_collection_packed"):
                self._collection_packed = torch.empty(
                    self.n_envs * env.features.limit,
                    ENTITY_WIDTH,
                    dtype=torch.int32,
                    device=self.device,
                )
            # Queue immutable public inputs and decision evidence before simulation.
            if new.any():
                columns = torch.cat(
                    (
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
                        details["placement_gap"][:, None],
                    ),
                    dim=1,
                )
                self._scheduler.retain_species_draws(
                    new,
                    torch.rand(self.n_envs, device=self.device),
                    torch.randint(
                        1,
                        len(self.cfg["environment"]["plants"]),
                        (self.n_envs,),
                        device=self.device,
                    ),
                )
            else:
                columns = torch.zeros(self.n_envs, 24, device=self.device)
            columns, issued = self._scheduler.retain(columns, new)
            actions = columns[:, 1].long()
            executing = issued
            with env.profiler.track("encoding"):
                counts = torch.where(executing, env.features.summary_tensor[: self.n_envs, 0], 0)
                offsets = counts.cumsum(0) - counts
                env.features.pack(executing, offsets, self._collection_packed)
            with env.profiler.track("transfer"):
                host = env.handoff.enqueue_fields(
                    "behavior",
                    dict(
                        entities=self._collection_packed[
                            : int(env.features.summary_host[issued_host, 0].sum())
                        ],
                        counts=counts,
                        offsets=offsets,
                        globals=self._last_obs.globals.index_select(0, live_slots),
                        action_masks=masks & executing[:, None],
                        plant_available=(env.header_tensor[:, 2, None] >= self._plant_costs)
                        & (torch.from_dlpack(env.batch.cooldowns) == 0),
                        columns=columns,
                        valid=self._runner.valid,
                    ),
                )
            env.step_device(actions, active=executing)
            species_coin = self._plant_exploration.observe(
                actions,
                env.header_tensor[:, 12].bool(),
                columns[:, 21].long(),
                executing,
                env._step_device_result[1],
                draws=(self._scheduler.species_coins, self._scheduler.species_choices),
            )
            with env.profiler.track("transfer"):
                exploration_host = env.handoff.enqueue_fields(
                    "exploration", dict(coins=species_coin, pending=self._plant_exploration.pending)
                )
            previous_wait = env.handoff.wait_seconds
            env.handoff.wait()
            for encoder in self.policy.encoders:
                encoder.release_retired()
            for encoder in self._teacher.encoders:
                encoder.release_retired()
            env.profiler.host("host_synchronization", env.handoff.wait_seconds - previous_wait)
            active = executing.cpu().numpy().copy()
            next_obs, _, dones, _, _, infos = env.step_host(autoreset=False, active=active)
            rows["active"] = active
            self._scheduler.commit(issued_host, active)
            if not bool(host["valid"]):
                raise ValueError("Q values must be finite and every decision needs a legal action")
            globals_host = np.zeros((self.n_envs, self._last_obs.globals.shape[-1]), np.float32)
            globals_host[issued_host] = host["globals"].numpy()
            globals_host[~active] = 0
            observations = PackedEntityBatch(
                host["entities"].numpy()[: int(host["counts"].sum())],
                host["offsets"].numpy(),
                host["counts"].numpy(),
                globals_host,
            )
            packed = host["columns"].numpy()
            rows["entity_count"] = host["counts"].numpy()
            rows["tick"], rows["action"] = packed[:, 0], packed[:, 1]
            rows["branch_value"], rows["tile_value"] = packed[:, 2], packed[:, 3]
            rows["placement_gap"] = packed[:, 23]
            rows["greedy_action"] = packed[:, 4]
            rows["species_coin"] = exploration_host["coins"].numpy()
            rows["tile_coin"] = packed[:, 6]
            rows["policy_action"] = packed[:, 20]
            rows["exploration_mode"] = packed[:, 21]
            rows["pending_species"] = packed[:, 22]
            rows["next_species"] = exploration_host["pending"].numpy()
            rows["entity_omitted"] = packed[:, 17:20]
            rows["mask"] = np.packbits(host["action_masks"].numpy(), axis=-1, bitorder="little")
            rows["reward"] = env.last_transition_host[:, 3]
            rows["duration"] = env.last_transition_host[:, 2]
            rows["done"] = env.last_transition_host[:, 0].astype(bool)
            rows["accepted"] = env.last_action_result_host[:, 0].astype(bool)
            rows["rejection_reason"] = env.last_action_result_host[:, 1]
            # Policy masks remain geometry-only. Attribution uses immutable
            # pre-action card facts captured through the existing stream fence.
            legal = host["action_masks"].numpy().copy()
            plant_geometry = legal[:, 1:361].reshape(self.n_envs, 8, 45)
            occupied = ~plant_geometry[:, 0].copy()
            plant_geometry &= host["plant_available"].numpy()[:, :, None]
            legal[:, 361:] = occupied & active[:, None]
            rows["route"], rows["branch_usable"] = classify_actions(
                rows["action"], rows["accepted"], legal, rows["rejection_reason"]
            )
            rows["components"] = env.last_reward_parts_host[
                :, [REWARD_FIELDS.index(k) for k in COMPONENTS]
            ]
            rows["components_valid"] = active
            rows["won"] = rows["components"][:, 0] > 0
            rows["home_entries"] = env.last_reward_parts_host[
                :, [REWARD_FIELDS.index(k) for k in ("home_outer_entries", "home_inner_entries")]
            ]
            staging = env.handoff.nbytes + self._scheduler.retained_bytes
            if not self._buffer.reserve_staging(staging):
                raise MemoryError("RAM budget cannot hold actual collection staging")
            presentation_started = perf_counter()
            env.action_journal.record_batch_safe(
                rows,
                packed[:, 7:17],
                env.last_action_result_host,
                env._episode_serial,
            )
            env.profiler.host("presentation", perf_counter() - presentation_started)
            rows["executed_action"] = np.where(rows["accepted"], rows["action"], 0)
            rows[~active] = np.zeros((), dtype=rows.dtype)
            rows["env"] = np.arange(self.n_envs)
            storage_started = perf_counter()
            if active.any():
                self._buffer.append(rows, observations)
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
            self.collection_execution = dict(
                capacity_blocked_slots=int(self._scheduler.pending.sum()),
                scope="current_cohort",
                active_games=int(env.enabled_envs.sum()),
                logical_rows=self._stats["collection_logical_rows"],
                padded_rows=self._stats["collection_padded_rows"],
                entity_widths=dict(widths),
                current_logical_rows=plan.logical_rows,
                current_padded_rows=plan.padded_rows,
                online_rows=dict(logical=plan.logical_rows, padded=plan.padded_rows),
                actual_ema_rows=dict(logical=0, padded=0),
                graph_cache={
                    name: {
                        stage: dict(encoder.collection_metrics)
                        for stage, encoder in zip(
                            ("species", "tile"), network.encoders, strict=True
                        )
                    }
                    for name, network in (("online", self.policy), ("ema", self._teacher))
                },
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
            if not env.enabled_envs.any() and not self._scheduler.pending.any():
                self._phase(CohortPhase.FINALIZE_REWARDS, callback)
            if not callback.on_step() and not draining:
                raise KeyboardInterrupt

    def _reset_execution(self):
        precision = self.cfg["training"].get("performance", {}).get("fit_precision", "bf16_master")
        performance = self.cfg["training"].get("performance", {})
        self.execution_state = dict(
            requested_precision=precision,
            precision=precision,
            microbatch=max(
                performance.get(
                    "fit_species_microbatch", performance.get("fit_encoder_microbatch", 192)
                ),
                performance.get(
                    "fit_tile_microbatch", performance.get("fit_encoder_microbatch", 192)
                ),
            ),
            species_microbatch=performance.get(
                "fit_species_microbatch", performance.get("fit_encoder_microbatch", 192)
            ),
            tile_microbatch=performance.get(
                "fit_tile_microbatch", performance.get("fit_encoder_microbatch", 192)
            ),
            token_budget=performance.get("fit_token_budget", 65536),
            fit_batch_groups=performance.get("fit_batch_groups", 1),
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
            != cfg["training"]["performance"].get("fit_precision", "bf16_master")
            or previous["training"].get("performance") != cfg["training"].get("performance")
            or previous["policy"]["encoder_microbatch"] != cfg["policy"]["encoder_microbatch"]
            or previous["policy"]["encoder_token_budget"] != cfg["policy"]["encoder_token_budget"]
            or previous["training"].get("performance", {}).get("fit_batch_groups", 1)
            != cfg["training"].get("performance", {}).get("fit_batch_groups", 1)
        ):
            self._reset_execution()
        for encoder in self.policy.encoders:
            encoder.token_budget = cfg["policy"]["encoder_token_budget"]
            if self.phase != CohortPhase.FIT:
                encoder.microbatch = cfg["policy"]["encoder_microbatch"]
            encoder.configure_collection(self.cfg["training"]["performance"])
        if getattr(self, "_teacher", None) is not None:
            for encoder in self._teacher.encoders:
                encoder.configure_collection(self.cfg["training"]["performance"])
        for encoder in self.policy.encoders:
            for layer in encoder.layers:
                layer.query_chunk = cfg["policy"]["attention_query_chunk"]

    def _drain(self):
        iterator = getattr(self, "_batch_iterator", None)
        if isinstance(iterator, BatchPrefetch):
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

    def _close_batches(self):
        iterator = getattr(self, "_batch_iterator", None)
        if iterator is not None:
            iterator.close()
        self._batch_iterator = None

    def _restart_pass(self, reason):
        self._close_batches()
        self.policy.zero_grad(set_to_none=True)
        self.execution_state["fallbacks"].append(reason)
        self._fit_pass_start = None
        self._drain()
        for network in (self.policy, getattr(self, "_teacher", None)):
            if network is not None:
                for encoder in network.encoders:
                    encoder.release_fitting()
                    encoder.release_collection()

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
            self._restart_pass("nonfinite_bf16")
            raise
        except EncoderCompilationError as exc:
            if (
                self._optimizer_committing
                or "encoder_compilation" in self.execution_state["fallbacks"]
            ):
                raise
            for encoder in self.policy.encoders:
                encoder.disable_compilation(exc)
            retry = "encoder_compilation"
        # Retry outside the exception frame, after failed activation graphs are released.
        if retry:
            if retry == "encoder_compilation":
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
        if self.execution_state["precision"] != "bf16_master":
            raise RuntimeError("BF16 master-weight fitting is required")
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 runtime inference is unavailable on this CUDA device")
        self.policy.zero_grad(set_to_none=True)
        self._pass_metrics = torch.zeros(10, device=self.device, dtype=torch.float64)
        for encoder in self.policy.encoders:
            encoder.fitting = True
            encoder.microbatch = min(
                self.execution_state["microbatch"],
                self.execution_state.get(
                    f"{encoder.stage}_microbatch", self.execution_state["microbatch"]
                ),
            )
            encoder.token_budget = self.execution_state["token_budget"]
            encoder.activation_checkpointing = self.execution_state["checkpointing"]
        for encoder in self._teacher.encoders:
            encoder.fitting = True
            encoder.microbatch = min(
                self.execution_state["microbatch"],
                self.execution_state.get(
                    f"{encoder.stage}_microbatch", self.execution_state["microbatch"]
                ),
            )
            encoder.token_budget = self.execution_state["token_budget"]
        groups = self.execution_state["fit_batch_groups"]
        if getattr(self, "_fitting_spool", None) is None:
            self._fitting_spool = FittingSpool(self._buffer)
        args = self._buffer, self.batch_size * groups, self._fitting_spool
        self._fit_pass_start = torch.cuda.Event(enable_timing=True)
        self._fit_pass_start.record()
        if self.cfg["training"].get("performance", {}).get("prefetch", False):
            try:
                self._batch_iterator = BatchPrefetch(
                    args[0],
                    args[1],
                    self.device,
                    self.cfg["encoding"]["max_entities"],
                    spool=args[2],
                )
            except MemoryError:
                self._batch_iterator = iter(flat_batches(*args))
        else:
            self._batch_iterator = iter(flat_batches(*args))

    def _publish_fit_metrics(self, *, final=False):
        if final:
            loss, selected_sum, selected_n, gap_sum, gap_n, consistency, *totals = (
                self._pass_metrics.cpu().tolist()
            )
            for key, value in consistency_diagnostics(consistency, totals).items():
                self.logger.record("train/" + key, value)
            self._cached_metrics = (
                loss,
                selected_sum / selected_n if selected_n else None,
                gap_sum / gap_n if gap_n else None,
            )
        loss, selected_error, placement_gap = self._cached_metrics
        self.logger.record("train/q_loss", loss)
        self.logger.record("train/selected_action_error", selected_error)
        self.logger.record("train/placement_gap", placement_gap)
        self._last_metrics_at = perf_counter()

    def _fit_chunk(self, callback):
        viewer = getattr(self.env, "live_view", None)
        if viewer is not None and viewer.session.enabled:
            viewer.session.poll()
            viewer.session.publish()
        if self._fit_epoch >= self.n_epochs:
            self._phase(CohortPhase.SYNCHRONIZE, callback)
            return
        if getattr(self, "_batch_iterator", None) is None:
            self._start_pass()
        item = next(self._batch_iterator, None)
        if item is None:
            if self._fit_pass_start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._device_events.append((self._fit_pass_start, end))
                self._fit_pass_start = None
            self._publish_fit_metrics(final=True)
            if not torch.isfinite(self._pass_metrics).all():
                raise FloatingPointError("Non-finite two-stage Q loss")
            try:
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.master_parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                raise FloatingPointError("Non-finite two-stage gradients") from exc
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self._optimizer_committing = True
            self.policy.optimizer.step()
            self.policy.sync_runtime_from_masters()
            self._update_teacher()
            self._optimizer_committing = False
            end.record()
            self._optimizer_events.append((start, end))
            self.policy.zero_grad(set_to_none=True)
            self._stats["q_optimizer_steps"] += 1
            self._fit_epoch += 1
            self._close_batches()
            self.logger.record("train/q_grad_norm", float(norm))
        else:
            rows, observations, *metadata = item
            self.policy.set_training_mode(True)
            loss, selected_error, placement_gap = batch_loss(
                self.policy,
                rows,
                self._buffer.outcome_counts,
                observations,
                metadata[0] if metadata else None,
                teacher=self._teacher,
                consistency_counts=self._buffer.consistency_counts,
            )
            loss.backward()
            self.policy.accumulate_master_gradients()
            self.policy.clear_runtime_grad(set_to_none=True)
            self._pass_metrics += torch.cat(
                (
                    torch.stack(
                        (
                            loss.detach().double(),
                            selected_error.detach().double().sum(),
                            loss.new_tensor(selected_error.numel()).double(),
                            placement_gap.detach().double().sum(),
                            loss.new_tensor(placement_gap.numel()).double(),
                            self.policy._last_species_consistency_loss.double(),
                        )
                    ),
                    self.policy._last_species_consistency_metrics,
                )
            )
            if perf_counter() - self._last_metrics_at >= self.cfg["logging"]["progress_seconds"]:
                self._publish_fit_metrics()
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _synchronize(self, callback):
        self._drain()
        self._stats["trajectory_bytes"] = (
            self._buffer.size * self._buffer.dtype.itemsize + self._buffer.entity_size * 44
        )
        self._stats["trajectory_ram_bytes"] = self._buffer.ram_used
        self._stats["ema_version"] = self._teacher_version
        self._stats["ema_compilation_status"] = self._teacher.compilation_status
        self._stats["optimizer_seconds"] = self._optimizer_seconds
        self._stats["fit_precision"] = self.execution_state["precision"]
        self._stats["fit_batch_groups"] = self.execution_state["fit_batch_groups"]
        self._stats["encoder_microbatch"] = self.execution_state["microbatch"]
        self._stats["encoder_token_budget"] = self.execution_state["token_budget"]
        self._stats["execution_fallbacks"] = list(self.execution_state["fallbacks"])
        self._stats["fitting_spool_rows"] = len(getattr(self, "_fitting_spool", None) or ())
        self._stats["compilation_status"] = getattr(self.policy, "compilation_status", "disabled")
        self._stats["compilation_backend"] = getattr(self.policy, "compilation_backend", None)
        self._stats["collection_graphs"] = {
            name: {
                stage: dict(encoder.collection_metrics)
                for stage, encoder in zip(("species", "tile"), network.encoders, strict=True)
            }
            for name, network in (("online", self.policy), ("ema", self._teacher))
        }
        if getattr(self, "_fitting_spool", None) is not None:
            self._fitting_spool.close()
            self._fitting_spool = None
        super()._synchronize(callback)

    def save(self, *args, **kwargs):
        with atomic_transition():
            self._drain()
            if self.phase == CohortPhase.FIT:
                self._close_batches()
                self.policy.zero_grad(set_to_none=True)
                if getattr(self, "_fitting_spool", None) is not None:
                    self._fitting_spool.close()
                    self._fitting_spool = None
            return super().save(*args, **kwargs)

    def learn(self, *args, **kwargs):
        try:
            return super().learn(*args, **kwargs)
        finally:
            self._drain()
            self._close_batches()
            if getattr(self, "_fitting_spool", None) is not None:
                self._fitting_spool.close()
                self._fitting_spool = None

    def _restore_runtime(self):
        extra = self.runtime_state
        if extra and extra.get("environment") is not None:
            if extra.get("objective") != self.cfg["training"].get("objective"):
                raise ValueError("Checkpoint objective differs; use weights-only initialization")
            if self.phase != CohortPhase.IDLE and any(
                extra.get(key) is None
                for key in (
                    "teacher",
                    "pending_episodes",
                    "compilation",
                    "plant_exploration",
                    "collection_scheduler",
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
            for name, network in (("policy", self.policy), ("teacher", self._teacher)):
                for stage, encoder in zip(("species", "tile"), network.encoders, strict=True):
                    compilation = extra["compilation"][name][stage]
                    if compilation.get("status") not in (
                        "disabled",
                        "requested",
                        "compiled",
                        "fallback",
                        "unavailable",
                    ):
                        raise ValueError("Invalid encoder execution recovery state")
                    if compilation["status"] == "fallback":
                        encoder.disable_compilation(compilation["error"])
                    elif self.cfg["training"]["performance"]["compile_kernels"]:
                        encoder.enable_compilation()
        self._scheduler = CollectionScheduler(self.n_envs, self.device)
        if extra and extra.get("collection_scheduler") is not None:
            self._scheduler.restore(extra["collection_scheduler"])
        self.policy.zero_grad(set_to_none=True)
        self._batch_iterator = None
        if not getattr(self, "execution_state", None):
            self._reset_execution()
        self._device_events = []
        self._optimizer_events = []
        self._last_metrics_at = 0.0
        self._optimizer_seconds = getattr(self, "_optimizer_seconds", 0.0)
        self._fit_pass_start = None
        self.collection_execution = {}
        if self.phase == CohortPhase.COLLECT:
            for network in (self.policy, getattr(self, "_teacher", None)):
                if network is not None:
                    for encoder in network.encoders:
                        encoder.microbatch = self.cfg["policy"]["encoder_microbatch"]
                        encoder.token_budget = self.cfg["policy"]["encoder_token_budget"]
                        encoder.activation_checkpointing = False

    def _extra_runtime(self):
        if getattr(self, "_teacher", None) is None:
            return {}
        return dict(
            collection_scheduler=self._scheduler.snapshot()
            if getattr(self, "_scheduler", None)
            else None,
            plant_exploration=self._plant_exploration.snapshot()
            if getattr(self, "_plant_exploration", None) is not None
            else None,
            teacher={k: v.cpu() for k, v in self._teacher.runtime_state_dict().items()},
            teacher_version=self._teacher_version,
            pending_episodes=getattr(self, "_pending_episodes", {}),
            objective=dict(self.cfg["training"]["objective"]),
            compilation={
                name: {
                    stage: dict(status=encoder.compilation_status, error=encoder.compilation_error)
                    for stage, encoder in zip(("species", "tile"), network.encoders, strict=True)
                }
                for name, network in (("policy", self.policy), ("teacher", self._teacher))
            },
        )

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "_batch_iterator",
            "_fitting_spool",
            "_pass_metrics",
            "_device_events",
            "_optimizer_events",
            "_fit_pass_start",
            "_collection_packed",
            "_scheduler",
            "_collection_callback",
            "_teacher",
            "_plant_exploration",
            "_plant_costs",
            "_pending_episodes",
        ]
