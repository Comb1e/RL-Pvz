"""Exact private pretraining with larger simulation waves and original fit panels."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter, sleep

import numpy as np
import torch
from pvz_game.cuda import CudaBatch, schema
from pvz_game.cuda.backend import projectile_bound

from pvz_rl.envs.cuda_features import CudaFeatures
from pvz_rl.envs.cuda_snapshots import download_snapshots, upload_snapshots
from pvz_rl.envs.encoding import width_bucket
from pvz_rl.envs.fork_service import ForkService
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_objective import PreparedAbsoluteEffects
from pvz_rl.learning.forecast_pretraining import LegacyForecastPretraining
from pvz_rl.learning.pretraining_sampling import play_commands, prepare_roots, sample_actions

PROTOCOL = "device_pipeline_v1"
FIT_ROOTS = 32  # Existing optimizer sequence, not an execution tuning parameter.


def device_storage(cp, *owners):
    """Count each allocation once, including feature views into shared storage."""
    allocations = {}
    for owner in owners:
        for value in vars(owner).values():
            if isinstance(value, cp.ndarray):
                allocations[value.data.mem.ptr] = value.data.mem.size
    return sum(allocations.values())


class DeviceForecastPretraining(LegacyForecastPretraining):
    """preparing → playing → labeling → ready → fitting → committed.

    Physical production never reads learned weights. Simulations may run ahead;
    optimizer commits retain the original panels and four-pass ordering.
    """

    def __init__(self, model, state=None):
        started = perf_counter()
        if state is not None and (state.get("version") != 1 or state.get("protocol") != PROTOCOL):
            raise ValueError("Incompatible private pretraining pipeline recovery")
        self.model, self.cfg = model, model.cfg
        self.settings = dict(self.cfg["training"]["forecasts"])
        if state is not None:
            self.settings.update(state["settings"])
        self.total = self.settings["pretrain_roots"]
        self.size = min(FIT_ROOTS * self.settings["pretrain_queue_panels"], max(1, self.total))
        performance = self.cfg["training"]["performance"]
        combined_budget = performance.get("auxiliary_vram_mib", 4096) * 2**20
        projectiles = 2 * projectile_bound(model.env.batch.rules)
        per_root = (
            8
            * (
                len(schema.HEADER)
                + 45 * len(schema.PLANT)
                + 8 * len(schema.ZOMBIE)
                + projectiles * len(schema.PROJECTILE)
                + 5 * len(schema.MOWER)
                + 8
            )
            + 65536
        )  # Conservative allowance for features, schedules and metadata.
        root_budget = max(combined_budget // 2, combined_budget - self.settings["vram_mib"] * 2**20)
        admitted = max(1, root_budget // per_root)
        if self.total >= FIT_ROOTS and admitted < FIT_ROOTS:
            raise MemoryError(
                "Private pretraining cannot retain one original 32-root optimizer panel"
            )
        if admitted >= FIT_ROOTS:
            admitted -= admitted % FIT_ROOTS
        self.size = min(self.size, admitted)
        if state is not None:
            # A pending wave is semantic state; never truncate it to fit new limits.
            self.size = state["size"]
        self.cursor = self.origin = self.fit_offset = self.fit_pass = self.play_tick = (
            self.count
        ) = 0
        self.phase = "generate"
        self.source = CudaBatch(
            self.size,
            zombie_capacity=8,
            projectile_capacity=projectiles,
            max_step_ticks=self.settings["chunk_ticks"],
            rules=model.env.batch.rules,
            device=model.env.batch.device,
        )
        self.cp = cp = self.source.cp
        self.features = CudaFeatures(self.source, self.cfg, "masked")
        root_bytes = device_storage(cp, self.source, self.features)
        # Reserve both outcome carriers plus conversion scratch and pinned status.
        result_reservation = self.size * 6 * 5 * 4 * 8 * 3 + 13 * 8
        remaining = combined_budget - root_bytes - result_reservation
        if remaining < per_root:
            raise MemoryError(
                "Private pretraining roots leave no worker within the auxiliary VRAM cap"
            )
        self.stream = cp.cuda.Stream(non_blocking=True)
        execution = dict(
            self.settings,
            **{
                k: v
                for k, v in self.cfg["training"]["performance"].items()
                if k.startswith("fork_")
            },
        )
        execution.update(
            lanes=self.settings["pretrain_simulation_lanes"],
            vram_mib=min(self.settings["vram_mib"], remaining // 2**20),
            fork_max_jobs=self.settings["pretrain_simulation_lanes"],
            chunk_ticks=self.settings["pretrain_simulation_ticks"],
            fork_suffix_cache=False,
            fork_candidates_per_root=6,
            headroom_mib=self.cfg["training"]["performance"].get("collection_vram_headroom_mib", 0),
        )
        with self.stream:
            self.service = ForkService(self.source, execution)
        self.progress_memory = cp.cuda.alloc_pinned_memory(13 * 8)
        self.progress_host = np.frombuffer(self.progress_memory, np.int64, count=13)
        self.event = cp.cuda.Event(disable_timing=True)
        self.pending = False
        self.graph = None
        self.labels = self.baseline = self.rows = self.prepared = self.summary = None
        self.jobs = []
        self.metrics = dict(
            protocol=PROTOCOL,
            roots=0,
            optimizer_commits=0,
            effect_loss=None,
            random_roots=0,
            scripted_roots=0,
            forks=0,
            simulation_ticks=0,
            play_ticks=0,
            fork_ticks=0,
            seconds=0.0,
            setup_seconds=0.0,
            progress_transfers=0,
            progress_bytes=0,
            label_downloads=0,
            simulation_lanes=self.service.batch.n,
            queue_roots=self.size,
            phase_seconds={},
            fitting_graphs=0,
            root_device_bytes=root_bytes,
            fork_device_bytes=self.service.metrics["device_bytes"],
            combined_device_bound_bytes=root_bytes
            + self.service.metrics["device_bytes"]
            + result_reservation,
            combined_device_cap_bytes=combined_budget,
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-pretrain-prepare")
        self.future = None
        self.future_origin = None
        self._encoder_settings = [
            (e, e.compile_fitting, e.compile_collection) for e in model.policy.encoders
        ]
        # Tiny one-shot pretraining must pay its own graph/compile startup cost.
        for encoder, _, _ in self._encoder_settings:
            encoder.compile_fitting = encoder.compile_collection = False
        try:
            if state is not None:
                for key in (
                    "cursor",
                    "origin",
                    "fit_offset",
                    "fit_pass",
                    "play_tick",
                    "count",
                    "phase",
                    "rows",
                    "jobs",
                    "metrics",
                    "summary",
                ):
                    setattr(self, key, state[key])
                if state["source"]:
                    upload_snapshots(self.source, state["source"], indices=range(self.count))
                ready = cp.cuda.Event(disable_timing=True)
                ready.record()
                self.stream.wait_event(ready)
                with self.stream:
                    self.service.restore(state["service"])
                    self.stream.synchronize()
                self.labels = cp.asarray(state["labels"]) if state["labels"] is not None else None
                self.baseline = (
                    cp.asarray(state["baseline"]) if state["baseline"] is not None else None
                )
                if self.phase == "fit":
                    self.features.encode(publish=False)
            self._prefetch(self.cursor if self.phase == "generate" else self.origin + self.count)
            self.metrics["setup_seconds"] += perf_counter() - started
        except BaseException:
            self.close()
            raise

    def _prepare(self, origin):
        return prepare_roots(
            self.cfg,
            self.source.rules,
            self.model.seed,
            origin,
            min(self.size, self.total - origin),
        )

    def _prefetch(self, origin):
        if origin < self.total:
            self.future_origin = origin
            self.future = self.pool.submit(self._prepare, origin)

    def _generate(self):
        if self.future_origin != self.cursor:
            raise RuntimeError("Private preparation frontier disagrees with committed roots")
        snapshots = self.future.result()
        self.future = None
        self.origin = self.cursor
        self.count = len(snapshots)
        self.source.header[:, 17] = 0
        upload_snapshots(self.source, snapshots, indices=range(self.count))
        self.play_tick, self.fit_offset, self.phase = 0, 0, "play"
        self._prefetch(self.origin + self.count)

    def _play(self):
        # Keep the precise sampling procedure and action cadence, including RNG keys.
        legal = self.source.action_masks_device().get()
        commands = play_commands(legal, self.model.seed, self.origin, self.play_tick, self.count)
        ticks = min(self.source.max_step_ticks, 3501 - self.play_tick)
        self.source.step_device(self.cp.asarray(commands), ticks=ticks, per_tick=False)
        self.metrics["play_ticks"] += ticks * self.count
        self.play_tick += ticks
        if self.play_tick == 3501:
            self._begin_labels()

    def _begin_labels(self):
        legal = self.source.action_masks_device().get()
        actions, valid, self.jobs = sample_actions(legal, self.model.seed, self.origin, self.count)
        self.rows = dict(actions=actions, valid=valid)
        ready = self.cp.cuda.Event(disable_timing=True)
        ready.record()
        self.stream.wait_event(ready)
        with self.stream:
            self.service.begin(
                list(range(self.count)) + [r for r, _ in self.jobs],
                [0] * self.count + [int(self.rows["actions"][r, t]) for r, t in self.jobs],
            )
        self.phase = "effects"

    def _effects(self):
        if self.pending:
            if not self.event.done:
                sleep(0.0001)  # Yield while the only dependency is the outstanding GPU chunk.
                return
            self.pending = False
            self.service._reported[:] = self.progress_host
            self.service.cursor = int(self.progress_host[0])
            if self.service.cursor == len(self.service.request["indices"]):
                self._finish_labels()
                return
        from pvz_game.config import ZOMBIE_TYPES

        with self.stream:
            progress = self.service.execution.advance(
                probe=ZOMBIE_TYPES.index(self.settings["probe"]), download=False
            )
            progress.get(out=self.progress_host, stream=self.stream, blocking=False)
            self.event.record(self.stream)
        self.pending = True
        self.metrics["progress_transfers"] += 1
        self.metrics["progress_bytes"] += self.progress_host.nbytes

    def _finish_labels(self):
        cp = self.cp
        output = self.service.execution.output[: self.service.cursor]
        # The completion event covers simulation and its progress copy.
        self.baseline = output[: self.count].copy()
        tau = reference_tau(self.source.rules)
        codes = output.copy()
        time = codes[..., 1]
        codes[..., 1] = cp.where(
            time < 0, 5, sum((time >= k * tau).astype(cp.int64) for k in (1, 2, 4, 8))
        )
        self.labels = cp.full((self.count, 6, 5, 4), -1, cp.int64)
        self.labels[:, 0] = codes[: self.count]
        if self.jobs:
            roots, slots = np.array(self.jobs).T
            self.labels[cp.asarray(roots), cp.asarray(slots + 1)] = codes[self.count :]
        self.metrics["forks"] += self.service.cursor
        self.metrics["fork_ticks"] += int(self.progress_host[1])
        self.service.request = self.service.results = None
        self.service.cursor = 0
        self.features.encode(publish=False)
        self.summary = self.features.summary[: self.count].get()
        self.fit_pass, self.prepared, self.phase = 0, None, "fit"

    def _fit(self):
        count = min(FIT_ROOTS, self.count - self.fit_offset)
        at = self.fit_offset
        width = width_bucket(
            int(self.summary[at : at + count, 0].max(initial=0)), self.features.limit
        )
        observations = self.features.view(at, count, width)
        if self.prepared is None:
            targets = torch.from_dlpack(self.labels[at : at + count])
            self.prepared = PreparedAbsoluteEffects(
                self.rows["actions"][at : at + count],
                self.rows["valid"][at : at + count],
                targets[:, 0],
                targets[:, 1:],
                padded=self.settings.get("pretrain_graphs", False),
            )
        policy, model = self.model.policy, self.model
        policy.auxiliary_scale, policy.injection_enabled = 1.0, False
        if self.settings.get("pretrain_graphs", False):
            from pvz_rl.learning.pretraining_graph import PretrainingGraph

            shape = (count, width, len(self.prepared.rows))
            if self.graph is not None and self.graph_shape != shape:
                self.graph.close()
                self.graph = None
            if self.graph is None:
                self.graph = PretrainingGraph(
                    policy, observations, self.prepared, model.max_grad_norm
                )
                self.graph_shape = shape
                self.metrics["fitting_graphs"] += 1
            loss, finite = self.graph.replay(observations, self.prepared)
        else:
            policy.zero_grad(set_to_none=True)
            with policy.board_context(observations) as memory:
                loss = self.prepared.loss(policy, observations, memory)
            loss.backward()
            eligible = {id(p) for p in policy.runtime_parameters("effect_pretrain")}
            for parameter in policy.parameters():
                if id(parameter) not in eligible:
                    parameter.grad = None
            policy.accumulate_master_gradients()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.master_parameters("effect_pretrain"), model.max_grad_norm
            )
            finite = torch.isfinite(loss.detach()) & torch.isfinite(norm)
        # One mandatory host check before commit, covering loss and gradients together.
        if not bool(finite):
            raise FloatingPointError("Non-finite private pretraining loss or gradients")
        policy.optimizer.step()
        policy.sync_runtime_from_masters()
        model._update_teacher("effect_pretrain")
        if self.graph is None:
            policy.zero_grad(set_to_none=True)
        policy._forecast_components = policy._primary_graph = None
        model.optimizer_commits += 1
        self.metrics["optimizer_commits"] += 1
        self.fit_pass += 1
        if self.fit_pass == self.settings["pretrain_passes"]:
            self.metrics["effect_loss"] = float(loss.detach())
            self.cursor += count
            self.fit_offset += count
            self.fit_pass, self.prepared = 0, None
            self.metrics.update(
                roots=self.cursor,
                random_roots=(self.cursor + 1) // 2,
                scripted_roots=self.cursor // 2,
            )
            if self.fit_offset == self.count:
                self.phase = "generate"

    def advance(self):
        if self.cursor == self.total:
            return True
        phase, started = self.phase, perf_counter()
        try:
            if phase == "generate":
                self._generate()
            elif phase == "play":
                self._play()
            elif phase == "effects":
                self._effects()
            elif phase == "fit":
                self._fit()
            else:
                raise ValueError("Invalid private pretraining pipeline phase")
            return self.cursor == self.total
        finally:
            elapsed = perf_counter() - started
            phases = self.metrics["phase_seconds"]
            phases[phase] = phases.get(phase, 0.0) + elapsed
            self.metrics["seconds"] += elapsed
            self.metrics["simulation_ticks"] = (
                self.metrics["play_ticks"] + self.metrics["fork_ticks"]
            )

    def snapshot(self):
        self.stream.synchronize()
        if self.pending:
            self.service._reported[:] = self.progress_host
            self.service.cursor = int(self.progress_host[0])
            self.pending = False
        return dict(
            protocol=PROTOCOL,
            version=1,
            settings=self.settings,
            size=self.size,
            **{
                key: getattr(self, key)
                for key in (
                    "cursor",
                    "origin",
                    "fit_offset",
                    "fit_pass",
                    "play_tick",
                    "count",
                    "phase",
                    "rows",
                    "jobs",
                    "metrics",
                    "summary",
                )
            },
            source=download_snapshots(self.source, indices=range(self.count)) if self.count else [],
            service=self.service.snapshot(),
            labels=self.labels.get() if self.labels is not None else None,
            baseline=self.baseline.get() if self.baseline is not None else None,
        )

    def close(self):
        if getattr(self, "pool", None) is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        if getattr(self, "stream", None) is not None:
            self.stream.synchronize()
        for encoder, fitting, collection in getattr(self, "_encoder_settings", []):
            encoder.compile_fitting, encoder.compile_collection = fitting, collection
        if getattr(self, "service", None) is not None:
            self.service.close()
        if getattr(self, "graph", None) is not None:
            self.graph.close()
            self.graph = None
        self.source = self.features = self.prepared = self.labels = self.baseline = None
