"""Budgeted, resumable work queues of isolated CUDA static forks."""

from time import perf_counter

import numpy as np

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.cuda_static_fork import StaticForkBatch
from pvz_rl.envs.fork_execution import PHYSICAL, ForkExecution
from pvz_rl.learning.forecast_labels import FORECAST_PROTOCOL


class ForkService:
    def __init__(self, source, settings, *, sky_sun_rate=1.0):
        import torch

        cp = source.cp
        self.source = source
        self.settings = {
            name: settings[name] for name in ("probe", "lanes", "vram_mib", "chunk_ticks")
        }
        self.settings["headroom_mib"] = settings.get("headroom_mib", 0)
        self.request = self.results = None
        self.cursor = 0
        self.metrics = dict(
            forks=0,
            ticks=0,
            seconds=0.0,
            peak_lanes=0,
            synchronizations=0,
            cache_hits=0,
            cache_misses=0,
            cache_mismatches=0,
            cache_spawn_misses=0,
            cache_exhausted=0,
            cache_empty=0,
            baseline_probes=0,
            baseline_probe_reuses=0,
            candidate_probes=0,
            field_only_rollouts=0,
            sampled_waits=0,
            skipped_waits=0,
        )
        torch.cuda.empty_cache()
        free, _ = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        budget = min(
            settings["vram_mib"] * 2**20, max(0, free - settings.get("headroom_mib", 0) * 2**20)
        )
        per_lane = sum(getattr(source, name)[0].nbytes for name in PHYSICAL) + 65536
        queue_fixed = 5 * source.n * (20 + 4) * 8 + 96 + source.n * 8
        available = max(0, budget - queue_fixed - source.n * 8)
        count = min(
            settings["lanes"],
            settings.get("fork_max_jobs", 5 * source.n),
            int(available // (per_lane + 8)),
        )
        self.batch = None
        while count:
            try:
                self.batch = StaticForkBatch(
                    count,
                    rules=source.rules,
                    zombie_capacity=source.zcap + 5,
                    projectile_capacity=source.qcap,
                    max_step_ticks=settings["chunk_ticks"],
                    device=source.device,
                    sky_sun_rate=sky_sun_rate,
                )
                break
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()
                count //= 2
        if self.batch is None:
            raise MemoryError(
                "Static forks cannot allocate one scratch lane within the VRAM cap/headroom. "
                f"Reusable VRAM: {free / 2**20:.0f} MiB; reserved headroom: "
                f"{settings.get('headroom_mib', 0)} MiB; fork cap: "
                f"{settings['vram_mib']} MiB. Free GPU memory or increase "
                "training.forecasts.vram_mib if the cap is limiting; recovery is retained."
            )
        occupied = sum(v.nbytes for v in vars(self.batch).values() if isinstance(v, cp.ndarray))
        queue_bytes = queue_fixed + count * 8
        cache_budget = min(
            settings.get("fork_cache_vram_mib", 128) * 2**20,
            max(0, budget - occupied - queue_bytes),
        )
        self.execution = ForkExecution(self.batch, source, settings, cache_budget=cache_budget)
        self.metrics.update(
            peak_lanes=count,
            cache_steps=self.execution.steps,
            cache_bytes=self.execution.cache_bytes,
            device_bytes=occupied + self.execution.device_bytes,
        )
        if self.metrics["device_bytes"] > budget:
            raise MemoryError("Static fork storage exceeds its VRAM budget; increase vram_mib")
        self._reported = np.zeros(11, np.int64)

    def begin(self, indices, actions=None, *, field_only=None):
        if self.request is not None:
            raise RuntimeError("Finish or recover the current fork request before starting another")
        indices = np.asarray(indices, np.int64)
        baseline = actions is None
        actions = np.zeros(len(indices), np.int64) if baseline else np.asarray(actions, np.int64)
        if (
            indices.ndim != 1
            or indices.shape != actions.shape
            or np.any((indices < 0) | (indices >= self.source.n))
            or np.any((actions < 0) | (actions >= ActionSchema.size))
            or (baseline and len(set(indices)) != len(indices))
        ):
            raise ValueError("One valid action is required per fork root; baselines must be unique")
        field_only = np.zeros(len(indices), bool) if field_only is None else np.asarray(field_only)
        if (
            field_only.shape != indices.shape
            or field_only.dtype != bool
            or (field_only.any() and not baseline)
        ):
            raise ValueError("Field-only jobs must be baseline roots")
        # Preparation has not launched a kernel. A failed allocation can retry
        # from the retained root instead of publishing a half-prepared queue.
        self.execution.begin(indices, actions, baseline=baseline, field_only=field_only)
        self.request = dict(
            version=2, indices=indices, actions=actions, baseline=baseline, field_only=field_only
        )
        self.results = None
        self.cursor = 0
        self._reported.fill(0)

    def advance(self):
        if self.request is None or self.results is not None:
            return True
        start = perf_counter()
        try:
            from pvz_game.config import ZOMBIE_TYPES

            progress = self.execution.advance(probe=ZOMBIE_TYPES.index(self.settings["probe"]))
            delta = progress - self._reported
            for key, value in zip(
                (
                    "forks",
                    "ticks",
                    "cache_hits",
                    "cache_misses",
                    "cache_mismatches",
                    "cache_spawn_misses",
                    "cache_exhausted",
                    "cache_empty",
                    "baseline_probes",
                    "candidate_probes",
                    "field_only_rollouts",
                ),
                delta,
                strict=True,
            ):
                self.metrics[key] += int(value)
            self.metrics["synchronizations"] += bool(len(self.request["indices"]))
            self._reported[:] = progress
            self.cursor = int(progress[0])
            if self.cursor == len(self.request["indices"]):
                self.results = self.execution.output[: self.cursor].get()
                return True
            return False
        finally:
            self.metrics["seconds"] += perf_counter() - start

    def take(self):
        if self.request is None or self.results is None:
            raise RuntimeError("A partial fork request cannot publish labels")
        result = self.results
        self.request = self.results = None
        self.cursor = 0
        return result

    def invalidate(self, indices=None, **kwargs):
        self.execution.invalidate(indices, **kwargs)

    def snapshot(self):
        return dict(
            version=3,
            protocol=FORECAST_PROTOCOL,
            settings=self.settings,
            metrics=self.metrics,
            request=self.request,
            cursor=self.cursor,
            results=self.results,
            reported=self._reported.copy(),
            execution=self.execution.snapshot() if self.request is not None else None,
        )

    def restore(self, state):
        if (
            state.get("version") not in (None, 2, 3)
            or state["protocol"] != FORECAST_PROTOCOL
            or state["settings"]["probe"] != self.settings["probe"]
        ):
            raise ValueError("Incompatible static-fork recovery")
        request = state["request"]
        if request is None:
            if state["cursor"] != 0 or state["results"] is not None:
                raise ValueError("Idle static-fork recovery contains pending storage")
        else:
            self.begin(
                request["indices"],
                None if request.get("baseline", False) else request["actions"],
                field_only=request.get("field_only"),
            )
            if state.get("version") in (2, 3):
                progress = state["execution"]["progress"]
                result = state["results"]
                if (
                    state["cursor"] != progress[0]
                    or not np.array_equal(state["reported"], progress)
                    or (
                        result is not None
                        and (
                            progress[0] != len(request["indices"])
                            or result.shape != (len(request["indices"]), 5, 4)
                            or result.dtype != np.int64
                            or not np.array_equal(result, state["execution"]["output"])
                        )
                    )
                ):
                    raise ValueError("Static-fork publication disagrees with pending progress")
                self.execution.restore(state["execution"])
                reported = state["reported"]
                self._reported[:] = np.pad(reported, (0, 11 - len(reported)))
                self.cursor, self.results = state["cursor"], state["results"]
            else:
                self._restore_panel(state)
        for key in (
            "forks",
            "ticks",
            "seconds",
            "synchronizations",
            "cache_hits",
            "cache_misses",
            "cache_mismatches",
            "cache_spawn_misses",
            "cache_exhausted",
            "cache_empty",
            "baseline_probes",
            "baseline_probe_reuses",
            "candidate_probes",
            "field_only_rollouts",
            "sampled_waits",
            "skipped_waits",
        ):
            self.metrics[key] = state["metrics"].get(key, 0)

    def _restore_panel(self, state):
        """Translate the previous panel payload without restarting completed work."""
        cp, execution = self.source.cp, self.execution
        prefix, panel = state["cursor"], state["batch"]
        indices = self.request["indices"]
        if (
            not 0 <= prefix <= len(indices)
            or state["results"].shape != (len(indices), 5, 4)
            or state["results"].dtype != np.int64
            or bool(state["request"]["panel_active"]) != (panel is not None)
        ):
            raise ValueError("Invalid legacy static-fork panel")
        execution.output[:prefix] = cp.asarray(state["results"][:prefix])
        execution.progress[0] = prefix
        execution.next_job[0] = prefix
        if panel is not None:
            if prefix + panel["count"] > len(indices):
                raise ValueError("Legacy fork panel exceeds its root set")
            self.batch.restore(panel)
            execution.workers = max(execution.workers, panel["count"])
            execution.job_ids[: panel["count"]] = cp.arange(prefix, prefix + panel["count"])
            execution.next_job[0] = prefix + panel["count"]
        self.batch.count = execution.workers
        self.cursor = prefix
        self._reported[0] = prefix
        execution.origins.fill(-1)

    def close(self):
        self.request = self.results = self.execution = self.batch = None

    @property
    def host_bytes(self):
        return (0 if self.results is None else self.results.nbytes) + sum(
            value.nbytes for value in (self.request or {}).values() if isinstance(value, np.ndarray)
        )
