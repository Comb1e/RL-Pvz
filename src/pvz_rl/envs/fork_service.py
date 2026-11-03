"""Budgeted, resumable panels of isolated CUDA static forks."""

from time import perf_counter

import numpy as np

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.cuda_static_fork import StaticForkBatch
from pvz_rl.learning.forecast_labels import FORECAST_PROTOCOL


class ForkService:
    def __init__(self, source, settings, *, sky_sun_rate=1.0):
        import torch

        cp = source.cp
        self.source = source
        # Predictor activation and optimizer schedules do not alter fork physics.
        # Resolve only this service's inputs so a mid-cohort resume cannot compare
        # a live activation flag with stale run-start metadata.
        self.settings = {
            name: settings[name] for name in ("probe", "lanes", "vram_mib", "chunk_ticks")
        }
        self.settings["headroom_mib"] = settings.get("headroom_mib", 0)
        self.batch = None
        self.request = None
        self.cursor = 0
        self.results = None
        self.metrics = dict(forks=0, ticks=0, seconds=0.0, peak_lanes=0)
        # Reclaim inactive framework reservations only at an allocation boundary.
        torch.cuda.empty_cache()
        free, _ = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        per_lane = sum(
            getattr(source, name)[0].nbytes
            for name in ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns")
        )
        per_lane += 5 * source.zombies.shape[-1] * 8 + 65536
        budget = min(
            settings["vram_mib"] * 2**20, max(0, free - settings.get("headroom_mib", 0) * 2**20)
        )
        count = min(settings["lanes"], int(budget // per_lane))
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
                "Static forks cannot allocate one lane. Lower training.forecasts.lanes/"
                "vram_mib or free GPU memory; recovery is retained."
            )
        self.metrics["peak_lanes"] = count

    def begin(self, indices, actions=None):
        if self.request is not None:
            raise RuntimeError("Finish or recover the current fork panel before starting another")
        indices = np.asarray(indices, np.int64)
        actions = (
            np.zeros(len(indices), np.int64) if actions is None else np.asarray(actions, np.int64)
        )
        if indices.shape != actions.shape or indices.ndim != 1:
            raise ValueError("One action is required per fork root")
        self.request = dict(indices=indices, actions=actions, panel_active=False)
        self.results = np.zeros((len(indices), 5, 4), np.int64)
        self.cursor = 0

    def advance(self):
        if self.request is None:
            return True
        start = perf_counter()
        try:
            req, batch = self.request, self.batch
            if self.cursor == len(req["indices"]):
                return True
            last = (
                self.cursor + batch.count
                if req["panel_active"]
                else min(len(req["indices"]), self.cursor + batch.n)
            )
            if not req["panel_active"]:
                batch.start(
                    self.source,
                    req["indices"][self.cursor : last],
                    req["actions"][self.cursor : last],
                )
                req["panel_active"] = True
            before = batch.header[: batch.count, 0].copy()
            from pvz_game.config import ZOMBIE_TYPES

            done = batch.advance(probe_kind=ZOMBIE_TYPES.index(self.settings["probe"]))
            # One bounded compact synchronization. No observations or policy calls.
            ready = bool(done.all())
            self.metrics["ticks"] += int((batch.header[: batch.count, 0] - before).sum())
            if ready:
                self.results[self.cursor : last] = batch.forecast[: batch.count].get()
                self.metrics["forks"] += last - self.cursor
                self.cursor, req["panel_active"] = last, False
            return self.cursor == len(req["indices"])
        finally:
            self.metrics["seconds"] += perf_counter() - start

    def take(self):
        if self.request is None or self.cursor != len(self.request["indices"]):
            raise RuntimeError("A partial fork panel cannot publish labels")
        result = self.results
        self.request = self.results = None
        self.cursor = 0
        return result

    def snapshot(self):
        return dict(
            protocol=FORECAST_PROTOCOL,
            settings=self.settings,
            metrics=self.metrics,
            request=self.request,
            cursor=self.cursor,
            results=self.results,
            batch=self.batch.snapshot()
            if self.request is not None and self.request["panel_active"]
            else None,
        )

    def restore(self, state):
        saved = {name: state["settings"].get(name, 0) for name in self.settings}
        if state["protocol"] != FORECAST_PROTOCOL or saved != self.settings:
            raise ValueError("Incompatible static-fork recovery")
        self.request, self.cursor, self.results = (
            state["request"],
            state["cursor"],
            state["results"],
        )
        if self.request is not None:
            indices, actions = self.request["indices"], self.request["actions"]
            if (
                indices.ndim != 1
                or actions.shape != indices.shape
                or np.any((indices < 0) | (indices >= self.source.n))
                or np.any((actions < 0) | (actions >= ActionSchema.size))
                or not 0 <= self.cursor <= len(indices)
                or self.results.shape != (len(indices), 5, 4)
                or self.results.dtype != np.int64
                or type(self.request["panel_active"]) is not bool
                or self.request["panel_active"] != (state["batch"] is not None)
            ):
                raise ValueError("Invalid pending static-fork panel")
        elif self.cursor != 0 or self.results is not None or state["batch"] is not None:
            raise ValueError("Idle static-fork recovery contains pending storage")
        self.metrics.update(state["metrics"])
        if state["batch"] is not None:
            if self.cursor + state["batch"]["count"] > len(self.request["indices"]):
                raise ValueError("Pending static-fork panel exceeds its root set")
            self.batch.restore(state["batch"])

    def close(self):
        self.request = self.results = self.batch = None

    @property
    def host_bytes(self):
        return (0 if self.results is None else self.results.nbytes) + sum(
            value.nbytes for value in (self.request or {}).values() if isinstance(value, np.ndarray)
        )
