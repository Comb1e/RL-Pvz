"""Owned work queue and exact, disposable field-suffix cache for static forks."""

from importlib.resources import files

import numpy as np
from pvz_game.cuda import schema

PHYSICAL = ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns")


def execution_source(source, *, packed):
    hooks = {
        "double sky_sun_rate;": (
            "double sky_sun_rate; I *suffix_depth, *suffix_ticks; I suffix_steps;"
        ),
        "    I offset = forecast_phase[0] == 0 ? 0 : 3;": """
    if (forecast_phase[0] == 0) for (I e = 0; e < suffix_steps; e++)
      if (suffix_ticks[e] >= 0 && h->tick >= suffix_ticks[e])
        suffix_depth[e * 5 + a.row] = hi(suffix_depth[e * 5 + a.row], code);
    I offset = forecast_phase[0] == 0 ? 0 : 3;""",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Static fork execution contract changed: {before}")
        source = source.replace(before, after)
    return (
        source
        + f"\n#define FORK_PACKED {int(packed)}\n"
        + files("pvz_rl.envs").joinpath("fork_execution.cu").read_text("utf-8")
    )


class ForkExecution:
    def __init__(self, batch, source, settings, *, cache_budget):
        self.batch, self.source, self.cp = batch, source, batch.cp
        cp = self.cp
        self.packed = settings.get("fork_packed", False)
        self.threads = settings.get("fork_threads", 32) if self.packed else 32
        self.wait_ticks = settings.get("fork_wait_ticks", 10)
        self.module = cp.RawModule(
            code=execution_source(batch.source, packed=self.packed),
            options=("--std=c++11", "--fmad=false"),
        )
        self.kernel = self.module.get_function("run_fork_jobs")
        self.invalidate_kernel = self.module.get_function("invalidate_fork_cache")
        self.job_ids = cp.full(batch.n, -1, cp.int64)
        self.next_job = cp.zeros(1, cp.int64)
        self.progress = cp.zeros(13, cp.int64)
        self.capacity = settings.get("fork_candidates_per_root", 5) * source.n
        self.roots = cp.empty(self.capacity, cp.int64)
        self.actions = cp.empty_like(self.roots)
        self.field_only = cp.zeros_like(self.roots)
        self.anchors = cp.full_like(self.roots, -1)
        self.cache_anchors = cp.full(source.n, -1, cp.int64)
        self.output = cp.zeros((self.capacity, 5, 4), cp.int64)
        self.cache_zcap = min(source.zcap, settings.get("fork_cache_zombies", 128))
        self.cache_qcap = min(source.qcap, settings.get("fork_cache_projectiles", 256))
        self.stride = (
            len(schema.HEADER)
            + 45 * len(schema.PLANT)
            + self.cache_zcap * len(schema.ZOMBIE)
            + self.cache_qcap * len(schema.PROJECTILE)
            + 5 * len(schema.MOWER)
            + 8
        )
        per_entry = source.n * (self.stride + 1 + 5 + 20) * 8
        self.steps = min(
            settings.get("fork_cache_steps", 8),
            max(0, int(cache_budget) - source.n * 8) // per_entry,
        )
        if not settings.get("fork_suffix_cache", True):
            self.steps = 0
        # All cache memory, including labels and metadata, shares the fork VRAM cap.
        while True:
            try:
                self.states = cp.empty((source.n, self.steps, self.stride), cp.int64)
                self.cache_ticks = cp.full((source.n, self.steps), -1, cp.int64)
                self.depths = cp.zeros((source.n, self.steps, 5), cp.int64)
                self.values = cp.empty((source.n, self.steps, 5, 4), cp.int64)
                self.origins = cp.full(source.n, -1, cp.int64)
                break
            except cp.cuda.memory.OutOfMemoryError:
                if self.steps == 0:
                    raise
                for name in ("states", "cache_ticks", "depths", "values", "origins"):
                    setattr(self, name, None)
                cp.get_default_memory_pool().free_all_blocks()
                self.steps //= 2
        self.count, self.baseline, self.workers = 0, False, 0

    @property
    def cache_bytes(self):
        return sum(
            getattr(self, name).nbytes
            for name in ("states", "cache_ticks", "depths", "values", "origins")
        )

    @property
    def device_bytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, self.cp.ndarray))

    def invalidate(self, indices=None, *, actions=None, active=None):
        if not self.steps:
            return
        if actions is None:
            if indices is None:
                self.cache_ticks.fill(-1)
            else:
                self.cache_ticks[self.cp.asarray(indices, self.cp.int64)] = -1
            return
        self.invalidate_kernel(
            ((self.source.n + 127) // 128,),
            (128,),
            (self.source.header, actions, active, self.cache_ticks, self.steps, self.source.n),
        )

    def begin(self, indices, actions, *, baseline=False, field_only=None):
        self.count, self.baseline = len(indices), baseline
        if self.count > self.capacity:
            raise ValueError("Fork request exceeds its configured candidate capacity")
        self.workers = min(self.count, self.batch.n)
        self.batch.count = self.workers
        self.roots[: self.count].set(np.ascontiguousarray(indices, dtype=np.int64))
        self.actions[: self.count].set(np.ascontiguousarray(actions, dtype=np.int64))
        self.field_only[: self.count] = 0 if field_only is None else self.cp.asarray(field_only)
        self.anchors[: self.count] = -1
        self.progress.fill(0)
        self.next_job.fill(0)
        self.job_ids.fill(-1)
        self.output[: self.count].fill(0)
        self.batch.header[:, schema.HEADER.index("enabled")] = 0

    def advance(self, *, probe, download=True):
        if not self.count:
            return np.zeros(13, np.int64) if download else self.progress
        b, source, cp = self.batch, self.source, self.cp
        grid = (
            ((self.workers + self.threads - 1) // self.threads,) if self.packed else (self.workers,)
        )
        self.kernel(
            grid,
            (self.threads,),
            (
                *(getattr(b, name) for name in PHYSICAL),
                b._schedules,
                b._events,
                b._event_counts,
                b.facts,
                b.forecast,
                b.forecast_phase,
                b.forecast_start,
                self.job_ids,
                self.workers,
                *(getattr(source, name) for name in PHYSICAL),
                self.roots,
                self.actions,
                self.field_only,
                self.anchors,
                self.cache_anchors,
                self.count,
                self.next_job,
                self.progress,
                self.output,
                b.max_step_ticks,
                probe,
                cp.float64(b.sky_sun_rate),
                cp.bool_(self.baseline),
                self.states,
                self.cache_ticks,
                self.depths,
                self.values,
                self.origins,
                self.steps,
                self.stride,
                source.zcap,
                source.qcap,
                self.wait_ticks,
                self.cache_zcap,
                self.cache_qcap,
            ),
        )
        # One progress synchronization: completion, ticks, hits and misses together.
        return self.progress.get() if download else self.progress

    def snapshot(self):
        return dict(
            version=2,
            workers=self.workers,
            baseline=self.baseline,
            batch=self.batch.snapshot(),
            job_ids=self.job_ids[: self.workers].get(),
            next_job=int(self.next_job.get()[0]),
            progress=self.progress.get(),
            output=self.output[: self.count].get(),
            anchors=self.anchors[: self.count].get(),
        )

    def restore(self, state):
        if state.get("version") not in (1, 2):
            raise ValueError("Unsupported static-fork execution recovery version")
        if state["workers"] > self.batch.n:
            raise MemoryError("Pending forks need more scratch lanes; free GPU memory and resume")
        n = state["workers"]
        ids, progress, output = state["job_ids"], state["progress"], state["output"]
        if state.get("version") == 1:
            progress = np.pad(progress, (0, 3))
        if progress.shape == (11,):
            progress = np.pad(progress, (0, 2))
        live = ids[ids >= 0]
        if (
            not 0 <= n <= self.count
            or (self.count > 0 and n == 0)
            or ids.shape != (n,)
            or ids.dtype != np.int64
            or np.any((ids < -1) | (ids >= self.count))
            or progress.shape != (13,)
            or progress.dtype != np.int64
            or np.any(progress < 0)
            or not 0 <= progress[0] <= self.count
            or output.shape != (self.count, 5, 4)
            or output.dtype != np.int64
            or bool(state["baseline"]) != self.baseline
            or not 0 <= state["next_job"] <= self.count + self.batch.n
            or state["batch"]["count"] != n
            or len(set(live)) != len(live)
            or np.any(live >= min(self.count, state["next_job"]))
            or progress[0] + len(live) != min(self.count, state["next_job"])
        ):
            raise ValueError("Invalid pending fork queue")
        self.batch.restore(state["batch"])
        self.workers = n
        self.job_ids[:n] = self.cp.asarray(ids)
        self.next_job[0] = state["next_job"]
        self.progress[:] = self.cp.asarray(progress)
        self.output[: self.count] = self.cp.asarray(output)
        anchors = state.get("anchors", np.full(self.count, -1, np.int64))
        if anchors.shape != (self.count,) or anchors.dtype != np.int64:
            raise ValueError("Invalid pending probe anchors")
        self.anchors[: self.count] = self.cp.asarray(anchors)
        # Completed caches are optional. Never publish partially reconstructed suffixes.
        self.origins.fill(-1)
        self.invalidate()
