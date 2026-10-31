"""Persistent, ordered and bounded CPU preparation for the bomb oracle."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from pvz_rl.envs.scratch_game import EngineSnapshot, scratch_rules
from pvz_rl.envs.tile_scenarios import (
    bomb_candidates,
    bomb_initials,
    generation_failure,
    sample_bomb,
)
from pvz_rl.learning.deadline import check_deadline

_WORKER_CFG = None


def _initialize(cfg):
    import signal

    global _WORKER_CFG
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_CFG = cfg
    scratch_rules()


def _prepare(job, cfg=None):
    cfg = _WORKER_CFG if cfg is None else cfg
    seed, start, labels, deadline = job
    if labels:
        return sample_bomb(cfg, seed, deadline=deadline)
    if cfg["training"].get("performance", {}).get("cherry_cuda_warmup", False):
        candidate = next(bomb_initials(cfg, seed, start=start, deadline=deadline), None)
        if candidate is None:
            raise generation_failure(cfg, seed)
        game, attempt = candidate
        return dict(snapshot=EngineSnapshot.capture(game), generation_failures=attempt)
    candidate = next(bomb_candidates(cfg, seed, start=start, deadline=deadline), None)
    if candidate is None:
        raise generation_failure(cfg, seed)
    game, target, attempt = candidate
    return dict(
        snapshot=EngineSnapshot.capture(game), target_id=target, generation_failures=attempt
    )


class BombPreparation:
    """One pending cohort; workers never import Torch or own CUDA contexts."""

    def __init__(self, cfg):
        import os

        self.cfg = cfg
        perf = cfg["training"].get("performance", {})
        requested = perf.get("cherry_prepare_workers", 0)
        # Leave at least half the configured host budget for trajectories/fitting.
        self.worker_ram_mib = perf.get("cherry_worker_ram_mib", 96)
        ram_workers = int(cfg["training"]["storage"]["ram_gib"] * 1024 / (2 * self.worker_ram_mib))
        self.workers = min(requested, os.cpu_count() or 1, max(0, ram_workers))
        self.oracle = perf.get("cherry_oracle", "cpu")
        self.pool = None
        self.cuda = None

    @property
    def reserved_bytes(self):
        return self.workers * self.worker_ram_mib * 1024**2

    def submit(self, seeds, deadline=None, *, starts=None):
        if self.workers and self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(self.cfg,),
            )
        starts = [0] * len(seeds) if starts is None else starts
        jobs = [
            (seed, start, self.oracle == "cpu", deadline)
            for seed, start in zip(seeds, starts, strict=True)
        ]
        return [self.pool.submit(_prepare, job) for job in jobs] if self.pool else jobs

    def _resolve(self, task, deadline):
        if self.pool is None:
            return _prepare(task, self.cfg)
        while True:
            from pvz_rl.learning.cohort import check_transition_interruption

            check_transition_interruption()
            check_deadline(deadline)
            try:
                return task.result(timeout=0.05)
            except TimeoutError:
                pass

    def collect(self, tasks, seeds, *, deadline=None):
        boards = [None] * len(tasks)
        remaining = list(range(len(tasks)))
        try:
            while remaining:
                candidates = [self._resolve(task, deadline) for task in tasks]
                if self.oracle == "cpu":
                    return candidates
                if self.cuda is None:
                    from pvz_rl.envs.cuda_bomb_oracle import CudaBombOracle

                    self.cuda = CudaBombOracle(self.cfg)
                if self.cfg["training"].get("performance", {}).get("cherry_cuda_warmup", False):
                    candidates = self.cuda.warmup(candidates, deadline=deadline)
                labels = self.cuda.labels(candidates, deadline=deadline)
                retry, starts = [], []
                for index, board, (positive, absolute) in zip(
                    remaining, candidates, labels, strict=True
                ):
                    if positive:
                        boards[index] = dict(
                            board, feasible_tiles=positive, absolute_infeasible_tiles=absolute
                        )
                    else:
                        retry.append(index)
                        starts.append(board["generation_failures"] + 1)
                remaining = retry
                tasks = self.submit([seeds[i] for i in retry], deadline, starts=starts)
            return boards
        except BaseException:
            self.cancel(tasks)
            raise

    @staticmethod
    def cancel(tasks):
        for task in tasks:
            if hasattr(task, "cancel"):
                task.cancel()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        if self.cuda is not None:
            self.cuda.close()
        self.cuda = None
