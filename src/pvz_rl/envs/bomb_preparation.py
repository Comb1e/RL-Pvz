"""Persistent, ordered and bounded CPU preparation for the bomb oracle."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from pvz_rl.envs.scratch_game import EngineSnapshot, scratch_rules
from pvz_rl.envs.tile_scenarios import (
    ATTACKER_EXERCISES,
    bomb_candidates,
    bomb_initials,
    generation_failure,
    preparation_settings,
    sample_bomb,
    valid_bomb_labels,
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
    seed, start, labels, deadline, skill, case = job
    if skill == "attacker_pre":
        from pvz_rl.envs.attacker_scenarios import attacker_initials, sample_attacker

        if not preparation_settings(cfg, skill).get("cuda_warmup", False):
            return sample_attacker(cfg, seed, start=start, deadline=deadline, case=case)
        candidate = next(
            attacker_initials(cfg, seed, start=start, deadline=deadline, case=case), None
        )
        if candidate is None:
            raise generation_failure(cfg, seed, skill)
        game, attempt, metadata = candidate
        return dict(snapshot=EngineSnapshot.capture(game), generation_failures=attempt, **metadata)
    if labels:
        return sample_bomb(cfg, seed, deadline=deadline, skill=skill)
    if cfg["training"].get("performance", {}).get("cherry_cuda_warmup", False):
        candidate = next(
            bomb_initials(cfg, seed, start=start, deadline=deadline, skill=skill), None
        )
        if candidate is None:
            raise generation_failure(cfg, seed, skill)
        game, attempt, metadata = candidate
        return dict(snapshot=EngineSnapshot.capture(game), generation_failures=attempt, **metadata)
    candidate = next(bomb_candidates(cfg, seed, start=start, deadline=deadline, skill=skill), None)
    if candidate is None:
        raise generation_failure(cfg, seed, skill)
    game, target, attempt, metadata = candidate
    return dict(
        snapshot=EngineSnapshot.capture(game),
        target_id=target,
        generation_failures=attempt,
        **metadata,
    )


class BombPreparation:
    """One pending cohort; workers never import Torch or own CUDA contexts."""

    def __init__(self, cfg, skill="cherry_bomb"):
        import os

        self.cfg = cfg
        self.skill = skill
        perf = preparation_settings(cfg, skill)
        requested = perf.get("prepare_workers", 0)
        # Leave at least half the configured host budget for trajectories/fitting.
        self.worker_ram_mib = perf.get("worker_ram_mib", 96)
        ram_workers = int(cfg["training"]["storage"]["ram_gib"] * 1024 / (2 * self.worker_ram_mib))
        self.workers = min(requested, os.cpu_count() or 1, max(0, ram_workers))
        self.oracle = perf.get("oracle", "cpu")
        self.pool = None
        self.cuda = None

    @property
    def reserved_bytes(self):
        return self.workers * self.worker_ram_mib * 1024**2

    def submit(
        self, seeds, deadline=None, *, starts=None, skill="cherry_bomb", cases=None, exercises=None
    ):
        if self.workers and self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(self.cfg,),
            )
        starts = [0] * len(seeds) if starts is None else starts
        cases = [None] * len(seeds) if cases is None else cases
        exercises = [skill] * len(seeds) if exercises is None else exercises
        oracles = {
            source: preparation_settings(self.cfg, source).get("oracle", "cpu")
            for source in set(exercises)
        }
        jobs = [
            (seed, start, oracles[source] == "cpu", deadline, source, case)
            for seed, start, case, source in zip(seeds, starts, cases, exercises, strict=True)
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

    def collect(self, tasks, seeds, *, deadline=None, skill="cherry_bomb", cases=None):
        boards = [None] * len(tasks)
        remaining = list(range(len(tasks)))
        cases = [None] * len(seeds) if cases is None else cases
        oracle = preparation_settings(self.cfg, skill).get("oracle", "cpu")
        try:
            while remaining:
                candidates = [self._resolve(task, deadline) for task in tasks]
                if oracle == "cpu":
                    return candidates
                if skill == "attacker_pre":
                    from pvz_rl.envs.attacker_scenarios import (
                        attacker_settings,
                        label_attacker,
                        warmup_ticks,
                    )

                    if preparation_settings(self.cfg, skill).get("cuda_warmup", False):
                        if self.cuda is None:
                            from pvz_rl.envs.cuda_bomb_oracle import CudaBombOracle

                            self.cuda = CudaBombOracle(self.cfg)
                        candidates = self.cuda.warmup(
                            candidates,
                            deadline=deadline,
                            ticks=warmup_ticks(self.cuda.rules, attacker_settings(self.cfg)),
                        )
                    retry, starts = [], []
                    for index, candidate in zip(remaining, candidates, strict=True):
                        board = label_attacker(candidate, scratch_rules())
                        if board is not None:
                            boards[index] = board
                        else:
                            retry.append(index)
                            starts.append(candidate["generation_failures"] + 1)
                    remaining = retry
                    tasks = self.submit(
                        [seeds[i] for i in retry],
                        deadline,
                        starts=starts,
                        skill=skill,
                        cases=[cases[i] for i in retry],
                    )
                    continue
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
                    if valid_bomb_labels(board, positive, skill):
                        boards[index] = dict(
                            board, feasible_tiles=positive, absolute_infeasible_tiles=absolute
                        )
                    else:
                        retry.append(index)
                        starts.append(board["generation_failures"] + 1)
                remaining = retry
                tasks = self.submit([seeds[i] for i in retry], deadline, starts=starts, skill=skill)
            return boards
        except BaseException:
            self.cancel(tasks)
            raise

    def collect_mixed(self, tasks, seeds, exercises, *, deadline=None):
        """Use one pool, rotating the CUDA scratch between bomb and attacker settings."""
        self.release_scratch()
        boards = [None] * len(seeds)
        try:
            for skill in ATTACKER_EXERCISES:
                indices = [i for i, source in enumerate(exercises) if source == skill]
                if not indices:
                    continue
                if skill == "attacker_pre":
                    self.release_scratch()
                prepared = self.collect(
                    [tasks[i] for i in indices],
                    [seeds[i] for i in indices],
                    deadline=deadline,
                    skill=skill,
                )
                for index, board in zip(indices, prepared, strict=True):
                    boards[index] = dict(board, exercise=skill)
            return boards
        except BaseException:
            self.cancel(tasks)
            raise

    @staticmethod
    def cancel(tasks):
        for task in tasks:
            if hasattr(task, "cancel"):
                task.cancel()

    def release_scratch(self):
        """Hand GPU storage to validation without discarding workers or CPU prefetch."""
        if self.cuda is not None:
            self.cuda.close()
            self.cuda = None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.release_scratch()
