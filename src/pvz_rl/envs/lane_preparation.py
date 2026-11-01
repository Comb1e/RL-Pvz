"""Publicly unreachable, owned lane projection; no model or CUDA imports."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from pvz_rl.envs.scratch_game import EngineSnapshot


def project_lane(snapshot, row):
    """Project a validated owned root; retain every physical field in the lane.

    Immutable rules, RNG-state metadata and cooldown values are shared for reading
    only. Entity dictionaries and mowers are independent mutable copies.
    External recovery inputs must pass the engine validator before this function.
    """
    result = dict(snapshot)
    for name in ("plants", "zombies", "projectiles"):
        result[name] = [dict(e) for e in snapshot[name] if e["row"] == row]
    result["cooldowns"] = dict(snapshot["cooldowns"])
    zombies = result["zombies"]
    result["level"] = dict(
        name="peashooter-visible-lane", initial_sun=result["sun"], plants=[], mowers=False,
        spawns=[dict(tick=result["tick"], zombie_type=z["kind"], row=row,
                     x=max(0, z["x"]), wave=1) for z in zombies],
    )
    result.update(spawn_index=len(zombies), defeated=sum(z["headless"] for z in zombies),
        wave=int(bool(zombies)), status="running" if any(not z["headless"] for z in zombies) else "won")
    result["mowers"] = [dict(m, state="spent") for m in snapshot["mowers"]]
    return EngineSnapshot._seal(result)


def _prepare(job):
    snapshot, rows = job
    return [(row, project_lane(snapshot, row)) for row in rows]


class LanePreparation:
    """One bounded ordered panel; workers persist across cohorts, never own CUDA."""

    def __init__(self, workers, ram_bytes):
        self.requested = workers
        self.workers = min(workers, os.cpu_count() or 1, ram_bytes // (128 * 2**20))
        self.pool = None

    def submit(self, jobs):
        if self.workers and self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"))
        return [self.pool.submit(_prepare, job) for job in jobs] if self.pool else [_prepare(job) for job in jobs]

    def ready(self, tasks):
        return not self.pool or all(task.done() for task in tasks)

    def resolve(self, tasks):
        return [task.result() for task in tasks] if self.pool else tasks

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
