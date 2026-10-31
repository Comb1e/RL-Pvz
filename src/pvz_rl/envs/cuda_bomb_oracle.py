"""Bounded device-resident bomb alternatives, with explicit source attribution."""

from functools import partial
from weakref import proxy

from pvz_game.cuda import CudaBatch
from pvz_game.cuda.backend import projectile_bound

from pvz_rl.envs.cuda_snapshots import GROUPS, pack_snapshots
from pvz_rl.envs.scratch_game import scratch_rules
from pvz_rl.learning.cohort import check_transition_interruption
from pvz_rl.learning.deadline import check_deadline


class _BombBatch(CudaBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit = self.cp.zeros((self.n, 3), dtype=self.cp.int64)
        replacements = {
            "double *f;": "double *f; I *credit;",
            "I ticks, I per_tick) {": "I ticks, I per_tick, I *credit) {",
            "facts + i * 9};": "facts + i * 9, credit + i * 3};",
            "    emit(5, a.id, source, dh, da);": "    if (a.id == credit[0] && source == credit[1] && dh + da > 0) credit[2] = 1;\n    emit(5, a.id, source, dh, da);",
        }
        for before, after in replacements.items():
            if self.source.count(before) != 1:
                raise RuntimeError("Pinned CUDA bomb attribution hook changed; verify the engine")
            self.source = self.source.replace(before, after)
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")
        self._step = partial(type(self)._attributed_step, proxy(self))

    def _attributed_step(self, grid, block, args):
        self._step_kernel(grid, block, (*args, self.credit))


class CudaBombOracle:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rules = scratch_rules()
        self.batch = None

    def close(self):
        if self.batch is not None:
            cp = self.batch.cp
            self.batch = None
            # Phase boundary: make unused CuPy storage available to normal fitting.
            cp.get_default_memory_pool().free_all_blocks()

    def _allocate(self, boards):
        from pvz_game.cuda.backend import cupy_runtime

        cp = cupy_runtime()
        perf = self.cfg["training"]["performance"]
        qcap = projectile_bound(self.rules) + max(len(b["snapshot"]["projectiles"]) for b in boards)
        zcap = max(1, *(len(b["snapshot"]["level"]["spawns"]) for b in boards))
        if self.batch is not None and self.batch.qcap >= qcap and self.batch.zcap >= zcap:
            return
        self.batch = None
        free, _ = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        budget = min(
            perf.get("cherry_oracle_vram_mib", 256) * 1024**2,
            max(0, free - perf.get("collection_vram_headroom_mib", 0) * 1024**2),
        )
        per_lane = qcap * 6 * 8 + zcap * 32 * 8 + 65536
        count = min(perf.get("cherry_oracle_lanes", 256), int(budget // per_lane))
        while count:
            try:
                self.batch = _BombBatch(
                    count,
                    zombie_capacity=zcap,
                    projectile_capacity=qcap,
                    rules=self.rules,
                    device=cp.cuda.runtime.getDevice(),
                    max_step_ticks=self.rules.plants["cherry_bomb"]["first_ticks"] + 1,
                )
                return
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()
                count //= 2
        raise MemoryError(
            "Cherry oracle cannot allocate one scratch lane; free VRAM or use cherry_oracle = 'cpu'"
        )

    def labels(self, boards, *, deadline=None):
        """CPU-validated starting states; one compact result per simulated tile."""
        if not boards:
            return []
        check_deadline(deadline)
        self._allocate(boards)
        batch = self.batch
        cp = batch.cp
        labels = []
        # Bound staging too; do not materialize a whole cohort × 45 device states.
        width = max(1, batch.n // 45)
        for start in range(0, len(boards), width):
            check_transition_interruption()
            check_deadline(deadline)
            chunk = boards[start : start + width]
            raw, packed = pack_snapshots(batch, [b["snapshot"] for b in chunk])
            jobs, absolute = [], [[] for _ in chunk]
            for i, row in enumerate(raw):
                occupied = {p["row"] * 9 + p["col"] for p in row["plants"]}
                usable = (
                    row["status"] == "running"
                    and row["cooldowns"]["cherry_bomb"] == 0
                    and row["sun"] >= self.rules.plants["cherry_bomb"]["cost"]
                )
                for tile in range(45):
                    if tile in occupied or not usable:
                        absolute[i].append(tile)
                    else:
                        jobs.append((i, tile))
            positive = [[] for _ in chunk]
            for offset in range(0, len(jobs), batch.n):
                check_deadline(deadline)
                panel = jobs[offset : offset + batch.n]
                count = len(panel)
                source = cp.asarray([i for i, _ in panel])
                batch.header[:, 17] = 0
                for name, slab in packed.items():
                    target = getattr(batch, name)
                    if name in GROUPS:
                        target[:count] = 0
                        if slab.shape[1]:
                            target[:count, : slab.shape[1]] = cp.asarray(slab)[source]
                    else:
                        target[:count] = cp.asarray(slab)[source]
                batch.credit[:] = 0
                batch.credit[:count, 0] = cp.asarray([chunk[i]["target_id"] for i, _ in panel])
                batch.credit[:count, 1] = batch.header[:count, 3]  # next_id is this new bomb
                actions = cp.zeros(batch.n, dtype=cp.int64)
                actions[:count] = cp.asarray([1 + 3 * 45 + tile for _, tile in panel])
                batch.step_device(actions, ticks=batch.max_step_ticks, per_tick=False)
                hits = batch.credit[:count, 2].get()
                for (i, tile), hit in zip(panel, hits, strict=True):
                    if hit:
                        positive[i].append(tile)
            labels.extend(zip(positive, absolute, strict=True))
        check_deadline(deadline)
        return labels
