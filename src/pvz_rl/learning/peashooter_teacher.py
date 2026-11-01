"""Deferred, resumable CUDA labels for learner-visited Peashooter boards."""

import copy
import json
import math
import re
import sys
from collections import deque
from time import perf_counter

import numpy as np

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_snapshots import download_snapshots
from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game
from pvz_rl.learning.action_routing import RepeatedRowCompactor
from pvz_rl.learning.peashooter_rules import (
    RULE_OBJECTIVE,
    RULE_PROTOCOL,
    candidate_targets,
    choose_rule,
    lane_facts,
    lane_snapshot,
    rule_settings,
    rules_enabled,
)


class SnapshotStore:
    """Owned snapshots in RAM; JSON only at spill and recovery boundaries."""

    def __init__(self, buffer):
        self.path = buffer.path / "peashooter"
        self.path.mkdir(exist_ok=True)
        self.limit = min(buffer.cfg["training"]["peashooter_rules"].get("snapshot_cache_mib", 256) * 2**20, buffer.ram_bytes // 8)
        self.cache, self.nbytes, self.spilled = {}, 0, 0
        self.trusted = set()
        self.shared = {}
        self.shared_ids = {}
        self.disk = {}
        self.spool = (self.path / "snapshots.bin").open("w+b")

    def _spill(self, key, data):
        self.spool.seek(0, 2)
        self.disk[key] = (self.spool.tell(), len(data))
        self.spool.write(data)
        self.spilled += len(data)

    def _read(self, key):
        offset, size = self.disk[key]
        self.spool.seek(offset)
        data = self.spool.read(size)
        if len(data) != size:
            raise OSError("Truncated owned Peashooter snapshot storage")
        return data

    def _encode(self, value):
        # Spills are process-owned. Refer to immutable metadata once rather
        # than serializing the rules, wave plan and RNG tuple for every root.
        metadata = {field: self.shared_ids[id(value[field])] for field in ("rules", "rng_state", "level")}
        return json.dumps(dict(metadata=metadata,
            snapshot={k: v for k, v in value.items() if k not in metadata}),
            separators=(",", ":"), allow_nan=False).encode()

    def _decode(self, data):
        value = json.loads(data)
        return dict(value["snapshot"], **{field: self.shared[key] for field, key in value["metadata"].items()})

    def put(self, key, value):
        self.trusted.discard(key)
        if isinstance(value, EngineSnapshot) and value.unchanged():
            self.trusted.add(key)
        previous = self.cache.pop(key, None)
        self.disk.pop(key, None)
        self.nbytes -= previous[1] if previous else 0
        # Engine rules and Python RNG metadata are immutable in this owner.
        # Share their values across roots instead of charging/copying thousands
        # of identical rule dictionaries and 625-element RNG states.
        from pvz_game.config import canonical_hash
        value = copy.copy(value)
        for field in ("rules", "rng_state", "level"):
            original = value[field]
            digest = self.shared_ids.get(id(original))
            if digest is None:
                digest = canonical_hash(original)
                # Keep the original alive: IDs cannot alias a later object.
                self.shared_ids[id(original)] = digest
            if digest not in self.shared:
                self.shared[digest] = original
                self.nbytes += snapshot_bytes(original)
            # Retain only canonical object IDs, avoiding unbounded metadata.
            if original is not self.shared[digest]:
                self.shared_ids.pop(id(original), None)
            value[field] = self.shared[digest]
        size = snapshot_bytes({k: v for k, v in value.items() if k not in ("rules", "rng_state", "level")})
        if self.nbytes + size <= self.limit:
            self.cache[key] = (value, size)
            self.nbytes += size
        else:
            self._spill(key, self._encode(value))

    def get(self, key):
        value = self.cache[key][0] if key in self.cache else self._decode(self._read(key))
        if key not in self.trusted:
            # Recovery files are external input. Validate before granting the
            # process-local provenance used by batched CUDA uploads.
            restored_game(value)
            self.trusted.add(key)
        return value

    def contains(self, key):
        return key in self.cache or key in self.disk

    def archive(self, archive):
        for key, (value, _) in self.cache.items():
            archive.writestr(f"peashooter/{key}.json", json.dumps(value, separators=(",", ":"), allow_nan=False))
        for key in self.disk:
            if key not in self.cache:
                archive.writestr(f"peashooter/{key}.json", json.dumps(self._decode(self._read(key)),
                    separators=(",", ":"), allow_nan=False))

    def flush(self):
        for key, (value, _) in self.cache.items():
            self._spill(key, self._encode(value))
        self.nbytes -= sum(size for _, size in self.cache.values())
        self.cache.clear()

    def restore(self, archive):
        for name in archive.namelist():
            if name.startswith("peashooter/") and name.endswith(".json"):
                leaf = name.removeprefix("peashooter/")
                if not re.fullmatch(r"(?:root-\d+|lane-\d+-[0-4]|progress-\d+)\.json", leaf):
                    raise ValueError("Invalid Peashooter private archive path")
                self.put(leaf.removesuffix(".json"), json.loads(archive.read(name)))

    def close(self):
        self.cache.clear()
        self.nbytes = 0
        self.trusted.clear()
        self.shared.clear()
        self.shared_ids.clear()
        self.disk.clear()
        if not self.spool.closed:
            self.spool.close()
        if self.path.is_dir():
            (self.path / "snapshots.bin").unlink(missing_ok=True)
            for path in self.path.glob("*.json"):
                path.unlink()
            self.path.rmdir()
        try:
            self.path.parent.rmdir()
        except OSError:
            pass  # The shared trajectory owner may still have live blocks.


def snapshot_bytes(value):
    """Conservative host charge, including repeated scalar/key references."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(snapshot_bytes(k) + snapshot_bytes(v) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(snapshot_bytes(v) for v in value)
    return size


class _LaneBatch(AccountingCudaBatch):
    def __init__(self, *args, cutoff, **kwargs):
        super().__init__(*args, **kwargs)
        before = "s.step(actions[i], ticks, per_tick);"
        if self.source.count(before) != 1:
            raise RuntimeError("Pinned lane-oracle cutoff hook changed")
        self.source = self.source.replace(before,
            f"s.step(actions[i], lo(ticks, hi(0LL, {cutoff}LL - s.h->tick)), per_tick);")
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")


class PeashooterTeacher:
    def __init__(self, env, buffer, identity, state=None, archive=None):
        self.env, self.buffer, self.cfg = env, buffer, buffer.cfg
        self.settings = copy.deepcopy(rule_settings(self.cfg))
        self.identity = tuple(identity)
        self.store = SnapshotStore(buffer)
        self.compactor = RepeatedRowCompactor(self.cfg["training"]["objective"]["invalid_fit_group_size"])
        self.roots, self.jobs, self.labels = [], [], {}
        self.queue = deque()
        self.batch, self.slots, self.capture = None, [], None
        self.transfers = None
        self.pool = None
        self.preparing = None
        from pvz_rl.envs.lane_preparation import LanePreparation
        workers = self.settings.get("prepare_workers", 0)
        preparation = getattr(env, "lane_preparation", None)
        if preparation is None or preparation.requested != workers:
            if preparation is not None:
                preparation.close()
            env.lane_preparation = LanePreparation(workers, buffer.ram_bytes)
        self.preparation = env.lane_preparation
        self.cutoff = math.ceil(self.cfg["environment"]["cutoff_seconds"] * env.batch.rules.game["tick_rate"])
        self.initialized = False
        self.next_root = 0
        self.metrics = dict(roots=0, labeled=0, rule_1=0, rule_2=0, rule_3=0,
            preferred_tiles=0, fallback_lanes=0, blocked_lanes=0, unlabeled=0,
            completed_rollouts=0, oracle_successes=0, oracle_timeouts=0,
            simulated_ticks=0, oracle_seconds=0.0, pending=0, spill_bytes=0)
        if state is not None:
            if tuple(state["identity"]) != self.identity or state["settings"] != self.settings:
                raise ValueError("Peashooter oracle identity/settings differ from recovery")
            self.roots = copy.deepcopy(state["roots"])
            self.jobs = copy.deepcopy(state["jobs"])
            self.labels = copy.deepcopy(state["labels"])
            self.metrics = dict(state["metrics"])
            self.initialized = state["initialized"]
            self.next_root = state["next_root"]
            self.compactor.__dict__.update(copy.deepcopy(state["compactor"]))
            self.store.restore(archive)
            for i in range(len(self.roots)):
                if not self.store.contains(f"root-{i}"):
                    raise ValueError(f"Missing Peashooter root {i}")
            for i, job in enumerate(self.jobs):
                if job["state"] != "completed":
                    if job.get("partial") and not self.store.contains(job["partial"]):
                        raise ValueError("Missing partial Peashooter rollout")
                    job["state"] = "queued"
                    self.queue.append(i)

    @property
    def estimated_bytes(self):
        return (self.store.nbytes + len(self.store.disk) * 256 + len(self.roots) * 1024 + len(self.jobs) * 512
                + self.preparation.workers * 128 * 2**20)

    @property
    def settled(self):
        return self.initialized and all(r["state"] == "labeled" for r in self.roots)

    def before(self, actions, active, branch_available):
        """Device copies before mutation; defer snapshot transfer until compaction."""
        cp, original = self.env.cp, self.env.batch
        branch = self.cfg["environment"]["plants"].index("peashooter") + 1
        action = cp.from_dlpack(actions.detach())
        selected = cp.flatnonzero(cp.from_dlpack(active) & cp.from_dlpack(branch_available[:, branch].contiguous())
            & (action >= 1 + (branch - 1) * A.tiles) & (action < 1 + branch * A.tiles))
        self.capture_slots = selected.get().tolist()
        if not self.capture_slots:
            self.capture = None
            return
        view = copy.copy(original)
        view.n = len(self.capture_slots)
        view.header = original.header[selected].copy()
        from pvz_game.cuda.schema import HEADER
        # Snapshot readers use the live counts. Copy no unoccupied projectile
        # reserve (the pinned engine reserves >20,000 shots per main lane).
        counts = view.header[:, [HEADER.index(k) for k in ("np", "nz", "nq")]].max(axis=0).get()
        for name, width in zip(("plants", "zombies", "projectiles"), counts, strict=True):
            setattr(view, name, getattr(original, name)[selected, :int(width)].copy())
        for name in ("mowers", "cooldowns"):
            setattr(view, name, getattr(original, name)[selected].copy())
        view._templates = [original._templates[i] for i in self.capture_slots]
        self.capture = view

    def observe(self, receipt):
        rows = self.buffer.take(receipt.rows)
        retained = self.compactor.select(rows)
        local = {slot: index for index, slot in enumerate(self.capture_slots)}
        selected = [i for i in retained if int(rows["env"][i]) in local]
        if selected:
            snapshots = download_snapshots(self.capture,
                indices=[local[int(rows["env"][i])] for i in selected], share_metadata=True)
            for i, snapshot in zip(selected, snapshots, strict=True):
                row = rows[i]
                identity = tuple(int(row[k]) for k in ("env", "episode", "decision"))
                index = len(self.roots)
                self.store.put(f"root-{index}", snapshot)
                self.roots.append(dict(identity=identity, state="queued", coverage={}, successes={}, pending=0))
        self.capture = None
        self.metrics.update(roots=len(self.roots), spill_bytes=self.store.spilled)

    def _enqueue(self, root_index, row, tile=-1):
        key = f"lane-{root_index}-{row}"
        if not self.store.contains(key):
            snap = lane_snapshot(self.store.get(f"root-{root_index}"), row, self.env.batch.rules)
            self.store.put(key, snap)
        self.jobs.append(dict(root=root_index, row=row, tile=tile, key=key, state="queued", placed=False))
        self.queue.append(len(self.jobs) - 1)
        self.roots[root_index]["pending"] += 1

    def _label(self, index, tiles, fallback=()):
        root = self.roots[index]
        label = dict(rule=root["rule"], tiles=tiles, fallback_rows=list(fallback), blocked=root["blocked"])
        self.labels[tuple(root["identity"])] = label
        root["state"] = "labeled"
        self.metrics[f"rule_{root['rule']}"] += 1
        self.metrics["preferred_tiles"] += len(tiles)
        self.metrics["fallback_lanes"] += len(fallback)
        self.metrics["blocked_lanes"] += root["blocked"]
        self.metrics["labeled" if tiles else "unlabeled"] += 1

    def _resolve(self, index):
        root = self.roots[index]
        if root["pending"] or root["state"] == "labeled":
            return
        snapshot = self.store.get(f"root-{index}")
        if root["state"] == "coverage":
            result = choose_rule(snapshot, root["coverage"])
            root.update(rule=result["rule"], rows=result["rows"], blocked=result["blocked"])
            if root["rule"] != 2:
                self._label(index, result["tiles"])
            else:
                root["state"] = "candidate_testing"
                legal = lane_facts(snapshot)[2]
                for row in root["rows"]:
                    for tile in legal[row]:
                        self._enqueue(index, row, tile)
                if not root["pending"]:
                    self._resolve(index)
        else:
            tiles, fallback = candidate_targets(snapshot, root["rows"], root["successes"])
            self._label(index, tiles, fallback)

    def _initialize(self):
        stop = min(len(self.roots), self.next_root + 32)
        if self.preparing is None:
            panel = []
            for index in range(self.next_root, stop):
                snapshot = self.store.get(f"root-{index}")
                panel.append((snapshot, list(lane_facts(snapshot)[1])))
            self.preparing = self.preparation.submit(panel)
        if not self.preparation.ready(self.preparing):
            # Do not spin while a process prepares roots. Preserve the existing
            # bounded deadline/interrupt boundary with a short host yield.
            from time import sleep
            sleep(0.001)
            return
        prepared = self.preparation.resolve(self.preparing)
        self.preparing = None
        start = self.next_root
        while self.next_root < stop:
            index = self.next_root
            root = self.roots[index]
            root["state"] = "coverage"
            for row, lane in prepared[index - start]:
                self.store.put(f"lane-{index}-{row}", lane)
                self._enqueue(index, row)
            self._resolve(index)
            self.next_root += 1
        self.initialized = self.next_root == len(self.roots)

    def _allocate(self):
        if self.batch is not None:
            return
        import torch
        from pvz_game.cuda.backend import projectile_bound

        cp, original = self.env.cp, self.env.batch
        torch.cuda.empty_cache()  # Oracle boundary only, after main collection.
        cp.get_default_memory_pool().free_all_blocks()
        free, _ = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        headroom = self.cfg["training"]["performance"].get("collection_vram_headroom_mib", 0) * 2**20
        budget = min(self.settings["oracle_vram_mib"] * 2**20, max(0, free - headroom))
        cache_bytes = min(64 * 2**20, budget // 8)
        qcap = original.qcap + projectile_bound(original.rules)
        per_lane = (qcap * 6 + original.zcap * 29 + 512) * 8 + 16384
        count = min(self.settings["oracle_lanes"], int((budget - cache_bytes) // per_lane))
        self.pool = cp.cuda.MemoryPool()
        while count:
            candidate = None
            try:
                with cp.cuda.using_allocator(self.pool.malloc):
                    candidate = _LaneBatch(count, cutoff=self.cutoff, zombie_capacity=original.zcap,
                        projectile_capacity=qcap, max_step_ticks=self.settings["oracle_chunk_ticks"],
                        rules=original.rules, device=original.device)
                actual = sum(v.nbytes for v in vars(candidate).values() if isinstance(v, cp.ndarray))
                if actual + cache_bytes > budget:
                    candidate = None
                    count //= 2
                    continue
                candidate.set_sky_sun_rate(original.sky_sun_rate)
                self.batch = candidate
                from pvz_rl.envs.cuda_lane_oracle import LaneTransfers
                self.transfers = LaneTransfers(cp, cache_bytes)
                self.slots = [None] * count
                self.slot_ticks = np.zeros(count, dtype=np.int64)
                self.metrics.update(scratch_lanes=count, scratch_mib=actual / 2**20,
                                    root_cache_limit_mib=cache_bytes / 2**20)
                return
            except cp.cuda.memory.OutOfMemoryError:
                candidate = None
                self.pool.free_all_blocks()
                count //= 2
        raise MemoryError("Peashooter oracle cannot allocate one lane; free VRAM or increase oracle_vram_mib")

    def _fill(self):
        entries = []
        for slot, current in enumerate(self.slots):
            if current is not None or not self.queue:
                continue
            index = self.queue.popleft()
            job = self.jobs[index]
            key = job.get("partial", job["key"])
            cached = self.transfers.cache.get(key)
            if cached is not None:
                # Repeated candidates use the immutable device root. Avoid
                # rereading spilled JSON and hashing an unused host snapshot.
                snapshot = raw = cached[0]
            else:
                raw = self.store.get(key)
                snapshot = raw if isinstance(raw, EngineSnapshot) else EngineSnapshot._seal(raw)
            entries.append((slot, key, snapshot))
            self.slot_ticks[slot] = raw["tick"]
            job["state"] = "running"
            self.slots[slot] = index
        self.transfers.fill(self.batch, entries)

    def advance(self):
        """One bounded engine chunk, including a recoverable commit of results."""
        started = perf_counter()
        try:
            if not self.initialized:
                self._initialize()
                if not self.initialized:
                    return False
            if self.settled:
                return True
            self._allocate()
            with self.env.cp.cuda.using_allocator(self.pool.malloc):
                return self._run_chunk()
        finally:
            self.metrics["oracle_seconds"] += perf_counter() - started
            self.metrics["spill_bytes"] = self.store.spilled
            self.metrics["pending"] = len(self.queue) + sum(i is not None for i in self.slots)
            self.metrics["queued_boards"] = len(self.roots) - self.next_root
            self.metrics["mean_preferred_set_size"] = self.metrics["preferred_tiles"] / max(1, self.metrics["labeled"])

    def _run_chunk(self):
        self._fill()
        cp, batch = self.env.cp, self.batch
        actions = np.zeros(batch.n, np.int64)
        for slot, index in enumerate(self.slots):
            if index is not None:
                job = self.jobs[index]
                if not job["placed"] and job["tile"] >= 0:
                    branch = self.cfg["environment"]["plants"].index("peashooter")
                    actions[slot] = 1 + branch * A.tiles + job["tile"]
        batch.step_device(cp.asarray(actions), ticks=self.settings["oracle_chunk_ticks"], per_tick=self.env._per_tick)
        headers = batch.header[:, [0, 1, 12]].get()
        finished = []
        for slot, index in enumerate(self.slots):
            if index is None:
                continue
            job = self.jobs[index]
            self.metrics["simulated_ticks"] += int(headers[slot, 0] - self.slot_ticks[slot])
            self.slot_ticks[slot] = headers[slot, 0]
            if not job["placed"] and job["tile"] >= 0 and not headers[slot, 2]:
                raise RuntimeError("Peashooter oracle rejected a legal candidate")
            job["placed"] = True
            tick, status = headers[slot, :2]
            if status or tick >= self.cutoff:
                success = bool(status == 1)
                self.metrics["completed_rollouts"] += 1
                self.metrics["oracle_successes"] += success
                self.metrics["oracle_timeouts"] += not status
                root = self.roots[job["root"]]
                root["coverage" if job["tile"] < 0 else "successes"][job["row"] if job["tile"] < 0 else job["tile"]] = success
                root["pending"] -= 1
                job.update(state="completed", success=success, tick=int(tick))
                finished.append(slot)
                self.slots[slot] = None
                self._resolve(job["root"])
        if finished:
            batch.header[cp.asarray(finished), 17] = 0
        self.metrics.update(root_cache_hits=self.transfers.hits,
            root_cache_misses=self.transfers.misses,
            root_upload_bytes=self.transfers.uploaded_bytes,
            root_cache_mib=self.transfers.nbytes / 2**20)
        self.metrics["pending"] = len(self.queue) + sum(i is not None for i in self.slots)
        return self.settled

    def snapshot(self):
        if self.capture is not None:
            raise RuntimeError("Cannot save an uncommitted Peashooter capture")
        if self.batch is not None:
            slots = [i for i, job in enumerate(self.slots) if job is not None]
            snapshots = download_snapshots(self.batch, indices=slots)
            for slot, snapshot in zip(slots, snapshots, strict=True):
                index = self.slots[slot]
                key = f"progress-{index}"
                self.store.put(key, snapshot)
                self.jobs[index]["partial"] = key
        return copy.deepcopy(dict(protocol=RULE_PROTOCOL, objective=RULE_OBJECTIVE, settings=self.settings,
            identity=self.identity, roots=self.roots, jobs=self.jobs, labels=self.labels,
            initialized=self.initialized, next_root=self.next_root,
            compactor=vars(self.compactor), metrics=self.metrics))

    def write_archive(self, archive):
        self.store.archive(archive)

    def release(self):
        if self.transfers is not None:
            self.transfers.close()
            self.transfers = None
        self.batch = None
        self.slots = []
        if self.pool is not None:
            self.pool.free_all_blocks()
            self.metrics["scratch_retained_mib"] = self.pool.used_bytes() / 2**20
            self.pool = None
        # Host roots remain charged through estimated_bytes during fitting.
        # Archive them only when recovery is actually saved; do not manufacture
        # thousands of spill files simply to enter the fitting phase.

    def close(self):
        if self.preparing is not None:
            for future in self.preparing:
                if hasattr(future, "cancel"):
                    future.cancel()
            self.preparing = None
        self.release()
        self.store.close()


def validate_rule_recovery(state, cfg):
    phase = (state.get("tile_scenarios") or {}).get("phase")
    payload = state.get("peashooter_teacher")
    if payload is None:
        if state.get("buffer") is not None and rules_enabled(cfg, phase):
            raise ValueError("Missing Peashooter rule recovery; use --init-from for fresh training")
        return
    if (not rules_enabled(cfg, phase) or payload.get("protocol") != RULE_PROTOCOL
            or payload.get("objective") != RULE_OBJECTIVE or payload.get("settings") != rule_settings(cfg)):
        raise ValueError("Peashooter rule objective differs from recovery; use --init-from")
    if any(key not in payload for key in (
        "identity", "roots", "jobs", "labels", "initialized", "next_root", "compactor", "metrics",
    )):
        raise ValueError("Incomplete Peashooter oracle recovery")
    if (len(payload["identity"]) != 2 or any(type(i) is not int or i < 0 for i in payload["identity"])
            or type(payload["next_root"]) is not int or not 0 <= payload["next_root"] <= len(payload["roots"])):
        raise ValueError("Invalid Peashooter oracle cohort identity/cursor")
    identities = set()
    pending = [0] * len(payload["roots"])
    for root in payload["roots"]:
        identity = tuple(root["identity"])
        if (len(identity) != 3 or any(type(i) is not int or i < 0 for i in identity)
                or identity in identities or root["state"] not in ("queued", "coverage", "candidate_testing", "labeled")):
            raise ValueError("Invalid Peashooter root recovery")
        identities.add(identity)
    for job in payload["jobs"]:
        if (not 0 <= job["root"] < len(payload["roots"]) or job["state"] not in ("queued", "running", "completed")
                or not 0 <= job["row"] < A.rows or not -1 <= job["tile"] < A.tiles
                or (job["tile"] >= 0 and job["tile"] // A.cols != job["row"])):
            raise ValueError("Invalid Peashooter oracle job")
        pending[job["root"]] += job["state"] != "completed"
    for index, root in enumerate(payload["roots"]):
        if root["pending"] != pending[index]:
            raise ValueError("Peashooter pending rollout count differs from saved jobs")
        key = tuple(root["identity"])
        if (root["state"] == "labeled") != (key in payload["labels"]):
            raise ValueError("Peashooter label commit differs from saved root state")
    for key, label in payload["labels"].items():
        tiles = label["tiles"]
        if (tuple(key) not in identities or label["rule"] not in (1, 2, 3)
                or len(tiles) != len(set(tiles)) or any(type(t) is not int or not 0 <= t < A.tiles for t in tiles)):
            raise ValueError("Invalid Peashooter preferred set")
