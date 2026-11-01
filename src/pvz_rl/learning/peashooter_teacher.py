"""Deferred, resumable CUDA labels for learner-visited Peashooter boards."""

import copy
import json
import math
import re
from collections import deque
from time import perf_counter

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_snapshots import download_snapshots, upload_snapshots
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
    """Bounded private JSON storage; only owned flat files enter recovery."""

    def __init__(self, buffer):
        self.path = buffer.path / "peashooter"
        self.path.mkdir(exist_ok=True)
        self.limit = min(64 * 2**20, buffer.ram_bytes // 16)
        self.cache, self.nbytes, self.spilled = {}, 0, 0
        self.trusted = set()

    def put(self, key, value):
        data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.trusted.discard(key)
        if isinstance(value, EngineSnapshot) and value.unchanged():
            self.trusted.add(key)
        self.nbytes -= len(self.cache.pop(key, b""))
        if self.nbytes + len(data) <= self.limit:
            self.cache[key] = data
            self.nbytes += len(data)
        else:
            (self.path / f"{key}.json").write_bytes(data)
            self.spilled += len(data)

    def get(self, key):
        value = json.loads(self.cache[key] if key in self.cache else (self.path / f"{key}.json").read_bytes())
        if key not in self.trusted:
            # Recovery files are external input. Validate before granting the
            # process-local provenance used by batched CUDA uploads.
            restored_game(value)
            self.trusted.add(key)
        return value

    def contains(self, key):
        return key in self.cache or (self.path / f"{key}.json").is_file()

    def archive(self, archive):
        for key, data in self.cache.items():
            archive.writestr(f"peashooter/{key}.json", data)
        for path in self.path.glob("*.json"):
            if path.stem not in self.cache:
                archive.write(path, f"peashooter/{path.name}")

    def flush(self):
        for key, data in self.cache.items():
            (self.path / f"{key}.json").write_bytes(data)
        self.cache.clear()
        self.nbytes = 0

    def restore(self, archive):
        for name in archive.namelist():
            if name.startswith("peashooter/") and name.endswith(".json"):
                leaf = name.removeprefix("peashooter/")
                if not re.fullmatch(r"(?:root-\d+|lane-\d+-[0-4]|progress-\d+)\.json", leaf):
                    raise ValueError("Invalid Peashooter private archive path")
                (self.path / leaf).write_bytes(archive.read(name))

    def close(self):
        self.cache.clear()
        self.nbytes = 0
        self.trusted.clear()
        if self.path.is_dir():
            for path in self.path.glob("*.json"):
                path.unlink()
            self.path.rmdir()
        try:
            self.path.parent.rmdir()
        except OSError:
            pass  # The shared trajectory owner may still have live blocks.


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
        return self.store.nbytes + len(self.roots) * 1024 + len(self.jobs) * 512

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
        for name in ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns"):
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
                indices=[local[int(rows["env"][i])] for i in selected])
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
        while self.next_root < stop:
            index = self.next_root
            root = self.roots[index]
            snapshot = self.store.get(f"root-{index}")
            root["state"] = "coverage"
            for row in lane_facts(snapshot)[1]:
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
        free, _ = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        headroom = self.cfg["training"]["performance"].get("collection_vram_headroom_mib", 0) * 2**20
        budget = min(self.settings["oracle_vram_mib"] * 2**20, max(0, free - headroom))
        qcap = original.qcap + projectile_bound(original.rules)
        per_lane = (qcap * 6 + original.zcap * 29 + 512) * 8 + 16384
        count = min(self.settings["oracle_lanes"], int(budget // per_lane))
        while count:
            candidate = None
            try:
                candidate = _LaneBatch(count, cutoff=self.cutoff, zombie_capacity=original.zcap,
                    projectile_capacity=qcap, max_step_ticks=self.settings["oracle_chunk_ticks"],
                    rules=original.rules, device=original.device)
                actual = sum(v.nbytes for v in vars(candidate).values() if isinstance(v, cp.ndarray))
                if actual > budget:
                    candidate = None
                    count //= 2
                    continue
                candidate.set_sky_sun_rate(original.sky_sun_rate)
                self.batch = candidate
                self.slots = [None] * count
                self.metrics.update(scratch_lanes=count, scratch_mib=actual / 2**20)
                return
            except cp.cuda.memory.OutOfMemoryError:
                candidate = None
                cp.get_default_memory_pool().free_all_blocks()
                count //= 2
        raise MemoryError("Peashooter oracle cannot allocate one lane; free VRAM or increase oracle_vram_mib")

    def _fill(self):
        snapshots, slots = [], []
        for slot, current in enumerate(self.slots):
            if current is not None or not self.queue:
                continue
            index = self.queue.popleft()
            job = self.jobs[index]
            raw = self.store.get(job.get("partial", job["key"]))
            snapshots.append(EngineSnapshot._seal(raw))
            slots.append(slot)
            job["state"] = "running"
            self.slots[slot] = index
        if snapshots:
            upload_snapshots(self.batch, snapshots, indices=slots)

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
            self._fill()
            cp, batch = self.env.cp, self.batch
            actions = cp.zeros(batch.n, cp.int64)
            for slot, index in enumerate(self.slots):
                if index is not None:
                    job = self.jobs[index]
                    if not job["placed"] and job["tile"] >= 0:
                        branch = self.cfg["environment"]["plants"].index("peashooter")
                        actions[slot] = 1 + branch * A.tiles + job["tile"]
            before = batch.header[:, 0].copy()
            batch.step_device(actions, ticks=self.settings["oracle_chunk_ticks"], per_tick=self.env._per_tick)
            headers = batch.header[:, [0, 1, 12]].get()
            self.metrics["simulated_ticks"] += int((batch.header[:, 0] - before).sum().get())
            for slot, index in enumerate(self.slots):
                if index is None:
                    continue
                job = self.jobs[index]
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
                    batch.header[slot, 17] = 0
                    self.slots[slot] = None
                    self._resolve(job["root"])
            self.metrics["pending"] = len(self.queue) + sum(i is not None for i in self.slots)
            return self.settled
        finally:
            self.metrics["oracle_seconds"] += perf_counter() - started
            self.metrics["spill_bytes"] = self.store.spilled
            self.metrics["pending"] = len(self.queue) + sum(i is not None for i in self.slots)
            self.metrics["queued_boards"] = len(self.roots) - self.next_root
            self.metrics["mean_preferred_set_size"] = self.metrics["preferred_tiles"] / max(1, self.metrics["labeled"])

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
        self.batch = None
        self.slots = []
        self.store.flush()

    def close(self):
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
