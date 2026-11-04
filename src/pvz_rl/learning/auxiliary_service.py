"""Bounded hybrid simulation queues; main-game execution always has priority."""

import copy
import multiprocessing
import os
import pickle
import threading
import traceback
import weakref
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter

import numpy as np

from pvz_rl.envs.auxiliary_cpu import private_game, validate_continuation
from pvz_rl.envs.auxiliary_worker import run as worker
from pvz_rl.envs.lane_preparation import project_lane
from pvz_rl.envs.snapshot_packet import SnapshotPacket, SnapshotRoot
from pvz_rl.envs.teacher_rules import candidate_targets, choose_rule, lane_facts
from pvz_rl.learning.auxiliary_settings import DEFAULTS

PROTOCOL = "hybrid-exact-simulation-v1"


def root_bytes(value):
    if isinstance(value, SnapshotRoot):
        return value.estimated_bytes
    return 8192 + sum(
        len(value[k]) * size
        for k, size in (("plants", 2048), ("zombies", 4096), ("projectiles", 2048))
    )


class RootStore:
    """Only the coordinator serializes or spills; producers retain owned values."""

    def __init__(self, ram_limit, spill_limit):
        self.directory = TemporaryDirectory(prefix="pvz-auxiliary-")
        self.spool = (Path(self.directory.name) / "roots.bin").open("w+b")
        self.limit, self.spill_limit = ram_limit, spill_limit
        self.entries, self.identities = {}, {}
        self.next_id = self.nbytes = self.spilled = 0
        self.pending = OrderedDict()

    def retain(self, value):
        identity = id(value)
        key = self.identities.get(identity)
        if key is not None:
            self.entries[key]["refs"] += 1
            return key
        key = self.next_id
        self.next_id += 1
        size = root_bytes(value)
        self.entries[key] = dict(value=value, size=size, refs=1, identity=identity)
        self.identities[identity] = key
        self.nbytes += size
        if isinstance(value, SnapshotRoot):
            self.pending[key] = None
        return key

    def get(self, key, *, wait=False, packed=False):
        entry = self.entries[key]
        if "value" not in entry:
            self.spool.seek(entry["offset"])
            data = self.spool.read(entry["length"])
            if len(data) != entry["length"]:
                raise OSError("Truncated auxiliary root spool")
            return pickle.loads(data)
        value = entry["value"]
        if packed and isinstance(value, SnapshotRoot) and value.packet.device is not None:
            return value
        return value.resolve(wait=wait) if isinstance(value, SnapshotRoot) else value

    def materialize(self):
        for _ in range(min(32, len(self.pending))):
            key, _ = self.pending.popitem(last=False)
            entry = self.entries.get(key)
            if entry is None or not isinstance(entry.get("value"), SnapshotRoot):
                continue
            value = entry["value"].resolve()
            if value is None:
                self.pending[key] = None
                continue
            if self.identities.get(entry["identity"]) == key:
                self.identities.pop(entry["identity"], None)
            entry["value"] = value
            entry["identity"] = -1
            size = root_bytes(value)
            self.nbytes += size - entry["size"]
            entry["size"] = size

    def spill(self):
        if self.nbytes <= self.limit:
            return
        for entry in list(self.entries.values()):
            if "value" not in entry:
                continue
            value = entry["value"]
            value = value.resolve() if isinstance(value, SnapshotRoot) else value
            if value is None:
                continue
            data = pickle.dumps(value, protocol=5)
            if self.spilled + len(data) > self.spill_limit:
                raise MemoryError("Auxiliary snapshot spill limit exhausted; exact work retained")
            self.spool.seek(0, 2)
            offset = self.spool.tell()
            self.spool.write(data)
            self.spool.flush()
            entry.update(offset=offset, length=len(data))
            entry.pop("value")
            self.identities.pop(entry["identity"], None)
            self.nbytes -= entry["size"]
            self.spilled += len(data)
            if self.nbytes <= self.limit:
                break

    def release(self, key):
        entry = self.entries[key]
        entry["refs"] -= 1
        if entry["refs"]:
            return
        if "value" in entry:
            self.nbytes -= entry["size"]
        if self.identities.get(entry["identity"]) == key:
            self.identities.pop(entry["identity"], None)
        del self.entries[key]
        self.pending.pop(key, None)

    def close(self):
        self.entries.clear()
        self.identities.clear()
        self.pending.clear()
        self.spool.close()
        self.directory.cleanup()


class AuxiliarySimulationService:
    """One coordinator owns CUDA, CPU workers, dependency expansion and spill I/O."""

    def __init__(self, env, buffer, state=None):
        import torch

        self.env, self.buffer = env, buffer
        self.settings = dict(
            DEFAULTS,
            **{k: v for k, v in env.cfg["training"]["performance"].items() if k in DEFAULTS},
        )
        # Saved services without a family directory used both queues. Missing
        # settings on old runs must never silently change their teacher workflow.
        self.families = set(
            state.get("families", ("outcomes", "teacher"))
            if state is not None
            else (("outcomes", "teacher") if self.settings["auxiliary_hybrid"] else ("outcomes",))
        )
        if self.families not in ({"outcomes"}, {"outcomes", "teacher"}):
            raise ValueError("Invalid saved auxiliary execution families")
        self.packed_only = self.families == {"outcomes"}
        self.forecasts = dict(env.cfg["training"]["forecasts"])
        self.teacher_settings = dict(env.cfg["training"].get("peashooter_rules", {}))
        self.teacher_settings.update(
            cutoff=int(
                np.ceil(
                    env.cfg["environment"]["cutoff_seconds"] * env.batch.rules.game["tick_rate"]
                )
            ),
            per_tick=env._per_tick,
        )
        self.condition = threading.Condition(threading.RLock())
        self.jobs, self.boards, self.results = {}, {}, {}
        self.board_queue = deque()
        self.published = set()
        self.queues = {family: deque() for family in ("outcomes", "teacher")}
        self.turn = 0
        self.sealed = set()
        self.failure = None
        self.closing = self.pause_requested = self.paused = self.main_busy = False
        self.drain_started = None
        self.draining = False
        self.defer_until_seal = False
        self.main_event = None
        self.gpu_active = None
        self.gpu = {}
        self.workers = []
        self.estimates = {}
        self.retired_transfers = dict(uploaded_bytes=0, progress_bytes=0)
        self.metrics = dict(
            cpu_jobs=0,
            cuda_jobs=0,
            cpu_ticks=0,
            cuda_ticks=0,
            field_ticks=0,
            probe_ticks=0,
            teacher_jobs=0,
            teacher_successes=0,
            teacher_timeouts=0,
            teacher_cpu_seconds=0.0,
            teacher_cuda_seconds=0.0,
            teacher_ticks=0,
            submitted=0,
            completed=0,
            published=0,
            peak_jobs=0,
            queue_wait_seconds=0.0,
            cpu_seconds=0.0,
            cuda_seconds=0.0,
            decode_seconds=0.0,
            host_wait_seconds=0.0,
            packet_bytes=0,
            root_ram_bytes=0,
            root_spill_bytes=0,
            device_bytes=0,
            worker_ram_bytes=0,
            admission_denials=0,
            migrations=0,
            drain_seconds=0.0,
            peak_root_ram_bytes=0,
            peak_packet_bytes=0,
            peak_device_bytes=0,
            peak_queue_age_seconds=0.0,
            uploaded_bytes=0,
            progress_bytes=0,
        )
        requested = 0 if self.packed_only else self.settings["auxiliary_cpu_workers"]
        reservation = self.settings["auxiliary_worker_ram_mib"] * 2**20
        count = min(
            requested, max(0, (os.cpu_count() or 1) - 4), buffer.ram_bytes // (2 * reservation)
        )
        self.metrics["worker_ram_bytes"] = count * reservation
        self.store = RootStore(
            min(256 * 2**20, max(2**20, buffer.ram_bytes // 8)),
            self.settings["auxiliary_spill_gib"] * 2**30,
        )
        # Switching at a quiescent cohort boundary preserves existing dependencies.
        previous = env.stream
        self.main_stream = torch.cuda.Stream(device=env.batch.device, priority=-1)
        self.main_stream.wait_stream(previous)
        env.stream = env.handoff.stream = self.main_stream
        cp = env.cp
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.transfer_stream = cp.cuda.Stream(non_blocking=True)
        try:
            self._prewarm_cuda()
            context = multiprocessing.get_context("spawn")
            self.workers = getattr(env, "_auxiliary_cpu_workers", [])
            if len(self.workers) != count:
                close_cpu_workers(env)
                self.workers = []
            for _ in range(count - len(self.workers)):
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child,), daemon=True)
                process.start()
                child.close()
                if not parent.poll(45):
                    process.terminate()
                    raise RuntimeError("CPU simulation worker failed to initialize")
                if parent.recv()[0] != "ready":
                    raise RuntimeError("Invalid CPU simulation worker handshake")
                self.workers.append(dict(process=process, connection=parent, job=None, start=0.0))
            env._auxiliary_cpu_workers = self.workers
            if not self.gpu and not self.workers:
                raise MemoryError("No auxiliary simulation worker fits the resource budget")
            if state is not None:
                self._restore(state)
            self.pause_requested = True
            self.thread = threading.Thread(target=self._run, name="pvz-auxiliary", daemon=True)
            self.thread.start()
            with self.condition:
                self.condition.wait_for(lambda: self.paused or self.failure)
            self._check()
        except BaseException:
            self.closing = True
            if hasattr(self, "thread"):
                with self.condition:
                    self.pause_requested = False
                    self.condition.notify_all()
                self.thread.join(timeout=5)
            for slot in self.workers:
                if slot["process"].is_alive():
                    slot["process"].terminate()
                    slot["process"].join(timeout=5)
                slot["connection"].close()
            env._auxiliary_cpu_workers = []
            with self.stream:
                for backend in self.gpu.values():
                    backend.close()
            self.store.close()
            raise

    def _prewarm_cuda(self):
        from pvz_rl.envs.auxiliary_cuda import CudaWorkers, device_bytes
        from pvz_rl.envs.snapshot_packet import PACK

        cp, b = self.env.cp, self.env.batch
        b._snapshot_pack = cp.RawKernel(PACK, "pack_physical")
        b._snapshot_pack.compile()
        self.env.queue.release_scratch()
        free, _ = cp.cuda.runtime.memGetInfo()
        headroom = (
            self.env.cfg["training"]["performance"].get("collection_vram_headroom_mib", 0) * 2**20
        )
        capture = (
            2
            * b.n
            * sum(
                getattr(b, k)[0].nbytes
                for k in ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns")
            )
            + b.n * 32
        )
        budget = min(self.settings["auxiliary_vram_mib"] * 2**20, max(0, free - headroom))
        if budget < capture + 2**20:
            raise MemoryError("Exact boundary captures exceed the combined auxiliary VRAM cap")
        packet_reserve = min(128 * 2**20, (budget - capture) // 8)
        remaining = budget - capture - packet_reserve
        self.metrics["capture_bytes"] = capture
        per_lane = device_bytes(b) // b.n + 131072
        for family in ("outcomes", "teacher"):
            if family not in self.families:
                continue
            if family == "teacher" and not self.teacher_settings.get("oracle_lanes"):
                continue
            child_cap = (
                self.forecasts["vram_mib"] * 2**20 - capture - packet_reserve
                if family == "outcomes"
                else self.teacher_settings["oracle_vram_mib"] * 2**20
            )
            maximum = (
                min(
                    b.n,
                    self.forecasts["lanes"],
                    self.settings["auxiliary_interval_lanes"] if self.packed_only else b.n,
                )
                if family == "outcomes"
                else self.teacher_settings["oracle_lanes"]
            )
            available = min(
                remaining // (2 if family == "outcomes" and "teacher" in self.families else 1),
                child_cap,
            )
            count = min(maximum, max(0, int((available - 16 * 2**20) // per_lane)))
            while count:
                try:
                    backend = CudaWorkers(
                        b, family, count, self.forecasts, self.teacher_settings, self.stream
                    )
                    if backend.nbytes + backend.transfers.limit > available:
                        backend.close()
                        count //= 2
                        continue
                    self.gpu[family] = backend
                    remaining -= backend.nbytes + backend.transfers.limit
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    count //= 2
        self.stream.synchronize()
        self.metrics["device_bytes"] = budget - remaining - packet_reserve
        self.packet_limit = min(128 * 2**20, remaining + packet_reserve)
        self.packets = weakref.WeakSet() if self.packed_only else []
        self.live_packet_bytes = 0
        if not self.gpu and not self.settings["auxiliary_cpu_workers"]:
            raise MemoryError("No exact auxiliary simulation worker fits the resource budget")

    def capture(self, batch, indices, headers):
        if not indices:
            return []
        if not self.packed_only:
            self.packets = [p for p in self.packets if p.device is not None]
        # Packet capacity is charged separately from worker and capture slabs.
        from pvz_game.cuda.schema import HEADER, MOWER, PLANT, PROJECTILE, ZOMBIE

        estimated = (
            len(indices) * (len(HEADER) + 5 * len(MOWER) + 8)
            + sum(
                int(headers[:, HEADER.index(key)].sum()) * len(schema)
                for key, schema in (("np", PLANT), ("nz", ZOMBIE), ("nq", PROJECTILE))
            )
        ) * 8
        live = self.live_packet_bytes if self.packed_only else sum(p.nbytes for p in self.packets)
        if live + estimated > self.packet_limit:
            # This is storage backpressure, never a dependency on completed labels.
            if estimated > self.packet_limit:
                raise MemoryError(
                    "One exact capture exceeds auxiliary staging; root retained in the main state"
                )
            started = perf_counter()
            with self.condition:
                while (
                    self.live_packet_bytes
                    if self.packed_only
                    else sum(p.nbytes for p in self.packets if p.device is not None)
                ) + estimated > self.packet_limit:
                    self._check()
                    self.condition.wait(timeout=0.001)
            self.metrics["host_wait_seconds"] += perf_counter() - started
        packet = SnapshotPacket(
            batch, indices, headers, stream=self.transfer_stream, device_only=self.packed_only
        )
        if self.packed_only:
            with self.condition:
                self.packets.add(packet)
                self.live_packet_bytes += packet.nbytes
            # CuPy views retain their base ndarray. Charge the allocation until
            # both the packet and any in-flight private-copy views release it.
            weakref.finalize(packet.device, self._release_packet, packet.nbytes)
        else:
            self.packets.append(packet)
        self.metrics["packet_bytes"] += packet.nbytes
        live = (
            self.live_packet_bytes
            if self.packed_only
            else sum(p.nbytes for p in self.packets if p.device is not None)
        )
        self.metrics["peak_packet_bytes"] = max(self.metrics["peak_packet_bytes"], live)
        self.metrics["peak_device_bytes"] = max(
            self.metrics["peak_device_bytes"], self.metrics["device_bytes"] + live
        )
        return packet.roots()

    def _release_packet(self, size):
        with self.condition:
            self.live_packet_bytes -= size
            self.condition.notify_all()

    def start(self):
        with self.condition:
            self.pause_requested = False
            self.condition.notify_all()

    def release_completed_intervals(self):
        """Drain boundary: return idle interval scratch before the teacher allocates."""
        if not self.packed_only or "outcomes" not in self.gpu:
            return
        with self.quiescent():
            if self.jobs or self.results:
                raise RuntimeError("Cannot release interval scratch with unpublished exact work")
            with self.stream:
                backend = self.gpu.pop("outcomes")
                self.metrics["device_bytes"] -= backend.nbytes + backend.transfers.limit
                self.retired_transfers["uploaded_bytes"] += backend.transfers.uploaded_bytes
                self.retired_transfers["progress_bytes"] += backend.progress_bytes
                backend.close()

    def import_result(self, identity, result):
        job = self.jobs[identity]
        self.store.release(job["root"])
        del self.jobs[identity]
        self.results[identity] = result

    @property
    def estimated_bytes(self):
        with self.condition:
            return (
                self.metrics["worker_ram_bytes"]
                + self.store.nbytes
                + len(self.jobs) * 2048
                + len(self.boards) * 2048
                + len(self.results) * 1024
                + len(self.published) * 128
            )

    def submit(
        self, identity, snapshot, *, family="outcomes", action=0, continuation=None, **metadata
    ):
        identity = tuple(identity)
        with self.condition:
            self._check()
            if identity in self.jobs or identity in self.results or identity in self.published:
                return identity
            if family in self.sealed:
                raise RuntimeError("Cannot append work to a sealed auxiliary family")
            root = self.store.retain(snapshot)
            job = dict(
                id=identity,
                root=root,
                family=family,
                action=int(action),
                state="capturing",
                queued=monotonic(),
                probe=self.forecasts["probe"],
                sky_sun_rate=self.env.batch.sky_sun_rate,
                cutoff=self.teacher_settings["cutoff"],
                **metadata,
            )
            if continuation is not None:
                job["continuation"] = continuation
                job["placed"] = True
            self.jobs[identity] = job
            self.queues[family].append(identity)
            self.metrics["submitted"] += 1
            self.metrics["peak_jobs"] = max(self.metrics["peak_jobs"], len(self.jobs))
            self.condition.notify_all()
        return identity

    def submit_board(self, identity, snapshot, *, record=None):
        identity = tuple(identity)
        with self.condition:
            self._check()
            if identity in self.boards or identity in self.results or identity in self.published:
                return identity
            self.boards[identity] = dict(
                id=identity,
                root=self.store.retain(snapshot),
                state="capturing",
                coverage={},
                successes={},
                pending=0,
            )
            self.boards[identity].update(record or {})
            if self.boards[identity]["state"] == "capturing":
                self.board_queue.append(identity)
            self.condition.notify_all()
        return identity

    def _check(self):
        if self.failure:
            raise RuntimeError(
                "Exact auxiliary simulation failed; recovery retained:\n" + self.failure
            )

    def poll(self, owner):
        with self.condition:
            self._check()
            return [
                (key, copy.deepcopy(value))
                for key, value in list(self.results.items())
                if key[0] == owner
            ]

    def acknowledge(self, identity):
        with self.condition:
            self.results.pop(tuple(identity), None)
            self.boards.pop(tuple(identity), None)
            self.published.add(tuple(identity))
            self.metrics["published"] = len(self.published)

    def seal(self, owner):
        with self.condition:
            self.sealed.add(owner)
            if not self.draining:
                self.drain_started = perf_counter()
            self.draining = True
            self.condition.notify_all()

    def main_start(self):
        with self.condition:
            self._check()
            self.main_busy = True

    def main_end(self):
        with self.condition:
            event = self.env.cp.cuda.Event(disable_timing=True)
            with self.env.device_context():
                event.record()
            self.main_event = event
            self.main_busy = False
            self.condition.notify_all()

    def _ready(self, job):
        if job["state"] not in ("capturing", "queued"):
            return False
        root = self.store.get(job["root"], packed=self.packed_only)
        if root is None:
            return False
        job["state"] = "queued"
        return True

    def _next(self, family):
        queue = self.queues[family]
        for _ in range(len(queue)):
            identity = queue.popleft()
            job = self.jobs.get(identity)
            if job is None or job["state"] not in ("capturing", "queued"):
                continue
            queue.append(identity)
            if self._ready(job):
                return job
        return None

    def _payload(self, job):
        return dict(job, snapshot=self.store.get(job["root"], packed=self.packed_only))

    def _dispatch_cpu(self, gpu_available):
        for slot in self.workers:
            if slot["job"] is not None or slot.get("pausing"):
                continue
            for offset in range(2):
                family = ("outcomes", "teacher")[(self.turn + offset) % 2]
                job = self._next(family)
                if job is None or job.get("prefer_cuda"):
                    continue
                rates = self.estimates.get(family, {})
                # One CPU pilot per family establishes a measured rate. Later
                # assignments account for current CUDA backlog and transfer cost.
                cpu_live = sum(
                    w["job"] is not None and self.jobs[w["job"]]["family"] == family
                    for w in self.workers
                )
                if gpu_available and family in self.gpu and "cpu" in rates and "cuda" in rates:
                    queued = len(self.queues[family])
                    gpu_wait = rates["cuda"] * (1 + queued / max(1, len(self.gpu[family].slots)))
                    if rates["cpu"] > gpu_wait:
                        continue
                if "cpu" not in rates and cpu_live:
                    continue
                job["state"] = "running_cpu"
                slot.update(job=job["id"], start=perf_counter())
                slot["connection"].send(("run", self._payload(job)))
                self.metrics["cpu_jobs"] += 1
                self.metrics["queue_wait_seconds"] += monotonic() - job["queued"]
                self.metrics["peak_queue_age_seconds"] = max(
                    self.metrics["peak_queue_age_seconds"], monotonic() - job["queued"]
                )
                self.turn = (self.turn + 1) % 2
                break

    def _harvest_cpu(self):
        for slot in self.workers:
            connection = slot["connection"]
            if not slot["process"].is_alive():
                if slot["job"] is not None:
                    self.jobs[slot["job"]]["state"] = "queued"
                    slot["job"] = None
                raise RuntimeError("CPU simulation process exited; its committed root is retained")
            if not connection.poll():
                continue
            kind, identity, result = connection.recv()
            if kind == "paused":
                slot["pausing"] = False
                if identity is not None:
                    job = self.jobs[identity]
                    self._account_progress(job, result["ticks"], "cpu")
                    self.metrics["cpu_seconds"] += perf_counter() - slot["start"]
                    job.update(continuation=result, state="queued", placed=True, prefer_cuda=True)
                    self.queues[job["family"]].append(identity)
                    self.metrics["migrations"] += 1
                slot["job"] = None
                continue
            if kind == "failed":
                if identity in self.jobs and result.get("continuation"):
                    self.jobs[identity].update(continuation=result["continuation"], state="queued")
                    self._account_progress(
                        self.jobs[identity], result["continuation"]["ticks"], "cpu"
                    )
                slot["job"] = None
                raise RuntimeError(result["error"])
            if kind != "completed" or identity != slot["job"]:
                raise RuntimeError("CPU simulation completion identity changed")
            elapsed = perf_counter() - slot["start"]
            self._complete(identity, result, "cpu", elapsed)
            slot["job"] = None

    def _account_progress(self, job, ticks, backend):
        previous = job.setdefault("accounted_ticks", {})
        total = 0
        for key in ("field_ticks", "probe_ticks", "teacher_ticks"):
            current = ticks.get(key, 0)
            delta = current - previous.get(key, 0)
            if delta < 0:
                raise ValueError("A migrated continuation lost simulated ticks")
            self.metrics[key] += delta
            previous[key] = current
            total += delta
        self.metrics[backend + "_ticks"] += total

    def _complete(self, identity, result, backend, seconds):
        job = self.jobs[identity]
        family = job["family"]
        ticks = result["ticks"]
        self._account_progress(job, ticks, backend)
        if backend == "cpu":
            self.metrics["cpu_seconds"] += seconds
        rates = self.estimates.setdefault(family, {})
        rates[backend] = seconds if backend not in rates else rates[backend] * 0.8 + seconds * 0.2
        self.metrics["completed"] += 1
        if family == "teacher":
            self.metrics["teacher_jobs"] += 1
            self.metrics["teacher_successes"] += result["success"]
            self.metrics["teacher_timeouts"] += result["timeout"]
            if backend == "cpu":
                self.metrics["teacher_cpu_seconds"] += seconds
            board = self.boards[job["board"]]
            key = job["row"] if job["tile"] < 0 else job["tile"]
            board["coverage" if job["tile"] < 0 else "successes"][key] = result["success"]
            board["pending"] -= 1
            self._resolve_board(board)
        else:
            self.results[identity] = result
        self.store.release(job["root"])
        del self.jobs[identity]

    def _board_job(self, board, row, tile=-1, *, snapshot=None, continuation=None):
        if snapshot is None:
            snapshot = project_lane(self.store.get(board["root"]), row)
        action = (
            0
            if tile < 0
            else 1 + self.env.cfg["environment"]["plants"].index("peashooter") * 45 + tile
        )
        identity = ("teacher-job", board["id"], row, tile)
        # Teacher sealing closes the root frontier, not dependency expansion.
        sealed = "teacher" in self.sealed
        self.sealed.discard("teacher")
        try:
            self.submit(
                identity,
                snapshot,
                family="teacher",
                action=action,
                board=board["id"],
                row=row,
                tile=tile,
                continuation=continuation,
            )
        finally:
            if sealed:
                self.sealed.add("teacher")
        board["pending"] += 1

    def _resolve_board(self, board):
        if board["pending"] or board["state"] == "labeled":
            return
        snapshot = self.store.get(board["root"])
        if board["state"] == "coverage":
            choice = choose_rule(snapshot, board["coverage"])
            board.update(rule=choice["rule"], rows=choice["rows"], blocked=choice["blocked"])
            if choice["rule"] == 2:
                board["state"] = "candidate_testing"
                legal = lane_facts(snapshot)[2]
                for row in board["rows"]:
                    lane = project_lane(snapshot, row)
                    for tile in legal[row]:
                        self._board_job(board, row, tile, snapshot=lane)
                if board["pending"]:
                    return
            else:
                tiles, fallback = choice["tiles"], []
        if board["state"] == "candidate_testing":
            tiles, fallback = candidate_targets(snapshot, board["rows"], board["successes"])
        board["state"] = "labeled"
        self.results[board["id"]] = dict(
            rule=board["rule"], tiles=tiles, fallback_rows=fallback, blocked=board["blocked"]
        )
        self.store.release(board["root"])
        board.pop("root")

    def _prepare_boards(self):
        for _ in range(min(8, len(self.board_queue))):
            identity = self.board_queue.popleft()
            board = self.boards[identity]
            snapshot = self.store.get(board["root"])
            if snapshot is None:
                self.board_queue.append(identity)
                continue
            board["state"] = "coverage"
            for row in lane_facts(snapshot)[1]:
                self._board_job(board, row)
            self._resolve_board(board)

    def _cuda_available(self):
        return not self.main_busy and (self.main_event is None or self.main_event.done)

    def _harvest_cuda(self):
        if self.gpu_active is None:
            return
        backend = self.gpu[self.gpu_active]
        completed = backend.poll()
        if completed is None:
            return
        results, milliseconds = completed
        for job in backend.slots:
            if job is not None and "ticks" in job:
                self._account_progress(self.jobs[job["id"]], job["ticks"], "cuda")
        elapsed = milliseconds / 1000
        self.metrics["cuda_seconds"] += elapsed
        if backend.family == "teacher":
            self.metrics["teacher_cuda_seconds"] += elapsed
        for identity, result in results:
            self._complete(
                identity,
                result,
                "cuda",
                max(elapsed, perf_counter() - self.jobs[identity]["started"])
                / max(1, len(results)),
            )
        target = self.settings["auxiliary_cuda_target_ms"]
        maximum = (
            self.forecasts["chunk_ticks"]
            if backend.family == "outcomes"
            else self.teacher_settings["oracle_chunk_ticks"]
        )
        backend.chunk_ticks = max(
            1,
            min(
                maximum,
                int(backend.chunk_ticks * np.clip(target / max(0.01, milliseconds), 0.5, 2)),
            ),
        )
        self.gpu_active = None

    def _dispatch_cuda(self):
        if self.gpu_active is not None or not self._cuda_available():
            self.metrics["admission_denials"] += 1
            return
        for offset in range(2):
            family = ("outcomes", "teacher")[(self.turn + offset) % 2]
            backend = self.gpu.get(family)
            if backend is None:
                continue
            limit = (
                len(backend.slots)
                if self.draining
                else min(
                    len(backend.slots),
                    self.settings[
                        "auxiliary_interval_lanes"
                        if family == "outcomes"
                        else "auxiliary_teacher_lanes"
                    ],
                )
            )
            entries = []
            for slot in range(limit):
                if backend.slots[slot] is not None:
                    continue
                job = self._next(family)
                if job is None:
                    break
                job.update(state="running_cuda", started=perf_counter())
                entries.append((slot, self._payload(job)))
                self.metrics["cuda_jobs"] += 1
                self.metrics["queue_wait_seconds"] += monotonic() - job["queued"]
                self.metrics["peak_queue_age_seconds"] = max(
                    self.metrics["peak_queue_age_seconds"], monotonic() - job["queued"]
                )
            if entries:
                backend.fill(entries)
            if not any(job is not None for job in backend.slots):
                continue
            backend.launch(backend.chunk_ticks)
            self.gpu_active = family
            self.turn = (self.turn + 1) % 2
            return

    def _pause_workers(self):
        # Finish only the current bounded chunk, never the whole job.
        if self.gpu_active is not None:
            self.gpu[self.gpu_active].event.synchronize()
            self._harvest_cuda()
        for backend in self.gpu.values():
            for identity, state in backend.continuations():
                self._account_progress(self.jobs[identity], state["ticks"], "cuda")
                self.jobs[identity].update(continuation=state, state="queued", placed=True)
            backend.slots[:] = [None] * len(backend.slots)
        for slot in self.workers:
            if slot["job"] is None and not slot.get("pausing"):
                continue
            if not slot.get("pausing"):
                slot["connection"].send(("pause", None))
            while True:
                if not slot["connection"].poll(10):
                    raise RuntimeError(
                        "CPU checkpoint boundary timed out; committed continuation retained"
                    )
                kind, identity, state = slot["connection"].recv()
                if kind == "completed":
                    self._complete(identity, state, "cpu", perf_counter() - slot["start"])
                    continue
                if kind == "failed":
                    if identity in self.jobs and state.get("continuation"):
                        self.jobs[identity].update(
                            continuation=state["continuation"], state="queued"
                        )
                    slot["job"] = None
                    raise RuntimeError(state["error"])
                if kind != "paused":
                    raise RuntimeError("Invalid CPU checkpoint handshake")
                if identity is not None:
                    self._account_progress(self.jobs[identity], state["ticks"], "cpu")
                    self.metrics["cpu_seconds"] += perf_counter() - slot["start"]
                    self.jobs[identity].update(continuation=state, state="queued", placed=True)
                slot["job"] = None
                slot["pausing"] = False
                break
        self.queues = {
            family: deque(key for key, j in self.jobs.items() if j["family"] == family)
            for family in ("outcomes", "teacher")
        }

    def _migrate_drain(self):
        if not self.draining or self.gpu_active is not None:
            return
        for slot in self.workers:
            if slot["job"] is None or slot.get("pausing"):
                continue
            job = self.jobs[slot["job"]]
            backend = self.gpu.get(job["family"])
            if (
                backend is not None
                and not any(j is not None for j in backend.slots)
                and self._next(job["family"]) is None
            ):
                slot["connection"].send(("pause", None))
                slot["pausing"] = True
                return

    def _run(self):
        try:
            with self.env.cp.cuda.Device(self.env.batch.device), self.stream:
                while True:
                    with self.condition:
                        if self.closing:
                            return
                        pause = self.pause_requested
                    if pause:
                        self._pause_workers()
                        with self.condition:
                            self.paused = True
                            self.condition.notify_all()
                            self.condition.wait_for(
                                lambda: not self.pause_requested or self.closing
                            )
                            self.paused = False
                        continue
                    self._harvest_cpu()
                    self._harvest_cuda()
                    decoding = perf_counter()
                    # Compaction can discard every row in a capture. Such packets
                    # still need their transfer consumed, without retaining a root.
                    if not self.packed_only:
                        for packet in [p for p in list(self.packets) if p.device is not None][:32]:
                            packet.resolve()
                        self.store.materialize()
                    elif self.live_packet_bytes > self.packet_limit * 0.75:
                        # Pressure only: download cold queued roots on the
                        # coordinator. Ordinary jobs remain on device end to end.
                        self.store.materialize()
                    self.metrics["decode_seconds"] += perf_counter() - decoding
                    if self.draining or not self.defer_until_seal:
                        self._prepare_boards()
                        self._dispatch_cuda()
                        self._dispatch_cpu(self._cuda_available())
                        self._migrate_drain()
                    self.store.spill()
                    cpu_live = sum(w["job"] is not None for w in self.workers)
                    cuda_live = sum(
                        sum(j is not None for j in backend.slots) for backend in self.gpu.values()
                    )
                    self.metrics.update(
                        root_ram_bytes=self.store.nbytes,
                        peak_root_ram_bytes=max(
                            self.metrics["peak_root_ram_bytes"], self.store.nbytes
                        ),
                        uploaded_bytes=self.retired_transfers["uploaded_bytes"]
                        + sum(backend.transfers.uploaded_bytes for backend in self.gpu.values()),
                        progress_bytes=self.retired_transfers["progress_bytes"]
                        + sum(backend.progress_bytes for backend in self.gpu.values()),
                        root_spill_bytes=self.store.spilled,
                        queued_jobs=max(0, len(self.jobs) - cpu_live - cuda_live),
                        live_jobs=len(self.jobs),
                        cpu_live=cpu_live,
                    )
                    if self.drain_started is not None:
                        self.metrics["drain_seconds"] = perf_counter() - self.drain_started
                    with self.condition:
                        self.condition.wait(timeout=0.001)
        except BaseException:
            failure = traceback.format_exc()
            # Roots and committed continuations stay authoritative even if a
            # device/process fails while publishing its newest chunk.
            try:
                with self.env.cp.cuda.Device(self.env.batch.device), self.stream:
                    self._pause_workers()
            except BaseException:
                failure += "\nRecovery boundary: " + traceback.format_exc()
            with self.condition:
                self.failure = failure
                self.condition.notify_all()

    @contextmanager
    def quiescent(self):
        with self.condition:
            self.pause_requested = True
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.paused or self.failure)
        try:
            yield
        finally:
            with self.condition:
                self.pause_requested = False
                self.condition.notify_all()

    def snapshot(self):
        if not self.paused and not self.failure:
            with self.quiescent():
                return self.snapshot()
        with self.condition:
            jobs = copy.deepcopy(self.jobs)
            for job in jobs.values():
                job["queued_age"] = max(0, monotonic() - job["queued"])
            return dict(
                protocol=PROTOCOL,
                families=sorted(self.families),
                jobs=jobs,
                boards=copy.deepcopy(self.boards),
                roots={key: self.store.get(key, wait=True) for key in self.store.entries},
                results=copy.deepcopy(self.results),
                published=set(self.published),
                sealed=set(self.sealed),
                metrics=dict(self.metrics),
                estimates=copy.deepcopy(self.estimates),
            )

    def _restore(self, state):
        if state.get("protocol") != PROTOCOL:
            raise ValueError("Unsupported hybrid simulation recovery")
        roots = {}
        for key, value in state["roots"].items():
            private_game(value, self.env.batch.rules)
            roots[key] = self.store.retain(value)
            self.store.entries[roots[key]]["refs"] = 0
        self.jobs, self.boards = copy.deepcopy(state["jobs"]), copy.deepcopy(state["boards"])
        self.board_queue = deque(
            key for key, board in self.boards.items() if board["state"] == "capturing"
        )
        for record in (*self.jobs.values(), *self.boards.values()):
            if "root" in record:
                record["root"] = roots[record["root"]]
                self.store.entries[record["root"]]["refs"] += 1
        for key, job in self.jobs.items():
            if (
                tuple(job["id"]) != key
                or job["family"] not in self.queues
                or type(job["action"]) is not int
                or not 0 <= job["action"] <= 405
                or job["state"] not in ("capturing", "queued", "running_cpu", "running_cuda")
            ):
                raise ValueError("Invalid hybrid simulation job identity")
            if job.get("continuation"):
                validate_continuation(job, job["continuation"], self.env.batch.rules)
            job["queued"] = monotonic() - job.pop("queued_age", 0)
            job["state"] = "queued"
            self.queues[job["family"]].append(key)
        self.results, self.published, self.sealed = (
            copy.deepcopy(state["results"]),
            set(state["published"]),
            set(state["sealed"]),
        )
        if (
            set(self.results) & self.published
            or set(self.jobs) & (set(self.results) | self.published)
            or not self.sealed <= {"outcomes", "teacher"}
        ):
            raise ValueError("Auxiliary recovery duplicated a publication")
        pending = Counter(job["board"] for job in self.jobs.values() if job["family"] == "teacher")
        for identity, board in self.boards.items():
            if (
                board["id"] != identity
                or board["pending"] != pending[identity]
                or board["state"] not in ("capturing", "coverage", "candidate_testing", "labeled")
                or (board["state"] == "labeled")
                != (identity in self.results or identity in self.published)
            ):
                raise ValueError("Teacher recovery lost a dependency or preferred set")
        if any(
            job["family"] == "teacher" and job["board"] not in self.boards
            for job in self.jobs.values()
        ):
            raise ValueError("Teacher simulation lost its owned root")
        self.metrics.update(
            {
                k: v
                for k, v in state["metrics"].items()
                if k
                not in (
                    "device_bytes",
                    "capture_bytes",
                    "worker_ram_bytes",
                    "root_ram_bytes",
                    "root_spill_bytes",
                )
            }
        )
        self.estimates = copy.deepcopy(state.get("estimates", {}))
        self.retired_transfers = {
            key: state["metrics"].get(key, 0) for key in ("uploaded_bytes", "progress_bytes")
        }
        self.draining = bool(self.sealed)
        if self.draining:
            self.drain_started = perf_counter() - self.metrics.get("drain_seconds", 0)

    def close(self):
        if getattr(self, "thread", None) is not None and self.thread.is_alive():
            with self.quiescent():
                with self.condition:
                    self.closing = True
                    self.condition.notify_all()
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Auxiliary coordinator did not close at a chunk boundary")
        with self.stream:
            for backend in self.gpu.values():
                backend.close()
        self.transfer_stream.synchronize()
        self.packets.clear()
        if self.failure:
            close_cpu_workers(self.env)
        self.store.close()
        self.env.auxiliary = None


def close_cpu_workers(env):
    for slot in getattr(env, "_auxiliary_cpu_workers", []):
        if slot["process"].is_alive():
            slot["connection"].send(("close", None))
            slot["process"].join(timeout=5)
            if slot["process"].is_alive():
                slot["process"].terminate()
                slot["process"].join(timeout=5)
        slot["connection"].close()
    env._auxiliary_cpu_workers = []
