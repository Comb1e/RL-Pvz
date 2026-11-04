"""Absolute auxiliary targets from real Wait prefixes and isolated suffix jobs."""

import copy
import pickle
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter

import numpy as np
from pvz_game.cuda import CudaBatch
from pvz_game.cuda.schema import HEADER

from pvz_rl.envs.cuda_snapshots import download_snapshots, upload_snapshots
from pvz_rl.envs.fork_execution import PHYSICAL
from pvz_rl.envs.fork_service import ForkService
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.envs.wait_capture import WaitCapture
from pvz_rl.learning.forecast_labels import outcome_codes


class IntervalPhase(StrEnum):
    FOLLOWING = "following_real_waits"
    FIELD_PENDING = "field_continuation_pending"
    FIELD_RUNNING = "field_continuation_running"
    PROBE_PENDING = "probe_pending"
    RUNNING = "probe_running"
    READY = "ready"


class WaitOutcomes:
    protocol = "absolute-wait-trajectory-v3"

    def __init__(self, env, buffer, state=None):
        self.env, self.buffer = env, buffer
        self.current = np.full(env.num_envs, -1, np.int64)
        self.intervals, self.rows, self.jobs = {}, {}, []
        self.next_id = 0
        self.live = []
        self.source = self.service = None
        self.published = False
        self.store = TemporaryDirectory(prefix="pvz-outcomes-")
        self.metrics = dict(
            borrowed_real_ticks=0,
            field_ticks=0,
            probe_ticks=0,
            interval_probes=0,
            probe_reuses=0,
            alternative_jobs=0,
            sampled_waits=0,
            skipped_waits=0,
            queued_jobs=0,
            peak_queued_jobs=0,
            root_disk_bytes=0,
            capture_bytes=0,
            labeling_seconds=0.0,
            published_rows=0,
            live_jobs=0,
            device_bytes=0,
        )
        env.batch.wait_capture = WaitCapture(env.batch)
        self.metrics["capture_bytes"] = env.batch.wait_capture.nbytes
        if not env.batch.reach_enabled:
            env.batch.enable_reach()
        if state is not None:
            self.restore(state)

    @property
    def estimated_bytes(self):
        return (
            len(self.rows) * 512
            + len(self.intervals) * 512
            + sum(sum(d.nbytes for _, d in r["segments"]) for r in self.intervals.values())
            + sum(job["root_bytes"] for job in self.jobs if "snapshot" in job)
            + sum(job["root_bytes"] for job in self.live)
        )

    def arm(self, request):
        watch = np.ones(self.env.num_envs, bool)
        for slot, identity in enumerate(self.current):
            if identity >= 0:
                watch[slot] = self.intervals[int(identity)]["phase"] == IntervalPhase.FOLLOWING
        watch |= request["legal_proposal"] & (request["actions"][:, 0] != 0)
        self.env.batch.wait_capture.arm(watch)

    def _new(self, slot, tick):
        identity = self.next_id
        self.next_id += 1
        self.intervals[identity] = dict(
            open=True,
            phase=IntervalPhase.FOLLOWING,
            start=int(tick),
            anchor=-1,
            origin=-1,
            segments=[],
            homes=np.full(5, -1, np.int64),
            outcome=None,
            decision_uses=0,
        )
        self.current[slot] = identity
        return identity

    def _root(self, job):
        if "snapshot" in job:
            return job["snapshot"]
        return pickle.loads((Path(self.store.name) / job["file"]).read_bytes())

    def _spill_roots(self):
        # Commit references before filesystem work. An allocation or I/O failure
        # retains the in-memory root in semantic recovery, with no resampling.
        for job in self.jobs:
            if "snapshot" not in job:
                continue
            path = Path(self.store.name) / job["file"]
            temporary = path.with_suffix(".tmp")
            payload = pickle.dumps(job["snapshot"], protocol=5)
            temporary.write_bytes(payload)
            temporary.replace(path)
            job.pop("snapshot")
            self.metrics["root_disk_bytes"] += len(payload)

    def _enqueue(self, snapshot, *, interval=None, row=None, token=None, action=0):
        key = f"root-{self.next_id}-{len(self.jobs)}-{self.metrics['queued_jobs']}.pkl"
        size = (
            8192
            + len(snapshot["level"]["spawns"]) * 128
            + len(snapshot["plants"]) * 256
            + len(snapshot["zombies"]) * 512
            + len(snapshot["projectiles"]) * 256
        )
        self.jobs.append(
            dict(
                file=key,
                interval=interval,
                row=row,
                token=token,
                action=int(action),
                snapshot=snapshot,
                root_bytes=size,
            )
        )
        self.metrics["queued_jobs"] += 1
        self.metrics["peak_queued_jobs"] = max(self.metrics["peak_queued_jobs"], len(self.jobs))

    def _finish_interval(self, identity, snapshot=None, *, cleared=False):
        record = self.intervals[identity]
        if record["phase"] != IntervalPhase.FOLLOWING:
            return
        if snapshot is None:
            raise RuntimeError("Unfinished interval lost its exact physical continuation")
        record["origin"] = int(snapshot["tick"])
        record["phase"] = IntervalPhase.PROBE_PENDING if cleared else IntervalPhase.FIELD_PENDING
        if cleared:
            record["anchor"] = int(snapshot["tick"])
        self._enqueue(snapshot, interval=identity)

    def _close(self, identity, snapshot=None):
        self.intervals[identity]["open"] = False
        self._finish_interval(identity, snapshot)

    def observe(self, request, rows, receipt):
        """Consume completed real actions atomically with their buffer receipts."""
        active = rows["active"]
        meta, captures = self.env.batch.wait_capture.read(active)
        slots_to_snapshot = [
            int(slot)
            for slot in np.flatnonzero(active & rows["done"])
            if meta[slot, 0] < 0 and meta[slot, 1] < 0
        ]
        terminals = dict(
            zip(
                slots_to_snapshot,
                download_snapshots(self.env.batch, indices=slots_to_snapshot),
                strict=True,
            )
        )
        locations = dict(zip(receipt.slots.tolist(), receipt.rows.tolist(), strict=True))
        for slot in np.flatnonzero(active):
            slot = int(slot)
            row_index = int(locations[slot])
            tick = int(rows["tick"][slot])
            old = int(self.current[slot])
            if old < 0:
                old = self._new(slot, tick)
            accepted = bool(rows["accepted"][slot] and rows["action"][slot] != 0)
            if accepted != bool(
                request["legal_proposal"][slot] and request["actions"][slot, 0] != 0
            ):
                raise RuntimeError("Saved action legality disagrees with actual execution")
            if accepted:
                self._close(old, request["pre_states"].get(slot))
                actual = self._new(slot, tick)
            else:
                actual = old
            record = self.intervals[actual]
            count = int(rows["duration"][slot])
            boundary = int(meta[slot, 0])
            spawned = bool(self.env._last_spawned[slot])
            if spawned != (boundary >= 0):
                raise RuntimeError("Spawn counter changed without an exact pre-tick snapshot")
            if (
                spawned
                and int(self.env.batch.header[slot, HEADER.index("spawn_index")].get())
                <= meta[slot, 2]
            ):
                raise RuntimeError("Predicted spawn boundary was not confirmed by execution")
            stop = boundary if spawned else tick + count
            trace = np.array(rows["forecast_trace"][slot, 1 : stop - tick + 1], dtype=np.uint8)
            if len(trace):
                record["segments"].append((tick + 1, trace))
            for lane, when in enumerate(rows["forecast_breach"][slot]):
                if tick < when <= stop and record["homes"][lane] < 0:
                    record["homes"][lane] = when
            self.metrics["borrowed_real_ticks"] += stop - tick
            if slot in captures[1]:
                self._finish_interval(actual, captures[1][slot], cleared=True)
            if spawned:
                self._close(actual, captures[0][slot])
                self.current[slot] = -1
            elif rows["done"][slot]:
                self._close(actual, terminals.get(slot))
                self.current[slot] = -1
            for identity in {old, actual}:
                interval = self.intervals[identity]
                self.metrics["probe_reuses"] += int(interval["decision_uses"] > 0)
                interval["decision_uses"] += 1
            self.rows[row_index] = dict(
                tick=tick,
                null=old,
                actual=actual,
                initial=np.array(rows["forecast_trace"][slot, 0], copy=True),
                actions=request["actions"][slot].copy(),
                valid=request["valid"][slot].copy(),
                alternatives={},
            )
            rows["outcome_interval"][slot] = actual
            block, offset = divmod(row_index, self.buffer.block_rows)
            self.buffer.blocks[block][offset]["outcome_interval"] = actual
            for token in np.flatnonzero(request["valid"][slot])[1:]:
                self._enqueue(
                    request["pre_states"][slot],
                    row=row_index,
                    token=int(token),
                    action=int(request["actions"][slot, token]),
                )
                self.metrics["alternative_jobs"] += 1
            self.metrics["sampled_waits"] += int(
                request["sampled"][slot] and request["effective_wait"][slot]
            )
            self.metrics["skipped_waits"] += int(
                not request["sampled"][slot] and request["effective_wait"][slot]
            )

    def _allocate(self):
        if self.source is not None:
            return
        self.env.queue.release_scratch()
        b, settings = self.env.batch, self.env.cfg["training"]["forecasts"]
        per_root = sum(getattr(b, name)[0].nbytes for name in PHYSICAL)
        remaining = settings["vram_mib"] * 2**20 - self.metrics["capture_bytes"]
        count = min(32, b.n, settings["lanes"], max(0, remaining // (3 * (per_root + 65536))))
        if count < 1:
            raise MemoryError("Wait snapshots and one exact worker exceed the shared fork VRAM cap")
        source = CudaBatch(
            int(count),
            rules=b.rules,
            zombie_capacity=b.zcap,
            projectile_capacity=b.qcap,
            max_step_ticks=settings["chunk_ticks"],
            device=b.device,
        )
        root_bytes = sum(v.nbytes for v in vars(source).values() if isinstance(v, b.cp.ndarray))
        remaining -= root_bytes
        cfg = dict(
            settings,
            **{
                k: v
                for k, v in self.env.cfg["training"]["performance"].items()
                if k.startswith("fork_")
            },
        )
        cfg["vram_mib"] = int(remaining // 2**20)
        cfg["lanes"] = min(cfg["lanes"], int(count))
        cfg["headroom_mib"] = self.env.cfg["training"]["performance"].get(
            "collection_vram_headroom_mib", 0
        )
        service = ForkService(source, cfg, sky_sun_rate=b.sky_sun_rate)
        self.source, self.service = source, service

    def advance(self):
        """One bounded labeling unit; collection need not await its completion."""
        if not self.jobs:
            return True
        started = perf_counter()
        try:
            self._spill_roots()
            self._allocate()
            if not self.live:
                self.live = self.jobs[: self.source.n]
            if self.service.request is None:
                snapshots = [self._root(job) for job in self.live]
                upload_snapshots(self.source, snapshots, indices=range(len(snapshots)))
                self.service.invalidate()
                self.service.begin(range(len(self.live)), [job["action"] for job in self.live])
            complete = self.service.advance()
            anchors = self.service.execution.anchors[: len(self.live)].get()
            for job, anchor in zip(self.live, anchors, strict=True):
                if job["interval"] is not None:
                    record = self.intervals[job["interval"]]
                    record["phase"] = (
                        IntervalPhase.RUNNING if anchor >= 0 else IntervalPhase.FIELD_RUNNING
                    )
                    record["anchor"] = int(anchor)
            if not complete:
                return False
            outcomes = self.service.results
            for job, outcome, anchor in zip(self.live, outcomes, anchors, strict=True):
                if job["interval"] is not None:
                    record = self.intervals[job["interval"]]
                    record.update(outcome=outcome, anchor=int(anchor), phase=IntervalPhase.READY)
                    if record.get("counted") is not True:
                        self.metrics["interval_probes"] += 1
                        record["counted"] = True
                else:
                    self.rows[job["row"]]["alternatives"][job["token"]] = outcome_codes(
                        outcome, reference_tau(self.env.batch.rules)
                    )
            self.service.take()
            completed = self.live
            del self.jobs[: len(completed)]
            self.live = []
            for job in completed:
                path = Path(self.store.name) / job["file"]
                self.metrics["root_disk_bytes"] -= path.stat().st_size
                path.unlink()
            return not self.jobs
        finally:
            self.metrics["live_jobs"] = len(self.live)
            if self.service is not None:
                self.metrics["field_ticks"] = self.service.metrics.get("field_ticks", 0)
                self.metrics["probe_ticks"] = self.service.metrics.get("probe_ticks", 0)
                self.metrics["device_bytes"] = (
                    self.metrics["capture_bytes"]
                    + self.service.metrics["device_bytes"]
                    + sum(
                        value.nbytes
                        for value in vars(self.source).values()
                        if isinstance(value, self.source.cp.ndarray)
                    )
                )
            self.metrics["labeling_seconds"] += perf_counter() - started

    def _index(self, record):
        if "suffix_depth" in record:
            return
        ticks, depths = [], []
        for start, values in record["segments"]:
            ticks.append(np.arange(start, start + len(values)))
            depths.append(values)
        record["ticks"] = np.concatenate(ticks) if ticks else np.empty(0, np.int64)
        record["suffix_depth"] = (
            np.maximum.accumulate(np.concatenate(depths)[::-1], axis=0)[::-1]
            if depths
            else np.empty((0, 5), np.uint8)
        )

    def target(self, identity, tick, initial):
        record = self.intervals[identity]
        if record["phase"] != IntervalPhase.READY:
            raise RuntimeError("An unfinished interval cannot publish auxiliary targets")
        self._index(record)
        result = record["outcome"].copy()
        at = np.searchsorted(record["ticks"], tick, side="right")
        observed = (
            record["suffix_depth"][at] if at < len(record["ticks"]) else np.zeros(5, np.int64)
        )
        result[:, 0] = np.maximum(
            np.maximum(initial, observed), result[:, 0] if tick <= record["origin"] else 0
        )
        homes = record["homes"].copy()
        tail = record["outcome"][:, 1]
        tail = np.where(tail >= 0, tail + record["origin"], -1)
        homes = np.where((homes < 0) | ((tail >= 0) & (tail < homes)), tail, homes)
        result[:, 1] = np.where(homes >= tick, homes - tick, -1)
        result[:, 2] = result[:, 1] >= 0
        return result

    def finalize(self):
        if self.published:
            return True
        # Diagnostic collection windows can stop with live games. Preserve the
        # observed prefix and resolve the untouched remainder privately.
        slots = np.flatnonzero(self.current >= 0).tolist()
        need = [
            slot
            for slot in slots
            if self.intervals[int(self.current[slot])]["phase"] == IntervalPhase.FOLLOWING
        ]
        snapshots = dict(zip(need, download_snapshots(self.env.batch, indices=need), strict=True))
        for slot in slots:
            self._close(int(self.current[slot]), snapshots.get(slot))
            self.current[slot] = -1
        if not self.advance():
            return False
        tau = reference_tau(self.env.batch.rules)
        for index, record in self.rows.items():
            block, offset = divmod(index, self.buffer.block_rows)
            row = self.buffer.blocks[block][offset]
            before = self.target(record["null"], record["tick"], record["initial"])
            after = self.target(record["actual"], record["tick"], record["initial"])
            row["baseline_outcomes"] = before
            row["effect_labels"][0] = outcome_codes(after, tau)
            for token, label in record["alternatives"].items():
                row["effect_labels"][token] = label
            if any(
                int(token) not in record["alternatives"]
                for token in np.flatnonzero(record["valid"])[1:]
            ):
                raise RuntimeError("A sampled alternative lost its exact target")
            row["outcome_ready"] = True
        self.metrics["published_rows"] = len(self.rows)
        self.published = True
        return True

    def snapshot(self):
        return dict(
            protocol=self.protocol,
            current=self.current.copy(),
            next_id=self.next_id,
            intervals={
                identity: {
                    key: copy.deepcopy(value)
                    for key, value in record.items()
                    if key not in ("ticks", "suffix_depth")
                }
                for identity, record in self.intervals.items()
            },
            rows=copy.deepcopy(self.rows),
            jobs=[dict(job, snapshot=self._root(job)) for job in self.jobs],
            live=len(self.live),
            metrics=dict(self.metrics),
            published=self.published,
            service=self.service.snapshot() if self.service is not None else None,
            source=download_snapshots(self.source, indices=range(len(self.live)))
            if self.live and self.service.request is not None
            else [],
        )

    def restore(self, state):
        if state.get("protocol") != self.protocol:
            raise ValueError("Incompatible absolute Wait trajectory recovery")
        self.current = np.asarray(state["current"]).copy()
        if self.current.shape != (self.env.num_envs,) or self.current.dtype != np.int64:
            raise ValueError("Invalid saved Wait interval identities")
        self.next_id, self.intervals, self.rows = (
            state["next_id"],
            copy.deepcopy(state["intervals"]),
            copy.deepcopy(state["rows"]),
        )
        if (
            type(self.next_id) is not int
            or self.next_id < 0
            or any(type(key) is not int or not 0 <= key < self.next_id for key in self.intervals)
        ):
            raise ValueError("Invalid saved interval directory")
        for identity in self.current[self.current >= 0]:
            if int(identity) not in self.intervals or not self.intervals[int(identity)]["open"]:
                raise ValueError("Current Wait interval is missing or closed")
        for record in self.intervals.values():
            if record["phase"] not in set(IntervalPhase) or type(record["open"]) is not bool:
                raise ValueError("Invalid saved interval lifecycle")
            if record["start"] < 0 or record["anchor"] < -1 or record["origin"] < -1:
                raise ValueError("Invalid interval anchor time")
            if record["phase"] == IntervalPhase.READY:
                outcome = np.asarray(record["outcome"])
                if outcome.shape != (5, 4) or outcome.dtype != np.int64 or record["anchor"] < 0:
                    raise ValueError("Completed interval lost its probe outcome")
        if not 0 <= state["live"] <= len(state["jobs"]):
            raise ValueError("Invalid live Wait job prefix")
        for record in self.rows.values():
            if record["null"] not in self.intervals or record["actual"] not in self.intervals:
                raise ValueError("Saved outcome row lost its interval")
        self.jobs = []
        for saved in state["jobs"]:
            job = dict(saved)
            snapshot = job.pop("snapshot")
            if Path(job["file"]).name != job["file"]:
                raise ValueError("Saved Wait root escaped its owned storage")
            job["snapshot"] = snapshot
            if job["interval"] is not None and job["interval"] not in self.intervals:
                raise ValueError("Pending probe lost its interval")
            if job["row"] is not None and job["row"] not in self.rows:
                raise ValueError("Pending alternative lost its decision row")
            self.jobs.append(job)
        pending = {job["interval"] for job in self.jobs if job["interval"] is not None}
        if any(
            record["phase"]
            in (
                IntervalPhase.FIELD_PENDING,
                IntervalPhase.FIELD_RUNNING,
                IntervalPhase.PROBE_PENDING,
                IntervalPhase.RUNNING,
            )
            and identity not in pending
            for identity, record in self.intervals.items()
        ):
            raise ValueError("Unfinished interval has no semantic probe job")
        self.metrics = dict(state["metrics"])
        self.metrics["root_disk_bytes"] = 0
        self.published = bool(state["published"])
        if state["service"] is not None:
            self._allocate()
            self.live = self.jobs[: state["live"]]
            if state["source"]:
                upload_snapshots(self.source, state["source"], indices=range(len(self.live)))
            self.service.restore(state["service"])

    def close(self):
        if self.service is not None:
            self.service.close()
        self.source = self.service = None
        self.env.batch.wait_capture = None
        target = Path(self.store.name).resolve()
        if target.parent != Path(gettempdir()).resolve() or not target.name.startswith(
            "pvz-outcomes-"
        ):
            raise RuntimeError("Wait root directory escaped its owned temporary location")
        self.store.cleanup()
