"""Private factual target membership, compressed CUDA traces and resumable labels."""

from enum import StrEnum
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.reach_tracking import position_category, suffix_labels

REACH_PROTOCOL = "visible-zombie-suffix-reach-v1"
TARGET_DTYPE = np.dtype([("id", "<i8"), ("lane", "<i8"), ("initial", "<i8")])


class ReachPhase(StrEnum):
    COLLECTING = "collecting"
    TRACES_COMPLETE = "traces_complete"
    LABELS_READY = "labels_ready"
    FITTING = "fitting"
    COMMITTED = "committed"


class ReachSupervision:
    def __init__(self, env, buffer, state=None, archive=None):
        self.env, self.buffer = env, buffer
        self.path = Path(buffer.path) / "visible-reach-targets.bin"
        self.phase, self.cursor, self.targets = ReachPhase.COLLECTING, 0, 0
        self.metrics = dict(
            tracked_zombies=0, eligible_decisions=0, tracking_seconds=0.0, storage_bytes=0
        )
        if state is not None:
            if state.get("protocol") != REACH_PROTOCOL:
                raise ValueError("Incompatible reach tracking recovery")
            self.phase, self.cursor, self.targets = (
                ReachPhase(state["phase"]),
                state["cursor"],
                state["targets"],
            )
            if (
                type(self.cursor) is not int
                or not 0 <= self.cursor <= buffer.size
                or type(self.targets) is not int
                or self.targets < 0
            ):
                raise ValueError("Corrupt reach recovery progress")
            trace = state["trace"]
            if (
                trace.dtype != torch.int64
                or trace.ndim != 3
                or state["errors"].shape != (env.batch.n,)
            ):
                raise ValueError("Corrupt reach recovery trace")
            self.metrics.update(state["metrics"])
            if archive is None:
                raise ValueError("Reach recovery requires its private target archive")
            with archive.open("reach/targets.bin") as source, self.path.open("wb") as target:
                import shutil

                shutil.copyfileobj(source, target, length=1024**2)
            if self.path.stat().st_size != self.targets * TARGET_DTYPE.itemsize:
                raise ValueError("Truncated reach target recovery")
            env.batch.enable_reach(state["trace"].numpy())
            env.batch.reach_error[:] = env.cp.asarray(state["errors"].numpy())
        else:
            self.path.write_bytes(b"")
            env.batch.enable_reach()
        self.stream = self.path.open("ab")
        self._trace = None

    @property
    def estimated_bytes(self):
        return int(self.env.batch.reach_trace.nbytes) + (
            0 if self._trace is None else self._trace.nbytes
        )

    def capture(self, rows, observations, identities):
        """Called after the existing handoff, before the trajectory append commits."""
        if self.phase != ReachPhase.COLLECTING:
            raise RuntimeError("Cannot append to completed reach traces")
        started = perf_counter()
        for slot in np.flatnonzero(rows["active"] & rows["branch_usable"] & (rows["action"] != 0)):
            count, offset = int(observations.counts[slot]), int(observations.offsets[slot])
            ids = identities[slot, :count]
            selected = np.flatnonzero(ids > 0)
            rows["reach_offset"][slot] = self.targets
            rows["reach_count"][slot] = len(selected)
            if not len(selected):
                continue
            public = observations.entities[offset : offset + count][selected]
            targets = np.empty(len(selected), TARGET_DTYPE)
            targets["id"] = ids[selected]
            targets["lane"] = public[:, 2]
            targets["initial"] = [
                position_category(x, self.env.batch.rules.game["units_per_tile"])
                for x in public[:, 3]
            ]
            self.stream.write(targets.tobytes())
            self.targets += len(targets)
            self.metrics["tracked_zombies"] += len(targets)
            self.metrics["eligible_decisions"] += 1
        self.metrics["tracking_seconds"] += perf_counter() - started
        self.metrics["storage_bytes"] = self.targets * TARGET_DTYPE.itemsize + self.estimated_bytes

    def prepare(self):
        """One bounded, idempotent metadata block per deadline/interruption boundary."""
        started = perf_counter()
        if self.phase == ReachPhase.COLLECTING:
            self.stream.flush()
            if bool(self.env.batch.reach_error.any()):
                raise RuntimeError(
                    "CUDA reach tracker exhausted capacity or lost identity; recovery retained"
                )
            self.phase = ReachPhase.TRACES_COMPLETE
        if self.phase != ReachPhase.TRACES_COMPLETE:
            return True
        if self._trace is None:
            self._trace = self.env.batch.reach_trace.get()
        last = min(self.buffer.size, self.cursor + self.buffer.block_rows)
        if self.targets:
            data = np.memmap(self.path, dtype=TARGET_DTYPE, mode="r", shape=(self.targets,))
        else:
            data = np.empty(0, TARGET_DTYPE)
        try:
            for index in range(self.cursor, last):
                block, offset = divmod(index, self.buffer.block_rows)
                row = self.buffer.blocks[block][offset]
                labels = np.full(5, -1, np.int8)
                if row["reach_count"]:
                    begin, count = int(row["reach_offset"]), int(row["reach_count"])
                    if begin + count > self.targets:
                        raise ValueError("Reach membership exceeds private target archive")
                    targets = data[begin : begin + count].view("<i8").reshape(-1, 3)
                    if (
                        np.any(targets[:, 0] <= 0)
                        or np.any((targets[:, 1] < 0) | (targets[:, 1] >= 5))
                        or np.any((targets[:, 2] < 0) | (targets[:, 2] > 9))
                    ):
                        raise ValueError("Invalid private reach target membership")
                    labels = suffix_labels(targets, self._trace[int(row["env"])], int(row["tick"]))
                row["reach_labels"] = labels
                examples = self.metrics.setdefault("examples", [])
                if (
                    row["reach_count"]
                    and len(examples) < 4
                    and not any(
                        item["env"] == int(row["env"]) and item["labels"] == labels.tolist()
                        for item in examples
                    )
                ):
                    examples.append(
                        dict(
                            row=index,
                            env=int(row["env"]),
                            tick=int(row["tick"]),
                            prediction=row["reach_prediction"].astype(float).tolist(),
                            labels=labels.tolist(),
                        )
                    )
            self.cursor = last
        finally:
            if isinstance(data, np.memmap):
                data._mmap.close()
        self.metrics["tracking_seconds"] += perf_counter() - started
        self.metrics["storage_bytes"] = self.targets * TARGET_DTYPE.itemsize + self.estimated_bytes
        if self.cursor == self.buffer.size:
            self.phase = ReachPhase.LABELS_READY
        return self.phase == ReachPhase.LABELS_READY

    def snapshot(self):
        self.stream.flush()
        return dict(
            protocol=REACH_PROTOCOL,
            phase=self.phase.value,
            cursor=self.cursor,
            targets=self.targets,
            metrics=dict(self.metrics),
            trace=torch.from_numpy(self.env.batch.reach_trace.get()),
            errors=torch.from_numpy(self.env.batch.reach_error.get()),
        )

    def write_archive(self, archive):
        self.stream.flush()
        archive.write(self.path, "reach/targets.bin")

    def close(self):
        self.stream.close()
        self._trace = None
        self.env.batch.disable_reach()
