"""Private factual target membership, compressed CUDA traces and resumable labels."""

from enum import StrEnum
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.reach_tracking import position_category, suffix_labels
from pvz_rl.envs.tile_scenarios import is_placement_skill

REACH_PROTOCOL = "visible-zombie-suffix-reach-v1"
EXERCISE_PROTOCOL = "placement-reach-endpoints-v1"
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
            tracked_zombies=0,
            eligible_decisions=0,
            tracking_seconds=0.0,
            storage_bytes=0,
            continuation_seconds=0.0,
            continuation_ticks=0,
            observed_lanes=0,
            conservative_lanes=0,
            stop_reasons={},
        )
        self.exercise = is_placement_skill(env.tile_scenario)
        self._exercise_state = None
        self.conservative_lanes = np.zeros((env.batch.n, 5), bool)
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
            if self.exercise:
                from pvz_rl.envs.reach_exercise import EXERCISE_WIDTH

                if state.get("exercise_protocol") != EXERCISE_PROTOCOL:
                    raise ValueError("Missing placement reach recovery identity")
                saved = state["exercise"]
                if saved.shape != (env.batch.n, EXERCISE_WIDTH) or saved.dtype != torch.int64:
                    raise ValueError("Corrupt placement reach continuation")
                values = saved.numpy()
                mode = 2 if env.tile_scenario == "attacker_pre" else 1
                if (
                    np.any(~np.isin(values[:, 0], (0, mode)))
                    or np.any((values[:, 2] < 0) | (values[:, 2] > 7))
                    or np.any(values[:, 1:7] < 0)
                    or np.any(~np.isin(values[:, 7:9], (0, 1)))
                ):
                    raise ValueError("Invalid placement reach endpoint state")
                env.batch.reach_exercise[:] = env.cp.asarray(saved.numpy())
                self._exercise_state = saved.numpy().copy()
                flags = state["conservative_lanes"]
                if flags.shape != (env.batch.n, 5) or flags.dtype != torch.bool:
                    raise ValueError("Corrupt reach label provenance")
                self.conservative_lanes[:] = flags.numpy()
        else:
            self.path.write_bytes(b"")
            env.batch.enable_reach()
        self.stream = self.path.open("ab")
        self._trace = None

    @property
    def estimated_bytes(self):
        return (
            int(self.env.batch.reach_trace.nbytes + self.env.batch.reach_exercise.nbytes)
            + self.conservative_lanes.nbytes
            + (0 if self._exercise_state is None else self._exercise_state.nbytes)
            + (0 if self._trace is None else self._trace.nbytes)
        )

    def before(self, actions, active):
        """Arm private endpoint hooks without changing the main action or cadence."""
        if not self.exercise:
            return
        from pvz_rl.envs.reach_exercise import EXERCISE_WIDTH

        cp, batch = self.env.cp, self.env.batch
        data = np.zeros((batch.n, EXERCISE_WIDTH), np.int64)
        for i, board in self.env.tile_boards.items():
            data[i, 0] = 2 if self.env.tile_scenario == "attacker_pre" else 1
            wanted = board.get("target_ids", [])
            if board.get("supplied_plant") == "potato_mine":
                wanted = wanted[:1]
            data[i, 5 : 5 + len(wanted)] = wanted
        data[:, 4] = int(
            self.env.cfg["environment"]["cutoff_seconds"] * batch.rules.game["tick_rate"]
        )
        staged = cp.asarray(data)
        staged[:, 1] = batch.header[:, 3]
        selected = cp.from_dlpack(active)
        batch.reach_exercise[:] = cp.where(selected[:, None], staged, batch.reach_exercise)
        self._exercise_state = None

    def continue_exercises(self):
        """One bounded Wait-only device chunk, outside all episode accounting."""
        if not self.exercise:
            return True
        from pvz_rl.envs.reach_exercise import STOP_REASONS

        cp, batch = self.env.cp, self.env.batch
        started = perf_counter()
        current = self._exercise_state
        if current is None:
            current = batch.reach_exercise.get()
        pending = (current[:, 0] != 0) & (current[:, 2] == 0)
        if pending.any():
            batch.reach_exercise[:, 8] = 1
            batch.header[:, 17] = cp.asarray(pending)
            try:
                batch.step_device(
                    cp.zeros(batch.n, cp.int64),
                    ticks=min(
                        batch.max_step_ticks,
                        self.env.cfg["training"]["performance"].get("reach_chunk_ticks", 256),
                    ),
                    per_tick=False,
                )
                progress = cp.concatenate(
                    (batch.reach_exercise, batch.header[:, 14:15]), axis=1
                ).get()
                current = progress[:, :-1]
                self.metrics["continuation_ticks"] += int(progress[pending, -1].sum())
            finally:
                batch.header[:, 17] = 0
            pending = (current[:, 0] != 0) & (current[:, 2] == 0)
        self._exercise_state = current
        self.metrics["continuation_seconds"] += perf_counter() - started
        if pending.any():
            return False
        self.metrics["stop_reasons"] = {
            name: int(np.count_nonzero((current[:, 0] != 0) & (current[:, 2] == code)))
            for code, name in STOP_REASONS.items()
        }
        return True

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
            if not self.continue_exercises():
                return False
            started = perf_counter()
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
        observed_lanes = conservative_lanes = 0
        try:
            for index in range(self.cursor, last):
                block, offset = divmod(index, self.buffer.block_rows)
                row = self.buffer.blocks[block][offset]
                labels = np.full(5, -1, np.int8)
                conservative = np.zeros(5, bool)
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
                    if self.exercise:
                        trace = {int(x[0]): x for x in self._trace[int(row["env"])] if x[0]}
                        for identity, lane, _ in targets:
                            if not trace[int(identity)][3]:
                                conservative[int(lane)] = True
                        labels[conservative] = 0
                        self.conservative_lanes[int(row["env"])] = conservative
                row["reach_labels"] = labels
                observed_lanes += int(((labels >= 0) & ~conservative).sum())
                conservative_lanes += int(conservative.sum())
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
                            conservative_lanes=conservative.tolist(),
                            stop_reason=int(self._exercise_state[int(row["env"]), 2])
                            if self.exercise
                            else None,
                        )
                    )
            self.cursor = last
            self.metrics["observed_lanes"] += observed_lanes
            self.metrics["conservative_lanes"] += conservative_lanes
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
            exercise_protocol=EXERCISE_PROTOCOL if self.exercise else None,
            exercise=torch.from_numpy(self.env.batch.reach_exercise.get())
            if self.exercise
            else None,
            conservative_lanes=torch.from_numpy(self.conservative_lanes.copy())
            if self.exercise
            else None,
        )

    def write_archive(self, archive):
        self.stream.flush()
        archive.write(self.path, "reach/targets.bin")

    def close(self):
        self.stream.close()
        self._trace = None
        self.env.batch.disable_reach()
