"""Resumable future-visit targets; one-placement exercises use snapshot constraints."""

from enum import StrEnum
from time import perf_counter

import torch

from pvz_rl.envs.reach_tracking import LABEL_POLICY, future_labels
from pvz_rl.envs.tile_scenarios import is_placement_skill

REACH_PROTOCOL = "future-cell-home-supervision-v2"


class ReachPhase(StrEnum):
    COLLECTING = "collecting"
    TRACES_COMPLETE = "traces_complete"
    LABELS_READY = "labels_ready"
    FITTING = "fitting"
    COMMITTED = "committed"


class ReachSupervision:
    def __init__(self, env, buffer, state=None, archive=None):
        self.env, self.buffer = env, buffer
        self.snapshot_only = is_placement_skill(env.tile_scenario)
        self.exercise = env.tile_scenario == "peashooter"
        self.phase, self.cursor = ReachPhase.COLLECTING, 0
        self._started = env.cp.zeros(env.batch.n, env.cp.bool_)
        self._trace = self._final_header = self._final_facts = None
        self.metrics = dict(
            label_policy=LABEL_POLICY,
            endpoint_policy="snapshot_only"
            if self.snapshot_only
            else ("natural_game_end" if self.exercise else "factual_game_or_cutoff"),
            eligible_decisions=0,
            tracking_seconds=0.0,
            storage_bytes=0,
            continuation_seconds=0.0,
            continuation_ticks=0,
            observed_cells=0,
            snapshot_cells=0,
            unknown_cells=0,
            home_targets=0,
            unknown_homes=0,
            stop_reasons={},
        )
        if state is None:
            env.batch.enable_reach()
        else:
            if state.get("protocol") != REACH_PROTOCOL or state.get("label_policy") != LABEL_POLICY:
                raise ValueError(
                    "Incompatible reach recovery; start fresh with future cell/home supervision"
                )
            self.phase, self.cursor = ReachPhase(state["phase"]), state["cursor"]
            if type(self.cursor) is not int or not 0 <= self.cursor <= buffer.size:
                raise ValueError("Corrupt reach preparation cursor")
            trace, started = state["trace"], state["started"]
            if (
                trace.dtype != torch.int64
                or trace.shape != (env.batch.n, 5, 10)
                or (trace < 0).any()
                or state["errors"].dtype != torch.int64
                or state["errors"].shape != (env.batch.n,)
                or started.dtype != torch.bool
                or started.shape != (env.batch.n,)
            ):
                raise ValueError("Corrupt future-reach recovery")
            env.batch.enable_reach(trace.numpy())
            env.batch.reach_error[:] = env.cp.asarray(state["errors"].numpy())
            self._started[:] = env.cp.asarray(started.numpy())
            self.metrics.update(state["metrics"])
            for name in ("trace", "final_header", "final_facts"):
                value = state.get("prepared_" + name)
                setattr(self, "_" + name, None if value is None else value.numpy().copy())
            if self.phase != ReachPhase.COLLECTING and (
                self._trace is None
                or self._trace.shape != (env.batch.n, 5, 10)
                or self._final_header is None
                or self._final_header.shape != env.batch.header.shape
                or self._final_facts is None
                or self._final_facts.shape != (env.batch.n, 5, 3)
            ):
                raise ValueError("Incomplete prepared future-reach recovery")

    @property
    def estimated_bytes(self):
        return int(
            self.env.batch.reach_trace.nbytes
            + self.env.batch.reach_error.nbytes
            + self._started.nbytes
        ) + sum(
            x.nbytes for x in (self._trace, self._final_header, self._final_facts) if x is not None
        )

    def before(self, actions, active):
        self._started |= self.env.cp.from_dlpack(active)

    def continue_exercises(self):
        if not self.exercise:
            return True
        cp, batch = self.env.cp, self.env.batch
        start = perf_counter()
        header = batch.header.get()
        pending = self._started & cp.asarray(header[:, 1] == 0)
        if bool(pending.any()):
            batch.header[:, 17] = pending
            try:
                batch.step_device(
                    cp.zeros(batch.n, cp.int64),
                    ticks=min(
                        batch.max_step_ticks,
                        self.env.cfg["training"]["performance"].get("reach_chunk_ticks", 256),
                    ),
                    per_tick=False,
                )
                header = batch.header.get()
                self.metrics["continuation_ticks"] += int(header[cp.asnumpy(pending), 14].sum())
            finally:
                batch.header[:, 17] = 0
        self.metrics["continuation_seconds"] += perf_counter() - start
        pending_host = cp.asnumpy(self._started) & (header[:, 1] == 0)
        if pending_host.any():
            return False
        started = cp.asnumpy(self._started)
        self.metrics["stop_reasons"] = {
            "natural_win": int(((header[:, 1] == 1) & started).sum()),
            "home_breach": int(((header[:, 1] == 2) & started).sum()),
        }
        return True

    def capture(self, rows):
        if self.phase != ReachPhase.COLLECTING:
            raise RuntimeError("Cannot append to completed reach traces")
        self.metrics["eligible_decisions"] += int(rows["reach_valid"].sum())
        self.metrics["storage_bytes"] = self.estimated_bytes

    def prepare(self):
        start = perf_counter()
        if self.phase == ReachPhase.COLLECTING:
            if not self.continue_exercises():
                return False
            if bool(self.env.batch.reach_error.any()):
                raise RuntimeError("CUDA future-reach tracker failed; recovery retained")
            self._trace = self.env.batch.reach_trace.get()
            self._final_header = self.env.batch.header.get()
            # Full public endpoint facts, without policy inference or trajectory changes.
            saved = self.env.batch.header[:, 17].copy()
            try:
                self.env.batch.header[:, 17] = self._started
                self.env.features.encode(publish=False)
                self._final_facts = self.env.features.reach_facts[: self.env.batch.n].get()
            finally:
                self.env.batch.header[:, 17] = saved
            self.phase = ReachPhase.TRACES_COMPLETE
        if self.phase != ReachPhase.TRACES_COMPLETE:
            return True
        last = min(self.buffer.size, self.cursor + self.buffer.block_rows)
        counts = dict(
            observed_cells=0, snapshot_cells=0, unknown_cells=0, home_targets=0, unknown_homes=0
        )
        examples = list(self.metrics.get("examples", []))
        for index in range(self.cursor, last):
            block, offset = divmod(index, self.buffer.block_rows)
            row = self.buffer.blocks[block][offset]
            row["reach_labels"], row["home_labels"] = -1, -1
            if not row["reach_valid"]:
                continue
            env = int(row["env"])
            cells, home = future_labels(
                row["reach_facts"],
                self._trace[env],
                int(row["tick"]),
                snapshot_only=self.snapshot_only,
                terminal=self._final_header[env, 1] != 0,
                final_facts=self._final_facts[env],
            )
            row["reach_labels"], row["home_labels"] = cells, home
            row["reach_provenance"] = 1 if self.snapshot_only else 2
            counts["snapshot_cells" if self.snapshot_only else "observed_cells"] += int(
                (cells >= 0).sum()
            )
            counts["unknown_cells"] += int((cells < 0).sum())
            counts["home_targets"] += int((home >= 0).sum())
            counts["unknown_homes"] += int((home < 0).sum())
            if len(examples) < 4 and not any(
                item["env"] == env and item["labels"] == cells.tolist() for item in examples
            ):
                examples.append(
                    dict(
                        row=index,
                        env=env,
                        tick=int(row["tick"]),
                        labels=cells.tolist(),
                        home_labels=home.tolist(),
                        prediction=row["reach_prediction"].astype(float).tolist(),
                        home_prediction=row["home_prediction"].astype(float).tolist(),
                        provenance="snapshot constraints"
                        if self.snapshot_only
                        else "factual future / safe constraints",
                    )
                )
        # Commit one bounded block together; retry overwrites identical labels.
        for key, value in counts.items():
            self.metrics[key] += value
        self.metrics["examples"] = examples
        self.cursor = last
        self.metrics["tracking_seconds"] += perf_counter() - start
        self.metrics["storage_bytes"] = self.estimated_bytes
        if self.cursor == self.buffer.size:
            self.phase = ReachPhase.LABELS_READY
        return self.phase == ReachPhase.LABELS_READY

    def snapshot(self):
        return dict(
            protocol=REACH_PROTOCOL,
            label_policy=LABEL_POLICY,
            phase=self.phase.value,
            cursor=self.cursor,
            metrics=dict(self.metrics),
            trace=torch.from_numpy(self.env.batch.reach_trace.get()),
            errors=torch.from_numpy(self.env.batch.reach_error.get()),
            started=torch.from_numpy(self.env.cp.asnumpy(self._started)),
            **{
                "prepared_" + name: None
                if getattr(self, "_" + name) is None
                else torch.from_numpy(getattr(self, "_" + name).copy())
                for name in ("trace", "final_header", "final_facts")
            },
        )

    def close(self):
        self._trace = self._final_header = self._final_facts = None
        self.env.batch.disable_reach()
