"""Resumable hindsight threat windows over the factual episode, never a fork."""

from enum import StrEnum
from time import perf_counter

import numpy as np

from pvz_rl.envs.reach_tracking import FORECAST_HEADER
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_labels import FORECAST_PROTOCOL, time_bin

REACH_PROTOCOL = FORECAST_PROTOCOL


class ReachPhase(StrEnum):
    COLLECTING = "collecting"
    TRACES_COMPLETE = "traces_complete"
    LABELS_READY = "labels_ready"
    FITTING = "fitting"
    COMMITTED = "committed"


class ReachSupervision:
    def __init__(self, env, buffer, state=None, archive=None):
        self.env, self.buffer = env, buffer
        self.exercise = env.tile_scenario == "peashooter"
        self.phase, self.cursor = ReachPhase.COLLECTING, 0
        self._started = env.cp.zeros(env.batch.n, env.cp.bool_)
        self.tail = []
        self._timelines = {}
        self._contacts = {}
        self.metrics = dict(
            label_policy=FORECAST_PROTOCOL,
            eligible_decisions=0,
            tracking_seconds=0.0,
            storage_bytes=0,
            continuation_seconds=0.0,
            continuation_ticks=0,
            exact_depths=0,
            censored_depths=0,
            breach_targets=0,
            stop_reasons={},
            examples=[],
            defensive_placements=0,
            empty_lane_defenses=0,
            contact_lead_ticks=0,
            contact_observations=0,
        )
        if state is None:
            env.batch.enable_reach()
        else:
            if state["protocol"] != REACH_PROTOCOL:
                raise ValueError("Incompatible old reach supervision; start fresh")
            self.phase, self.cursor = ReachPhase(state["phase"]), state["cursor"]
            self._started[:] = env.cp.asarray(state["started"])
            self.tail = list(state["tail"])
            self.metrics.update(state["metrics"])
            env.batch.enable_reach(state["trace"])
            env.batch.reach_error[:] = env.cp.asarray(state["errors"])

    @property
    def estimated_bytes(self):
        return int(
            self.env.batch.reach_trace.nbytes
            + self.env.batch.reach_error.nbytes
            + self._started.nbytes
            + sum(x[2].nbytes + x[3].nbytes + x[4].nbytes for x in self.tail)
            + sum(v[0].nbytes + v[1].nbytes for v in self._timelines.values())
            + sum(sum(a.nbytes for a in lanes) for lanes in self._contacts.values())
        )

    def before(self, actions, active):
        self._started |= self.env.cp.from_dlpack(active)

    def capture(self, rows):
        self.metrics["eligible_decisions"] += int(rows["reach_valid"].sum())

    def continue_exercises(self):
        if not self.exercise:
            return True
        cp, batch = self.env.cp, self.env.batch
        header = batch.header.get()
        active = cp.asnumpy(self._started) & (header[:, 1] == 0)
        if not active.any():
            return True
        start = perf_counter()
        before = header[:, 0].copy()
        saved = batch.header[:, 17].copy()
        try:
            batch.header[:, 17] = cp.asarray(active)
            batch.step_device(
                cp.zeros(batch.n, cp.int64), ticks=batch.max_step_ticks, per_tick=False
            )
            header = batch.header.get()
            trace = batch.reach_trace.get()
            if bool(batch.reach_error.any()):
                raise RuntimeError("Private threat continuation tracker failed; recovery retained")
            for env in np.flatnonzero(active):
                count = int(header[env, 0] - before[env])
                values = (
                    trace[env, FORECAST_HEADER : FORECAST_HEADER + (count + 1) * 5]
                    .reshape(-1, 5)
                    .astype(np.uint8)
                )
                self.tail.append(
                    (
                        int(env),
                        int(before[env]),
                        values,
                        trace[env, 1:6].copy(),
                        trace[env, 6:11].copy(),
                    )
                )
            self.metrics["continuation_ticks"] += int((header[active, 0] - before[active]).sum())
            self.metrics["continuation_seconds"] += perf_counter() - start
        finally:
            batch.header[:, 17] = saved
        return not np.any(active & (header[:, 1] == 0))

    def _timeline(self, env, episode):
        key = env, episode
        if key in self._timelines:
            return self._timelines[key]
        pieces, homes, end = [], np.full(5, -1, np.int64), 0
        contacts = [[] for _ in range(5)]
        for block, length in zip(
            self.buffer.blocks, range(0, self.buffer.size, self.buffer.block_rows)
        ):
            rows = block[: min(self.buffer.block_rows, self.buffer.size - length)]
            for row in rows[(rows["env"] == env) & (rows["episode"] == episode)]:
                tick, count = int(row["tick"]), int(row["duration"])
                values = row["forecast_trace"][: count + 1]
                pieces.append((tick, values))
                seen = row["forecast_breach"] >= 0
                homes[seen & ((homes < 0) | (row["forecast_breach"] < homes))] = row[
                    "forecast_breach"
                ][seen & ((homes < 0) | (row["forecast_breach"] < homes))]
                end = max(end, tick + count)
                for lane, contact in enumerate(row["forecast_contacts"]):
                    if contact >= 0:
                        contacts[lane].append(int(contact))
        for slot, tick, values, entries, bites in self.tail:
            if slot == env:
                pieces.append((tick, values))
                end = max(end, tick + len(values) - 1)
                seen = entries >= 0
                take = seen & ((homes < 0) | (entries < homes))
                homes[take] = entries[take]
                for lane, contact in enumerate(bites):
                    if contact >= 0:
                        contacts[lane].append(int(contact))
        trace = np.zeros((end + 1, 5), np.uint8)
        for tick, values in pieces:
            np.maximum(
                trace[tick : tick + len(values)], values, out=trace[tick : tick + len(values)]
            )
        self._timelines[key] = trace, homes
        self._contacts[key] = [np.unique(lane) for lane in contacts]
        return trace, homes

    def prepare(self):
        start = perf_counter()
        if self.phase == ReachPhase.COLLECTING:
            if not self.continue_exercises():
                return False
            if bool(self.env.batch.reach_error.any()):
                raise RuntimeError("Threat tracker failed; recovery retained")
            self.phase = ReachPhase.TRACES_COMPLETE
        if self.phase != ReachPhase.TRACES_COMPLETE:
            return True
        tau = reference_tau(self.env.batch.rules)
        last = min(self.buffer.size, self.cursor + self.buffer.block_rows)
        headers = self.env.batch.header.get()
        for index in range(self.cursor, last):
            block, offset = divmod(index, self.buffer.block_rows)
            row = self.buffer.blocks[block][offset]
            row["threat_depth"], row["threat_time"], row["threat_loss"] = -1, -1, -1
            row["reach_labels"], row["home_labels"] = -1, -1
            if not row["reach_valid"]:
                continue
            env, tick = int(row["env"]), int(row["tick"])
            trace, homes = self._timeline(env, int(row["episode"]))
            action = int(row["action"])
            if row["accepted"] and 45 < action <= 360:
                lane = ((action - 1) % 45) // 9
                self.metrics["defensive_placements"] += 1
                self.metrics["empty_lane_defenses"] += int(row["lane_zombies"][lane] == 0)
                contacts = self._contacts[env, int(row["episode"])][lane]
                future = contacts[contacts >= tick]
                if len(future):
                    self.metrics["contact_lead_ticks"] += int(future[0] - tick)
                    self.metrics["contact_observations"] += 1
            end = len(trace) - 1
            won = headers[env, 1] == 1
            for horizon, stop in enumerate((tick + 2 * tau, tick + 9 * tau, end)):
                row["threat_depth"][:, horizon] = trace[tick : min(stop, end) + 1].max(0)
                censored = not won and (horizon == 2 or stop > end)
                row["threat_censored"][:, horizon] = censored
                self.metrics["censored_depths" if censored else "exact_depths"] += 5
            observed = homes >= tick
            row["threat_time"][observed] = time_bin(homes[observed] - tick, tau)
            row["threat_loss"][observed] = 1
            if won:
                row["threat_time"][~observed], row["threat_loss"][~observed] = 5, 0
            self.metrics["breach_targets"] += int(observed.sum())
            # Existing viewers keep a masked physical map. Censored depths mean
            # at-least reach: only observed positive penetration is colored.
            depths = row["threat_depth"][:, 2]
            for lane in range(5):
                if won:
                    row["reach_labels"][lane] = 0
                row["reach_labels"][lane, np.arange(9) >= 9 - min(9, int(depths[lane]))] = 1
            row["home_labels"] = row["threat_loss"]
            row["reach_provenance"] = 3
            if len(self.metrics["examples"]) < 3:
                self.metrics["examples"].append(
                    dict(
                        env=env,
                        tick=tick,
                        prediction=row["reach_prediction"].tolist(),
                        home_prediction=row["home_prediction"].tolist(),
                        labels=row["reach_labels"].tolist(),
                        home_labels=row["home_labels"].tolist(),
                        provenance="Realized penetration"
                        if won
                        else "Observed lower bound; grey = censored",
                    )
                )
        self.cursor = last
        self.metrics["tracking_seconds"] += perf_counter() - start
        self.metrics["storage_bytes"] = self.estimated_bytes
        if self.cursor == self.buffer.size:
            self.phase = ReachPhase.LABELS_READY
            self.metrics["stop_reasons"] = dict(
                natural_win=int((headers[:, 1] == 1).sum()),
                home_breach=int((headers[:, 1] == 2).sum()),
                censored=int(((headers[:, 1] == 0) & self.env.cp.asnumpy(self._started)).sum()),
            )
            n, d = self.metrics["contact_observations"], self.metrics["defensive_placements"]
            self.metrics["defense_lead_time_to_lane_bite_seconds"] = (
                self.metrics["contact_lead_ticks"] / n / self.env.batch.rules.game["tick_rate"]
                if n
                else None
            )
            self.metrics["defense_empty_lane_fraction"] = (
                self.metrics["empty_lane_defenses"] / d if d else None
            )
        return self.phase == ReachPhase.LABELS_READY

    def snapshot(self):
        return dict(
            protocol=REACH_PROTOCOL,
            phase=self.phase.value,
            cursor=self.cursor,
            metrics=dict(self.metrics),
            trace=self.env.batch.reach_trace.get(),
            errors=self.env.batch.reach_error.get(),
            started=self.env.cp.asnumpy(self._started),
            tail=list(self.tail),
        )

    def close(self):
        self.tail.clear()
        self._timelines.clear()
        self._contacts.clear()
        self.env.batch.disable_reach()
