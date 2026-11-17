"""One exact pre-action field/probe target per placement decision.

Prediction records forecasts until the lawn is quiet, then runs one probe.
"""

import copy
from time import perf_counter

import numpy as np
from pvz_game.cuda.schema import HEADER, STATUSES

from pvz_rl.envs.prediction_stop import PREDICTION_QUIET
from pvz_rl.learning.auxiliary_service import root_bytes
from pvz_rl.learning.prediction_labels import (
    accumulate_static,
    freeze_static,
    prediction_label_action,
    write_episode_target,
)

LOST = STATUSES.index("lost")


class PlacementOutcomes:
    protocol = "single-placement-outcomes-v3"

    def __init__(self, env, buffer, state=None):
        self.env, self.buffer, self.auxiliary = env, buffer, env.auxiliary
        if self.auxiliary is None:
            raise RuntimeError("Single-frame outcomes require the bounded simulation service")
        self.family = "prediction" if env.tile_scenario == "prediction" else "placement"
        self.jobs_metric = f"{self.family}_jobs"
        self.jobs = {}
        self.opening_seen = set()
        self.episodes = {}
        self.published = False
        self.metrics = dict(
            published_rows=0,
            field_ticks=0,
            probe_ticks=0,
            labeling_seconds=0.0,
            live_jobs=0,
        )
        self.metrics[self.jobs_metric] = 0
        if self.family == "prediction" and not env.batch.field_enabled:
            env.batch.enable_field_tracking()
        if state is not None:
            if state.get("protocol") != self.protocol:
                raise ValueError("Incompatible single-placement outcome recovery")
            if self.family == "prediction" and (
                "episodes" not in state or state.get("quiet") != PREDICTION_QUIET
            ):
                raise ValueError("Prediction outcome recovery predates the idle-plant stop")
            self.jobs = {tuple(key): value for key, value in state["jobs"]}
            self.opening_seen = {int(index) for index in state.get("opening_seen", ())}
            for job in self.jobs.values():
                if job.get("opening"):
                    self.opening_seen.add(int(job["env"]))
            for item in state.get("episodes", ()):
                episode = dict(item)
                episode["depth"] = np.array(episode["depth"], dtype=np.int64)
                episode["home"] = np.array(episode["home"], dtype=np.int64)
                self.episodes[int(episode["env"])] = episode
            self.published = state["published"]
            self.metrics = dict(state["metrics"])

    @property
    def estimated_bytes(self):
        rows = sum(len(episode["rows"]) for episode in self.episodes.values())
        return 512 * (len(self.jobs) + rows) + sum(
            root_bytes(job["root"]) for job in self.jobs.values() if "root" in job
        )

    def arm(self, request):
        # The ticket already owns its pre-mutation root; no real Wait capture.
        pass

    def _episode(self, env_index):
        episode = self.episodes.get(env_index)
        if episode is None:
            episode = dict(
                env=env_index,
                depth=np.zeros(5, np.int64),
                home=np.full(5, -1, np.int64),
                origin=None,
                rows=[],
                closed=False,
                field=None,
                delay=None,
            )
            self.episodes[env_index] = episode
        return episode

    def _observe_prediction(self, rows, receipt):
        timed_out = self.env.last_transition_host[:, 1].astype(bool)
        status = self.env._physical_header[:, HEADER.index("status")]
        quiet = []
        for slot, index in zip(receipt.slots, receipt.rows, strict=True):
            env_index = int(rows["env"][slot])
            episode = self._episode(env_index)
            if episode["closed"] or episode["field"] is not None:
                raise RuntimeError("A finished prediction episode recorded another forecast")
            if episode["origin"] is None:
                episode["origin"] = int(rows["tick"][slot])
            accumulate_static(
                episode["depth"],
                episode["home"],
                rows["field_trace"][slot],
                rows["field_breach"][slot],
            )
            episode["rows"].append(int(index))
            action = prediction_label_action(
                bool(rows["done"][slot]),
                bool(timed_out[slot]),
                int(status[slot]) == LOST,
                bool(np.any(episode["home"] >= 0)),
            )
            if action == "drop":
                episode["closed"] = True
            elif action == "probe":
                field, delay = freeze_static(episode["depth"], episode["home"], episode["origin"])
                episode["field"] = field
                episode["delay"] = delay
                quiet.append(env_index)
        if not quiet:
            return
        roots = self.auxiliary.capture(
            self.env.batch, quiet, self.env._physical_header[quiet]
        )
        for env_index, root in zip(quiet, roots, strict=True):
            episode = self.episodes[env_index]
            identity = ("outcomes", self.family, env_index)
            self.jobs[identity] = dict(
                env=env_index,
                rows=list(episode["rows"]),
                field=episode["field"],
                delay=episode["delay"],
                root=root,
                submitted=False,
            )
            self.metrics[self.jobs_metric] += 1

    def observe(self, request, rows, receipt):
        if self.family == "prediction":
            self._observe_prediction(rows, receipt)
            return
        for slot, index in zip(receipt.slots, receipt.rows, strict=True):
            identity = ("outcomes", self.family, int(index))
            action = int(request["actions"][slot, 0])
            env_index = int(rows["env"][slot])
            if request["valid"][slot].sum() != 1:
                raise RuntimeError("A single-frame exercise must own exactly one proposal")
            if int(rows["executed_action"][slot]) != action:
                raise RuntimeError("Saved proposal legality differs from execution")
            opening = env_index not in self.opening_seen
            self.opening_seen.add(env_index)
            self.jobs[identity] = dict(
                row=int(index),
                env=env_index,
                opening=opening,
                action=action,
                root=request["pre_states"][int(slot)],
                submitted=False,
            )
            self.metrics[self.jobs_metric] += 1

    def _submit_pending(self):
        for identity, job in self.jobs.items():
            if not job["submitted"]:
                # Retain ownership until queue admission succeeds. Resource failure
                # leaves an exact root in recovery, even after the real action.
                self.auxiliary.submit(identity, job["root"], action=0)
                job["submitted"] = True
                job.pop("root")

    def _publish_probe(self, job, result):
        probe = np.asarray(result["outcome"])[:, 2]
        if probe.shape != (5,) or np.any((probe < 0) | (probe > 10)):
            raise RuntimeError("Prediction probe depths are not five lane classes")
        write_episode_target(self.buffer, job["rows"], job["field"], job["delay"], probe)
        self.metrics["published_rows"] += len(job["rows"])

    def advance(self):
        started = perf_counter()
        self._submit_pending()
        for identity, result in self.auxiliary.poll("outcomes"):
            job = self.jobs[identity]
            if "rows" in job:
                self._publish_probe(job, result)
            else:
                index = job["row"]
                block, offset = divmod(index, self.buffer.block_rows)
                row = self.buffer.blocks[block][offset]
                row["baseline_outcomes"] = result["outcome"]
                row["null_target_valid"] = True
                row["outcome_ready"] = True
                self.metrics["published_rows"] += 1
            self.auxiliary.acknowledge(identity)
            del self.jobs[identity]
        for key in ("field_ticks", "probe_ticks"):
            self.metrics[key] = self.auxiliary.metrics.get(key, 0)
        self.metrics["live_jobs"] = len(self.jobs)
        self.metrics["labeling_seconds"] += perf_counter() - started
        return not self.jobs

    def finalize(self):
        if not self.published:
            self.auxiliary.seal("outcomes")
            if not self.advance():
                return False
            self.published = True
            self.auxiliary.release_completed_intervals()
        return True

    def snapshot(self):
        jobs = []
        for identity, job in self.jobs.items():
            saved = dict(job)
            if "root" in saved and hasattr(saved["root"], "resolve"):
                saved["root"] = saved["root"].resolve(wait=True)
            jobs.append((identity, saved))
        saved = dict(
            protocol=self.protocol,
            jobs=jobs,
            opening_seen=sorted(self.opening_seen),
            episodes=[self.episodes[index] for index in sorted(self.episodes)],
            published=self.published,
            metrics=self.metrics,
        )
        if self.family == "prediction":
            saved["quiet"] = PREDICTION_QUIET
        return copy.deepcopy(saved)

    def close(self):
        pass
