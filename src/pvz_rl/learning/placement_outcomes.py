"""One exact pre-action field/probe target per single-frame exercise."""

import copy
from time import perf_counter

from pvz_rl.learning.auxiliary_service import root_bytes


class PlacementOutcomes:
    protocol = "single-placement-outcomes-v3"

    def __init__(self, env, buffer, state=None):
        self.env, self.buffer, self.auxiliary = env, buffer, env.auxiliary
        if self.auxiliary is None:
            raise RuntimeError("Single-frame outcomes require the bounded simulation service")
        self.family = "prediction" if env.tile_scenario == "prediction" else "placement"
        self.jobs_metric = f"{self.family}_jobs"
        self.jobs = {}
        self.published = False
        self.metrics = dict(
            published_rows=0,
            field_ticks=0,
            probe_ticks=0,
            labeling_seconds=0.0,
            live_jobs=0,
        )
        self.metrics[self.jobs_metric] = 0
        if state is not None:
            if state.get("protocol") != self.protocol:
                raise ValueError("Incompatible single-placement outcome recovery")
            self.jobs = {tuple(key): value for key, value in state["jobs"]}
            self.published = state["published"]
            self.metrics = dict(state["metrics"])

    @property
    def estimated_bytes(self):
        return 512 * len(self.jobs) + sum(
            root_bytes(job["root"]) for job in self.jobs.values() if "root" in job
        )

    def arm(self, request):
        # The ticket already owns its pre-mutation root; no real Wait capture.
        pass

    def observe(self, request, rows, receipt):
        for slot, index in zip(receipt.slots, receipt.rows, strict=True):
            identity = ("outcomes", self.family, int(index))
            action = int(request["actions"][slot, 0])
            if request["valid"][slot].sum() != 1:
                raise RuntimeError("A single-frame exercise must own exactly one proposal")
            if int(rows["executed_action"][slot]) != action:
                raise RuntimeError("Saved proposal legality differs from execution")
            self.jobs[identity] = dict(
                row=int(index),
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

    def advance(self):
        started = perf_counter()
        self._submit_pending()
        for identity, result in self.auxiliary.poll("outcomes"):
            index = self.jobs[identity]["row"]
            block, offset = divmod(index, self.buffer.block_rows)
            row = self.buffer.blocks[block][offset]
            row["baseline_outcomes"] = result["outcome"]
            row["null_target_valid"] = True
            row["outcome_ready"] = True
            self.auxiliary.acknowledge(identity)
            del self.jobs[identity]
            self.metrics["published_rows"] += 1
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
        return copy.deepcopy(
            dict(
                protocol=self.protocol,
                jobs=jobs,
                published=self.published,
                metrics=self.metrics,
            )
        )

    def close(self):
        pass
