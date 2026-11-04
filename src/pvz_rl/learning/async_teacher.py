"""The unchanged Peashooter teacher backed by the shared asynchronous service."""

import copy
import json

import numpy as np
from pvz_game.config import PLANT_TYPES
from pvz_game.cuda.schema import HEADER

from pvz_rl.learning.action_routing import RepeatedRowCompactor
from pvz_rl.learning.peashooter_rules import RULE_OBJECTIVE, RULE_PROTOCOL, rule_settings


class AsyncPeashooterTeacher:
    def __init__(self, env, buffer, identity, state=None, archive=None):
        self.env, self.buffer, self.cfg = env, buffer, buffer.cfg
        self.settings = copy.deepcopy(rule_settings(self.cfg))
        self.identity = tuple(identity)
        self.service = getattr(env, "auxiliary", None)
        self.compactor = RepeatedRowCompactor(
            self.cfg["training"]["objective"]["invalid_fit_group_size"]
        )
        self.roots, self.jobs, self.labels = [], [], {}
        self.root_lookup = {}
        self.capture = None
        self.capture_slots = []
        self.sealed = False
        self.metrics = dict(
            roots=0,
            labeled=0,
            rule_1=0,
            rule_2=0,
            rule_3=0,
            preferred_tiles=0,
            fallback_lanes=0,
            blocked_lanes=0,
            unlabeled=0,
            completed_rollouts=0,
            oracle_successes=0,
            oracle_timeouts=0,
            simulated_ticks=0,
            oracle_seconds=0.0,
            pending=0,
            spill_bytes=0,
        )
        if state is not None:
            if tuple(state["identity"]) != self.identity or state["settings"] != self.settings:
                raise ValueError("Peashooter teacher identity/settings differ from recovery")
            self.roots = copy.deepcopy(state["roots"])
            self.labels = copy.deepcopy(state["labels"])
            self.metrics.update(state["metrics"])
            self.compactor.__dict__.update(copy.deepcopy(state["compactor"]))
            self.sealed = state.get("sealed", False)
            if not state.get("async_execution"):
                self._import_legacy(state, archive)
            self.root_lookup = {tuple(root["identity"]): root for root in self.roots}

    @property
    def estimated_bytes(self):
        return len(self.roots) * 1024 + len(self.labels) * 512

    @property
    def settled(self):
        return self.sealed and len(self.labels) == len(self.roots)

    def before(self, actions, active, branch_available, request=None, *, issued=None):
        branch = self.cfg["environment"]["plants"].index("peashooter")
        kind = PLANT_TYPES.index("peashooter")
        h, cd = self.env._physical_header, self.env._physical_cooldowns
        available = (cd[:, kind] == 0) & (
            h[:, HEADER.index("sun")] >= self.env.batch.rules.plants["peashooter"]["cost"]
        )
        available &= (h[:, HEADER.index("allowed")] & (1 << kind)) != 0
        if request is None:
            # With forecasts disabled there is no effect ticket. Freeze the
            # available public lanes and filter the committed proposal receipt.
            selected = np.asarray(issued, dtype=bool) & available
        else:
            proposed = request["actions"][:, 0]
            selected = (
                request["valid"][:, 0]
                & available
                & (proposed >= 1 + branch * 45)
                & (proposed < 1 + (branch + 1) * 45)
            )
        self.capture_slots = np.flatnonzero(selected).tolist()
        self.capture = (
            self.service.capture(self.env.batch, self.capture_slots, h[self.capture_slots])
            if self.capture_slots
            else None
        )

    def observe(self, receipt):
        rows = self.buffer.take(receipt.rows)
        retained = self.compactor.select(rows)
        local = {slot: index for index, slot in enumerate(self.capture_slots)}
        for index in retained:
            slot = int(rows["env"][index])
            action = int(rows["action"][index])
            branch = self.cfg["environment"]["plants"].index("peashooter")
            if slot not in local or not 1 + branch * 45 <= action < 1 + (branch + 1) * 45:
                continue
            identity = tuple(int(rows[k][index]) for k in ("env", "episode", "decision"))
            root = dict(identity=identity, state="queued", pending=0)
            self.roots.append(root)
            self.root_lookup[identity] = root
            self.service.submit_board(("teacher", *identity), self.capture[local[slot]])
        self.capture = None
        self.metrics["roots"] = len(self.roots)

    def seal(self):
        if not self.sealed:
            self.sealed = True
            self.service.seal("teacher")

    def advance(self):
        for identity, label in self.service.poll("teacher"):
            key = tuple(identity[1:])
            if key not in self.labels:
                self.labels[key] = label
                self.metrics[f"rule_{label['rule']}"] += 1
                self.metrics["preferred_tiles"] += len(label["tiles"])
                self.metrics["fallback_lanes"] += len(label["fallback_rows"])
                self.metrics["blocked_lanes"] += label["blocked"]
                self.metrics["labeled" if label["tiles"] else "unlabeled"] += 1
            self.root_lookup[key]["state"] = "labeled"
            self.service.acknowledge(identity)
        metrics = self.service.metrics
        self.metrics.update(
            pending=len(self.roots) - len(self.labels),
            simulated_ticks=metrics["teacher_ticks"],
            spill_bytes=metrics["root_spill_bytes"],
            completed_rollouts=metrics["teacher_jobs"],
            oracle_successes=metrics["teacher_successes"],
            oracle_timeouts=metrics["teacher_timeouts"],
            oracle_seconds=metrics["teacher_cpu_seconds"] + metrics["teacher_cuda_seconds"],
            mean_preferred_set_size=self.metrics["preferred_tiles"] / max(1, len(self.labels)),
        )
        return self.settled

    def _import_legacy(self, state, archive):
        """Import exact old roots and already-applied partial candidate states."""

        def read(key):
            return json.loads(archive.read(f"peashooter/{key}.json"))

        for index, root in enumerate(state["roots"]):
            identity = ("teacher", *root["identity"])
            if root["state"] == "labeled":
                self.service.published.add(identity)
                continue
            record = {
                k: copy.deepcopy(v) for k, v in root.items() if k not in ("identity", "pending")
            }
            record["pending"] = 0
            if record["state"] == "queued":
                record["state"] = "capturing"
            self.service.submit_board(identity, read(f"root-{index}"), record=record)
            board = self.service.boards[identity]
            for job in state["jobs"]:
                if job["root"] != index or job["state"] == "completed":
                    continue
                snapshot = read(job.get("partial", job["key"]))
                continuation = (
                    dict(snapshot=snapshot, ticks={}, action_applied=True)
                    if job["placed"]
                    else None
                )
                self.service._board_job(
                    board, job["row"], job["tile"], snapshot=snapshot, continuation=continuation
                )
            if board["state"] != "capturing":
                self.service._resolve_board(board)

    def snapshot(self):
        if self.capture is not None:
            raise RuntimeError("Cannot checkpoint an uncommitted teacher capture")
        return copy.deepcopy(
            dict(
                protocol=RULE_PROTOCOL,
                objective=RULE_OBJECTIVE,
                settings=self.settings,
                identity=self.identity,
                roots=self.roots,
                jobs=[],
                labels=self.labels,
                initialized=self.sealed,
                next_root=len(self.roots),
                compactor=vars(self.compactor),
                metrics=self.metrics,
                async_execution=True,
                sealed=self.sealed,
            )
        )

    def write_archive(self, archive):
        pass  # Owned roots and continuations are in the service's semantic state.

    def release(self):
        pass  # The shared service is closed once both owners have published.

    def close(self):
        self.capture = None
