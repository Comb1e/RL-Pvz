"""Resident exact simulation workers. Importable without Torch or CUDA."""

import traceback

import numpy as np
from pvz_game import Rules, Status
from pvz_game.config import LevelSpec

from pvz_rl.envs.outcome_math import OUTCOME_SHAPE
from pvz_rl.envs.scratch_game import restored_game
from pvz_rl.envs.static_fork import StaticFork, apply_action

_RULES = {}


def validate_continuation(job, state, rules):
    private_game(state["snapshot"], rules)
    if state.get("action_applied") is not True:
        raise ValueError("A private continuation must retain its applied action")
    if any(
        key not in ("field_ticks", "probe_ticks", "teacher_ticks")
        or type(value) is not int
        or value < 0
        for key, value in state["ticks"].items()
    ):
        raise ValueError("Invalid private simulation tick counters")
    if job["family"] == "outcomes":
        labels = np.asarray(state["labels"])
        if (
            labels.shape != OUTCOME_SHAPE
            or labels.dtype != np.int64
            or (labels[:, [0, 2]] < 0).any()
            or (labels[:, [0, 2]] > 10).any()
            or (labels[:, 1] < -1).any()
            or not np.array_equal(labels[:, 0] == 10, labels[:, 1] >= 0)
        ):
            raise ValueError("Invalid absolute outcome continuation")
        if (
            state["phase"] not in ("field", "probe")
            or not 0 <= state["start"] <= state["snapshot"]["tick"]
            or (state["phase"] == "field" and state["anchor"] != -1)
            or (
                state["phase"] == "probe"
                and not state["start"] <= state["anchor"] <= state["snapshot"]["tick"]
            )
        ):
            raise ValueError("Invalid field/probe continuation lifecycle")


def private_game(snapshot, rules=None):
    """Validate physical records, then restore private fork accounting verbatim.

    Forks deliberately remove breached lanes and install unscheduled probes.
    Those counters are not valid public-game spawn history. A synthetic past is
    used only for the engine validator; none of it enters the continuation.
    """
    if rules is None:
        digest = snapshot["rules_hash"]
        if digest not in _RULES:
            candidate = Rules(snapshot["rules"])
            if candidate.digest != digest:
                raise ValueError("Private continuation rules hash changed")
            if len(_RULES) >= 16:
                _RULES.pop(next(iter(_RULES)))
            _RULES[digest] = candidate
        rules = _RULES[digest]
    normalized = dict(snapshot)
    zombies = snapshot["zombies"]
    normalized.update(
        level=dict(
            name="private-continuation",
            initial_sun=snapshot["sun"],
            plants=[],
            mowers=False,
            spawns=[
                dict(
                    tick=snapshot["tick"],
                    zombie_type=z["kind"],
                    row=z["row"],
                    x=max(0, z["x"]),
                    wave=1,
                )
                for z in zombies
            ],
        ),
        spawn_index=len(zombies),
        defeated=sum(z["headless"] for z in zombies),
        wave=int(bool(zombies)),
        status="lost"
        if any(not z["headless"] and z["x"] <= rules.game["house_x"] for z in zombies)
        else "running",
    )
    game = restored_game(normalized, rules)
    game._level = LevelSpec.from_dict(snapshot["level"])
    game._total_waves = max((s.wave for s in game._level.spawns), default=0)
    for key in ("spawn_index", "defeated", "wave"):
        setattr(game, "_" + key, snapshot[key])
    game._status = Status(snapshot["status"])
    return game


class CpuSimulation:
    def __init__(self, job):
        self.job = job
        state = job.get("continuation")
        game = private_game(state["snapshot"] if state else job["snapshot"])
        self.ticks = dict(field_ticks=0, probe_ticks=0, teacher_ticks=0)
        if job["family"] == "outcomes":
            # Constructor hooks remain bound to this owned fork after recovery.
            self.fork = StaticFork(
                game,
                action=0 if state else job["action"],
                probe=job["probe"],
                sky_sun_rate=job["sky_sun_rate"],
            )
            if state:
                self.fork.phase = state["phase"]
                self.fork.start, self.fork.anchor_tick = state["start"], state["anchor"]
                self.fork.labels[:] = state["labels"]
                self.fork.field_ticks = state["ticks"].get("field_ticks", 0)
                self.fork.probe_ticks = state["ticks"].get("probe_ticks", 0)
            self.game = self.fork.game
        else:
            self.fork = None
            self.game = game
            self.game._emit = lambda *args, **kwargs: None
            self.game._events = []
            if not state and job["action"] and not apply_action(game, job["action"]):
                raise RuntimeError("CPU teacher rejected a saved legal candidate")
            self.ticks.update(state["ticks"] if state else {})

    def advance(self, ticks=256):
        if self.fork is not None:
            done = self.fork.advance(ticks)
            self.ticks.update(field_ticks=self.fork.field_ticks, probe_ticks=self.fork.probe_ticks)
            return done
        game = self.game
        for _ in range(ticks):
            if game._status != Status.RUNNING or game._tick >= self.job["cutoff"]:
                return True
            drops = game._sky_drops
            game._advance()
            if game._sky_drops != drops and self.job["sky_sun_rate"] != 1:
                game._sky_due = game._tick + max(
                    1, int((game._sky_due - game._tick) / self.job["sky_sun_rate"])
                )
            self.ticks["teacher_ticks"] += 1
        return game._status != Status.RUNNING or game._tick >= self.job["cutoff"]

    def snapshot(self):
        state = dict(snapshot=self.game.snapshot(), ticks=dict(self.ticks), action_applied=True)
        if self.fork is not None:
            state.update(
                phase=self.fork.phase,
                start=self.fork.start,
                anchor=self.fork.anchor_tick,
                labels=self.fork.labels.copy(),
            )
        return state

    def result(self):
        if self.fork is not None:
            return dict(
                outcome=self.fork.labels.copy(),
                anchor=self.fork.anchor_tick,
                ticks=dict(self.ticks),
            )
        return dict(
            success=self.game._status == Status.WON,
            timeout=self.game._status == Status.RUNNING,
            tick=self.game._tick,
            ticks=dict(self.ticks),
        )


def worker(connection):
    """Keep hot states resident; snapshot only at checkpoint/migration boundaries."""
    simulation = identity = None
    connection.send(("ready", None, None))
    try:
        while True:
            if simulation is None or connection.poll():
                command, payload = connection.recv()
                if command == "close":
                    return
                if command == "pause":
                    connection.send(
                        ("paused", identity, simulation.snapshot() if simulation else None)
                    )
                    simulation = identity = None
                    continue
                if command != "run" or simulation is not None:
                    raise RuntimeError("Invalid CPU simulation worker transition")
                identity = payload["id"]
                simulation = CpuSimulation(payload)
            if simulation.advance(256):
                connection.send(("completed", identity, simulation.result()))
                simulation = identity = None
    except EOFError:
        return
    except BaseException:
        state = simulation.snapshot() if simulation is not None else None
        connection.send(
            ("failed", identity, dict(error=traceback.format_exc(), continuation=state))
        )
    finally:
        connection.close()
