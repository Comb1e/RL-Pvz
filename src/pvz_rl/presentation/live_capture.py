"""Read-only CUDA staging for the live viewer, separate from policy observations."""

from time import perf_counter

import numpy as np
import torch
from pvz_game import Dig
from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES
from pvz_game.cuda import schema as s
from pvz_game.types import (
    API_VERSION,
    CardView,
    MowerView,
    Observation,
    PlantView,
    ProjectileView,
    Status,
    ZombieCounts,
    ZombieView,
)

from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.presentation.live_view import Activity, DecisionLayout, LiveSession, PanelState


def public_observation(arrays, rules, level):
    """Match Game.observe using present-state arrays only (never templates/seeds)."""
    h = dict(zip(s.HEADER, map(int, arrays["header"]), strict=True))
    tick, g = h["tick"], rules.game
    plants, zombies = [], []
    for values in arrays["plants"][: h["np"]]:
        p = dict(zip(s.PLANT, map(int, values), strict=True))
        kind = PLANT_TYPES[p["kind"]]
        plants.append(
            PlantView(
                p["id"],
                kind,
                p["row"],
                p["col"],
                p["health"],
                rules.plants[kind]["health"],
                s.PLANT_STATES[p["state"]],
                max(0, p["due"] - tick),
            )
        )
    for values in arrays["zombies"][: h["nz"]]:
        z = dict(zip(s.ZOMBIE, map(int, values), strict=True))
        state = s.ZOMBIE_STATES[z["state"]]
        timer = 0
        if state == "vaulting":
            timer = max(0, z["vault_until"] - tick)
        elif state == "biting":
            slow = tick < z["slow_until"]
            timer = (max(0, g["bite_ticks"] * 2 - z["bite_progress"]) + (0 if slow else 1)) // (
                1 if slow else 2
            )
        zombies.append(
            ZombieView(
                z["id"],
                ZOMBIE_TYPES[z["kind"]],
                z["row"],
                z["x"],
                z["health"],
                z["armor"],
                state,
                max(0, z["slow_until"] - tick),
                bool(z["has_pole"]),
                timer,
                bool(z["headless"]),
            )
        )
    return Observation(
        API_VERSION,
        tick,
        tick / g["tick_rate"],
        g["tick_rate"],
        g["units_per_tile"],
        g["rows"],
        g["cols"],
        level,
        Status(s.STATUSES[h["status"]]),
        h["sun"],
        tuple(
            CardView(k, rules.plants[k]["cost"], int(c), rules.plants[k]["recharge_ticks"])
            for k, c in zip(PLANT_TYPES, arrays["cooldowns"], strict=True)
        ),
        tuple(plants),
        tuple(zombies),
        tuple(
            ProjectileView(int(p[0]), int(p[1]), int(p[2]), int(p[3]), bool(p[4]))
            for p in arrays["projectiles"][: h["nq"]]
        ),
        tuple(MowerView(int(m[0]), int(m[1]), s.MOWER_STATES[int(m[2])]) for m in arrays["mowers"]),
        h["wave"],
        h["total_waves"],
        ZombieCounts(
            h["total_spawns"],
            h["spawn_index"],
            h["spawn_index"] - h["defeated"],
            h["defeated"],
            h["total_spawns"] - h["spawn_index"],
            h["total_spawns"] - h["defeated"],
        ),
    )


class CudaLiveCapture:
    """Collector-owned double buffer. CPU publication waits for copy completion."""

    def __init__(self, env, settings, *, session=None, notify=None):
        self.env = env
        self.session = session or LiveSession(env.num_envs, settings, notify=notify)
        self.codec = ActionCodec(env.cfg)
        self.source = {k: torch.from_dlpack(v) for k, v in env.batch.public_state_device().items()}
        self.shapes = {k: v.shape[1:] for k, v in self.source.items()}
        self.widths = {k: int(np.prod(shape)) for k, shape in self.shapes.items()}
        self.width = sum(self.widths.values())
        self.capacity = len(self.session.selection.panels)
        self.slots = [
            dict(
                device=torch.empty(
                    (self.capacity, self.width), dtype=torch.int64, device=env.header_tensor.device
                ),
                host=torch.empty((self.capacity, self.width), dtype=torch.int64, pin_memory=True),
                event=torch.cuda.Event(),
                pending=None,
            )
            for _ in range(2)
        ]
        self.indices = None
        self.sequence = 0
        self.last_capture = -float("inf")
        self.last_decision = -float("inf")
        self.decisions = {}
        self.decision_pending = None
        self.decision_host = None
        self.prepared = False
        self.role = None
        self.stats = {"captures": 0, "capture_seconds": 0.0, "skipped": 0}

    @property
    def enabled(self):
        return self.session.enabled

    def fail(self, exc):
        self.session.fail(exc)

    def set_activity(self, activity):
        self.session.set_activity(activity)
        if hasattr(self.session, "set_role"):
            self.session.set_role(self.role)

    def prepare(self):
        changed = self.session.selection.refresh(np.flatnonzero(self.env.enabled_envs))
        changed = self.session.poll() or changed
        if not self.enabled:
            return
        self.set_activity(
            Activity.ACTOR_COLLECTING
            if self.role == "actor"
            else Activity.CRITIC_COLLECTING
            if self.role == "critic"
            else Activity.COLLECTING
        )
        self.drain()
        if self.indices is None or changed:
            self.indices = torch.tensor(
                [p.env for p in self.session.selection.panels], device=self.env.header_tensor.device
            )
            self.last_capture = -float("inf")
            self.session.publish(force=True)
        self.prepared = True

    def decision_indices(self):
        self.prepare()
        if not self.enabled or perf_counter() - self.last_decision < 1 / self.session.fps:
            return None
        return self.indices

    def capture_decision(self, diagnostic, ticks):
        packed = torch.cat(
            (
                diagnostic["q"],
                diagnostic["plants"],
                diagnostic["tiles"].flatten(1),
                diagnostic["actions"][:, None],
                ticks[self.indices, None],
            ),
            -1,
        )
        if self.decision_host is None:
            self.decision_host = torch.empty_like(packed, device="cpu", pin_memory=True)
        self.decision_host.copy_(packed, non_blocking=True)
        self.decision_pending = [
            (i, p.generation, self.env._episode_serial[p.env])
            for i, p in enumerate(self.session.selection.panels)
        ]
        self.last_decision = perf_counter()

    def diagnostics(self):
        h = self.env.header_tensor
        return torch.stack(
            (
                self.env.executed_actions.index_select(0, self.indices),
                h[:, 0].index_select(0, self.indices),
            ),
            dim=1,
        ).flatten()

    def after_step(self, compact, actions):
        started = perf_counter()
        if self.decision_pending is not None:
            for row, (i, generation, episode) in zip(
                self.decision_host.numpy(), self.decision_pending
            ):
                self.decisions[i] = dict(
                    generation=generation,
                    episode=episode,
                    q=[float(x) if np.isfinite(x) else None for x in row[:3]],
                    plants=row[DecisionLayout.species].tolist(),
                    tiles=row[DecisionLayout.tiles].reshape(A.plant_types, A.tiles).tolist(),
                    action=int(row[DecisionLayout.action]),
                    tick=int(row[DecisionLayout.tick]),
                )
            self.decision_pending = None
        panels = self.session.selection.panels
        terminal = []
        for i, p in enumerate(panels):
            if p.state == PanelState.RESULT or i in self.session.selection.pending:
                continue
            action, tick = map(int, actions[i])
            if action and self.env.enabled_envs[p.env]:
                command = self.codec.decode(action)
                label = (
                    "Dig"
                    if isinstance(command, Dig)
                    else f"Plant {command.plant_type.replace('_', ' ')}"
                )
                tile = f"{chr(65 + command.row)}{command.col + 1}"
                p.actions.append(
                    f"{tick / self.env.batch.rules.game['tick_rate']:.2f}s: {label} at {tile}"
                )
            if compact[p.env, 0] or not self.env.enabled_envs[p.env]:
                terminal.append(i)
        now = perf_counter()
        if not terminal and now - self.last_capture < 1 / self.session.fps:
            return
        watched = [
            i
            for i, p in enumerate(panels)
            if p.state != PanelState.RESULT and i not in self.session.selection.pending
        ]
        if not watched:
            self.session.publish()
            return
        self.drain()
        slot = next((slot for slot in self.slots if slot["pending"] is None), None)
        if slot is None and terminal:
            # Terminal states cannot be reconstructed after autoreset. Normally
            # the existing compact host transfer has already completed both copies.
            self.slots[0]["event"].synchronize()
            self.drain()
            slot = self.slots[0]
        if slot is None:
            self.stats["skipped"] += 1
            return
        offset = 0
        for key, tensor in self.source.items():
            width = self.widths[key]
            torch.index_select(
                tensor,
                0,
                self.indices,
                out=slot["device"][:, offset : offset + width].view(
                    self.capacity, *self.shapes[key]
                ),
            )
            offset += width
        slot["host"].copy_(slot["device"], non_blocking=True)
        slot["event"].record(self.env.stream)
        self.sequence += 1
        pending = []
        for i in watched:
            p = panels[i]
            outcome = "truncated" if compact[p.env, 1] else self.env.finished_outcomes.get(p.env)
            pending.append(
                (
                    i,
                    p.generation,
                    dict(
                        level=self.env._level_names[p.env],
                        task=self.env._tasks[p.env],
                        episode=self.env._episode_serial[p.env],
                        actions=list(p.actions),
                        sequence=self.sequence,
                        outcome=outcome,
                        decision=self.decisions.get(i)
                        if self.decisions.get(i, {}).get("generation") == p.generation
                        and self.decisions[i]["episode"] == self.env._episode_serial[p.env]
                        else None,
                    ),
                )
            )
            if i in terminal:
                p.state = PanelState.RESULT
        slot["pending"] = pending
        self.last_capture = now
        self.stats["captures"] += 1
        self.stats["capture_seconds"] += perf_counter() - started

    def drain(self, *, wait=False):
        for slot in self.slots:
            if slot["pending"] is None:
                continue
            if wait:
                slot["event"].synchronize()
            if not slot["event"].query():
                continue
            raw = slot["host"].numpy()
            for i, generation, frame in slot["pending"]:
                arrays, offset = {}, 0
                for key, width in self.widths.items():
                    arrays[key] = raw[i, offset : offset + width].reshape(self.shapes[key])
                    offset += width
                obs = public_observation(arrays, self.env.batch.rules, frame.pop("level"))
                frame["observation"] = obs
                frame["outcome"] = frame["outcome"] or obs.status.value
                panel = self.session.selection.panels[i]
                # A previous ordinary capture must not undo a pending terminal state.
                pending_terminal = panel.state == PanelState.RESULT
                if self.session.selection.accept(i, generation, frame) and pending_terminal:
                    panel.state = PanelState.RESULT
            slot["pending"] = None
        self.session.publish()

    def close(self):
        self.drain(wait=True)
        self.session.close()
