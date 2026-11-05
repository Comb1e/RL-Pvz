"""Read-only CUDA staging for the live viewer, separate from policy observations."""

from time import perf_counter

import numpy as np
import torch
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

from pvz_rl.envs.cuda_features import METRIC_INDICES
from pvz_rl.envs.rewards import REWARD_METRICS
from pvz_rl.envs.tile_scenarios import is_placement_skill
from pvz_rl.learning.objective import COMPONENTS, components, training_rewards
from pvz_rl.presentation.live_view import Activity, LiveSession, PanelState


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
        self.session = session or LiveSession(env.num_envs, settings, cfg=env.cfg, notify=notify)
        self.session.journal = env.action_journal
        self.source = {k: torch.from_dlpack(v) for k, v in env.batch.public_state_device().items()}
        self.reward_totals = torch.from_dlpack(env.features.totals)
        self.reward_columns = [METRIC_INDICES[REWARD_METRICS.index(k)] for k in COMPONENTS]
        self.shapes = {k: v.shape[1:] for k, v in self.source.items()}
        self.widths = {k: int(np.prod(shape)) for k, shape in self.shapes.items()}
        self.width = sum(self.widths.values())
        self.capacity = len(self.session.selection.panels)
        self.slots = self._allocate_slots()
        self.indices = None
        self.sequence = 0
        self.last_capture = -float("inf")
        self.prepared = False
        self.stats = {"captures": 0, "capture_seconds": 0.0, "skipped": 0}

    def _allocate_slots(self):
        return [
            dict(
                device=torch.empty(
                    (self.capacity, self.width),
                    dtype=torch.int64,
                    device=self.env.header_tensor.device,
                ),
                host=torch.empty((self.capacity, self.width), dtype=torch.int64, pin_memory=True),
                reward_device=torch.empty(
                    (self.capacity, self.reward_totals.shape[1]),
                    dtype=torch.float64,
                    device=self.env.header_tensor.device,
                ),
                reward_host=torch.empty(
                    (self.capacity, self.reward_totals.shape[1]),
                    dtype=torch.float64,
                    pin_memory=True,
                ),
                event=torch.cuda.Event(),
                pending=None,
            )
            for _ in range(2)
        ]

    def _resize(self, capacity, shapes):
        if capacity == self.capacity and shapes == self.shapes:
            return
        self.drain(wait=True)
        # Drop old owners before allocating replacements. Placement snapshots use
        # actual live counts, not every main environment's worst-case shot slab.
        self.slots = []
        self.capacity, self.shapes = capacity, shapes
        self.widths = {k: int(np.prod(shape)) for k, shape in shapes.items()}
        self.width = sum(self.widths.values())
        self.slots = self._allocate_slots()

    @property
    def enabled(self):
        return self.session.enabled

    def fail(self, exc):
        self.session.fail(exc)

    def set_activity(self, activity):
        self.session.set_activity(activity)

    def poll(self):
        """Serve cached results and switches during collection and fitting."""
        self.drain()
        changed = self.session.selection.refresh(np.flatnonzero(self.env.enabled_envs))
        changed = self.session.poll() or changed
        if changed:
            self.indices = None
            self.session.publish(force=True)

    def prepare(self):
        self.poll()
        if not self.enabled:
            return
        self.set_activity(Activity.COLLECTING)
        # One-shot exercises finish every worker before a switch can be polled.
        # Copy their public results once for CPU-only browsing. Size staging
        # after the step, from headers already transferred for episode outcomes.
        capacity = (
            self.env.num_envs
            if is_placement_skill(self.env.tile_scenario)
            else len(self.session.selection.panels)
        )
        if not is_placement_skill(self.env.tile_scenario):
            self._resize(capacity, {k: v.shape[1:] for k, v in self.source.items()})
        if self.indices is not None and self.indices.numel() != capacity:
            self.indices = None
        if self.indices is None:
            self.indices = torch.tensor(
                list(range(capacity))
                if is_placement_skill(self.env.tile_scenario)
                else [p.env for p in self.session.selection.panels],
                device=self.env.header_tensor.device,
            )
            self.last_capture = -float("inf")
            self.session.publish(force=True)
        self.prepared = True

    def after_step(self, compact, *, completed_headers=None):
        started = perf_counter()
        panels = self.session.selection.panels
        cache_completed = is_placement_skill(self.env.tile_scenario)
        terminal = []
        for i, p in enumerate(panels):
            if p.state == PanelState.RESULT or i in self.session.selection.pending:
                continue
            if compact[p.env, 0] or not self.env.enabled_envs[p.env]:
                terminal.append(i)
        now = perf_counter()
        if not cache_completed and not terminal and now - self.last_capture < 1 / self.session.fps:
            return
        watched = [
            i
            for i, p in enumerate(panels)
            if p.state != PanelState.RESULT and i not in self.session.selection.pending
        ]
        captured = np.flatnonzero(self.env._step_active).tolist() if cache_completed else watched
        if not captured:
            self.session.publish()
            return
        if cache_completed:
            if completed_headers is None:
                raise RuntimeError("Placement viewer requires completed public headers")
            indices, headers = completed_headers
            if list(indices) != captured:
                raise RuntimeError("Placement viewer headers disagree with completed exercises")
            shapes = {k: v.shape[1:] for k, v in self.source.items()}
            for name, count in (("plants", "np"), ("zombies", "nz"), ("projectiles", "nq")):
                live = int(headers[:, s.HEADER.index(count)].max())
                if not 0 <= live <= shapes[name][0]:
                    raise RuntimeError("Placement viewer live count exceeds physical storage")
                shapes[name] = (live, shapes[name][1])
            self._resize(self.env.num_envs, shapes)
        self.drain()
        slot = next((slot for slot in self.slots if slot["pending"] is None), None)
        if slot is None and (terminal or cache_completed):
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
            if not width:
                continue
            if tensor.ndim == 3:
                tensor = tensor[:, : self.shapes[key][0]]
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
        torch.index_select(self.reward_totals, 0, self.indices, out=slot["reward_device"])
        slot["reward_host"].copy_(slot["reward_device"], non_blocking=True)
        slot["event"].record(self.env.stream)
        self.sequence += 1
        pending = []
        for row in captured:
            env = row if cache_completed else panels[row].env
            subscribers = [(i, p.generation) for i, p in enumerate(panels) if p.env == env]
            outcome = "truncated" if compact[env, 1] else self.env.finished_outcomes.get(env)
            pending.append(
                (
                    row,
                    subscribers,
                    cache_completed,
                    dict(
                        env=env,
                        level=self.env._level_names[env],
                        task=self.env._tasks[env],
                        episode=self.env._episode_serial[env],
                        sequence=self.sequence,
                        outcome=outcome,
                    ),
                )
            )
            for i, _ in subscribers:
                if compact[env, 0] or not self.env.enabled_envs[env]:
                    panels[i].state = PanelState.RESULT
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
            reward_totals = slot["reward_host"].numpy()
            for row, subscribers, cache_completed, frame in slot["pending"]:
                arrays, offset = {}, 0
                for key, width in self.widths.items():
                    arrays[key] = raw[row, offset : offset + width].reshape(self.shapes[key])
                    offset += width
                obs = public_observation(arrays, self.env.batch.rules, frame.pop("level"))
                frame["observation"] = obs
                frame["reward"] = float(
                    training_rewards(reward_totals[row, self.reward_columns], self.env.cfg)
                )
                frame["outcome"] = frame["outcome"] or obs.status.value
                frame["reward_pending"] = frame["outcome"] != "running"
                history = self.env.action_journal.page(frame["env"], frame["episode"])
                frame["history"] = history
                frame["decision"] = history["latest"] if history else None
                frame["outcome_decision"] = history.get("latest_outcome") if history else None

                if cache_completed:
                    self.session.selection.remember_completed(frame)
                for i, generation in subscribers:
                    panel = self.session.selection.panels[i]
                    # An ordinary capture must not undo a pending terminal state.
                    pending_terminal = panel.state == PanelState.RESULT
                    if self.session.selection.accept(i, generation, frame) and pending_terminal:
                        panel.state = PanelState.RESULT
            slot["pending"] = None
        self.session.publish()

    def reset_episodes(self, indices):
        self.session.selection.reset_episodes(
            ((int(env), int(self.env._episode_serial[env])) for env in indices),
            active=set(np.flatnonzero(self.env.enabled_envs)),
        )
        self.indices = None

    def finalize_rewards(self, pending):
        """At the cohort boundary, publish authoritative host episode totals."""
        self.drain(wait=True)
        episodes = [
            (
                int(env),
                int(item["episode"]),
                float(training_rewards(components(item["metrics"]), self.env.cfg)),
            )
            for env, item in pending.items()
        ]
        self.session.finalize_rewards(episodes)

    def close(self):
        self.drain(wait=True)
        self.session.close()
