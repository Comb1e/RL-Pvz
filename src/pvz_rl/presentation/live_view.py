"""Bounded, CPU-only presentation of selected live training games.

Viewer code owns no model or simulator; it renders public snapshots. Selection
uses a private RNG. Commands change subscriptions, never game state. The
gradient-scale shares and the 3× restart flag are the training values the view can write.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import warnings
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from queue import Empty, Full
from time import monotonic

from pvz_game.config import PLANT_TYPES

from pvz_rl.config import outcome_prediction_weights
from pvz_rl.learning.curriculum import SELECTABLE_STAGES

PANELS = 4
RESULT_SECONDS = 1.0
UI_FPS = 30

DECISION_BRANCHES = ("wait", *PLANT_TYPES, "dig")


class PanelState(StrEnum):
    WATCHING = "watching"
    RESULT = "showing result"
    SELECTING = "selecting"


class Activity(IntEnum):
    STARTING = 0
    COLLECTING = 1
    UPDATING = 2
    VALIDATING = 3
    REPORTING = 4


ACTIVITY_TEXT = {
    Activity.STARTING: "Starting",
    Activity.COLLECTING: "Collecting",
    Activity.UPDATING: "Updating",
    Activity.VALIDATING: "Validating",
    Activity.REPORTING: "Reporting",
}


@dataclass
class Panel:
    env: int
    generation: int = 0
    state: PanelState = PanelState.WATCHING
    frame: dict | None = None


class Selection:
    def __init__(self, count, *, rng=None):
        self.count = count
        self.rng = rng or random.Random()
        self.panels = [Panel(i) for i in self.rng.sample(range(count), min(PANELS, count))]
        self.active = set(range(count))
        self.pending = set()
        self.finalized_rewards = {}
        self.episode_floor = {}
        self.completed_frames = {}

    def refresh(self, active):
        self.active = set(active)
        changed = False
        for index in sorted(self.pending.copy()):
            changed = self.switch(index, self.panels[index].generation) or changed
        return changed

    def switch(self, panel, generation):
        if not 0 <= panel < len(self.panels):
            return False
        old = self.panels[panel]
        if generation != old.generation:
            return False
        occupied = {p.env for p in self.panels}
        candidates = sorted((self.active | self.completed_frames.keys()) - occupied)
        if not candidates and old.env in self.completed_frames:
            candidates = [old.env]  # All games are already displayed; acknowledge the request.
        if not candidates:
            self.pending.add(panel)
            old.state = PanelState.SELECTING
            return False
        selected = self.rng.choice(candidates)
        self.pending.discard(panel)
        replacement = Panel(selected, old.generation + 1, PanelState.SELECTING)
        self.panels[panel] = replacement
        if selected in self.completed_frames:
            self.accept(panel, replacement.generation, self.completed_frames[selected])
        else:
            replacement.frame = old.frame
        return True

    def _finalized_frame(self, frame):
        env = frame["env"]
        if frame["episode"] < self.episode_floor.get(env, 0):
            return None
        saved = self.finalized_rewards.get(env)
        if saved is not None:
            episode, reward = saved
            if frame["episode"] < episode:
                return None
            if frame["episode"] == episode:
                frame = dict(frame, reward=reward, reward_pending=False)
            else:
                del self.finalized_rewards[env]
        return frame

    def remember_completed(self, frame):
        """Keep one public result per worker for browsing a one-decision cohort."""
        frame = self._finalized_frame(frame)
        if frame is None or frame["outcome"] == "running":
            return
        previous = self.completed_frames.get(frame["env"])
        if previous is None or frame["sequence"] > previous["sequence"]:
            self.completed_frames[frame["env"]] = frame

    def accept(self, panel, generation, frame):
        p = self.panels[panel]
        if generation != p.generation or frame.get("env", p.env) != p.env:
            return False
        previous = p.frame
        if (
            previous is not None
            and previous.get("env", p.env) == p.env
            and previous["episode"] == frame["episode"]
            and frame["sequence"] <= previous["sequence"]
        ):
            return False
        frame = self._finalized_frame(frame)
        if frame is None:
            return False
        p.frame = frame
        if panel not in self.pending:
            p.state = PanelState.RESULT if frame["outcome"] != "running" else PanelState.WATCHING
        return True

    def reset_episodes(self, episodes, *, active=None):
        """Expire finalized overrides and fence captures from reset episodes."""
        episodes = dict(episodes)
        self.completed_frames.clear()
        for env, episode in episodes.items():
            self.episode_floor[env] = max(episode, self.episode_floor.get(env, 0))
            self.finalized_rewards.pop(env, None)
        for index, panel in enumerate(self.panels):
            if panel.env in episodes:
                panel.generation += 1
                panel.state = PanelState.SELECTING
                self.pending.discard(index)
            elif active is not None and panel.env not in active:
                panel.state = PanelState.SELECTING
                self.pending.add(index)

    def finalize_rewards(self, episodes):
        """Replace rewards by stable episode identity, including delayed captures."""
        for env, episode, reward in episodes:
            if episode < self.episode_floor.get(env, 0):
                continue
            saved = self.finalized_rewards.get(env)
            if saved is not None and episode < saved[0]:
                continue
            self.finalized_rewards[env] = (episode, reward)
            cached = self.completed_frames.get(env)
            if cached is not None and cached["episode"] == episode:
                self.completed_frames[env] = dict(cached, reward=reward, reward_pending=False)
            for panel in self.panels:
                frame = panel.frame
                if (
                    panel.env == env
                    and frame is not None
                    and frame.get("env", env) == env
                    and frame["episode"] == episode
                ):
                    panel.frame = dict(frame, reward=reward, reward_pending=False)


def accept_grad_scale(value):
    """Finite positive coefficient. Booleans and other types stay rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        return None
    return value


def coerce_grad_scale(value):
    """Parse a typed field, then apply the same acceptance rule."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    return accept_grad_scale(value)


def format_grad_scale(value):
    accepted = accept_grad_scale(value)
    return "1" if accepted is None else format(accepted, ".6g")


def restored_grad_scale(extra, key):
    """Missing or rejected recovery fields stay at 1."""
    if not isinstance(extra, dict):
        return 1.0
    accepted = accept_grad_scale(extra.get(key, 1.0))
    return 1.0 if accepted is None else accepted


def pull_shared_grad_scale(requested, current):
    """Read the view's pending coefficient without changing the active scale."""
    with requested.get_lock():
        accepted = accept_grad_scale(requested.value)
        if accepted is None:
            requested.value = float(current)
            return float(current)
        return accepted


def latch_shared_grad_scale(requested, active, current):
    """Copy the pending coefficient onto the scale this cohort will use."""
    with requested.get_lock():
        accepted = accept_grad_scale(requested.value)
        if accepted is None:
            accepted = float(current)
            requested.value = accepted
        active.value = float(accepted)
        return accepted


def publish_shared_grad_scale(requested, active, requested_value, active_value):
    """Restore both shares from checkpoint state."""
    with requested.get_lock():
        requested.value = float(requested_value)
        active.value = float(active_value)


def read_shared_grad_scale(requested, active):
    """Pending coefficient, then the scale already in force."""
    if requested is None and active is None:
        return 1.0, 1.0
    if requested is None:
        return 1.0, float(active.value)
    with requested.get_lock():
        pending = float(requested.value)
        current = float(active.value) if active is not None else 1.0
        return pending, current


def accept_lr_restart(value):
    """Armed is 1 and idle is 0. Every other value stays rejected."""
    if isinstance(value, bool) or type(value) is not int or value not in (0, 1):
        return None
    return value


def restored_lr_restart(extra, key):
    """Missing or rejected recovery fields stay idle."""
    if not isinstance(extra, dict):
        return 0
    accepted = accept_lr_restart(extra.get(key, 0))
    return 0 if accepted is None else accepted


def pull_shared_lr_restart(requested, current):
    """Read the armed request without starting the open cohort's rate."""
    with requested.get_lock():
        accepted = accept_lr_restart(int(requested.value))
        if accepted is None:
            requested.value = int(current)
            return int(current)
        return accepted


def latch_shared_lr_restart(requested, active):
    """Move an armed request onto this cohort and clear the request."""
    with requested.get_lock():
        armed = 1 if accept_lr_restart(int(requested.value)) else 0
        requested.value = 0
        active.value = armed
        return armed


def publish_shared_lr_restart(requested, active, requested_value, active_value):
    """Restore both restart shares from checkpoint state."""
    with requested.get_lock():
        requested.value = int(requested_value)
        active.value = int(active_value)


def clear_shared_lr_restart_active(active):
    """The finished cohort releases its rate. A request for the next one stays."""
    active.value = 0


def read_shared_lr_restart(requested, active):
    """Armed request, then whether this cohort is already using the high rate."""
    if requested is None and active is None:
        return 0, 0
    if requested is None:
        return 0, int(active.value)
    with requested.get_lock():
        pending = int(requested.value)
        current = int(active.value) if active is not None else 0
        return pending, current


def toggle_shared_lr_restart(requested):
    """A second click before the cohort begins cancels the request."""
    with requested.get_lock():
        requested.value = 0 if int(requested.value) else 1
        return int(requested.value)


def offer_latest(queue, item):
    """Never wait for a slow renderer. Terminal panels also persist in the sender."""
    try:
        queue.put_nowait(item)
        return True
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            return False  # A multiprocessing feeder may not have exposed it yet.
        try:
            queue.put_nowait(item)
            return True
        except Full:
            return False


def history_page(selection, journal, command):
    """Respond only to the current subscription, with a bounded current-game page."""
    panel = command["panel"]
    if journal is None or not 0 <= panel < len(selection.panels):
        return None
    current = selection.panels[panel]
    if current.generation != command["generation"] or current.env != command["env"]:
        return None
    return journal.page(current.env, command["episode"], command.get("start"))


def learning_settings(cfg):
    """Snapshot effective learning values without reading defaults or device state."""
    training = cfg["training"]
    return (
        (
            "Learning",
            tuple(
                f"{key}={training[key]}"
                for key in (
                    "learning_rate",
                    "max_grad_norm",
                    "batch_size",
                    "gamma",
                )
            )
            + (f"fit_batch_groups={training['performance']['fit_batch_groups']}",),
        ),
        (
            "Passes",
            (
                tuple(
                    f"{stage}={table[stage]}"
                    for table in (training.get("stage_epochs") or {},)
                    for stage in SELECTABLE_STAGES
                    if stage in table
                )
                + (
                    (f"cohorts_per_epoch={training['cohorts_per_epoch']}",)
                    if type(training.get("cohorts_per_epoch")) is int
                    and not isinstance(training.get("cohorts_per_epoch"), bool)
                    else ()
                )
                or ((f"{training['n_epochs']}",) if "n_epochs" in training else ())
            )
            + tuple(
                item
                for item in (
                    (
                        f"learning_rate_restart={training['learning_rate_restart']}"
                        if type(training.get("learning_rate_restart")) in (int, float)
                        and not isinstance(training.get("learning_rate_restart"), bool)
                        else None
                    ),
                    (
                        f"stage_streak={training['stage_streak']}"
                        if type(training.get("stage_streak")) is int
                        and not isinstance(training.get("stage_streak"), bool)
                        else None
                    ),
                )
                if item
            ),
        ),
        (
            "Outcome weights",
            tuple(
                f"{stage}: static_field={field}, probe={probe}"
                for stage in (
                    "cherry_bomb",
                    "cherry_bomb_obstacle",
                    "attacker_pre",
                    "prediction",
                    "attacker",
                    "peashooter",
                    "normal",
                )
                for field, probe in (outcome_prediction_weights(cfg, stage),)
            ),
        ),
        (
            "Rewards",
            tuple(f"{key}={value}" for key, value in cfg["reward"].items() if key != "version"),
        ),
        (
            "Tile exploration",
            tuple(
                f"{key}={value}"
                for key, value in training["exploration"].items()
                if key != "objective"
            ),
        ),
    )


class LiveSession:
    """Process lifecycle and bounded mailboxes, owned by the training process."""

    def __init__(self, count, settings, *, cfg=None, notify=None):
        self.selection = Selection(count)
        self.fps = settings["live_fps"]
        self.notify = notify or (lambda text: warnings.warn(text, RuntimeWarning, stacklevel=2))
        ctx = mp.get_context("spawn")
        self.frames = ctx.Queue(maxsize=1)
        self.commands = ctx.Queue(maxsize=8)
        self.errors = ctx.Queue(maxsize=1)
        self.history_responses = ctx.Queue(maxsize=4)
        self.journal = None
        self.closed = ctx.Event()
        self.ready = ctx.Event()
        self.activity = ctx.Value("i", int(Activity.STARTING))
        self.grad_scale_requested = ctx.Value("d", 1.0)
        self.grad_scale_active = ctx.Value("d", 1.0)
        self.lr_restart_requested = ctx.Value("i", 0)
        self.lr_restart_active = ctx.Value("i", 0)
        # A requested nonzero port is retained until the web child binds. A
        # zero value preserves the existing ephemeral-port behavior.
        self.web_port = ctx.Value("i", int(settings.get("viewer_port", 0)))
        self.viewer = settings.get("viewer", "auto")
        target = viewer_main
        if self.viewer == "web" or (
            self.viewer == "auto" and os.name != "nt" and not os.environ.get("DISPLAY")
        ):
            from pvz_rl.presentation.web_viewer import web_viewer_main

            target = web_viewer_main
        self.process = ctx.Process(
            target=target,
            args=(
                self.frames,
                self.commands,
                self.history_responses,
                self.errors,
                self.closed,
                self.ready,
                self.activity,
                settings["live_window_size"],
                count,
                learning_settings(cfg) if cfg is not None else (),
                self.web_port,
                self.grad_scale_requested,
                self.grad_scale_active,
                self.lr_restart_requested,
                self.lr_restart_active,
            ),
            name="pvz-live-view",
            daemon=True,
        )
        self.failed = False
        self.stopped = False
        self.last_publish = -float("inf")
        self.started = monotonic()
        self.stats = {"published": 0, "dropped": 0}
        try:
            self.process.start()
        except Exception as exc:
            self.fail(exc)

    @property
    def proxy_url(self):
        """Relative path suitable for Jupyter Server Proxy and JupyterHub."""
        return f"./proxy/{self.web_port.value}/" if self.web_port.value else None

    @property
    def enabled(self):
        return not self.stopped and not self.closed.is_set()

    def fail(self, error):
        if not self.failed:
            self.notify(f"Live viewer disabled; training continues: {error}")
            self.failed = True
        self.closed.set()

    def poll(self):
        try:
            self.fail(self.errors.get_nowait())
        except Empty:
            pass
        if not self.enabled:
            return False
        if not self.process.is_alive():
            self.fail("viewer process exited unexpectedly")
            return False
        if not self.ready.is_set() and monotonic() - self.started > 15:
            self.fail("display startup timed out")
            return False
        changed = False
        for _ in range(8):
            try:
                command = self.commands.get_nowait()
            except Empty:
                break
            if isinstance(command, dict):
                page = history_page(self.selection, self.journal, command)
                if page is not None:
                    offer_latest(self.history_responses, dict(**command, page=page))
            else:
                changed = self.selection.switch(*command) or changed
        return changed

    def pull_grad_scale(self, current):
        return pull_shared_grad_scale(self.grad_scale_requested, current)

    def latch_grad_scale(self, current):
        return latch_shared_grad_scale(
            self.grad_scale_requested, self.grad_scale_active, current
        )

    def publish_grad_scale(self, requested, active):
        publish_shared_grad_scale(
            self.grad_scale_requested, self.grad_scale_active, requested, active
        )

    def pull_lr_restart(self, current):
        return pull_shared_lr_restart(self.lr_restart_requested, current)

    def latch_lr_restart(self):
        return latch_shared_lr_restart(self.lr_restart_requested, self.lr_restart_active)

    def publish_lr_restart(self, requested, active):
        publish_shared_lr_restart(
            self.lr_restart_requested, self.lr_restart_active, requested, active
        )

    def clear_lr_restart_active(self):
        clear_shared_lr_restart_active(self.lr_restart_active)

    def publish(self, *, force=False):
        if not self.enabled or not self.ready.is_set():
            return
        now = monotonic()
        if not force and now - self.last_publish < 1 / self.fps:
            return
        packet = [
            dict(env=p.env, generation=p.generation, state=p.state.value, frame=p.frame)
            for p in self.selection.panels
        ]
        key = "published" if offer_latest(self.frames, packet) else "dropped"
        self.stats[key] += 1
        self.last_publish = now

    def finalize_rewards(self, episodes):
        self.selection.finalize_rewards(episodes)
        self.publish(force=True)

    def set_activity(self, activity):
        if self.enabled:
            self.activity.value = int(activity)

    def close(self):
        if self.stopped:
            return
        self.stopped = True
        self.closed.set()
        if self.process.pid is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        for queue in (self.frames, self.commands, self.history_responses, self.errors):
            queue.cancel_join_thread()
            queue.close()


def decision_reason(selected_kind, values):
    """Explain the recorded greedy decision using its pre-action Q values."""
    selected = values[selected_kind]
    if selected is None or any(q is not None and not isfinite(q) for q in values):
        return "Q comparison unavailable"
    alternatives = [(i, q) for i, q in enumerate(values) if i != selected_kind and q is not None]
    if not alternatives:
        return "only recorded branch"
    runner_up, value = max(alternatives, key=lambda item: item[1])
    gap = selected - value
    if gap < 0:
        return "recorded choice differs from Q ranking"
    if gap == 0:
        return "tied best; priority wait > species order > dig"
    return f"highest Q; lead {gap:+.6g} over {DECISION_BRANCHES[runner_up]}"


from .live_layout import draw_view, viewer_main  # noqa: E402, F401
