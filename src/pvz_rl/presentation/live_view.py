"""Bounded, CPU-only presentation of selected live training games.

Viewer code owns no model or simulator; it renders public snapshots. Selection
uses a private RNG; commands change subscriptions, never game state.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import warnings
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from queue import Empty, Full
from time import monotonic

from pvz_game.config import PLANT_TYPES

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
    Activity.STARTING: "Starting training",
    Activity.COLLECTING: "Live training collection (sampled actions, including exploration)",
    Activity.UPDATING: "Learner updating / preparing cohort - training boards unchanged",
    Activity.VALIDATING: "Validation - training games paused",
    Activity.REPORTING: "Reporting - training games paused",
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
        candidates = sorted(self.active - occupied)
        if not candidates:
            self.pending.add(panel)
            old.state = PanelState.SELECTING
            return False
        selected = self.rng.choice(candidates)
        self.pending.discard(panel)
        self.panels[panel] = Panel(selected, old.generation + 1, PanelState.SELECTING, old.frame)
        return True

    def accept(self, panel, generation, frame):
        p = self.panels[panel]
        if generation != p.generation:
            return False
        if p.frame is not None and frame["sequence"] <= p.frame["sequence"]:
            return False
        p.frame = frame
        if panel not in self.pending:
            p.state = PanelState.RESULT if frame["outcome"] != "running" else PanelState.WATCHING
        return True


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


class LiveSession:
    """Process lifecycle and bounded mailboxes, owned by the training process."""

    def __init__(self, count, settings, *, notify=None):
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
        self.process = ctx.Process(
            target=viewer_main,
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
    """Explain the recorded greedy decision using its legal, pre-action Q values."""
    selected = values[selected_kind]
    if selected is None or any(q is not None and not isfinite(q) for q in values):
        return "Q comparison unavailable"
    alternatives = [(i, q) for i, q in enumerate(values) if i != selected_kind and q is not None]
    if not alternatives:
        return "only legal branch"
    runner_up, value = max(alternatives, key=lambda item: item[1])
    gap = selected - value
    if gap < 0:
        return "recorded choice differs from Q ranking"
    if gap == 0:
        return "tied best; priority wait > species order > dig"
    return f"highest legal Q; lead {gap:+.6g} over {DECISION_BRANCHES[runner_up]}"


from .live_layout import draw_view, viewer_main  # noqa: E402, F401
