"""Bounded, CPU-only presentation of selected live training games.

The viewer process never imports torch or owns a simulator. Selection uses a
private RNG; commands change subscriptions, never game state.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from queue import Empty, Full
from time import monotonic

PANELS = 4
ACTION_HISTORY = 3
RESULT_SECONDS = 1.0
UI_FPS = 30


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
    Activity.UPDATING: "Learner updating / preparing rollout - training boards unchanged",
    Activity.VALIDATING: "Validation - training games paused",
    Activity.REPORTING: "Reporting - training games paused",
}


@dataclass
class Panel:
    env: int
    generation: int = 0
    state: PanelState = PanelState.WATCHING
    actions: deque = field(default_factory=lambda: deque(maxlen=ACTION_HISTORY))
    frame: dict | None = None


class Selection:
    def __init__(self, count, *, rng=None):
        self.count = count
        self.rng = rng or random.Random()
        self.panels = [Panel(i) for i in self.rng.sample(range(count), min(PANELS, count))]

    def switch(self, panel, generation):
        if not 0 <= panel < len(self.panels):
            return False
        old = self.panels[panel]
        if generation != old.generation:
            return False
        occupied = {p.env for p in self.panels}
        candidates = sorted(set(range(self.count)) - occupied)
        selected = self.rng.choice(candidates) if candidates else old.env
        self.panels[panel] = Panel(selected, old.generation + 1, PanelState.SELECTING)
        return True

    def accept(self, panel, generation, frame):
        p = self.panels[panel]
        if generation != p.generation:
            return False
        if p.frame is not None and frame["sequence"] <= p.frame["sequence"]:
            return False
        p.frame = frame
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
        self.closed = ctx.Event()
        self.ready = ctx.Event()
        self.activity = ctx.Value("i", int(Activity.STARTING))
        self.process = ctx.Process(
            target=viewer_main,
            args=(
                self.frames,
                self.commands,
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
                panel, generation = self.commands.get_nowait()
            except Empty:
                break
            changed = self.selection.switch(panel, generation) or changed
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
        for queue in (self.frames, self.commands, self.errors):
            queue.cancel_join_thread()
            queue.close()


def draw_view(surface, packets, activity, *, renderers, boards, pending=(), count=128):
    """Draw through the pinned renderer; reusable in offscreen visual controls."""
    import pygame
    from pvz_game.rendering import BoardRenderer, RenderContext

    surface.fill((19, 28, 24))
    width, height = surface.get_size()
    if "fonts" not in renderers:
        renderers["fonts"] = (
            pygame.font.SysFont("Segoe UI", 17),
            pygame.font.SysFont("Segoe UI", 15),
        )
    font, small = renderers["fonts"]
    surface.blit(font.render(ACTIVITY_TEXT[Activity(activity)], True, (225, 238, 225)), (12, 8))
    buttons = []
    original_clip = surface.get_clip()
    cell_w, cell_h = width // 2, (height - 36) // 2
    for index in range(PANELS):
        x, y = index % 2 * cell_w, 36 + index // 2 * cell_h
        rect = pygame.Rect(x + 4, y + 4, cell_w - 8, cell_h - 8)
        surface.set_clip(rect)
        pygame.draw.rect(surface, (37, 51, 43), rect, border_radius=5)
        if index >= len(packets):
            surface.blit(
                font.render(
                    "Waiting for training state" if index < count else "No additional environment",
                    True,
                    (175, 185, 175),
                ),
                (x + 12, y + 12),
            )
            continue
        packet = packets[index]
        frame = packet["frame"]
        button = pygame.Rect(x + cell_w - 104, y + 9, 92, 30)
        enabled = count > len(packets) and index not in pending
        pygame.draw.rect(
            surface, (71, 112, 83) if enabled else (60, 65, 62), button, border_radius=4
        )
        surface.blit(
            small.render(
                "Switch" if enabled else "Waiting" if index in pending else "No others",
                True,
                (240, 243, 235),
            ),
            (button.x + 9, button.y + 5),
        )
        if enabled:
            buttons.append((button, index, packet["generation"]))
        title = f"Env {packet['env']}"
        if frame:
            title += f" | {frame['task']} | episode {frame['episode']}"
        title_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(x + 12, y + 9, max(1, cell_w - 122), 32))
        surface.blit(font.render(title, True, (236, 239, 224)), (x + 12, y + 12))
        surface.set_clip(title_clip)
        if not frame:
            surface.blit(
                small.render("Selecting - waiting for collection", True, (200, 210, 200)),
                (x + 12, y + 52),
            )
            continue
        size = (max(1, cell_w - 16), max(1, cell_h - 116))
        key = (index, size)
        if key not in renderers:
            renderers[key] = BoardRenderer(size=size)
        renderer = renderers[key]
        identity = (packet["generation"], frame["sequence"], size)
        if index not in boards or boards[index][0] != identity:
            boards[index] = (
                identity,
                renderer.render(
                    frame["observation"], context=RenderContext(outcome=frame["outcome"])
                ),
            )
        surface.blit(boards[index][1], (x + 8, y + 46))
        obs = frame["observation"]
        caption = f"{obs.elapsed_seconds:.2f}s | sun {obs.sun} | {frame['outcome']}"
        if index in pending:
            caption += " | switch pending"
        surface.blit(small.render(caption, True, (237, 222, 163)), (x + 12, y + cell_h - 64))
        for offset, text in enumerate(frame["actions"][-2:]):
            surface.blit(
                small.render(text, True, (213, 227, 214)), (x + 12, y + cell_h - 44 + offset * 18)
            )
    surface.set_clip(original_clip)
    return buttons


def viewer_main(frames, commands, errors, closed, ready, activity, size, count):
    """Spawn entry point: no model, CUDA context, engine instance, or replay."""
    try:
        import pygame

        pygame.display.init()
        pygame.font.init()
        desktop = pygame.display.get_desktop_sizes()[0]
        size = (min(size[0], max(640, desktop[0] - 80)), min(size[1], max(480, desktop[1] - 100)))
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("PVZ | Four live training games")
        ready.set()
        clock = pygame.time.Clock()
        packets, buttons, renderers, boards, pending, results = [], [], {}, {}, {}, {}
        dirty, displayed_activity = True, None

        def switch(index, generation):
            nonlocal dirty
            try:
                commands.put_nowait((index, generation))
                pending[index] = generation
                dirty = True
            except Full:
                pass

        while not closed.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    closed.set()
                elif event.type in (pygame.WINDOWEXPOSED, pygame.WINDOWRESTORED):
                    dirty = True
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(
                        (max(640, event.w), max(480, event.h)), pygame.RESIZABLE
                    )
                    renderers.clear()
                    boards.clear()
                    dirty = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, index, generation in buttons:
                        if rect.collidepoint(event.pos):
                            switch(index, generation)
            try:
                incoming = frames.get_nowait()
                dirty = True
                # Never regress even if a delayed transport frame arrives.
                packets = [
                    old
                    if i < len(packets) and (old := packets[i])["generation"] > p["generation"]
                    else p
                    for i, p in enumerate(incoming)
                ]
            except Empty:
                pass
            for index, packet in enumerate(packets):
                generation = packet["generation"]
                if index in pending and pending[index] < generation:
                    del pending[index]
                if packet["frame"] and packet["frame"]["outcome"] != "running":
                    key = (index, generation)
                    until = results.setdefault(key, monotonic() + RESULT_SECONDS)
                    if monotonic() >= until and index not in pending:
                        switch(index, generation)
                for key in list(results):
                    if key[0] == index and key[1] != generation:
                        del results[key]
            current_activity = activity.value
            if dirty or current_activity != displayed_activity:
                buttons = draw_view(
                    screen,
                    packets,
                    current_activity,
                    renderers=renderers,
                    boards=boards,
                    pending=pending,
                    count=count,
                )
                pygame.display.flip()
                dirty, displayed_activity = False, current_activity
            clock.tick(UI_FPS)
    except Exception as exc:
        try:
            errors.put_nowait(f"{type(exc).__name__}: {exc}")
        except Full:
            pass
        # Parent polls this diagnostic before deciding to disable presentation.
    finally:
        try:
            import pygame

            pygame.quit()
        except ImportError:
            pass
