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

from pvz_rl.envs.actions import ActionSchema as A

PANELS = 4
ACTION_HISTORY = 3
RESULT_SECONDS = 1.0
UI_FPS = 30


class DecisionLayout:
    species = slice(3, 3 + A.plant_types)
    tiles = slice(species.stop, species.stop + A.plant_types * A.tiles)
    action = tiles.stop
    tick = action + 1


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
    CRITIC_COLLECTING = 5
    ACTOR_COLLECTING = 6
    CRITIC_UPDATING = 7
    ACTOR_UPDATING = 8


ACTIVITY_TEXT = {
    Activity.STARTING: "Starting training",
    Activity.COLLECTING: "Live training collection (sampled actions, including exploration)",
    Activity.UPDATING: "Learner updating / preparing cohort - training boards unchanged",
    Activity.VALIDATING: "Validation - training games paused",
    Activity.REPORTING: "Reporting - training games paused",
    Activity.CRITIC_COLLECTING: "Critic role | collecting complete games | both networks frozen",
    Activity.ACTOR_COLLECTING: "Actor role | collecting complete games | both networks frozen",
    Activity.CRITIC_UPDATING: "Critic role | fitting returns | training boards paused",
    Activity.ACTOR_UPDATING: "Actor role | conditional planting PPO | training boards paused",
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
        self.panels[panel] = Panel(selected, old.generation + 1, PanelState.SELECTING)
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
        self.role = ctx.Value("i", 0)
        self.process = ctx.Process(
            target=viewer_main,
            args=(
                self.frames,
                self.commands,
                self.errors,
                self.closed,
                self.ready,
                self.activity,
                self.role,
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

    def set_role(self, role):
        self.role.value = int(role == "actor")

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


def draw_view(surface, packets, activity, *, renderers, boards, pending=(), count=32, role=None):
    """Draw through the pinned renderer; reusable in offscreen visual controls."""
    import pygame
    from pvz_game.rendering import BoardRenderer, RenderContext

    surface.fill((19, 28, 24))
    width, height = surface.get_size()
    # Scale the complete layout on small screens instead of clipping the board
    # behind the decision panel. Convert hit boxes back to window coordinates.
    if width < 1280 or height < 950:
        size = (max(width, 1280), max(height, 950))
        canvas = renderers.get("canvas")
        if canvas is None or canvas.get_size() != size:
            canvas = renderers["canvas"] = pygame.Surface(size)
        buttons = draw_view(
            canvas,
            packets,
            activity,
            renderers=renderers,
            boards=boards,
            pending=pending,
            count=count,
            role=role,
        )
        surface.blit(pygame.transform.smoothscale(canvas, (width, height)), (0, 0))
        sx, sy = width / size[0], height / size[1]
        return [
            (
                pygame.Rect(
                    round(r.x * sx),
                    round(r.y * sy),
                    max(1, round(r.w * sx)),
                    max(1, round(r.h * sy)),
                ),
                i,
                target,
            )
            for r, i, target in buttons
        ]
    if "fonts" not in renderers:
        renderers["fonts"] = (
            pygame.font.SysFont("Segoe UI", 17),
            pygame.font.SysFont("Segoe UI", 15),
        )
    font, small = renderers["fonts"]
    label = ACTIVITY_TEXT[Activity(activity)]
    if role and "role" not in label:
        label = f"{role.capitalize()} role | " + label
    surface.blit(font.render(label, True, (225, 238, 225)), (12, 8))
    buttons = []
    species_selection = renderers.setdefault("species_selection", {})
    current_selections = {(i, packet["generation"]) for i, packet in enumerate(packets)}
    for key in list(species_selection):
        if key not in current_selections:
            del species_selection[key]
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
        diagnostic = frame.get("decision")
        footer = 230 if diagnostic else 70
        size = (max(1, cell_w - 16), max(1, cell_h - footer - 46))
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
        surface.blit(
            small.render(caption, True, (237, 222, 163)), (x + 12, y + cell_h - footer + 4)
        )
        for offset, text in enumerate(frame["actions"][-2:]):
            surface.blit(
                small.render(text, True, (213, 227, 214)),
                (x + 12, y + cell_h - footer + 24 + offset * 18),
            )
        if diagnostic:
            from pvz_game.config import PLANT_TYPES

            base_y = y + cell_h - footer + 63
            action = diagnostic["action"]
            selected_kind = 0 if action == 0 else 1 if action < A.dig_start else 2
            q_text = "Estimated returns: " + " | ".join(
                (">" if i == selected_kind else "")
                + name
                + " "
                + ("n/a" if q is None else f"{q:+.3f}")
                for i, (name, q) in enumerate(
                    zip(("Q(wait)", "Q(plant)", "Q(dig best)"), diagnostic["q"])
                )
            )
            surface.blit(small.render(q_text, True, (245, 229, 177)), (x + 12, base_y))
            decoded = (
                "wait"
                if action == 0
                else (
                    f"dig tile {action - A.dig_start}"
                    if action >= A.dig_start
                    else f"{PLANT_TYPES[(action - 1) // A.tiles]} tile {(action - 1) % A.tiles}"
                )
            )
            surface.blit(
                small.render(
                    f"Latest decision {diagnostic['tick'] / obs.tick_rate:.2f}s: {decoded} | If planting:",
                    True,
                    (219, 230, 219),
                ),
                (x + 12, base_y + 20),
            )
            key = (index, packet["generation"])
            default = (
                (action - 1) // A.tiles
                if 0 < action < A.dig_start
                else max(range(A.plant_types), key=lambda k: diagnostic["plants"][k])
            )
            species = species_selection.get(key, default)
            button_w = max(50, (cell_w - 242) // 2)
            for k, (name, probability) in enumerate(zip(PLANT_TYPES, diagnostic["plants"])):
                r = pygame.Rect(
                    x + 12 + k % 2 * button_w, base_y + 43 + k // 2 * 24, button_w - 4, 22
                )
                pygame.draw.rect(surface, (68, 106, 78) if k == species else (46, 65, 52), r)
                surface.blit(
                    small.render(
                        f"{name.replace('_', ' ')} {probability:.1%}",
                        True,
                        (233, 239, 227) if probability else (141, 157, 144),
                    ),
                    (r.x + 3, r.y + 1),
                )
                buttons.append((r, index, (packet["generation"], k)))
            tile_probs = diagnostic["tiles"][species]
            maximum = max(tile_probs) or 1
            for tile, probability in enumerate(tile_probs):
                r = pygame.Rect(
                    x + cell_w - 223 + tile % A.cols * 23, base_y + 44 + tile // A.cols * 19, 21, 17
                )
                intensity = probability / maximum
                pygame.draw.rect(
                    surface,
                    (int(40 + 150 * intensity), int(55 + 160 * intensity), 60)
                    if probability
                    else (29, 35, 31),
                    r,
                )
            surface.blit(
                small.render("Tile heatmap: brighter = higher", True, (203, 219, 204)),
                (x + cell_w - 225, base_y + 140),
            )
    surface.set_clip(original_clip)
    return buttons


def viewer_main(frames, commands, errors, closed, ready, activity, role, size, count):
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
        displayed_role = None

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
                    selections = renderers.get("species_selection", {})
                    renderers.clear()
                    renderers["species_selection"] = selections
                    boards.clear()
                    dirty = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, index, generation in buttons:
                        if rect.collidepoint(event.pos):
                            if isinstance(generation, tuple):
                                gen, species = generation
                                renderers.setdefault("species_selection", {})[(index, gen)] = (
                                    species
                                )
                                dirty = True
                            else:
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
            if dirty or current_activity != displayed_activity or role.value != displayed_role:
                buttons = draw_view(
                    screen,
                    packets,
                    current_activity,
                    renderers=renderers,
                    boards=boards,
                    pending=pending,
                    count=count,
                    role="actor" if role.value else "critic",
                )
                pygame.display.flip()
                dirty, displayed_activity = False, current_activity
                displayed_role = role.value
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
