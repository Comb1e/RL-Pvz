"""Resizable board overview and readable paged decision tables. CPU process only."""

from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Full
from time import monotonic

from pvz_game.config import PLANT_TYPES
from pvz_game.cuda.schema import REASONS

from pvz_rl.envs.actions import ActionSchema as A

NAMES = ("wait", *PLANT_TYPES, "dig")
SHORT = (
    "Wait",
    "Sunflower",
    "Peashooter",
    "Wall-nut",
    "Cherry",
    "Mine",
    "Snow pea",
    "Chomper",
    "Repeater",
    "Dig",
)


def branch(action):
    return 0 if action == 0 else 1 + (action - 1) // A.tiles


class HistoryMode(StrEnum):
    FOLLOWING = "following latest"
    BROWSING = "browsing history"


@dataclass
class Browse:
    mode: HistoryMode = HistoryMode.FOLLOWING
    start: int = 0
    horizontal: int = 0
    selected: dict | None = None
    page: dict | None = None
    request: int = 0
    sent: float = 0

    @property
    def follow(self):
        return self.mode == HistoryMode.FOLLOWING

    @follow.setter
    def follow(self, value):
        self.mode = HistoryMode.FOLLOWING if value else HistoryMode.BROWSING

    def choose(self, row):
        self.follow = False
        self.selected = row
        if self.page is not None:
            offset = next(
                (i for i, r in enumerate(self.page["rows"]) if r["sequence"] == row["sequence"]), 0
            )
            self.start = self.page["start"] + offset
            self.page = {**self.page, "start": self.start, "rows": self.page["rows"][offset:]}
        self.request += 1
        self.sent = 0


def accept_history_response(packets, browse, response):
    """A page belongs to the displayed game, never a pending replacement."""
    index = response["panel"]
    if not 0 <= index < len(packets):
        return False
    packet = packets[index]
    frame = packet["frame"]
    if (
        packet["state"] == "selecting"
        or frame is None
        or packet["env"] != response["env"]
        or packet["generation"] != response["generation"]
        or frame["episode"] != response["episode"]
    ):
        return False
    state = browse.get((index, response["generation"], response["episode"]))
    if state is None or state.follow or state.request != response["request"]:
        return False
    state.page = response["page"]
    state.sent = float("inf")
    return True


def draw_view(surface, packets, activity, *, renderers, boards, pending=(), count=128):
    import pygame
    from pvz_game.rendering import BoardRenderer, RenderContext

    from .live_view import ACTIVITY_TEXT, Activity, decision_reason

    surface.fill((18, 27, 24))
    width, height = surface.get_size()
    font, small = renderers.setdefault(
        "fonts", (pygame.font.SysFont("Segoe UI", 17), pygame.font.SysFont("Consolas", 15))
    )
    focus = renderers.get("focus")
    browse = renderers.setdefault("browse", {})
    regions = renderers["regions"] = {}
    valid = {
        (i, p["generation"], p["frame"].get("episode") if p["frame"] else None)
        for i, p in enumerate(packets)
    }
    for key in list(browse):
        if key not in valid:
            del browse[key]
    label = (
        ACTIVITY_TEXT[Activity(activity)]
        + " | F11 fullscreen | Focus / Esc | wheel: history, Shift+wheel: columns"
    )
    surface.blit(small.render(label, True, (225, 237, 226)), (12, 8))
    buttons = []
    old_clip = surface.get_clip()
    indices = [focus] if focus is not None else list(range(4))
    for index in indices:
        cell_w, cell_h = (
            (width, height - 34) if focus is not None else (width // 2, (height - 34) // 2)
        )
        x, y = (0, 34) if focus is not None else (index % 2 * cell_w, 34 + index // 2 * cell_h)
        rect = pygame.Rect(x + 4, y + 3, cell_w - 8, cell_h - 6)
        regions[index] = rect
        surface.set_clip(rect)
        pygame.draw.rect(surface, (34, 47, 40), rect, border_radius=5)
        if index >= len(packets):
            surface.blit(
                font.render(
                    "No additional environment" if index >= count else "Awaiting first board",
                    True,
                    (211, 224, 210),
                ),
                (x + 12, y + 15),
            )
            continue
        packet, frame = packets[index], packets[index]["frame"]
        gen = packet["generation"]
        for label, offset, operation in [
            ("Switch", 92, "switch"),
            ("Grid" if focus is not None else "Focus", 170, "focus"),
            ("Follow latest", 290, "follow"),
        ]:
            r = pygame.Rect(x + cell_w - offset, y + 8, 112 if operation == "follow" else 72, 27)
            pygame.draw.rect(surface, (60, 99, 75), r, border_radius=4)
            surface.blit(small.render(label, True, (241, 243, 230)), (r.x + 4, r.y + 4))
            buttons.append((r, index, (operation, gen)))
        title = (
            f"Env {packet['env']}"
            if not frame
            else f"Env {packet['env']} | {frame['task']} | game {frame['episode']}"
        )
        switching = packet["state"] == "selecting" or index in pending
        if switching:
            title = f"Switching to env {packet['env']} — previous board retained"
        title_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(x + 10, y + 8, max(1, cell_w - 308), 28))
        surface.blit(font.render(title, True, (233, 237, 224)), (x + 12, y + 11))
        surface.set_clip(title_clip)
        if not frame:
            surface.blit(
                font.render("Waiting for collection", True, (201, 213, 201)), (x + 12, y + 60)
            )
            continue
        key = (index, gen, frame["episode"])
        state = browse.setdefault(key, Browse())
        incoming = frame.get("history")
        if state.follow and incoming is not None:
            state.page = incoming
        elif incoming is not None and state.page is not None:
            state.page["total"] = incoming["total"]
        page = state.page or incoming or dict(start=0, total=0, rows=[])
        board_h = max(
            85,
            int(
                cell_h
                * (
                    0.34
                    if focus is not None and height < 700
                    else 0.46
                    if focus is not None
                    else 0.40
                )
            ),
        )
        size = (max(1, cell_w - 20), board_h)
        cache_key = (index, size)
        if cache_key not in renderers:
            # At most one board renderer per panel, plus ordinary UI state.
            for old in list(renderers):
                if isinstance(old, tuple) and old[0] == index:
                    del renderers[old]
            renderers[cache_key] = BoardRenderer(size=size)
        identity = (gen, frame["sequence"], size)
        if index not in boards or boards[index][0] != identity:
            boards[index] = (
                identity,
                renderers[cache_key].render(
                    frame["observation"], context=RenderContext(outcome=frame["outcome"])
                ),
            )
        surface.blit(boards[index][1], (x + 10, y + 42))
        obs = frame["observation"]
        top = y + 45 + board_h
        caption = f"Board {obs.elapsed_seconds:.2f}s | sun {obs.sun} | {frame['outcome']}"
        if switching:
            caption += " | waiting for unfinished replacement"
        surface.blit(small.render(caption, True, (234, 219, 159)), (x + 12, top))
        decision = state.selected if not state.follow and state.selected else frame.get("decision")
        if decision:
            greedy, chosen = branch(decision["greedy_action"]), branch(decision["action"])
            legal = decision.get("legal", [q is not None for q in decision["q"]])
            values = [q if ok else None for q, ok in zip(decision["q"], legal)]
            mode = (
                "Latest decision" if state.follow or state.selected is None else "Historical action"
            )
            summary = f"{mode} #{decision.get('sequence', '?')} @ {decision['tick'] / obs.tick_rate:.2f}s: {NAMES[chosen]}"
            surface.blit(
                small.render(
                    summary + " | coins S/T " + "/".join(str(int(c)) for c in decision["coins"]),
                    True,
                    (244, 232, 175),
                ),
                (x + 12, top + 21),
            )
            reason = f"Greedy {NAMES[greedy]}: {decision_reason(greedy, values)}"
            if decision["action"] != decision["greedy_action"]:
                reason += " | exploration override"
            surface.blit(small.render(reason, True, (216, 230, 214)), (x + 12, top + 41))
            surface.blit(
                small.render(
                    "Estimated returns (raw Q); * = illegal; no probabilities",
                    True,
                    (220, 232, 215),
                ),
                (x + 12, top + 61),
            )
            # All ten outputs at fixed font size; horizontal scroll also applies here.
            for k, (q, ok) in enumerate(zip(decision["q"], legal)):
                text = (
                    f"{SHORT[k]} {q:+.5g}{'' if ok else '*'}"
                    if q is not None
                    else f"{SHORT[k]} unavailable"
                )
                color = (
                    (252, 223, 131) if k == chosen else (221, 231, 215) if ok else (146, 157, 150)
                )
                surface.blit(
                    small.render(text, True, color),
                    (x + 12 + k % 5 * 157 - state.horizontal, top + 81 + k // 5 * 20),
                )
        table_y = top + 126
        row_height = 22
        visible = max(0, (y + cell_h - table_y - 27 - row_height) // row_height)
        rows = (
            (page["rows"][-visible:] if state.follow else page["rows"][:visible]) if visible else []
        )
        headings = ["Decision", "Time(s)", "Action", "Status", *SHORT]
        widths = [78, 85, 112, 105] + [96] * 10
        tx = x + 12 - state.horizontal
        pygame.draw.rect(surface, (48, 67, 54), (x + 8, table_y, cell_w - 16, row_height))
        for heading, col_w in zip(headings, widths):
            surface.blit(small.render(heading, True, (229, 237, 221)), (tx, table_y + 2))
            tx += col_w
        for offset, row in enumerate(rows):
            ry = table_y + (offset + 1) * row_height
            if state.selected and row["sequence"] == state.selected["sequence"]:
                pygame.draw.rect(surface, (70, 86, 56), (x + 8, ry, cell_w - 16, row_height))
            cells = [
                str(row["sequence"]),
                f"{row['tick'] / obs.tick_rate:.2f}",
                NAMES[branch(row["action"])],
                "accepted" if row["accepted"] else "REJECTED",
            ]
            cells += [f"{q:+.5g}{'' if ok else '*'}" for q, ok in zip(row["q"], row["legal"])]
            tx = x + 12 - state.horizontal
            for c, (text, col_w) in enumerate(zip(cells, widths)):
                color = (220, 231, 218) if c < 4 or row["legal"][c - 4] else (142, 153, 146)
                surface.blit(small.render(text, True, color), (tx, ry + 2))
                tx += col_w
            buttons.append(
                (pygame.Rect(x + 8, ry, cell_w - 16, row_height), index, ("row", gen, row))
            )
        if not rows:
            surface.blit(
                small.render("No plant/dig actions in this game yet", True, (184, 205, 186)),
                (x + 12, table_y + 25),
            )
        if state.selected and not state.selected["accepted"]:
            reason = REASONS[state.selected["reason"]]
            surface.blit(
                small.render(f"Rejected: {reason}", True, (250, 181, 137)),
                (x + 12, y + cell_h - 22),
            )
        else:
            surface.blit(
                small.render(
                    f"{'Follow latest' if state.follow else 'Browsing'} | {page['total']} actions | scroll for more",
                    True,
                    (169, 192, 177),
                ),
                (x + 12, y + cell_h - 22),
            )
    surface.set_clip(old_clip)
    return buttons


def viewer_main(frames, commands, history_responses, errors, closed, ready, activity, size, count):
    import pygame

    from .live_view import RESULT_SECONDS, UI_FPS

    try:
        pygame.display.init()
        pygame.font.init()
        desktop = pygame.display.get_desktop_sizes()[0]
        size = (min(size[0], max(640, desktop[0] - 80)), min(size[1], max(480, desktop[1] - 100)))
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("PVZ | Live boards and complete game decision history")
        ready.set()
        clock = pygame.time.Clock()
        packets, buttons, renderers, boards, pending, results = [], [], {}, {}, {}, {}
        fullscreen = False
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
                elif event.type in (
                    pygame.WINDOWEXPOSED,
                    pygame.WINDOWRESTORED,
                    pygame.WINDOWMAXIMIZED,
                ):
                    dirty = True
                elif event.type == pygame.VIDEORESIZE:
                    size = (max(640, event.w), max(480, event.h))
                    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
                    boards.clear()
                    dirty = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        renderers["focus"] = None
                    elif event.key == pygame.K_F11:
                        fullscreen = not fullscreen
                        screen = pygame.display.set_mode(
                            desktop if fullscreen else size,
                            pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE,
                        )
                    dirty = True
                elif event.type == pygame.MOUSEWHEEL:
                    for index, rect in renderers.get("regions", {}).items():
                        if (
                            rect.collidepoint(pygame.mouse.get_pos())
                            and index < len(packets)
                            and packets[index]["frame"]
                        ):
                            p = packets[index]
                            key = (index, p["generation"], p["frame"]["episode"])
                            state = renderers["browse"].setdefault(key, Browse())
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT or event.x:
                                state.horizontal = max(
                                    0, min(1300, state.horizontal - (event.x or event.y) * 60)
                                )
                            else:
                                page = (
                                    state.page
                                    or p["frame"].get("history")
                                    or dict(total=0, start=0)
                                )
                                state.start = max(
                                    0,
                                    min(
                                        page["total"],
                                        (max(0, page["total"] - 8) if state.follow else state.start)
                                        - event.y * 8,
                                    ),
                                )
                                state.follow = False
                                state.request += 1
                                state.sent = 0
                            dirty = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, index, target in buttons:
                        if not rect.collidepoint(event.pos):
                            continue
                        operation, generation, *extra = target
                        p = packets[index]
                        if operation == "switch":
                            switch(index, generation)
                        elif operation == "focus":
                            renderers["focus"] = (
                                None if renderers.get("focus") is not None else index
                            )
                        elif p["frame"]:
                            key = (index, generation, p["frame"]["episode"])
                            state = renderers["browse"].setdefault(key, Browse())
                            if operation == "follow":
                                state.follow, state.selected = True, None
                            else:
                                state.choose(extra[0])
                        dirty = True
                        break
            try:
                incoming = frames.get_nowait()
                packets = [
                    packets[i]
                    if i < len(packets) and packets[i]["generation"] > p["generation"]
                    else p
                    for i, p in enumerate(incoming)
                ]
                dirty = True
            except Empty:
                pass
            for _ in range(4):
                try:
                    response = history_responses.get_nowait()
                except Empty:
                    break
                if accept_history_response(packets, renderers.get("browse", {}), response):
                    dirty = True
            for index, packet in enumerate(packets):
                generation, frame = packet["generation"], packet["frame"]
                if (
                    index in pending
                    and pending[index] < generation
                    and packet["state"] != "selecting"
                ):
                    del pending[index]
                if frame and frame["outcome"] != "running" and packet["state"] != "selecting":
                    until = results.setdefault((index, generation), monotonic() + RESULT_SECONDS)
                    if monotonic() >= until and index not in pending:
                        switch(index, generation)
                for key in list(results):
                    if key[0] == index and key[1] != generation:
                        del results[key]
                if not frame or packet["state"] == "selecting":
                    continue
                state = renderers.get("browse", {}).get((index, generation, frame["episode"]))
                if state and not state.follow and monotonic() - state.sent > 0.5:
                    try:
                        commands.put_nowait(
                            dict(
                                panel=index,
                                generation=generation,
                                env=packet["env"],
                                episode=frame["episode"],
                                start=state.start,
                                request=state.request,
                            )
                        )
                        state.sent = monotonic()
                    except Full:
                        pass
            if dirty or activity.value != displayed_activity:
                buttons = draw_view(
                    screen,
                    packets,
                    activity.value,
                    renderers=renderers,
                    boards=boards,
                    pending=pending,
                    count=count,
                )
                pygame.display.flip()
                dirty, displayed_activity = False, activity.value
            clock.tick(UI_FPS)
    except Exception as exc:
        try:
            errors.put_nowait(f"{type(exc).__name__}: {exc}")
        except Full:
            pass
    finally:
        pygame.quit()
