"""Resizable board overview and readable paged decision tables. CPU process only."""

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Full
from time import monotonic

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.tile_scenarios import is_placement_skill
from pvz_rl.presentation.forecast_display import scores_recorded, tile_label
from pvz_rl.presentation.outcome_view import draw_live_outcomes
from pvz_rl.presentation.rendering import render_context

TABLE_WIDTHS = (36, 44, 18, 36, *([48] * 10))
QUIET_WIDTHS = (36, 44, 18, 36)
CARD_COLOR = (
    (140, 148, 142),
    (248, 207, 94),
    (127, 186, 86),
    (197, 140, 77),
    (216, 92, 88),
    (201, 156, 89),
    (130, 201, 207),
    (180, 149, 229),
    (112, 181, 91),
    (196, 164, 120),
)


def horizontal_limit(viewport_width, widths=None):
    return max(0, sum(widths or TABLE_WIDTHS) - viewport_width)


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
    selected_offset: int | None = None
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
        self.selected = deepcopy(row)
        if self.page is not None:
            offset = next(
                (i for i, r in enumerate(self.page["rows"]) if r["sequence"] == row["sequence"]), 0
            )
            self.selected_offset = self.page["start"] + offset
            self.start = self.page["start"]
        self.request += 1
        self.sent = float("inf")

    def resume_follow(self, page=None):
        """Return to the live tail while retaining horizontal table scrolling."""
        self.mode = HistoryMode.FOLLOWING
        self.selected = None
        self.selected_offset = None
        self.page = page
        if page is not None:
            self.start = max(0, int(page.get("total", 0)))
        else:
            self.start = 0
        # Invalidate historical page requests without changing horizontal scroll.
        self.request += 1
        self.sent = float("inf")


def resume_visible_history(packets, renderers, indices=None):
    """Resume the displayed games; a replacement never inherits an old page."""
    if indices is None:
        focus = renderers.get("focus")
        indices = [focus] if focus is not None else range(min(4, len(packets)))
    for index in indices:
        if not 0 <= index < len(packets):
            continue
        packet = packets[index]
        frame = packet.get("frame")
        if packet.get("state") == "selecting" or frame is None:
            continue
        key = (index, packet["generation"], frame["episode"])
        if renderers.get("browse_keys", {}).get(index) != key:
            continue
        state = renderers.setdefault("browse", {}).setdefault(key, Browse())
        state.resume_follow(frame.get("history"))


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


def draw_learning_settings(surface, font, renderers):
    """Wrap immutable run settings above the boards, including before the first frame."""
    settings = renderers.get("learning_settings", ())
    if not settings or not renderers.get("show_settings", False):
        return 28
    width = surface.get_width()
    y = 28
    for _title, entries in settings:
        x = 12
        for text in entries:
            label = font.render(str(text), True, (216, 226, 212))
            if x > 12 and x + label.get_width() > width - 12:
                x = 12
                y += 18
            surface.blit(label, (x, y))
            x += label.get_width() + 14
        y += 18
    return y + 6


def _dash(surface, rect, color):
    import pygame

    step = 8
    for x in range(rect.left, rect.right, step * 2):
        pygame.draw.line(surface, color, (x, rect.top), (min(rect.right, x + step), rect.top), 2)
        pygame.draw.line(
            surface, color, (x, rect.bottom - 1), (min(rect.right, x + step), rect.bottom - 1), 2
        )
    for y in range(rect.top, rect.bottom, step * 2):
        pygame.draw.line(surface, color, (rect.left, y), (rect.left, min(rect.bottom, y + step)), 2)
        pygame.draw.line(
            surface, color, (rect.right - 1, y), (rect.right - 1, min(rect.bottom, y + step)), 2
        )


def _swatch(surface, color, center, branch_id):
    import pygame

    if branch_id == 0:
        pygame.draw.circle(surface, (214, 222, 214), center, 5, 1)
    else:
        pygame.draw.circle(surface, color, center, 5)


def _mark(surface, font, origin, color, text, branch_id):
    _swatch(surface, color, (origin[0] + 6, origin[1] + 8), branch_id)
    if text:
        label = font.render(text, True, (232, 237, 226))
        surface.blit(label, (origin[0] + 16, origin[1]))


def draw_view(surface, packets, activity, *, renderers, boards, pending=(), count=128):
    import pygame
    from pvz_game.rendering import BoardRenderer

    from .live_view import ACTIVITY_TEXT, Activity

    surface.fill((18, 27, 24))
    width, height = surface.get_size()
    font, small = renderers.setdefault(
        "fonts", (pygame.font.SysFont("Segoe UI", 17), pygame.font.SysFont("Consolas", 15))
    )
    focus = renderers.get("focus")
    browse = renderers.setdefault("browse", {})
    regions = renderers["regions"] = {}
    previous = renderers.get("browse_keys", {})
    keys = {
        i: previous[i]
        if p["state"] == "selecting" and i in previous
        else (i, p["generation"], p["frame"].get("episode") if p["frame"] else None)
        for i, p in enumerate(packets)
    }
    renderers["browse_keys"] = keys
    valid = set(keys.values())
    for key in list(browse):
        if key not in valid:
            del browse[key]
    buttons = []
    surface.blit(
        small.render(ACTIVITY_TEXT[Activity(activity)], True, (225, 237, 226)), (12, 8)
    )
    if renderers.get("learning_settings"):
        mark = "v" if renderers.get("show_settings") else ">"
        glyph = small.render(mark, True, (216, 226, 212))
        toggle = glyph.get_rect(topright=(width - 12, 8))
        surface.blit(glyph, toggle)
        buttons.append((toggle, None, ("settings", None)))
    header_height = draw_learning_settings(surface, small, renderers)
    old_clip = surface.get_clip()
    indices = [focus] if focus is not None else list(range(4))
    for index in indices:
        cell_w, cell_h = (
            (width, height - header_height)
            if focus is not None
            else (width // 2, (height - header_height) // 2)
        )
        x, y = (
            (0, header_height)
            if focus is not None
            else (index % 2 * cell_w, header_height + index // 2 * cell_h)
        )
        rect = pygame.Rect(x + 4, y + 3, cell_w - 8, cell_h - 6)
        regions[index] = rect
        surface.set_clip(rect)
        pygame.draw.rect(surface, (34, 47, 40), rect, border_radius=5)
        if index >= len(packets):
            continue
        packet, frame = packets[index], packets[index]["frame"]
        gen = packet["generation"]
        key = keys[index]
        state = browse.setdefault(key, Browse()) if frame else None
        if frame:
            incoming = frame.get("history")
            if state.follow and incoming is not None:
                state.page = incoming
            elif incoming is not None and state.page is not None:
                state.page["total"] = incoming["total"]
        page = (state.page if state else None) or (frame or {}).get("history") or dict(
            start=0, total=0, rows=[]
        )
        decision = (
            state.selected
            if state and not state.follow and state.selected
            else (frame or {}).get("decision")
        )
        for label, offset, operation in (
            ("v", 52, "follow"),
            ("+", 96, "focus"),
            ("<>", 148, "switch"),
        ):
            r = pygame.Rect(x + cell_w - offset - 36, y + 8, 40, 24)
            following = operation == "follow" and state is not None and state.follow
            pygame.draw.rect(
                surface, (90, 122, 78) if following else (60, 99, 75), r, border_radius=4
            )
            glyph = small.render(label, True, (241, 243, 230))
            surface.blit(glyph, glyph.get_rect(center=r.center))
            buttons.append((r, index, (operation, gen)))
        switching = packet["state"] == "selecting" or index in pending
        finished = bool(frame) and frame.get("outcome") not in (None, "running")
        if switching:
            _dash(surface, rect, (212, 168, 74))
        elif finished:
            pygame.draw.rect(surface, (212, 168, 74), rect, width=2, border_radius=5)
        title = str(packet["env"]) if not frame else f"{packet['env']}  {frame['task']}"
        task = frame.get("task") if frame else None
        if frame and task != "prediction" and not is_placement_skill(task):
            reward = frame.get("reward")
            if isinstance(reward, (int, float)):
                title += f"  {reward:+.3g}{'~' if frame.get('reward_pending') else ''}"
        elif (
            task == "prediction"
            and decision
            and decision.get("effect_prediction_valid")
            and frame.get("observation") is not None
        ):
            rate = max(1, frame["observation"].tick_rate)
            title += f"  {decision.get('tick', 0) / rate:.1f}"
        title_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(x + 10, y + 8, max(1, cell_w - 196), 26))
        surface.blit(font.render(title, True, (233, 237, 224)), (x + 12, y + 10))
        surface.set_clip(title_clip)
        if not frame:
            continue
        recorded = scores_recorded(decision)
        widths = TABLE_WIDTHS if recorded else QUIET_WIDTHS
        state.horizontal = max(0, min(state.horizontal, horizontal_limit(cell_w - 24, widths)))
        extras = (44 if recorded else 22 if decision and decision.get("action") else 0) + (
            120 if page.get("total") else 0
        )
        board_h = max(85, cell_h - 48 - extras)
        outcome_w = min(360, (cell_w - 28) * 2 // 5)
        size = (max(1, cell_w - 28 - outcome_w), board_h)
        cache_key = (index, size)
        if cache_key not in renderers:
            # At most one board renderer per panel, plus ordinary UI state.
            for old in list(renderers):
                if isinstance(old, tuple) and old[0] == index:
                    del renderers[old]
            renderers[cache_key] = BoardRenderer(size=size)
        identity = (gen, frame["sequence"], frame.get("reward"), frame.get("reward_pending"), size)
        if index not in boards or boards[index][0] != identity:
            boards[index] = (
                identity,
                renderers[cache_key].render(frame["observation"], context=render_context(frame)),
            )
        surface.blit(boards[index][1], (x + 10, y + 42))
        draw_live_outcomes(
            surface, frame, pygame.Rect(x + 18 + size[0], y + 42, outcome_w, board_h)
        )
        obs = frame["observation"]
        top = y + 48 + board_h
        if recorded and decision:
            chosen = branch(decision["action"])
            legal = decision.get("legal", [q is not None for q in decision["q"]])
            explored = decision["action"] != decision.get("greedy_action")
            stride = max(44, (cell_w - 24) // 5)
            for k, (q, ok) in enumerate(zip(decision["q"], legal)):
                origin = (x + 12 + (k % 5) * stride, top + (k // 5) * 20)
                slot = pygame.Rect(origin[0], origin[1] - 1, stride - 4, 18)
                surface.set_clip(rect.clip(slot))
                if k == chosen:
                    pygame.draw.rect(
                        surface,
                        (120, 78, 36) if explored else (92, 78, 40),
                        slot,
                        border_radius=3,
                    )
                text = "" if q is None else f"{q:+.1f}"
                color = CARD_COLOR[k] if ok else (90, 98, 92)
                _mark(surface, small, origin, color, text, k)
                surface.set_clip(rect)
            top += 42
        elif decision and decision.get("action"):
            chosen_branch = branch(decision["action"])
            _mark(
                surface,
                small,
                (x + 12, top),
                CARD_COLOR[chosen_branch],
                tile_label(decision["action"]),
                chosen_branch,
            )
            top += 22
        if not page.get("total"):
            continue
        table_y = top + 6
        row_height = 22
        if table_y + row_height > y + cell_h - 8:
            continue
        visible = max(0, (y + cell_h - table_y - 8 - row_height) // row_height)
        rows = (
            (page["rows"][-visible:] if state.follow else page["rows"][:visible]) if visible else []
        )
        tx = x + 12 - state.horizontal
        for offset, row in enumerate(rows):
            ry = table_y + offset * row_height
            rejected = row.get("accepted") is False
            if state.selected and row["sequence"] == state.selected["sequence"]:
                pygame.draw.rect(surface, (70, 86, 56), (x + 8, ry, cell_w - 16, row_height))
            elif rejected:
                pygame.draw.rect(surface, (92, 48, 40), (x + 8, ry, cell_w - 16, row_height))
            cells = [
                str(row["sequence"]),
                f"{row['tick'] / obs.tick_rate:.1f}",
                branch(row["action"]),
                tile_label(row["action"]),
            ]
            if recorded:
                cells += [None if q is None else f"{q:+.1f}" for q in row["q"]]
            tx = x + 12 - state.horizontal
            for text, col_w in zip(cells, widths):
                surface.set_clip(rect.clip(pygame.Rect(tx, ry, col_w - 2, row_height)))
                if isinstance(text, int):
                    _swatch(surface, CARD_COLOR[text], (tx + 7, ry + 11), text)
                else:
                    surface.blit(small.render(text, True, (220, 231, 218)), (tx, ry + 2))
                tx += col_w
            surface.set_clip(rect)
            buttons.append(
                (pygame.Rect(x + 8, ry, cell_w - 16, row_height), index, ("row", gen, row))
            )
    surface.set_clip(old_clip)
    return buttons


def viewer_main(
    frames,
    commands,
    history_responses,
    errors,
    closed,
    ready,
    activity,
    size,
    count,
    settings=(),
    port_value=None,
):
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
        renderers["learning_settings"] = settings
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
                    elif event.key == pygame.K_f:
                        resume_visible_history(packets, renderers)
                    elif event.key == pygame.K_s:
                        renderers["show_settings"] = not renderers.get("show_settings", False)
                    dirty = True
                elif event.type == pygame.MOUSEWHEEL:
                    for index, rect in renderers.get("regions", {}).items():
                        if (
                            rect.collidepoint(pygame.mouse.get_pos())
                            and index < len(packets)
                            and packets[index]["frame"]
                        ):
                            p = packets[index]
                            key = renderers["browse_keys"][index]
                            state = renderers["browse"].setdefault(key, Browse())
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT or event.x:
                                limit = horizontal_limit(max(1, rect.width - 16))
                                state.horizontal = max(
                                    0, min(limit, state.horizontal - (event.x or event.y) * 60)
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
                        if operation == "settings":
                            renderers["show_settings"] = not renderers.get("show_settings", False)
                            dirty = True
                            break
                        p = packets[index]
                        if operation == "switch":
                            switch(index, generation)
                        elif operation == "focus":
                            renderers["focus"] = (
                                None if renderers.get("focus") is not None else index
                            )
                        elif p["frame"] and p["state"] != "selecting":
                            key = renderers["browse_keys"][index]
                            state = renderers["browse"].setdefault(key, Browse())
                            if operation == "follow":
                                resume_visible_history(packets, renderers, [index])
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
