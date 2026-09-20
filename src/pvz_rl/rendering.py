"""Offscreen public-state renderer shared by the Gym adapter and video exports."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _font(size):
    import pygame

    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(None, size)


def render_observation(obs, *, context=None):
    import pygame
    from pvz_game.ui import plant_art, zombie_art

    context = context or {}
    surface = pygame.Surface((1000, 600))
    surface.fill((20, 39, 36))

    def label(text, x, y, size=22, color=(228, 239, 226)):
        surface.blit(_font(size).render(str(text), True, color), (x, y))

    level = context.get("level", obs.level)
    if context.get("family") == "diagnostic":
        level = f"diagnostic / {level}"
    seed = f" | seed {context['seed']}" if "seed" in context else ""
    label(f"PVZ  /  {level}{seed}", 20, 12, 26)
    label(f"Time {obs.elapsed_seconds:6.1f}s    Sun {obs.sun}", 440, 15)
    label(f"Wave {obs.wave}/{obs.total_waves}", 790, 15)
    label(
        f"Zombies: {obs.counts.alive} alive / {obs.counts.defeated} defeated / "
        f"{obs.counts.remaining} remaining",
        20,
        41,
        20,
    )
    ready_mowers = sum(m.state == "ready" for m in obs.mowers)
    label(f"Ready mowers {ready_mowers}/{obs.rows}", 790, 41, 20)
    for i, card in enumerate(obs.cards):
        x = 20 + i * 122
        available = card.cooldown_ticks == 0 and obs.sun >= card.cost
        color = (60, 99, 70) if available else (47, 61, 59)
        pygame.draw.rect(surface, color, (x, 70, 116, 60), border_radius=5)
        label(card.plant_type.replace("_", " "), x + 6, 77, 18)
        label(f"Sun {card.cost}", x + 6, 96, 18)
        cooldown = card.cooldown_ticks / obs.tick_rate
        label(f"{cooldown:.1f}s" if cooldown else "Ready", x + 66, 112, 16)

    x0, y0, tile = 75, 150, 84
    for row in range(obs.rows):
        label(row + 1, 54, y0 + row * tile + 32, 18)
        for col in range(obs.cols):
            color = (92, 134, 67) if (row + col) % 2 else (101, 145, 73)
            pygame.draw.rect(surface, color, (x0 + col * tile, y0 + row * tile, tile - 1, tile - 1))
    for col in range(obs.cols):
        label(col + 1, x0 + col * tile + 36, 133, 16)
    for mower in obs.mowers:
        if mower.state != "spent":
            x = x0 + mower.x / obs.units_per_tile * tile
            y = y0 + (mower.row + 0.5) * tile
            pygame.draw.rect(surface, (221, 87, 70), (x - 16, y - 10, 30, 20), border_radius=3)
            pygame.draw.circle(surface, (30, 35, 33), (int(x - 9), int(y + 11)), 5)
            pygame.draw.circle(surface, (30, 35, 33), (int(x + 9), int(y + 11)), 5)
    for plant in obs.plants:
        x, y = x0 + (plant.col + 0.5) * tile, y0 + (plant.row + 0.5) * tile
        plant_art(surface, plant.plant_type, x, y, state=plant.state)
        pygame.draw.rect(surface, (45, 50, 43), (x - 20, y + 32, 40, 4))
        pygame.draw.rect(
            surface, (177, 231, 127), (x - 20, y + 32, 40 * plant.health / plant.max_health, 4)
        )
    for zombie in obs.zombies:
        zombie_art(
            surface,
            zombie.zombie_type,
            x0 + zombie.x / obs.units_per_tile * tile,
            y0 + (zombie.row + 0.5) * tile,
            tick=obs.tick,
            state=zombie.state,
            armor=zombie.armor,
            slow=zombie.slow_ticks > 0,
        )
    for projectile in obs.projectiles:
        pygame.draw.circle(
            surface,
            (130, 220, 250) if projectile.icy else (160, 225, 90),
            (
                int(x0 + projectile.x / obs.units_per_tile * tile),
                int(y0 + (projectile.row + 0.5) * tile),
            ),
            6,
        )
    label(context.get("action", "Waiting for the next decision"), 20, 578, 19)
    outcome = context.get("outcome", obs.status.value)
    if outcome in ("won", "lost", "truncated"):
        pygame.draw.rect(surface, (20, 39, 36), (350, 284, 300, 84), border_radius=8)
        label(outcome.upper(), 405, 305, 42)
        if outcome == "truncated":
            label("External time limit", 425, 344, 18)
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
