"""Off-screen rendering from observations; no game state or display mutation."""

import numpy as np


def render_observation(obs):
    import pygame
    from pvz_game.ui import plant_art, zombie_art

    surface = pygame.Surface((1000, 600))
    surface.fill((20, 39, 36))
    x0, y0, tile = 55, 65, 96
    for row in range(obs.rows):
        for col in range(obs.cols):
            color = (92, 134, 67) if (row + col) % 2 else (101, 145, 73)
            pygame.draw.rect(surface, color, (x0 + col * tile, y0 + row * tile, tile - 1, tile - 1))
    for p in obs.plants:
        plant_art(
            surface,
            p.plant_type,
            x0 + (p.col + 0.5) * tile,
            y0 + (p.row + 0.5) * tile,
            state=p.state,
        )
    for z in obs.zombies:
        zombie_art(
            surface,
            z.zombie_type,
            x0 + z.x / obs.units_per_tile * tile,
            y0 + (z.row + 0.5) * tile,
            tick=obs.tick,
            state=z.state,
            armor=z.armor,
            slow=z.slow_ticks > 0,
        )
    for p in obs.projectiles:
        pygame.draw.circle(
            surface,
            (130, 220, 250) if p.icy else (160, 225, 90),
            (int(x0 + p.x / obs.units_per_tile * tile), int(y0 + (p.row + 0.5) * tile)),
            6,
        )
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
