"""A versioned, ID-free spatial representation of public observations.

Integer accumulation precedes scaling, making entity permutation immaterial.
Spatial aggregation deliberately trades exact entity identity for a compact input.
"""

import numpy as np
from gymnasium import spaces
from pvz_game import Observation, Rules


class ObservationEncoder:
    def __init__(self, cfg: dict, rules: Rules):
        env = cfg["environment"]
        self.cfg, self.rules = cfg, rules
        self.plants = {kind: i for i, kind in enumerate(env["plants"])}
        self.zombies = {kind: i for i, kind in enumerate(env["zombies"])}
        self.plant_states = {state: i for i, state in enumerate(env["plant_states"])}
        self.zombie_states = {state: i for i, state in enumerate(env["zombie_states"])}
        self.mower_states = {state: i for i, state in enumerate(env["mower_states"])}
        self.rows, self.cols, self.bins = env["rows"], env["cols"], env["bins"]
        self.plant_width = len(self.plants) + 2 + len(self.plant_states)
        self.zombie_width = len(self.zombies) + 6 + len(self.zombie_states)
        self.global_width = 10 + len(self.plants) * 3 + self.rows * 4
        sizes = [
            self.rows * self.cols * self.plant_width,
            self.rows * self.bins * self.zombie_width,
            self.rows * self.bins * 3,
            self.global_width,
        ]
        offsets = np.cumsum([0, *sizes])
        self.slices = dict(
            zip(
                ("plants", "zombies", "projectiles", "globals"),
                (slice(int(a), int(b)) for a, b in zip(offsets, offsets[1:])),
            )
        )
        self.size = int(offsets[-1])
        self.space = spaces.Box(-np.inf, np.inf, shape=(self.size,), dtype=np.float32)
        self.count_scale = cfg["encoding"]["count_scale"]
        self.timer_scale = max(
            v for d in rules.plants.values() for k, v in d.items() if k.endswith("_ticks")
        )
        self.hp_scale = max(z["health"] for z in rules.zombies.values())
        self.armor_scale = max(z["armor"] for z in rules.zombies.values())
        self.damage_scale = max(p.get("damage", 0) for p in rules.plants.values())
        self.position_scale = rules.game["spawn_x"] - rules.game["house_x"]
        self.cost_scale = max(p["cost"] for p in rules.plants.values())

    def bin_index(self, x: int) -> int:
        shifted = x - self.rules.game["house_x"]
        return max(0, min(self.bins - 1, shifted * self.bins // self.position_scale))

    def encode(self, obs: Observation) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float32)
        plant = result[self.slices["plants"]].reshape(self.rows, self.cols, self.plant_width)
        nplants, nzombies = len(self.plants), len(self.zombies)
        for p in obs.plants:
            tile = plant[p.row, p.col]
            tile[self.plants[p.plant_type]] = 1
            tile[nplants] = p.health / p.max_health
            tile[nplants + 1] = p.timer_ticks / self.timer_scale
            tile[nplants + 2 + self.plant_states[p.state]] = 1

        # All sums are exact integer operations; IDs and tuple order are never features.
        zombies = np.zeros((self.rows, self.bins, self.zombie_width), dtype=np.int64)
        for z in obs.zombies:
            cell = zombies[z.row, self.bin_index(z.x)]
            cell[self.zombies[z.zombie_type]] += 1
            cell[nzombies : nzombies + 6] += (
                z.health,
                z.armor,
                z.x,
                z.slow_ticks,
                int(z.has_pole),
                z.timer_ticks,
            )
            cell[nzombies + 6 + self.zombie_states[z.state]] += 1
        zscale = np.full(self.zombie_width, self.count_scale, dtype=np.float32)
        zscale[nzombies : nzombies + 6] *= (
            self.hp_scale,
            max(1, self.armor_scale),
            self.position_scale,
            self.timer_scale,
            1,
            self.timer_scale,
        )
        result[self.slices["zombies"]] = (zombies / zscale).ravel()
        projectiles = np.zeros((self.rows, self.bins, 3), dtype=np.int64)
        for p in obs.projectiles:
            projectiles[p.row, self.bin_index(p.x)] += (1, p.damage, int(p.icy))
        result[self.slices["projectiles"]] = (
            projectiles / (self.count_scale * np.array([1, self.damage_scale, 1]))
        ).ravel()
        counts = obs.counts
        values = [
            obs.sun / self.rules.game["sun_cap"],
            obs.elapsed_seconds / self.cfg["environment"]["cutoff_seconds"],
            obs.wave / self.cfg["encoding"]["wave_scale"],
            obs.total_waves / self.cfg["encoding"]["wave_scale"],
        ]
        values.extend(
            x / self.count_scale
            for x in (
                counts.initial_total,
                counts.spawned,
                counts.alive,
                counts.defeated,
                counts.not_yet_spawned,
                counts.remaining,
            )
        )
        cards = {c.plant_type: c for c in obs.cards}
        for kind in self.plants:
            card = cards[kind]
            values.extend(
                (
                    card.cost / self.cost_scale,
                    card.cooldown_ticks / self.timer_scale,
                    card.recharge_ticks / self.timer_scale,
                )
            )
        for mower in sorted(obs.mowers, key=lambda m: m.row):
            values.extend(
                (
                    mower.x / self.position_scale,
                    *(float(mower.state == state) for state in self.mower_states),
                )
            )
        result[self.slices["globals"]] = values
        return result
