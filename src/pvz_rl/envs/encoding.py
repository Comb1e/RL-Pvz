"""A versioned, ID-free spatial representation of public observations.

Integer accumulation precedes scaling, making entity permutation immaterial.
Spatial aggregation deliberately trades exact entity identity for a compact input.
"""

import numpy as np
from gymnasium import spaces
from pvz_game import Observation, Rules

# Internal animation phases share compact public behavior categories.
PLANT_BEHAVIOR = {
    "ready": "ready",
    "arming": "arming",
    "rising": "arming",
    "armed": "armed",
    "fusing": "fusing",
    "digesting": "digesting",
    "biting": "digesting",
    "biting_got_one": "digesting",
    "recovering": "digesting",
    "exploding": "exploding",
    "detonating": "detonating",
}


class ObservationEncoder:
    def __init__(self, cfg: dict, rules: Rules):
        env = cfg["environment"]
        self.cfg, self.rules = cfg, rules
        self.plants = {kind: i for i, kind in enumerate(env["plants"])}
        self.zombies = {kind: i for i, kind in enumerate(env["zombies"])}
        self.plant_states = {state: i for i, state in enumerate(env["plant_states"])}
        self.rows, self.cols, self.bins = env["rows"], env["cols"], env["bins"]
        if cfg["encoding"]["version"] != "event_v7":
            raise ValueError("Only event_v7 observations are supported")
        self.plant_width = 3
        # Per lane-region: one count per zombie type, aggregate health and armor,
        # nearest zombie distance, and nearest carrier of an unused pole.
        self.zombie_fields = {
            name: i
            for i, name in enumerate((*self.zombies, "health", "armor", "nearest", "nearest_pole"))
        }
        self.zombie_width = len(self.zombie_fields)
        self.empty_distance = -1.0
        self.global_fields = {
            name: i
            for i, name in enumerate(
                ("sun", "elapsed", "wave", "total_waves", "initial", "defeated")
                + tuple(f"mower_spent_{r}" for r in range(self.rows))
            )
        }
        self.global_width = len(self.global_fields)
        sizes = [
            self.rows * self.cols * self.plant_width,
            self.rows * self.bins * self.zombie_width,
            self.global_width,
            self.rows,
        ]
        names = ["plants", "zombies", "globals", "headless"]
        offsets = np.cumsum([0, *sizes])
        self.slices = dict(
            zip(
                names,
                (slice(int(a), int(b)) for a, b in zip(offsets, offsets[1:])),
            )
        )
        self.size = int(offsets[-1])
        self.space = spaces.Box(-np.inf, np.inf, shape=(self.size,), dtype=np.float32)
        self.count_scale = cfg["encoding"]["count_scale"]
        self.local_count_scale = cfg["encoding"]["local_count_scale"]
        self.hp_scale = max(z["health"] for z in rules.zombies.values())
        self.armor_scale = max(z["armor"] for z in rules.zombies.values())
        self.position_scale = rules.game["spawn_x"] - rules.game["house_x"]
        self.cost_scale = max(p["cost"] for p in rules.plants.values())

    def bin_index(self, x: int) -> int:
        shifted = x - self.rules.game["house_x"]
        return max(0, min(self.bins - 1, shifted * self.bins // self.position_scale))

    def encode(self, obs: Observation) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float32)
        plant = result[self.slices["plants"]].reshape(self.rows, self.cols, self.plant_width)
        for p in obs.plants:
            tile = plant[p.row, p.col]
            tile[:] = (
                self.plants[p.plant_type] + 1,
                p.health / p.max_health,
                self.plant_states[PLANT_BEHAVIOR[p.state]] + 1,
            )

        # All sums are exact integer operations; IDs and tuple order are never features.
        shape = (self.rows, self.bins)
        zombies = np.zeros((*shape, self.zombie_width), dtype=np.int64)
        nearest = np.full((*shape, 2), np.iinfo(np.int64).max, dtype=np.int64)
        fields = self.zombie_fields
        for z in obs.zombies:
            r, b = z.row, self.bin_index(z.x)
            nearest[r, b, 0] = min(nearest[r, b, 0], z.x)
            cell = zombies[r, b]
            cell[self.zombies[z.zombie_type]] += 1
            cell[fields["health"]] += z.health
            cell[fields["armor"]] += z.armor
            if z.has_pole and not z.headless:
                nearest[r, b, 1] = min(nearest[r, b, 1], z.x)
        zscale = np.full(self.zombie_width, self.local_count_scale, dtype=np.float32)
        zscale[fields["health"]] *= self.hp_scale
        zscale[fields["armor"]] *= max(1, self.armor_scale)
        encoded = zombies / zscale
        # The negative sentinel distinguishes absence from a carrier at spawn_x.
        encoded[:, :, fields["nearest"] : fields["nearest_pole"] + 1] = np.where(
            nearest == np.iinfo(np.int64).max,
            self.empty_distance,
            (nearest.astype(np.float64) - self.rules.game["house_x"]) / self.position_scale,
        )
        result[self.slices["zombies"]] = encoded.ravel()
        counts = obs.counts
        values = {
            "sun": obs.sun / self.cost_scale,
            "elapsed": obs.elapsed_seconds / self.cfg["environment"]["cutoff_seconds"],
            "wave": obs.wave / self.cfg["encoding"]["wave_scale"],
            "total_waves": obs.total_waves / self.cfg["encoding"]["wave_scale"],
            "initial": counts.initial_total / self.count_scale,
            "defeated": counts.defeated / self.count_scale,
        }
        values.update({f"mower_spent_{m.row}": float(m.state == "spent") for m in obs.mowers})
        globals_ = result[self.slices["globals"]]
        for name, value in values.items():
            globals_[self.global_fields[name]] = value
        for row in range(self.rows):
            front = min(
                (z for z in obs.zombies if z.row == row),
                key=lambda z: (z.x, z.headless),
                default=None,
            )
            result[self.slices["headless"].start + row] = bool(front and front.headless)
        return result
