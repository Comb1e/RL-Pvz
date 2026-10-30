"""Public-only projection and approximate conditional worlds for the search teacher.

No factual snapshot, seed, schedule or hidden countdown enters this interface.
The prior is deliberately a valid-state approximation, not a history posterior.
"""

import tomllib
from collections import Counter
from importlib.resources import files

import numpy as np
from pvz_game import Dig, Game, LevelSpec, Rules
from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES, Spawn, WaveSpec
from pvz_game.cuda import schema as S
from pvz_game.mechanics import MOVE_DENOMINATOR, SCALE, SPEC, attack_contact
from pvz_game.types import (
    CardView,
    MowerView,
    Observation,
    PlantView,
    ProjectileView,
    Status,
    ZombieCounts,
    ZombieView,
)

from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.sky_sun import scale_sky_delay

# Exhaustive ownership for the pinned snapshot contract. New engine fields must
# be classified deliberately; reset() must never leave an unnoticed latent.
WORLD_FIELDS = {
    "snapshot": {
        "public": "tick status sun cooldowns spawn_index defeated wave plants zombies projectiles mowers",
        "derived": "snapshot_version engine_version rules rules_hash next_id",
        "prior": "level seed rng_state gameplay_rng sky_due sky_drops",
    },
    "plants": {
        "public": "id kind row col health state",
        "derived": "due",
        "prior": "burst_due",
    },
    "zombies": {
        "public": "id kind row x health armor state has_pole headless",
        "derived": "slow_until vault_until target_id",
        "prior": "landing_x bite_progress move_remainder age speed pole_speed gait gait_phase phase_remainder vault_start_x vault_start_tick",
    },
    "projectiles": {
        "public": "id row x damage icy",
        "derived": "",
        "prior": "move_remainder",
    },
    "mowers": {
        "public": "row x state",
        "derived": "",
        "prior": "move_remainder chomp_ticks",
    },
}


def validate_world_fields(raw):
    for kind, groups in WORLD_FIELDS.items():
        expected = {key for fields in groups.values() for key in fields.split()}
        for record in [raw] if kind == "snapshot" else raw[kind]:
            if set(record) != expected:
                raise RuntimeError(
                    f"Unclassified pinned-engine {kind} fields: {set(record) ^ expected}"
                )


def public_observations(batch, levels, indices=None):
    """Project *only* public columns, in one bounded transfer per entity group."""
    indices = list(range(batch.n)) if indices is None else list(indices)
    keys = "tick status sun spawn_index defeated wave total_waves total_spawns np nz nq".split()
    header = batch.header[indices][:, [S.HEADER.index(k) for k in keys]].get()
    cd = batch.cooldowns[indices].get()
    specs = (
        ("plants", S.PLANT, "id kind row col health state due".split(), 8),
        (
            "zombies",
            S.ZOMBIE,
            "id kind row x health armor state slow_until has_pole vault_until bite_progress headless".split(),
            9,
        ),
        ("projectiles", S.PROJECTILE, "id row x damage icy".split(), 10),
        ("mowers", S.MOWER, "row x state".split(), None),
    )
    arrays = {}
    for name, schema, columns, count in specs:
        cap = 5 if count is None else int(header[:, count].max(initial=0))
        arrays[name] = getattr(batch, name)[indices, :cap][
            :, :, [schema.index(k) for k in columns]
        ].get()
    g, rules = batch.rules.game, batch.rules
    result = []
    for j, index in enumerate(indices):
        h = dict(zip(keys, map(int, header[j])))
        t = h["tick"]
        plants = tuple(
            PlantView(
                int(i),
                PLANT_TYPES[k],
                int(r),
                int(c),
                int(hp),
                rules.plants[PLANT_TYPES[k]]["health"],
                S.PLANT_STATES[st],
                max(0, int(due) - t),
            )
            for i, k, r, c, hp, st, due in arrays["plants"][j, : h["np"]]
        )
        zombies = []
        for i, k, r, x, hp, armor, st, slow, pole, vault, bite, headless in arrays["zombies"][
            j, : h["nz"]
        ]:
            state = S.ZOMBIE_STATES[st]
            timer = (
                max(0, int(vault) - t)
                if state == "vaulting"
                else (
                    (max(0, g["bite_ticks"] * 2 - int(bite)) + (0 if t < slow else 1))
                    // (1 if t < slow else 2)
                    if state == "biting"
                    else 0
                )
            )
            zombies.append(
                ZombieView(
                    int(i),
                    ZOMBIE_TYPES[k],
                    int(r),
                    int(x),
                    int(hp),
                    int(armor),
                    state,
                    max(0, int(slow) - t),
                    bool(pole),
                    timer,
                    bool(headless),
                )
            )
        result.append(
            Observation(
                1,
                t,
                t / g["tick_rate"],
                g["tick_rate"],
                g["units_per_tile"],
                g["rows"],
                g["cols"],
                str(levels[index]),
                Status(S.STATUSES[h["status"]]),
                h["sun"],
                tuple(
                    CardView(
                        k, rules.plants[k]["cost"], int(cd[j, n]), rules.plants[k]["recharge_ticks"]
                    )
                    for n, k in enumerate(PLANT_TYPES)
                ),
                plants,
                tuple(zombies),
                tuple(
                    ProjectileView(int(i), int(r), int(x), int(d), bool(ice))
                    for i, r, x, d, ice in arrays["projectiles"][j, : h["nq"]]
                ),
                tuple(
                    MowerView(int(r), int(x), S.MOWER_STATES[st])
                    for r, x, st in arrays["mowers"][j]
                ),
                h["wave"],
                h["total_waves"],
                ZombieCounts(
                    h["total_spawns"],
                    h["spawn_index"],
                    sum(not z.headless for z in zombies),
                    h["defeated"],
                    h["total_spawns"] - h["spawn_index"],
                    h["total_spawns"] - h["defeated"],
                ),
            )
        )
    return result


def legal_actions(obs, cfg, *, allowed=255, digging=True, peashooter=False):
    codec = ActionCodec(cfg)
    occupied = {(p.row, p.col) for p in obs.plants}
    legal = [0]
    for n, card in enumerate(obs.cards):
        if not allowed & (1 << n) or card.cooldown_ticks or obs.sun < card.cost:
            continue
        if peashooter and card.plant_type != "peashooter":
            continue
        legal.extend(
            1 + n * 45 + r * 9 + c for r in range(5) for c in range(9) if (r, c) not in occupied
        )
    if digging and not peashooter:
        legal.extend(codec.encode(Dig(r, c)) for r, c in sorted(occupied))
    return np.asarray(legal, dtype=np.int64)


class WorldGenerationError(RuntimeError):
    pass


class PublicWorldSampler:
    protocol = "public_board_prior_v1"

    def __init__(self, cfg, rules=None):
        self.cfg, self.rules = cfg, rules or Rules()
        self.presets = tomllib.loads(
            files("pvz_game.data").joinpath("levels.toml").read_text("utf-8")
        )
        self.failures = 0

    def sample(self, obs, level, rng, *, rate=1.0):
        if level not in self.presets:
            raise ValueError(f"Public search prior does not support scenario {level!r}")
        for _ in range(self.cfg["training"]["annotation"]["sampling_max_attempts"]):
            try:
                raw = self._sample(obs, level, rng, rate)
                validate_world_fields(raw)
                game = Game(self.rules)
                reconstructed = game.restore(raw)
                if reconstructed != obs:
                    raise ValueError("Public world reconstruction changed the observation")
                return raw
            except (ValueError, AssertionError):
                self.failures += 1
        raise WorldGenerationError("Public-board prior exhausted its sampling-attempt limit")

    def _sample(self, obs, level, rng, rate):
        g, t = self.rules.game, obs.tick
        spec = WaveSpec.from_dict({"name": level, **self.presets[level]})
        counts = [len(w) for w in spec.waves]
        assert sum(counts) == obs.counts.initial_total
        remaining = obs.counts.spawned
        spawns = []
        for w, kinds in enumerate(spec.waves):
            start = round((spec.preparation_seconds + w * spec.wave_seconds) * g["tick_rate"])
            end = start + round(spec.jitter_seconds * g["tick_rate"])
            past = min(remaining, len(kinds))
            remaining -= past
            shuffled = list(kinds)
            rng.shuffle(shuffled)
            for j, kind in enumerate(shuffled):
                lo, hi = (start, min(t, end)) if j < past else (max(t + 1, start), end)
                if hi < lo:
                    raise ValueError("Wave counts incompatible with public elapsed time")
                spawns.append(
                    Spawn(int(rng.integers(lo, hi + 1)), kind, int(rng.integers(5)), w + 1)
                )
        past_kinds = Counter(s.zombie_type for s in spawns if s.tick <= t)
        visible = Counter(z.zombie_type for z in obs.zombies)
        if visible - past_kinds:
            raise ValueError("Visible roster incompatible with sampled past slots")
        initial = LevelSpec(
            obs.level,
            tuple(spawns),
            initial_sun=obs.sun,
            mowers=any(m.state != "spent" for m in obs.mowers),
        )
        game = Game(self.rules)
        # This fresh seed is oracle-owned, never the factual game seed.
        game.reset(initial, int(rng.integers(2**31)))
        raw = game.snapshot()
        raw.update(
            tick=t,
            sun=obs.sun,
            status=obs.status.value,
            spawn_index=obs.counts.spawned,
            defeated=obs.counts.defeated,
            wave=obs.wave,
            cooldowns={c.plant_type: c.cooldown_ticks for c in obs.cards},
            gameplay_rng=int(rng.integers(1, 2**32)),
        )
        due = scale_sky_delay(
            int(rng.integers(g["sky_first_min_ticks"], g["sky_first_max_ticks"] + 1)), rate
        )
        drops = 0
        while due <= t:
            drops += 1
            delay = min(
                g["sky_interval_max_ticks"],
                g["sky_interval_base_ticks"] + drops * g["sky_interval_increment_ticks"],
            ) + int(rng.integers(g["sky_interval_jitter_ticks"] + 1))
            due += scale_sky_delay(delay, rate)
        raw.update(sky_due=due, sky_drops=drops)
        raw["plants"] = []
        for p in obs.plants:
            # Public due/state; only repeater burst phase is latent.
            burst = self.rules.plants[p.plant_type].get("burst_ticks", 0)
            raw["plants"].append(
                dict(
                    id=p.id,
                    kind=p.plant_type,
                    row=p.row,
                    col=p.col,
                    health=p.health,
                    state=p.state,
                    due=t + p.timer_ticks,
                    burst_due=t + int(rng.integers(1, burst + 1))
                    if burst and rng.integers(2)
                    else 0,
                )
            )
        raw["zombies"] = []
        for z in obs.zombies:
            r = self.rules.zombies[z.zombie_type]
            pole_speed = int(
                rng.integers(
                    r.get("pole_speed", r["speed"]), r.get("pole_max_speed", r["max_speed"]) + 1
                )
            )
            speed = pole_speed if z.has_pole else int(rng.integers(r["speed"], r["max_speed"] + 1))
            gait = (
                2
                if z.has_pole
                else 3
                if z.zombie_type == "pole_vaulting"
                else (1 if z.zombie_type == "flag" else int(rng.integers(2)))
            )
            candidates = [
                b
                for b in range(g["bite_ticks"] * 2)
                if (max(0, g["bite_ticks"] * 2 - b) + (0 if z.slow_ticks else 1))
                // (1 if z.slow_ticks else 2)
                == z.timer_ticks
            ]
            bite = int(rng.choice(candidates)) if z.state == "biting" and candidates else 0
            target = next(
                (
                    p.id
                    for p in obs.plants
                    if p.row == z.row
                    and attack_contact(p.col, z.x, g["units_per_tile"], z.has_pole)
                ),
                0,
            )
            start_x, landing_x, start_tick = z.x, z.x, t
            if z.state == "vaulting":
                duration = (SPEC["jump_frames"] * g["tick_rate"] + SPEC["jump_fps"] - 1) // SPEC[
                    "jump_fps"
                ]
                elapsed = duration - z.timer_ticks
                fraction = min(elapsed * 24, 4300)
                possible = []
                for col in range(g["cols"]):
                    landing = col * g["units_per_tile"] + 116 * g["units_per_tile"] // 80
                    if fraction == 4300:
                        if z.x == landing:
                            possible.append((max(z.x, g["spawn_x"]), landing))
                        continue
                    estimate = (z.x * 4300 - landing * fraction) // (4300 - fraction)
                    for start in range(estimate - 2, estimate + 3):
                        if (
                            0 <= start <= g["spawn_x"]
                            and start - (start - landing) * fraction // 4300 == z.x
                        ):
                            possible.append((start, landing))
                if not possible:
                    raise ValueError("Public vault pose has no valid latent trajectory")
                start_x, landing_x = possible[int(rng.integers(len(possible)))]
                start_tick = t - elapsed
            raw["zombies"].append(
                dict(
                    id=z.id,
                    kind=z.zombie_type,
                    row=z.row,
                    x=z.x,
                    health=z.health,
                    armor=z.armor,
                    state=z.state,
                    slow_until=t + z.slow_ticks,
                    has_pole=z.has_pole,
                    vault_until=t + z.timer_ticks if z.state == "vaulting" else 0,
                    landing_x=landing_x,
                    bite_progress=bite,
                    move_remainder=int(rng.integers(MOVE_DENOMINATOR)),
                    target_id=target,
                    headless=z.headless,
                    age=int(rng.integers(g["bite_ticks"] * 2)),
                    speed=speed,
                    pole_speed=pole_speed,
                    gait=gait,
                    gait_phase=int(rng.integers(SCALE)),
                    phase_remainder=int(rng.integers(4700)),
                    vault_start_x=start_x,
                    vault_start_tick=start_tick,
                )
            )
        raw["projectiles"] = [
            dict(
                id=p.id,
                row=p.row,
                x=p.x,
                damage=p.damage,
                icy=p.icy,
                move_remainder=int(rng.integers(g["tick_rate"] * g["slow_denominator"])),
            )
            for p in obs.projectiles
        ]
        raw["mowers"] = [
            dict(
                row=m.row,
                x=m.x,
                state=m.state,
                move_remainder=int(rng.integers(g["tick_rate"] * g["slow_denominator"])),
                chomp_ticks=int(rng.integers(g.get("mower_hit_ticks", 1))),
            )
            for m in obs.mowers
        ]
        raw["next_id"] = 1 + max(
            [0]
            + [p.id for p in obs.plants]
            + [z.id for z in obs.zombies]
            + [p.id for p in obs.projectiles]
        )
        return raw
