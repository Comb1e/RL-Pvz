"""Frozen scenario families, separate from policy observations and selection."""

import hashlib
import random
from dataclasses import replace

from pvz_game import LevelSpec, Rules, Spawn, WaveSpec
from pvz_game.config import bundled, resolve_level


def namespace_seed(family: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"pvz-rl/v1/{family}/{seed}".encode()).digest()[:8], "big")


def scenario(level: str, family: str, seed: int, rules: Rules, cfg=None):
    if family == "preset":
        return level
    if family == "diagnostic":
        # A peashooter anywhere in row 2 can kill this basic zombie; no mower rescue.
        return LevelSpec(
            "diagnostic",
            (Spawn(5 * rules.game["tick_rate"], "basic", 2),),
            initial_sun=100,
            mowers=False,
        )
    if family in ("saving",):
        from pvz_rl.config import lesson_settings

        lesson = lesson_settings(cfg)[family]
        rng = random.Random(namespace_seed(family, seed))
        count = lesson.get("lanes_per_spawn", 1)
        # Preserve archived single-lane seed mappings exactly.
        lanes = (
            [rng.randrange(rules.game["rows"])]
            if count == 1
            else sorted(rng.sample(range(rules.game["rows"]), count))
        )
        return LevelSpec(
            family,
            tuple(Spawn(tick, "basic", lane) for tick in lesson["spawn_ticks"] for lane in lanes),
            initial_sun=lesson["initial_sun"],
            mowers=False,
        )
    if family not in ("redistributed", "faster", "concentrated"):
        raise ValueError(f"Unknown scenario family: {family}")
    raw = bundled("levels.toml")[level]
    spec = WaveSpec.from_dict({"name": f"{family}-{level}", **raw})
    rng = random.Random(namespace_seed(family, seed))
    if family == "redistributed":
        tail = [kind for wave in spec.waves[3:] for kind in wave]
        rng.shuffle(tail)
        waves, offset = list(spec.waves[:3]), 0
        for wave in spec.waves[3:]:
            waves.append(tuple(tail[offset : offset + len(wave)]))
            offset += len(wave)
        spec = replace(spec, waves=tuple(waves))
    if family == "faster":
        spec = replace(spec, wave_seconds=20)
    # Reuse the original engine seed so changed families can be paired with a preset.
    resolved = resolve_level(spec, rules, random.Random(seed))
    if family == "concentrated":
        lanes = sorted(rng.sample(range(rules.game["rows"]), 3))
        resolved = replace(
            resolved, spawns=tuple(replace(s, row=rng.choice(lanes)) for s in resolved.spawns)
        )
    return resolved


def difficulty_weights(cfg: dict, decisions: int, total: int, curriculum: bool) -> list[float]:
    c = cfg["curriculum"]
    if total is None:
        # Teaching selects its explicit task mix independently of a budget fraction.
        return c["final_weights"]
    progress = decisions / max(1, total)
    if not curriculum or progress >= c["boundaries"][1]:
        return c["final_weights"]
    return c["easy_weights"] if progress < c["boundaries"][0] else c["intermediate_weights"]
