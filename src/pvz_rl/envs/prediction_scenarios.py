"""Seeded, engine-validated current frames for forecast-only training."""

import math
from random import Random

from pvz_game import LevelSpec, Spawn, Status, Wait
from pvz_game.config import InitialPlant
from pvz_game.engine import _Zombie

from pvz_rl.envs.motion_history import MotionHistory
from pvz_rl.envs.prediction_counts import SCENE_PROTOCOL
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game, scratch_game, set_initial_mowers
from pvz_rl.envs.tile_scenarios import prediction_settings


def _feasible_surviving_totals(live_range, unspawned_range):
    """Totals of on-field plus count-only zombies that fit both ranges."""
    live_lo, live_hi = live_range
    unspawned_lo, unspawned_hi = unspawned_range
    totals = []
    for total in range(max(live_lo + unspawned_lo, 1), live_hi + unspawned_hi + 1):
        low = max(live_lo, total - unspawned_hi)
        high = min(live_hi, total - unspawned_lo)
        if low <= high:
            totals.append(total)
    return totals


def _surviving_weights(totals, curve):
    """Relative exponential weights. The two curve points fix the slope."""
    first, first_weight, second, second_weight = curve
    log_first = math.log(first_weight)
    slope = (math.log(second_weight) - log_first) / (second - first)
    logs = [log_first + (total - first) * slope for total in totals]
    peak = max(logs)
    return [math.exp(log - peak) for log in logs]


def _draw_surviving_counts(rng, settings):
    """On-field and count-only counts whose sum follows the surviving curve."""
    totals = _feasible_surviving_totals(settings["live_zombies"], settings["unspawned_zombies"])
    if not totals:
        raise ValueError("Prediction scenes need a surviving zombie")
    weights = _surviving_weights(totals, settings["surviving_curve"])
    scale = 0.0
    for weight in weights:
        scale += weight
    mark = rng.random() * scale
    chosen = totals[-1]
    acc = 0.0
    for total, weight in zip(totals, weights, strict=True):
        acc += weight
        if mark < acc:
            chosen = total
            break
    live_lo, live_hi = settings["live_zombies"]
    unspawned_lo, unspawned_hi = settings["unspawned_zombies"]
    low = max(live_lo, chosen - unspawned_hi)
    high = min(live_hi, chosen - unspawned_lo)
    live = rng.randint(low, high)
    return live, chosen - live


def sample_hp_fraction(rng, settings):
    """Truncate by rejection, rather than concentrating mass at the bounds."""
    while True:
        value = rng.gauss(settings["mean"], settings["std"])
        if settings["min"] <= value <= settings["max"]:
            return value


def prediction_snapshot(cfg, seed, rules, *, skill="prediction"):
    settings = prediction_settings(cfg, skill)
    rng = Random(namespace_seed(SCENE_PROTOCOL, seed))
    g = rules.game
    plant_types, zombie_types = (cfg["environment"][key] for key in ("plants", "zombies"))
    plant_count = rng.randint(*settings["plants"])
    defeated = rng.randint(*settings["defeated_zombies"])
    live, unspawned = _draw_surviving_counts(rng, settings)
    tiles = sorted(rng.sample(range(g["rows"] * g["cols"]), plant_count))
    plants = tuple(InitialPlant(rng.choice(plant_types), *divmod(t, g["cols"])) for t in tiles)
    tick = 1
    live_spawns = sorted(
        (
            Spawn(
                tick,
                rng.choice(zombie_types),
                rng.randrange(g["rows"]),
                x=rng.randrange(max(0, g["house_x"] + 1), g["cols"] * g["units_per_tile"]),
            )
            for _ in range(live)
        ),
        key=lambda spawn: (spawn.row, spawn.x, spawn.zombie_type),
    )
    # Only physical history belongs to the engine ledger. Unspawned zombies have
    # no schedule, type, lane or RNG draws and never enter either continuation.
    spawns = [Spawn(tick, zombie_types[0], 0)] * defeated + live_spawns
    game = scratch_game(rules)
    game.reset(
        LevelSpec("tile-prediction", tuple(spawns), rng.randint(*settings["sun"]), plants),
        0,
    )
    ready = settings["ready_probability"]
    set_initial_mowers(game, [row for row in range(g["rows"]) if rng.random() < ready])
    game._tick, game._wave = tick, int(defeated + live > 0)
    game._spawn_index, game._defeated = defeated + live, defeated
    game._cooldowns = {
        kind: 0 if rng.random() < ready else rng.randint(1, rules.plants[kind]["recharge_ticks"])
        for kind in plant_types
    }
    # Engine-initialized gait and RNG in canonical public board order. Hidden
    # combat state does not draw from the scene sampler.
    for spawn in live_spawns:
        kind = spawn.zombie_type
        spec = rules.zombies[kind]
        pole = kind == "pole_vaulting"
        zombie = _Zombie(
            game._id(),
            kind,
            spawn.row,
            spawn.x,
            spec["health"],
            spec["armor"],
            "carrying_pole" if pole else "walking",
            has_pole=pole,
        )
        game._start_walk(zombie, pole=pole)
        game._zombies[zombie.id] = zombie
        maximum = spec["health"] + spec["armor"]
        remaining = max(
            1,
            min(maximum, round(maximum * sample_hp_fraction(rng, settings["zombie_hp_fraction"]))),
        )
        game._damage(zombie, maximum - remaining, 0)
    game._events.clear()
    # An unmoved frame has an empty ring. Eight waits make the summary real motion,
    # and the forecast fork is this warmed frame.
    motion = MotionHistory(cfg, rules)
    motion.sync(game.observe())
    for _ in range(motion.ticks):
        previous, history = game.snapshot(), motion.export()
        result = game.step(Wait(), ticks=1)
        if result.status == Status.LOST:
            # Keep the last pre-entry frame if a warmup ever reaches the house.
            return EngineSnapshot.capture(restored_game(previous, rules)), history, unspawned
        if result.ticks_advanced:
            motion.advance(result.observation, result.events)
        if result.status != Status.RUNNING:
            break
    return EngineSnapshot.capture(restored_game(game.snapshot(), rules)), motion.export(), unspawned
