"""Seeded, engine-validated current frames for forecast-only training."""

from random import Random

from pvz_game import LevelSpec, Spawn
from pvz_game.config import InitialPlant
from pvz_game.engine import _Zombie

from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game, scratch_game, set_initial_mowers
from pvz_rl.envs.tile_scenarios import prediction_settings


def prediction_snapshot(cfg, seed, rules):
    settings = prediction_settings(cfg)
    rng = Random(namespace_seed("prediction-frame-v1", seed))
    g = rules.game
    plant_types, zombie_types = (cfg["environment"][key] for key in ("plants", "zombies"))
    plant_count = rng.randint(*settings["plants"])
    live = rng.randint(*settings["live_zombies"])
    defeated = rng.randint(*settings["defeated_zombies"])
    pending = rng.randint(*settings["pending_zombies"])
    if not live and not pending:
        if settings["pending_zombies"][1]:
            pending = 1
        else:
            live = 1
    tiles = sorted(rng.sample(range(g["rows"] * g["cols"]), plant_count))
    plants = tuple(InitialPlant(rng.choice(plant_types), *divmod(t, g["cols"])) for t in tiles)
    tick = 1
    live_spawns = sorted(
        (
            Spawn(
                tick,
                rng.choice(zombie_types),
                rng.randrange(g["rows"]),
                x=rng.randint(max(0, g["house_x"] + 1), g["spawn_x"]),
            )
            for _ in range(live)
        ),
        key=lambda spawn: (spawn.row, spawn.x, spawn.zombie_type),
    )
    # The ledger represents already-defeated, live, and future zombies. Future
    # entries supply the public total but are suppressed in every target fork.
    spawns = [Spawn(tick, zombie_types[0], 0)] * defeated + live_spawns
    spawns.extend(Spawn(tick + offset + 1, zombie_types[0], 0) for offset in range(pending))
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
    return EngineSnapshot.capture(restored_game(game.snapshot(), rules))
