"""Engine-authored tile exercises and private, rollout-verified bomb labels."""

import random
from dataclasses import replace

from pvz_game import Game, LevelSpec, Place, Spawn, Status, Wait
from pvz_game.config import InitialPlant, resolve_level

from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.learning.deadline import check_deadline

SKILLS = ("peashooter", "cherry_bomb")


def peashooter_level(rules, seed):
    return replace(
        resolve_level("easy", rules, random.Random(seed)),
        name="tile-peashooter",
        initial_sun=rules.game["initial_sun"],
        mowers=False,
    )


def bomb_hit(snapshot, tile, target_id, *, rules=None):
    """Credit only damage caused by this bomb, including its real fuse/geometry."""
    game = Game(rules)
    game.restore(snapshot)
    row, col = divmod(tile, game.rules.game["cols"])
    action = Place("cherry_bomb", row, col)
    if not game.validate_action(action).accepted:
        return False
    result = game.step(action, ticks=game.rules.plants["cherry_bomb"]["first_ticks"] + 1)
    source = next(
        event.entity_id
        for event in result.events
        if event.kind == "PlantPlaced" and event.get("plant_type") == "cherry_bomb"
    )
    return any(
        event.kind == "DamageApplied"
        and event.entity_id == target_id
        and event.get("source") == source
        and event.get("health_damage", 0) + event.get("armor_damage", 0) > 0
        for event in result.events
    )


def bomb_labels(snapshot, target_id, *, rules=None, deadline=None):
    game = Game(rules)
    game.restore(snapshot)
    positive, absolute = [], []
    for tile in range(game.rules.game["rows"] * game.rules.game["cols"]):
        check_deadline(deadline)
        action = Place("cherry_bomb", *divmod(tile, game.rules.game["cols"]))
        if not game.validate_action(action).accepted:
            absolute.append(tile)
        elif bomb_hit(snapshot, tile, target_id, rules=game.rules):
            positive.append(tile)
    return positive, absolute


def sample_bomb(cfg, seed, *, rules=None, deadline=None):
    """Generate a legal single-decision exercise; never edit simulator internals."""
    game = Game(rules)
    rules = game.rules
    settings = cfg["training"]["tile_scenarios"]
    attempts = cfg["training"]["annotation"]["sampling_max_attempts"]
    rng = random.Random(namespace_seed("tile-cherry-bomb", seed))
    rows, cols = rules.game["rows"], rules.game["cols"]
    kinds = [kind for kind in cfg["environment"]["plants"] if kind != "cherry_bomb"]
    tick = rules.plants["cherry_bomb"]["initial_recharge_ticks"] + 2
    if cfg["environment"]["cutoff_seconds"] * rules.game["tick_rate"] <= tick:
        raise ValueError(
            "Cherry Bomb exercises need a cutoff longer than the initial card cooldown"
        )
    for attempt in range(attempts):
        check_deadline(deadline)
        count = rng.randint(*settings["bomb_plants"])
        tiles = rng.sample(range(rows * cols), count)
        plants = tuple(InitialPlant(rng.choice(kinds), *divmod(tile, cols)) for tile in tiles)
        zombie = rng.choice(cfg["environment"]["zombies"])
        row = rng.randrange(rows)
        x = rng.randrange(cols * rules.game["units_per_tile"])
        level = LevelSpec(
            "tile-cherry-bomb",
            (Spawn(tick, zombie, row, x=x),),
            initial_sun=settings["bomb_initial_sun"],
            plants=plants,
            mowers=False,
        )
        game.reset(level, seed)
        observation = game.step(Wait(), ticks=tick).observation
        targets = [z for z in observation.zombies if z.health > 0 and not z.headless]
        if observation.status != Status.RUNNING or len(targets) != 1:
            continue
        snapshot = game.snapshot()
        positive, absolute = bomb_labels(snapshot, targets[0].id, rules=rules, deadline=deadline)
        if positive:
            return dict(
                snapshot=snapshot,
                target_id=targets[0].id,
                feasible_tiles=positive,
                absolute_infeasible_tiles=absolute,
                generation_failures=attempt,
            )
    raise RuntimeError(f"No solvable Cherry Bomb board after {attempts} attempts (seed {seed})")
