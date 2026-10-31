"""Engine-authored tile exercises and private, rollout-verified bomb labels."""

import random
from dataclasses import replace

from pvz_game import LevelSpec, Place, Spawn, Status, Wait
from pvz_game.config import InitialPlant, resolve_level

from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.scratch_game import (
    EngineSnapshot,
    fork_game,
    restored_game,
    scratch_game,
    set_initial_mowers,
)
from pvz_rl.learning.deadline import check_deadline

SKILLS = ("peashooter", "cherry_bomb")


class BombGenerationFailure(RuntimeError):
    pass


def peashooter_level(rules, seed):
    return replace(
        resolve_level("easy", rules, random.Random(seed)),
        name="tile-peashooter",
        initial_sun=rules.game["initial_sun"],
        mowers=False,
    )


def bomb_hit(snapshot, tile, target_id, *, rules=None):
    """Credit only damage caused by this bomb, including its real fuse/geometry."""
    return _bomb_hit(restored_game(snapshot, rules), tile, target_id)


def _bomb_hit(game, tile, target_id):
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
    return _bomb_labels(restored_game(snapshot, rules), target_id, deadline=deadline)


def _bomb_labels(game, target_id, *, deadline=None):
    positive, absolute = [], []
    for tile in range(game.rules.game["rows"] * game.rules.game["cols"]):
        check_deadline(deadline)
        action = Place("cherry_bomb", *divmod(tile, game.rules.game["cols"]))
        if not game.validate_action(action).accepted:
            absolute.append(tile)
        elif _bomb_hit(fork_game(game), tile, target_id):
            positive.append(tile)
    return positive, absolute


def sample_bomb(cfg, seed, *, rules=None, deadline=None):
    for game, target_id, attempt in bomb_candidates(cfg, seed, rules=rules, deadline=deadline):
        positive, absolute = _bomb_labels(game, target_id, deadline=deadline)
        if positive:
            return dict(
                snapshot=EngineSnapshot.capture(game),
                target_id=target_id,
                feasible_tiles=positive,
                absolute_infeasible_tiles=absolute,
                generation_failures=attempt,
            )
    raise generation_failure(cfg, seed)


def generation_failure(cfg, seed):
    attempts = cfg["training"]["annotation"]["sampling_max_attempts"]
    return BombGenerationFailure(
        f"No solvable Cherry Bomb board after {attempts} attempts (seed {seed})"
    )


def bomb_candidates(cfg, seed, *, start=0, rules=None, deadline=None):
    """Engine-authored candidates in stable seed/attempt order, without labels."""
    for game, attempt in bomb_initials(cfg, seed, start=start, rules=rules, deadline=deadline):
        tick = game.rules.plants["cherry_bomb"]["initial_recharge_ticks"] + 2
        observation = game.step(Wait(), ticks=tick).observation
        targets = [z for z in observation.zombies if z.health > 0 and not z.headless]
        if observation.status == Status.RUNNING and len(targets) == 1:
            yield game, targets[0].id, attempt


def bomb_initials(cfg, seed, *, start=0, rules=None, deadline=None):
    """Validated initial boards, shared by CPU and CUDA natural cooldown advances."""
    game = scratch_game(rules)
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
        if attempt < start:
            continue
        game.reset(level, seed)
        # A separate stream leaves board proposals and gameplay RNG unchanged.
        mower_rng = random.Random(namespace_seed(f"tile-cherry-bomb-mowers/{attempt}", seed))
        ready = settings.get("bomb_mower_ready_probability", 0.0)
        set_initial_mowers(game, [row for row in range(rows) if mower_rng.random() < ready])
        yield game, attempt
