"""Engine-authored tile exercises and private, rollout-verified bomb labels."""

import random
from dataclasses import replace
from fractions import Fraction

from pvz_game import LevelSpec, Place, Spawn, Status, Wait
from pvz_game.config import InitialPlant, resolve_level
from pvz_game.mechanics import blast_hits

from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.scratch_game import (
    EngineSnapshot,
    fork_game,
    restored_game,
    scratch_game,
    set_initial_mowers,
)
from pvz_rl.learning.deadline import check_deadline

LEGACY_SKILLS = ("peashooter", "cherry_bomb")
BOMB_SKILLS = ("cherry_bomb", "cherry_bomb_obstacle")
ATTACKER_EXERCISES = (*BOMB_SKILLS, "attacker_pre")
SKILLS = (*LEGACY_SKILLS, "cherry_bomb_obstacle", "attacker_pre", "attacker")


def is_bomb_skill(skill):
    return skill in BOMB_SKILLS


def is_placement_skill(skill):
    return skill in (*ATTACKER_EXERCISES, "attacker")


def attacker_mix(cfg):
    return cfg["training"].get("tile_scenarios", {}).get("attacker_mix")


def attacker_quotas(ratios, size):
    """Largest remainder, breaking ties in source exercise order."""
    shares = [size * Fraction(str(ratio)) for ratio in ratios]
    counts = [int(share) for share in shares]
    order = sorted(range(len(shares)), key=lambda i: counts[i] - shares[i])
    for i in order[: size - sum(counts)]:
        counts[i] += 1
    return counts


def shuffled_attacker_exercises(cfg, seeds, *, validation=False):
    counts = attacker_quotas(attacker_mix(cfg)["ratios"], len(seeds))
    exercises = [
        skill for skill, count in zip(ATTACKER_EXERCISES, counts, strict=True) for _ in range(count)
    ]
    # Derive a separate stream from the panel identity without drawing any board RNG.
    random.Random(repr(("attacker-exercises-v1", validation, tuple(seeds)))).shuffle(exercises)
    return exercises


def placement_source(stage, board):
    return board["exercise"] if stage == "attacker" else stage


def attacker_exercise_scores(rows):
    scores = {skill: dict(games=0, wins=0) for skill in ATTACKER_EXERCISES}
    for row in rows:
        score = scores[row["exercise"]]
        score["games"] += 1
        score["wins"] += bool(row["win"])
    return scores


def placement_settings(cfg, skill):
    settings = cfg["training"].get("tile_scenarios", {})
    return settings["attacker"] if skill == "attacker_pre" else settings


def preparation_settings(cfg, skill):
    """One resource interface for all single-placement exercises."""
    perf = cfg["training"].get("performance", {})
    result = {k.removeprefix("cherry_"): v for k, v in perf.items() if k.startswith("cherry_")}
    if skill in ("attacker_pre", "attacker"):
        result.update(
            {k.removeprefix("attacker_"): v for k, v in perf.items() if k.startswith("attacker_")}
        )
        result["oracle"] = "distance"
    return result


def skill_plant(skill, board=None):
    if skill == "attacker_pre":
        return board["supplied_plant"]
    return "cherry_bomb" if is_bomb_skill(skill) else skill


def bomb_tile_distances(positive, illegal, *, rows, cols):
    """Private spatial targets: illegal placements get the full board distance."""
    illegal = set(illegal)
    maximum = rows + cols - 2
    return [
        maximum
        if tile in illegal
        else min(
            abs(tile // cols - target // cols) + abs(tile % cols - target % cols)
            for target in positive
        )
        for tile in range(rows * cols)
    ]


def valid_bomb_labels(board, positive, skill):
    if not positive:
        return False
    if skill == "cherry_bomb_obstacle":
        occupied = {p["row"] * 9 + p["col"] for p in board["snapshot"]["plants"]}
        return positive == [board["reserved_tile"]] and set(board["blocking_tiles"]) <= occupied
    return True


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


def sample_bomb(cfg, seed, *, rules=None, deadline=None, skill="cherry_bomb"):
    for game, target_id, attempt, metadata in bomb_candidates(
        cfg, seed, rules=rules, deadline=deadline, skill=skill
    ):
        positive, absolute = _bomb_labels(game, target_id, deadline=deadline)
        board = dict(snapshot=EngineSnapshot.capture(game), **metadata)
        if valid_bomb_labels(board, positive, skill):
            return dict(
                board,
                target_id=target_id,
                feasible_tiles=positive,
                absolute_infeasible_tiles=absolute,
                generation_failures=attempt,
            )
    raise generation_failure(cfg, seed, skill)


def generation_failure(cfg, seed, skill="cherry_bomb"):
    attempts = cfg["training"]["tile_scenarios"]["sampling_max_attempts"]
    return BombGenerationFailure(f"No valid {skill} board after {attempts} attempts (seed {seed})")


def bomb_candidates(cfg, seed, *, start=0, rules=None, deadline=None, skill="cherry_bomb"):
    """Engine-authored candidates in stable seed/attempt order, without labels."""
    for game, attempt, metadata in bomb_initials(
        cfg, seed, start=start, rules=rules, deadline=deadline, skill=skill
    ):
        tick = game.rules.plants["cherry_bomb"]["initial_recharge_ticks"] + 2
        observation = game.step(Wait(), ticks=tick).observation
        targets = [z for z in observation.zombies if z.health > 0 and not z.headless]
        if observation.status == Status.RUNNING and len(targets) == 1:
            yield game, targets[0].id, attempt, metadata


def bomb_initials(cfg, seed, *, start=0, rules=None, deadline=None, skill="cherry_bomb"):
    """Validated initial boards, shared by CPU and CUDA natural cooldown advances."""
    game = scratch_game(rules)
    rules = game.rules
    settings = cfg["training"]["tile_scenarios"]
    if not is_bomb_skill(skill):
        raise ValueError(f"Unknown bomb scenario: {skill}")
    attempts = cfg["training"]["tile_scenarios"]["sampling_max_attempts"]
    namespace = "tile-cherry-bomb" if skill == "cherry_bomb" else "tile-cherry-bomb-obstacle"
    rng = random.Random(namespace_seed(namespace, seed))
    rows, cols = rules.game["rows"], rules.game["cols"]
    kinds = [kind for kind in cfg["environment"]["plants"] if kind != "cherry_bomb"]
    tick = rules.plants["cherry_bomb"]["initial_recharge_ticks"] + 2
    if cfg["environment"]["cutoff_seconds"] * rules.game["tick_rate"] <= tick:
        raise ValueError(
            "Cherry Bomb exercises need a cutoff longer than the initial card cooldown"
        )
    for attempt in range(attempts):
        check_deadline(deadline)
        metadata = {}
        if skill == "cherry_bomb":
            # Preserve the original proposal stream for saved runs and CPU/CUDA agreement.
            count = rng.randint(*settings["bomb_plants"])
            tiles = rng.sample(range(rows * cols), count)
            plants = tuple(InitialPlant(rng.choice(kinds), *divmod(tile, cols)) for tile in tiles)
            zombie = rng.choice(cfg["environment"]["zombies"])
            row = rng.randrange(rows)
            x = rng.randrange(cols * rules.game["units_per_tile"])
        else:
            zombie = rng.choice(cfg["environment"]["zombies"])
            row = rng.randrange(rows)
            x = rng.randrange(cols * rules.game["units_per_tile"])
            hit_cells = [
                tile
                for tile in range(rows * cols)
                if blast_hits(tile % cols, tile // cols, x, row, rules.game["units_per_tile"], True)
            ]
            reserved = rng.choice(hit_cells)
            blockers = [tile for tile in hit_cells if tile != reserved]
            lower, upper = settings["bomb_obstacle_plants"]
            if len(blockers) > upper:
                continue
            count = rng.randint(max(lower, len(blockers)), upper)
            extra = rng.sample(
                [tile for tile in range(rows * cols) if tile not in hit_cells],
                count - len(blockers),
            )
            plants = tuple(
                InitialPlant(rng.choice(kinds), *divmod(tile, cols)) for tile in (*blockers, *extra)
            )
            metadata = dict(reserved_tile=reserved, blocking_tiles=blockers)
        level = LevelSpec(
            "tile-" + skill.replace("_", "-"),
            (Spawn(tick, zombie, row, x=x),),
            initial_sun=settings["bomb_initial_sun"],
            plants=plants,
            mowers=False,
        )
        if attempt < start:
            continue
        game.reset(level, seed)
        # A separate stream leaves board proposals and gameplay RNG unchanged.
        mower_rng = random.Random(namespace_seed(f"{namespace}-mowers/{attempt}", seed))
        ready = settings.get("bomb_mower_ready_probability", 0.0)
        set_initial_mowers(game, [row for row in range(rows) if mower_rng.random() < ready])
        yield game, attempt, metadata
