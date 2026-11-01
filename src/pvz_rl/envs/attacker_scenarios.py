"""Private conservative placement labels; collection never rolls out alternatives."""

import math
import random
from functools import lru_cache

from pvz_game import LevelSpec, Place, Spawn, Status, Wait
from pvz_game.config import InitialPlant
from pvz_game.mechanics import GAITS, MOVE_DENOMINATOR, SCALE, SPEC

from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game, scratch_game, set_initial_mowers
from pvz_rl.learning.deadline import check_deadline

ATTACKER_PROTOCOL = "conservative-attacker-v1"
ATTACKERS = ("potato_mine", "peashooter", "repeater", "snow_pea")


def attacker_settings(cfg):
    return cfg["training"]["tile_scenarios"]["attacker"]


def warmup_ticks(rules, settings):
    return max(rules.plants[k].get("initial_recharge_ticks", -1) for k in settings["provided_plants"]) + 2


def _integral(deltas, start, length):
    """Integral of the periodic, piecewise-constant integer gait."""
    width = SCALE / len(deltas)
    total = 0.0
    while length > 1e-7:
        phase = start % SCALE
        index = min(len(deltas) - 1, int(phase / width + 1e-10))
        step = min(length, (index + 1) * width - phase)
        if step < 1e-7:
            start += 1e-7
            length -= min(length, 1e-7)
            continue
        total += deltas[index] * step
        start += step
        length -= step
    return total


@lru_cache(maxsize=32768)
def motion_bound(ticks, units, minimum_speed, maximum_speed, lower=False):
    """Bound every Basic gait/phase, including arbitrary slowing in the upper bound.

    Integrate the gait over phase. A left-sample can overshoot the integral only
    at a downward jump (undershoot at an upward jump). Charge a full maximum
    phase step at every such boundary, including integer phase rounding. The
    bound does not assume monotone ordering between different sampled speeds.
    """
    values = []
    for deltas in GAITS[:2]:
        frames = len(deltas) + 1
        rates = [s * frames * SPEC["animation_factor"] * SCALE // (100 * sum(deltas))
                 for s in (minimum_speed, maximum_speed)]
        phase_step = rates[1] / (100 * frames)
        advance = ticks * rates[0 if lower else 1] / (100 * frames)
        cycles, rest = divmod(advance, SCALE)
        width = SCALE / len(deltas)
        starts = [j * width for j in range(len(deltas))]
        starts += [(x - rest) % SCALE for x in starts]
        partial = (min if lower else max)(_integral(deltas, x, rest) for x in starts)
        integral = cycles * width * sum(deltas) + partial
        variation = sum(max(0, (b - a) if lower else (a - b))
                        for a, b in zip(deltas, (*deltas[1:], deltas[0])))
        crossings = math.ceil(ticks * phase_step / SCALE) + 2
        error = crossings * variation * (math.ceil(phase_step) + 1)
        factor = 100 * frames * units / MOVE_DENOMINATOR
        values.append((integral - error if lower else integral + error) * factor)
    # Cover movement remainder and floating-point evaluation at phase boundaries.
    return max(0, math.floor(min(values)) - 2) if lower else math.ceil(max(values)) + 2


def _motion(rules, ticks, *, lower=False):
    basic = rules.zombies["basic"]
    return motion_bound(ticks, rules.game["units_per_tile"], basic["speed"], basic["max_speed"], lower)


@lru_cache(maxsize=8)
def attacker_bounds(rules):
    """One rules-derived table per process, independent of all scenario seeds."""
    g, basic = rules.game, rules.zombies["basic"]
    units = g["units_per_tile"]
    # Maximum x relative to the plant at which an ordinary bite can begin.
    contact = (SPEC["plant_inset_pixels"] + SPEC["plant_width_pixels"]
               - SPEC["contact_overlap_pixels"] - SPEC["attack_offset_pixels"]) * units / SPEC["source_tile_pixels"]
    flight = math.ceil(g["cols"] * units * g["tick_rate"] / g["projectile_speed"]) + 1
    table = {}
    for kind in ATTACKERS:
        spec = rules.plants[kind]
        if kind == "potato_mine":
            # Arm before entering the trigger rectangle, without relying on bite immunity.
            trigger = 55 * units / SPEC["source_tile_pixels"]
            horizon = spec["first_ticks"] + spec["rise_ticks"] + 2
            cells = math.ceil((_motion(rules, horizon) + max(contact, trigger) + 1) / units)
            table[kind] = {str(n): dict(cells=cells, horizon=horizon) for n in (1, 2)}
            continue
        full = math.ceil((basic["health"] + basic["armor"]) / spec["damage"])
        defeat = math.ceil((basic["armor"] + basic["health"] - basic["health"] // 3 + 1) / spec["damage"])
        table[kind] = {}
        for n in (1, 2):
            shots = (n - 1) * full + defeat
            fire = spec["interval_ticks"] + spec["windup_ticks"]
            if kind == "repeater":
                fire += ((shots - 1) // 2) * spec["interval_ticks"]
                if shots % 2 == 0:
                    fire += spec["interval_ticks"] - spec["burst_ticks"]
            else:
                fire += (shots - 1) * spec["interval_ticks"]
            horizon = fire + flight + 2
            table[kind][str(n)] = dict(
                cells=math.ceil((_motion(rules, horizon) + contact + 1) / units),
                horizon=horizon,
            )
    return dict(protocol=ATTACKER_PROTOCOL, rules_hash=rules.digest, table=table)


def _arrival(rules, distance, *, latest):
    """Conservative arrival-time interval, without reading sampled hidden speed."""
    if distance <= 0:
        return 0
    lo, hi = 0, 1
    while _motion(rules, hi, lower=latest) < distance:
        hi *= 2
        if hi > 2**20:
            raise ValueError("Attacker arrival bounds exceed representable exercise duration")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _motion(rules, mid, lower=latest) >= distance:
            hi = mid
        else:
            lo = mid
    return hi


def attacker_initials(cfg, seed, *, start=0, rules=None, deadline=None, case=None):
    settings = attacker_settings(cfg)
    game = scratch_game(rules)
    rules, rng = game.rules, random.Random(namespace_seed("tile-attacker-pre", seed))
    rows, cols, units = (rules.game[k] for k in ("rows", "cols", "units_per_tile"))
    tick = warmup_ticks(rules, settings)
    kinds = [k for k in cfg["environment"]["plants"] if k != "cherry_bomb"]
    for attempt in range(cfg["training"]["annotation"]["sampling_max_attempts"]):
        check_deadline(deadline)
        supplied = rng.choice(settings["provided_plants"])
        count = rng.randint(*settings["zombie_count"])
        if case is not None:
            supplied, count = case
        row = rng.randrange(rows)
        background = [r for r in range(rows) if r != row]
        tiles = [r * cols + rng.randrange(cols) for r in background]
        count_plants = rng.randint(*settings["plants"])
        tiles += rng.sample([t for t in range(rows * cols) if t // cols != row and t not in tiles],
                            count_plants - len(tiles))
        plants = tuple(InitialPlant(rng.choice(kinds), *divmod(t, cols)) for t in tiles)
        positions = sorted(rng.sample(range(cols * units), count))
        level = LevelSpec("tile-attacker-pre", tuple(Spawn(tick, "basic", row, x=x) for x in positions),
                          initial_sun=settings["initial_sun"], plants=plants, mowers=False)
        if attempt < start:
            continue
        game.reset(level, seed)
        mower_rng = random.Random(namespace_seed(f"tile-attacker-mowers/{attempt}", seed))
        set_initial_mowers(game, [r for r in range(rows) if mower_rng.random() < settings["mower_ready_probability"]])
        yield game, attempt, dict(supplied_plant=supplied, zombie_count=count, target_row=row,
                                 bounds=attacker_bounds(rules), scenario="attacker_pre")


def label_attacker(board, rules):
    """Return labels after natural cooldown warmup; never simulate placements."""
    snap, kind = board["snapshot"], board["supplied_plant"]
    targets = sorted((z for z in snap["zombies"] if z["health"] > 0 and not z["headless"]),
                     key=lambda z: (z["x"], z["id"]))
    row, n = board["target_row"], board["zombie_count"]
    occupied = {p["row"] * 9 + p["col"] for p in snap["plants"]}
    if (snap["status"] != "running" or len(targets) != n
            or any(z["row"] != row or z["kind"] != "basic" for z in targets)
            or any(t // 9 == row for t in occupied)
            or snap["sun"] < rules.plants[kind]["cost"] or snap["cooldowns"][kind] != 0
            or board["bounds"] != attacker_bounds(rules)):
        return None
    units = rules.game["units_per_tile"]
    bound = board["bounds"]["table"][kind][str(n)]
    positive = []
    for col in range(9):
        if targets[0]["x"] - col * units < bound["cells"] * units:
            continue
        if kind == "potato_mine" and n == 2:
            # First target must reach even the *left* trigger boundary before
            # the other can enter the *right* boundary. Includes one tick of
            # mine-before-movement ordering; no hidden sampled speed is used.
            front = targets[0]["x"] - (col * units + 40 * units / SPEC["source_tile_pixels"])
            rear = targets[1]["x"] - (col * units + 55 * units / SPEC["source_tile_pixels"])
            if _arrival(rules, front, latest=True) + 2 >= _arrival(rules, rear, latest=False):
                continue
        positive.append(row * 9 + col)
    return dict(board, target_ids=[z["id"] for z in targets],
                feasible_tiles=positive or [row * 9], absolute_infeasible_tiles=sorted(occupied),
                fallback=not positive, required_cells=bound["cells"])


def sample_attacker(cfg, seed, *, start=0, rules=None, deadline=None, case=None):
    for game, attempt, metadata in attacker_initials(cfg, seed, start=start, rules=rules,
                                                    deadline=deadline, case=case):
        ticks = warmup_ticks(game.rules, attacker_settings(cfg))
        for offset in range(0, ticks, 256):
            check_deadline(deadline)
            game.step(Wait(), ticks=min(256, ticks - offset))
        board = label_attacker(dict(metadata, snapshot=EngineSnapshot.capture(game),
                                    generation_failures=attempt), game.rules)
        if board is not None:
            return board
    from pvz_rl.envs.tile_scenarios import generation_failure

    raise generation_failure(cfg, seed, "attacker_pre")


def _attributed_defeats(events, source, projectiles):
    """The damage that actually defeats a target must belong to the new plant."""
    damage, defeated = {}, set()
    for event in events:
        if event.kind == "ProjectileFired" and event.get("source") == source:
            projectiles.add(event.entity_id)
        elif event.kind == "DamageApplied":
            damage[event.tick, event.entity_id] = (
                event.get("source") == source or event.get("source") in projectiles
            )
        elif event.kind == "ZombieDefeated" and damage.get((event.tick, event.entity_id), False):
            defeated.add(event.entity_id)
    return defeated


def attacker_result(cfg, board, tile, *, rules=None, deadline=None):
    """Independent CPU validation with exact plant→projectile→defeat attribution."""
    game = restored_game(board["snapshot"], rules)
    action = Place(board["supplied_plant"], *divmod(tile, 9))
    accepted = game.validate_action(action).accepted
    source, projectiles, defeated = game._next_id, set(), set()
    cutoff = int(cfg["environment"]["cutoff_seconds"] * game.rules.game["tick_rate"])
    wanted = set(board["target_ids"][:1] if board["supplied_plant"] == "potato_mine" else board["target_ids"])
    pending = action
    while accepted and game.observe().status == Status.RUNNING and game._tick < cutoff:
        check_deadline(deadline)
        result = game.step(pending, ticks=min(256, cutoff - game._tick))
        pending = Wait()
        defeated.update(_attributed_defeats(result.events, source, projectiles))
        if wanted <= defeated:
            break
    killed = wanted <= defeated
    correct = accepted and tile in board["feasible_tiles"]
    return dict(win=bool(correct and (board["fallback"] or killed)), target_correct=bool(correct),
                actual_kill=bool(killed), attributed_kills=len(set(board["target_ids"]) & defeated),
                fallback=board["fallback"], supplied_plant=board["supplied_plant"],
                zombie_count=board["zombie_count"], engine_status=game.observe().status.value)
