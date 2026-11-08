"""Exact visible-lane teacher rules without model or CUDA dependencies."""

import copy

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game


def lane_facts(snapshot):
    counts = [0] * A.rows
    occupied = {p["row"] * A.cols + p["col"] for p in snapshot["plants"]}
    for plant in snapshot["plants"]:
        if plant["kind"] == "peashooter" and plant["health"] > 0:
            counts[plant["row"]] += 1
    threats = {}
    for zombie in snapshot["zombies"]:
        if zombie["health"] > 0 and not zombie["headless"]:
            row = zombie["row"]
            threats[row] = min(threats.get(row, zombie["x"]), zombie["x"])
    legal = [
        [r * A.cols + c for c in range(A.cols) if r * A.cols + c not in occupied]
        for r in range(A.rows)
    ]
    return counts, threats, legal


def choose_rule(snapshot, covered):
    """Return a fixed preferred set, or the urgent rows requiring candidate tests."""
    counts, threats, legal = lane_facts(snapshot)
    blocked = sum(not legal[r] for r in threats if not covered.get(r, False))
    if not threats or all(covered.get(r, False) for r in threats):
        minimum = min(counts)
        tiles = [t for r in range(A.rows) if counts[r] == minimum for t in legal[r]]
        return dict(rule=1, tiles=tiles, rows=[], blocked=blocked)
    empty = [r for r in threats if counts[r] == 0]
    if empty:
        closest = min(threats[r] for r in empty)
        return dict(
            rule=2, tiles=[], rows=[r for r in empty if threats[r] == closest], blocked=blocked
        )
    endangered = [r for r in threats if not covered.get(r, False) and legal[r]]
    if endangered:
        closest = min(threats[r] for r in endangered)
        tiles = [legal[r][0] for r in endangered if threats[r] == closest]
    else:
        tiles = []
    return dict(rule=3, tiles=tiles, rows=[], blocked=blocked)


def candidate_targets(snapshot, rows, successful):
    legal = lane_facts(snapshot)[2]
    targets, fallback = [], []
    for row in rows:
        winners = [t for t in legal[row] if successful.get(t, False)]
        if winners:
            targets.extend(winners)
        elif legal[row]:
            targets.append(legal[row][0])
            fallback.append(row)
    return targets, fallback


def lane_snapshot(snapshot, row, rules):
    """A validated current lane, with no future spawns or other-lane RNG draws.

    Synthetic *past* spawn entries reconcile engine counters with the retained
    live entities and defeated bodies. Physical fields and RNG are copied, not
    reconstructed; the original sky countdown is never rescaled.
    """
    result = copy.deepcopy(snapshot)
    for name in ("plants", "zombies", "projectiles"):
        result[name] = [e for e in result[name] if e["row"] == row]
    zombies = result["zombies"]
    result["level"] = dict(
        name="peashooter-visible-lane",
        initial_sun=result["sun"],
        plants=[],
        mowers=False,
        spawns=[
            dict(tick=result["tick"], zombie_type=z["kind"], row=row, x=max(0, z["x"]), wave=1)
            for z in zombies
        ],
    )
    result.update(
        spawn_index=len(zombies),
        defeated=sum(z["headless"] for z in zombies),
        wave=int(bool(zombies)),
        status="running" if any(not z["headless"] for z in zombies) else "won",
    )
    for mower in result["mowers"]:
        mower["state"] = "spent"
    game = restored_game(result, rules)
    return EngineSnapshot.capture(game)
