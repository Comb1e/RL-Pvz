"""Terminal, potential, and public-event combat rewards, reported separately."""

from functools import lru_cache

from pvz_game import Observation, Rules, Status

REWARD_METRICS = (
    "plant_kills",
    "mower_kills",
    "nonlethal_health_damage",
    "empty_mower_activations",
    "plant_kill_reward",
    "mower_kill_penalty",
    "damage_reward",
    "mower_activation_penalty",
)


@lru_cache(maxsize=1)
def _default_rules():
    return Rules()


def potential(obs: Observation, cfg: dict) -> float:
    if obs.status != Status.RUNNING:
        return 0.0
    r = cfg["reward"]
    progress = r["defeated_weight"] * obs.counts.defeated / max(1, obs.counts.initial_total)
    if r.get("potential_mode", "legacy") == "plant_value":
        costs = {card.plant_type: card.cost for card in obs.cards}
        plant_value = sum(costs[p.plant_type] for p in obs.plants)
        return progress + r["economy_weight"] * (obs.sun + plant_value) / r["economy_scale"]
    flowers = sum(p.plant_type == "sunflower" for p in obs.plants)
    return (
        progress
        + r["sun_weight"] * min(obs.sun / r["sun_target"], 1)
        + r["flower_weight"] * min(flowers / r["flower_target"], 1)
    )


def _attribute_events(events):
    """Separate lethal blows, earlier damage, and empty mower activations.

    In the pinned game every death follows DamageApplied within the same tick.
    Positive sources are plant/projectile IDs; negative sources encode mower rows.
    IDs are used transiently to join public events, never as policy features.
    """
    damage_indices, defeated, lethal = {}, set(), set()
    mower_kill_ticks = set()
    counts = {"plant_kills": 0, "mower_kills": 0}
    for index, event in enumerate(events):
        if event.kind == "DamageApplied":
            damage_indices[event.entity_id] = index
        elif event.kind == "ZombieDefeated":
            damage_index = damage_indices.pop(event.entity_id, None)
            hit = None if damage_index is None else events[damage_index]
            source = None if hit is None else hit.get("source")
            if (
                event.entity_id in defeated
                or type(source) is not int
                or source == 0
                or hit.tick != event.tick
            ):
                raise RuntimeError("Cannot attribute public ZombieDefeated event to killing damage")
            defeated.add(event.entity_id)
            lethal.add(damage_index)
            counts["mower_kills" if source < 0 else "plant_kills"] += 1
            if source < 0:
                mower_kill_ticks.add((event.tick, -source - 1))
    nonlethal = [
        event
        for index, event in enumerate(events)
        if event.kind == "DamageApplied" and index not in lethal and event.get("source", 0) > 0
    ]
    counts["empty_mower_activations"] = sum(
        event.kind == "MowerActivated" and (event.tick, event.get("row")) not in mower_kill_ticks
        for event in events
    )
    return counts, nonlethal


def defeat_counts(events) -> dict:
    """Compatibility interface for plant/mower killing-blow counts."""
    counts, _ = _attribute_events(tuple(events))
    return {key: counts[key] for key in ("plant_kills", "mower_kills")}


def combat_metrics(before: Observation, events, rules: Rules | None = None) -> dict:
    events = tuple(events)
    counts, nonlethal = _attribute_events(events)
    types = {z.id: z.zombie_type for z in before.zombies}
    types.update(
        (event.entity_id, event.get("zombie_type"))
        for event in events
        if event.kind == "ZombieSpawned"
    )
    rules = rules if rules is not None else _default_rules()
    health_damage, damage_fraction = 0, 0.0
    for event in nonlethal:
        damage = event.get("health_damage", 0)
        if damage <= 0:  # Armor-only damage does not remove base HP.
            continue
        zombie_type = types.get(event.entity_id)
        if zombie_type not in rules.zombies:
            raise RuntimeError("Cannot identify public zombie type for nonlethal damage")
        health_damage += damage
        damage_fraction += damage / rules.zombies[zombie_type]["health"]
    return {
        **counts,
        "nonlethal_health_damage": health_damage,
        "nonlethal_damage_fraction": damage_fraction,
    }


def reward_parts(
    before: Observation, after: Observation, cfg: dict, shaped: bool, *, events=(), rules=None
) -> dict:
    terminal = 1.0 if after.status == Status.WON else -1.0 if after.status == Status.LOST else 0.0
    shaping = (
        cfg["reward"]["gamma"] * potential(after, cfg) - potential(before, cfg) if shaped else 0.0
    )
    combat = combat_metrics(before, events, rules)
    settings = cfg["reward"]
    denominator = (
        max(1, before.counts.initial_total) if settings.get("normalize_kills", True) else 1
    )
    # Absent weights preserve legacy checkpoint/configuration rewards exactly.
    plant_reward = settings.get("plant_kill_weight", 0.0) * combat["plant_kills"] / denominator
    mower_penalty = -settings.get("mower_kill_weight", 0.0) * combat["mower_kills"] / denominator
    damage_reward = (
        settings.get("damage_weight", 0.0)
        * combat["nonlethal_damage_fraction"]
        / max(1, before.counts.initial_total)
    )
    activation_penalty = (
        -settings.get("empty_mower_activation_penalty", 0.0) * combat["empty_mower_activations"]
    )
    return {
        "terminal": terminal,
        "shaping": shaping,
        **combat,
        "plant_kill_reward": plant_reward,
        "mower_kill_penalty": mower_penalty,
        "damage_reward": damage_reward,
        "mower_activation_penalty": activation_penalty,
        "total": terminal
        + shaping
        + plant_reward
        + mower_penalty
        + damage_reward
        + activation_penalty,
    }
