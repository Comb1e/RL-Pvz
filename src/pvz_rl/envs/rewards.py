"""Net realized value from public assets and events, in sun-equivalent units."""

from functools import lru_cache

from pvz_game import Dig, Observation, Place, Rules, Status

# Combat diagnostics and additive accounting components share one reporting schema.
REWARD_METRICS = (
    "plant_kills",
    "mower_kills",
    "nonlethal_health_damage",
    "empty_mower_activations",
    "mower_activations",
    "mower_activation_sun",
    "wall_nut_damage",
    "empty_explosions",
    "mower_activation_penalty",
    "invalid_plant_penalty",
    "empty_dig_penalty",
    "sky_income",
    "produced_sun",
    "effective_damage",
    "plant_value_loss",
    "combat_value",
    "mower_expenditure",
    "net_value",
    "development",
    "terminal",
)

LEDGER_METRICS = (
    "discounted_return",
    "cumulative_net_value",
    "maximum_net_value",
    "value_drawdown",
    "discounted_outcome_return",
    "discounted_development_return",
)


@lru_cache(maxsize=1)
def _default_rules():
    return Rules()


def plant_value(obs: Observation) -> float:
    """Remaining asset value; natural termination never clears physical assets."""
    costs = {card.plant_type: card.cost for card in obs.cards}
    return sum(costs[p.plant_type] * p.health / p.max_health for p in obs.plants)


def asset_value(obs: Observation) -> float:
    return obs.sun + plant_value(obs)


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
        **defense_metrics(before, events, rules),
    }


def defense_metrics(before: Observation, events, rules: Rules) -> dict:
    """Attribute defense events using only public state and ordered public events.

    Reconstruct spend/income in order so activation uses its actual sun even in
    legacy multi-tick batches. Explosion damage must share source AND tick; armor
    damage counts as a hit. Merely arming, digging, or losing a mine is no miss.
    """
    plants = {p.id: p.plant_type for p in before.plants}
    costs = {c.plant_type: c.cost for c in before.cards}
    sun = before.sun
    activations = activation_sun = nut_damage = empty_explosions = 0
    damaging_sources = set()
    for event in events:
        if event.kind == "PlantPlaced":
            kind = event.get("plant_type")
            plants[event.entity_id] = kind
            sun -= costs[kind]
        elif event.kind == "PlantRemoved":
            plants.pop(event.entity_id, None)
        elif event.kind == "SunProduced":
            sun += event.get("amount", 0)  # Actual income after the engine sun cap.
        elif event.kind == "MowerActivated":
            activations += 1
            activation_sun += sun
        elif event.kind == "PlantDamaged" and plants.get(event.entity_id) == "wall_nut":
            nut_damage += event.get("damage", 0)  # Engine caps bites at remaining HP.
        elif event.kind == "DamageApplied":
            if event.get("health_damage", 0) + event.get("armor_damage", 0) > 0:
                damaging_sources.add((event.tick, event.get("source")))
        elif event.kind == "PlantExploded":
            empty_explosions += (event.tick, event.entity_id) not in damaging_sources
    return {
        "mower_activations": activations,
        "mower_activation_sun": activation_sun,
        "wall_nut_damage": nut_damage,
        "wall_nut_damage_fraction": nut_damage / rules.plants["wall_nut"]["health"],
        "empty_explosions": empty_explosions,
    }


def reward_parts(
    before: Observation,
    after: Observation,
    cfg: dict,
    *,
    events=(),
    rules=None,
    action=None,
    action_result=None,
) -> dict:
    """Count actual gains/losses once, independent of action names and difficulty."""
    rules = rules if rules is not None else _default_rules()
    events = tuple(events)
    settings = cfg["reward"]
    terminal = (
        settings["win_reward"]
        if after.status == Status.WON
        else -settings["loss_penalty"]
        if after.status == Status.LOST
        else 0.0
    )
    combat = combat_metrics(before, events, rules)
    sky = sum(
        e.get("amount", 0) for e in events if e.kind == "SunProduced" and e.get("source") == "sky"
    )
    produced = sum(
        e.get("amount", 0) for e in events if e.kind == "SunProduced" and e.get("source") != "sky"
    )
    damage = sum(
        e.get("health_damage", 0) + e.get("armor_damage", 0)
        for e in events
        if e.kind == "DamageApplied" and e.get("source", 0) > 0
    )
    resource_delta = asset_value(after) - asset_value(before) - sky
    lost_value = produced - resource_delta
    combat_value = settings["basic_zombie_value"] * damage / rules.zombies["basic"]["health"]
    mower_cost = settings["mower_value"] * combat["mower_activations"]
    net = resource_delta + combat_value - mower_cost
    scale = settings["progress_weight"] / settings["value_scale"]
    development = scale * net
    invalid_plant = 0.0
    empty_dig = 0.0
    if action_result is not None and not action_result.accepted:
        if isinstance(action, Place):
            invalid_plant = -float(settings.get("invalid_plant_penalty", 0))
        elif isinstance(action, Dig) and action_result.reason == "empty_tile":
            empty_dig = -float(settings.get("empty_dig_penalty", 0))
    return {
        **combat,
        "terminal": terminal,
        "development": development,
        "sky_income": sky,
        "produced_sun": produced,
        "effective_damage": damage,
        "plant_value_loss": lost_value,
        "combat_value": combat_value,
        "mower_expenditure": mower_cost,
        "net_value": net,
        # A reported component of development, never added a second time.
        "mower_activation_penalty": -scale * mower_cost,
        "invalid_plant_penalty": invalid_plant,
        "empty_dig_penalty": empty_dig,
        "total": terminal + development + invalid_plant + empty_dig,
    }
