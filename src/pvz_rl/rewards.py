"""Terminal, potential, and public-event kill rewards, reported separately."""

from pvz_game import Observation, Status


def potential(obs: Observation, cfg: dict) -> float:
    if obs.status != Status.RUNNING:
        return 0.0
    r = cfg["reward"]
    flowers = sum(p.plant_type == "sunflower" for p in obs.plants)
    return (
        r["defeated_weight"] * obs.counts.defeated / max(1, obs.counts.initial_total)
        + r["sun_weight"] * min(obs.sun / r["sun_target"], 1)
        + r["flower_weight"] * min(flowers / r["flower_target"], 1)
    )


def defeat_counts(events) -> dict:
    """Attribute the killing damage, not earlier damage or mower activation.

    In the pinned game every death follows DamageApplied within the same tick.
    Positive sources are plant/projectile IDs; negative sources encode mower rows.
    IDs are used transiently to join public events, never as policy features.
    """
    sources, defeated = {}, set()
    counts = {"plant_kills": 0, "mower_kills": 0}
    for event in events:
        if event.kind == "DamageApplied":
            sources[event.entity_id] = event.get("source")
        elif event.kind == "ZombieDefeated":
            source = sources.pop(event.entity_id, None)
            if event.entity_id in defeated or type(source) is not int or source == 0:
                raise RuntimeError("Cannot attribute public ZombieDefeated event to killing damage")
            defeated.add(event.entity_id)
            counts["mower_kills" if source < 0 else "plant_kills"] += 1
    return counts


def reward_parts(
    before: Observation, after: Observation, cfg: dict, shaped: bool, *, events=()
) -> dict:
    terminal = 1.0 if after.status == Status.WON else -1.0 if after.status == Status.LOST else 0.0
    shaping = (
        cfg["reward"]["gamma"] * potential(after, cfg) - potential(before, cfg) if shaped else 0.0
    )
    kills = defeat_counts(events)
    settings = cfg["reward"]
    denominator = (
        max(1, before.counts.initial_total) if settings.get("normalize_kills", True) else 1
    )
    # Absent weights preserve legacy checkpoint/configuration rewards exactly.
    plant_reward = settings.get("plant_kill_weight", 0.0) * kills["plant_kills"] / denominator
    mower_penalty = -settings.get("mower_kill_weight", 0.0) * kills["mower_kills"] / denominator
    return {
        "terminal": terminal,
        "shaping": shaping,
        **kills,
        "plant_kill_reward": plant_reward,
        "mower_kill_penalty": mower_penalty,
        "total": terminal + shaping + plant_reward + mower_penalty,
    }
