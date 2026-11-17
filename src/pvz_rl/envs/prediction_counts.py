"""Count-only Prediction zombies, separate from the physical spawn ledger."""

from dataclasses import replace

SCENE_PROTOCOL = "prediction-frame-v2"
COLLECTION_PROTOCOL = "prediction-first-entry-v1"


def continuation_lifecycle(snapshot, phase):
    """Semantic worker state, independent of backend/event handles."""
    if phase == "complete":
        return "ready"
    if phase == "probe":
        return "probing"
    return (
        "static_continuation"
        if any(not z["headless"] and z["health"] > 0 for z in snapshot["zombies"])
        else "settling"
    )


def public_total(physical_total, unspawned):
    """Works on scalars and arrays without importing a computational backend."""
    return physical_total + unspawned


def public_counts(observation, unspawned=0):
    if type(unspawned) is not int or unspawned < 0:
        raise ValueError("Prediction unspawned count must be a nonnegative integer")
    counts = observation.counts
    return replace(
        observation,
        counts=replace(
            counts,
            initial_total=public_total(counts.initial_total, unspawned),
            not_yet_spawned=public_total(counts.not_yet_spawned, unspawned),
            remaining=public_total(counts.remaining, unspawned),
        ),
    )


def validate_scene_state(state, count):
    if not isinstance(state, dict) or state.get("protocol") != SCENE_PROTOCOL:
        raise ValueError("Prediction scene recovery changed; start a fresh run")
    values = state.get("unspawned")
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(n) is not int or n < 0 for n in values)
    ):
        raise ValueError("Prediction recovery needs one nonnegative unspawned count per env")
    return values
