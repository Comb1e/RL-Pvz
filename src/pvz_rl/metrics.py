"""Task-labelled diagnostics; never inputs to the policy or reward."""

from collections import Counter


def task_name(level, family):
    return (
        level
        if family == "preset"
        else family
        if family in ("placement", "saving", "diagnostic")
        else f"{family}/{level}"
    )


def episode_task(row):
    return task_name(row.get("level", "unknown"), row.get("family", "unknown"))


def task_statistics(rows):
    count = len(rows)
    usage = sum((Counter(row.get("plant_usage", {})) for row in rows), Counter())
    plants = sum(usage.values())
    digs = sum(row.get("early_voluntary_digs", 0) for row in rows)
    return {
        "completed_games": count,
        "win_rate": sum(row["win"] for row in rows) / count if count else None,
        "simulated_seconds": sum(row["simulated_seconds"] for row in rows) / count
        if count
        else None,
        "early_digs_per_game": digs / count if count else None,
        "early_digs_per_planting": digs / plants if plants else None,
        "attacker_purchases_per_game": sum(row.get("attacker_purchases", 0) for row in rows) / count
        if count
        else None,
        "plant_usage": dict(usage),
    }
