"""Prediction recovery with separate house-exit and real-kill accounting."""

from pvz_game.config import integer

from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game

ACCOUNTING_PROTOCOL = "prediction-accounting-v1"


def _snapshot_error(snapshot, index, reason, exited=None):
    row = snapshot if isinstance(snapshot, dict) else {}
    zombies = row.get("zombies")
    present = (
        sum(not zombie["headless"] for zombie in zombies)
        if isinstance(zombies, list)
        and all(isinstance(z, dict) and type(z.get("headless")) is bool for z in zombies)
        else "invalid"
    )
    if (
        exited is None
        and type(present) is int
        and type(row.get("spawn_index")) is int
        and type(row.get("defeated")) is int
    ):
        exited = row["spawn_index"] - row["defeated"] - present
    return ValueError(
        f"Prediction snapshot env {index} "
        f"(tick={row.get('tick')}, spawned={row.get('spawn_index')}, "
        f"defeated={row.get('defeated')}, present={present}, exited={exited}): {reason}"
    )


def _exited_count(snapshot):
    integer(snapshot["tick"], "tick")
    spawned = integer(snapshot["spawn_index"], "spawn_index")
    defeated = integer(snapshot["defeated"], "defeated")
    zombies = snapshot["zombies"]
    if not isinstance(zombies, list) or any(
        not isinstance(z, dict) or type(z.get("headless")) is not bool for z in zombies
    ):
        raise ValueError("Zombie records require boolean headless flags")
    present = sum(not zombie["headless"] for zombie in zombies)
    exited = spawned - defeated - present
    if spawned > len(snapshot["level"]["spawns"]) or exited < 0:
        raise ValueError("Inconsistent Prediction zombie counts")
    return exited


def capture_prediction_accounting(snapshots):
    """Record exits from the authoritative spawn, kill and occupied-entity ledger."""
    exits = []
    for index, snapshot in enumerate(snapshots):
        try:
            exits.append(_exited_count(snapshot))
        except (KeyError, TypeError, ValueError) as exc:
            raise _snapshot_error(snapshot, index, exc) from exc
    return dict(protocol=ACCOUNTING_PROTOCOL, exited=exits)


def validated_prediction_snapshots(snapshots, accounting, rules):
    """Validate the whole panel before any device mutation.

    Missing metadata belongs to compatible Prediction recovery written before
    exits were recorded. Only the resolved count is adjusted for the pinned
    validator. Returned owned roots retain the actual kills and original RNG.
    """
    if accounting is None:
        accounting = capture_prediction_accounting(snapshots)
    if not isinstance(accounting, dict) or accounting.get("protocol") != ACCOUNTING_PROTOCOL:
        raise ValueError("Unsupported Prediction snapshot accounting protocol")
    exits = accounting.get("exited")
    if not isinstance(exits, list) or len(exits) != len(snapshots):
        raise ValueError("Prediction accounting requires one exited count per environment")
    roots = []
    for index, (snapshot, exited) in enumerate(zip(snapshots, exits, strict=True)):
        try:
            integer(exited, "exited")
            if _exited_count(snapshot) != exited:
                raise ValueError("Exited count disagrees with the saved zombie ledger")
            validation = dict(snapshot, defeated=snapshot["defeated"] + exited)
            game = restored_game(validation, rules)
            game._defeated = snapshot["defeated"]
            roots.append(EngineSnapshot.capture(game))
        except (KeyError, TypeError, ValueError) as exc:
            raise _snapshot_error(snapshot, index, exc, exited) from exc
    return roots
