"""Validated snapshot packing with one host transfer per storage slab."""

import numpy as np
from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES
from pvz_game.cuda import schema as s
from pvz_game.cuda.backend import CapacityError, projectile_bound

from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game

GROUPS = ("plants", "zombies", "projectiles", "mowers")


def pack_snapshots(batch, snapshots):
    """Validate everything before mutation, using the engine's own validator."""
    raw = [
        snapshot
        if isinstance(snapshot, EngineSnapshot) and snapshot.unchanged()
        else restored_game(snapshot, batch.rules).snapshot()
        for snapshot in snapshots
    ]
    if any(snapshot["rules_hash"] != batch.rules.digest for snapshot in raw):
        raise ValueError("Snapshot rules differ from this batch")
    count = len(raw)
    packed = {
        "header": np.zeros((count, len(s.HEADER)), dtype=np.int64),
        "cooldowns": np.zeros((count, 8), dtype=np.int64),
        "_schedules": np.zeros((count, batch.zcap, 5), dtype=np.int64),
    }
    schemas = (s.PLANT, s.ZOMBIE, s.PROJECTILE, s.MOWER)
    for name, schema in zip(GROUPS, schemas, strict=True):
        # Transfer only occupied prefixes. Clear the destination's unused tail.
        maximum = max((len(row[name]) for row in raw), default=0)
        packed[name] = np.zeros((count, maximum, len(schema)), dtype=np.int64)
    for index, row in enumerate(raw):
        if len(row["level"]["spawns"]) > batch.zcap or len(row["zombies"]) > batch.zcap:
            raise CapacityError("Snapshot exceeds scratch zombie capacity")
        if len(row["projectiles"]) + projectile_bound(batch.rules) > batch.qcap:
            raise CapacityError("Snapshot needs extra projectile headroom")
        fields = dict(
            row,
            status=s.STATUSES.index(row["status"]),
            total_waves=max((x["wave"] for x in row["level"]["spawns"]), default=0),
            total_spawns=len(row["level"]["spawns"]),
            np=len(row["plants"]),
            nz=len(row["zombies"]),
            nq=len(row["projectiles"]),
            allowed=255,
            dig=1,
            enabled=1,
        )
        packed["header"][index] = [fields.get(key, 0) for key in s.HEADER]
        packed["cooldowns"][index] = [row["cooldowns"][key] for key in PLANT_TYPES]
        for name, schema in zip(GROUPS, schemas, strict=True):
            for j, entity in enumerate(row[name]):
                entity = dict(entity)
                if name in ("plants", "zombies"):
                    kinds = PLANT_TYPES if name == "plants" else ZOMBIE_TYPES
                    entity["kind"] = kinds.index(entity["kind"])
                if "state" in entity:
                    entity["state"] = {
                        "plants": s.PLANT_STATES,
                        "zombies": s.ZOMBIE_STATES,
                        "mowers": s.MOWER_STATES,
                    }[name].index(entity["state"])
                packed[name][index, j] = [entity.get(key, 0) for key in schema]
        for j, spawn in enumerate(row["level"]["spawns"]):
            packed["_schedules"][index, j] = (
                spawn["tick"],
                ZOMBIE_TYPES.index(spawn["zombie_type"]),
                spawn["row"],
                spawn["wave"],
                batch.rules.game["spawn_x"] if spawn["x"] is None else spawn["x"],
            )
    return raw, packed


def upload_snapshots(batch, snapshots, *, indices):
    indices = batch._indices(indices)
    if len(indices) != len(snapshots):
        raise ValueError("One snapshot required per index")
    raw, packed = pack_snapshots(batch, snapshots)
    cp = batch.cp
    ix = cp.asarray(indices)
    for name, slab in packed.items():
        target = getattr(batch, name)
        if name in GROUPS:
            target[ix] = 0
            if slab.shape[1]:
                target[ix, : slab.shape[1]] = cp.asarray(slab)
        else:
            target[ix] = cp.asarray(slab)
    for index, row in zip(indices, raw, strict=True):
        batch._templates[index] = dict(row)
    batch.facts[ix] = 0
    batch._event_counts[ix] = 0
    batch.action_masks_device()
