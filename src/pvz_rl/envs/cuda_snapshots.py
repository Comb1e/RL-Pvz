"""Validated snapshot transport with one transfer per occupied storage slab."""

from copy import deepcopy

import numpy as np
from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES
from pvz_game.cuda import schema as s
from pvz_game.cuda.backend import CapacityError, projectile_bound

from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game, scratch_game

GROUPS = ("plants", "zombies", "projectiles", "mowers")


def reset_owned(batch, levels, seeds, *, indices):
    """Resolve fresh scenarios with shared validated rules, then batch one upload.

    No external snapshot bypasses validation. The engine still resolves every
    level and seed; only repeated construction/restoration of Rules is removed.
    """
    if len(levels) != len(seeds) or len(levels) != len(indices):
        raise ValueError("One level and seed required per reset index")
    snapshots = []
    for level, seed in zip(levels, seeds, strict=True):
        game = scratch_game(batch.rules)
        game.reset(level, seed)
        snapshots.append(EngineSnapshot.capture(game))
    upload_snapshots(batch, snapshots, indices=indices)
    if batch.sky_sun_rate != 1.0:
        cp = batch.cp
        ix = cp.asarray(indices)
        due = s.HEADER.index("sky_due")
        batch.header[ix, due] = cp.maximum(1, cp.floor(batch.header[ix, due] / batch.sky_sun_rate)).astype(cp.int64)


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


def download_snapshots(batch, *, indices, share_metadata=False):
    """Capture owned CUDA states using the pinned snapshot schema, in one batch."""
    indices = batch._indices(indices)
    if not indices:
        return []
    if any(batch._templates[i] is None for i in indices):
        raise RuntimeError("Reset before taking snapshots")
    ix = batch.cp.asarray(indices)
    headers = batch.header[ix].get()
    cooldowns = batch.cooldowns[ix].get().tolist()
    counts = [headers[:, s.HEADER.index(name)] for name in ("np", "nz", "nq")]
    counts.append(np.full(len(indices), batch.rules.game["rows"]))
    slabs = [
        getattr(batch, name)[ix, : int(size.max())].get().tolist()
        for name, size in zip(GROUPS, counts, strict=True)
    ]
    snapshots = []
    for i, index in enumerate(indices):
        # Internal oracle consumers never mutate rules, level or RNG metadata.
        # Other callers retain the ordinary fully detached snapshot contract.
        raw = dict(batch._templates[index]) if share_metadata else deepcopy(batch._templates[index])
        header = dict(zip(s.HEADER, headers[i].tolist(), strict=True))
        for key in (
            "tick",
            "sun",
            "next_id",
            "spawn_index",
            "defeated",
            "wave",
            "gameplay_rng",
            "sky_due",
            "sky_drops",
        ):
            raw[key] = header[key]
        raw["status"] = s.STATUSES[header["status"]]
        raw["cooldowns"] = dict(zip(PLANT_TYPES, cooldowns[i], strict=True))
        for name, schema, slab, size in zip(
            GROUPS, (s.PLANT, s.ZOMBIE, s.PROJECTILE, s.MOWER), slabs, counts, strict=True
        ):
            raw[name] = []
            for values in slab[i][: int(size[i])]:
                item = dict(zip(schema, values, strict=True))
                item.pop("previous_x", None)
                if name in ("plants", "zombies"):
                    kinds = PLANT_TYPES if name == "plants" else ZOMBIE_TYPES
                    item["kind"] = kinds[item["kind"]]
                if "state" in item:
                    states = {
                        "plants": s.PLANT_STATES,
                        "zombies": s.ZOMBIE_STATES,
                        "mowers": s.MOWER_STATES,
                    }[name]
                    item["state"] = states[item["state"]]
                for key in ("icy", "has_pole", "headless"):
                    if key in item:
                        item[key] = bool(item[key])
                raw[name].append(item)
        snapshots.append(EngineSnapshot._seal(raw))
    return snapshots
