"""Versioned snapshot and dual tile/action annotation storage.

The browser stores rich public observations for display and the canonical entity
observation separately for fitting.  Source metadata is intentionally kept out
of the tensors passed to the policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

SNAPSHOT_PROTOCOL = "pvz-rl/tile-annotation-snapshot-v1"
# v2 adds action records and explicit absolute-infeasible labels.  v1 records
# remain readable; their absolute tile set is derived from the legal complement.
ANNOTATION_PROTOCOL = "pvz-rl/annotation-v2"
LEGACY_ANNOTATION_PROTOCOL = "pvz-rl/tile-annotation-v1"
TILES = 45
ROWS, COLS = 5, 9


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported annotation value: {type(value).__name__}")


def _required(mapping, keys, label):
    if not isinstance(mapping, dict) or any(key not in mapping for key in keys):
        raise ValueError(f"{label} is missing required fields")


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


def _nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _string(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def validate_observation(observation):
    """Validate the rich public observation used by the browser renderer."""
    _required(
        observation,
        (
            "api_version",
            "tick",
            "elapsed_seconds",
            "tick_rate",
            "units_per_tile",
            "rows",
            "cols",
            "level",
            "status",
            "sun",
            "cards",
            "plants",
            "zombies",
            "projectiles",
            "mowers",
            "wave",
            "total_waves",
            "counts",
        ),
        "observation",
    )
    if observation["rows"] != ROWS or observation["cols"] != COLS:
        raise ValueError("annotation observations must use the pinned 5x9 board")
    for key in ("api_version", "tick", "rows", "cols", "wave", "total_waves"):
        _nonnegative_int(observation[key], f"observation.{key}")
    _string(observation["level"], "observation.level")
    if observation["status"] not in ("running", "won", "lost"):
        raise ValueError("observation.status must be running, won, or lost")
    for key in ("elapsed_seconds", "units_per_tile", "sun"):
        _finite_number(observation[key], f"observation.{key}")
        if observation[key] < 0:
            raise ValueError(f"observation.{key} must be nonnegative")
    _nonnegative_int(observation["tick_rate"], "observation.tick_rate")
    if observation["tick_rate"] < 1 or observation["units_per_tile"] < 1:
        raise ValueError("observation tick rate and units per tile must be positive")
    for key in ("cards", "plants", "zombies", "projectiles", "mowers"):
        if not isinstance(observation[key], list):
            raise ValueError(f"observation.{key} must be a list")
    if not isinstance(observation["counts"], dict):
        raise ValueError("observation.counts must be an object")
    for key in (
        "initial_total",
        "spawned",
        "alive",
        "defeated",
        "not_yet_spawned",
        "remaining",
    ):
        _nonnegative_int(observation["counts"].get(key), f"observation.counts.{key}")
    for card in observation["cards"]:
        _required(card, ("plant_type", "cost", "cooldown_ticks", "recharge_ticks"), "card")
        _string(card["plant_type"], "card.plant_type")
        for key in ("cost", "cooldown_ticks", "recharge_ticks"):
            _nonnegative_int(card[key], f"card.{key}")
    for plant in observation["plants"]:
        _required(
            plant,
            ("id", "plant_type", "row", "col", "health", "max_health", "state", "timer_ticks"),
            "plant",
        )
        _nonnegative_int(plant["id"], "plant.id")
        _string(plant["plant_type"], "plant.plant_type")
        if type(plant["row"]) is not int or not 0 <= plant["row"] < ROWS:
            raise ValueError("plant.row is outside the board")
        if type(plant["col"]) is not int or not 0 <= plant["col"] < COLS:
            raise ValueError("plant.col is outside the board")
        for key in ("health", "max_health", "timer_ticks"):
            _nonnegative_int(plant[key], f"plant.{key}")
        _string(plant["state"], "plant.state")
    for zombie in observation["zombies"]:
        _required(
            zombie,
            (
                "id",
                "zombie_type",
                "row",
                "x",
                "health",
                "armor",
                "state",
                "slow_ticks",
                "has_pole",
                "timer_ticks",
                "headless",
            ),
            "zombie",
        )
        _nonnegative_int(zombie["id"], "zombie.id")
        _string(zombie["zombie_type"], "zombie.zombie_type")
        if type(zombie["row"]) is not int or not 0 <= zombie["row"] < ROWS:
            raise ValueError("zombie.row is outside the board")
        # The public fixed-point coordinate can pass left of the lawn before
        # a house breach (or while a headless zombie is decaying).
        if type(zombie["x"]) is not int:
            raise ValueError("zombie.x must be an integer")
        for key in ("health", "armor", "slow_ticks", "timer_ticks"):
            _nonnegative_int(zombie[key], f"zombie.{key}")
        _string(zombie["state"], "zombie.state")
        if type(zombie["has_pole"]) is not bool or type(zombie["headless"]) is not bool:
            raise ValueError("zombie flags must be boolean")
    for projectile in observation["projectiles"]:
        _required(projectile, ("id", "row", "x", "damage", "icy"), "projectile")
        if type(projectile["row"]) is not int or not 0 <= projectile["row"] < ROWS:
            raise ValueError("projectile.row is outside the board")
        for key in ("id", "x", "damage"):
            _nonnegative_int(projectile[key], f"projectile.{key}")
        if type(projectile["icy"]) is not bool:
            raise ValueError("projectile.icy must be boolean")
    for mower in observation["mowers"]:
        _required(mower, ("row", "x", "state"), "mower")
        if type(mower["row"]) is not int or not 0 <= mower["row"] < ROWS:
            raise ValueError("mower.row is outside the board")
        _nonnegative_int(mower["x"], "mower.x")
        _string(mower["state"], "mower.state")


def validate_encoded_observation(encoded):
    _required(encoded, ("entities", "globals"), "encoded_observation")
    entities = np.asarray(encoded["entities"])
    if entities.size == 0:
        entities = entities.reshape(0, 11)
    globals_ = np.asarray(encoded["globals"])
    if entities.ndim != 2 or entities.shape[1] != 11:
        raise ValueError("encoded_observation.entities must have shape (n, 11)")
    if globals_.shape != (18,):
        raise ValueError("encoded_observation.globals must have shape (18,)")
    if not np.isfinite(entities).all() or not np.isfinite(globals_).all():
        raise ValueError("encoded_observation contains non-finite values")
    if not np.equal(entities, entities.astype(np.int32)).all():
        raise ValueError("encoded_observation.entities must contain integer fields")


def annotation_kind(record):
    """Return the annotation kind, accepting legacy tile records."""
    kind = record.get("annotation_kind", "tile")
    if kind not in ("tile", "action"):
        raise ValueError("annotation_kind must be tile or action")
    return kind


def action_names(plant_types):
    return ("wait", *tuple(plant_types or ()), "dig")


def _validate_labels(values, label, *, upper):
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"{label} must be a list of integers")
    if len(set(values)) != len(values) or any(value < 0 or value >= upper for value in values):
        raise ValueError(f"{label} must contain unique in-range labels")
    if values != sorted(values):
        raise ValueError(f"{label} must be sorted")


def _action_metadata(record, plant_types):
    metadata = record.get("action_availability", record.get("branch_availability"))
    if not isinstance(metadata, dict):
        raise ValueError("action records require action_availability metadata")
    inferred = tuple(name for name in metadata if name not in ("wait", "dig")) if plant_types is None else plant_types
    names = action_names(inferred)
    if set(metadata) != set(names):
        raise ValueError("action_availability must contain wait, every plant, and dig")
    for name in names:
        item = metadata[name]
        _required(item, ("feasible", "absolute_infeasible", "reason"), f"action_availability.{name}")
        if type(item["feasible"]) is not bool or type(item["absolute_infeasible"]) is not bool:
            raise ValueError("action availability flags must be boolean")
        if not isinstance(item["reason"], (str, type(None))):
            raise ValueError("action availability reason must be a string or null")
    if metadata["wait"]["feasible"] is not True or metadata["wait"]["absolute_infeasible"] is not False:
        raise ValueError("wait must be feasible in a running annotation snapshot")
    expected_dig = bool(record["observation"]["plants"])
    if metadata["dig"]["feasible"] is not expected_dig or metadata["dig"]["absolute_infeasible"] is expected_dig:
        raise ValueError("dig availability does not match the board")
    return metadata


def _automatic_action_labels(record, plant_types):
    """Return automatic action labels for annotation records.

    Simulator availability remains the source of truth for the audit metadata.
    The annotation protocol additionally starts every new action record with
    ``dig`` in the absolute-infeasible set.  This default keeps the action
    annotator from silently inheriting the sampler's rollout choice; legacy
    records that predate the default are still accepted by ``validate_record``.
    """
    metadata = _action_metadata(record, plant_types)
    inferred = tuple(
        name for name in metadata if name not in ("wait", "dig")
    ) if plant_types is None else plant_types
    names = action_names(inferred)
    labels = {
        name for name, item in metadata.items() if item["absolute_infeasible"]
    }
    labels.add("dig")
    return sorted(labels, key=names.index)


def validate_record(record, *, annotation=False, plant_types=None):
    if not isinstance(record, dict):
        raise ValueError("annotation record must be an object")
    protocol = record.get("protocol")
    expected = ANNOTATION_PROTOCOL if annotation else SNAPSHOT_PROTOCOL
    if protocol not in (expected, LEGACY_ANNOTATION_PROTOCOL if annotation else None, None):
        raise ValueError(f"unsupported annotation protocol: {protocol!r}")
    for key in (
        "id",
        "group_id",
        "observation",
        "encoded_observation",
    ):
        if key not in record:
            raise ValueError(f"record is missing {key}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("record.id must be a non-empty string")
    if not isinstance(record["group_id"], str) or not record["group_id"]:
        raise ValueError("record.group_id must be a non-empty string")
    # Snapshot records normally predate the annotation-kind field and are
    # tile records.  Action snapshots carry the field before validation so
    # they can omit the tile-only selected plant target.
    kind = annotation_kind(record) if annotation or record.get("annotation_kind") == "action" else "tile"
    if kind == "tile":
        if not isinstance(record.get("plant_type"), str) or not record["plant_type"]:
            raise ValueError("tile records require a non-empty plant_type")
        if plant_types is not None and record["plant_type"] not in plant_types:
            raise ValueError(f"unknown plant type: {record['plant_type']}")
    elif record.get("plant_type") not in (None, "") and plant_types is not None and record.get("plant_type") not in plant_types:
        raise ValueError(f"unknown plant type: {record['plant_type']}")
    if "rules_hash" in record:
        _string(record["rules_hash"], "record.rules_hash")
    if "observation_schema" in record and not isinstance(record["observation_schema"], dict):
        raise ValueError("record.observation_schema must be an object")
    validate_observation(record["observation"])
    if annotation and record["observation"]["status"] != "running":
        raise ValueError("annotation records must use a running snapshot")
    validate_encoded_observation(record["encoded_observation"])
    availability = record.get("availability", {})
    if kind == "action":
        _action_metadata(record, plant_types)
        availability = availability if isinstance(availability, dict) else {}
    else:
        _required(
            availability,
            ("available", "cost", "sun", "cooldown_ticks", "legal_tiles"),
            "availability",
        )
    if not annotation and "availability" not in record:
        raise ValueError("record is missing availability")
    if kind == "tile" and availability["available"] is not True:
        raise ValueError("annotation records must select an available plant")
    if kind == "tile":
        for key in ("cost", "sun", "cooldown_ticks"):
            _nonnegative_int(availability[key], f"availability.{key}")
    cards = [
        card
        for card in record["observation"]["cards"]
        if kind == "tile" and card["plant_type"] == record["plant_type"]
    ]
    if kind == "tile" and len(cards) != 1:
        raise ValueError("annotation plant must have exactly one matching card")
    card = cards[0] if cards else None
    if kind == "tile" and (
        availability["cost"] != card["cost"]
        or availability["cooldown_ticks"] != card["cooldown_ticks"]
        or availability["sun"] != record["observation"]["sun"]
        or availability["sun"] < card["cost"]
        or card["cooldown_ticks"] != 0
    ):
        raise ValueError("annotation plant availability metadata is inconsistent")
    legal = availability.get("legal_tiles", [])
    if kind == "tile" and (not isinstance(legal, list) or any(
        type(tile) is not int or not 0 <= tile < TILES for tile in legal
    )):
        raise ValueError("availability.legal_tiles must contain valid tile indices")
    if kind == "tile" and (len(set(legal)) != len(legal) or legal != sorted(legal) or not legal):
        raise ValueError("availability.legal_tiles must contain at least one unique tile")
    occupied = {plant["row"] * COLS + plant["col"] for plant in record["observation"]["plants"]}
    if kind == "tile" and occupied.intersection(legal):
        raise ValueError("availability.legal_tiles contains an occupied tile")
    if annotation:
        if "revision" in record:
            _nonnegative_int(record["revision"], "record.revision")
        status = record.get("status")
        if status not in ("pending", "labeled", "discarded"):
            raise ValueError("annotation status must be pending, labeled, or discarded")
        if kind == "tile":
            tiles = record.get("feasible_tiles", [])
            absolute = record.get("absolute_infeasible_tiles")
            automatic = record.get("automatic_absolute_infeasible_tiles")
            _validate_labels(tiles, "feasible_tiles", upper=TILES)
            if absolute is None:
                absolute = sorted(set(range(TILES)) - set(legal))
            _validate_labels(absolute, "absolute_infeasible_tiles", upper=TILES)
            if automatic is None:
                automatic = sorted(set(range(TILES)) - set(legal))
            _validate_labels(automatic, "automatic_absolute_infeasible_tiles", upper=TILES)
            derived_automatic = sorted(set(range(TILES)) - set(legal))
            if set(automatic) != set(derived_automatic):
                raise ValueError("automatic absolute tile labels must equal the legal complement")
            if not set(automatic).issubset(set(absolute)):
                raise ValueError("automatic absolute tile labels cannot be removed")
            if set(tiles) & set(automatic):
                raise ValueError("feasible_tiles contains a tile that is not legal for the selected plant")
            if set(tiles) & set(absolute):
                raise ValueError("feasible_tiles and absolute_infeasible_tiles overlap")
            if status in ("pending", "discarded") and (tiles or set(absolute) - set(automatic)):
                raise ValueError(f"{status} records cannot contain human tile labels")
            record["absolute_infeasible_tiles"] = sorted(set(absolute))
            record["automatic_absolute_infeasible_tiles"] = sorted(set(automatic))
        else:
            metadata = record.get("action_availability", {})
            inferred = tuple(name for name in metadata if name not in ("wait", "dig")) if plant_types is None else plant_types
            names = action_names(inferred)
            feasible = record.get("feasible_actions", [])
            absolute = record.get("absolute_infeasible_actions", [])
            automatic = record.get("automatic_absolute_infeasible_actions", [])
            for value, label in ((feasible, "feasible_actions"), (absolute, "absolute_infeasible_actions"), (automatic, "automatic_absolute_infeasible_actions")):
                if not isinstance(value, list) or any(item not in names for item in value) or len(set(value)) != len(value) or value != sorted(value, key=names.index):
                    raise ValueError(f"{label} must contain unique known action names in canonical order")
            metadata = _action_metadata(record, plant_types)
            derived_automatic = [
                name for name in names if metadata[name]["absolute_infeasible"]
            ]
            # New action snapshots default Dig to absolute-infeasible even on
            # boards where the simulator could currently execute a dig.  Keep
            # pre-default records readable by accepting their availability-
            # derived set as well; newly generated records carry the expanded
            # set and remain locked in the browser.
            derived_default = _automatic_action_labels(record, plant_types)
            if (
                set(automatic) != set(derived_automatic)
                and set(automatic) != set(derived_default)
            ):
                raise ValueError("automatic absolute action labels do not match availability")
            if not set(automatic).issubset(set(absolute)):
                raise ValueError("automatic absolute action labels cannot be removed")
            if set(feasible) & set(absolute):
                raise ValueError("feasible_actions and absolute_infeasible_actions overlap")
            if not feasible and status == "labeled":
                raise ValueError("labeled action records require at least one feasible action")
            if status in ("pending", "discarded") and (feasible or set(absolute) - set(automatic)):
                raise ValueError(f"{status} records cannot contain human action labels")
            record["feasible_actions"] = list(feasible)
            record["absolute_infeasible_actions"] = list(absolute)
            record["automatic_absolute_infeasible_actions"] = list(automatic)
        if status == "pending" and record.get("discard_reason") is not None:
            raise ValueError("pending records cannot contain a discard reason")
        if (
            status == "discarded"
            and record.get("discard_reason") is not None
            and not isinstance(record["discard_reason"], str)
        ):
            raise ValueError("discard_reason must be a string or null")
    return record


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def _atomic_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_manifest(path: str | Path, manifest: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(temporary, path)


def load_records(path: str | Path, *, annotation=False, plant_types=None) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    valid = []
    for row in rows:
        validate_record(row, annotation=annotation, plant_types=plant_types)
        if row["id"] in seen:
            raise ValueError(f"duplicate annotation record id: {row['id']}")
        seen.add(row["id"])
        valid.append(row)
    return valid


def dataset_digest(rows: Iterable[dict]) -> str:
    payload = "\n".join(
        json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def annotation_input_key(record: dict) -> str:
    """Return the model-input identity used for duplicate annotation checks."""
    encoded = record["encoded_observation"]
    kind = annotation_kind(record)
    payload = {
        "kind": kind,
        "plant": record.get("plant_type", "") if kind == "tile" else "",
        "entities": np.asarray(encoded["entities"], np.int32).reshape(-1, 11).tolist(),
        "globals": np.asarray(encoded["globals"], np.float32).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def annotation_conflict_groups(records: Iterable[dict]) -> list[dict]:
    """Group labeled records whose identical inputs have incompatible sets."""
    grouped = {}
    for record in records:
        if record.get("status", "labeled") != "labeled":
            continue
        grouped.setdefault(annotation_input_key(record), []).append(record)
    conflicts = []
    for key, rows in grouped.items():
        if len(rows) < 2:
            continue
        kind = annotation_kind(rows[0])
        feasible_key = "feasible_tiles" if kind == "tile" else "feasible_actions"
        absolute_key = "absolute_infeasible_tiles" if kind == "tile" else "absolute_infeasible_actions"
        intersection = set(rows[0].get(feasible_key, []))
        for row in rows[1:]:
            intersection.intersection_update(row.get(feasible_key, []))
        incompatible = not intersection
        # Every absolute label must rank below every non-absolute label. A
        # cycle in the combined constraints is unsatisfiable.
        universe = set(range(TILES)) if kind == "tile" else set(action_names(rows[0].get("action_availability", {}).keys()))
        edges = set()
        for row in rows:
            absolute = set(row.get(absolute_key, []))
            for low in absolute:
                for high in universe - absolute:
                    edges.add((low, high))
        graph = {}
        for low, high in edges:
            graph.setdefault(low, set()).add(high)
        visiting, visited = set(), set()
        def has_cycle(node):
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(has_cycle(child) for child in graph.get(node, ())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False
        incompatible = incompatible or any(has_cycle(node) for node in graph)
        if not incompatible:
            continue
        conflicts.append({"key": key, "records": list(rows)})
    return conflicts


class AnnotationStore:
    """Atomic latest-record store used by the standalone annotator."""

    def __init__(self, snapshots: str | Path, output: str | Path, *, plant_types=None):
        self.snapshots_path = Path(snapshots)
        self.output_path = Path(output)
        if self.snapshots_path.resolve() == self.output_path.resolve():
            raise ValueError("annotation output must be separate from the snapshot input")
        self.plant_types = tuple(plant_types or ())
        self.snapshots = load_records(
            self.snapshots_path, annotation=False, plant_types=self.plant_types or None
        )
        if not self.snapshots:
            raise ValueError("snapshot set is empty")
        manifest_path = self.snapshots_path.with_suffix(
            self.snapshots_path.suffix + ".manifest.json"
        )
        self.snapshot_manifest = (
            json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
        )
        self.by_id = {row["id"]: dict(row) for row in self.snapshots}
        self.annotations = {}
        if self.output_path.exists():
            for row in load_records(
                self.output_path, annotation=True, plant_types=self.plant_types or None
            ):
                if row["id"] not in self.by_id:
                    raise ValueError(f"annotation is not present in snapshots: {row['id']}")
                self.annotations[row["id"]] = row

    @property
    def total(self):
        return len(self.snapshots)

    def _index(self, index):
        if type(index) is not int or not 0 <= index < self.total:
            raise ValueError("record index is outside the snapshot set")
        return self.snapshots[index]

    def record(self, index):
        source = self._index(index)
        return self.annotations.get(source["id"], source)

    def pending_index(self, start=0):
        for index in range(max(0, start), self.total):
            if self.snapshots[index]["id"] not in self.annotations:
                return index
        for index in range(0, max(0, start)):
            if self.snapshots[index]["id"] not in self.annotations:
                return index
        return None

    def _persist(self):
        rows = [
            self.annotations[row["id"]] for row in self.snapshots if row["id"] in self.annotations
        ]
        _atomic_jsonl(self.output_path, rows)
        manifest = {
            "protocol": ANNOTATION_PROTOCOL,
            "snapshot_protocol": SNAPSHOT_PROTOCOL,
            "total_snapshots": self.total,
            "labeled": sum(row.get("status") == "labeled" for row in rows),
            "discarded": sum(row.get("status") == "discarded" for row in rows),
            "pending": self.total - len(rows),
            "counts": {
                "snapshots": self.total,
                "labeled": sum(row.get("status") == "labeled" for row in rows),
                "discarded": sum(row.get("status") == "discarded" for row in rows),
                "pending": self.total - len(rows),
            },
            "snapshot_digest": dataset_digest(self.snapshots),
            "annotation_digest": dataset_digest(rows),
            "source_digest": self.snapshot_manifest.get("snapshot_digest"),
            "config_digest": self.snapshot_manifest.get("config_digest"),
            "observation_schema": self.snapshot_manifest.get("observation_schema"),
            "rules_hash": self.snapshot_manifest.get("rules_hash")
            or (
                self.snapshot_manifest.get("observation_schema", {}).get("rules_hash")
                if isinstance(self.snapshot_manifest.get("observation_schema"), dict)
                else None
            ),
        }
        write_manifest(
            self.output_path.with_suffix(self.output_path.suffix + ".manifest.json"), manifest
        )

    def commit(self, index, tiles):
        source = self._index(index)
        if source["id"] in self.annotations:
            raise ValueError("record has already been committed")
        if not isinstance(tiles, (list, tuple, set)) or any(
            type(tile) is not int for tile in tiles
        ):
            raise ValueError("feasible tiles must be integers")
        tiles = sorted(set(tiles))
        row = dict(source)
        row.update(
            protocol=ANNOTATION_PROTOCOL,
            status="labeled",
            feasible_tiles=tiles,
            discard_reason=None,
        )
        validate_record(row, annotation=True, plant_types=self.plant_types or None)
        self.annotations[row["id"]] = row
        self._persist()
        return row

    def discard(self, index, reason=None):
        source = self._index(index)
        if source["id"] in self.annotations:
            raise ValueError("record has already been committed")
        row = dict(source)
        row.update(
            protocol=ANNOTATION_PROTOCOL,
            status="discarded",
            feasible_tiles=[],
            discard_reason=None if reason in (None, "") else str(reason),
        )
        validate_record(row, annotation=True, plant_types=self.plant_types or None)
        self.annotations[row["id"]] = row
        self._persist()
        return row


class LiveAnnotationStore:
    """Durable one-at-a-time store used by the on-demand annotator.

    The active map is kept in the same JSONL file as completed records.  A
    pending record is never considered training data, but survives a browser
    or process restart so an interrupted annotation can be resumed exactly.
    """

    def __init__(
        self, output: str | Path, *, plant_types=None, config_digest=None, rules_hash=None,
        kind="tile"
    ):
        self.output_path = Path(output)
        self.manifest_path = self.output_path.with_suffix(
            self.output_path.suffix + ".manifest.json"
        )
        self.plant_types = tuple(plant_types or ())
        if kind not in ("tile", "action"):
            raise ValueError("annotation kind must be tile or action")
        self.kind = kind
        self.config_digest = config_digest
        self.rules_hash = rules_hash
        self.records = (
            load_records(self.output_path, annotation=True, plant_types=self.plant_types or None)
            if self.output_path.is_file()
            else []
        )
        if any(annotation_kind(row) != self.kind for row in self.records):
            raise ValueError("annotation dataset kind does not match the annotator mode")
        self.by_id = {row["id"]: row for row in self.records}
        pending = [row for row in self.records if row.get("status") == "pending"]
        if len(pending) > 1:
            raise ValueError("annotation data contains more than one pending record")
        self._pending_id = pending[0]["id"] if pending else None
        self.manifest = (
            json.loads(self.manifest_path.read_text("utf-8"))
            if self.manifest_path.is_file()
            else {}
        )

    @property
    def pending(self):
        return self.by_id.get(self._pending_id) if self._pending_id is not None else None

    @property
    def last_difficulty(self):
        value = self.manifest.get("last_difficulty")
        return value if isinstance(value, str) else None

    @property
    def total(self):
        return len(self.records)

    def index(self, record_id):
        if not isinstance(record_id, str) or record_id not in self.by_id:
            raise ValueError("unknown annotation record")
        return next(i for i, row in enumerate(self.records) if row["id"] == record_id)

    def record(self, record_id):
        return self.records[self.index(record_id)]

    def _counts(self):
        return {
            "generated": self.total,
            "labeled": sum(row.get("status") == "labeled" for row in self.records),
            "discarded": sum(row.get("status") == "discarded" for row in self.records),
            "pending": sum(row.get("status") == "pending" for row in self.records),
        }

    @property
    def counts(self):
        return self._counts()

    def conflict_groups(self):
        """Return deterministic, user-facing groups of contradictory labels."""
        result = []
        for group in annotation_conflict_groups(self.records):
            rows = group["records"]
            result.append(
                {
                    "key": group["key"],
                    "record": dict(rows[0]),
                    "record_ids": [row["id"] for row in rows],
                    "members": [
                        {
                            "id": row["id"],
                            "revision": row.get("revision", 0),
                            "status": row.get("status"),
                    "feasible_tiles": list(row.get("feasible_tiles", [])),
                    "absolute_infeasible_tiles": list(row.get("absolute_infeasible_tiles", [])),
                    "feasible_actions": list(row.get("feasible_actions", [])),
                    "absolute_infeasible_actions": list(row.get("absolute_infeasible_actions", [])),
                            "source": dict(row.get("source", {})),
                        }
                        for row in rows
                    ],
                    "annotation_kind": annotation_kind(rows[0]),
                    "plant_type": rows[0].get("plant_type", ""),
                    "legal_tiles": list(rows[0].get("availability", {}).get("legal_tiles", range(TILES))),
                }
            )
        return result

    def _persist(self, *, last_difficulty=None, sampler_seed=None):
        if last_difficulty is not None:
            self.manifest["last_difficulty"] = last_difficulty
        if sampler_seed is not None:
            self.manifest["sampler_seed"] = int(sampler_seed)
        counts = self._counts()
        manifest = {
            **self.manifest,
            "protocol": ANNOTATION_PROTOCOL,
            "annotation_kind": self.kind,
            "snapshot_protocol": SNAPSHOT_PROTOCOL,
            "counts": {**counts, "usable_training": counts["labeled"]},
            "total_snapshots": counts["generated"],
            "generated": counts["generated"],
            "labeled": counts["labeled"],
            "discarded": counts["discarded"],
            "pending": counts["pending"],
            "usable_training": counts["labeled"],
            "annotation_digest": dataset_digest(self.records),
            "source_digest": self.manifest.get("source_digest"),
            "split_digest": self.manifest.get("split_digest"),
            "config_digest": self.config_digest or self.manifest.get("config_digest"),
            "rules_hash": self.rules_hash or self.manifest.get("rules_hash"),
        }
        if self.records and self.records[0].get("observation_schema") is not None:
            manifest["observation_schema"] = self.records[0]["observation_schema"]
        _atomic_jsonl(self.output_path, self.records)
        write_manifest(self.manifest_path, manifest)
        self.manifest = manifest

    def set_difficulty(self, difficulty):
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError("difficulty must be a non-empty string")
        self._persist(last_difficulty=difficulty)

    def add_pending(self, record, *, difficulty, sampler_seed=None):
        if self.pending is not None:
            raise ValueError("an annotation map is already pending")
        row = dict(record)
        row.update(
            protocol=ANNOTATION_PROTOCOL,
            annotation_kind=self.kind,
            status="pending",
            discard_reason=None,
            difficulty=difficulty,
        )
        if self.kind == "tile":
            row.update(feasible_tiles=[])
        else:
            automatic = set(row.get("automatic_absolute_infeasible_actions", []))
            # New action records always start with Dig locked as an
            # absolute-infeasible annotation.  The sampler already persists
            # this label; enforcing it here also covers imported snapshots.
            automatic.add("dig")
            row["automatic_absolute_infeasible_actions"] = sorted(
                automatic, key=action_names(self.plant_types).index
            )
            row.update(feasible_actions=[], absolute_infeasible_actions=list(row["automatic_absolute_infeasible_actions"]))
        validate_record(row, annotation=True, plant_types=self.plant_types or None)
        if row["id"] in self.by_id:
            raise ValueError(f"duplicate annotation record id: {row['id']}")
        self.records.append(row)
        self.by_id[row["id"]] = row
        self._pending_id = row["id"]
        self._persist(last_difficulty=difficulty, sampler_seed=sampler_seed)
        return row

    def resolve(self, record_id, *, status, tiles=None, reason=None, revision=0, feasible=None, absolute=None):
        row = self.pending
        if row is None or row["id"] != record_id:
            raise ValueError("stale annotation record")
        return self._save_resolution(row, status, tiles, reason, revision, feasible=feasible, absolute=absolute)

    def revise(self, record_id, *, status, tiles=None, reason=None, revision, feasible=None, absolute=None):
        row = self.record(record_id)
        if row["status"] == "pending":
            raise ValueError("pending annotations must be resolved before revision")
        return self._save_resolution(row, status, tiles, reason, revision, feasible=feasible, absolute=absolute)

    def revise_conflict(self, key, members, tiles=None, *, feasible=None, absolute=None):
        """Atomically apply one corrected label set to a conflict group."""
        groups = self.conflict_groups()
        group = next((item for item in groups if item["key"] == key), None)
        if group is None:
            raise ValueError("stale annotation conflict")
        if not isinstance(members, list) or any(not isinstance(item, dict) for item in members):
            raise ValueError("conflict members must be a list of objects")
        expected = {
            item.get("id"): item.get("revision")
            for item in members
            if isinstance(item.get("id"), str)
        }
        if set(expected) != set(group["record_ids"]) or len(expected) != len(members):
            raise ValueError("stale annotation conflict members")
        for member in group["members"]:
            revision = expected[member["id"]]
            if type(revision) is not int or revision != member["revision"]:
                raise ValueError("stale annotation conflict revision")
        kind = group.get("annotation_kind", self.kind)
        if kind == "action":
            names = action_names(self.plant_types)
            feasible = [] if feasible is None else feasible
            absolute = [] if absolute is None else absolute
            if not isinstance(feasible, list) or not feasible:
                raise ValueError("conflict correction must contain a feasible action")
            if any(item not in names for item in (*feasible, *absolute)):
                raise ValueError("unknown conflict action label")
            feasible = sorted(set(feasible), key=names.index)
            absolute = sorted(set(absolute), key=names.index)
        else:
            if not isinstance(tiles, list) or any(type(tile) is not int for tile in tiles):
                raise ValueError("conflict feasible tiles must be a list of integers")
            tiles = sorted(set(tiles))
            if not tiles:
                raise ValueError("conflict correction must contain at least one tile")
            legal = set(group["legal_tiles"])
            if not set(tiles).issubset(legal):
                raise ValueError("conflict feasible tiles must be legal for the selected plant")
            absolute = [] if absolute is None else absolute
            if not isinstance(absolute, list) or any(type(tile) is not int or not 0 <= tile < TILES for tile in absolute):
                raise ValueError("conflict absolute tiles must be valid indices")

        updates = {}
        for row in self.records:
            if row["id"] not in group["record_ids"]:
                continue
            updated = dict(row)
            updated.update(status="labeled", discard_reason=None, revision=row.get("revision", 0) + 1)
            if kind == "action":
                updated.update(feasible_actions=feasible, absolute_infeasible_actions=sorted(set(absolute) | set(row.get("automatic_absolute_infeasible_actions", [])), key=action_names(self.plant_types).index))
            else:
                automatic = set(row.get("automatic_absolute_infeasible_tiles", []))
                updated.update(feasible_tiles=tiles, absolute_infeasible_tiles=sorted(set(absolute) | automatic))
            validate_record(updated, annotation=True, plant_types=self.plant_types or None)
            updates[row["id"]] = updated
        old_records, old_by_id = self.records, self.by_id
        self.records = [updates.get(row["id"], row) for row in self.records]
        self.by_id = {row["id"]: row for row in self.records}
        try:
            self._persist(last_difficulty=self.last_difficulty)
        except Exception:
            self.records, self.by_id = old_records, old_by_id
            raise
        return [updates[record_id] for record_id in group["record_ids"]]

    def _save_resolution(self, row, status, tiles, reason, revision, *, feasible=None, absolute=None):
        if type(revision) is not int or revision != row.get("revision", 0):
            raise ValueError("stale annotation revision; reload the map before saving")
        if status not in ("labeled", "discarded"):
            raise ValueError("resolution status must be labeled or discarded")
        if self.kind == "action":
            feasible = [] if feasible is None else feasible
            absolute = list(row.get("automatic_absolute_infeasible_actions", [])) if absolute is None else absolute
            if not isinstance(feasible, list) or not isinstance(absolute, list):
                raise ValueError("action labels must be lists")
            if status == "discarded":
                feasible, absolute = [], list(row.get("automatic_absolute_infeasible_actions", []))
            absolute = sorted(set(absolute) | set(row.get("automatic_absolute_infeasible_actions", [])), key=action_names(self.plant_types).index)
            updated = dict(row)
            updated.update(status=status, feasible_actions=sorted(set(feasible), key=action_names(self.plant_types).index), absolute_infeasible_actions=sorted(set(absolute), key=action_names(self.plant_types).index), discard_reason=reason if status == "discarded" else None, revision=revision + 1)
        else:
            if tiles is None:
                tiles = []
            if not isinstance(tiles, (list, tuple, set)) or any(
                type(tile) is not int for tile in tiles
            ):
                raise ValueError("feasible tiles must be integers")
            tiles = sorted(set(tiles))
            if status == "discarded":
                tiles = []
            updated = dict(row)
            automatic = set(row.get("automatic_absolute_infeasible_tiles", []))
            chosen_absolute = automatic if absolute is None else set(absolute) | automatic
            updated.update(status=status, feasible_tiles=tiles, absolute_infeasible_tiles=sorted(chosen_absolute), discard_reason=reason if status == "discarded" else None, revision=revision + 1)
        validate_record(updated, annotation=True, plant_types=self.plant_types or None)
        index = self.index(row["id"])
        self.records[index] = updated
        self.by_id[row["id"]] = updated
        if self._pending_id == row["id"]:
            self._pending_id = None
        self._persist(last_difficulty=self.last_difficulty or updated.get("difficulty"))
        return updated
