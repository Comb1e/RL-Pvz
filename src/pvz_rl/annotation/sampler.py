"""Generate reachable, plant-available annotation snapshots."""

from __future__ import annotations

import dataclasses
import random
import secrets
import uuid
from pathlib import Path

from pvz_game import Place, Status, Wait

from pvz_rl.annotation.dataset import (
    SNAPSHOT_PROTOCOL,
    _atomic_jsonl,
    _automatic_action_labels,
    _jsonable,
    action_names,
    dataset_digest,
    validate_record,
    write_manifest,
)
from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.envs.env import PvZEnv


class ReachableSnapshotSampler:
    """Generate one usable, reachable annotation state at a time."""

    def __init__(
        self,
        cfg=None,
        *,
        seed=None,
        family="preset",
        max_steps=None,
        max_attempts=100,
    ):
        self.cfg = cfg or load_config()
        validate_config(self.cfg)
        settings = {
            **load_config()["training"]["annotation"],
            **self.cfg["training"].get("annotation", {}),
        }
        if max_steps is None:
            max_steps = settings["sampling_max_steps"]
        if family not in ("preset", "diagnostic", "redistributed", "faster", "concentrated"):
            raise ValueError(f"unknown annotation scenario family: {family}")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if seed is None:
            seed = secrets.randbits(64)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer or None")
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.family = family
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.plant_interval = tuple(settings["sampling_plant_interval_steps"])
        self.max_plants = settings["sampling_max_plants"]
        self.episodes = 0
        self.env = PvZEnv(
            self.cfg,
            level=self.cfg["evaluation"]["levels"][0],
            family=family,
            training=False,
        )

    @property
    def levels(self):
        return tuple(self.cfg["evaluation"]["levels"])

    def sample(self, level, *, annotation_kind="tile"):
        """Return one validated record for ``level`` or raise a bounded error."""
        if level not in self.levels:
            raise ValueError(f"unknown annotation difficulty: {level}")
        if annotation_kind not in ("tile", "action"):
            raise ValueError("annotation_kind must be tile or action")
        for _attempt in range(self.max_attempts):
            self.episodes += 1
            scenario_seed = self.rng.randrange(1, 2**31 - 1)
            self.env.level, self.env.family = level, self.family
            self.env.reset(seed=scenario_seed, options={"level": level, "family": self.family})
            group_id = f"{self.family}:{level}:{scenario_seed}"
            target_steps = _warmup_steps(self.rng, self.max_steps)
            next_plant_step = self.rng.randint(*self.plant_interval)
            for step in range(target_steps):
                if self.env.state != "running" or self.env.public.status != Status.RUNNING:
                    break
                action = Wait()
                if step >= next_plant_step:
                    action = _rollout_action(self.env, self.rng, max_plants=self.max_plants)
                    # A failed planting attempt also waits a full interval;
                    # newly available sunlight is never spent immediately.
                    next_plant_step = step + self.rng.randint(*self.plant_interval)
                self.env.step(self.env.codec.encode(action))
            else:
                # Inspect the endpoint only. Never reuse a pre-loss frame or
                # continue beyond the randomly selected simulation limit.
                if self.env.state != "running" or self.env.public.status != Status.RUNNING:
                    continue
                # Tile records deliberately condition on one usable card so
                # their plot labels have a concrete branch.  Action records
                # label the complete species branch, so a sampler-selected
                # plant is neither required nor supervision; any running
                # endpoint is useful, including boards with no usable card.
                candidate = None
                if annotation_kind == "tile":
                    available = _available(self.env)
                    if not available:
                        continue
                    candidate = self.rng.choice(available)["plant_type"]
                record = snapshot_record(
                    self.env,
                    plant_type=candidate,
                    group_id=group_id,
                    record_id=f"annotation-{uuid.uuid4().hex}",
                    annotation_kind=annotation_kind,
                )
                record["difficulty"] = level
                record["source"]["simulation_steps"] = target_steps
                return record
        raise RuntimeError(
            f"could not generate a usable {level} annotation map after "
            f"{self.max_attempts} attempts of {self.max_steps} steps"
        )


def _public_observation(observation):
    if dataclasses.is_dataclass(observation):
        return _jsonable(dataclasses.asdict(observation))
    if hasattr(observation, "_asdict"):
        return _jsonable(observation._asdict())
    raise TypeError("the game observation must be a dataclass or named tuple")


def _available(env):
    legal = env.public_board().legal
    by_plant = {kind: [] for kind in env.codec.plants}
    for action in legal:
        if isinstance(action, Place):
            by_plant.setdefault(action.plant_type, []).append(
                int(action.row) * env.codec.cols + int(action.col)
            )
    cards = {card.plant_type: card for card in env.public.cards}
    result = []
    for kind, tiles in by_plant.items():
        card = cards.get(kind)
        if card is None or card.cooldown_ticks != 0 or env.public.sun < card.cost or not tiles:
            continue
        result.append(
            {
                "plant_type": kind,
                "available": True,
                "cost": int(card.cost),
                "sun": int(env.public.sun),
                "cooldown_ticks": int(card.cooldown_ticks),
                "legal_tiles": sorted(set(tiles)),
            }
        )
    return result


def _warmup_steps(rng, max_steps):
    """Draw a uniform, inclusive simulation limit, including the reset state."""
    return rng.randint(0, max_steps)


def _rollout_action(env, rng, *, max_plants=None):
    """Draw wait or any species equally; unavailable plant draws become wait."""
    kind = rng.choice((None, *env.codec.plants))
    if kind is None or (max_plants is not None and len(env.public.plants) >= max_plants):
        return Wait()
    available = next((row for row in _available(env) if row["plant_type"] == kind), None)
    if available is None:
        return Wait()
    # _available sorts legal tiles, keeping seeded rollouts independent of the
    # public board's frozenset iteration order. Never redraw a failed species.
    tile = rng.choice(available["legal_tiles"])
    return Place(kind, *divmod(tile, env.codec.cols))


def _action_availability(env):
    cards = {card.plant_type: card for card in env.public.cards}
    usable = {row["plant_type"] for row in _available(env)}
    result = {"wait": {"feasible": True, "absolute_infeasible": False, "reason": None}}
    for kind in env.codec.plants:
        card = cards.get(kind)
        if card is None:
            reason = "absent_card"
        elif card.cooldown_ticks:
            reason = "cooldown"
        elif env.public.sun < card.cost:
            reason = "insufficient_sun"
        elif kind not in usable:
            reason = "no_legal_placement"
        else:
            reason = None
        legal_tiles = sorted(set(next((row["legal_tiles"] for row in _available(env) if row["plant_type"] == kind), [])))
        result[kind] = {"feasible": reason is None, "absolute_infeasible": reason is not None, "reason": reason, "legal_tiles": legal_tiles, "cost": int(card.cost) if card else None, "cooldown_ticks": int(card.cooldown_ticks) if card else None}
    result["dig"] = {"feasible": bool(env.public.plants), "absolute_infeasible": not bool(env.public.plants), "reason": None if env.public.plants else "empty_board"}
    return result


def snapshot_record(env, *, plant_type=None, group_id, record_id, annotation_kind="tile"):
    if env.state != "running" or env.public.status != Status.RUNNING:
        raise ValueError("annotation snapshots must come from a running game")
    if annotation_kind not in ("tile", "action"):
        raise ValueError("annotation_kind must be tile or action")
    available = {row["plant_type"]: row for row in _available(env)}
    if annotation_kind == "tile" and plant_type not in available:
        raise ValueError(f"plant is not currently available: {plant_type}")
    observation = env.public
    encoder = env.encoder
    encoded = encoder.encode(observation)
    record = {
        "protocol": SNAPSHOT_PROTOCOL,
        "id": record_id,
        "group_id": group_id,
        "plant_type": plant_type,
        "observation": _public_observation(observation),
        "encoded_observation": _jsonable(encoded),
        "observation_schema": encoder.schema(),
        "rules_hash": env.rules.digest,
        # Action records do not carry a sampler-selected branch.  Keep the
        # generic field for protocol compatibility, but action fitting uses
        # action_availability below instead.
        "availability": available[plant_type] if annotation_kind == "tile" else {},
        "source": {
            "level": env.episode_level,
            "family": env.episode_family,
            "seed": int(env.episode_seed),
            "tick": int(observation.tick),
        },
    }
    if annotation_kind == "action":
        record["annotation_kind"] = "action"
        metadata = _action_availability(env)
        record["action_availability"] = metadata
        record["branch_availability"] = metadata
        # Dig is intentionally an annotation default rather than a sampler
        # choice.  Even when a board currently contains a plant, the action
        # annotator starts with Dig in the absolute-infeasible set so users
        # must explicitly opt into another action set instead of inheriting a
        # rollout artifact.  It is a human-editable default, so it must not be
        # part of the automatic (locked) labels.  Keep the simulator's actual
        # dig feasibility in ``action_availability`` for display and audit
        # purposes.
        automatic = set(_automatic_action_labels(record, env.codec.plants))
        record["automatic_absolute_infeasible_actions"] = sorted(
            automatic, key=action_names(env.codec.plants).index
        )
        record["absolute_infeasible_actions"] = sorted(
            automatic | {"dig"}, key=action_names(env.codec.plants).index
        )
        record["feasible_actions"] = []
    else:
        record["annotation_kind"] = "tile"
        legal_tiles = set(record["availability"]["legal_tiles"])
        record["automatic_absolute_infeasible_tiles"] = sorted(set(range(45)) - legal_tiles)
        record["absolute_infeasible_tiles"] = list(record["automatic_absolute_infeasible_tiles"])
        record["feasible_tiles"] = []
    validate_record(record, plant_types=env.codec.plants)
    return record


def generate_snapshots(
    output: str | Path,
    *,
    cfg=None,
    count=100,
    seed=101,
    levels=None,
    family="preset",
    max_steps=None,
    output_manifest=None,
):
    """Write ``count`` reachable snapshots and a reproducible manifest.

    States are sampled after legal actions from the pinned simulator.  The
    generator never edits engine arrays directly, so every snapshot can be
    reconstructed by the normal observation encoder.
    """
    cfg = cfg or load_config()
    validate_config(cfg)
    if type(count) is not int or count < 1:
        raise ValueError("count must be positive")
    levels = tuple(levels or cfg["evaluation"]["levels"])
    unknown = set(levels) - set(cfg["evaluation"]["levels"])
    if unknown:
        raise ValueError(f"unknown levels: {sorted(unknown)}")
    rng = random.Random(seed)
    sampler = ReachableSnapshotSampler(cfg, seed=seed, family=family, max_steps=max_steps)
    rows = [sampler.sample(rng.choice(levels)) for _ in range(count)]
    env = sampler.env
    episode = sampler.episodes
    output = Path(output)
    _atomic_jsonl(output, rows)
    manifest = {
        "protocol": SNAPSHOT_PROTOCOL,
        "count": len(rows),
        "counts": {"snapshots": len(rows), "episodes": episode},
        "episodes": episode,
        "seed": seed,
        "levels": list(levels),
        "family": family,
        "max_steps": sampler.max_steps,
        "config_digest": digest(cfg),
        "observation_schema": ObservationEncoder(cfg, env.rules).schema(),
        "rules_hash": env.rules.digest,
        "source_digest": None,
        "snapshot_digest": dataset_digest(rows),
    }
    write_manifest(
        output_manifest or output.with_suffix(output.suffix + ".manifest.json"), manifest
    )
    return manifest


def generate_from_json(output, **kwargs):
    """Convenience wrapper used by command-line dispatch and tests."""
    return generate_snapshots(output, **kwargs)
