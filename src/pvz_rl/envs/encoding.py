"""Compact ID-free public entity records and shared ragged batching."""

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces
from pvz_game import Observation, Rules
from pvz_game.cuda.schema import MOWER_STATES, PLANT_STATES, ZOMBIE_STATES

ENTITY_FIELDS = (
    "type",
    "state",
    "row",
    "x",
    "health",
    "armor",
    "phase_ticks",
    "slow_ticks",
    "has_pole",
    "headless",
    "damage",
)
ENTITY_WIDTH = len(ENTITY_FIELDS)
GLOBAL_FIELDS = (
    "sun",
    "elapsed",
    "wave",
    "total_waves",
    "initial",
    "defeated",
    *(f"cooldown_{i}" for i in range(8)),
    "plant_count",
    "zombie_count",
    "projectile_count",
    "mower_count",
)
GLOBAL_WIDTH = len(GLOBAL_FIELDS)
OBSERVATION_PROTOCOL = "entity_v1"
MINIMUM_PADDING = 32


def width_bucket(count, limit):
    """Fixed transport padding (32, 64, 128, ... capped at ``limit``) holding ``count``.

    Padding never carries information; a few fixed widths keep device buffers and
    compiled inference shapes reusable while preserving every kept entity.
    """
    width = MINIMUM_PADDING
    while width < count:
        width *= 2
    return min(width, limit)


def validate_entity_records(raw):
    """Validate packed host records at archive/storage ingestion, once per slab."""
    kind, state = raw[..., 0], raw[..., 1]
    plants, zombies = len(PLANT_STATES), len(ZOMBIE_STATES)
    correct = (
        ((kind >= 1) & (kind <= 8) & (state >= 1) & (state <= plants))
        | ((kind >= 9) & (kind <= 13) & (state > plants) & (state <= plants + zombies))
        | (((kind == 14) | (kind == 15)) & (state == 0))
        | (
            (kind == 16)
            & (state > plants + zombies)
            & (state <= plants + zombies + len(MOWER_STATES))
        )
    )
    if not np.all(correct):
        raise ValueError("Unknown public entity type/state or state does not match its type")


@dataclass
class PackedEntityBatch:
    entities: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    globals: np.ndarray

    def validate(self):
        if self.offsets.shape != self.counts.shape or self.globals.shape != (
            *self.counts.shape,
            GLOBAL_WIDTH,
        ):
            raise ValueError("Packed observation shapes disagree")
        if self.entities.ndim != 2 or self.entities.shape[1] != ENTITY_WIDTH:
            raise ValueError("Invalid packed entity slab")
        if (
            self.entities.dtype != np.int32
            or self.offsets.dtype != np.int64
            or self.counts.dtype != np.int64
        ):
            raise ValueError("Invalid packed observation dtype")
        if (
            np.any(self.offsets < 0)
            or np.any(self.counts < 0)
            or np.any(self.offsets + self.counts > len(self.entities))
        ):
            raise ValueError("Packed entity offset/count outside slab")
        validate_entity_records(self.entities)
        if not np.isfinite(self.globals).all():
            raise ValueError("Non-finite trajectory globals")


@dataclass
class EntityBatch:
    """Padded batch (or batch/time sequence); padding exists only in transport."""

    entities: torch.Tensor
    entity_mask: torch.Tensor
    globals: torch.Tensor
    validated: bool = False

    def __len__(self):
        return len(self.globals)

    @property
    def shape(self):
        return self.globals.shape[:-1]

    @property
    def device(self):
        return self.globals.device

    def __getitem__(self, index):
        return EntityBatch(
            self.entities[index], self.entity_mask[index], self.globals[index], self.validated
        )

    def clone(self):
        return EntityBatch(*(x.clone() for x in self.tensors()))

    def tensors(self):
        return self.entities, self.entity_mask, self.globals

    def to(self, device, *, non_blocking=False):
        return EntityBatch(
            *(x.to(device, non_blocking=non_blocking) for x in self.tensors()),
            validated=self.validated,
        )

    def cpu(self):
        return self.to("cpu")

    def reshape(self, *shape):
        n = self.entities.shape[-2]
        globals_ = self.globals.reshape(*shape, GLOBAL_WIDTH)
        batch_shape = globals_.shape[:-1]
        return EntityBatch(
            self.entities.reshape(*batch_shape, n, ENTITY_WIDTH),
            self.entity_mask.reshape(*batch_shape, n),
            globals_,
            self.validated,
        )

    def observations(self):
        """Unpad a flattened batch for storage and public API returns."""
        host = self.reshape(-1).cpu()
        return [
            {"entities": e[m].numpy().copy(), "globals": g.numpy().copy()}
            for e, m, g in zip(*host.tensors(), strict=True)
        ]


def collate_observations(observations, device=None):
    if isinstance(observations, EntityBatch):
        return observations if device is None else observations.to(device)
    if isinstance(observations, dict):
        observations = [observations]
    observations = list(observations)
    count = len(observations)
    width = max((len(o["entities"]) for o in observations), default=0)
    entities = np.zeros((count, width, ENTITY_WIDTH), np.int32)
    valid = np.zeros((count, width), bool)
    globals_ = np.zeros((count, GLOBAL_WIDTH), np.float32)
    for i, obs in enumerate(observations):
        raw = np.asarray(obs["entities"])
        if raw.size == 0:
            raw = raw.reshape(0, ENTITY_WIDTH)
        if raw.ndim != 2 or raw.shape[1] != ENTITY_WIDTH:
            raise ValueError("entities must have shape (n, 11)")
        if not np.isfinite(raw).all() or not np.equal(raw, raw.astype(np.int32)).all():
            raise ValueError("entity fields must be finite int32 values")
        g = np.asarray(obs["globals"], dtype=np.float32)
        if g.shape != (GLOBAL_WIDTH,) or not np.isfinite(g).all():
            raise ValueError("globals must contain 18 finite values")
        entities[i, : len(raw)] = raw
        valid[i, : len(raw)] = True
        globals_[i] = g
    return EntityBatch(
        torch.as_tensor(entities, device=device),
        torch.as_tensor(valid, device=device),
        torch.as_tensor(globals_, device=device),
    )


def observation_json(observation):
    return {name: np.asarray(observation[name]).tolist() for name in ("entities", "globals")}


def observations_equal(left, right, *, atol=0):
    return all(
        np.asarray(left[k]).shape == np.asarray(right[k]).shape
        and np.allclose(left[k], right[k], atol=atol, rtol=0)
        for k in ("entities", "globals")
    )


class ObservationEncoder:
    def __init__(self, cfg: dict, rules: Rules):
        if cfg["encoding"]["version"] != OBSERVATION_PROTOCOL:
            raise ValueError("Retired observations; entity_v1 requires fresh initialization")
        self.cfg, self.rules = cfg, rules
        env = cfg["environment"]
        self.rows, self.cols = env["rows"], env["cols"]
        names = [
            *(f"plant:{k}" for k in env["plants"]),
            *(f"zombie:{k}" for k in env["zombies"]),
            "projectile:pea",
            "projectile:icy",
            "mower",
        ]
        self.types = {name: i + 1 for i, name in enumerate(names)}
        self.plants = {k: self.types[f"plant:{k}"] for k in env["plants"]}
        self.zombies = {k: self.types[f"zombie:{k}"] for k in env["zombies"]}
        states = [
            *(f"plant:{s}" for s in PLANT_STATES),
            *(f"zombie:{s}" for s in ZOMBIE_STATES),
            *(f"mower:{s}" for s in MOWER_STATES),
        ]
        self.states = {name: i + 1 for i, name in enumerate(states)}
        self.global_fields = {name: i for i, name in enumerate(GLOBAL_FIELDS)}
        self.global_width = GLOBAL_WIDTH
        self.count_scale = cfg["encoding"]["count_scale"]
        self.max_entities = cfg["encoding"]["max_entities"]
        self.cost_scale = max(p["cost"] for p in rules.plants.values())
        self.position_scale = rules.game["spawn_x"] - rules.game["house_x"]
        self.armor_scale = max(1, *(z["armor"] for z in rules.zombies.values()))
        self.health_scales = np.ones(len(self.types) + 1, np.float32)
        for kinds, specs in ((self.plants, rules.plants), (self.zombies, rules.zombies)):
            for name, code in kinds.items():
                self.health_scales[code] = specs[name]["health"]
        self.space = spaces.Dict(
            {
                "entities": spaces.Sequence(
                    spaces.Box(
                        np.iinfo(np.int32).min,
                        np.iinfo(np.int32).max,
                        shape=(ENTITY_WIDTH,),
                        dtype=np.int32,
                    ),
                    stack=True,
                ),
                "globals": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_WIDTH,), dtype=np.float32),
            }
        )

    def schema(self):
        return dict(
            version=OBSERVATION_PROTOCOL,
            fields=list(ENTITY_FIELDS),
            globals=list(GLOBAL_FIELDS),
            types=self.types,
            states=self.states,
            normalization="pinned_rules_v1",
            rules_hash=self.rules.digest,
            max_entities=self.max_entities,
            selection="mowers_plants_zombies_projectiles_v1",
        )

    def encode(self, obs: Observation):
        rows = []
        unit = self.rules.game["units_per_tile"]
        try:
            for p in obs.plants:
                rows.append(
                    (
                        self.plants[p.plant_type],
                        self.states[f"plant:{p.state}"],
                        p.row,
                        p.col * unit + unit // 2,
                        p.health,
                        0,
                        p.timer_ticks,
                        0,
                        0,
                        0,
                        0,
                    )
                )
            for z in obs.zombies:
                rows.append(
                    (
                        self.zombies[z.zombie_type],
                        self.states[f"zombie:{z.state}"],
                        z.row,
                        z.x,
                        z.health,
                        z.armor,
                        z.timer_ticks,
                        z.slow_ticks,
                        z.has_pole,
                        z.headless,
                        0,
                    )
                )
            for p in obs.projectiles:
                rows.append(
                    (
                        self.types["projectile:icy" if p.icy else "projectile:pea"],
                        0,
                        p.row,
                        p.x,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        p.damage,
                    )
                )
            for m in obs.mowers:
                rows.append(
                    (
                        self.types["mower"],
                        self.states[f"mower:{m.state}"],
                        m.row,
                        m.x,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                )
        except KeyError as exc:
            raise ValueError(f"Unknown public entity category: {exc}") from exc

        def priority(record):
            kind = record[0]
            group = 0 if kind == 16 else 1 if kind <= 8 else 2 if kind <= 13 else 3
            lane = record[2] if group <= 1 else 0
            return group, lane, record[3], *record

        retained = sorted(rows, key=priority)[: self.max_entities]
        if any(not np.iinfo(np.int32).min <= v <= np.iinfo(np.int32).max for r in rows for v in r):
            raise ValueError("Public entity fields exceed int32 range")
        raw = np.asarray(retained, dtype=np.int32).reshape(-1, ENTITY_WIDTH)
        self.last_truncation = {
            "total": len(rows) - len(retained),
            **{
                name: sum(lo <= r[0] <= hi for r in rows) - sum(lo <= r[0] <= hi for r in retained)
                for name, lo, hi in (("plants", 1, 8), ("zombies", 9, 13), ("projectiles", 14, 15))
            },
        }
        cards = {c.plant_type: c for c in obs.cards}
        cooldowns = []
        for name in self.plants:
            if name not in cards or cards[name].recharge_ticks < 0:
                raise ValueError(f"Missing or invalid public card {name!r}")
            c = cards[name]
            cooldowns.append(c.cooldown_ticks / (c.recharge_ticks + 1))
        globals_ = np.asarray(
            [
                obs.sun / self.cost_scale,
                obs.elapsed_seconds / self.cfg["environment"]["cutoff_seconds"],
                obs.wave / self.cfg["encoding"]["wave_scale"],
                obs.total_waves / self.cfg["encoding"]["wave_scale"],
                obs.counts.initial_total / self.count_scale,
                obs.counts.defeated / self.count_scale,
                *cooldowns,
                *(
                    len(xs) / self.count_scale
                    for xs in (obs.plants, obs.zombies, obs.projectiles, obs.mowers)
                ),
            ],
            dtype=np.float32,
        )
        return {"entities": raw, "globals": globals_}

    def validate(self, batch):
        """Validate once at ingestion, outside encoder microbatches/recomputation."""
        if batch.validated:
            return
        raw = batch.entities
        kind, state = raw[..., 0].long(), raw[..., 1].long()
        valid = batch.entity_mask
        invalid_type = valid & ((kind < 1) | (kind > len(self.types)))
        invalid_state = valid & ((state < 0) | (state > len(self.states)))
        correct_state = (
            ((kind <= 8) & (state >= 1) & (state <= len(PLANT_STATES)))
            | (
                (kind >= 9)
                & (kind <= 13)
                & (state > len(PLANT_STATES))
                & (state <= len(PLANT_STATES) + len(ZOMBIE_STATES))
            )
            | (((kind == 14) | (kind == 15)) & (state == 0))
            | ((kind == 16) & (state > len(PLANT_STATES) + len(ZOMBIE_STATES)))
        )
        if torch.any(invalid_type | invalid_state | (valid & ~correct_state)):
            raise ValueError("Unknown public entity type/state or state does not match its type")
        batch.validated = True

    def normalize(self, batch, *, health_scales=None):
        self.validate(batch)
        raw = batch.entities
        kind, state, valid = raw[..., 0].long(), raw[..., 1].long(), batch.entity_mask
        kind = torch.where(valid, kind, 0)
        state = torch.where(valid, state, 0)
        values = torch.where(valid[..., None], raw[..., 2:], 0).float()
        scales = (
            torch.as_tensor(self.health_scales, device=raw.device)
            if health_scales is None
            else health_scales
        )
        values[..., 0] /= max(1, self.rows - 1)
        values[..., 1] = (values[..., 1] - self.rules.game["house_x"]) / self.position_scale
        values[..., 2] /= scales[kind]
        values[..., 3] /= self.armor_scale
        values[..., 4:6] /= self.rules.game["tick_rate"]
        values[..., 8] /= self.rules.plants["peashooter"]["damage"]
        return kind, state, values
