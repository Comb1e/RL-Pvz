"""Device encoders/rewards from the versioned public-observation schema."""

from importlib.resources import files

import torch
from pvz_game.cuda.backend import kernel_source
from pvz_game.cuda.schema import REASONS

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.encoding import (
    ENTITY_FIELDS,
    ENTITY_WIDTH,
    GLOBAL_WIDTH,
    EntityBatch,
    ObservationEncoder,
)
from pvz_rl.envs.rewards import LEDGER_METRICS, REWARD_METRICS
from pvz_rl.monitoring.cuda_diagnostics import DeviceProfiler

REWARD_FIELDS = (*REWARD_METRICS, "total")
METRIC_INDICES = (*range(20, 29), *range(81, 81 + len(REWARD_METRICS) - 9))
LEDGER_INDICES = {name: max(METRIC_INDICES) + 1 + i for i, name in enumerate(LEDGER_METRICS)}
METRIC_SIZE = (
    max(LEDGER_INDICES.values()) + 1
)  # Preserve original totals, early digs and planting timestamps.


class CudaFeatures:
    def __init__(self, batch, cfg, condition):
        self.batch, self.cfg = batch, cfg
        cp = batch.cp
        self.profiler = DeviceProfiler(cp, cfg.get("simulation", {}).get("profile", False))
        encoder = self.encoder = ObservationEncoder(cfg, batch.rules)
        params = {
            "ENTITY_WIDTH": ENTITY_WIDTH,
            "GLOBAL_WIDTH": GLOBAL_WIDTH,
            "ACTION_DIG_START": ActionSchema.dig_start,
            "ACTION_COUNT": ActionSchema.size,
            "ACTION_TILES": ActionSchema.tiles,
            "ZOMBIE_TYPE_START": min(encoder.zombies.values()),
            "ZOMBIE_STATE_START": encoder.states["zombie:walking"],
            "PROJECTILE_TYPE_START": encoder.types["projectile:pea"],
            "MOWER_TYPE": encoder.types["mower"],
            "MOWER_STATE_START": encoder.states["mower:ready"],
            "COST_SCALE": encoder.cost_scale,
            "CUTOFF_SECONDS": cfg["environment"]["cutoff_seconds"],
            "EARLY_DIG_TICKS": cfg.get("diagnostics", {}).get("early_dig_seconds", 5)
            * batch.rules.game["tick_rate"],
            "WAVE_SCALE": cfg["encoding"]["wave_scale"],
            "COUNT_SCALE": encoder.count_scale,
            "GAMMA": cfg["training"]["gamma"],
            "BASIC_HP": batch.rules.zombies["basic"]["health"],
            "REWARD_SIZE": len(REWARD_FIELDS),
            "METRIC_SIZE": METRIC_SIZE,
            "EMPTY_TILE_REASON": REASONS.index("empty_tile"),
        }
        params.update({f"O_{k}": v for k, v in encoder.global_fields.items()})
        params.update({f"E_{k}": i for i, k in enumerate(ENTITY_FIELDS)})
        for index, kind in enumerate(encoder.plants):
            params[f"CD_{index}"] = batch.rules.plants[kind]["recharge_ticks"] + 1
        reward_keys = (
            "win_reward",
            "loss_penalty",
            "basic_zombie_value",
            "mower_value",
            "progress_weight",
            "value_scale",
            "invalid_plant_penalty",
            "empty_dig_penalty",
        )
        params.update({f"R_{k}": float(cfg["reward"].get(k, 0)) for k in reward_keys})
        params.update({f"F_{k}": i for i, k in enumerate(REWARD_FIELDS)})
        params.update({f"T_{k}": i for k, i in zip(REWARD_METRICS, METRIC_INDICES)})
        params.update({f"T_{k}": i for k, i in LEDGER_INDICES.items()})
        source = kernel_source(batch.rules, batch.zcap, batch.qcap, batch.ecap, batch.diagnostic)
        source += "\n" + "\n".join(f"#define {k} {v}" for k, v in params.items())
        source += "\n" + files("pvz_rl.envs").joinpath("cuda_features.cu").read_text(
            "utf-8"
        ).replace(
            "METRIC_ACCUMULATION", "\n".join(f"t[T_{k}] += v[F_{k}];" for k in REWARD_METRICS)
        )
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self.encode_kernel = self.module.get_function("encode_state")
        self.policy_mask_kernel = self.module.get_function("policy_masks")
        self.reward_kernel = self.module.get_function("reward_metrics")
        self.globals = cp.zeros((batch.n, encoder.global_width), cp.float32)
        self.obs_tensor = None
        self.assets = cp.zeros(batch.n, cp.float64)
        self.rewards = cp.zeros(batch.n, cp.float32)
        self.parts = cp.zeros((batch.n, len(REWARD_FIELDS)), cp.float64)
        self.totals = cp.zeros((batch.n, METRIC_SIZE), cp.float64)
        self.totals[:, 11] = -1
        self.reward_tensor = torch.from_dlpack(self.rewards)
        self.mask_tensor = torch.from_dlpack(batch.masks)

    def encode(self):
        b = self.batch
        cp = b.cp
        counts = b.header[:, 9] + b.header[:, 10] + b.header[:, 11] + 5
        width = int(counts.max().get())
        if width < 5 or width > 45 + b.zcap + b.qcap + 5:
            raise ValueError("Invalid public entity count in CUDA state")
        entities = cp.zeros((b.n, width, ENTITY_WIDTH), cp.int32)
        self.encode_kernel(
            (b.n,),
            (128,),
            (
                b.header,
                b.plants,
                b.zombies,
                b.projectiles,
                b.mowers,
                b.cooldowns,
                entities,
                self.globals,
                self.assets,
                width,
                b.n,
            ),
        )
        valid = cp.arange(width)[None] < counts[:, None]
        # Lexicographic public-field ordering, with padding sorted last.
        kind = entities[:, :, 0]
        group = cp.where(kind == 16, 0, cp.where(kind <= 8, 1, cp.where(kind <= 13, 2, 3)))
        lane = cp.where(group <= 1, entities[:, :, 2], 0)
        keys = [entities[:, :, j].ravel() for j in reversed(range(ENTITY_WIDTH))]
        keys += [
            entities[:, :, 3].ravel(),
            lane.ravel(),
            group.ravel(),
            (~valid).ravel(),
            cp.repeat(cp.arange(b.n), width),
        ]
        order = cp.lexsort(cp.stack(keys))
        entities = entities.reshape(-1, ENTITY_WIDTH)[order].reshape(b.n, width, ENTITY_WIDTH)
        limit = self.encoder.max_entities
        kept = cp.minimum(counts, limit)
        self.truncation_counts = cp.stack(
            (
                cp.maximum(0, b.header[:, 9] - (limit - 5)),
                cp.maximum(0, b.header[:, 10] - cp.maximum(0, limit - 5 - b.header[:, 9])),
                cp.maximum(
                    0, b.header[:, 11] - cp.maximum(0, limit - 5 - b.header[:, 9] - b.header[:, 10])
                ),
            ),
            axis=1,
        )
        if width > limit:
            entities = cp.ascontiguousarray(entities[:, :limit])
            width = limit
        valid = cp.arange(width)[None] < kept[:, None]
        self.obs_tensor = EntityBatch(
            torch.from_dlpack(entities),
            torch.from_dlpack(valid),
            torch.from_dlpack(self.globals.copy()),
        )
        self.policy_mask_kernel(
            ((b.n * ActionSchema.size + 255) // 256,),
            (256,),
            (b.header, b.plants, b.masks, b.n),
        )
        return self.obs_tensor

    def step(self, actions, *, ticks=1, per_tick=True):
        b = self.batch
        before_assets = self.assets.copy()
        before_header, before_cd = b.header.copy(), b.cooldowns.copy()
        with self.profiler.track("simulation"):
            b.step_device(actions, ticks=ticks, per_tick=per_tick)
        with self.profiler.track("encoding_masks"):
            self.encode()
        with self.profiler.track("rewards_metrics"):
            self.reward_kernel(
                ((b.n + 63) // 64,),
                (64,),
                (
                    b.header,
                    before_header,
                    before_cd,
                    actions,
                    b.facts,
                    b.accounting,
                    before_assets,
                    self.assets,
                    self.rewards,
                    self.parts,
                    self.totals,
                    b.n,
                ),
            )
        return self.obs_tensor, self.reward_tensor
