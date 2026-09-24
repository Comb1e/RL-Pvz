"""Device encoders/rewards from the versioned public-observation schema."""

from importlib.resources import files

import torch
from pvz_game.cuda.backend import kernel_source

from .actions import ActionSchema
from .cuda_diagnostics import DeviceProfiler
from .encoding import ObservationEncoder
from .rewards import LEDGER_METRICS, REWARD_METRICS

REWARD_FIELDS = (*REWARD_METRICS, "total")
METRIC_INDICES = (*range(20, 29), *range(81, 81 + len(REWARD_METRICS) - 9))
LEDGER_INDICES = dict(zip(LEDGER_METRICS, range(max(METRIC_INDICES) + 1, max(METRIC_INDICES) + 5)))
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
            "ACTION_DIG_START": ActionSchema.dig_start,
            "ACTION_COUNT": ActionSchema.size,
            "ACTION_TILES": ActionSchema.tiles,
            "BINS": encoder.bins,
            "ZOMBIE_WIDTH": encoder.zombie_width,
            "ZOMBIE_OFFSET": encoder.slices["zombies"].start,
            "EMPTY_DISTANCE": encoder.empty_distance,
            "OBS_SIZE": encoder.size,
            "LOCAL_COUNT": encoder.local_count_scale,
            "HP_SCALE": encoder.hp_scale,
            "ARMOR_SCALE": max(1, encoder.armor_scale),
            "POSITION_SCALE": encoder.position_scale,
            "COST_SCALE": encoder.cost_scale,
            "CUTOFF_SECONDS": cfg["environment"]["cutoff_seconds"],
            "EARLY_DIG_TICKS": cfg.get("diagnostics", {}).get("early_dig_seconds", 5)
            * batch.rules.game["tick_rate"],
            "WAVE_SCALE": cfg["encoding"]["wave_scale"],
            "COUNT_SCALE": encoder.count_scale,
            "GLOBAL_OFFSET": encoder.slices["globals"].start,
            "GAMMA": cfg["training"]["gamma"],
            "BASIC_HP": batch.rules.zombies["basic"]["health"],
            "REWARD_SIZE": len(REWARD_FIELDS),
            "METRIC_SIZE": METRIC_SIZE,
        }
        params.update({f"Z_{k}": v for k, v in encoder.zombie_fields.items()})
        params.update({f"O_{k}": v for k, v in encoder.global_fields.items()})
        reward_keys = (
            "win_reward",
            "loss_penalty",
            "basic_zombie_value",
            "mower_value",
            "progress_weight",
            "value_scale",
        )
        params.update({f"R_{k}": float(cfg["reward"][k]) for k in reward_keys})
        params.update({f"F_{k}": i for i, k in enumerate(REWARD_FIELDS)})
        params.update({f"T_{k}": i for k, i in zip(REWARD_METRICS, METRIC_INDICES)})
        params.update({f"T_{k}": i for k, i in LEDGER_INDICES.items()})
        source = kernel_source(batch.rules, batch.zcap, batch.qcap, batch.ecap, batch.diagnostic)
        source += "\n" + "\n".join(f"#define {k} {v}" for k, v in params.items())
        source += "\n" + files("pvz_rl").joinpath("cuda_features.cu").read_text("utf-8").replace(
            "METRIC_ACCUMULATION", "\n".join(f"t[T_{k}] += v[F_{k}];" for k in REWARD_METRICS)
        )
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self.encode_kernel = self.module.get_function("encode_state")
        self.reward_kernel = self.module.get_function("reward_metrics")
        self.observations = cp.zeros((batch.n, encoder.size), cp.float32)
        self.assets = cp.zeros(batch.n, cp.float64)
        self.rewards = cp.zeros(batch.n, cp.float32)
        self.parts = cp.zeros((batch.n, len(REWARD_FIELDS)), cp.float64)
        self.totals = cp.zeros((batch.n, METRIC_SIZE), cp.float64)
        self.totals[:, 11] = -1
        self.obs_tensor = torch.from_dlpack(self.observations)
        self.reward_tensor = torch.from_dlpack(self.rewards)
        self.mask_tensor = torch.from_dlpack(batch.masks)

    def encode(self):
        b = self.batch
        self.encode_kernel(
            (b.n,),
            (32,),
            (
                b.header,
                b.plants,
                b.zombies,
                b.mowers,
                self.observations,
                self.assets,
                b.n,
            ),
        )
        b.action_masks_device()
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
