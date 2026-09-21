"""Device encoders/rewards from the versioned public-observation schema."""

from importlib.resources import files

import torch
from pvz_game.cuda.backend import kernel_source

from .cuda_diagnostics import DeviceProfiler
from .encoding import ObservationEncoder
from .rewards import REWARD_METRICS

REWARD_FIELDS = ("terminal", "shaping", *REWARD_METRICS, "total")
METRIC_SIZE = 81  # 35 original totals + early digs + 45 planting timestamps


class CudaFeatures:
    def __init__(self, batch, cfg, condition):
        self.batch, self.cfg = batch, cfg
        cp = batch.cp
        self.profiler = DeviceProfiler(cp, cfg.get("simulation", {}).get("profile", False))
        encoder = self.encoder = ObservationEncoder(cfg, batch.rules)
        params = {
            "BINS": encoder.bins,
            "OBS_SIZE": encoder.size,
            "TACTICAL": int(encoder.tactical),
            "TIMER_SCALE": encoder.timer_scale,
            "LOCAL_COUNT": encoder.local_count_scale,
            "HP_SCALE": encoder.hp_scale,
            "ARMOR_SCALE": max(1, encoder.armor_scale),
            "POSITION_SCALE": encoder.position_scale,
            "DAMAGE_SCALE": encoder.damage_scale,
            "COST_SCALE": encoder.cost_scale,
            "CUTOFF_SECONDS": cfg["environment"]["cutoff_seconds"],
            "EARLY_DIG_TICKS": cfg.get("diagnostics", {}).get("early_dig_seconds", 5) * 20,
            "WAVE_SCALE": cfg["encoding"]["wave_scale"],
            "COUNT_SCALE": encoder.count_scale,
            "FIREPOWER_SCALE": cfg["encoding"].get("firepower_scale", 1),
            "LANE_COUNT_SCALE": cfg["encoding"].get("lane_count_scale", 1),
            "PROJECTILE_OFFSET": encoder.slices["projectiles"].start,
            "GLOBAL_OFFSET": encoder.slices["globals"].start,
            "PLANT_VALUE": int(cfg["reward"].get("potential_mode", "legacy") == "plant_value"),
            "SHAPED": int(cfg["conditions"][condition]["shaped"]),
            "REWARD_SIZE": len(REWARD_FIELDS),
            "METRIC_SIZE": METRIC_SIZE,
        }
        defaults = dict(
            win_reward=1.0,
            loss_penalty=1.0,
            normalize_kills=True,
            mower_sun_scale=300.0,
            sun_target=300.0,
            flower_target=8.0,
            economy_scale=300.0,
        )
        reward_keys = "win_reward loss_penalty normalize_kills gamma defeated_weight economy_weight economy_scale sun_weight sun_target flower_weight flower_target plant_kill_weight mower_kill_weight damage_weight empty_mower_activation_penalty mower_sun_weight mower_sun_scale wall_nut_damage_weight empty_explosion_penalty".split()
        params.update(
            {f"R_{k}": float(cfg["reward"].get(k, defaults.get(k, 0.0))) for k in reward_keys}
        )
        source = kernel_source(batch.rules, batch.zcap, batch.qcap, batch.ecap, batch.diagnostic)
        source += "\n" + "\n".join(f"#define {k} {v}" for k, v in params.items())
        source += "\n" + files("pvz_rl").joinpath("cuda_features.cu").read_text("utf-8")
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self.encode_kernel = self.module.get_function("encode_state")
        self.reward_kernel = self.module.get_function("reward_metrics")
        self.observations = cp.zeros((batch.n, encoder.size), cp.float32)
        self.potential = cp.zeros(batch.n, cp.float64)
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
                b.projectiles,
                b.mowers,
                b.cooldowns,
                self.observations,
                self.potential,
                b.n,
            ),
        )
        b.action_masks_device()
        return self.obs_tensor

    def step(self, actions, *, ticks=1, per_tick=True):
        b = self.batch
        before_phi = self.potential.copy()
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
                    before_phi,
                    self.potential,
                    self.rewards,
                    self.parts,
                    self.totals,
                    b.n,
                ),
            )
        return self.obs_tensor, self.reward_tensor
