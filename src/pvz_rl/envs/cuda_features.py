"""Device encoders/rewards from the versioned public-observation schema."""

from contextlib import nullcontext
from importlib.resources import files

import torch
from pvz_game.cuda.backend import kernel_source
from pvz_game.cuda.schema import REASONS

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.cuda_accounting import ACCOUNTING_WIDTH
from pvz_rl.envs.encoding import (
    ENTITY_FIELDS,
    ENTITY_WIDTH,
    GLOBAL_WIDTH,
    EntityBatch,
    ObservationEncoder,
    width_bucket,
)
from pvz_rl.envs.history import EVENT_WIDTH
from pvz_rl.envs.rewards import LEDGER_METRICS, REWARD_METRICS

REWARD_FIELDS = (*REWARD_METRICS, "total")
METRIC_INDICES = (*range(20, 29), *range(81, 81 + len(REWARD_METRICS) - 9))
LEDGER_INDICES = {name: max(METRIC_INDICES) + 1 + i for i, name in enumerate(LEDGER_METRICS)}
METRIC_SIZE = (
    max(LEDGER_INDICES.values()) + 1
)  # Preserve original totals, early digs and planting timestamps.


class CudaFeatures:
    """Canonical observations, geometry masks and rewards for one simulator batch.

    Records are written into a persistent ``[rows, max_entities]`` store; ``rows``
    may exceed the batch so successive probe passes keep their observations
    (``offset`` selects the destination rows). Encoding never reads the device:
    the per-row summary (kept count and truncations) reaches the host through
    the caller's step handoff, and :meth:`publish` then chooses a fixed padding
    width. Views stay valid until the next encode on the same stream.
    """

    def __init__(self, batch, cfg, condition, *, store_rows=None):
        self.batch, self.cfg = batch, cfg
        cp = batch.cp
        self.profiler = None
        encoder = self.encoder = ObservationEncoder(cfg, batch.rules)
        self.limit = encoder.max_entities
        params = {
            "ENTITY_WIDTH": ENTITY_WIDTH,
            "ACCOUNTING_WIDTH": ACCOUNTING_WIDTH,
            "HISTORY_WIDTH": EVENT_WIDTH,
            "ENTITY_LIMIT": self.limit,
            "RECORD_CAPACITY": 50 + batch.zcap,
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
        for index, value in enumerate(cfg["reward"].get("home_entry_penalties", (0, 0))):
            params[f"HOME_PENALTY_{index}"] = float(value)
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
        self.encode_kernel = self.module.get_function("encode_canonical")
        self.policy_mask_kernel = self.module.get_function("policy_masks")
        self.reward_kernel = self.module.get_function("reward_metrics")
        self.home_kernel = self.module.get_function("home_entries")
        self.dedup_kernel = self.module.get_function("dedup_probes")
        self.pack_kernel = self.module.get_function("pack_entities")
        rows = batch.n if store_rows is None else store_rows
        self.entities = cp.zeros((rows, self.limit, ENTITY_WIDTH), cp.int32)
        self.entity_mask = cp.zeros((rows, self.limit), cp.bool_)
        self.globals = cp.zeros((rows, GLOBAL_WIDTH), cp.float32)
        # Kept count, then plants/zombies/projectiles omitted by the entity cap.
        self.summary = cp.zeros((rows, 4), cp.int64)
        self.history = cp.zeros((rows, 7), cp.int64)
        self.history_tensor = torch.from_dlpack(self.history)
        self.records = cp.zeros((batch.n, 50 + batch.zcap, ENTITY_WIDTH), cp.int32)
        self.truncation_counts = self.summary[: batch.n, 1:]
        self.entity_tensor = torch.from_dlpack(self.entities)
        self.entity_mask_tensor = torch.from_dlpack(self.entity_mask)
        self.globals_tensor = torch.from_dlpack(self.globals)
        self.summary_tensor = torch.from_dlpack(self.summary)
        self.width, self.obs_tensor, self.summary_host = None, None, None
        self.home_ledger = cp.zeros((batch.n, batch.zcap, 2), cp.int64)
        self.home_entries = cp.zeros((batch.n, 2), cp.int64)
        self.assets = cp.zeros(batch.n, cp.float64)
        self.rewards = cp.zeros(batch.n, cp.float32)
        self.parts = cp.zeros((batch.n, len(REWARD_FIELDS)), cp.float64)
        self.totals = cp.zeros((batch.n, METRIC_SIZE), cp.float64)
        self.totals[:, 11] = -1
        # Reusable pre-step copies; the hot path allocates nothing.
        self.before_header = cp.zeros_like(batch.header)
        self.before_cooldowns = cp.zeros_like(batch.cooldowns)
        self.before_assets = cp.zeros_like(self.assets)
        self.enabled = cp.zeros(batch.n, cp.bool_)
        self.reward_tensor = torch.from_dlpack(self.rewards)
        self.mask_tensor = torch.from_dlpack(batch.masks)

    def _track(self, phase):
        return self.profiler.track(phase) if self.profiler is not None else nullcontext()

    def initialize_home(self, indices):
        self.home_ledger[indices] = 0
        selected = self.batch.cp.zeros(self.batch.n, self.batch.cp.bool_)
        selected[indices] = True
        self.home_kernel(
            ((self.batch.n + 63) // 64,),
            (64,),
            (
                self.batch.header,
                self.batch.zombies,
                self.home_ledger,
                self.home_entries,
                selected,
                self.batch.n,
            ),
        )
        self.home_entries[indices] = 0

    def encode(self, *, offset=0, publish=True):
        """Write canonical records for every game into store rows ``offset:offset+n``.

        ``publish`` reads the kept counts once to select the padding width; the
        collection hot path instead publishes from its existing host handoff.
        """
        b = self.batch
        rows = slice(offset, offset + b.n)
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
                self.records,
                self.entities[rows],
                self.entity_mask[rows],
                self.globals[rows],
                self.assets,
                self.summary[rows],
                b.n,
            ),
        )
        self.policy_mask_kernel(
            ((b.n * ActionSchema.size + 255) // 256,),
            (256,),
            (b.header, b.plants, b.masks, b.n),
        )
        if not publish:
            return None
        return self.publish(self.summary[rows].get(), b.header[:, 17].get() != 0)

    def view(self, offset, rows, width):
        """Validated canonical rows at a fixed padding width (no copy)."""
        return EntityBatch(
            self.entity_tensor[offset : offset + rows, :width],
            self.entity_mask_tensor[offset : offset + rows, :width],
            self.globals_tensor[offset : offset + rows],
            validated=True,
        )

    def publish(self, summary, active):
        """Expose the batch at the smallest fixed width holding every active observation.

        ``summary`` is the host copy of this encode's per-game summary rows.
        """
        self.summary_host = summary
        self.width = width_bucket(int(summary[active, 0].max(initial=0)), self.limit)
        self.obs_tensor = self.view(0, self.batch.n, self.width)
        return self.obs_tensor

    def step(self, actions, *, ticks=1, per_tick=True, offset=0, publish=True, phase="simulation"):
        b, cp = self.batch, self.batch.cp
        with self._track(phase):
            cp.copyto(self.before_assets, self.assets)
            cp.copyto(self.before_header, b.header)
            cp.copyto(self.before_cooldowns, b.cooldowns)
            b.step_device(actions, ticks=ticks, per_tick=per_tick)
        with self._track("encoding"):
            observation = self.encode(offset=offset, publish=publish)
            cp.copyto(self.history[offset : offset + b.n], b.accounting[:, 3:])
        with self._track(phase):
            cp.copyto(self.enabled, b.header[:, 17], casting="unsafe")
            self.home_kernel(
                ((b.n + 63) // 64,),
                (64,),
                (b.header, b.zombies, self.home_ledger, self.home_entries, self.enabled, b.n),
            )
            self.reward_kernel(
                ((b.n + 63) // 64,),
                (64,),
                (
                    b.header,
                    self.before_header,
                    self.before_cooldowns,
                    actions,
                    b.facts,
                    b.accounting,
                    self.before_assets,
                    self.assets,
                    self.rewards,
                    self.parts,
                    self.totals,
                    self.home_entries,
                    b.n,
                ),
            )
        return observation, self.reward_tensor

    def dedup(self, events, valid, source, slots):
        """Mark probe rows whose stored and recurrent inputs repeat an earlier slot."""
        cp = self.batch.cp
        n = len(self.entities) // slots
        self.dedup_kernel(
            (n,),
            (128,),
            (
                self.entities,
                self.summary,
                self.globals,
                cp.from_dlpack(events.contiguous()),
                cp.from_dlpack(valid),
                cp.from_dlpack(source),
                n,
                slots,
            ),
        )

    def pack(self, selected, offsets, destination):
        """Copy selected rows' kept records contiguously; sizes come from the host."""
        cp = self.batch.cp
        rows = len(selected)
        self.pack_kernel(
            (rows,),
            (128,),
            (
                self.entities,
                self.summary,
                cp.from_dlpack(offsets),
                cp.from_dlpack(selected),
                cp.from_dlpack(destination),
                rows,
            ),
        )
