"""Private random/scripted play pretraining, with independent counters and recovery."""

import random
from time import perf_counter

import numpy as np
import torch
from pvz_game import LevelSpec, Spawn
from pvz_game.cuda import CudaBatch
from pvz_game.cuda.backend import projectile_bound

from pvz_rl.config import absolute_outcomes
from pvz_rl.envs.cuda_features import CudaFeatures
from pvz_rl.envs.cuda_snapshots import download_snapshots, upload_snapshots
from pvz_rl.envs.encoding import BOARD_GLOBAL_FIELDS
from pvz_rl.envs.fork_service import ForkService
from pvz_rl.envs.scratch_game import scratch_game, set_initial_mowers
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.cuda_buffer import trajectory_dtype
from pvz_rl.learning.forecast_labels import baseline_features, effect_labels, outcome_codes
from pvz_rl.learning.forecast_objective import forecast_auxiliary


class ForecastPretraining:
    """generate → play → baseline → effects → fit → next panel.

    Only optimizer commits advance here. Main episodes, validation clocks,
    exploration RNG and the physical main environments remain untouched.
    """

    def __init__(self, model, state=None):
        self.model, self.cfg = model, model.cfg
        self.settings = dict(self.cfg["training"]["forecasts"])
        self.total, self.cursor, self.phase, self.fit_pass = (
            self.settings["pretrain_roots"],
            0,
            "generate",
            0,
        )
        self.size = min(32, max(1, self.total))
        self.source = CudaBatch(
            self.size,
            zombie_capacity=8,
            projectile_capacity=2 * projectile_bound(model.env.batch.rules),
            max_step_ticks=self.settings["chunk_ticks"],
            rules=model.env.batch.rules,
            device=model.env.batch.device,
        )
        self.features = CudaFeatures(self.source, self.cfg, "masked")
        self.service = ForkService(
            self.source,
            dict(
                self.settings,
                **{
                    k: v
                    for k, v in self.cfg["training"]["performance"].items()
                    if k.startswith("fork_") and k != "fork_suffix_cache"
                },
                fork_suffix_cache=False,  # Each private pretraining board is queried once.
                headroom_mib=self.cfg["training"]["performance"].get(
                    "collection_vram_headroom_mib", 0
                ),
            ),
        )
        self.rows = self.baseline = None
        self.jobs = []
        self.play_tick = 0
        self.count = 0
        self.metrics = dict(
            roots=0,
            optimizer_commits=0,
            effect_loss=None,
            random_roots=0,
            scripted_roots=0,
            forks=0,
            simulation_ticks=0,
            play_ticks=0,
            seconds=0.0,
        )
        if state is not None:
            for name in (
                "cursor",
                "phase",
                "fit_pass",
                "rows",
                "baseline",
                "jobs",
                "play_tick",
                "count",
                "metrics",
            ):
                setattr(self, name, state[name])
            if state["source"]:
                upload_snapshots(self.source, state["source"], indices=range(self.count))
            self.service.restore(state["service"])

    def _generate(self):
        self.count = min(self.size, self.total - self.cursor)
        snapshots = []
        for index in range(self.count):
            rng = random.Random(
                (int(self.model.seed or 0) << 32) ^ (self.cursor + index) ^ 0xF0EC457
            )
            count = rng.randint(1, 8)
            level = LevelSpec(
                "forecast-pretraining",
                tuple(
                    Spawn(
                        3501,
                        rng.choice(self.cfg["environment"]["zombies"]),
                        rng.randrange(5),
                        x=rng.randrange(9 * self.source.rules.game["units_per_tile"]),
                    )
                    for _ in range(count)
                ),
                initial_sun=500,
                mowers=False,
            )
            game = scratch_game(self.source.rules)
            game.reset(level, rng.randrange(2**31))
            set_initial_mowers(game, [r for r in range(5) if rng.random() < 0.5])
            snapshots.append(game.snapshot())
        self.source.header[:, 17] = 0
        upload_snapshots(self.source, snapshots, indices=range(self.count))
        self.play_tick, self.phase = 0, "play"

    def _play(self):
        legal = self.source.action_masks_device().get()
        commands = np.zeros(self.size, np.int64)
        for slot in range(self.count):
            choices = np.flatnonzero(legal[slot])
            rng = np.random.default_rng(
                [int(self.model.seed or 0), self.cursor + slot, self.play_tick, 17]
            )
            if (self.cursor + slot) % 2 == 0:
                commands[slot] = int(rng.choice(choices))
            else:
                # Public scripted economy/defense mixture: economy first, then
                # spread shooters toward the left, paying every real cooldown.
                priority = [1, 2, 8, 6, 3, 5, 7, 4]
                candidates = [
                    a
                    for branch in priority
                    for a in choices
                    if a > 0 and (a - 1) // 45 + 1 == branch
                ]
                commands[slot] = int(candidates[0]) if candidates else 0
        ticks = min(self.source.max_step_ticks, 3501 - self.play_tick)
        self.source.step_device(self.source.cp.asarray(commands), ticks=ticks, per_tick=False)
        self.metrics["play_ticks"] += ticks * self.count
        self.play_tick += ticks
        if self.play_tick == 3501:
            self.service.begin(range(self.count))
            self.phase = "baseline"

    def _effects(self):
        if not self.service.advance():
            return
        after = self.service.take()
        for (root, token), outcome in zip(self.jobs, after, strict=True):
            self.rows["effect_labels"][root, token] = (
                outcome_codes(outcome, reference_tau(self.source.rules))
                if absolute_outcomes(self.cfg)
                else effect_labels(self.baseline[root], outcome, reference_tau(self.source.rules))
            )
        self.features.encode()
        if not absolute_outcomes(self.cfg):
            self.features.globals[: self.count, len(BOARD_GLOBAL_FIELDS) :] = (
                self.source.cp.asarray(
                    baseline_features(self.baseline, reference_tau(self.source.rules))
                )
            )
        self.fit_pass, self.phase = 0, "fit"

    def _fit(self):
        policy, model = self.model.policy, self.model
        observations = self.features.obs_tensor[: self.count]
        # Recovery reconstructs the public input from the saved private source.
        if not absolute_outcomes(self.cfg):
            self.features.globals[: self.count, len(BOARD_GLOBAL_FIELDS) :] = (
                self.source.cp.asarray(
                    baseline_features(self.baseline, reference_tau(self.source.rules))
                )
            )
        policy.zero_grad(set_to_none=True)
        policy.auxiliary_scale = 1.0
        policy.injection_enabled = False
        with policy.board_context(observations) as memory:
            loss, metrics = forecast_auxiliary(
                policy,
                observations,
                self.rows,
                memory,
                decision_count=(self.count, self.count),
                loss_weights=(0, 1),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite fork pretraining loss")
        loss.backward()
        eligible = {id(p) for p in policy.runtime_parameters("effect_pretrain")}
        for parameter in policy.parameters():
            if id(parameter) not in eligible:
                parameter.grad = None
        policy.accumulate_master_gradients()
        torch.nn.utils.clip_grad_norm_(
            policy.master_parameters("effect_pretrain"),
            model.max_grad_norm,
            error_if_nonfinite=True,
        )
        policy.optimizer.step()
        policy.sync_runtime_from_masters()
        model._update_teacher("effect_pretrain")
        policy.zero_grad(set_to_none=True)
        policy._forecast_components = None
        policy._primary_graph = None
        model.optimizer_commits += 1
        self.metrics["optimizer_commits"] += 1
        self.metrics["effect_loss"] = float(metrics[4])
        self.fit_pass += 1
        if self.fit_pass == self.settings["pretrain_passes"]:
            self.cursor += self.count
            self.metrics["roots"] = self.cursor
            self.metrics["random_roots"] = (self.cursor + 1) // 2
            self.metrics["scripted_roots"] = self.cursor // 2
            self.phase = "generate"

    def advance(self):
        start = perf_counter()
        try:
            return self._advance()
        finally:
            self.metrics["seconds"] = self.metrics.get("seconds", 0.0) + perf_counter() - start
            self.metrics["fork_ticks"] = self.service.metrics["ticks"]
            self.metrics["simulation_ticks"] = (
                self.metrics["play_ticks"] + self.metrics["fork_ticks"]
            )

    def _advance(self):
        if self.cursor == self.total:
            return True
        if self.phase == "generate":
            self._generate()
        elif self.phase == "play":
            self._play()
        elif self.phase == "baseline":
            if self.service.advance():
                self.baseline = self.service.take()
                self.rows = np.zeros(self.count, dtype=trajectory_dtype(self.cfg))
                self.rows["active"], self.rows["reach_valid"], self.rows["branch_usable"] = (
                    True,
                    True,
                    True,
                )
                self.rows["threat_depth"], self.rows["threat_time"], self.rows["threat_loss"] = (
                    -1,
                    -1,
                    -1,
                )
                self.rows["baseline_outcomes"] = self.baseline
                self.rows["effect_labels"] = (
                    -1 if absolute_outcomes(self.cfg) else np.array([10, 5, 1, 10])
                )
                legal = self.source.action_masks_device().get()
                self.jobs = []
                for root in range(self.count):
                    rng = np.random.default_rng([int(self.model.seed or 0), self.cursor + root, 71])
                    candidates = np.flatnonzero(legal[root])[1:]
                    selected = rng.choice(candidates, min(5, len(candidates)), replace=False)
                    for token, action in enumerate(selected):
                        self.rows["effect_actions"][root, token] = action
                        self.rows["effect_valid"][root, token] = True
                        self.jobs.append((root, token))
                    self.rows["action"][root] = selected[0] if len(selected) else 0
                self.service.begin(
                    [root for root, _ in self.jobs],
                    [self.rows["effect_actions"][r, t] for r, t in self.jobs],
                )
                self.phase = "effects"
        elif self.phase == "effects":
            self._effects()
        elif self.phase == "fit":
            if self.features.obs_tensor is None:
                self.features.encode()
            self._fit()
        else:
            raise ValueError("Invalid forecast pretraining phase")
        self.metrics["forks"] = self.service.metrics["forks"]
        return self.cursor == self.total

    def snapshot(self):
        return dict(
            **{
                name: getattr(self, name)
                for name in (
                    "cursor",
                    "phase",
                    "fit_pass",
                    "rows",
                    "baseline",
                    "jobs",
                    "play_tick",
                    "count",
                    "metrics",
                )
            },
            source=download_snapshots(self.source, indices=range(self.count)) if self.count else [],
            service=self.service.snapshot(),
        )

    def close(self):
        self.service.close()
        self.source = self.features = None
