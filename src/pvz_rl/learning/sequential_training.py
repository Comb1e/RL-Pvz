"""Stateless two-stage fitting on the shared CUDA cohort lifecycle."""

import copy
from contextlib import closing
from dataclasses import replace
from time import perf_counter

import numpy as np
import torch

from pvz_rl.config import (
    absolute_outcomes,
    bomb_max_illegal_distance,
    bomb_ranking_margin,
    bomb_softmax_temperature,
    effect_prediction_weight,
    forecast_forks_enabled,
    forecast_mode,
    forecast_version,
    passivity_settings,
    plant_wait_check_ticks,
    threat_prediction_weight,
)
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_features import REWARD_FIELDS
from pvz_rl.envs.encoding import ENTITY_WIDTH, PackedEntityBatch
from pvz_rl.envs.tile_scenarios import is_placement_skill, placement_settings
from pvz_rl.learning.action_routing import classify_actions
from pvz_rl.learning.batch_transport import (
    BatchPrefetch,
    FittingSpool,
    batch_rows,
    fitting_metadata,
)
from pvz_rl.learning.bomb_objective import (
    BOMB_OBJECTIVE,
    DISTANCE_OBJECTIVES,
    bomb_distance_loss,
    bomb_loss_metrics,
    bomb_set_cross_entropy,
)
from pvz_rl.learning.cohort import CohortPhase, atomic_transition, check_transition_interruption
from pvz_rl.learning.collection_scheduler import CollectionScheduler
from pvz_rl.learning.cuda_q import CudaCohortLifecycle
from pvz_rl.learning.exploration import (
    CommittedPlantExploration,
    configure_exploration,
    exploration_status,
    refresh_wait_exploration,
)
from pvz_rl.learning.flat_q import CohortCoefficients, batch_loss
from pvz_rl.learning.objective import (
    COMPONENTS,
    consistency_diagnostics,
    finalize_episode_metrics,
)
from pvz_rl.learning.peashooter_rules import RULE_OBJECTIVE, PeashooterObjective, rules_enabled
from pvz_rl.learning.performance import compilation_requested
from pvz_rl.learning.tile_curriculum import TILE_SCENARIO_PROTOCOL, TileSkillState
from pvz_rl.learning.training_requirements import FIT_SCHEDULE_PROTOCOL, fitting_schedule
from pvz_rl.policy.active_batch import ActiveBatchPlan
from pvz_rl.policy.cudagraph_backend import EncoderCompilationError
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.sequential_q import action_parts
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy


def flat_batches(buffer, decision_budget, spool=None):
    if spool is not None:
        for rows, packed in spool.iter_batches(decision_budget):
            width = spool.buckets[np.searchsorted(spool.buckets, int(packed.counts.max(initial=0)))]
            yield (
                rows,
                buffer.observations_from_packed(packed, validated=True, width=width),
                fitting_metadata(rows, "cpu"),
            )
        return
    with closing(batch_rows(buffer, decision_budget, spool=spool)) as source:
        for rows in source:
            yield rows, buffer.observations(rows)


class CudaSequentialQ(CudaCohortLifecycle):
    policy_protocol = "shared_public_forecast_q_v4"
    optimizer_version = FIT_SCHEDULE_PROTOCOL

    @property
    def fit_schedule(self):
        schedule = fitting_schedule(self.cfg, self.normal_updates + 1)
        return replace(schedule, stage="tile_only") if self.tile_skills.active else schedule

    @property
    def tile_skills(self):
        return TileSkillState(**self.tile_skill_state)

    @property
    def total_fit_passes(self):
        return self.fit_schedule.passes

    @property
    def fit_stage(self):
        return self.fit_schedule.stage

    def _setup_model(self):
        super()._setup_model()
        if not hasattr(self, "tile_skill_state"):
            self.tile_skill_state = TileSkillState().to_dict()
        if not hasattr(self, "normal_updates"):
            self.normal_updates = self._n_updates
        # SB3 restores ordinary attributes before rebuilding the policy. Keep a
        # saved cohort's FP32/memory fallback instead of overwriting it here.
        if not getattr(self, "execution_state", None):
            self._reset_execution()
        if not hasattr(self, "_passivity_requested"):
            self._passivity_requested = passivity_settings(self.cfg)
        if not hasattr(self, "_passivity_active"):
            self._passivity_active = None

    def refresh_passivity(self, cfg):
        """Apply only the requested residual-sun settings at a cohort boundary."""
        self._passivity_requested = passivity_settings(cfg)
        return dict(self._passivity_requested)

    def _new_runner(self):
        return PolicyRunner(self.policy)

    @property
    def peashooter_rule_training(self):
        return rules_enabled(self.cfg, self.tile_skills.phase)

    @property
    def reach_training(self):
        available = not self.tile_skills.active or self.tile_skills.phase in self.cfg[
            "training"
        ].get("stage_threat_weights", {})
        return (
            available
            and self.cfg.get("diagnostics", {}).get("reach_tracking", True)
            and self.cfg["training"].get("forecasts", {}).get("enabled", False)
        )

    @property
    def reach_weight(self):
        return threat_prediction_weight(self.cfg, self.tile_skills.phase) * getattr(
            self.policy, "auxiliary_scale", 1
        )

    @property
    def home_weight(self):
        return effect_prediction_weight(self.cfg, self.tile_skills.phase) * getattr(
            self.policy, "auxiliary_scale", 1
        )

    @property
    def fitting_parameter_stage(self):
        if self._fit_kind != "tile_only":
            return None
        return (
            (
                "tile_reach"
                if forecast_forks_enabled(self.cfg, self.tile_skills.phase)
                else "tile_threat"
            )
            if self.reach_training and (self.reach_weight > 0 or self.home_weight > 0)
            else "tile"
        )

    def _prepare_collection(self, callback):
        if absolute_outcomes(self.cfg):
            self._scheduler.draining = getattr(self, "_pending_effect", None) is not None
            owner = getattr(self, "_wait_outcomes", None)
            if owner is not None:
                started = perf_counter()
                try:
                    with atomic_transition(), self.env.device_context():
                        owner.advance()
                        if not self.env.enabled_envs.any() and not self._scheduler.pending.any():
                            if not owner.finalize():
                                if hasattr(callback, "log_progress"):
                                    callback.log_progress()
                                return False
                finally:
                    self._phase_times["collect"] += perf_counter() - started
                    self._stats["wait_outcomes"] = dict(owner.metrics)
        elif forecast_forks_enabled(self.cfg, self.tile_skills.phase):
            from pvz_rl.learning.effect_collection import finish_effects

            if hasattr(callback, "check_deadline"):
                callback.check_deadline()
            started = perf_counter()
            with atomic_transition(), self.env.device_context():
                pending = getattr(self, "_pending_effect", None)
                self._scheduler.draining = pending is not None
                if pending is None and self._scheduler.pending.any():
                    raise ValueError(
                        "Pending decision tickets lack effect recovery records; use an intact checkpoint"
                    )
                if pending is not None:
                    ready = finish_effects(self)
                    if ready:
                        self.env.restore_baseline_inputs(np.flatnonzero(self._scheduler.pending))
                else:
                    ready = self.env.prepare_baseline(self._scheduler.ready(self.env.enabled_envs))
            self._phase_times["collect"] += perf_counter() - started
            if not ready:
                if hasattr(callback, "log_progress"):
                    callback.log_progress()
                return False
        if self.reach_training or self.peashooter_rule_training:
            if self.env.enabled_envs.any() or self._scheduler.pending.any():
                return True
            if hasattr(callback, "check_deadline"):
                callback.check_deadline()
            started = perf_counter()
            try:
                with atomic_transition(), self.env.device_context():
                    ready, staging = True, 0
                    if self.reach_training:
                        ready = self._reach.prepare()
                        staging += self._reach.estimated_bytes
                        self._stats["reach_tracking"] = dict(self._reach.metrics)
                    if self.peashooter_rule_training:
                        ready = self._peashooter_teacher.advance() and ready
                        staging += self._peashooter_teacher.estimated_bytes
                        self._stats["peashooter_teacher"] = dict(self._peashooter_teacher.metrics)
                    if not self._buffer.reserve_staging(staging):
                        raise MemoryError("Shared RAM budget cannot hold placement/reach metadata")
                    if ready:
                        if self.reach_training:
                            self.env.action_journal.finalize_reach(self._buffer)
                            viewer = getattr(self.env, "live_view", None)
                            if viewer is not None:
                                viewer.finalize_reach()
                        self._phase(CohortPhase.FINALIZE_REWARDS, callback)
            finally:
                self._phase_times["collect"] += perf_counter() - started
            if hasattr(callback, "log_progress"):
                callback.log_progress()
            return False
        return True

    def _begin(self, callback):
        # A restored cohort keeps the buffer configuration it was collected
        # with. New buffers receive the current requested reward settings.
        self.cfg["reward"]["passivity"] = copy.deepcopy(self._passivity_requested)
        self._passivity_active = copy.deepcopy(self._passivity_requested)
        self._drain()
        self._close_batches()
        if getattr(self, "_reach", None) is not None:
            self._reach.close()
            self._reach = None
        if getattr(self, "_peashooter_teacher", None) is not None:
            self._peashooter_teacher.close()
            self._peashooter_teacher = None
        if (
            not self.peashooter_rule_training
            and getattr(self.env, "lane_preparation", None) is not None
        ):
            self.env.lane_preparation.close()
            self.env.lane_preparation = None
        boundary = self._phase_memory("before_collection_release")
        for encoder in self.policy.encoders:
            encoder.release_fitting()
        if getattr(self, "_teacher", None) is not None:
            for encoder in self._teacher.encoders:
                encoder.release_fitting()
        for encoder in self.policy.encoders:
            encoder.microbatch = self.cfg["policy"]["encoder_microbatch"]
            encoder.token_budget = self.cfg["policy"]["encoder_token_budget"]
            encoder.activation_checkpointing = False
        torch.cuda.empty_cache()
        self.env.cp.get_default_memory_pool().free_all_blocks()
        released = self._phase_memory("after_collection_release")
        self.collection_execution = {}
        for name in ("_collection_reach", "_collection_home", "_collection_effect"):
            if hasattr(self, name):
                delattr(self, name)
        if getattr(self, "_wait_outcomes", None) is not None:
            self._wait_outcomes.close()
            self._wait_outcomes = None
        super()._begin(callback)
        self._stats["phase_memory"] = [boundary, released]
        self._stats["collection_logical_rows"] = 0
        self._stats["collection_padded_rows"] = 0
        self._stats["collection_entity_widths"] = {}
        self._plant_exploration = CommittedPlantExploration(
            self.n_envs,
            self.device,
            self.plant_exploration_rate,
            self.exploration_rate,
            wait_check_ticks=None if self.tile_skills.active else plant_wait_check_ticks(self.cfg),
        )
        self._stats.update(
            plant_exploration_checks=0,
            wait_exploration_checks=0,
            plant_exploration_commitments=0,
            wait_exploration_commitments=0,
            passivity_requested=copy.deepcopy(self._passivity_requested),
            passivity_active=copy.deepcopy(self._passivity_active),
            **exploration_status(self),
        )
        self._buffer.exploration = dict(
            plant_epsilon=self.plant_exploration_rate, tile_epsilon=self.exploration_rate
        )
        self._plant_costs = torch.tensor(
            [
                self.env.batch.rules.plants[kind]["cost"]
                for kind in self.cfg["environment"]["plants"]
            ],
            device=self.device,
        )
        self._ensure_teacher()
        settings = self.cfg["training"].get("forecasts", {})
        injection = forecast_forks_enabled(
            self.cfg, self.tile_skills.phase
        ) and self.training_games >= settings.get("injection_after_games", 4096)
        total = self.training_games + self.tile_skills.total
        half_life = (
            settings.get("discount_half_life_steps", 600)
            if injection
            else settings.get("initial_discount_half_life_steps", 60)
        )
        self._resolved_forecast_schedule = dict(
            injection=injection,
            auxiliary_scale=min(
                1.0, (total + self.cohort_games) / max(1, settings.get("warmup_games", 2048))
            ),
            n_step=settings.get("injected_n_step", 20) if injection else settings.get("n_step", 5),
            gamma_tick=0.5 ** (1 / (half_life * self.cfg["environment"]["decision_ticks"])),
        )
        self.cfg["training"]["forecasts"]["injection_active"] = injection
        for network in (self.policy, self._teacher):
            network.forecast_stage = self.tile_skills.phase
            network.injection_enabled = injection
            network.auxiliary_scale = self._resolved_forecast_schedule["auxiliary_scale"]
        self._stats["forecast_schedule"] = dict(self._resolved_forecast_schedule)
        self._stats["forecast_mode"] = forecast_mode(self.cfg, self.tile_skills.phase)
        self._stats["forecast_pretraining_state"] = getattr(
            self, "forecast_pretraining_state", "pending"
        )
        if self.reach_training:
            from pvz_rl.learning.reach_supervision import ReachSupervision

            self._reach = ReachSupervision(self.env, self._buffer)
        if self.peashooter_rule_training:
            from pvz_rl.learning.peashooter_teacher import PeashooterTeacher

            self._stats["fitting_objective"] = RULE_OBJECTIVE
            self._peashooter_teacher = PeashooterTeacher(
                self.env,
                self._buffer,
                (self.cohort_number, self.optimizer_commits),
            )
        self._scheduler = CollectionScheduler(self.n_envs, self.device)
        self._pending_effect = None
        self._wait_outcomes = None
        if (
            absolute_outcomes(self.cfg)
            and self.reach_training
            and forecast_forks_enabled(self.cfg, self.tile_skills.phase)
        ):
            from pvz_rl.learning.wait_outcomes import WaitOutcomes

            self._wait_outcomes = WaitOutcomes(self.env, self._buffer)
        if compilation_requested(self.cfg["training"]["performance"]):
            for encoder in self._teacher.encoders:
                if encoder.compilation_status == "disabled":
                    encoder.enable_compilation()
        self._batch_iterator = None
        self._fitting_spool = None
        self._reset_execution()

    def _ensure_teacher(self):
        if getattr(self, "_teacher", None) is None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self._teacher = (
                    TwoStageTransformerPolicy(self.cfg).to(self.device).eval().requires_grad_(False)
                )
            self._teacher.load_state_dict(self.policy.runtime_state_dict())
            self._teacher_version = getattr(self, "_teacher_version", 0)

    @torch.no_grad()
    def _update_teacher(self, stage=None):
        self._ensure_teacher()
        decay = self.cfg["training"]["objective"]["ema_decay"]
        teacher_parameters = self._teacher.runtime_parameters(stage)
        online_parameters = self.policy.runtime_parameters(stage)
        torch._foreach_lerp_(
            list(teacher_parameters),
            list(online_parameters),
            1 - decay,
        )
        self._teacher_version += 1

    def _finalize_rewards(self, callback):
        self.env.handoff.clear()
        if not self._buffer.reserve_staging(
            (self._peashooter_teacher.estimated_bytes if self.peashooter_rule_training else 0)
            + (self._reach.estimated_bytes if self.reach_training else 0)
        ):
            raise MemoryError("Shared RAM budget cannot hold placement/reach metadata")
        pending = getattr(self, "_pending_episodes", {})
        summary = self._buffer.finalize_rewards(pending)
        passivity = summary.get("passivity", {}).get("episodes", {})
        for slot, item in pending.items():
            episode = item.get("episode")
            details = passivity.get(f"{int(slot)}:{int(episode)}", {})
            item["metrics"]["wait_passivity_penalty"] = float(
                -details.get("loss_wait_penalty", 0.0)
            )
            item["metrics"].update(
                {
                    "wait_passivity_attribution": summary.get("passivity", {}).get(
                        "loss_wait_attribution"
                    ),
                    "wait_passivity_wait_rows": int(details.get("wait_rows", 0)),
                    "wait_passivity_run_count": int(details.get("run_count", 0)),
                    "wait_passivity_shortest_run": int(details.get("shortest_run", 0)),
                    "wait_passivity_longest_run": int(details.get("longest_run", 0)),
                    "wait_passivity_run_details": details.get("run_details", []),
                    "wait_passivity_penalty_minimum": float(details.get("minimum_penalty", 0.0)),
                    "wait_passivity_penalty_maximum": float(details.get("maximum_penalty", 0.0)),
                    "wait_passivity_skipped_no_wait": bool(details.get("skipped_no_wait", False)),
                }
            )
        finalize_episode_metrics(
            [p["metrics"] for p in pending.values()],
            self.cfg,
            median=summary["duration_median_seconds"],
        )
        viewer = getattr(self.env, "live_view", None)
        if viewer is not None and viewer.enabled:
            try:
                viewer.finalize_rewards(pending)
            except Exception as exc:
                viewer.fail(exc)
        if hasattr(callback, "finalize_cohort_rewards"):
            callback.finalize_cohort_rewards(pending)
        self._phase(CohortPhase.RETURNS, callback)

    def _phase_memory(self, boundary):
        from pvz_rl.learning.performance import phase_memory

        memory = phase_memory(self.device)
        return dict(
            boundary=boundary,
            torch_allocated_mib=memory["allocated_mib"],
            torch_reserved_mib=memory["reserved_mib"],
            cupy_reserved_mib=self.env.cp.get_default_memory_pool().total_bytes() / 2**20,
            free_mib=memory["free_mib"],
            total_mib=memory["total_mib"],
        )

    def _phase(self, phase, callback):
        if phase == CohortPhase.FIT:
            if getattr(self, "_wait_outcomes", None) is not None:
                if not self._wait_outcomes.published:
                    raise RuntimeError("Cannot fit unresolved absolute outcomes")
                self._stats["wait_outcomes"] = dict(self._wait_outcomes.metrics)
                self._wait_outcomes.close()
                self._wait_outcomes = None
            if self.env.static_forks is not None:
                if self.env.static_forks.request is not None:
                    raise RuntimeError("Cannot fit with an unfinished static fork")
                self._stats["static_forks"] = dict(self.env.static_forks.metrics)
                self.env.static_forks.close()
                self.env.static_forks = None
            if getattr(self, "_peashooter_teacher", None) is not None:
                self._peashooter_teacher.release()
                if not self._buffer.reserve_staging(
                    self._peashooter_teacher.estimated_bytes
                    + (self._reach.estimated_bytes if self.reach_training else 0)
                ):
                    raise MemoryError("Shared RAM budget cannot hold placement/reach metadata")
            for encoder in self.policy.encoders:
                encoder.release_collection()
            if getattr(self, "_teacher", None) is not None:
                for encoder in self._teacher.encoders:
                    encoder.release_collection()
            if hasattr(self, "_collection_packed"):
                del self._collection_packed
            torch.cuda.empty_cache()
            self.env.cp.get_default_memory_pool().free_all_blocks()
            self._stats.setdefault("phase_memory", []).append(self._phase_memory("fit_entry"))
        super()._phase(phase, callback)

    def _collect_step(self, callback, *, draining=False):
        if not draining:
            check_transition_interruption()
        self._collection_callback = callback
        env = self.env
        with torch.no_grad(), env.device_context():
            unfinished = env.enabled_envs.copy()
            new = self._scheduler.ready(unfinished)
            issued_host = new | self._scheduler.pending
            enabled = torch.as_tensor(new, device=self.device)
            plan = ActiveBatchPlan.from_counts(
                new,
                env.features.summary_host[:, 0],
                env.features.limit,
                self.policy.species_encoder.microbatch,
                self.policy.species_encoder.token_budget,
            )
            live_slots = torch.as_tensor(np.flatnonzero(issued_host), device=self.device)
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            rows["reset"] = self._scheduler.resets
            rows["env"] = np.arange(self.n_envs)
            source_decisions = self._scheduler.decisions.copy()
            masks = env.action_masks().clone()
            branch_available = env.branch_availability()
            if new.any():
                supplied = None
                if self.tile_skills.active:
                    supplied = env.supplied_branches(self.tile_skills.phase, branch_available)
                with env.profiler.track("online_inference"):
                    result = self._runner.decide(
                        self._last_obs,
                        masks,
                        active=enabled,
                        deterministic=False,
                        defer_check=True,
                        plan=plan,
                        branch_usable=branch_available,
                        supplied_branches=supplied,
                    )
                actions, values, tile_values, details = result
                if self.tile_skills.active:
                    details.update(
                        policy_action=actions,
                        exploration_mode=torch.zeros_like(actions),
                        pending_species=torch.zeros_like(actions),
                    )
                else:
                    actions, values, tile_values, details = self._plant_exploration.apply(
                        self.policy,
                        actions,
                        details,
                        masks,
                        env.header_tensor[:, 2],
                        torch.from_dlpack(env.batch.cooldowns),
                        self._plant_costs,
                        enabled,
                        branch_usable=branch_available,
                        plan=plan,
                    )
                self._runner.valid &= details["valid"]
                if self.reach_training:
                    if not hasattr(self, "_collection_reach"):
                        self._collection_reach = torch.zeros(
                            (self.n_envs, 5, 9), device=self.device
                        )
                    if not hasattr(self, "_collection_home"):
                        self._collection_home = torch.zeros((self.n_envs, 5), device=self.device)
                    self._collection_home[enabled] = self.policy.home_map(details["reach_logits"])[
                        enabled
                    ]
                    self._collection_reach[enabled] = self.policy.reach_map(
                        details["reach_logits"]
                    )[enabled]
                    if forecast_version(self.cfg) >= 2:
                        if not hasattr(self, "_collection_effect"):
                            self._collection_effect = torch.zeros(
                                (self.n_envs, 2, 5, 10), device=self.device
                            )
                        self._collection_effect[enabled] = self.policy.effect_action_display(
                            details["observations"],
                            actions,
                            details["board_memory"],
                            branch_available,
                        )[enabled]
                self.policy._effect_decision_cache = None
            if not hasattr(self, "_collection_packed"):
                self._collection_packed = torch.empty(
                    self.n_envs * env.features.limit,
                    ENTITY_WIDTH,
                    dtype=torch.int32,
                    device=self.device,
                )
            # Queue immutable public inputs and decision evidence before simulation.
            if new.any():
                branches, _ = action_parts(actions)
                usable = branch_available.gather(1, branches[:, None]).flatten()
                minimum_q = torch.where(
                    (branches > 0) & usable,
                    details["tile_q"].float().amin(-1),
                    details["branch_q"].float().amin(-1),
                )
                columns = torch.cat(
                    (
                        env.header_tensor[:, :1].float(),
                        actions[:, None].float(),
                        values[:, None],
                        tile_values[:, None],
                        details["greedy_actions"][:, None].float(),
                        details["coins"].float(),
                        details["branch_q"],
                        torch.from_dlpack(env.features.truncation_counts).float(),
                        details["policy_action"][:, None].float(),
                        details["exploration_mode"][:, None].float(),
                        details["pending_species"][:, None].float(),
                        details["placement_gap"][:, None],
                        minimum_q[:, None],
                    ),
                    dim=1,
                )
                self._scheduler.retain_species_draws(
                    new,
                    torch.rand(self.n_envs, device=self.device),
                    torch.randint(
                        1,
                        len(self.cfg["environment"]["plants"]),
                        (self.n_envs,),
                        device=self.device,
                    ),
                    torch.randint(1, A.plant_types + 1, (self.n_envs,), device=self.device)
                    if self._plant_exploration.wait_check_ticks is not None
                    else self._scheduler.wait_species_choices,
                )
            else:
                columns = torch.zeros(self.n_envs, 25, device=self.device)
            columns, issued = self._scheduler.retain(columns, new)
            actions = columns[:, 1].long()
            executing = issued
            if (
                forecast_forks_enabled(self.cfg, self.tile_skills.phase)
                and (not absolute_outcomes(self.cfg) or self.reach_training)
                and getattr(self, "_pending_effect", None) is None
                and issued_host.any()
            ):
                from pvz_rl.learning.effect_collection import begin_effects

                self._pending_effect = begin_effects(self, actions, details, issued_host)
                self._scheduler.pending |= issued_host
                if not absolute_outcomes(self.cfg):
                    return
            with env.profiler.track("encoding"):
                counts = torch.where(executing, env.features.summary_tensor[: self.n_envs, 0], 0)
                offsets = counts.cumsum(0) - counts
                env.features.pack(executing, offsets, self._collection_packed)
            with env.profiler.track("transfer"):
                host = env.handoff.enqueue_fields(
                    "behavior",
                    dict(
                        entities=self._collection_packed[
                            : int(env.features.summary_host[issued_host, 0].sum())
                        ],
                        counts=counts,
                        offsets=offsets,
                        globals=self._last_obs.globals.index_select(0, live_slots),
                        branch_available=branch_available & executing[:, None],
                        columns=columns,
                        valid=self._runner.valid,
                        **(
                            {
                                "reach_errors": torch.from_dlpack(env.batch.reach_error),
                                "reach_prediction": self._collection_reach,
                                "home_prediction": self._collection_home,
                                "lane_zombies": torch.from_dlpack(
                                    env.features.lane_zombies[: self.n_envs]
                                ),
                                **(
                                    {
                                        "forecast_facts": torch.from_dlpack(
                                            env.features.forecast_facts[: self.n_envs]
                                        )
                                    }
                                    if forecast_version(self.cfg) >= 2
                                    else {}
                                ),
                                **(
                                    {"effect_prediction": self._collection_effect}
                                    if forecast_version(self.cfg) >= 2
                                    else {}
                                ),
                            }
                            if self.reach_training
                            else {}
                        ),
                    ),
                )
            if self.peashooter_rule_training:
                self._peashooter_teacher.before(actions, executing, branch_available)
            if self.reach_training:
                self._reach.before(actions, executing)
            if getattr(self, "_wait_outcomes", None) is not None:
                self._wait_outcomes.arm(self._pending_effect)
            env.step_device(actions, active=executing)
            forecast_host = None
            if (
                self.reach_training
                and forecast_mode(self.cfg, self.tile_skills.phase) != "snapshot_rules"
            ):
                from pvz_rl.envs.reach_tracking import FORECAST_HEADER

                count = self.cfg["environment"]["decision_ticks"] + 1
                forecast_host = env.handoff.enqueue_fields(
                    "forecast_step",
                    dict(
                        trace=torch.from_dlpack(
                            env.batch.reach_trace[:, FORECAST_HEADER : FORECAST_HEADER + count * 5]
                        ).to(torch.uint8),
                        breaches=torch.from_dlpack(env.batch.reach_trace[:, 1:6]),
                        contacts=torch.from_dlpack(env.batch.reach_trace[:, 6:11]),
                        **(
                            {"finished": torch.from_dlpack(env.features.forecast_facts)}
                            if forecast_version(self.cfg) >= 2
                            else {}
                        ),
                    ),
                )
            checks = self._plant_exploration.observe(
                actions,
                env.header_tensor[:, 12].bool(),
                columns[:, 21].long(),
                env._device_active,
                env._step_device_result[1],
                env.transition_ticks,
                draws=(
                    self._scheduler.species_coins,
                    self._scheduler.species_choices,
                    self._scheduler.wait_species_choices,
                ),
            )
            with env.profiler.track("transfer"):
                exploration_host = env.handoff.enqueue_fields(
                    "exploration",
                    dict(
                        coins=checks.fired,
                        pending=self._plant_exploration.pending,
                        plant_checks=checks.plant_checks,
                        wait_checks=checks.wait_checks,
                    ),
                )
            previous_wait = env.handoff.wait_seconds
            env.handoff.wait()
            for encoder in self.policy.encoders:
                encoder.release_retired()
            for encoder in self._teacher.encoders:
                encoder.release_retired()
            env.profiler.host("host_synchronization", env.handoff.wait_seconds - previous_wait)
            active = env._step_state_host["executing"].numpy().copy()
            if np.any(active & ~issued_host):
                raise ValueError("Executed slots differ from issued decision tickets")
            next_obs, _, dones, _, _, step = env.step_host(
                autoreset=False, active=active, packed=True
            )
            rows["active"] = active
            self._scheduler.commit(issued_host, active)
            if not bool(host["valid"]):
                raise ValueError("Q values must be finite and every decision needs a legal action")
            globals_host = np.zeros((self.n_envs, self._last_obs.globals.shape[-1]), np.float32)
            globals_host[issued_host] = host["globals"].numpy()
            globals_host[~active] = 0
            observations = PackedEntityBatch(
                host["entities"].numpy()[: int(host["counts"].sum())],
                host["offsets"].numpy(),
                host["counts"].numpy(),
                globals_host,
            )
            packed = host["columns"].numpy()
            rows["entity_count"] = host["counts"].numpy()
            rows["tick"], rows["action"] = packed[:, 0], packed[:, 1]
            rows["branch_value"], rows["tile_value"] = packed[:, 2], packed[:, 3]
            rows["placement_gap"] = packed[:, 23]
            rows["minimum_q"] = packed[:, 24]
            rows["greedy_action"] = packed[:, 4]
            rows["species_coin"] = exploration_host["coins"].numpy()
            rows["tile_coin"] = packed[:, 6]
            rows["policy_action"] = packed[:, 20]
            rows["exploration_mode"] = packed[:, 21]
            rows["pending_species"] = packed[:, 22]
            rows["next_species"] = exploration_host["pending"].numpy()
            rows["entity_omitted"] = packed[:, 17:20]
            rows["reward"] = env.last_transition_host[:, 3]
            rows["duration"] = env.last_transition_host[:, 2]
            rows["done"] = env.last_transition_host[:, 0].astype(bool)
            rows["accepted"] = env.last_action_result_host[:, 0].astype(bool)
            rows["rejection_reason"] = env.last_action_result_host[:, 1]
            branch_available_host = host["branch_available"].numpy().copy()
            rows["route"][active], rows["branch_usable"][active] = classify_actions(
                rows["action"][active],
                rows["accepted"][active],
                np.broadcast_to(True, (np.count_nonzero(active), A.size)),
                rows["rejection_reason"][active],
                branch_available=branch_available_host[active],
            )
            rows["components"] = env.last_reward_parts_host[
                :, [REWARD_FIELDS.index(k) for k in COMPONENTS]
            ]
            rows["components_valid"] = active
            rows["won"] = rows["components"][:, 0] > 0
            rows["home_entries"] = env.last_reward_parts_host[
                :, [REWARD_FIELDS.index(k) for k in ("home_outer_entries", "home_inner_entries")]
            ]
            staging = env.handoff.nbytes + self._scheduler.retained_bytes
            if env.static_forks is not None:
                staging += env.static_forks.host_bytes
            if getattr(self, "_pending_effect", None) is not None:
                staging += sum(
                    v.nbytes for v in self._pending_effect.values() if isinstance(v, np.ndarray)
                )
            if self.peashooter_rule_training:
                staging += self._peashooter_teacher.estimated_bytes
            if not self._buffer.reserve_staging(staging):
                raise MemoryError("RAM budget cannot hold actual collection staging")
            if self.reach_training:
                rows["reach_prediction"] = host["reach_prediction"].numpy()
                rows["reach_valid"] = active
                rows["home_prediction"] = host["home_prediction"].numpy()
                rows["lane_zombies"] = host["lane_zombies"].numpy()
                if forecast_host is not None:
                    rows["forecast_trace"] = (
                        forecast_host["trace"].numpy().reshape(self.n_envs, -1, 5)
                    )
                    rows["forecast_breach"] = forecast_host["breaches"].numpy()
                    rows["forecast_contacts"] = forecast_host["contacts"].numpy()
                if forecast_version(self.cfg) >= 2:
                    facts = host["forecast_facts"].numpy()
                    rows["forecast_pending"] = facts[:, 0]
                    rows["effect_prediction"] = host["effect_prediction"].numpy()
                    rows["effect_prediction_valid"] = forecast_forks_enabled(
                        self.cfg, self.tile_skills.phase
                    )
                    rows["forecast_mowers"] = (facts[:, 6, None] & (1 << np.arange(5))) != 0
                    if forecast_host is not None:
                        finished = forecast_host["finished"].numpy()
                        rows["forecast_finished"] = (finished[:, 0, None] == 0) & (
                            finished[:, 1:6] == 0
                        )
            rows["threat_depth"], rows["threat_time"], rows["threat_loss"] = -1, -1, -1
            pending = getattr(self, "_pending_effect", None)
            if pending is not None:
                if not pending["ready"]:
                    raise RuntimeError("An incomplete effect fork cannot commit a main action")
                rows["effect_actions"] = pending["actions"]
                rows["effect_valid"] = pending["valid"]
                rows["effect_labels"] = pending["labels"]
                if not absolute_outcomes(self.cfg):
                    rows["baseline_outcomes"] = env._baseline_outcomes
                from pvz_rl.learning.effect_collection import promote_executed_probes

                promote_executed_probes(env, pending, active, rows["done"])
                if not self._scheduler.pending.any():
                    self._pending_effect = None
                    self._scheduler.draining = False
            rows["reach_labels"], rows["home_labels"] = -1, -1
            presentation_started = perf_counter()
            env.action_journal.record_batch_safe(
                rows,
                packed[:, 7:17],
                env.last_action_result_host,
                env._episode_serial,
            )
            env.profiler.host("presentation", perf_counter() - presentation_started)
            rows["executed_action"] = np.where(rows["accepted"], rows["action"], 0)
            rows[~active] = np.zeros((), dtype=rows.dtype)
            rows["env"] = np.arange(self.n_envs)
            if self.reach_training:
                if bool(host["reach_errors"].any()):
                    raise RuntimeError("CUDA future-reach tracker failure; recovery retained")
                self._reach.capture(rows)
                staging += self._reach.estimated_bytes
                if not self._buffer.reserve_staging(staging):
                    raise MemoryError("RAM budget cannot hold reach tracking staging")
            storage_started = perf_counter()
            receipt = self._buffer.append(
                rows, observations, identities=(env._episode_serial, source_decisions)
            )
            if getattr(self, "_wait_outcomes", None) is not None:
                self._wait_outcomes.observe(pending, rows, receipt)
                if not self._buffer.reserve_staging(staging + self._wait_outcomes.estimated_bytes):
                    raise MemoryError("Shared RAM budget cannot hold Wait target metadata")
            if self.peashooter_rule_training:
                self._peashooter_teacher.observe(receipt)
            env.profiler.host("storage", perf_counter() - storage_started)
            self._stats["collection_logical_rows"] = (
                self._stats.get("collection_logical_rows", 0) + plan.logical_rows
            )
            self._stats["collection_padded_rows"] = (
                self._stats.get("collection_padded_rows", 0) + plan.padded_rows
            )
            widths = self._stats.setdefault("collection_entity_widths", {})
            for bucket in plan.buckets:
                key = str(bucket.entity_width)
                widths[key] = widths.get(key, 0) + len(bucket.slot_ids)
            self.collection_execution = dict(
                capacity_blocked_slots=int(self._scheduler.pending.sum()),
                scope="current_cohort",
                active_games=int(env.enabled_envs.sum()),
                logical_rows=self._stats["collection_logical_rows"],
                padded_rows=self._stats["collection_padded_rows"],
                entity_widths=dict(widths),
                current_logical_rows=plan.logical_rows,
                current_padded_rows=plan.padded_rows,
                online_rows=dict(logical=plan.logical_rows, padded=plan.padded_rows),
                actual_ema_rows=dict(logical=0, padded=0),
                graph_cache={
                    name: {
                        stage: dict(encoder.collection_metrics)
                        for stage, encoder in zip(
                            ("species", "tile"), network.encoders, strict=True
                        )
                    }
                    for name, network in (("online", self.policy), ("ema", self._teacher))
                },
                phase_memory=list(self._stats.get("phase_memory", [])),
                timings=env.profiler.snapshot(),
            )
            self._planting_samples += int(
                np.count_nonzero(active & (rows["action"] > 0) & (rows["action"] < 361))
            )
            self._stats["species_exploration_coins"] += int(rows["species_coin"][active].sum())
            self._stats["tile_exploration_coins"] += int(rows["tile_coin"][active].sum())
            for trigger in ("plant", "wait"):
                selected = exploration_host[f"{trigger}_checks"].numpy() & active
                for suffix, count in (
                    ("checks", np.count_nonzero(selected)),
                    ("commitments", np.count_nonzero(selected & rows["species_coin"])),
                ):
                    key = f"{trigger}_exploration_{suffix}"
                    self._stats[key] = self._stats.get(key, 0) + int(count)
            for name, selected in (
                ("committed_wait_decisions", rows["exploration_mode"] == 1),
                ("committed_plant_attempts", rows["exploration_mode"] == 2),
                ("committed_plant_successes", (rows["exploration_mode"] == 2) & rows["accepted"]),
            ):
                self._stats[name] = self._stats.get(name, 0) + int(
                    np.count_nonzero(active & selected)
                )
            self._stats["exploratory_changes"] += int(
                np.count_nonzero(active & (rows["action"] != rows["greedy_action"]))
            )
            self._simulation_ticks += int(rows["duration"][active].sum())
            self._first, self._last_obs = False, next_obs
            self.num_timesteps += int(active.sum())
            for index, info in step.completed.items():
                info["episode_metrics"]["time_reward_pending"] = True
                info["episode_slot"] = index
                self._pending_episodes[index] = dict(
                    episode=int(env._episode_serial[index]),
                    metrics=info["episode_metrics"],
                    record_id=f"cohort:{self._n_updates}:slot:{index}",
                )
            if not hasattr(callback, "cfg"):
                if self.tile_skills.active:
                    skills = self.tile_skills
                    skills.episodes[skills.phase] += len(step.completed)
                    self.tile_skill_state = skills.to_dict()
                else:
                    self.training_games += len(step.completed)
            callback.update_locals(
                {
                    "infos": step.infos(active_only=True),
                    "dones": dones,
                    "step_batch": step,
                    "episode_infos": list(step.completed.values()),
                }
            )
            # Publish the phase before callbacks may request an interruption.
            if (
                not env.enabled_envs.any()
                and not self._scheduler.pending.any()
                and not self.peashooter_rule_training
                and not self.reach_training
            ):
                self._phase(CohortPhase.FINALIZE_REWARDS, callback)
            if not callback.on_step() and not draining:
                raise KeyboardInterrupt

    def _reset_execution(self):
        precision = self.cfg["training"].get("performance", {}).get("fit_precision", "bf16_master")
        performance = self.cfg["training"].get("performance", {})
        self.execution_state = dict(
            requested_precision=precision,
            precision=precision,
            microbatch=max(
                performance.get(
                    "fit_species_microbatch", performance.get("fit_encoder_microbatch", 192)
                ),
                performance.get(
                    "fit_tile_microbatch", performance.get("fit_encoder_microbatch", 192)
                ),
            ),
            species_microbatch=performance.get(
                "fit_species_microbatch", performance.get("fit_encoder_microbatch", 192)
            ),
            tile_microbatch=performance.get(
                "fit_tile_microbatch", performance.get("fit_encoder_microbatch", 192)
            ),
            token_budget=performance.get("fit_token_budget", 65536),
            fit_batch_groups=performance.get("fit_batch_groups", 1),
            checkpointing=False,
            fallbacks=[],
        )
        self._device_events = []
        self._optimizer_events = []
        self._device_seconds = getattr(self, "_device_seconds", 0.0)
        self._optimizer_seconds = 0.0
        self._last_metrics_at = 0.0
        self._cached_metrics = (None, None, None)
        self._fit_pass_start = None

    def configure_execution(self, cfg):
        """Apply approved execution settings and the requested validation interval."""
        from pvz_rl.learning.budget import refresh_validation_schedule
        from pvz_rl.learning.performance import refresh_performance

        previous = self.cfg
        self.policy.cfg = refresh_wait_exploration(refresh_performance(previous, cfg), cfg)
        self.policy.cfg = refresh_validation_schedule(self.policy.cfg, cfg)
        self.policy.cfg["logging"] = dict(cfg["logging"])
        self.policy.features_extractor.cfg = self.policy.cfg
        self.policy_kwargs["features_extractor_kwargs"]["layout_cfg"] = self.policy.cfg
        configure_exploration(self, self.cfg)
        if (
            getattr(self, "execution_state", {}).get("requested_precision")
            != cfg["training"]["performance"].get("fit_precision", "bf16_master")
            or previous["training"].get("performance") != cfg["training"].get("performance")
            or previous["policy"]["encoder_microbatch"] != cfg["policy"]["encoder_microbatch"]
            or previous["policy"]["encoder_token_budget"] != cfg["policy"]["encoder_token_budget"]
            or previous["training"].get("performance", {}).get("fit_batch_groups", 1)
            != cfg["training"].get("performance", {}).get("fit_batch_groups", 1)
        ):
            self._reset_execution()
        for encoder in self.policy.encoders:
            encoder.token_budget = cfg["policy"]["encoder_token_budget"]
            if self.phase != CohortPhase.FIT:
                encoder.microbatch = cfg["policy"]["encoder_microbatch"]
            encoder.configure_collection(self.cfg["training"]["performance"])
        if getattr(self, "_teacher", None) is not None:
            for encoder in self._teacher.encoders:
                encoder.configure_collection(self.cfg["training"]["performance"])
        for encoder in self.policy.encoders:
            for layer in encoder.layers:
                layer.query_chunk = cfg["policy"]["attention_query_chunk"]

    def _drain(self):
        iterator = getattr(self, "_batch_iterator", None)
        if isinstance(iterator, BatchPrefetch):
            for future in iterator.futures:
                future.result()
            iterator.stream.synchronize()
        for start, end in getattr(self, "_device_events", []):
            end.synchronize()
            self._device_seconds += start.elapsed_time(end) / 1000
        self._device_events = []
        for start, end in getattr(self, "_optimizer_events", []):
            end.synchronize()
            self._optimizer_seconds += start.elapsed_time(end) / 1000
        self._optimizer_events = []

    def _close_batches(self):
        iterator = getattr(self, "_batch_iterator", None)
        if iterator is not None:
            iterator.close()
        self._batch_iterator = None

    def _restart_pass(self, reason):
        self._close_batches()
        self.policy.zero_grad(set_to_none=True)
        self.execution_state["fallbacks"].append(reason)
        self._fit_pass_start = None
        self._drain()
        for network in (self.policy, getattr(self, "_teacher", None)):
            if network is not None:
                for encoder in network.encoders:
                    encoder.release_fitting()
                    encoder.release_collection()

    def _fit_step(self, callback):
        self._optimizer_committing = False
        retry = None
        try:
            self._fit_chunk(callback)
        except torch.cuda.OutOfMemoryError:
            if self._optimizer_committing:
                raise
            retry = "memory"
        except MemoryError:
            if self._optimizer_committing or self.execution_state.get("prefetch_disabled"):
                raise
            self.execution_state["prefetch_disabled"] = True
            retry = "pinned_budget"
        except FloatingPointError:
            self._restart_pass("nonfinite_bf16")
            raise
        except EncoderCompilationError as exc:
            if (
                self._optimizer_committing
                or "encoder_compilation" in self.execution_state["fallbacks"]
            ):
                raise
            for encoder in self.policy.encoders:
                encoder.disable_compilation(exc)
            retry = "encoder_compilation"
        # Retry outside the exception frame, after failed activation graphs are released.
        if retry:
            if retry in ("encoder_compilation", "pinned_budget"):
                pass  # Replay the uncommitted pass eagerly with the same precision.
            elif self.execution_state["microbatch"] > 16:
                self.execution_state["microbatch"] = max(
                    16, self.execution_state["microbatch"] // 2
                )
            elif not self.execution_state["checkpointing"]:
                self.execution_state["checkpointing"] = True
            else:
                self._restart_pass("memory_exhausted")
                raise RuntimeError(
                    "Encoder allocation failed after bounded memory fallbacks; no entities removed"
                )
            self._restart_pass(retry)
            if retry == "memory":
                torch.cuda.empty_cache()

    def _prepare_bomb_temperature(self):
        settings = placement_settings(self.cfg, self.tile_skills.phase)
        if (
            not is_placement_skill(self.tile_skills.phase)
            or settings.get("ranking_objective") != BOMB_OBJECTIVE
        ):
            return
        # These attributed-hit outcomes and the settings are already checkpointed.
        # Reconstruct the same scalar on pass replay; never use validation or a
        # rolling window, and never change it between microbatches/passes.
        episodes = self._pending_episodes
        if not episodes or len(episodes) != self.cohort_games:
            raise RuntimeError("Bomb fitting temperature requires a completed cohort")
        rate = sum(bool(item["metrics"]["win"]) for item in episodes.values()) / len(episodes)
        self._bomb_fit_temperature = bomb_softmax_temperature(
            settings,
            rate,
            skill=self.tile_skills.phase,
        )
        prefix = "attacker" if self.tile_skills.phase == "attacker_pre" else "bomb"
        self.logger.record(f"train/{prefix}_softmax_temperature", self._bomb_fit_temperature)

    def _start_pass(self):
        from pvz_rl.learning.forecast_metrics import reset_calibration

        reset_calibration(self.policy)
        self._prepare_bomb_temperature()
        self._ensure_teacher()
        if self.execution_state["precision"] != "bf16_master":
            raise RuntimeError("BF16 master-weight fitting is required")
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 runtime inference is unavailable on this CUDA device")
        self.policy.zero_grad(set_to_none=True)
        self._pass_metrics = torch.zeros(10, device=self.device, dtype=torch.float64)
        self._reach_pass_metrics = torch.zeros(10, device=self.device, dtype=torch.float64)
        self._bomb_pass_metrics = torch.zeros(
            len(bomb_loss_metrics(placement_settings(self.cfg, self.tile_skills.phase))),
            device=self.device,
        )
        for encoder in self.policy.encoders:
            encoder.fitting = True
            encoder.microbatch = min(
                self.execution_state["microbatch"],
                self.execution_state.get(
                    f"{encoder.stage}_microbatch", self.execution_state["microbatch"]
                ),
            )
            encoder.token_budget = self.execution_state["token_budget"]
            encoder.activation_checkpointing = self.execution_state["checkpointing"]
        for encoder in self._teacher.encoders:
            encoder.fitting = True
            encoder.microbatch = min(
                self.execution_state["microbatch"],
                self.execution_state.get(
                    f"{encoder.stage}_microbatch", self.execution_state["microbatch"]
                ),
            )
            encoder.token_budget = self.execution_state["token_budget"]
        groups = self.execution_state["fit_batch_groups"]
        if getattr(self, "_fitting_spool", None) is None:
            self._fitting_spool = FittingSpool(self._buffer)
        if self.peashooter_rule_training:
            self._peashooter_objective = PeashooterObjective(
                self._peashooter_teacher,
                self._fitting_spool,
                self.device,
            )
            self._peashooter_pass_metrics = torch.zeros(3, device=self.device)
        if self.reach_training:
            from pvz_rl.learning.reach_supervision import ReachPhase

            if self._reach.phase not in (ReachPhase.LABELS_READY, ReachPhase.FITTING):
                raise RuntimeError("Reach labels must finish before fitting")
            self._reach.phase = ReachPhase.FITTING
        if self.reach_training:
            targets = self._fitting_spool.rows
            from pvz_rl.learning.forecast_labels import threat_eligible

            threat = threat_eligible(targets)
            self._reach_decisions = (int(threat.sum()), len(targets))
        else:
            self._reach_decisions = (0, 0)
        self._fit_kind = self.fit_stage
        self._fit_outcome_counts = (
            self._fitting_spool.tile_outcome_counts
            if self._fit_kind == "tile_only"
            else self._fitting_spool.outcome_counts
        )
        self._fit_consistency_counts = self._fitting_spool.consistency_counts
        if not hasattr(self._fitting_spool, "coefficients"):
            self._fitting_spool.coefficients = CohortCoefficients.prepare(
                self._fitting_spool.outcome_counts,
                self._fit_consistency_counts,
                self.device,
                self.cfg["training"]["objective"]["accepted_outcome_share"],
            )
        if not hasattr(self._fitting_spool, "tile_coefficients"):
            self._fitting_spool.tile_coefficients = CohortCoefficients.prepare(
                self._fitting_spool.tile_outcome_counts,
                np.zeros((2, 2), dtype=np.int64),
                self.device,
                self.cfg["training"]["objective"]["accepted_outcome_share"],
            )
        self._fit_coefficients = self._fitting_spool.coefficients
        if self._fit_kind == "tile_only":
            self._fit_coefficients = self._fitting_spool.tile_coefficients
        args = self._buffer, self.batch_size * groups, self._fitting_spool
        self._fit_pass_start = torch.cuda.Event(enable_timing=True)
        self._fit_pass_start.record()
        if self.cfg["training"].get("performance", {}).get(
            "prefetch", False
        ) and not self.execution_state.get("prefetch_disabled"):
            try:
                self._batch_iterator = BatchPrefetch(
                    args[0],
                    args[1],
                    self.device,
                    self.cfg["encoding"]["max_entities"],
                    spool=args[2],
                )
            except MemoryError:
                self._batch_iterator = iter(flat_batches(*args))
        else:
            self._batch_iterator = iter(flat_batches(*args))

    def _publish_fit_metrics(self, *, final=False):
        if final:
            loss, selected_sum, selected_n, gap_sum, gap_n, consistency, *totals = (
                self._pass_metrics.cpu().tolist()
            )
            for key, value in consistency_diagnostics(consistency, totals).items():
                self.logger.record("train/" + key, value)
            self._cached_metrics = (
                loss,
                selected_sum / selected_n if selected_n else None,
                gap_sum / gap_n if gap_n else None,
            )
            self._cached_fit_examples = int(selected_n)
            if self.reach_training:
                (
                    threat,
                    depth_brier,
                    depths,
                    decisions,
                    effect,
                    effect_brier,
                    effects,
                    effect_decisions,
                    depth_correct,
                    effect_correct,
                ) = self._reach_pass_metrics.cpu().tolist()
                from pvz_rl.learning.forecast_metrics import calibration_report

                self._stats["forecast_calibration"] = calibration_report(self.policy)
                (
                    time_brier,
                    times,
                    time_correct,
                    loss_brier,
                    losses,
                    loss_correct,
                    changed_correct,
                    changed,
                ) = self.policy._forecast_score_totals[:8].cpu().tolist()
                tp, fp, fn, change_brier, unchanged_correct, unchanged = (
                    self.policy._forecast_score_totals[8:14].cpu().tolist()
                )
                conditional_correct, conditional_count = (
                    self.policy._forecast_score_totals[14:].cpu().tolist()
                )
                for key, value in dict(
                    threat_loss=threat,
                    effect_loss=effect if effects else None,
                    threat_depth_brier=depth_brier / depths if depths else None,
                    effect_brier=effect_brier / effects if effects else None,
                    threat_depth_accuracy=depth_correct / depths if depths else None,
                    effect_accuracy=effect_correct / effects if effects else None,
                    threat_time_brier=time_brier / times if times else None,
                    threat_time_accuracy=time_correct / times if times else None,
                    threat_loss_brier=loss_brier / losses if losses else None,
                    threat_loss_accuracy=loss_correct / losses if losses else None,
                    changed_effect_accuracy=changed_correct / changed if changed else None,
                    unchanged_effect_accuracy=unchanged_correct / unchanged if unchanged else None,
                    effect_change_precision=tp / (tp + fp) if tp + fp else None,
                    effect_change_recall=tp / (tp + fn) if tp + fn else None,
                    effect_change_brier=change_brier / effects
                    if effects and not absolute_outcomes(self.cfg)
                    else None,
                    effect_conditional_accuracy=conditional_correct / conditional_count
                    if conditional_count
                    else None,
                    threat_exact_depths=depths,
                    effect_labeled_terms=effects,
                    threat_eligible_decisions=decisions,
                    effect_eligible_decisions=effect_decisions,
                    total_loss=loss,
                ).items():
                    self.logger.record("train/" + key, value)
                    self._stats[key] = value
                if absolute_outcomes(self.cfg):
                    self._stats["outcome_objective"] = "absolute_outcome_v3"
                    self._stats["outcome_loss"] = effect if effects else None
                    self._stats["outcome_accuracy"] = effect_correct / effects if effects else None
                    self._stats["outcome_brier"] = effect_brier / effects if effects else None
                loss -= self.reach_weight * threat + self.home_weight * effect
                self._cached_metrics = (loss, self._cached_metrics[1], self._cached_metrics[2])
            self.logger.record("train/primary_loss", loss)
            self._stats["primary_loss"] = loss
            if is_placement_skill(self.tile_skills.phase):
                self._cached_fit_examples = self.cohort_games
                for key, number in zip(
                    bomb_loss_metrics(
                        placement_settings(self.cfg, self.tile_skills.phase),
                        prefix="attacker" if self.tile_skills.phase == "attacker_pre" else "bomb",
                    ),
                    self._bomb_pass_metrics.cpu().tolist(),
                    strict=True,
                ):
                    self.logger.record("train/" + key, number)
                    self._stats[key] = number
            if self.peashooter_rule_training:
                values = self._peashooter_pass_metrics.cpu().tolist()
                self._cached_fit_examples = self._peashooter_objective.examples
                self.logger.record("train/peashooter_cross_entropy_loss", values[0])
                self.logger.record(
                    "train/peashooter_preferred_column_accuracy",
                    values[1] / values[2] if values[2] else None,
                )
        loss, selected_error, placement_gap = self._cached_metrics
        self.logger.record("train/q_loss", loss)
        self.logger.record("train/selected_action_error", selected_error)
        self.logger.record("train/placement_gap", placement_gap)
        self._stats.update(
            q_loss=loss, selected_action_error=selected_error, placement_gap=placement_gap
        )
        self._last_metrics_at = perf_counter()

    def _fit_chunk(self, callback):
        viewer = getattr(self.env, "live_view", None)
        if viewer is not None and viewer.session.enabled:
            viewer.poll()
            viewer.session.publish()
        if self._fit_epoch >= self.total_fit_passes:
            self._phase(CohortPhase.SYNCHRONIZE, callback)
            return
        if getattr(self, "_batch_iterator", None) is None:
            self._start_pass()
        item = next(self._batch_iterator, None)
        if item is None:
            if self._fit_pass_start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._device_events.append((self._fit_pass_start, end))
                self._fit_pass_start = None
            self._publish_fit_metrics(final=True)
            if not torch.isfinite(self._pass_metrics).all():
                raise FloatingPointError("Non-finite two-stage Q loss")
            parameters = self.policy.master_parameters(self.fitting_parameter_stage)
            try:
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.max_grad_norm, error_if_nonfinite=True
                )
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                raise FloatingPointError("Non-finite two-stage gradients") from exc
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self._optimizer_committing = True
            self.policy.optimizer.step()
            self.policy.sync_runtime_from_masters()
            self._update_teacher(self.fitting_parameter_stage)
            self._optimizer_committing = False
            end.record()
            self._optimizer_events.append((start, end))
            self.policy.zero_grad(set_to_none=True)
            self._stats["q_optimizer_steps"] += 1
            self.optimizer_commits += 1
            phase_key = self._fit_kind
            self._stats[f"{phase_key}_optimizer_steps"] += 1
            self._stats[f"{phase_key}_fit_examples"] += self._cached_fit_examples
            self._fit_epoch += 1
            self._close_batches()
            self.logger.record("train/q_grad_norm", float(norm))
            self._stats["q_grad_norm"] = float(norm)
        else:
            rows, observations, *metadata = item
            self.policy.set_training_mode(True)
            if self.tile_skills.active and self.reach_training:
                loss, selected_error, placement_gap = self._placement_reach_loss(rows, observations)
            elif self.peashooter_rule_training:
                loss, selected_error, placement_gap, detail = self._peashooter_objective.loss(
                    self.policy,
                    rows,
                    observations,
                )
                self._peashooter_pass_metrics += detail
            elif is_placement_skill(self.tile_skills.phase):
                loss, selected_error, placement_gap = self._bomb_loss(rows, observations)
            else:
                loss, selected_error, placement_gap = batch_loss(
                    self.policy,
                    rows,
                    self._fit_outcome_counts,
                    observations,
                    metadata[0] if metadata else None,
                    teacher=self._teacher,
                    consistency_counts=self._fit_consistency_counts,
                    coefficients=self._fit_coefficients,
                    fit_stage=self._fit_kind,
                    reach_decision_count=self._reach_decisions,
                    reach_enabled=self.reach_training,
                )
            if self.reach_training:
                self._reach_pass_metrics += self.policy._last_reach_metrics
                if (
                    self.cfg["training"]["forecasts"]["measure_gradient_norms"]
                    and "forecast_gradient_ratios" not in self._stats
                ):
                    from pvz_rl.learning.forecast_objective import gradient_norm_ratios

                    self._stats["forecast_gradient_ratios"] = gradient_norm_ratios(self.policy)
            loss.backward()
            self.policy.accumulate_master_gradients()
            self.policy.clear_runtime_grad(set_to_none=True)
            self.policy._primary_graph = None
            self.policy._forecast_components = None
            self._pass_metrics += torch.cat(
                (
                    torch.stack(
                        (
                            loss.detach().double(),
                            selected_error.detach().double().sum(),
                            loss.new_tensor(selected_error.numel()).double(),
                            placement_gap.detach().double().sum(),
                            loss.new_tensor(placement_gap.numel()).double(),
                            self.policy._last_species_consistency_loss.double(),
                        )
                    ),
                    self.policy._last_species_consistency_metrics,
                )
            )
            if perf_counter() - self._last_metrics_at >= self.cfg["logging"]["progress_seconds"]:
                self._publish_fit_metrics()
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _placement_reach_loss(self, rows, observations):
        from pvz_rl.envs.encoding import collate_observations
        from pvz_rl.learning.forecast_objective import forecast_auxiliary

        observations = collate_observations(observations, self.device)
        branches, _ = action_parts(torch.as_tensor(np.array(rows["action"]), device=self.device))
        requested = np.flatnonzero(rows["active"] & rows["branch_usable"] & (rows["action"] > 0))
        at = torch.as_tensor(requested, device=self.device)
        features = next(self.policy.parameters()).new_zeros(
            (len(rows), 45, self.cfg["policy"]["shared_width"])
        )
        with self.policy.board_context(observations) as memory:
            if len(at):
                features = features.index_copy(
                    0,
                    at,
                    self.policy.tile_features(observations, branches[at], rows=at, memory=memory),
                )
            null = self.policy.null_features(memory)
            if self.peashooter_rule_training:
                loss, error, gap, detail = self._peashooter_objective.loss(
                    self.policy, rows, observations, features=features, null=null
                )
                self._peashooter_pass_metrics += detail
            else:
                loss, error, gap = self._bomb_loss(rows, observations, features=features, null=null)
            self.policy._last_reach_metrics = torch.zeros(
                10, device=self.device, dtype=torch.float64
            )
            self.policy._primary_graph = loss
            auxiliary, metrics = forecast_auxiliary(
                self.policy,
                observations,
                rows,
                memory,
                features=features[at],
                feature_rows=requested,
                decision_count=self._reach_decisions,
                stage=self.tile_skills.phase,
            )
            loss = loss + auxiliary
            self.policy._last_reach_metrics = metrics
        return loss, error, gap

    def _bomb_loss(self, rows, observations, *, features=None, null=None):
        from pvz_rl.envs.encoding import collate_observations
        from pvz_rl.learning.placement_labels import set_ranking_loss

        labels = [self.env.tile_boards[int(index)] for index in rows["env"]]
        observations = collate_observations(observations, self.device)
        branches = [
            self.cfg["environment"]["plants"].index(label.get("supplied_plant", "cherry_bomb")) + 1
            for label in labels
        ]
        values = (
            self.policy.fitting_tiles(
                observations,
                torch.tensor(branches, device=self.device, dtype=torch.long),
                rows=torch.arange(len(rows), device=self.device),
            ).float()
            if features is None
            else self.policy.tile_values(features, null=null).float()
        )
        settings = placement_settings(self.cfg, self.tile_skills.phase)
        positives = [label["feasible_tiles"] for label in labels]
        objective = settings.get("ranking_objective", "top1_hinge_v1")
        if objective == BOMB_OBJECTIVE:
            loss = bomb_set_cross_entropy(
                values,
                positives,
                temperature=self._bomb_fit_temperature,
            )
            detail = loss.detach().reshape(1)
        elif objective in DISTANCE_OBJECTIVES:
            illegal = [
                label["absolute_infeasible_tiles"] if bomb_max_illegal_distance(settings) else []
                for label in labels
            ]
            loss, detail = bomb_distance_loss(
                values,
                positives,
                illegal,
                rows=self.cfg["environment"]["rows"],
                cols=self.cfg["environment"]["cols"],
                margin=bomb_ranking_margin(settings),
                weight=settings["distance_weight"],
                temperature=settings["distance_temperature"],
            )
        else:
            # Saved runs retain their original objective, including its margin.
            loss = set_ranking_loss(values, positives, margin=settings["ranking_margin"])
            detail = torch.stack((loss.detach(), loss.detach() * 0))
        fraction = len(rows) / self.cohort_games
        loss = loss * fraction
        self._bomb_pass_metrics += detail * fraction
        self.policy._last_species_consistency_loss = loss.detach() * 0
        self.policy._last_species_consistency_metrics = loss.new_zeros(4, dtype=torch.float64)
        return loss, values.new_empty(0), values.new_empty(0)

    def _synchronize(self, callback):
        self._drain()
        if self.reach_training:
            from pvz_rl.learning.reach_supervision import ReachPhase

            self._reach.phase = ReachPhase.COMMITTED
            self._stats["reach_tracking"] = dict(self._reach.metrics)
        self._stats["trajectory_bytes"] = (
            self._buffer.size * self._buffer.dtype.itemsize + self._buffer.entity_size * 44
        )
        self._stats["trajectory_ram_bytes"] = self._buffer.ram_used
        self._stats["ema_version"] = self._teacher_version
        self._stats["ema_compilation_status"] = self._teacher.compilation_status
        self._stats["optimizer_seconds"] = self._optimizer_seconds
        self._stats["fit_precision"] = self.execution_state["precision"]
        self._stats["fit_batch_groups"] = self.execution_state["fit_batch_groups"]
        self._stats["encoder_microbatch"] = self.execution_state["microbatch"]
        self._stats["encoder_token_budget"] = self.execution_state["token_budget"]
        self._stats["execution_fallbacks"] = list(self.execution_state["fallbacks"])
        if is_placement_skill(self.tile_skills.phase):
            distances = [
                item["metrics"]["selected_tile_distance"]
                for item in self._pending_episodes.values()
                if item["metrics"].get("selected_tile_distance") is not None
            ]
            self._stats["bomb_selected_tile_distance"] = (
                sum(distances) / len(distances) if distances else None
            )
            if self.tile_skills.phase == "attacker_pre":
                metrics = [item["metrics"] for item in self._pending_episodes.values()]
                self._stats["attacker_selected_tile_distance"] = self._stats.pop(
                    "bomb_selected_tile_distance"
                )
                self._stats["attacker_fallback_rate"] = sum(m["fallback"] for m in metrics) / len(
                    metrics
                )
                self._stats["attacker_accuracy"] = sum(m["win"] for m in metrics) / len(metrics)
                self._stats["attacker_breakdown"] = {
                    f"{kind}/{count}": dict(
                        episodes=len(group),
                        correct=sum(m["win"] for m in group),
                        fallbacks=sum(m["fallback"] for m in group),
                    )
                    for kind in self.cfg["training"]["tile_scenarios"]["attacker"][
                        "provided_plants"
                    ]
                    for count in (1, 2)
                    if (
                        group := [
                            m
                            for m in metrics
                            if m["supplied_plant"] == kind and m["zombie_count"] == count
                        ]
                    )
                }
        self._stats["fit_schedule"] = self.fit_schedule.summary()
        self._stats["fitting_spool_rows"] = len(getattr(self, "_fitting_spool", None) or ())
        spool = getattr(self, "_fitting_spool", None)
        self._stats["fitting_compaction"] = dict(spool.compaction) if spool is not None else {}
        self._stats["fitting_workloads"] = dict(spool.workloads) if spool is not None else {}
        self._stats["spool_preparation_seconds"] = (
            spool.preparation_seconds if spool is not None else 0
        )
        self._stats["compilation_status"] = getattr(self.policy, "compilation_status", "disabled")
        self._stats["compilation_backend"] = getattr(self.policy, "compilation_backend", None)
        self._stats["collection_graphs"] = {
            name: {
                stage: dict(encoder.collection_metrics)
                for stage, encoder in zip(("species", "tile"), network.encoders, strict=True)
            }
            for name, network in (("online", self.policy), ("ema", self._teacher))
        }
        if getattr(self, "_fitting_spool", None) is not None:
            self._fitting_spool.close()
            self._fitting_spool = None
        self._peashooter_objective = None
        super()._synchronize(callback)

    def save(self, *args, **kwargs):
        with atomic_transition():
            self._drain()
            if self.phase == CohortPhase.FIT:
                self._close_batches()
                self.policy.zero_grad(set_to_none=True)
                if getattr(self, "_fitting_spool", None) is not None:
                    self._fitting_spool.close()
                    self._fitting_spool = None
            return super().save(*args, **kwargs)

    def learn(self, *args, **kwargs):
        try:
            return super().learn(*args, **kwargs)
        finally:
            self._drain()
            self._close_batches()
            if getattr(self, "_fitting_spool", None) is not None:
                self._fitting_spool.close()
                self._fitting_spool = None

    def _restore_runtime(self):
        original = self.runtime_state
        try:
            self._restore_runtime_payload()
        except BaseException:
            self.runtime_state = original
            raise

    def _restore_runtime_payload(self):
        extra = self.runtime_state
        self._rollout_start_notified = (
            extra.get("rollout_start_notified", False) if extra else False
        )
        self.forecast_pretraining_state = (
            extra.get("forecast_pretraining_state", "pending") if extra else "pending"
        )
        if extra:
            from pvz_rl.learning.peashooter_teacher import validate_rule_recovery

            validate_rule_recovery(extra, self.cfg)
        if extra and extra.get("tile_scenarios") is not None:
            if (
                is_placement_skill(self.tile_skills.phase)
                or (self.tile_skills.active and self.tile_skills.stage_order is None)
            ) and (
                placement_settings(self.cfg, self.tile_skills.phase).get(
                    "ranking_objective", "top1_hinge_v1"
                )
                != extra.get("tile_ranking_objective")
            ):
                raise ValueError("Cherry Bomb objective changed; use --init-from for fresh fitting")
            if extra.get("tile_scenario_protocol") != TILE_SCENARIO_PROTOCOL:
                raise ValueError("Unsupported tile prerequisite recovery protocol")
            if TileSkillState(**extra["tile_scenarios"]).to_dict() != self.tile_skills.to_dict():
                raise ValueError("Tile prerequisite state disagrees with checkpoint")
        elif self.tile_skills.active:
            raise ValueError("Missing tile prerequisite recovery state")
        if extra and extra.get("environment") is not None:
            if extra.get("objective") != self.cfg["training"].get("objective"):
                raise ValueError("Checkpoint objective differs; use weights-only initialization")
            if self.phase not in (CohortPhase.IDLE, CohortPhase.PRETRAIN) and any(
                extra.get(key) is None
                for key in (
                    "teacher",
                    "pending_episodes",
                    "compilation",
                    "plant_exploration",
                    "collection_scheduler",
                )
            ):
                raise ValueError("Incomplete objective recovery state")
        super()._restore_runtime()
        if self.phase == CohortPhase.PRETRAIN:
            from pvz_rl.learning.forecast_pretraining import ForecastPretraining

            self._ensure_teacher()
            self._teacher.load_state_dict(extra["teacher"])
            self._teacher_version = extra["teacher_version"]
            self._pretrainer = ForecastPretraining(self, extra["forecast_pretraining"])
            self._batch_iterator = None
            self._device_events, self._optimizer_events = [], []
            return
        if extra and extra.get("reach_supervision") is not None:
            from zipfile import ZipFile

            from pvz_rl.learning.reach_supervision import ReachSupervision

            with ZipFile(self._checkpoint_source) as archive:
                self._reach = ReachSupervision(
                    self.env, self._buffer, extra["reach_supervision"], archive
                )
        elif self.reach_training and self.phase != CohortPhase.IDLE:
            if self.phase == CohortPhase.COLLECT and self._buffer.size == 0:
                from pvz_rl.learning.reach_supervision import ReachSupervision

                self._reach = ReachSupervision(self.env, self._buffer)
            else:
                raise ValueError("Missing private reach recovery; start fresh")
        if extra and extra.get("peashooter_teacher") is not None:
            from zipfile import ZipFile

            from pvz_rl.learning.peashooter_teacher import PeashooterTeacher

            identity = tuple(extra["peashooter_teacher"]["identity"])
            if self.phase == CohortPhase.COLLECT and identity != (
                self.cohort_number,
                self.optimizer_commits,
            ):
                raise ValueError("Peashooter oracle weights differ from saved cohort")
            with ZipFile(self._checkpoint_source) as archive:
                self._peashooter_teacher = PeashooterTeacher(
                    self.env,
                    self._buffer,
                    identity,
                    extra["peashooter_teacher"],
                    archive,
                )
        if extra and extra.get("plant_exploration") is not None:
            self._plant_exploration = CommittedPlantExploration(
                self.n_envs, self.device, self.plant_exploration_rate, self.exploration_rate
            )
            self._plant_exploration.restore(extra["plant_exploration"])
            for trigger in ("plant", "wait"):
                for suffix in ("checks", "commitments"):
                    self._stats.setdefault(f"{trigger}_exploration_{suffix}", 0)
            self._stats.update(exploration_status(self))
            if self._buffer is not None and self._buffer.exploration != dict(
                plant_epsilon=self.plant_exploration_rate, tile_epsilon=self.exploration_rate
            ):
                raise ValueError("Trajectory exploration probabilities differ from recovery state")
            self._plant_costs = torch.tensor(
                [
                    self.env.batch.rules.plants[kind]["cost"]
                    for kind in self.cfg["environment"]["plants"]
                ],
                device=self.device,
            )
        self._passivity_requested = copy.deepcopy(
            extra.get("passivity_requested", passivity_settings(self.cfg))
            if extra
            else passivity_settings(self.cfg)
        )
        self._passivity_active = copy.deepcopy(extra.get("passivity_active")) if extra else None
        if self._buffer is not None and self._buffer.cfg != self.cfg:
            from pvz_rl.learning.training_requirements import resume_protocol

            if resume_protocol(self._buffer.cfg, "masked") != resume_protocol(self.cfg, "masked"):
                raise ValueError("Trajectory objective/configuration differs from policy")
        if extra and "pending_episodes" in extra:
            self._pending_episodes = extra["pending_episodes"]
        if extra and "teacher" in extra:
            self._ensure_teacher()
            self._teacher.load_state_dict(extra["teacher"])
            self._teacher_version = extra["teacher_version"]
            for name, network in (("policy", self.policy), ("teacher", self._teacher)):
                for stage, encoder in zip(("species", "tile"), network.encoders, strict=True):
                    compilation = extra["compilation"][name][stage]
                    if compilation.get("status") not in (
                        "disabled",
                        "requested",
                        "compiled",
                        "fallback",
                        "unavailable",
                    ):
                        raise ValueError("Invalid encoder execution recovery state")
                    if compilation["status"] == "fallback":
                        encoder.disable_compilation(compilation["error"])
                    elif compilation_requested(self.cfg["training"]["performance"]):
                        encoder.enable_compilation()
        if extra and extra.get("collection_reach") is not None:
            self._collection_reach = extra["collection_reach"].to(self.device)
            self._collection_home = extra["collection_home"].to(self.device)
            if extra.get("collection_effect") is not None:
                self._collection_effect = extra["collection_effect"].to(self.device)
        self._scheduler = CollectionScheduler(self.n_envs, self.device)
        if extra and extra.get("collection_scheduler") is not None:
            self._scheduler.restore(extra["collection_scheduler"])
        self._pending_effect = extra.get("pending_effect") if extra else None
        self._wait_outcomes = None
        if extra and extra.get("wait_outcomes") is not None:
            from pvz_rl.learning.wait_outcomes import WaitOutcomes

            self._wait_outcomes = WaitOutcomes(self.env, self._buffer, extra["wait_outcomes"])
        elif (
            absolute_outcomes(self.cfg)
            and self.reach_training
            and forecast_forks_enabled(self.cfg, self.tile_skills.phase)
            and self.phase == CohortPhase.COLLECT
        ):
            if self._buffer.size:
                raise ValueError("Missing semantic Wait trajectory recovery")
            from pvz_rl.learning.wait_outcomes import WaitOutcomes

            self._wait_outcomes = WaitOutcomes(self.env, self._buffer)
        if (
            forecast_forks_enabled(self.cfg, self.tile_skills.phase)
            and self.phase == CohortPhase.COLLECT
        ):
            from pvz_rl.learning.effect_collection import validate_effect_request

            validate_effect_request(self._pending_effect, self)
        if extra and extra.get("collection_valid") is not None and self._runner is not None:
            self._runner.valid &= extra["collection_valid"].to(self.device)
        if extra and extra.get("forecast_schedule"):
            self._resolved_forecast_schedule = extra["forecast_schedule"]
            self.cfg["training"]["forecasts"]["injection_active"] = (
                self._resolved_forecast_schedule["injection"]
            )
            for network in (self.policy, self._teacher):
                network.forecast_stage = self.tile_skills.phase
                network.injection_enabled = self._resolved_forecast_schedule["injection"]
                network.auxiliary_scale = self._resolved_forecast_schedule["auxiliary_scale"]
        self._td_builder = None
        if extra and extra.get("td_preparation"):
            from pvz_rl.learning.td_targets import TDTargetBuilder

            self._td_builder = TDTargetBuilder(self, extra["td_preparation"])
        self.policy.zero_grad(set_to_none=True)
        self._batch_iterator = None
        if not getattr(self, "execution_state", None):
            self._reset_execution()
        self._device_events = []
        self._optimizer_events = []
        self._last_metrics_at = 0.0
        self._optimizer_seconds = getattr(self, "_optimizer_seconds", 0.0)
        self._fit_pass_start = None
        self.collection_execution = {}
        if self.phase == CohortPhase.COLLECT:
            for network in (self.policy, getattr(self, "_teacher", None)):
                if network is not None:
                    for encoder in network.encoders:
                        encoder.microbatch = self.cfg["policy"]["encoder_microbatch"]
                        encoder.token_budget = self.cfg["policy"]["encoder_token_budget"]
                        encoder.activation_checkpointing = False

    def _extra_runtime(self):
        skills = dict(
            forecast_pretraining=self._pretrainer.snapshot()
            if getattr(self, "_pretrainer", None)
            else None,
            forecast_schedule=getattr(self, "_resolved_forecast_schedule", None),
            td_preparation=self._td_builder.snapshot()
            if getattr(self, "_td_builder", None)
            else None,
            pending_effect=getattr(self, "_pending_effect", None),
            wait_outcomes=self._wait_outcomes.snapshot()
            if getattr(self, "_wait_outcomes", None) is not None
            else None,
            collection_valid=self._runner.valid.cpu() if getattr(self, "_runner", None) else None,
            collection_reach=getattr(self, "_collection_reach", None),
            collection_home=getattr(self, "_collection_home", None),
            collection_effect=getattr(self, "_collection_effect", None),
            forecast_pretraining_state=getattr(self, "forecast_pretraining_state", "pending"),
            rollout_start_notified=getattr(self, "_rollout_start_notified", False),
            reach_supervision=self._reach.snapshot() if getattr(self, "_reach", None) else None,
            tile_scenarios=self.tile_skill_state,
            tile_scenario_protocol=TILE_SCENARIO_PROTOCOL,
            tile_ranking_objective=placement_settings(self.cfg, self.tile_skills.phase).get(
                "ranking_objective"
            ),
            peashooter_teacher=self._peashooter_teacher.snapshot()
            if getattr(self, "_peashooter_teacher", None)
            else None,
        )
        if getattr(self, "_teacher", None) is None:
            return skills
        return dict(
            **skills,
            collection_scheduler=self._scheduler.snapshot()
            if getattr(self, "_scheduler", None)
            else None,
            plant_exploration=self._plant_exploration.snapshot()
            if getattr(self, "_plant_exploration", None) is not None
            else None,
            teacher={k: v.cpu() for k, v in self._teacher.runtime_state_dict().items()},
            teacher_version=self._teacher_version,
            pending_episodes=getattr(self, "_pending_episodes", {}),
            passivity_requested=copy.deepcopy(
                getattr(self, "_passivity_requested", passivity_settings(self.cfg))
            ),
            passivity_active=copy.deepcopy(getattr(self, "_passivity_active", None)),
            objective=dict(self.cfg["training"]["objective"]),
            compilation={
                name: {
                    stage: dict(status=encoder.compilation_status, error=encoder.compilation_error)
                    for stage, encoder in zip(("species", "tile"), network.encoders, strict=True)
                }
                for name, network in (("policy", self.policy), ("teacher", self._teacher))
            },
        )

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "_reach",
            "_reach_pass_metrics",
            "_collection_reach",
            "_collection_home",
            "_collection_effect",
            "_batch_iterator",
            "_fitting_spool",
            "_pass_metrics",
            "_device_events",
            "_optimizer_events",
            "_fit_pass_start",
            "_collection_packed",
            "_scheduler",
            "_collection_callback",
            "_teacher",
            "_plant_exploration",
            "_plant_costs",
            "_peashooter_teacher",
            "_peashooter_objective",
            "_peashooter_pass_metrics",
            "_bomb_pass_metrics",
            "_pending_episodes",
            "_pretrainer",
            "_td_builder",
            "_pending_effect",
            "_wait_outcomes",
        ]
