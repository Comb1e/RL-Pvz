"""Q training with validation-only checkpoint selection and resumable artifacts."""

from __future__ import annotations

import copy
import json
import math
import secrets
import shutil
from collections import Counter, deque
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3.common.callbacks import BaseCallback

from pvz_rl.config import (
    bomb_ranking_margin,
    bomb_softmax_temperature,
    cohort_envs,
    curriculum_probe_seeds,
    output_settings,
    random_validation_seeds,
    runtime_settings,
    simulator,
    validate_config,
)
from pvz_rl.envs.rewards import LEDGER_METRICS, REWARD_METRICS
from pvz_rl.envs.scenarios import difficulty_weights
from pvz_rl.envs.sky_sun import peashooter_sky_sun_rate
from pvz_rl.envs.tile_scenarios import is_placement_skill, placement_settings
from pvz_rl.evaluation.runner import evaluate, summarize
from pvz_rl.learning.budget import (
    budget_target,
    effective_limits,
    evaluation_interval,
    progress_value,
    until_stage_complete,
    uses_games,
)
from pvz_rl.learning.cohort import atomic_transition
from pvz_rl.learning.curriculum import (
    CurriculumState,
    initial_state,
    pre_stage,
    selected_stage,
    stage_distribution,
    teaching_enabled,
    validation_after_stage,
)
from pvz_rl.learning.deadline import BudgetExpired, RunBudget
from pvz_rl.learning.exploration import (
    EXPLORATION_PROTOCOL,
    apply_exploration_state,
    configure_exploration,
    exploration_state,
)
from pvz_rl.learning.objective import components, training_rewards
from pvz_rl.learning.peashooter_rules import RULE_OBJECTIVE
from pvz_rl.learning.performance import compilation_requested
from pvz_rl.learning.tile_curriculum import initial_tile_state, required_validation_wins
from pvz_rl.learning.training_requirements import (
    FIT_SCHEDULE_PROTOCOL,
    parameter_changes,
    require_cuda_training,
    require_supported_policy,
    resume_protocol,
    transfer_protocol,
)
from pvz_rl.learning.validation_trigger import ValidationTrigger
from pvz_rl.monitoring.metrics import episode_task, mean_agent_actions, task_statistics
from pvz_rl.monitoring.progress import CollectionProgress, Phase, ProgressReporter
from pvz_rl.monitoring.timing import TrainingTimings
from pvz_rl.provenance import append_jsonl, file_hash, metadata, verify_engine, write_json


def vector_env(cfg, condition, learner_seed, family="preset"):
    from pvz_rl.envs.cuda_env import CudaVecEnv

    require_cuda_training(cfg, condition)
    return CudaVecEnv(cfg, condition, learner_seed, family)


def build_model(cfg, condition, env, seed, log_dir=None):
    """Build the shared learner; train() enables prerequisites for fresh runs."""
    from pvz_rl.envs.cuda_env import CudaVecEnv

    require_cuda_training(cfg, condition)
    if not isinstance(env, CudaVecEnv):
        raise ValueError("Training requires the CUDA tensor environment")
    t = cfg["training"]
    from pvz_rl.learning.sequential_training import CudaSequentialQ
    from pvz_rl.policy.two_stage_policy import TwoStageQPolicy

    algorithm, policy_class = CudaSequentialQ, TwoStageQPolicy
    policy_kwargs = {"features_extractor_kwargs": {"layout_cfg": cfg}}
    model = algorithm(
        policy_class,
        env,
        learning_rate=t["learning_rate"],
        batch_size=t["batch_size"],
        n_epochs=t["n_epochs"],
        max_grad_norm=t["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=t["device"],
        verbose=0,
        tensorboard_log=str(log_dir) if log_dir else None,
    )
    model.training_games = 0
    if compilation_requested(t.get("performance", {})):
        model.policy.enable_compilation()
    configure_exploration(model, cfg)
    return model


class TrainingDeadline(Exception):
    """Raised only between complete collect/update cycles."""


class TrainingGamesComplete(Exception):
    """The requested game count was reached; the last rollout has been optimized."""


class TrainingStageComplete(Exception):
    """The selected stage passed its mastery gates after a complete Q update."""


def optimizer_step(model):
    """Count commits independently of selectively updated Adam parameters."""
    if hasattr(model, "optimizer_commits"):
        return model.optimizer_commits
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    for state in getattr(optimizer, "state", {}).values():
        step = state.get("step")
        if isinstance(step, torch.Tensor) and step.device.type == "cpu":
            return int(step)
        if isinstance(step, (int, float)):
            return int(step)
    return 0


def seconds_text(value):
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.0f}s"


class ResearchCallback(BaseCallback):
    def __init__(
        self,
        cfg,
        condition,
        learner_seed,
        output,
        validation_limit=None,
        family="preset",
        progress=None,
        deadline=None,
        wall_budget=None,
    ):
        super().__init__()
        self.cfg, self.condition, self.learner_seed = cfg, condition, learner_seed
        self.output, self.validation_limit, self.family = Path(output), validation_limit, family
        self.best_score = -math.inf
        self.next_eval = evaluation_interval(cfg)
        self.last_eval = -1
        self.started = perf_counter()
        self.initial_steps = 0
        self.initial_games = 0
        self.initial_budget_progress = 0
        self.eval_seconds = 0.0
        self.stream = None
        self.settings = output_settings(cfg)
        self.owns_progress = progress is None
        self.progress = progress or ProgressReporter.from_settings(self.output / "train.log", cfg)
        self.collection_progress = CollectionProgress(clock=self.progress.clock)
        self.recent = deque(maxlen=self.settings["logging"]["rolling_window"])
        self.recent_by_task = {}
        self.initial_task_counts = {}
        self.metrics_stream = None
        self.report_seconds = 0.0
        self.last_updates = 0
        self.last_weights = None
        self.timings = TrainingTimings()
        self.deadline = deadline
        self.wall_budget = wall_budget
        self.validation_pending = False
        self.pending_stage_validation = None
        self.validation_panels = {}
        self.evaluation_cache = {}
        self.cache_steps = -1
        self.budget_stopped = False
        self.curriculum = None
        self.validation_trigger = ValidationTrigger(cfg)
        self.hardware = None
        self.hardware_activity = "training"
        self.simulation_ticks = 0
        self.instant_actions = 0
        t = cfg["training"]
        self.target = budget_target(cfg) if uses_games(cfg) else t["total_steps"]

    def _init_callback(self):
        if self.cfg["training"].get("performance", {}).get("telemetry", False):
            from pvz_rl.monitoring.hardware import HardwareMonitor

            self.hardware = HardwareMonitor(
                self.output / "hardware-metrics.jsonl",
                seconds=self.settings["logging"]["hardware_sample_seconds"],
                device=getattr(self.model, "device", None),
            ).__enter__()

    def hardware_context(self, activity=None):
        if activity is not None:
            self.hardware_activity = activity
        viewer = getattr(getattr(self.model, "env", None), "live_view", None)
        if viewer is not None:
            from pvz_rl.presentation.live_view import Activity

            viewer.set_activity(
                {
                    "training": Activity.UPDATING,
                    "idle": Activity.UPDATING,
                    "forecast_pretraining": Activity.UPDATING,
                    "validation": Activity.VALIDATING,
                    "reporting": Activity.REPORTING,
                    "collect": Activity.COLLECTING,
                    "finalize_rewards": Activity.UPDATING,
                    "fit": Activity.UPDATING,
                    "returns": Activity.UPDATING,
                    "td_targets": Activity.UPDATING,
                    "synchronize": Activity.UPDATING,
                }[self.hardware_activity]
            )
        if self.hardware:
            cohort = getattr(self.model, "cohort_metrics", {})
            self.hardware.update(
                stage=self.active_stage,
                phase=getattr(self.model, "phase", "starting"),
                activity=self.hardware_activity,
                training_steps=self.model.num_timesteps,
                **self.q_status(),
            )
            if cohort:
                self.hardware.record_window(cohort)

    def q_status(self):
        from pvz_rl.learning.exploration import exploration_status

        buffer = getattr(self.model, "_buffer", None)
        stats = getattr(self.model, "_stats", {})
        return dict(
            planting_samples=int(buffer.group_counts[1])
            if buffer is not None and buffer.finalized
            else getattr(self.model, "_planting_samples", 0),
            q_optimizer_steps=stats.get("q_optimizer_steps", 0),
            fit_schedule=self.model.fit_schedule.summary(),
            tile_exploration_epsilon=getattr(
                getattr(self.model, "policy", None), "tile_exploration_epsilon", 0.0
            ),
            species_counts=buffer.species_counts.tolist()
            if buffer is not None and buffer.finalized
            else getattr(self.model, "cohort_metrics", {}).get("species_counts"),
            species_exploration_coins=stats.get("species_exploration_coins", 0),
            tile_exploration_coins=stats.get("tile_exploration_coins", 0),
            exploratory_changes=stats.get("exploratory_changes", 0),
            **exploration_status(self.model),
            **{
                f"{trigger}_exploration_{suffix}": stats.get(f"{trigger}_exploration_{suffix}", 0)
                for trigger in ("plant", "wait")
                for suffix in ("checks", "commitments")
            },
            passivity_requested=getattr(self.model, "_passivity_requested", None),
            passivity_active=getattr(self.model, "_passivity_active", None),
        )

    def task_counts(self):
        env = getattr(self.model, "env", None)
        current = getattr(env, "task_counts", lambda: {})()
        combined = {}
        for task in set(current) | set(self.initial_task_counts):
            now, before = current.get(task, {}), self.initial_task_counts.get(task, {})
            combined[task] = {
                key: now.get(key, 0) + (0 if key == "active_games" else before.get(key, 0))
                for key in ("started_games", "active_games", "completed_games", "transitions")
            }
        self.model.training_task_counts = combined
        return combined

    def snapshot(self):
        elapsed = perf_counter() - self.started
        active = max(elapsed - self.eval_seconds - self.report_seconds, 1e-9)
        collected = self.model.num_timesteps - self.initial_steps
        rate = collected / active
        games = getattr(self.model, "training_games", 0)
        game_rate = (games - self.initial_games) / active
        progress = progress_value(self.cfg, self.model)
        budget_rate = (progress - self.initial_budget_progress) / active
        rows = list(self.recent)
        decisions = sum(r["decisions"] for r in rows)
        purchases = sum(sum(r.get("plant_usage", {}).values()) for r in rows)
        weights = difficulty_weights(
            self.cfg,
            progress,
            budget_target(self.cfg),
            self.cfg["conditions"][self.condition]["curriculum"],
        )
        self.observe_collection_progress()
        execution = getattr(self.model, "collection_execution", None)
        if execution is None:
            execution = getattr(self.model, "_stats", {}).get("collection_execution", {})
        return {
            "collection_progress": self.collection_progress.snapshot(),
            "collection_execution": copy.deepcopy(execution),
            "lifetime_throughput": {
                "scope": "current_session",
                "denominator": "wall time excluding validation and reporting; includes fitting",
                "transitions_per_second": rate,
                "games_per_second": game_rate,
                "simulation_ticks_per_second": self.simulation_ticks / active,
                "end_to_end_transitions_per_second": collected / max(elapsed, 1e-9),
            },
            "current_cohort": self.cohort_summary(),
            **self.q_status(),
            "live_view": dict(self.model.env.live_view.stats)
            if getattr(getattr(self.model, "env", None), "live_view", None)
            else None,
            "hardware": self.hardware.latest if self.hardware else {},
            "task_counts": self.task_counts(),
            "rolling_by_task": {
                task: task_statistics(rows) for task, rows in sorted(self.recent_by_task.items())
            },
            **self.timings.snapshot(),
            "state": self.progress.state.value,
            "steps": self.model.num_timesteps,
            "training_steps": self.model.num_timesteps,
            "training_games": games,
            "tile_scenarios": self.model.tile_skills.to_dict(),
            "reach_tracking": dict(
                getattr(
                    getattr(self.model, "_reach", None),
                    "metrics",
                    self.model.cohort_metrics.get("reach_tracking", {}),
                )
            ),
            "peashooter_teacher": dict(
                getattr(
                    getattr(self.model, "_peashooter_teacher", None),
                    "metrics",
                    self.model.cohort_metrics.get("peashooter_teacher", {}),
                )
            ),
            "static_forks": dict(
                getattr(
                    getattr(self.model.env, "static_forks", None),
                    "metrics",
                    self.model.cohort_metrics.get("static_forks", {}),
                )
            ),
            "wait_outcomes": dict(
                getattr(
                    getattr(self.model, "_wait_outcomes", None),
                    "metrics",
                    self.model.cohort_metrics.get("wait_outcomes", {}),
                )
            ),
            "collection_policy": "network",
            "fitting_objective": (
                RULE_OBJECTIVE
                if self.model.peashooter_rule_training
                else "nstep_fork_forecast_v3"
                if not self.model.tile_skills.active
                else None
            ),
            "validation_trigger": self.validation_trigger.snapshot(
                self.model.tile_skills.validation_domain
                if self.model.tile_skills.active
                else selected_stage(self.cfg)
                if pre_stage(self.cfg)
                else "normal"
            ),
            "peashooter_sky_sun_rate": peashooter_sky_sun_rate(self.cfg),
            "active_skill": self.model.tile_skills.phase if self.model.tile_skills.active else None,
            "tile_validation_pending": self.pending_validation == "tile"
            if self.validation_trigger.enabled
            else self.model.tile_skills.due(self.cfg),
            "normal_cohorts": self.model.normal_updates,
            "budget_unit": "games" if uses_games(self.cfg) else "decisions",
            "budget_progress": progress,
            "budget_target": self.target,
            "target_games": self.target if uses_games(self.cfg) else None,
            "target_steps": None if uses_games(self.cfg) else self.target,
            "games_per_second": game_rate,
            "wall_seconds": elapsed,
            "validation_seconds": self.eval_seconds,
            "report_seconds": self.report_seconds,
            "decisions_per_second": rate,
            "simulation_ticks": self.simulation_ticks,
            "simulation_ticks_per_second": self.simulation_ticks / active,
            "instant_action_fraction": self.instant_actions / max(1, collected),
            "end_to_end_decisions_per_second": collected / max(elapsed, 1e-9),
            "estimated_remaining_training_seconds": (
                max(0, self.target - progress) / budget_rate
                if self.target is not None and budget_rate > 0
                else None
            ),
            "until_stage_complete": until_stage_complete(self.cfg),
            "next_episode_difficulty_weights": None
            if self.curriculum or self.model.tile_skills.active or pre_stage(self.cfg)
            else weights,
            "curriculum": self.curriculum.to_dict() if self.curriculum else None,
            "curriculum_stage": self.active_stage,
            "training_phase": str(getattr(self.model, "phase", "starting")),
            "q_prefit": getattr(self.model, "prefit_errors", {}),
            "exploration_rate": getattr(self.model, "exploration_rate", 0.0),
            "tile_exploration_epsilon": getattr(self.model, "exploration_rate", 0.0),
            "plant_exploration_epsilon": getattr(self.model, "plant_exploration_rate", 0.0),
            "plant_exploration_progress": getattr(self.model, "plant_exploration_progress", 0.0),
            "plant_exploration_at_floor": getattr(self.model, "plant_exploration_at_floor", False),
            "exploration_phase": getattr(self.model, "exploration_phase", None),
            "exploration_progress": getattr(self.model, "exploration_progress", 0.0),
            "exploration_at_floor": getattr(self.model, "exploration_at_floor", False),
            # Small test doubles and legacy reports may not carry a policy object.
            # Keep diagnostics optional so progress logging remains usable before
            # model construction and never embeds compiler tracebacks in reports.
            "compilation_status": getattr(
                getattr(self.model, "policy", None), "compilation_status", "unknown"
            ),
            "compilation_backend": getattr(
                getattr(self.model, "policy", None), "compilation_backend", None
            ),
            "compilation_error": (
                str(error)
                if (
                    error := getattr(getattr(self.model, "policy", None), "compilation_error", None)
                )
                else None
            ),
            "selected_stage": selected_stage(self.cfg),
            "stage_mastered": self.stage_mastered,
            "curriculum_incomplete": bool(
                self.curriculum and not self.curriculum.complete(self.cfg)
            ),
            "next_episode_tasks": ([self.model.tile_skills.phase], [1.0])
            if self.model.tile_skills.active
            else (stage_distribution(self.cfg, self.curriculum.stage) if self.curriculum else None),
            "rolling_episodes": len(rows),
            "rolling_pending_rewards": sum(bool(r.get("time_reward_pending")) for r in rows),
            "rolling_truncations": sum(r.get("status") == "truncated" for r in rows),
            "rolling_agent_actions": mean_agent_actions(rows),
            "rolling_tasks": dict(
                Counter(
                    r.get("level", "unknown")
                    if r.get("family") == "preset"
                    else r.get("family", "unknown")
                    for r in rows
                )
            ),
            "rolling_mower_free_lessons": sum(r.get("family") in ("diagnostic",) for r in rows),
            "rolling_win_rate": sum(r["win"] for r in rows) / len(rows) if rows else None,
            "rolling_return": sum(r["return"] for r in rows) / len(rows) if rows else None,
            "rolling_seconds": (
                sum(r["simulated_seconds"] for r in rows) / len(rows) if rows else None
            ),
            "rolling_plant_purchases": purchases / len(rows) if rows else None,
            "rolling_attacker_purchases": sum(r.get("attacker_purchases", 0) for r in rows)
            / len(rows)
            if rows
            else None,
            **{
                f"rolling_{key}": sum(r.get(key, 0) for r in rows) / len(rows) if rows else None
                for key in (*REWARD_METRICS, *LEDGER_METRICS)
            },
            "rolling_first_attacker_seconds": (
                sum(
                    r["first_attacker_seconds"]
                    for r in rows
                    if r.get("first_attacker_seconds") is not None
                )
                / sum(r.get("first_attacker_seconds") is not None for r in rows)
            )
            if any(r.get("first_attacker_seconds") is not None for r in rows)
            else None,
            "rolling_early_voluntary_digs": sum(r.get("early_voluntary_digs", 0) for r in rows)
            / len(rows)
            if rows
            else None,
            "early_digs_per_planting": (
                sum(r.get("early_voluntary_digs", 0) for r in rows) / purchases
                if purchases
                else None
            ),
            "time_budget": self.wall_budget.state() if self.wall_budget else None,
            "validation_pending": self.validation_pending or self.pending_validation is not None,
            "validation_schedule": "stage_success" if self.stage_validation else "periodic",
            "pending_stage_validation": self.pending_stage_validation,
            "validation_deferred": self.stage_validation and not self.pending_stage_validation,
            "rolling_maximum_sun": sum(r.get("maximum_sun", 0) for r in rows) / len(rows)
            if rows
            else None,
            "invalid_action_rate": (
                sum(r["invalid_actions"] for r in rows) / max(1, decisions) if rows else None
            ),
            "best_validation_win_rate": None if self.best_score == -math.inf else self.best_score,
            "cohort": getattr(self.model, "cohort_metrics", {}),
        }

    def cohort_summary(self):
        rows = [item["metrics"] for item in getattr(self.model, "_pending_episodes", {}).values()]
        count = len(rows)
        return dict(
            games=getattr(self.model, "cohort_games", 0),
            completed=count,
            wins=sum(bool(row["win"]) for row in rows),
            raw_return=sum(row["return"] for row in rows) / count if count else None,
            training_return=sum(float(training_rewards(components(row), self.cfg)) for row in rows)
            / count
            if count
            else None,
            provisional=any(row.get("time_reward_pending", False) for row in rows),
        )

    def cohort_text(self):
        summary = self.cohort_summary()
        if is_placement_skill(self.model.tile_skills.phase) or is_placement_skill(
            getattr(self.model.env, "tile_scenario", None)
        ):
            return (
                f"done {summary['completed']}/{summary['games']} "
                f"correct {summary['wins']}/{summary['games']}; placement supervision"
            )
        raw = "n/a" if summary["raw_return"] is None else f"{summary['raw_return']:.3g}"
        fitted = (
            "n/a" if summary["training_return"] is None else f"{summary['training_return']:.3g}"
        )
        marker = "~" if summary["provisional"] else ""
        return (
            f"done {summary['completed']}/{summary['games']} wins {summary['wins']}/{summary['games']} "
            f"Rraw={raw} Rtrain={fitted}{marker}"
        )

    def log_progress(self, *, force=False):
        """Write the detailed status snapshot; offer one compact line to the terminal."""
        if not force and not self.progress.due():
            return
        self.hardware_context()
        row = self.snapshot()
        write_json(self.output / "status.json", row)
        text, change = self.compact_status(row)
        self.progress.emit(text, force=force, change=change)
        self.collection_progress.advance()

    def observe_collection_progress(self):
        """Measure live work using only the already-materialized host counters."""
        model = self.model
        phase = str(getattr(model, "phase", "starting"))
        enabled = getattr(getattr(model, "env", None), "enabled_envs", None)
        size = getattr(model, "cohort_games", 0)
        active = int(enabled[:size].sum()) if phase == "collect" and enabled is not None else 0
        self.collection_progress.observe(
            transitions=model.num_timesteps,
            active_games=active,
            cohort=getattr(model, "cohort_number", getattr(model, "_n_updates", 0) + 1),
            phase=phase,
        )

    def compact_status(self, row):
        """Terminal summary from host-side status; never reads device tensors."""
        model = self.model
        phase = str(row["training_phase"])
        cohort = getattr(model, "cohort_number", getattr(model, "_n_updates", 0) + 1)
        if phase == "collect" and getattr(model, "env", None) is not None:
            size = getattr(model, "cohort_games", 0)
            finished = max(0, size - int(model.env.enabled_envs[:size].sum()))
            marker = f"{finished}/{size} games"
        elif phase == "fit":
            if model._fit_epoch >= model.total_fit_passes:
                marker = "fit complete"
            else:
                stage = "tile-only" if model.fit_stage == "tile_only" else "joint"
                marker = f"{stage} pass {model._fit_epoch + 1}/{model.total_fit_passes}"
        else:
            marker = phase
        games = (
            f"scenario episodes {row['budget_progress']:,}"
            if pre_stage(self.cfg)
            else f"normal games total {row['training_games']:,}"
        )
        if row["target_games"] is not None:
            games += f"/{row['target_games']:,}"
        step = optimizer_step(model)
        parts = [
            f"tile stage {self.model.tile_skills.phase} "
            f"({self.model.tile_skills.progress:,} episodes)"
            if self.model.tile_skills.active
            else f"{selected_stage(self.cfg)} stage complete"
            if pre_stage(self.cfg)
            else "normal training",
            f"cohort {cohort} {self.cohort_text()}",
            CollectionProgress.text(row["collection_progress"], getattr(model, "cohort_games", 0)),
            marker,
            games,
            f"transitions total {row['training_steps']:,}",
            "collect "
            + seconds_text(row["last_collection_seconds"])
            + " fit "
            + seconds_text(row["last_optimization_seconds"]),
            f"optimizer step {step:,}",
        ]
        hardware = row["hardware"]
        usage = [
            f"{name} {hardware[key]:.0f}%"
            for name, key in (("GPU", "gpu_percent"), ("CPU", "system_cpu_percent"))
            if isinstance(hardware.get(key), (int, float)) and math.isfinite(hardware[key])
        ]
        if usage:
            parts.append(" ".join(usage))
        change = (
            phase,
            cohort,
            marker,
            row["training_games"],
            row["training_steps"],
            step,
            tuple(row["current_cohort"].values()),
            row["collection_progress"]["active_games"],
        )
        return " | ".join(parts), change

    def cohort_phase(self, previous, phase):
        """Print cohort boundaries: collection start/completion; fitting is reported at its end."""
        from pvz_rl.learning.cohort import CohortPhase

        self.progress.phase(Phase.COLLECTING if phase == CohortPhase.COLLECT else Phase.UPDATING)
        self.observe_collection_progress()
        if phase == CohortPhase.COLLECT:
            self.log_progress(force=True)
        elif previous == CohortPhase.COLLECT:
            model = self.model
            seconds = model._phase_times["collect"]
            transitions = model.num_timesteps - getattr(model, "cohort_start_steps", 0)
            self.progress.emit(
                f"Cohort {model._n_updates + 1} collected: {getattr(model, 'cohort_games', 0)} "
                f"games, {transitions:,} transitions in {seconds:.1f}s "
                f"({transitions / max(seconds, 1e-9):,.0f}/s); {self.cohort_text()}",
                force=True,
            )
        else:
            self.log_progress(force=True)

    def capture_update(self):
        if self.model._n_updates <= self.last_updates:
            return
        row = self.snapshot()
        values = self.model.logger.name_to_value
        keys = (
            "peashooter_cross_entropy_loss",
            "peashooter_preferred_column_accuracy",
            "total_loss",
            "primary_loss",
            "threat_loss",
            "effect_loss",
            "threat_depth_brier",
            "effect_brier",
            "threat_depth_accuracy",
            "effect_accuracy",
            "threat_exact_depths",
            "threat_time_brier",
            "threat_time_accuracy",
            "threat_loss_brier",
            "threat_loss_accuracy",
            "changed_effect_accuracy",
            "unchanged_effect_accuracy",
            "effect_change_precision",
            "effect_change_recall",
            "effect_change_brier",
            "effect_conditional_accuracy",
            "effect_labeled_terms",
            "effect_eligible_decisions",
            "threat_eligible_decisions",
            "q_loss",
            "tile_ranking_loss",
            "bomb_cross_entropy_loss",
            "bomb_softmax_temperature",
            "attacker_cross_entropy_loss",
            "attacker_softmax_temperature",
            "bomb_hinge_loss",
            "bomb_distance_loss",
            "selected_action_error",
            "species_consistency_loss",
            "species_consistency_mae",
            "species_consistency_saturation_fraction",
            "species_selection_evaluation_gap",
            "placement_gap",
            "q_grad_norm",
            "fit_batch_groups",
            "q_optimizer_steps",
            "tile_only_optimizer_steps",
            "joint_optimizer_steps",
            "species_exploration_coins",
            "tile_exploration_coins",
            "exploratory_changes",
            "plant_exploration_checks",
            "wait_exploration_checks",
            "plant_exploration_commitments",
            "wait_exploration_commitments",
            "plant_exploration_epsilon",
            "plant_wait_check_seconds",
            "requested_plant_wait_check_seconds",
            "learning_rate",
            "n_updates",
        )
        keys += tuple(
            f"{metric}_{action}"
            for action in ("wait", "plant", "dig")
            for metric in ("action_count", "value_target_error", "placement_gap")
        )
        metrics = {}
        for key in keys:
            value = values.get(f"train/{key}")
            metrics[key] = (
                float(value) if value is not None and math.isfinite(float(value)) else None
            )
        row.update(optimization=metrics, completed_updates=self.model._n_updates)
        if is_placement_skill(row.get("active_skill")):
            if (
                metrics["bomb_cross_entropy_loss"] is None
                and metrics["attacker_cross_entropy_loss"] is None
            ):
                metrics["tile_ranking_loss"] = metrics["q_loss"]
            metrics["q_loss"] = None
        append_jsonl(self.metrics_stream, row)
        self.last_updates = self.model._n_updates
        self.last_optimizer_metrics = dict(values)
        self.model.logger.dump(step=self.model.num_timesteps)

    def refresh_report(self):
        if not self.settings["visualization"]["enabled"]:
            return
        from pvz_rl.presentation.visualization import build_run_report

        started = perf_counter()
        previous_activity = self.hardware_activity
        try:
            self.hardware_context("reporting")
            for stream in (self.stream, self.metrics_stream):
                if stream and not stream.closed:
                    stream.flush()
            report = build_run_report(self.output, self.cfg)
            self.progress.emit(f"Report refreshed: {report}", force=True, key="report")
        except Exception as exc:
            write_json(
                self.output / "visualizations" / "status.json",
                {
                    "state": "failed",
                    "error": repr(exc),
                },
            )
            self.progress.emit(f"Report refresh failed: {exc}", force=True, key="report")
        finally:
            self.report_seconds += perf_counter() - started
            self.hardware_context(previous_activity)

    def _on_training_start(self):
        self.initial_task_counts = (
            {}
            if getattr(self.model, "resumed_cohort", False)
            else copy.deepcopy(getattr(self.model, "training_task_counts", {}))
        )
        self.initial_steps = self.model.num_timesteps
        self.initial_games = getattr(self.model, "training_games", 0)
        progress = progress_value(self.cfg, self.model)
        self.initial_budget_progress = progress
        self.restore_schedule()
        self.training_env.env_method("set_progress", progress)
        self.stream = (self.output / "training-episodes.jsonl").open("a", encoding="utf-8")
        self.metrics_stream = (self.output / "training-metrics.jsonl").open("a", encoding="utf-8")
        self.last_updates = self.model._n_updates
        if teaching_enabled(self.cfg) and self.family == "preset" and not pre_stage(self.cfg):
            self.curriculum = CurriculumState(
                **getattr(self.model, "curriculum_state", initial_state(self.cfg).to_dict())
            )
            self.training_env.env_method("set_curriculum_stage", self.curriculum.stage)
            self.sync_curriculum()
        self.progress.phase(Phase.COLLECTING)
        self.log_progress(force=True)

    def _on_step(self):
        self.observe_collection_progress()
        completed = 0
        step = self.locals.get("step_batch")
        if step is not None:
            self.simulation_ticks += step.ticks
            self.instant_actions += step.instant_actions
        for info in self.locals.get("episode_infos", self.locals["infos"]):
            if step is None:
                self.simulation_ticks += info["ticks_advanced"]
                self.instant_actions += int(info["ticks_advanced"] == 0)
            if "episode_metrics" in info:
                skills = self.model.tile_skills
                if skills.active:
                    skills.episodes[skills.phase] += 1
                    self.model.tile_skill_state = skills.to_dict()
                else:
                    self.model.training_games = getattr(self.model, "training_games", 0) + 1
                completed += 1
                if (
                    not skills.active
                    and self.curriculum
                    and (
                        self.cfg["curriculum"].get("residency") == "episode_start_stage"
                        or self.cfg["training"]["exploration"].get("decay_games", 0) > 0
                    )
                ):
                    self.curriculum.completed_episode(
                        info["episode_metrics"].get("episode_start_stage")
                    )
                self.recent.append(info["episode_metrics"])
                task = episode_task(info["episode_metrics"])
                self.recent_by_task.setdefault(
                    task, deque(maxlen=self.settings["logging"]["rolling_window"])
                ).append(info["episode_metrics"])
                report = {
                    **info["episode_metrics"],
                    "policy": self.condition,
                    "learner_seed": self.learner_seed,
                    "training_steps": self.num_timesteps,
                    "training_games": self.model.training_games,
                    "tile_scenarios": self.model.tile_skills.to_dict(),
                }
                if info["episode_metrics"].get("time_reward_pending"):
                    self.model._pending_episodes[info["episode_slot"]]["report"] = report
                else:
                    append_jsonl(self.stream, report)
        progress = progress_value(self.cfg, self.model)
        if hasattr(self, "progress"):
            self.log_progress()
        if completed and uses_games(self.cfg):
            # Finished workers pause. Update progress for the next cohort reset.
            self.training_env.env_method("set_progress", progress)
        weights = difficulty_weights(
            self.cfg,
            progress,
            budget_target(self.cfg),
            self.cfg["conditions"][self.condition]["curriculum"],
        )
        if self.family == "preset" and not self.curriculum and weights != self.last_weights:
            self.progress.emit(
                f"Curriculum for next episode resets: easy/standard/hard = {weights}; "
                "continuing the same policy and optimizer",
                force=True,
            )
            self.last_weights = list(weights)
        # Optimization metrics are finalized after the selected fitting phase.
        return True

    def finalize_cohort_rewards(self, pending):
        """Publish finalized rewards once, even when replaying an older recovery ZIP."""
        self.stream.flush()
        path = self.output / "training-episodes.jsonl"
        seen = set()
        if path.exists():
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        seen.add(json.loads(line).get("record_id"))
        for item in pending.values():
            report = item.get("report")
            if report is not None and item["record_id"] not in seen:
                append_jsonl(
                    self.stream, {**report, **item["metrics"], "record_id": item["record_id"]}
                )
        self.stream.flush()

    def _on_rollout_start(self):
        # This hook runs after the preceding Q update. A step callback would
        # save pre-update weights while labeling them with the newly collected steps.
        self.capture_update()
        if self.pending_validation in ("evaluation", "stage_evaluation"):
            self.validate()
            self.require_validation_complete()
        elif self.pending_validation == "curriculum":
            self.probe_curriculum()
            self.require_validation_complete()
        if self.model.tile_skills.active:
            self.check_deadline()
            self.probe_tile_skills()
            self.require_validation_complete()
            self.check_deadline()
        self.model.env.queue.deadline = self.operation_deadline()
        if pre_stage(self.cfg):
            self.check_stage_complete()
            if self.target is not None and progress_value(self.cfg, self.model) >= self.target:
                raise TrainingGamesComplete()
        if self.model.tile_skills.active:
            skills = self.model.tile_skills
            state = exploration_state(self.cfg, skills.episodes[skills.phase])
            apply_exploration_state(self.model, state)
            if is_placement_skill(skills.phase) and skills.stage_order is not None:
                from pvz_rl.learning.exploration import set_exploration_rate

                set_exploration_rate(self.model, 0.0, plant_rate=0.0)
            self.model.plant_exploration_rate = 0.0
            self.hardware_context("training")
            self.progress.phase(Phase.COLLECTING)
            return
        self.finish_stage_validation()
        self.check_stage_complete()
        if (
            uses_games(self.cfg)
            and self.target is not None
            and progress_value(self.cfg, self.model) >= self.target
        ):
            raise TrainingGamesComplete()
        self.check_deadline()
        if not self.stage_validation and progress_value(self.cfg, self.model) >= self.next_eval:
            nominal = self.next_eval
            while self.next_eval <= progress_value(self.cfg, self.model):
                self.next_eval += evaluation_interval(self.cfg)
            self.validate(nominal_games=nominal)
            self.require_validation_complete()
        self.check_deadline()
        self.probe_curriculum()
        self.require_validation_complete()
        self.finish_stage_validation()
        self.check_stage_complete()
        self.check_deadline()
        stage_games = (
            self.curriculum.completed_stage_games
            if self.curriculum
            else getattr(self.model, "training_games", 0)
        )
        state = exploration_state(self.cfg, stage_games)
        apply_exploration_state(self.model, state)
        self.hardware_context("training")
        self.progress.phase(Phase.COLLECTING)

    @property
    def active_stage(self):
        if self.model.tile_skills.active:
            skills = self.model.tile_skills
            return "pre" if skills.stage_order is None else skills.phase
        if pre_stage(self.cfg):
            return selected_stage(self.cfg)
        return self.curriculum.name if self.curriculum else "fixed"

    @property
    def stage_mastered(self):
        if pre_stage(self.cfg):
            return not self.model.tile_skills.active
        return bool(self.curriculum and self.curriculum.mastered)

    @property
    def stage_validation(self):
        return validation_after_stage(self.cfg, self.family)

    def restore_schedule(self):
        interval = evaluation_interval(self.cfg)
        progress = progress_value(self.cfg, self.model)
        schedule = getattr(self.model, "research_schedule", {})
        self.next_eval = schedule.get("next_eval", ((progress // interval) + 1) * interval)
        self.last_eval = schedule.get("last_eval", -1)
        self.pending_stage_validation = copy.deepcopy(schedule.get("pending_stage_validation"))
        self.validation_pending = self.pending_stage_validation is not None
        self.validation_trigger = ValidationTrigger(self.cfg, schedule.get("validation_trigger"))
        self.validation_panels = copy.deepcopy(schedule.get("validation_panels", {}))
        self.validation_pending |= self.pending_validation is not None

    @property
    def pending_validation(self):
        """Random panels remain pending even for recipes without cohort gating."""
        return self.validation_trigger.pending or next(iter(self.validation_panels), None)

    def request_validation(self, kind):
        skills = self.model.tile_skills
        domain = skills.validation_domain if kind == "tile" else "normal"
        progress = skills.progress if kind == "tile" else self.model.training_games
        cohort = self.model.cohort_metrics.get("validation_cohort", {})
        current = cohort.get("training_steps") == self.model.num_timesteps and cohort.get(
            "optimizer_step"
        ) == optimizer_step(self.model)
        if self.validation_trigger.enabled and not current:
            if self.validation_trigger.pending:
                raise ValueError("Pending validation weights or cohort changed")
            return False
        allowed, checked = self.validation_trigger.request(kind, domain, progress, cohort)
        if allowed:
            if self.validation_panels and kind not in self.validation_panels:
                raise ValueError("Another validation panel remains pending")
            if kind not in self.validation_panels:
                self.validation_panels[kind] = (
                    curriculum_probe_seeds(self.cfg)
                    if kind in ("tile", "curriculum")
                    else random_validation_seeds(self.cfg, limit=self.validation_limit)
                )
            self.validation_pending = True
        if checked:
            decision = self.validation_trigger.state["decision"]
            with (self.output / "validation-checks.jsonl").open("a", encoding="utf-8") as stream:
                append_jsonl(
                    stream,
                    dict(
                        **decision,
                        kind=kind,
                        training_steps=self.model.num_timesteps,
                        optimizer_step=optimizer_step(self.model),
                    ),
                )
            self.progress.emit(
                f"Validation check: {cohort.get('wins', 0)}/{cohort.get('games', 0)} "
                f"cohort successes; {'launch' if allowed else 'skip'}; "
                f"next check after {progress + self.validation_trigger.interval_for(domain):,} "
                f"{domain} episodes",
                force=True,
            )
        if allowed or checked:
            self.save_checkpoint("latest.zip")
            write_json(self.output / "status.json", self.snapshot())
        if checked and not allowed:
            self.refresh_report()
        return allowed

    def complete_validation(self, kind):
        self.validation_trigger.complete(kind)
        self.validation_panels.pop(kind, None)
        self.validation_pending = False

    def require_validation_complete(self):
        if self.pending_validation:
            self.budget_stopped = True
            raise TrainingDeadline()

    def finish_stage_validation(self):
        if self.stage_validation and self.pending_stage_validation:
            if self.validate() is False:
                # Do not update past an unevaluated mastery checkpoint. Finalization
                # can retry with its reserved time, or resume can retry these weights.
                self.budget_stopped = True
                raise TrainingDeadline()

    def check_stage_complete(self):
        if self.stage_mastered:
            raise TrainingStageComplete()

    def check_deadline(self):
        estimated = self.timings.last_window_seconds or (
            self.timings.last_collection_seconds or 0
        ) + (self.timings.last_optimization_seconds or 0)
        if (
            (self.deadline is not None and perf_counter() >= self.deadline)
            or self.wall_budget is not None
            and not self.wall_budget.can_collect(estimated)
        ):
            self.budget_stopped = True
            raise TrainingDeadline()

    def operation_deadline(self, final=False):
        wall = (
            None
            if self.wall_budget is None
            else (self.wall_budget.deadline if final else self.wall_budget.learning_deadline)
        )
        limits = [value for value in (self.deadline, wall) if value is not None]
        return min(limits) if limits else None

    def save_checkpoint(self, name):
        skills = self.model.tile_skills
        skills.generation_failures = max(
            skills.generation_failures,
            self.model.env.queue.generation_failures,
        )
        self.model.tile_skill_state = skills.to_dict()
        self.task_counts()
        if self.wall_budget:
            self.model.wall_budget_state = self.wall_budget.state()
        self.model.research_schedule = {
            "next_eval": self.next_eval,
            "last_eval": self.last_eval,
            "pending_stage_validation": copy.deepcopy(self.pending_stage_validation),
            "validation_trigger": copy.deepcopy(self.validation_trigger.state),
            "validation_panels": copy.deepcopy(self.validation_panels),
        }
        self.sync_curriculum()
        self.model.save(self.output / name)
        self.progress.emit(
            f"Checkpoint saved: {name}; transitions {self.model.num_timesteps:,}; "
            f"optimizer step {optimizer_step(self.model):,}; {self.cohort_text()}",
            force=True,
        )

    def cached_evaluation(self, seeds, levels, family, destination, split, final=False):
        if self.cache_steps != self.model.num_timesteps:
            self.evaluation_cache.clear()
            self.cache_steps = self.model.num_timesteps
        keys = [(family, level, seed) for level in levels for seed in seeds]
        if all(key in self.evaluation_cache for key in keys):
            return [self.evaluation_cache[key] for key in keys]
        attempt = destination
        retry = 1
        while attempt.exists():
            attempt = destination.with_name(destination.name + f"-retry-{retry}")
            retry += 1
        rows = evaluate(
            self.cfg,
            policy=self.model,
            condition=self.condition,
            learner_seed=self.learner_seed,
            seeds=seeds,
            levels=levels,
            family=family,
            split=split,
            output=attempt,
            training_steps=self.model.num_timesteps,
            progress=self.progress,
            deadline=self.operation_deadline(final),
        )
        self.evaluation_cache.update(
            {(r["family"], r["level"], r["scenario_seed"]): r for r in rows}
        )
        return rows

    def sync_curriculum(self):
        if self.curriculum:
            self.model.curriculum_state = self.curriculum.to_dict()
            write_json(self.output / "curriculum.json", self.model.curriculum_state)

    def evaluation_event(self, operation, **kwargs):
        previous_activity = self.hardware_activity
        self.hardware_context("validation")
        started = perf_counter()
        try:
            # Validation owns a separate environment. Retire idle collection
            # scratch and graph buffers before it asks CuPy for another panel.
            self.model.env.queue.release_scratch()
            for network in (self.model.policy, getattr(self.model, "_teacher", None)):
                if network is not None:
                    for encoder in network.encoders:
                        encoder.release_fitting()
                        encoder.release_collection()
            from pvz_rl.learning.performance import phase_memory

            phase_memory(self.model.device, empty_cache=True)
            self.model.env.cp.get_default_memory_pool().free_all_blocks()
            return operation(**kwargs)
        finally:
            self.eval_seconds += perf_counter() - started
            self.refresh_report()
            self.hardware_context(previous_activity)

    def probe_curriculum(self, *, final=False):
        if (
            self.model.tile_skills.active
            or pre_stage(self.cfg)
            or not self.curriculum
            or not self.curriculum.due(progress_value(self.cfg, self.model), self.cfg)
        ):
            return
        if not self.request_validation("curriculum"):
            return
        return self.evaluation_event(self._probe_curriculum, final=final)

    def probe_tile_skills(self, *, final=False):
        skills = self.model.tile_skills
        if not skills.due(self.cfg) and self.pending_validation != "tile":
            return
        if not self.request_validation("tile"):
            return
        from pvz_rl.evaluation.tile_scenarios import validate_tile_skills

        def probe():
            names = ", ".join(skills.validation_skills)
            self.progress.phase(Phase.VALIDATING, f"Tile stage validation: {names}")
            try:
                results = validate_tile_skills(
                    self.cfg,
                    self.model,
                    self.validation_panels["tile"],
                    deadline=self.operation_deadline(final),
                    progress=self.progress,
                    skills=skills.validation_skills,
                )
            except BudgetExpired:
                self.progress.emit(
                    "Tile validation incomplete; mastery remains pending", force=True
                )
                return
            count = self.cfg["curriculum"]["probe_cases"]
            if any(len(rows) != count for rows in results.values()):
                raise ValueError("Incomplete tile-stage validation cannot certify mastery")
            scores = {skill: sum(row["win"] for row in rows) for skill, rows in results.items()}
            previous = skills.phase
            with atomic_transition():
                skills.observe(scores, self.cfg)
                self.complete_validation("tile")
                self.model.tile_skill_state = skills.to_dict()
                with (self.output / "tile-skill-probes.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    append_jsonl(
                        stream,
                        dict(
                            episodes=skills.total,
                            scores=scores,
                            previous=previous,
                            next=skills.phase,
                            stage=previous if skills.stage_order is not None else "pre",
                            stage_episodes=skills.episodes[previous],
                            cases=results,
                            optimizer_step=optimizer_step(self.model),
                            training_steps=self.model.num_timesteps,
                        ),
                    )
                self.save_checkpoint("latest.zip")
            self.progress.emit(
                f"Tile validation: {scores}; next phase {skills.phase}",
                force=True,
            )
            if not skills.active:
                self.recent.clear()
                self.recent_by_task.clear()
                self.progress.emit(
                    f"{selected_stage(self.cfg)} stage mastered"
                    if pre_stage(self.cfg)
                    else "Tile stages mastered; normal game budget and schedules start at zero",
                    force=True,
                )
            elif skills.phase != previous:
                self.recent.clear()
                self.recent_by_task.clear()
                self.progress.emit(
                    f"{previous} stage mastered; starting {skills.phase} with its own episode clock",
                    force=True,
                )
            write_json(self.output / "status.json", self.snapshot())

        return self.evaluation_event(probe)

    def _probe_curriculum(self, *, final=False):
        progress = progress_value(self.cfg, self.model)
        if not self.curriculum or not self.curriculum.due(progress, self.cfg):
            return
        previous = self.curriculum.name
        self.progress.phase(Phase.VALIDATING, f"Curriculum probe: {previous}")
        wins = {}
        for task in self.curriculum.requirements(self.cfg):
            try:
                rows = self.cached_evaluation(
                    self.validation_panels["curriculum"],
                    [task],
                    "preset",
                    self.output / "curriculum-probes" / str(self.model.num_timesteps) / task,
                    "curriculum_validation",
                    final=final,
                )
            except BudgetExpired:
                self.progress.emit("Curriculum probe pending: time allowance exhausted", force=True)
                return
            if len(rows) != self.cfg["curriculum"]["probe_cases"]:
                raise ValueError("Incomplete curriculum probe cannot certify mastery")
            wins[task] = sum(bool(row["win"]) for row in rows)
        with atomic_transition():
            advanced = self.curriculum.observe(
                wins, progress, self.cfg, advance=selected_stage(self.cfg) is None
            )
            seeds = self.validation_panels["curriculum"]
            self.complete_validation("curriculum")
            self.sync_curriculum()
            if advanced:
                self.training_env.env_method("set_curriculum_stage", self.curriculum.stage)
            if self.stage_validation and (advanced or self.curriculum.mastered):
                self.pending_stage_validation = {
                    "stage": previous,
                    "training_steps": self.model.num_timesteps,
                    "training_games": getattr(self.model, "training_games", 0),
                }
                self.validation_pending = True
            with (self.output / "curriculum-probes.jsonl").open("a", encoding="utf-8") as stream:
                append_jsonl(
                    stream,
                    {
                        "training_steps": self.model.num_timesteps,
                        "training_games": getattr(self.model, "training_games", 0),
                        "stage": previous,
                        "wins": wins,
                        "seeds": seeds,
                        "advanced": advanced,
                        "stage_mastered": self.curriculum.mastered,
                        "state": self.curriculum.to_dict(),
                    },
                )
            self.progress.emit(
                f"Curriculum {previous}: {wins}; next-reset stage {self.curriculum.name}; "
                f"consecutive passes {self.curriculum.consecutive_passes}; "
                f"stage mastered {self.curriculum.mastered}",
                force=True,
            )
            if selected_stage(self.cfg) or self.stage_validation:
                self.save_checkpoint("latest.zip")
                self.progress.emit("Saved stage progress to latest.zip", force=True)
            write_json(self.output / "status.json", self.snapshot())

    def _on_rollout_end(self):
        self.timings.record_window(self.model.cohort_metrics)
        self.hardware_context()
        self.sync_curriculum()
        self.progress.phase(Phase.UPDATING)
        cohort = self.model.cohort_metrics
        self.progress.emit(
            f"Cohort {self.model._n_updates} fitted: {cohort.get('q_optimizer_steps', 0)} passes "
            f"({cohort.get('tile_only_optimizer_steps', 0)} tile-only + "
            f"{cohort.get('joint_optimizer_steps', 0)} joint) "
            f"in {cohort['fit_seconds']:.1f}s after {cohort['collection_seconds']:.1f}s "
            f"collection; {cohort['transitions_per_second']:,.0f} transitions/s; "
            f"optimizer step {optimizer_step(self.model):,}; {self.cohort_text()}",
            force=True,
        )
        write_json(self.output / "status.json", self.snapshot())

    def validate(self, *, nominal_games=None, final=False):
        if self.stage_validation and not self.pending_stage_validation:
            return False
        kind = "stage_evaluation" if self.pending_stage_validation else "evaluation"
        if not self.request_validation(kind):
            return False
        return self.evaluation_event(self._validate, nominal_games=nominal_games, final=final)

    def _validate(self, *, nominal_games=None, final=False):
        if self.stage_validation and not self.pending_stage_validation:
            return False
        milestone = self.pending_stage_validation
        self.progress.phase(
            Phase.VALIDATING,
            f"Validation at {progress_value(self.cfg, self.model):,} {'games' if uses_games(self.cfg) else 'decisions'}"
            + (f" after {milestone['stage']} mastery" if milestone else ""),
        )
        write_json(self.output / "status.json", self.snapshot())
        try:
            rows = self.cached_evaluation(
                self.validation_panels["stage_evaluation" if milestone else "evaluation"],
                ["easy"] if self.family in ("diagnostic",) else self.cfg["evaluation"]["levels"],
                self.family,
                self.output / "validation" / str(self.model.num_timesteps),
                "validation",
                final=final,
            )
        except BudgetExpired:
            self.validation_pending = True
            self.progress.emit(
                "Validation pending; partial results cannot select best.zip", force=True
            )
            return False
        with atomic_transition():
            scores = summarize(rows)
            score = sum(r["win_rate"] for r in scores.values()) / len(scores)
            self.complete_validation("stage_evaluation" if milestone else "evaluation")
            self.pending_stage_validation = None
            self.last_eval = self.model.num_timesteps
            self.progress.emit(
                "Validation "
                + ", ".join(f"{key}: {value['win_rate']:.1%}" for key, value in scores.items())
                + f"; shared-policy mean {score:.1%}",
                force=True,
            )
            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint("best.zip")
                write_json(
                    self.output / "best.json",
                    {
                        "steps": self.model.num_timesteps,
                        "games": getattr(self.model, "training_games", 0),
                        "macro_win_rate": score,
                        "checkpoint_hash": file_hash(self.output / "best.zip"),
                        "stage_success": milestone,
                    },
                )
                self.progress.emit(f"Saved shared best.zip at {score:.1%}", force=True)
            self.save_checkpoint("latest.zip")
            self.progress.emit("Saved latest.zip", force=True)
            self.last_eval = self.model.num_timesteps
            with (self.output / "learning-curve.jsonl").open("a", encoding="utf-8") as stream:
                append_jsonl(
                    stream,
                    {
                        "training_steps": self.model.num_timesteps,
                        "training_games": getattr(self.model, "training_games", 0),
                        "budget_unit": "games" if uses_games(self.cfg) else "decisions",
                        "nominal_games": nominal_games,
                        "final": final,
                        "stage_success": milestone,
                        "wall_seconds": perf_counter() - self.started,
                        "macro_win_rate": score,
                        "levels": scores,
                    },
                )
            write_json(self.output / "status.json", self.snapshot())

    def _on_training_end(self):
        self.capture_update()
        pending = self.pending_validation
        if pending == "tile":
            self.probe_tile_skills(final=True)
        elif pending == "curriculum":
            self.probe_curriculum(final=True)
        elif pending:
            self.validate(final=True)
        if self.pending_validation:
            return
        if pre_stage(self.cfg):
            if not self.budget_stopped:
                self.probe_tile_skills(final=True)
            return
        if self.model.tile_skills.active:
            if not self.budget_stopped:
                self.probe_tile_skills(final=True)
            return
        if self.stage_validation:
            if not self.budget_stopped and not self.pending_stage_validation:
                self.probe_curriculum(final=True)
            if self.pending_stage_validation:
                self.validate(final=True)
            else:
                self.progress.emit(
                    "No additional normal-game validation without a new stage success; "
                    "checkpoints and reports are still saved",
                    force=True,
                )
            return
        if self.last_eval != self.model.num_timesteps and (
            not self.budget_stopped
            or self.wall_budget is not None
            and self.wall_budget.limit is not None
        ):
            self.validate(final=True)
        if not self.budget_stopped and (
            selected_stage(self.cfg) or "curriculum" in self.cfg["splits"]
        ):
            self.probe_curriculum(final=True)
        elif (
            not self.budget_stopped
            and self.cfg["curriculum"].get("residency") != "episode_start_stage"
        ):
            self.probe_curriculum()

    def close(self):
        if self.hardware:
            self.hardware.close()
        if self.stream:
            self.stream.close()
        if self.metrics_stream:
            self.metrics_stream.close()
        if self.owns_progress:
            self.progress.close()


def initial_weights(checkpoint, cfg):
    """Transfer validated weights with fresh optimizer, counters and curriculum."""
    from pvz_rl.learning.checkpoints import compatible_config, inspect_checkpoint

    require_supported_policy(cfg)
    from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.suffix:
        checkpoint = checkpoint.with_suffix(".zip")
    saved = inspect_checkpoint(checkpoint, weights_only=True)
    source_cfg = saved["config"]
    compatible_config(
        source_cfg,
        cfg,
        weights_only=True,
    )
    verify_engine(cfg)
    if saved["initialization_type"] == "demonstration":
        weights = saved["payload"]["model"]
    else:
        import io
        from zipfile import ZipFile

        with ZipFile(checkpoint) as archive:
            weights = torch.load(
                io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True
            )
    if not isinstance(weights, dict) or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in weights.values()
    ):
        raise ValueError("Checkpoint contains invalid or non-finite model weights")
    with torch.random.fork_rng(devices=[]):
        model = TwoStageTransformerPolicy(cfg)
    conversion = "none"
    if (
        source_cfg["policy"]["kind"] == "shared_fork_forecast_q_v3"
        and cfg["policy"]["kind"] == "shared_public_forecast_q_v4"
    ):
        from pvz_rl.envs.encoding import BOARD_GLOBAL_FIELDS

        weights = dict(weights)
        key = "board_encoder.global_projection.weight"
        weights[key] = weights[key][:, : len(BOARD_GLOBAL_FIELDS)].clone()
        initialized = model.state_dict()
        for key in ("effect_head.2.weight", "effect_head.2.bias"):
            weights[key] = initialized[key]
        conversion = "public_inputs_absolute_outcomes_v1"
    model.load_state_dict(weights, strict=True)
    weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return weights, {
        "mode": "weights_only",
        "type": saved["initialization_type"],
        "conversion": conversion,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": saved.get("snapshot_sha256") or file_hash(checkpoint),
        "source_learner_seed": saved["learner_seed"],
        **saved.get("source_counters", {"steps": 0, "games": 0, "updates": 0}),
        "source_structural_signature": transfer_protocol(source_cfg),
        "parameter_changes": parameter_changes(source_cfg, cfg),
    }


def load_policy(checkpoint, device="cpu", *, for_resume=False):
    from gymnasium import spaces
    from pvz_game import Rules

    from pvz_rl.envs.actions import ActionSchema
    from pvz_rl.envs.encoding import ObservationEncoder
    from pvz_rl.learning.checkpoints import inspect_checkpoint, model_class
    from pvz_rl.policy.two_stage_policy import TwoStageQPolicy

    data = inspect_checkpoint(checkpoint, weights_only=not for_resume)
    from pvz_rl.learning.checkpoints import execution_config
    from pvz_rl.learning.training_requirements import current_model_config, inference_config
    from pvz_rl.provenance import current_engine_config

    cfg = data["config"] if for_resume else execution_config(inference_config(data["config"]))
    verify_engine(cfg)
    require_supported_policy(cfg, data["condition"])
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if (
        data["initialization_type"] in ("demonstration", "weights")
        or not current_model_config(data["config"])
        or not current_engine_config(data["config"])
        or (not for_resume and data.get("fit_schedule_protocol") != FIT_SCHEDULE_PROTOCOL)
    ):
        model = model_class(cfg)(
            TwoStageQPolicy,
            device=device,
            _init_setup_model=False,
            policy_kwargs={"features_extractor_kwargs": {"layout_cfg": cfg}},
        )
        model.observation_space = ObservationEncoder(cfg, Rules()).space
        model.action_space = spaces.Discrete(ActionSchema.size)
        model.n_envs = cfg["training"]["n_envs"]
        model._setup_model()
        if data["initialization_type"] == "demonstration":
            weights = data.pop("payload")["model"]
        else:
            weights, _ = initial_weights(checkpoint, cfg)
        model.policy.load_state_dict(weights, strict=True)
        if data["initialization_type"] == "weights":
            counters = data["source_counters"]
            model.num_timesteps = counters["steps"]
            model.training_games = counters["games"]
            model._n_updates = counters["updates"]
    else:
        model = model_class(cfg).load(checkpoint, device=device)
        saved_reward = copy.deepcopy(model.cfg["reward"])
        current_reward = copy.deepcopy(cfg["reward"])
        if for_resume:
            # Passivity is an explicit cohort-scoped reward update. It is
            # refreshed after loading and takes effect at the next cohort;
            # every other reward field remains part of the strict identity.
            saved_reward.pop("passivity", None)
            current_reward.pop("passivity", None)
        if transfer_protocol(model.cfg) != transfer_protocol(cfg) or saved_reward != current_reward:
            raise ValueError("Checkpoint contents disagree with saved model configuration")
    if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model weights")
    model.policy.features_extractor.cfg = cfg
    model.policy_kwargs["features_extractor_kwargs"]["layout_cfg"] = cfg
    configure_exploration(model, cfg)
    return model, data


def train(
    cfg,
    condition,
    learner_seed,
    output,
    *,
    validation_limit=None,
    family="preset",
    resume=None,
    init_from=None,
    deadline=None,
):
    started = perf_counter()
    cfg = copy.deepcopy(cfg)
    if family == "diagnostic":
        cfg["training"].pop("peashooter_rules", None)
        cfg.setdefault("diagnostics", {})["reach_tracking"] = False
    validate_config(cfg)
    output = Path(output)
    require_cuda_training(cfg, condition, runtime=False)
    if resume and init_from:
        raise ValueError("Use either resume or init_from, not both")
    if selected_stage(cfg) and family != "preset":
        raise ValueError("Stage training uses family=preset and the stage's configured task mix")
    if until_stage_complete(cfg) and deadline is not None:
        raise ValueError(
            "until_stage_complete cannot be combined with an explicit training deadline"
        )
    initialized_model, initialization = None, None
    if init_from:
        initialized_model, initialization = initial_weights(init_from, cfg)
    if resume:
        from pvz_rl.learning.checkpoints import inspect_checkpoint, protocol_for
        from pvz_rl.learning.cuda_q import checkpoint_optimizer_protocol

        saved = inspect_checkpoint(resume)
        if learner_seed is None:
            learner_seed = saved["learner_seed"]
        if saved["initialization_type"] != "autonomous":
            raise ValueError(
                "Demonstration weights require --init-from; --resume needs an autonomous ZIP"
            )
        require_cuda_training(saved["config"], saved["condition"], runtime=False)
        if saved.get("exploration_protocol") != EXPLORATION_PROTOCOL:
            raise ValueError("Checkpoint uses the retired exploration schedule; start a fresh run")
        if (
            checkpoint_optimizer_protocol(resume)
            != protocol_for(cfg["policy"]["kind"])["optimizer"]
        ):
            raise ValueError("Optimizer protocol changed; use fresh training or --init-from")
        if (
            saved["condition"] != condition
            or resume_protocol(saved["config"], condition) != resume_protocol(cfg, condition)
            or saved["family"] != family
        ):
            raise ValueError(
                "Resume requires identical condition, family and configuration; start a fresh run for a changed profile"
            )
        if saved["learner_seed"] != learner_seed or saved["validation_limit"] != validation_limit:
            raise ValueError("Resume learner seed or validation limit differs")
    elif learner_seed is None:
        learner_seed = secrets.randbits(32)
    require_cuda_training(cfg, condition)
    resumed_model = (
        load_policy(resume, cfg["training"]["device"], for_resume=True)[0] if resume else None
    )
    if resumed_model is not None and hasattr(resumed_model, "refresh_passivity"):
        # Reward shaping is the only resume-scoped update. The restored buffer
        # keeps its saved cohort settings; the requested values activate when
        # the next cohort is created.
        resumed_model.refresh_passivity(cfg)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    # Complete Q rollouts are counted explicitly; no discarded partial optimization batch.
    game_budget = uses_games(cfg)
    unlimited = until_stage_complete(cfg)
    nominal_steps = None if game_budget else cfg["training"]["total_steps"]
    effective_steps = None if game_budget else nominal_steps
    in_place_resume = bool(resume and output.resolve() == Path(resume).resolve().parent)
    prior_time = 0.0
    if resume:
        previous_status = Path(resume).resolve().parent / "status.json"
        if previous_status.exists():
            prior_time = (
                json.loads(previous_status.read_text("utf-8")).get("time_budget") or {}
            ).get("elapsed_seconds", 0.0)
    output.mkdir(parents=True, exist_ok=in_place_resume)
    details = metadata(
        cfg,
        condition=condition,
        learner_seed=learner_seed,
        family=family,
        nominal_steps=nominal_steps,
        effective_steps=effective_steps,
        target_games=budget_target(cfg) if game_budget else None,
        until_stage_complete=unlimited,
        effective_limits=effective_limits(cfg),
        budget_unit="games" if game_budget else "decisions",
        validation_limit=validation_limit,
        resume=str(Path(resume).resolve()) if resume else None,
        initialization=initialization
        if initialization
        else saved.get("initialization")
        if resume
        else None,
        exploration_protocol=EXPLORATION_PROTOCOL,
        structural_signature=transfer_protocol(cfg),
        optimizer_protocol=cfg["training"]["method"],
        fit_schedule_protocol=FIT_SCHEDULE_PROTOCOL,
        model_family=cfg["policy"]["kind"],
        hardware_telemetry={
            "enabled": cfg["training"].get("performance", {}).get("telemetry", False),
            "sample_seconds": output_settings(cfg)["logging"]["hardware_sample_seconds"],
            "samples": "hardware-metrics.jsonl",
            "gpu_scope": "device-wide; memory percent is controller activity, not VRAM capacity",
            "cpu_scope": "system and busiest logical core: 0..100%; process: one core = 100%",
        },
        optimizer_settings={
            "learning_rate": cfg["training"]["learning_rate"],
            "adam_epsilon": 1e-8,
            "batch_size": cfg["training"]["batch_size"],
            "normal_fitting": "joint",
            "passes_per_cohort": cfg["training"]["n_epochs"],
            "shared_encoder": True,
            "runtime_dtype": "bf16",
            "master_dtype": "fp32",
        },
        collection_method=cfg["training"]["method"],
    )
    write_json(output / "metadata.json", details)
    write_json(output / "config.json", cfg)
    write_json(output / "status.json", {"state": "starting"})
    env, model, callback = None, None, None
    settings = output_settings(cfg)
    progress = ProgressReporter.from_settings(output / "train.log", cfg)
    progress.route_warnings()
    budget_label = (
        "until stage mastery; no time or game ceiling"
        if unlimited
        else f"{budget_target(cfg) if game_budget else effective_steps:,} {'games' if game_budget else 'decisions'}"
    )
    progress.emit(
        f"Shared {condition} {cfg['policy']['kind']} policy; learner seed {learner_seed}; {cfg['training']['device']}; "
        f"{cohort_envs(cfg, 'peashooter')} Peashooter lanes, {cohort_envs(cfg)} normal lanes, "
        f"{cohort_envs(cfg, 'cherry_bomb')} Cherry Bomb lanes, "
        f"{cfg['training'].get('tile_scenarios', {}).get('attacker', {}).get('n_envs', 0)} attacker-pre lanes; simulator {simulator(cfg)}; "
        "one complete game per environment per cohort; "
        f"{budget_label}; "
        f"family {family}; output {output.resolve()}",
        force=True,
    )
    progress.emit(f"Data transport: {runtime_settings(cfg)}", force=True)
    preparation = cfg["training"].get("performance", {})
    progress.emit(
        f"Cherry preparation: {preparation.get('cherry_prepare_workers', 0)} requested CPU workers; "
        f"{preparation.get('cherry_oracle', 'cpu')} oracle; "
        f"CUDA warmup {preparation.get('cherry_cuda_warmup', False)}; "
        f"prefetch {preparation.get('cherry_prefetch', False)}; workers/scratch respect RAM/VRAM limits",
        force=True,
    )
    target_kind = (
        "successful-set placement supervision"
        if is_placement_skill(selected_stage(cfg))
        else (
            "n-step Q targets with hindsight threats and absolute action outcomes"
            if cfg["encoding"]["version"] == "entity_public_v4"
            else "n-step Q targets with hindsight threats and signed fork effects"
        )
    )
    progress.emit(
        f"Policy: {cfg.get('policy', {'kind': 'flat'})}; {target_kind}; "
        "joint normal cohorts; supervised preliminary cohorts; "
        f"{cfg['training']['n_epochs']} passes per cohort; "
        f"minibatch {cfg['training']['batch_size']}",
        force=True,
    )
    progress.emit(f"Reward settings: {cfg['reward']}", force=True)
    if cfg["training"].get("peashooter_rules"):
        progress.emit(
            "Peashooter: prioritized visible-lane oracle targets; column-decaying cross-entropy; "
            "greedy main actions, no factual followers",
            force=True,
        )
    progress.emit(
        f"Committed plant/tile exploration {cfg['training']['exploration']}; "
        "normal branches greedy; accepted planting can commit to another species; "
        "Rraw/Rtrain are completed-cohort means (~ provisional)",
        force=True,
    )
    if validation_after_stage(cfg, family):
        probe_key = "probe_interval_games" if game_budget else "probe_interval"
        progress.emit(
            f"Mastery checks every {cfg['curriculum'][probe_key]:,} "
            f"{'completed games' if game_budget else 'decisions'}; "
            "normal easy/standard/hard validation only after each stage passes",
            force=True,
        )
    if selected_stage(cfg):
        progress.emit(
            f"Single stage: {selected_stage(cfg)}; no automatic promotion; "
            + ("stop only at mastery" if unlimited else "stop at mastery or budget")
            + ", save final.zip for the next stage",
            force=True,
        )
    progress.emit(
        f"Agent cadence: {cfg['environment'].get('action_frequency_hz', 'derived')} Hz; "
        f"{cfg['environment']['decision_ticks']} simulator ticks per decision; "
        f"accounting discount {cfg['training']['gamma']} per simulation tick; "
        "TD uses the saved forecast half-life schedule",
        force=True,
    )
    training_complete = False
    retain_weights = False
    wall_budget = RunBudget(cfg, elapsed=prior_time, started=started)
    try:
        env = vector_env(cfg, condition, learner_seed, family)
        env.start_live_view(notify=lambda text: progress.emit(text, force=True))
        if resume:
            model = resumed_model
            model.configure_execution(cfg)
            wall_budget.prior_elapsed = max(
                wall_budget.prior_elapsed,
                getattr(model, "wall_budget_state", {}).get("elapsed_seconds", 0.0),
            )
            model.set_env(env)
            if compilation_requested(cfg["training"].get("performance", {})):
                model.policy.enable_compilation()
            model.tensorboard_log = str(output / "tensorboard")
            details["resume_steps"] = model.num_timesteps
            details["resume_games"] = getattr(model, "training_games", 0)
            write_json(output / "metadata.json", details)
            progress.emit(
                f"Resumed shared policy and optimizer at {progress_value(cfg, model):,} {'games' if game_budget else 'decisions'}",
                force=True,
            )
        else:
            model = build_model(cfg, condition, env, learner_seed, output / "tensorboard")
            model.tile_skill_state = initial_tile_state(cfg, family)
            if initialized_model is not None:
                model.policy.load_state_dict(initialized_model, strict=True)
                progress.emit(
                    f"Initialized weights from {initialization['checkpoint']}; fresh optimizers, counters and budget",
                    force=True,
                )
            if teaching_enabled(cfg) and family == "preset":
                model.curriculum_state = initial_state(cfg).to_dict()
        model.research_metadata = details
        if model.tile_skills.active:
            from pvz_rl.learning.budget import validation_interval_games

            if is_placement_skill(model.tile_skills.phase):
                from pvz_rl.learning.bomb_objective import BOMB_OBJECTIVE

                bomb = placement_settings(cfg, model.tile_skills.phase)
                objective = bomb.get("ranking_objective", "top1_hinge_v1")
                fitting = (
                    "set cross-entropy only, temperature "
                    f"{bomb_softmax_temperature(bomb, 0.0, skill=model.tile_skills.phase):g}"
                    " at 0% success to "
                    f"{bomb_softmax_temperature(bomb, 1.0, skill=model.tile_skills.phase):g}"
                    " at 100% (linear in this cohort's target accuracy)"
                    if objective == BOMB_OBJECTIVE
                    else (
                        f"ranking margin {bomb_ranking_margin(bomb)}, "
                        f"distance weight {bomb.get('distance_weight', 0)}, "
                        f"temperature {bomb.get('distance_temperature', 1)}"
                    )
                )
                progress.emit(
                    f"Placement fitting: {objective}; {fitting}; no search or exploration",
                    force=True,
                )
            requirements = {
                skill: (
                    required_validation_wins(cfg, skill)
                    if model.tile_skills.stage_order is not None
                    else cfg["curriculum"]["probe_cases"]
                )
                for skill in model.tile_skills.validation_skills
            }
            progress.emit(
                f"Tile stage: {model.tile_skills.phase}; validate "
                f"{', '.join(model.tile_skills.validation_skills)} every "
                f"{validation_interval_games(cfg, model.tile_skills.validation_domain):,} "
                "stage episodes at a fitted "
                "cohort boundary; require "
                + ", ".join(
                    f"{skill} {wins}/{cfg['curriculum']['probe_cases']}"
                    for skill, wins in requirements.items()
                )
                + "; "
                + (
                    "scenario episodes count toward the selected tile stage budget"
                    if pre_stage(cfg)
                    else "normal game budget remains separate"
                ),
                force=True,
            )
        remaining = (
            None
            if unlimited
            else (
                budget_target(cfg) - progress_value(cfg, model)
                if game_budget
                else effective_steps - model.num_timesteps
            )
        )
        model.trajectory_root = output
        callback = ResearchCallback(
            cfg,
            condition,
            learner_seed,
            output,
            validation_limit,
            family,
            progress=progress,
            deadline=deadline,
            wall_budget=wall_budget,
        )
        if resume:
            previous = Path(resume).resolve().parent
            if (previous / "best.json").exists():
                best = json.loads((previous / "best.json").read_text("utf-8"))
                if best["steps"] <= model.num_timesteps:
                    if not in_place_resume:
                        shutil.copy2(previous / "best.zip", output / "best.zip")
                        shutil.copy2(previous / "best.json", output / "best.json")
                    callback.best_score = best["macro_win_rate"]
        env.env_method("set_progress", progress_value(cfg, model))
        if teaching_enabled(cfg) and family == "preset":
            env.env_method(
                "set_curriculum_stage", getattr(model, "curriculum_state", {}).get("stage", 0)
            )
        if remaining is None or remaining > 0 or model.phase != "idle":
            try:
                model.learn(
                    # Game count is the stopping criterion. SB3 requires a numeric
                    # timesteps bound; this sentinel does not schedule learning.
                    2**63 - 1 if game_budget else max(0, remaining),
                    callback=callback,
                    reset_num_timesteps=not bool(resume),
                    tb_log_name=condition,
                )
            except TrainingGamesComplete:
                progress.emit("Game target reached; final Q update completed", force=True)
                callback._on_training_end()
            except TrainingStageComplete:
                progress.emit("Mastery reached; finalizing its checkpoint", force=True)
                callback._on_training_end()
            except TrainingDeadline:
                progress.emit(
                    "Time allowance reached at a completed Q update; finalizing available results",
                    force=True,
                )
                callback._on_training_end()
        else:
            # Recover an interrupted final evaluation without collecting extra data.
            if model.runtime_state:
                model._restore_runtime()
            callback.init_callback(model)
            callback.last_updates = model._n_updates
            callback.initial_steps = model.num_timesteps
            callback.initial_games = getattr(model, "training_games", 0)
            callback.initial_budget_progress = progress_value(cfg, model)
            callback.restore_schedule()
            if teaching_enabled(cfg) and family == "preset" and not pre_stage(cfg):
                callback.curriculum = CurriculumState(**getattr(model, "curriculum_state", {}))
            callback._on_training_end()
        callback.save_checkpoint("final.zip")
        callback.save_checkpoint("latest.zip")
        final_status = {
            **callback.snapshot(),
            "state": "complete",
            "steps": model.num_timesteps,
            "wall_seconds": perf_counter() - started,
            "budget_stopped": callback.budget_stopped,
            "stop_reason": "stage_mastered"
            if selected_stage(cfg) and callback.stage_mastered
            else "curriculum_mastered"
            if callback.curriculum and callback.curriculum.mastered
            else "time_budget"
            if callback.budget_stopped
            else "game_budget"
            if game_budget
            else "decision_budget",
            "budget_complete": None
            if unlimited
            else progress_value(cfg, model) >= budget_target(cfg),
            "extra_games_in_final_rollout": max(0, progress_value(cfg, model) - budget_target(cfg))
            if game_budget and not unlimited
            else 0,
        }
        training_complete = True
        progress.emit("Saved final.zip; finalization status recorded", force=True)
        callback.hardware_context("reporting")
        write_json(output / "status.json", final_status)
        # Release worker processes before rendering. Exports never update the policy.
        env.close()
        env = None
        visual = None
        completed_learning = (
            not callback.budget_stopped
            and model.phase == "idle"
            and not (
                callback.validation_trigger.pending
                or callback.validation_pending
                or callback.pending_stage_validation
            )
        )
        if settings["visualization"]["enabled"]:
            from pvz_rl.presentation.visualization import visualize_run

            write_json(output / "status.json", {**final_status, "state": "exporting"})
            visual = visualize_run(
                output,
                cfg=cfg,
                videos=settings["visualization"]["videos"],
                progress=progress,
                deadline=wall_budget.deadline,
                report_only=not (output / "best.zip").exists(),
            )
            final_status.update(
                visualization_state=visual["state"], export_seconds=visual.get("export_seconds", 0)
            )
        final_status.update(time_budget=wall_budget.state(), wall_seconds=perf_counter() - started)
        write_json(output / "status.json", final_status)
        progress.phase(Phase.COMPLETE, f"Training complete: {output.resolve()}")
        callback.refresh_report()
        final_status.update(time_budget=wall_budget.state(), wall_seconds=perf_counter() - started)
        write_json(output / "status.json", final_status)
        retain_weights = completed_learning
        return output
    except BaseException as exc:
        if training_complete:
            # An interrupted/failed export cannot invalidate completed learning.
            write_json(
                output / "status.json",
                {
                    **final_status,
                    "visualization_state": "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    "visualization_error": repr(exc),
                },
            )
            progress.emit(f"Training is complete; artifact generation stopped: {exc}", force=True)
            raise
        if model is not None:
            model.wall_budget_state = wall_budget.state()
            if callback:
                callback.save_checkpoint("interrupted.zip")
            else:
                model.save(output / "interrupted.zip")
        failure_status = {}
        if callback is not None and model is not None:
            try:
                failure_status = callback.snapshot()
            except Exception:
                pass  # Preserve the original failure even if its environment is unavailable.
        write_json(
            output / "status.json",
            {
                **failure_status,
                "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": repr(exc),
                "steps": model.num_timesteps if model else 0,
                "tile_scenarios": model.tile_skill_state if model else None,
                "time_budget": wall_budget.state(),
            },
        )
        progress.phase(
            Phase.INTERRUPTED if isinstance(exc, KeyboardInterrupt) else Phase.FAILED,
            f"Training stopped: {exc}; recovery checkpoint saved when available",
        )
        if callback:
            callback.refresh_report()
        raise
    finally:
        if callback:
            callback.close()
        if model is not None and getattr(model, "_reach", None) is not None:
            model._reach.close()
            model._reach = None
        if model is not None and getattr(model, "_peashooter_teacher", None) is not None:
            model._peashooter_teacher.close()
        if model is not None and getattr(model, "_pretrainer", None) is not None:
            model._pretrainer.close()
            model._pretrainer = None
        if model is not None and getattr(model, "_wait_outcomes", None) is not None:
            model._wait_outcomes.close()
            model._wait_outcomes = None
        if model is not None and getattr(model, "_buffer", None) is not None:
            model._buffer.close()
        if model is not None and getattr(model, "_fitting_spool", None) is not None:
            model._close_batches()
        if model is not None and getattr(model, "_logger", None) is not None:
            model.logger.close()
        if env:
            env.close()
        progress.close()
        # Recovery must include interrupted presentation and cleanup, which can
        # happen well after the final model checkpoint was saved.
        status_path = output / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text("utf-8"))
            status.update(time_budget=wall_budget.state(), wall_seconds=perf_counter() - started)
            write_json(status_path, status)
        if retain_weights:
            from pvz_rl.learning.training_artifacts import finish_autonomous_run

            finish_autonomous_run(output)
