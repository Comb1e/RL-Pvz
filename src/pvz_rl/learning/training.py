"""Q training with validation-only checkpoint selection and resumable artifacts."""

from __future__ import annotations

import copy
import json
import math
import shutil
from collections import Counter, deque
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3.common.callbacks import BaseCallback

from pvz_rl.config import (
    curriculum_probe_seeds,
    output_settings,
    runtime_settings,
    seed_values,
    simulator,
    validate_config,
)
from pvz_rl.envs.rewards import LEDGER_METRICS, REWARD_METRICS
from pvz_rl.envs.scenarios import difficulty_weights
from pvz_rl.evaluation.runner import evaluate, summarize
from pvz_rl.learning.budget import (
    budget_target,
    effective_limits,
    evaluation_interval,
    progress_value,
    until_stage_complete,
    uses_games,
)
from pvz_rl.learning.curriculum import (
    CurriculumState,
    initial_state,
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
from pvz_rl.learning.training_requirements import (
    parameter_changes,
    require_cuda_training,
    require_supported_policy,
    resume_protocol,
    transfer_protocol,
)
from pvz_rl.monitoring.metrics import episode_task, mean_agent_actions, task_statistics
from pvz_rl.monitoring.progress import Phase, ProgressReporter
from pvz_rl.monitoring.timing import TrainingTimings
from pvz_rl.provenance import append_jsonl, file_hash, metadata, verify_engine, write_json


def vector_env(cfg, condition, learner_seed, family="preset"):
    from pvz_rl.envs.cuda_env import CudaVecEnv

    require_cuda_training(cfg, condition)
    return CudaVecEnv(cfg, condition, learner_seed, family)


def build_model(cfg, condition, env, seed, log_dir=None):
    from pvz_rl.envs.cuda_env import CudaVecEnv

    require_cuda_training(cfg, condition)
    if not isinstance(env, CudaVecEnv):
        raise ValueError("Training requires the CUDA tensor environment")
    t = cfg["training"]
    from pvz_rl.learning.recurrent_q import CudaRecurrentQ
    from pvz_rl.policy.recurrent_policy import RecurrentQPolicy

    algorithm, policy_class = CudaRecurrentQ, RecurrentQPolicy
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
    if t.get("performance", {}).get("compile_kernels", False):
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
    """Adam's host-side step counter; capturable device counters are never read."""
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
        self.eval_seconds = 0.0
        self.stream = None
        self.settings = output_settings(cfg)
        self.owns_progress = progress is None
        self.progress = progress or ProgressReporter.from_settings(self.output / "train.log", cfg)
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
        self.evaluation_cache = {}
        self.cache_steps = -1
        self.budget_stopped = False
        self.curriculum = None
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
                    "validation": Activity.VALIDATING,
                    "reporting": Activity.REPORTING,
                    "collect": Activity.COLLECTING,
                    "finalize_rewards": Activity.UPDATING,
                    "fit": Activity.UPDATING,
                    "returns": Activity.UPDATING,
                    "synchronize": Activity.UPDATING,
                }[self.hardware_activity]
            )
        if self.hardware:
            cohort = getattr(self.model, "cohort_metrics", {})
            self.hardware.update(
                stage=self.curriculum.name if self.curriculum else self.family,
                phase=getattr(self.model, "phase", "starting"),
                activity=self.hardware_activity,
                training_steps=self.model.num_timesteps,
                **self.q_status(),
            )
            if cohort:
                self.hardware.record_window(cohort)

    def q_status(self):
        buffer = getattr(self.model, "_buffer", None)
        stats = getattr(self.model, "_stats", {})
        return dict(
            planting_samples=int(buffer.group_counts[1])
            if buffer is not None and buffer.finalized
            else getattr(self.model, "_planting_samples", 0),
            q_optimizer_steps=stats.get("q_optimizer_steps", 0),
            tile_exploration_epsilon=getattr(
                getattr(self.model, "policy", None), "tile_exploration_epsilon", 0.0
            ),
            species_counts=buffer.species_counts.tolist()
            if buffer is not None and buffer.finalized
            else getattr(self.model, "cohort_metrics", {}).get("species_counts"),
            species_exploration_coins=stats.get("species_exploration_coins", 0),
            tile_exploration_coins=stats.get("tile_exploration_coins", 0),
            exploratory_changes=stats.get("exploratory_changes", 0),
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
        budget_rate = game_rate if uses_games(self.cfg) else rate
        rows = list(self.recent)
        decisions = sum(r["decisions"] for r in rows)
        purchases = sum(sum(r.get("plant_usage", {}).values()) for r in rows)
        weights = difficulty_weights(
            self.cfg,
            progress,
            budget_target(self.cfg),
            self.cfg["conditions"][self.condition]["curriculum"],
        )
        return {
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
            "next_episode_difficulty_weights": None if self.curriculum else weights,
            "curriculum": self.curriculum.to_dict() if self.curriculum else None,
            "curriculum_stage": self.curriculum.name if self.curriculum else "fixed",
            "training_phase": str(getattr(self.model, "phase", "starting")),
            "q_prefit": getattr(self.model, "prefit_errors", {}),
            "exploration_rate": getattr(self.model, "exploration_rate", 0.0),
            "tile_exploration_epsilon": getattr(self.model, "exploration_rate", 0.0),
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
            "stage_mastered": bool(self.curriculum and self.curriculum.mastered),
            "curriculum_incomplete": bool(
                self.curriculum and not self.curriculum.complete(self.cfg)
            ),
            "next_episode_tasks": stage_distribution(self.cfg, self.curriculum.stage)
            if self.curriculum
            else None,
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
            "validation_pending": self.validation_pending,
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

    def log_progress(self, *, force=False):
        """Write the detailed status snapshot; offer one compact line to the terminal."""
        if not force and not self.progress.due():
            return
        self.hardware_context()
        row = self.snapshot()
        write_json(self.output / "status.json", row)
        text, change = self.compact_status(row)
        self.progress.emit(text, force=force, change=change)

    def compact_status(self, row):
        """Terminal summary from host-side status; never reads device tensors."""
        model = self.model
        phase = str(row["training_phase"])
        cohort = getattr(model, "_n_updates", 0) + 1
        if phase == "collect" and getattr(model, "env", None) is not None:
            size = getattr(model, "cohort_games", 0)
            finished = max(0, size - int(model.env.enabled_envs[:size].sum()))
            marker = f"{finished}/{size} games"
        elif phase == "fit":
            marker = (
                f"pass {getattr(model, '_fit_epoch', 0) + 1}/{self.cfg['training']['n_epochs']}"
            )
        else:
            marker = phase
        games = f"games {row['training_games']:,}"
        if row["target_games"] is not None:
            games += f"/{row['target_games']:,}"
        step = optimizer_step(model)
        parts = [
            f"cohort {cohort} {marker}",
            games,
            f"transitions {row['training_steps']:,}",
            f"{row['decisions_per_second']:,.0f}/s",
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
        if self.progress.warning:
            parts.append(f"warning: {self.progress.warning[:96]}")
        change = (phase, cohort, marker, row["training_games"], row["training_steps"], step)
        return " | ".join(parts), change

    def cohort_phase(self, previous, phase):
        """Print cohort boundaries: collection start/completion; fitting is reported at its end."""
        from pvz_rl.learning.cohort import CohortPhase

        self.progress.phase(Phase.COLLECTING if phase == CohortPhase.COLLECT else Phase.UPDATING)
        if phase == CohortPhase.COLLECT:
            self.log_progress(force=True)
        elif previous == CohortPhase.COLLECT:
            model = self.model
            seconds = model._phase_times["collect"]
            transitions = model.num_timesteps - getattr(model, "cohort_start_steps", 0)
            self.progress.emit(
                f"Cohort {model._n_updates + 1} collected: {getattr(model, 'cohort_games', 0)} "
                f"games, {transitions:,} transitions in {seconds:.1f}s "
                f"({transitions / max(seconds, 1e-9):,.0f}/s)",
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
            "q_loss",
            "branch_loss",
            "tile_loss",
            "q_grad_norm",
            "q_optimizer_steps",
            "species_exploration_coins",
            "tile_exploration_coins",
            "exploratory_changes",
            "learning_rate",
            "n_updates",
        )
        keys += tuple(
            f"{metric}_{action}"
            for action in ("wait", "plant", "dig")
            for metric in ("action_count", "value_target_error", "tile_target_error")
        )
        metrics = {}
        for key in keys:
            value = values.get(f"train/{key}")
            metrics[key] = (
                float(value) if value is not None and math.isfinite(float(value)) else None
            )
        row.update(optimization=metrics, completed_updates=self.model._n_updates)
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
        self.restore_schedule()
        self.training_env.env_method("set_progress", progress)
        self.stream = (self.output / "training-episodes.jsonl").open("a", encoding="utf-8")
        self.metrics_stream = (self.output / "training-metrics.jsonl").open("a", encoding="utf-8")
        self.last_updates = self.model._n_updates
        if teaching_enabled(self.cfg) and self.family == "preset":
            self.curriculum = CurriculumState(
                **getattr(self.model, "curriculum_state", initial_state(self.cfg).to_dict())
            )
            self.training_env.env_method("set_curriculum_stage", self.curriculum.stage)
            self.sync_curriculum()
        self.progress.phase(Phase.COLLECTING)
        self.log_progress(force=True)

    def _on_step(self):
        completed = 0
        for info in self.locals["infos"]:
            self.simulation_ticks += info["ticks_advanced"]
            self.instant_actions += int(info["ticks_advanced"] == 0)
            if "episode_metrics" in info:
                self.model.training_games = getattr(self.model, "training_games", 0) + 1
                completed += 1
                if self.curriculum and (
                    self.cfg["curriculum"].get("residency") == "episode_start_stage"
                    or self.cfg["training"]["exploration"].get("decay_games", 0) > 0
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
        self.check_deadline()
        self.probe_curriculum()
        self.finish_stage_validation()
        self.check_stage_complete()
        self.check_deadline()
        stage_games = (
            self.curriculum.completed_stage_games
            if self.curriculum
            else getattr(self.model, "training_games", 0)
        )
        state = exploration_state(self.cfg, stage_games, staged=self.curriculum is not None)
        apply_exploration_state(self.model, self.cfg, state)
        self.hardware_context("training")
        self.progress.phase(Phase.COLLECTING)

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

    def finish_stage_validation(self):
        if self.stage_validation and self.pending_stage_validation:
            if self.validate() is False:
                # Do not update past an unevaluated mastery checkpoint. Finalization
                # can retry with its reserved time, or resume can retry these weights.
                self.budget_stopped = True
                raise TrainingDeadline()

    def check_stage_complete(self):
        if self.curriculum and self.curriculum.mastered:
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
        if self.wall_budget is None:
            return None
        return self.wall_budget.deadline if final else self.wall_budget.learning_deadline

    def save_checkpoint(self, name):
        self.task_counts()
        if self.wall_budget:
            self.model.wall_budget_state = self.wall_budget.state()
        self.model.research_schedule = {
            "next_eval": self.next_eval,
            "last_eval": self.last_eval,
            "pending_stage_validation": copy.deepcopy(self.pending_stage_validation),
        }
        self.sync_curriculum()
        self.model.save(self.output / name)
        self.progress.emit(
            f"Checkpoint saved: {name}; transitions {self.model.num_timesteps:,}; "
            f"optimizer step {optimizer_step(self.model):,}",
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
            return operation(**kwargs)
        finally:
            self.eval_seconds += perf_counter() - started
            self.refresh_report()
            self.hardware_context(previous_activity)

    def probe_curriculum(self, *, final=False):
        if not self.curriculum or not self.curriculum.due(
            progress_value(self.cfg, self.model), self.cfg
        ):
            return
        return self.evaluation_event(self._probe_curriculum, final=final)

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
                    curriculum_probe_seeds(self.cfg),
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
        advanced = self.curriculum.observe(
            wins, progress, self.cfg, advance=selected_stage(self.cfg) is None
        )
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

    def _on_rollout_end(self):
        self.timings.record_window(self.model.cohort_metrics)
        self.hardware_context()
        self.sync_curriculum()
        self.progress.phase(Phase.UPDATING)
        cohort = self.model.cohort_metrics
        self.progress.emit(
            f"Cohort {self.model._n_updates} fitted: {cohort.get('q_optimizer_steps', 0)} passes "
            f"in {cohort['fit_seconds']:.1f}s after {cohort['collection_seconds']:.1f}s "
            f"collection; {cohort['transitions_per_second']:,.0f} transitions/s; "
            f"optimizer step {optimizer_step(self.model):,}",
            force=True,
        )
        write_json(self.output / "status.json", self.snapshot())

    def validate(self, *, nominal_games=None, final=False):
        if self.stage_validation and not self.pending_stage_validation:
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
                seed_values(self.cfg, "validation", self.validation_limit),
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
        scores = summarize(rows)
        score = sum(r["win_rate"] for r in scores.values()) / len(scores)
        self.validation_pending = False
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
    saved = inspect_checkpoint(checkpoint)
    source_cfg = saved["config"]
    compatible_config(
        source_cfg,
        cfg,
        weights_only=saved["initialization_type"] == "demonstration",
    )
    verify_engine(source_cfg)
    model, _ = load_policy(checkpoint, "cpu")
    weights = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    return weights, {
        "mode": "weights_only",
        "type": saved["initialization_type"],
        "conversion": "none",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_learner_seed": saved["learner_seed"],
        "steps": model.num_timesteps,
        "games": getattr(model, "training_games", 0),
        "updates": model._n_updates,
        "source_structural_signature": transfer_protocol(source_cfg),
        "parameter_changes": parameter_changes(source_cfg, cfg),
    }


def load_policy(checkpoint, device="cpu", *, for_resume=False):
    from gymnasium import spaces
    from pvz_game import Rules

    from pvz_rl.envs.actions import ActionSchema
    from pvz_rl.envs.encoding import ObservationEncoder
    from pvz_rl.learning.checkpoints import inspect_checkpoint, model_class
    from pvz_rl.policy.recurrent_policy import RecurrentQPolicy

    data = inspect_checkpoint(checkpoint)
    cfg = data["config"]
    verify_engine(cfg)
    require_supported_policy(cfg, data["condition"])
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if data["initialization_type"] == "demonstration":
        model = model_class(cfg)(
            RecurrentQPolicy,
            device=device,
            _init_setup_model=False,
            policy_kwargs={"features_extractor_kwargs": {"layout_cfg": cfg}},
        )
        model.observation_space = ObservationEncoder(cfg, Rules()).space
        model.action_space = spaces.Discrete(ActionSchema.size)
        model.n_envs = cfg["training"]["n_envs"]
        model._setup_model()
        model.policy.load_state_dict(data.pop("payload")["model"], strict=True)
    else:
        model = model_class(cfg).load(checkpoint, device=device)
        if (
            transfer_protocol(model.cfg) != transfer_protocol(cfg)
            or model.cfg["reward"] != cfg["reward"]
        ):
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
    validate_config(cfg)
    output = Path(output)
    require_cuda_training(cfg, condition)
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
        if saved["initialization_type"] != "autonomous":
            raise ValueError(
                "Demonstration weights require --init-from; --resume needs an autonomous ZIP"
            )
        require_cuda_training(saved["config"], saved["condition"])
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
    resumed_model = (
        load_policy(resume, cfg["training"]["device"], for_resume=True)[0] if resume else None
    )
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
            "adam_epsilon": 1e-8 if cfg["policy"]["kind"] == "transformer_lstm_q_v2" else 1e-5,
            "batch_size": cfg["training"]["batch_size"],
            "shared_encoder": True,
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
        f"{cfg['training']['n_envs']} parallel games; simulator {simulator(cfg)}; "
        "one complete game per environment per cohort; "
        f"{budget_label}; "
        f"family {family}; output {output.resolve()}",
        force=True,
    )
    progress.emit(f"Data transport: {runtime_settings(cfg)}", force=True)
    progress.emit(
        f"Policy: {cfg.get('policy', {'kind': 'flat'})}; "
        f"actual returns, one sequential Q fitting phase; minibatch {cfg['training']['batch_size']}",
        force=True,
    )
    progress.emit(f"Reward settings: {cfg['reward']}", force=True)
    progress.emit(
        f"Tile-only exploration {cfg['training']['exploration']}; greedy wait/plant/dig values",
        force=True,
    )
    if validation_after_stage(cfg, family):
        probe_key = "probe_interval_games" if game_budget else "probe_interval"
        progress.emit(
            f"Mastery probes every {cfg['curriculum'][probe_key]:,} "
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
        f"Action timing: {cfg['environment'].get('action_timing', 'fixed')}; "
        f"{cfg['environment']['decision_ticks']} ticks per wait; "
        f"discount {cfg['training']['gamma']} per simulation tick",
        force=True,
    )
    training_complete = False
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
            if cfg["training"].get("performance", {}).get("compile_kernels", False):
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
            if initialized_model is not None:
                model.policy.load_state_dict(initialized_model, strict=True)
                progress.emit(
                    f"Initialized weights from {initialization['checkpoint']}; fresh optimizers, counters and budget",
                    force=True,
                )
            if teaching_enabled(cfg) and family == "preset":
                model.curriculum_state = initial_state(cfg).to_dict()
        model.research_metadata = details
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
            callback.init_callback(model)
            callback.initial_steps = model.num_timesteps
            callback.initial_games = getattr(model, "training_games", 0)
            callback.restore_schedule()
            if teaching_enabled(cfg) and family == "preset":
                callback.curriculum = CurriculumState(**getattr(model, "curriculum_state", {}))
            if callback.stage_validation:
                if callback.pending_stage_validation:
                    callback.validate(final=True)
            else:
                callback.validate(final=True)
        callback.save_checkpoint("final.zip")
        callback.save_checkpoint("latest.zip")
        final_status = {
            **callback.snapshot(),
            "state": "complete",
            "steps": model.num_timesteps,
            "wall_seconds": perf_counter() - started,
            "budget_stopped": callback.budget_stopped,
            "stop_reason": "stage_mastered"
            if selected_stage(cfg) and callback.curriculum and callback.curriculum.mastered
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
        if settings["visualization"]["enabled"] and (output / "best.zip").exists():
            from pvz_rl.presentation.visualization import visualize_run

            write_json(output / "status.json", {**final_status, "state": "exporting"})
            visual = visualize_run(
                output,
                cfg=cfg,
                videos=settings["visualization"]["videos"],
                progress=progress,
                deadline=wall_budget.deadline,
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
        write_json(
            output / "status.json",
            {
                "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": repr(exc),
                "steps": model.num_timesteps if model else 0,
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
        if model is not None and getattr(model, "_buffer", None) is not None:
            model._buffer.close()
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
