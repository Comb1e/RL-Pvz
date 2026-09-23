"""PPO training with validation-only checkpoint selection and resumable artifacts."""

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

from .budget import budget_target, evaluation_interval, progress_value, uses_games
from .config import (
    curriculum_probe_seeds,
    output_settings,
    runtime_settings,
    seed_values,
    simulator,
    validate_config,
)
from .curriculum import (
    LESSONS,
    CurriculumState,
    initial_state,
    selected_stage,
    stage_distribution,
    teaching_enabled,
    validation_after_stage,
)
from .deadline import BudgetExpired, RunBudget
from .evaluation import evaluate, summarize
from .exploration import configure_exploration
from .metrics import episode_task, mean_agent_actions, task_statistics
from .progress import Phase, ProgressReporter, duration
from .provenance import append_jsonl, file_hash, metadata, verify_engine, write_json
from .rewards import REWARD_METRICS
from .scenarios import difficulty_weights
from .spatial_policy import SpatialFeatures, SpatialGroupedPolicy
from .timing import TrainingTimings
from .training_requirements import (
    parameter_changes,
    require_cuda_training,
    require_supported_policy,
    resume_protocol,
    transfer_protocol,
)


def vector_env(cfg, condition, learner_seed, family="preset"):
    from .cuda_env import CudaVecEnv

    require_cuda_training(cfg, condition)
    return CudaVecEnv(cfg, condition, learner_seed, family)


def build_model(cfg, condition, env, seed, log_dir=None):
    from .cuda_env import CudaVecEnv
    from .cuda_ppo import CudaMaskablePPO, configure_tensor_buffer

    require_cuda_training(cfg, condition)
    if not isinstance(env, CudaVecEnv):
        raise ValueError("Training requires the CUDA tensor environment")
    t = cfg["training"]
    policy_kwargs = {
        "net_arch": {"pi": t["hidden_sizes"], "vf": t["hidden_sizes"]},
        "features_extractor_class": SpatialFeatures,
        "features_extractor_kwargs": {"layout_cfg": cfg},
        "critic_learning_rate": t.get("critic_learning_rate"),
    }
    model = CudaMaskablePPO(
        SpatialGroupedPolicy,
        env,
        learning_rate=t["learning_rate"],
        n_steps=t["rollout_size"] // t["n_envs"],
        batch_size=t["batch_size"],
        n_epochs=t["n_epochs"],
        gamma=cfg["reward"]["gamma"],
        gae_lambda=t["gae_lambda"],
        clip_range=t["clip_range"],
        ent_coef=0.0,
        vf_coef=t["vf_coef"],
        max_grad_norm=t["max_grad_norm"],
        normalize_advantage=t["normalize_advantage"],
        target_kl=t["target_kl"] or None,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=t["device"],
        verbose=0,
        tensorboard_log=str(log_dir) if log_dir else None,
    )
    configure_tensor_buffer(model)
    model.value_batch_size = t.get("value_batch_size", 1024)
    configure_exploration(model, cfg)
    if "initial_dig_logit" in cfg.get("policy", {}):
        model.policy.initialize_dig_logit(cfg["policy"]["initial_dig_logit"])
    return model


class TrainingDeadline(Exception):
    """Raised only between complete collect/update cycles."""


class TrainingGamesComplete(Exception):
    """The requested game count was reached; the last rollout has been optimized."""


class TrainingStageComplete(Exception):
    """The selected stage passed its mastery gates after a complete PPO update."""


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
        self.progress = progress or ProgressReporter(
            self.output / "train.log", self.settings["logging"]["progress_seconds"]
        )
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
        self.simulation_ticks = 0
        self.instant_actions = 0
        t = cfg["training"]
        self.target = (
            budget_target(cfg)
            if uses_games(cfg)
            else math.ceil(t["total_steps"] / t["rollout_size"]) * t["rollout_size"]
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
                max(0, self.target - progress) / budget_rate if budget_rate > 0 else None
            ),
            "next_episode_difficulty_weights": None if self.curriculum else weights,
            "curriculum": self.curriculum.to_dict() if self.curriculum else None,
            "curriculum_stage": self.curriculum.name if self.curriculum else "fixed",
            "selected_stage": selected_stage(self.cfg),
            "stage_mastered": bool(self.curriculum and self.curriculum.mastered),
            "curriculum_incomplete": bool(
                self.curriculum and not self.curriculum.complete(self.cfg)
            ),
            "next_episode_tasks": stage_distribution(self.cfg, self.curriculum.stage)
            if self.curriculum
            else None,
            "rolling_episodes": len(rows),
            "rolling_agent_actions": mean_agent_actions(rows),
            "rolling_tasks": dict(
                Counter(
                    r.get("level", "unknown")
                    if r.get("family") == "preset"
                    else r.get("family", "unknown")
                    for r in rows
                )
            ),
            "rolling_mower_free_lessons": sum(
                r.get("family") in (*LESSONS, "diagnostic") for r in rows
            ),
            "rolling_win_rate": sum(r["win"] for r in rows) / len(rows) if rows else None,
            "rolling_return": sum(r["return"] for r in rows) / len(rows) if rows else None,
            "rolling_seconds": (
                sum(r["simulated_seconds"] for r in rows) / len(rows) if rows else None
            ),
            "rolling_attacker_purchases": sum(r.get("attacker_purchases", 0) for r in rows)
            / len(rows)
            if rows
            else None,
            **{
                f"rolling_{key}": sum(r.get(key, 0) for r in rows) / len(rows) if rows else None
                for key in REWARD_METRICS
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
        }

    def log_progress(self, *, force=False):
        if not force and not self.progress.due():
            return
        row = self.snapshot()
        win = "pending" if row["rolling_win_rate"] is None else f"{row['rolling_win_rate']:.1%}"
        reward = "pending" if row["rolling_return"] is None else f"{row['rolling_return']:.3f}"
        best = "pending" if self.best_score == -math.inf else f"{self.best_score:.1%}"
        timing = ""
        if row["last_optimization_seconds"] is not None:
            timing = (
                f"; last rollout {row['last_collection_seconds']:.2f}s collect / "
                f"{row['last_optimization_seconds']:.2f}s update"
            )
        kills = (
            f"; plant/mower kills per game {row['rolling_plant_kills']:.2f}/{row['rolling_mower_kills']:.2f}"
            f"; attackers/game {row['rolling_attacker_purchases']:.2f}, early digs/game {row['rolling_early_voluntary_digs']:.2f}"
            f"; mower cost {row['rolling_mower_activation_penalty']:.3f}"
            if self.recent
            else ""
        )
        context = f"; stage {row['curriculum_stage']}"
        if row["rolling_agent_actions"] is not None:
            context += f"; plant+dig/game {row['rolling_agent_actions']:.2f}"
        if self.recent:
            context += "; tasks " + ", ".join(f"{k}={v}" for k, v in row["rolling_tasks"].items())
            if row["rolling_mower_free_lessons"] == len(self.recent):
                context += "; mowers disabled in these lessons"
        for task, metrics in row["rolling_by_task"].items():
            ratio = metrics["early_digs_per_planting"]
            context += (
                f"; {task} last {metrics['completed_games']}: win {metrics['win_rate']:.1%}"
                f", digs/plant {'pending' if ratio is None else f'{ratio:.1%}'}"
            )
        self.progress.emit(
            f"{row['budget_progress']:,}/{self.target:,} {row['budget_unit']} ({row['budget_progress'] / self.target:.1%}); "
            f"{row['games_per_second'] * 60:.2f} games/min; "
            f"{row['decisions_per_second']:.0f} transitions/s; elapsed {duration(row['wall_seconds'])}; "
            f"{row['simulation_ticks_per_second']:.0f} simulation ticks/s; "
            f"training ETA {duration(row['estimated_remaining_training_seconds'])}; "
            f"last {len(self.recent)} games win {win}, reward {reward}; best validation {best}"
            + timing
            + context
            + kills,
            force=force,
        )
        write_json(self.output / "status.json", row)

    def capture_update(self):
        if self.model._n_updates <= self.last_updates:
            return
        row = self.snapshot()
        values = self.model.logger.name_to_value
        keys = (
            "policy_gradient_loss",
            "value_loss",
            "entropy_loss",
            "joint_entropy",
            "exploration_bonus",
            "choice_fraction",
            "optimizer_steps",
            "actor_optimizer_steps",
            "critic_optimizer_steps",
            "actor_grad_norm",
            "critic_grad_norm",
            "critic_learning_rate",
            "critic_epochs_completed",
            "post_update_approx_kl",
            "post_update_type_kl",
            "dig_probability_when_legal",
            "dig_legal_observations",
            "epochs_completed",
            "kl_stopped",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "learning_rate",
            "n_updates",
            "loss",
        )
        metrics = {}
        if isinstance(self.model.policy, SpatialGroupedPolicy):
            for key, value in self.model.policy.pop_entropy_metrics().items():
                self.model.logger.record(f"train/{key}", value)
                metrics[key] = value
        for key in keys:
            value = values.get(f"train/{key}")
            metrics[key] = (
                float(value) if value is not None and math.isfinite(float(value)) else None
            )
        row.update(optimization=metrics, completed_updates=self.model._n_updates)
        append_jsonl(self.metrics_stream, row)
        self.last_updates = self.model._n_updates

    def refresh_report(self):
        if not self.settings["visualization"]["enabled"]:
            return
        from .visualization import build_run_report, pending_report

        if (
            self.wall_budget is not None
            and self.wall_budget.deadline is not None
            and perf_counter() >= self.wall_budget.deadline
        ):
            pending_report(self.output, self.snapshot())
            return

        started = perf_counter()
        try:
            build_run_report(self.output, self.cfg)
        except Exception as exc:
            write_json(
                self.output / "visualizations" / "status.json",
                {
                    "state": "failed",
                    "error": repr(exc),
                },
            )
            self.progress.emit(f"Report refresh failed: {exc}", force=True)
        finally:
            self.report_seconds += perf_counter() - started

    def _on_training_start(self):
        self.initial_task_counts = copy.deepcopy(getattr(self.model, "training_task_counts", {}))
        self.initial_steps = self.model.num_timesteps
        self.initial_games = getattr(self.model, "training_games", 0)
        progress = progress_value(self.cfg, self.model)
        self.restore_schedule()
        self.training_env.env_method("set_progress", progress)
        self.stream = (self.output / "training-episodes.jsonl").open("w", encoding="utf-8")
        self.metrics_stream = (self.output / "training-metrics.jsonl").open("w", encoding="utf-8")
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
                if (
                    self.curriculum
                    and self.cfg["curriculum"].get("residency") == "episode_start_stage"
                ):
                    self.curriculum.completed_episode(
                        info["episode_metrics"].get("episode_start_stage")
                    )
                self.recent.append(info["episode_metrics"])
                task = episode_task(info["episode_metrics"])
                self.recent_by_task.setdefault(
                    task, deque(maxlen=self.settings["logging"]["rolling_window"])
                ).append(info["episode_metrics"])
                append_jsonl(
                    self.stream,
                    {
                        **info["episode_metrics"],
                        "policy": self.condition,
                        "learner_seed": self.learner_seed,
                        "training_steps": self.num_timesteps,
                        "training_games": self.model.training_games,
                    },
                )
        progress = progress_value(self.cfg, self.model)
        if completed and uses_games(self.cfg):
            # VecEnv has already reset completed workers. Broadcast the global
            # count for subsequent resets; never alter any active episode.
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
        self.log_progress()
        return True

    def _on_rollout_start(self):
        # This hook runs after the preceding PPO update. A step callback would
        # save pre-update weights while labeling them with the newly collected steps.
        self.timings.end_update()
        self.capture_update()
        self.finish_stage_validation()
        self.check_stage_complete()
        if uses_games(self.cfg) and progress_value(self.cfg, self.model) >= self.target:
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
        self.progress.phase(Phase.COLLECTING)
        self.timings.begin_collection()

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
        estimated = (self.timings.last_collection_seconds or 0) + (
            self.timings.last_optimization_seconds or 0
        )
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

    def probe_curriculum(self, *, final=False):
        progress = progress_value(self.cfg, self.model)
        if not self.curriculum or not self.curriculum.due(progress, self.cfg):
            return
        started = perf_counter()
        previous = self.curriculum.name
        self.progress.phase(Phase.VALIDATING, f"Curriculum probe: {previous}")
        wins = {}
        for task in self.curriculum.requirements(self.cfg):
            try:
                rows = self.cached_evaluation(
                    curriculum_probe_seeds(self.cfg),
                    ["easy" if task in LESSONS else task],
                    task if task in LESSONS else "preset",
                    self.output / "curriculum-probes" / str(self.model.num_timesteps) / task,
                    "curriculum_validation",
                    final=final,
                )
            except BudgetExpired:
                self.progress.emit("Curriculum probe pending: time allowance exhausted", force=True)
                self.eval_seconds += perf_counter() - started
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
        self.eval_seconds += perf_counter() - started

    def _on_rollout_end(self):
        self.timings.end_collection()
        self.progress.phase(Phase.UPDATING)
        write_json(self.output / "status.json", self.snapshot())
        self.log_progress()

    def validate(self, *, nominal_games=None, final=False):
        if self.stage_validation and not self.pending_stage_validation:
            return False
        milestone = self.pending_stage_validation
        started = perf_counter()
        self.progress.phase(
            Phase.VALIDATING,
            f"Validation at {progress_value(self.cfg, self.model):,} {'games' if uses_games(self.cfg) else 'decisions'}"
            + (f" after {milestone['stage']} mastery" if milestone else ""),
        )
        write_json(self.output / "status.json", self.snapshot())
        try:
            rows = self.cached_evaluation(
                seed_values(self.cfg, "validation", self.validation_limit),
                ["easy"]
                if self.family in ("diagnostic", *LESSONS)
                else self.cfg["evaluation"]["levels"],
                self.family,
                self.output / "validation" / str(self.model.num_timesteps),
                "validation",
                final=final,
            )
        except BudgetExpired:
            self.validation_pending = True
            self.eval_seconds += perf_counter() - started
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
        self.eval_seconds += perf_counter() - started
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
        self.refresh_report()

    def _on_training_end(self):
        self.timings.end_update()
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
        if self.stream:
            self.stream.close()
        if self.metrics_stream:
            self.metrics_stream.close()
        if self.owns_progress:
            self.progress.close()


def initial_weights(checkpoint, cfg):
    """Read tensors only; the retired shared architecture is never instantiated."""
    from stable_baselines3.common.save_util import load_from_zip_file

    checkpoint = Path(checkpoint).resolve()
    saved = json.loads((checkpoint.parent / "metadata.json").read_text("utf-8"))
    source_cfg = saved["config"]
    validate_config(source_cfg)
    verify_engine(source_cfg)
    compatible = copy.deepcopy(source_cfg)
    shared = compatible["policy"]["kind"] == "spatial_grouped_v3"
    if shared:
        compatible["policy"]["kind"] = "spatial_grouped_v4"
    require_supported_policy(compatible, saved["condition"])
    if transfer_protocol(compatible) != transfer_protocol(cfg):
        raise ValueError(
            "Weights-only initialization requires matching engine, observation and network structure; start fresh for changed dimensions"
        )
    data, parameters, _ = load_from_zip_file(checkpoint, device="cpu")
    weights = {key: value.detach().clone() for key, value in parameters["policy"].items()}
    if shared:
        # SB3's shared module has three registered names in the saved state dict.
        # Reject contradictory aliases instead of silently choosing one set of weights.
        for key, value in list(weights.items()):
            if key.startswith("features_extractor."):
                suffix = key.removeprefix("features_extractor.")
                for prefix in ("pi_features_extractor.", "vf_features_extractor."):
                    if prefix + suffix not in weights or not torch.equal(
                        weights[prefix + suffix], value
                    ):
                        raise ValueError("Shared checkpoint contains inconsistent encoder tensors")
                    weights[prefix + suffix] = value.clone()
    return weights, {
        "mode": "weights_only",
        "conversion": "shared_to_independent" if shared else "none",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_learner_seed": saved["learner_seed"],
        "steps": data["num_timesteps"],
        "games": data.get("training_games", 0),
        "updates": data["_n_updates"],
        "source_structural_signature": transfer_protocol(source_cfg),
        "parameter_changes": parameter_changes(source_cfg, cfg),
    }


def load_policy(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent
    data = json.loads((run / "metadata.json").read_text("utf-8"))
    cfg, condition = data["config"], data["condition"]
    validate_config(cfg)
    verify_engine(cfg)
    require_supported_policy(cfg, condition)
    from .cuda_buffer import TensorRolloutBuffer
    from .cuda_ppo import CudaMaskablePPO

    torch.set_num_threads(cfg["training"]["torch_threads"])
    model = CudaMaskablePPO.load(
        checkpoint,
        device=device,
        custom_objects={"rollout_buffer_class": TensorRolloutBuffer, "rollout_buffer_kwargs": {}},
    )
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
    initialized_model, initialization = None, None
    if init_from:
        initialized_model, initialization = initial_weights(init_from, cfg)
    if resume:
        saved = json.loads((Path(resume).resolve().parent / "metadata.json").read_text("utf-8"))
        require_cuda_training(saved["config"], saved["condition"])
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
    torch.set_num_threads(cfg["training"]["torch_threads"])
    # Complete PPO rollouts are counted explicitly; no discarded partial optimization batch.
    game_budget = uses_games(cfg)
    nominal_steps = None if game_budget else cfg["training"]["total_steps"]
    rollout = cfg["training"]["rollout_size"]
    effective_steps = None if game_budget else math.ceil(nominal_steps / rollout) * rollout
    output.mkdir(parents=True, exist_ok=False)
    details = metadata(
        cfg,
        condition=condition,
        learner_seed=learner_seed,
        family=family,
        nominal_steps=nominal_steps,
        effective_steps=effective_steps,
        target_games=budget_target(cfg) if game_budget else None,
        budget_unit="games" if game_budget else "decisions",
        validation_limit=validation_limit,
        resume=str(Path(resume).resolve()) if resume else None,
        initialization=initialization,
        structural_signature=transfer_protocol(cfg),
        optimizer_settings={
            "actor_learning_rate": cfg["training"]["learning_rate"],
            "critic_learning_rate": cfg["training"].get("critic_learning_rate")
            or cfg["training"]["learning_rate"],
            "value_batch_size": cfg["training"].get("value_batch_size", 1024),
            "independent_encoders": True,
        },
    )
    write_json(output / "metadata.json", details)
    write_json(output / "config.json", cfg)
    write_json(output / "status.json", {"state": "starting"})
    env, model, callback = None, None, None
    settings = output_settings(cfg)
    progress = ProgressReporter(output / "train.log", settings["logging"]["progress_seconds"])
    progress.emit(
        f"Shared {condition} policy; learner seed {learner_seed}; {cfg['training']['device']}; "
        f"{cfg['training']['n_envs']} parallel games; simulator {simulator(cfg)}; "
        f"{cfg['training']['rollout_size'] // cfg['training']['n_envs']} transitions/env/update (includes waits, no game action cap); "
        f"{cfg['training']['rollout_size']} total rollout transitions; "
        f"{budget_target(cfg) if game_budget else effective_steps:,} {'games' if game_budget else 'decisions'}; "
        f"family {family}; output {output.resolve()}",
        force=True,
    )
    progress.emit(f"Data transport: {runtime_settings(cfg)}", force=True)
    progress.emit(
        f"Policy: {cfg.get('policy', {'kind': 'flat'})}; "
        "standard PPO minibatch reduction; "
        f"GAE lambda {cfg['training']['gae_lambda']}; minibatch {cfg['training']['batch_size']}",
        force=True,
    )
    progress.emit(f"Reward settings: {cfg['reward']}", force=True)
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
            "stop at mastery or budget, save final.zip for the next stage",
            force=True,
        )
    progress.emit(
        f"Action timing: {cfg['environment'].get('action_timing', 'fixed')}; "
        f"{cfg['environment']['decision_ticks']} ticks per wait; "
        f"discount {cfg['reward']['gamma']} per policy decision",
        force=True,
    )
    training_complete = False
    prior_time = 0.0
    if resume:
        previous_status = Path(resume).resolve().parent / "status.json"
        if previous_status.exists():
            prior_time = (
                json.loads(previous_status.read_text("utf-8")).get("time_budget") or {}
            ).get("elapsed_seconds", 0.0)
    wall_budget = RunBudget(cfg, elapsed=prior_time, started=started)
    try:
        env = vector_env(cfg, condition, learner_seed, family)
        if resume:
            model, old = load_policy(resume, cfg["training"]["device"])
            wall_budget.prior_elapsed = max(
                wall_budget.prior_elapsed,
                getattr(model, "wall_budget_state", {}).get("elapsed_seconds", 0.0),
            )
            model.set_env(env)
            from .cuda_ppo import configure_tensor_buffer

            configure_tensor_buffer(model)
            model.tensorboard_log = str(output / "tensorboard")
            details["resume_steps"] = model.num_timesteps
            details["resume_games"] = getattr(model, "training_games", 0)
            write_json(output / "metadata.json", details)
            progress.emit(
                f"Resumed shared policy and both optimizers at {progress_value(cfg, model):,} {'games' if game_budget else 'decisions'}",
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
        remaining = (
            budget_target(cfg) - progress_value(cfg, model)
            if game_budget
            else effective_steps - model.num_timesteps
        )
        if remaining < 0 and not game_budget:
            raise ValueError("Checkpoint exceeds this run's training budget")
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
                    shutil.copy2(previous / "best.zip", output / "best.zip")
                    shutil.copy2(previous / "best.json", output / "best.json")
                    callback.best_score = best["macro_win_rate"]
        env.env_method("set_progress", progress_value(cfg, model))
        if teaching_enabled(cfg) and family == "preset":
            env.env_method(
                "set_curriculum_stage", getattr(model, "curriculum_state", {}).get("stage", 0)
            )
        if remaining > 0:
            try:
                model.learn(
                    # Game count is the stopping criterion. SB3 requires a numeric
                    # timesteps bound; this sentinel does not schedule learning.
                    2**63 - 1 if game_budget else remaining,
                    callback=callback,
                    reset_num_timesteps=not bool(resume),
                    tb_log_name=condition,
                )
            except TrainingGamesComplete:
                progress.emit("Game target reached; final PPO update completed", force=True)
                callback._on_training_end()
            except TrainingStageComplete:
                progress.emit("Mastery reached; finalizing its checkpoint", force=True)
                callback._on_training_end()
            except TrainingDeadline:
                progress.emit(
                    "Time allowance reached at a completed PPO update; finalizing available results",
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
            "budget_complete": progress_value(cfg, model) >= budget_target(cfg),
            "extra_games_in_final_rollout": max(0, progress_value(cfg, model) - budget_target(cfg))
            if game_budget
            else 0,
        }
        training_complete = True
        progress.emit("Saved final.zip; finalization status recorded", force=True)
        write_json(output / "status.json", final_status)
        # Release worker processes before rendering. Exports never update the policy.
        env.close()
        env = None
        visual = None
        if settings["visualization"]["enabled"] and (output / "best.zip").exists():
            from .visualization import visualize_run

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
