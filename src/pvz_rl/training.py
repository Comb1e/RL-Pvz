"""PPO training with validation-only checkpoint selection and resumable artifacts."""

from __future__ import annotations

import copy
import json
import math
import shutil
from collections import deque
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .budget import budget_target, evaluation_interval, progress_value, uses_games
from .config import (
    output_settings,
    research_config,
    runtime_settings,
    seed_values,
    simulator,
    validate_config,
)
from .curriculum import LESSONS, CurriculumState, stage_distribution, teaching_enabled
from .env import PvZEnv
from .evaluation import evaluate, summarize
from .grouped_policy import GroupedPolicy
from .progress import Phase, ProgressReporter, duration
from .provenance import append_jsonl, file_hash, metadata, verify_engine, write_json
from .rewards import REWARD_METRICS
from .runtime import (
    CachedMaskVecEnv,
    MaskTransport,
    configure_rollout_buffer,
    rollout_buffer_class,
)
from .scenarios import difficulty_weights
from .timing import TrainingTimings


def make_env(cfg, condition, worker_seed, family):
    env = PvZEnv(cfg, condition=condition, training=True, worker_seed=worker_seed, family=family)
    if cfg["conditions"][condition]["masked"] and runtime_settings(cfg)["coalesce_masks"]:
        env = MaskTransport(env)
    return Monitor(env)


def vector_env(cfg, condition, learner_seed, family="preset"):
    if simulator(cfg) == "cuda":
        from .cuda_env import CudaVecEnv

        return CudaVecEnv(cfg, condition, learner_seed, family)
    factories = [
        partial(make_env, cfg, condition, learner_seed * 1000 + i, family)
        for i in range(cfg["training"]["n_envs"])
    ]
    env = (
        DummyVecEnv(factories)
        if len(factories) == 1
        else SubprocVecEnv(factories, start_method="spawn")
    )
    if cfg["conditions"][condition]["masked"] and runtime_settings(cfg)["coalesce_masks"]:
        env = CachedMaskVecEnv(env)
    return env


def build_model(cfg, condition, env, seed, log_dir=None):
    t = cfg["training"]
    grouped = cfg.get("policy", {}).get("kind", "flat") == "grouped_v1"
    if grouped and (
        not cfg["conditions"][condition]["masked"] or cfg["conditions"][condition]["hybrid"]
    ):
        raise ValueError(
            "Grouped profiles require direct masked PPO; use the baseline config for other conditions"
        )
    if teaching_enabled(cfg) and (
        not cfg["conditions"][condition]["curriculum"]
        or cfg["conditions"][condition]["hybrid"]
        or not cfg["conditions"][condition]["masked"]
    ):
        raise ValueError("Teaching curriculum requires a direct masked curriculum condition")
    algorithm = MaskablePPO if cfg["conditions"][condition]["masked"] else PPO
    if simulator(cfg) == "cuda":
        from .cuda_ppo import CudaMaskablePPO, CudaPPO

        algorithm = CudaMaskablePPO if cfg["conditions"][condition]["masked"] else CudaPPO
    model = algorithm(
        GroupedPolicy if grouped else "MlpPolicy",
        env,
        learning_rate=t["learning_rate"],
        n_steps=t["rollout_size"] // t["n_envs"],
        batch_size=t["batch_size"],
        n_epochs=t["n_epochs"],
        gamma=cfg["reward"]["gamma"],
        gae_lambda=t["gae_lambda"],
        clip_range=t["clip_range"],
        ent_coef=t["ent_coef"],
        policy_kwargs={"net_arch": {"pi": t["hidden_sizes"], "vf": t["hidden_sizes"]}},
        seed=seed,
        device=t["device"],
        verbose=0,
        tensorboard_log=str(log_dir) if log_dir else None,
        rollout_buffer_class=rollout_buffer_class(
            cfg["conditions"][condition]["masked"],
            runtime_settings(cfg)["cache_rollout_on_device"],
        ),
    )
    if simulator(cfg) == "cuda":
        from .cuda_ppo import configure_tensor_buffer

        configure_tensor_buffer(model, cfg["conditions"][condition]["masked"])
    return model


class TrainingDeadline(Exception):
    """Raised only between complete collect/update cycles."""


class TrainingGamesComplete(Exception):
    """The requested game count was reached; the last rollout has been optimized."""


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
        self.metrics_stream = None
        self.report_seconds = 0.0
        self.last_updates = 0
        self.last_weights = None
        self.timings = TrainingTimings()
        self.deadline = deadline
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
        weights = difficulty_weights(
            self.cfg,
            progress,
            budget_target(self.cfg),
            self.cfg["conditions"][self.condition]["curriculum"],
        )
        return {
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
            "next_episode_difficulty_weights": weights,
            "curriculum": self.curriculum.to_dict() if self.curriculum else None,
            "curriculum_stage": self.curriculum.name if self.curriculum else "fixed",
            "curriculum_incomplete": bool(self.curriculum and self.curriculum.name != "shared"),
            "next_episode_tasks": stage_distribution(self.cfg, self.curriculum.stage)
            if self.curriculum
            else None,
            "rolling_episodes": len(rows),
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
            f"; plant/mower kills per game {row['rolling_plant_kills']:.1f}/{row['rolling_mower_kills']:.1f}"
            f"; damage reward {row['rolling_damage_reward']:.3f}"
            f"; empty mowers {row['rolling_empty_mower_activations']:.1f}"
            f"; mower sun penalty {row['rolling_mower_sun_penalty']:.3f}"
            f"; wall-nut reward {row['rolling_wall_nut_reward']:.3f}"
            f"; empty blasts {row['rolling_empty_explosions']:.1f}"
            if self.recent
            else ""
        )
        self.progress.emit(
            f"{row['budget_progress']:,}/{self.target:,} {row['budget_unit']} ({row['budget_progress'] / self.target:.1%}); "
            f"{row['games_per_second'] * 60:.2f} games/min; "
            f"{row['decisions_per_second']:.0f} decisions/s; elapsed {duration(row['wall_seconds'])}; "
            f"{row['simulation_ticks_per_second']:.0f} simulation ticks/s; "
            f"training ETA {duration(row['estimated_remaining_training_seconds'])}; "
            f"last {len(self.recent)} games win {win}, reward {reward}; best validation {best}"
            + timing
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
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "learning_rate",
            "n_updates",
            "loss",
        )
        metrics = {}
        if isinstance(self.model.policy, GroupedPolicy):
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
        from .visualization import build_run_report

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
        self.initial_steps = self.model.num_timesteps
        self.initial_games = getattr(self.model, "training_games", 0)
        interval = evaluation_interval(self.cfg)
        progress = progress_value(self.cfg, self.model)
        self.next_eval = ((progress // interval) + 1) * interval
        self.training_env.env_method("set_progress", progress)
        self.stream = (self.output / "training-episodes.jsonl").open("w", encoding="utf-8")
        self.metrics_stream = (self.output / "training-metrics.jsonl").open("w", encoding="utf-8")
        self.last_updates = self.model._n_updates
        if teaching_enabled(self.cfg) and self.family == "preset":
            self.curriculum = CurriculumState(**getattr(self.model, "curriculum_state", {}))
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
                self.recent.append(info["episode_metrics"])
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
        if uses_games(self.cfg) and progress_value(self.cfg, self.model) >= self.target:
            raise TrainingGamesComplete()
        self.check_deadline()
        self.probe_curriculum()
        self.check_deadline()
        if progress_value(self.cfg, self.model) >= self.next_eval:
            self.validate()
            while self.next_eval <= progress_value(self.cfg, self.model):
                self.next_eval += evaluation_interval(self.cfg)
        self.check_deadline()
        self.progress.phase(Phase.COLLECTING)
        self.timings.begin_collection()

    def check_deadline(self):
        if self.deadline is not None and perf_counter() >= self.deadline:
            self.budget_stopped = True
            raise TrainingDeadline()

    def sync_curriculum(self):
        if self.curriculum:
            self.model.curriculum_state = self.curriculum.to_dict()
            write_json(self.output / "curriculum.json", self.model.curriculum_state)

    def probe_curriculum(self):
        progress = progress_value(self.cfg, self.model)
        if not self.curriculum or not self.curriculum.due(progress, self.cfg):
            return
        started = perf_counter()
        previous = self.curriculum.name
        self.progress.phase(Phase.VALIDATING, f"Curriculum probe: {previous}")
        wins = {}
        for task in self.curriculum.requirements(self.cfg):
            rows = evaluate(
                self.cfg,
                policy=self.model,
                condition=self.condition,
                learner_seed=self.learner_seed,
                seeds=seed_values(self.cfg, "validation", self.cfg["curriculum"]["probe_cases"]),
                levels=["easy" if task in LESSONS else task],
                family=task if task in LESSONS else "preset",
                split="curriculum_validation",
                output=self.output / "curriculum-probes" / str(self.model.num_timesteps) / task,
                training_steps=self.model.num_timesteps,
                progress=self.progress,
            )
            wins[task] = sum(row["win"] for row in rows)
        advanced = self.curriculum.observe(wins, progress, self.cfg)
        self.sync_curriculum()
        if advanced:
            self.training_env.env_method("set_curriculum_stage", self.curriculum.stage)
        with (self.output / "curriculum-probes.jsonl").open("a", encoding="utf-8") as stream:
            append_jsonl(
                stream,
                {
                    "training_steps": self.model.num_timesteps,
                    "training_games": getattr(self.model, "training_games", 0),
                    "stage": previous,
                    "wins": wins,
                    "advanced": advanced,
                    "state": self.curriculum.to_dict(),
                },
            )
        self.progress.emit(
            f"Curriculum {previous}: {wins}; next-reset stage {self.curriculum.name}; consecutive passes {self.curriculum.consecutive_passes}",
            force=True,
        )
        self.eval_seconds += perf_counter() - started

    def _on_rollout_end(self):
        self.timings.end_collection()
        self.progress.phase(Phase.UPDATING)
        write_json(self.output / "status.json", self.snapshot())
        self.log_progress()

    def validate(self):
        started = perf_counter()
        self.progress.phase(
            Phase.VALIDATING,
            f"Validation at {progress_value(self.cfg, self.model):,} {'games' if uses_games(self.cfg) else 'decisions'}",
        )
        write_json(self.output / "status.json", self.snapshot())
        rows = evaluate(
            self.cfg,
            policy=self.model,
            condition=self.condition,
            learner_seed=self.learner_seed,
            seeds=seed_values(self.cfg, "validation", self.validation_limit),
            levels=(
                ["easy"]
                if self.family in ("diagnostic", *LESSONS)
                else self.cfg["evaluation"]["levels"]
            ),
            family=self.family,
            output=self.output / "validation" / str(self.model.num_timesteps),
            training_steps=self.model.num_timesteps,
            progress=self.progress,
        )
        scores = summarize(rows)
        score = sum(r["win_rate"] for r in scores.values()) / len(scores)
        self.progress.emit(
            "Validation "
            + ", ".join(f"{key}: {value['win_rate']:.1%}" for key, value in scores.items())
            + f"; shared-policy mean {score:.1%}",
            force=True,
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output / "best.zip")
            write_json(
                self.output / "best.json",
                {
                    "steps": self.model.num_timesteps,
                    "games": getattr(self.model, "training_games", 0),
                    "macro_win_rate": score,
                    "checkpoint_hash": file_hash(self.output / "best.zip"),
                },
            )
            self.progress.emit(f"Saved shared best.zip at {score:.1%}", force=True)
        self.model.save(self.output / "latest.zip")
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
        if not self.budget_stopped:
            self.probe_curriculum()
        if self.last_eval != self.model.num_timesteps and not self.budget_stopped:
            self.validate()

    def close(self):
        if self.stream:
            self.stream.close()
        if self.metrics_stream:
            self.metrics_stream.close()
        if self.owns_progress:
            self.progress.close()


def load_policy(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent
    data = json.loads((run / "metadata.json").read_text("utf-8"))
    cfg, condition = data["config"], data["condition"]
    validate_config(cfg)
    verify_engine(cfg)
    algorithm = MaskablePPO if cfg["conditions"][condition]["masked"] else PPO
    if simulator(cfg) == "cuda":
        from .cuda_ppo import CudaMaskablePPO, CudaPPO

        algorithm = CudaMaskablePPO if cfg["conditions"][condition]["masked"] else CudaPPO
    torch.set_num_threads(cfg["training"]["torch_threads"])
    model = algorithm.load(checkpoint, device=device)
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
    deadline=None,
):
    cfg = copy.deepcopy(cfg)
    validate_config(cfg)
    output = Path(output)
    if condition not in cfg["conditions"]:
        raise ValueError("Unknown training condition")
    if cfg["training"]["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
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
    )
    write_json(output / "metadata.json", details)
    write_json(output / "config.json", cfg)
    write_json(output / "status.json", {"state": "starting"})
    env, model, callback = None, None, None
    started = perf_counter()
    settings = output_settings(cfg)
    progress = ProgressReporter(output / "train.log", settings["logging"]["progress_seconds"])
    progress.emit(
        f"Shared {condition} policy; learner seed {learner_seed}; {cfg['training']['device']}; "
        f"{cfg['training']['n_envs']} parallel games; simulator {simulator(cfg)}; "
        f"{cfg['training']['rollout_size'] // cfg['training']['n_envs']} decisions/game/rollout; "
        f"{cfg['training']['rollout_size']} total rollout decisions; "
        f"{budget_target(cfg) if game_budget else effective_steps:,} {'games' if game_budget else 'decisions'}; "
        f"family {family}; output {output.resolve()}",
        force=True,
    )
    progress.emit(f"Data transport: {runtime_settings(cfg)}", force=True)
    progress.emit(f"Reward settings: {cfg['reward']}", force=True)
    progress.emit(
        f"Action timing: {cfg['environment'].get('action_timing', 'fixed')}; "
        f"{cfg['environment']['decision_ticks']} ticks per wait; "
        f"discount {cfg['reward']['gamma']} per policy decision",
        force=True,
    )
    training_complete = False
    try:
        env = vector_env(cfg, condition, learner_seed, family)
        if resume:
            model, old = load_policy(resume, cfg["training"]["device"])
            if (
                old["condition"] != condition
                or research_config(old["config"]) != research_config(cfg)
                or old["family"] != family
            ):
                raise ValueError(
                    "Resume requires identical condition, family and configuration; start a fresh run for a new observation, policy, curriculum or reward profile"
                )
            if old["learner_seed"] != learner_seed or old["validation_limit"] != validation_limit:
                raise ValueError("Resume learner seed differs")
            model.set_env(env)
            if simulator(cfg) == "cuda":
                from .cuda_ppo import configure_tensor_buffer

                configure_tensor_buffer(model, cfg["conditions"][condition]["masked"])
            else:
                configure_rollout_buffer(
                    model,
                    cfg["conditions"][condition]["masked"],
                    runtime_settings(cfg)["cache_rollout_on_device"],
                )
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
            except TrainingDeadline:
                progress.emit(
                    "Pilot deadline reached at a completed PPO update; saving partial run",
                    force=True,
                )
                callback._on_training_end()
        else:
            # Recover an interrupted final evaluation without collecting extra data.
            callback.init_callback(model)
            callback.initial_steps = model.num_timesteps
            callback.initial_games = getattr(model, "training_games", 0)
            if teaching_enabled(cfg) and family == "preset":
                callback.curriculum = CurriculumState(**getattr(model, "curriculum_state", {}))
            callback.validate()
        model.save(output / "final.zip")
        final_status = {
            **callback.snapshot(),
            "state": "complete",
            "steps": model.num_timesteps,
            "wall_seconds": perf_counter() - started,
            "budget_stopped": callback.budget_stopped,
            "budget_complete": progress_value(cfg, model) >= budget_target(cfg),
            "extra_games_in_final_rollout": max(0, progress_value(cfg, model) - budget_target(cfg))
            if game_budget
            else 0,
        }
        training_complete = True
        progress.emit("Saved final.zip; optimization and validation finished", force=True)
        write_json(output / "status.json", final_status)
        # Release worker processes before rendering. Exports never update the policy.
        env.close()
        env = None
        visual = None
        if settings["visualization"]["enabled"] and (output / "best.zip").exists():
            from .visualization import visualize_run

            write_json(output / "status.json", {**final_status, "state": "exporting"})
            visual = visualize_run(
                output, cfg=cfg, videos=settings["visualization"]["videos"], progress=progress
            )
            final_status.update(
                visualization_state=visual["state"], export_seconds=visual.get("export_seconds", 0)
            )
        write_json(
            output / "status.json",
            final_status,
        )
        progress.phase(Phase.COMPLETE, f"Training complete: {output.resolve()}")
        callback.refresh_report()
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
            model.save(output / "interrupted.zip")
        write_json(
            output / "status.json",
            {
                "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": repr(exc),
                "steps": model.num_timesteps if model else 0,
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
