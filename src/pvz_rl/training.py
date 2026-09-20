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

from .config import output_settings, research_config, seed_values, validate_config
from .env import PvZEnv
from .evaluation import evaluate, summarize
from .progress import Phase, ProgressReporter, duration
from .provenance import append_jsonl, file_hash, metadata, verify_engine, write_json
from .scenarios import difficulty_weights


def make_env(cfg, condition, worker_seed, family):
    return Monitor(
        PvZEnv(cfg, condition=condition, training=True, worker_seed=worker_seed, family=family)
    )


def vector_env(cfg, condition, learner_seed, family="preset"):
    factories = [
        partial(make_env, cfg, condition, learner_seed * 1000 + i, family)
        for i in range(cfg["training"]["n_envs"])
    ]
    return (
        DummyVecEnv(factories)
        if len(factories) == 1
        else SubprocVecEnv(factories, start_method="spawn")
    )


def build_model(cfg, condition, env, seed, log_dir=None):
    t = cfg["training"]
    algorithm = MaskablePPO if cfg["conditions"][condition]["masked"] else PPO
    return algorithm(
        "MlpPolicy",
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
    )


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
    ):
        super().__init__()
        self.cfg, self.condition, self.learner_seed = cfg, condition, learner_seed
        self.output, self.validation_limit, self.family = Path(output), validation_limit, family
        self.best_score = -math.inf
        self.next_eval = cfg["training"]["eval_interval"]
        self.last_eval = -1
        self.started = perf_counter()
        self.initial_steps = 0
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
        t = cfg["training"]
        self.target = math.ceil(t["total_steps"] / t["rollout_size"]) * t["rollout_size"]

    def snapshot(self):
        elapsed = perf_counter() - self.started
        active = max(elapsed - self.eval_seconds - self.report_seconds, 1e-9)
        collected = self.model.num_timesteps - self.initial_steps
        rate = collected / active
        rows = list(self.recent)
        decisions = sum(r["decisions"] for r in rows)
        weights = difficulty_weights(
            self.cfg,
            self.model.num_timesteps,
            self.cfg["training"]["total_steps"],
            self.cfg["conditions"][self.condition]["curriculum"],
        )
        return {
            "state": self.progress.state.value,
            "steps": self.model.num_timesteps,
            "training_steps": self.model.num_timesteps,
            "target_steps": self.target,
            "wall_seconds": elapsed,
            "validation_seconds": self.eval_seconds,
            "report_seconds": self.report_seconds,
            "decisions_per_second": rate,
            "end_to_end_decisions_per_second": collected / max(elapsed, 1e-9),
            "estimated_remaining_training_seconds": (
                max(0, self.target - self.model.num_timesteps) / rate if rate > 0 else None
            ),
            "next_episode_difficulty_weights": weights,
            "rolling_episodes": len(rows),
            "rolling_win_rate": sum(r["win"] for r in rows) / len(rows) if rows else None,
            "rolling_return": sum(r["return"] for r in rows) / len(rows) if rows else None,
            "rolling_seconds": (
                sum(r["simulated_seconds"] for r in rows) / len(rows) if rows else None
            ),
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
        self.progress.emit(
            f"{row['steps']:,}/{self.target:,} decisions ({row['steps'] / self.target:.1%}); "
            f"{row['decisions_per_second']:.0f} decisions/s; elapsed {duration(row['wall_seconds'])}; "
            f"training ETA {duration(row['estimated_remaining_training_seconds'])}; "
            f"last {len(self.recent)} games win {win}, reward {reward}; best validation {best}",
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
        self.next_eval = (
            (self.initial_steps // self.cfg["training"]["eval_interval"]) + 1
        ) * self.cfg["training"]["eval_interval"]
        self.training_env.env_method("set_progress", self.initial_steps)
        self.stream = (self.output / "training-episodes.jsonl").open("w", encoding="utf-8")
        self.metrics_stream = (self.output / "training-metrics.jsonl").open("w", encoding="utf-8")
        self.last_updates = self.model._n_updates
        self.progress.phase(Phase.COLLECTING)
        self.log_progress(force=True)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_metrics" in info:
                self.recent.append(info["episode_metrics"])
                append_jsonl(
                    self.stream,
                    {
                        **info["episode_metrics"],
                        "policy": self.condition,
                        "learner_seed": self.learner_seed,
                        "training_steps": self.num_timesteps,
                    },
                )
        weights = difficulty_weights(
            self.cfg,
            self.num_timesteps,
            self.cfg["training"]["total_steps"],
            self.cfg["conditions"][self.condition]["curriculum"],
        )
        if weights != self.last_weights:
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
        self.capture_update()
        if self.model.num_timesteps >= self.next_eval:
            self.validate()
            while self.next_eval <= self.model.num_timesteps:
                self.next_eval += self.cfg["training"]["eval_interval"]
        self.progress.phase(Phase.COLLECTING)

    def _on_rollout_end(self):
        self.progress.phase(Phase.UPDATING)
        write_json(self.output / "status.json", self.snapshot())
        self.log_progress()

    def validate(self):
        started = perf_counter()
        self.progress.phase(
            Phase.VALIDATING, f"Validation at {self.model.num_timesteps:,} decisions"
        )
        write_json(self.output / "status.json", self.snapshot())
        rows = evaluate(
            self.cfg,
            policy=self.model,
            condition=self.condition,
            learner_seed=self.learner_seed,
            seeds=seed_values(self.cfg, "validation", self.validation_limit),
            levels=(["easy"] if self.family == "diagnostic" else self.cfg["evaluation"]["levels"]),
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
                    "wall_seconds": perf_counter() - self.started,
                    "macro_win_rate": score,
                    "levels": scores,
                },
            )
        write_json(self.output / "status.json", self.snapshot())
        self.refresh_report()

    def _on_training_end(self):
        self.capture_update()
        if self.last_eval != self.model.num_timesteps:
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
    torch.set_num_threads(cfg["training"]["torch_threads"])
    model = algorithm.load(checkpoint, device=device)
    return model, data


def train(
    cfg, condition, learner_seed, output, *, validation_limit=None, family="preset", resume=None
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
    nominal_steps = cfg["training"]["total_steps"]
    rollout = cfg["training"]["rollout_size"]
    effective_steps = math.ceil(nominal_steps / rollout) * rollout
    output.mkdir(parents=True, exist_ok=False)
    details = metadata(
        cfg,
        condition=condition,
        learner_seed=learner_seed,
        family=family,
        nominal_steps=nominal_steps,
        effective_steps=effective_steps,
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
        f"{cfg['training']['n_envs']} workers; {effective_steps:,} decisions; "
        f"family {family}; output {output.resolve()}",
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
                raise ValueError("Resume requires identical condition, family and configuration")
            if old["learner_seed"] != learner_seed or old["validation_limit"] != validation_limit:
                raise ValueError("Resume learner seed differs")
            model.set_env(env)
            model.tensorboard_log = str(output / "tensorboard")
            details["resume_steps"] = model.num_timesteps
            write_json(output / "metadata.json", details)
            progress.emit(
                f"Resumed shared policy and optimizer at {model.num_timesteps:,} decisions",
                force=True,
            )
        else:
            model = build_model(cfg, condition, env, learner_seed, output / "tensorboard")
        remaining = effective_steps - model.num_timesteps
        if remaining < 0:
            raise ValueError("Checkpoint exceeds this run's training budget")
        callback = ResearchCallback(
            cfg, condition, learner_seed, output, validation_limit, family, progress=progress
        )
        if resume:
            previous = Path(resume).resolve().parent
            if (previous / "best.json").exists():
                best = json.loads((previous / "best.json").read_text("utf-8"))
                if best["steps"] <= model.num_timesteps:
                    shutil.copy2(previous / "best.zip", output / "best.zip")
                    shutil.copy2(previous / "best.json", output / "best.json")
                    callback.best_score = best["macro_win_rate"]
        env.env_method("set_progress", model.num_timesteps)
        if remaining:
            model.learn(
                remaining,
                callback=callback,
                reset_num_timesteps=not bool(resume),
                tb_log_name=condition,
            )
        else:
            # Recover an interrupted final evaluation without collecting extra data.
            callback.init_callback(model)
            callback.initial_steps = model.num_timesteps
            callback.validate()
        model.save(output / "final.zip")
        final_status = {
            **callback.snapshot(),
            "state": "complete",
            "steps": model.num_timesteps,
            "wall_seconds": perf_counter() - started,
        }
        training_complete = True
        progress.emit("Saved final.zip; optimization and validation finished", force=True)
        write_json(output / "status.json", final_status)
        # Release worker processes before rendering. Exports never update the policy.
        env.close()
        env = None
        visual = None
        if settings["visualization"]["enabled"]:
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
