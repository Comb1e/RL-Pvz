"""PPO training with validation-only checkpoint selection and resumable artifacts."""

from __future__ import annotations

import copy
import json
import math
import shutil
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import seed_values, validate_config
from .env import PvZEnv
from .evaluation import evaluate, summarize
from .provenance import append_jsonl, file_hash, metadata, verify_engine, write_json


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
        self, cfg, condition, learner_seed, output, validation_limit=None, family="preset"
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

    def _on_training_start(self):
        self.initial_steps = self.model.num_timesteps
        self.next_eval = (
            (self.initial_steps // self.cfg["training"]["eval_interval"]) + 1
        ) * self.cfg["training"]["eval_interval"]
        self.training_env.env_method("set_progress", self.initial_steps)
        self.stream = (self.output / "training-episodes.jsonl").open("w", encoding="utf-8")

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_metrics" in info:
                append_jsonl(
                    self.stream,
                    {
                        **info["episode_metrics"],
                        "policy": self.condition,
                        "learner_seed": self.learner_seed,
                        "training_steps": self.num_timesteps,
                    },
                )
        return True

    def _on_rollout_start(self):
        # This hook runs after the preceding PPO update. A step callback would
        # save pre-update weights while labeling them with the newly collected steps.
        if self.model.num_timesteps >= self.next_eval:
            self.validate()
            while self.next_eval <= self.model.num_timesteps:
                self.next_eval += self.cfg["training"]["eval_interval"]

    def _on_rollout_end(self):
        elapsed = perf_counter() - self.started
        write_json(
            self.output / "status.json",
            {
                "state": "training",
                "steps": self.num_timesteps,
                "target_steps": self.cfg["training"]["total_steps"],
                "wall_seconds": elapsed,
                "validation_seconds": self.eval_seconds,
                "decisions_per_second": (self.num_timesteps - self.initial_steps)
                / max(elapsed, 1e-9),
                "best_validation_win_rate": None
                if self.best_score == -math.inf
                else self.best_score,
            },
        )

    def validate(self):
        started = perf_counter()
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
        )
        scores = summarize(rows)
        score = sum(r["win_rate"] for r in scores.values()) / len(scores)
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
        self.model.save(self.output / "latest.zip")
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

    def _on_training_end(self):
        if self.last_eval != self.model.num_timesteps:
            self.validate()

    def close(self):
        if self.stream:
            self.stream.close()


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
        resume=str(resume) if resume else None,
    )
    write_json(output / "metadata.json", details)
    write_json(output / "config.json", cfg)
    write_json(output / "status.json", {"state": "starting"})
    env, model, callback = None, None, None
    started = perf_counter()
    try:
        env = vector_env(cfg, condition, learner_seed, family)
        if resume:
            model, old = load_policy(resume, cfg["training"]["device"])
            if old["condition"] != condition or old["config"] != cfg or old["family"] != family:
                raise ValueError("Resume requires identical condition, family and configuration")
            if old["learner_seed"] != learner_seed or old["validation_limit"] != validation_limit:
                raise ValueError("Resume learner seed differs")
            model.set_env(env)
            model.tensorboard_log = str(output / "tensorboard")
        else:
            model = build_model(cfg, condition, env, learner_seed, output / "tensorboard")
        remaining = effective_steps - model.num_timesteps
        if remaining < 0:
            raise ValueError("Checkpoint exceeds this run's training budget")
        callback = ResearchCallback(cfg, condition, learner_seed, output, validation_limit, family)
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
            callback.validate()
        model.save(output / "final.zip")
        write_json(
            output / "status.json",
            {
                "state": "complete",
                "steps": model.num_timesteps,
                "wall_seconds": perf_counter() - started,
                "best_validation_win_rate": callback.best_score,
                "validation_seconds": callback.eval_seconds,
            },
        )
        return output
    except BaseException as exc:
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
        raise
    finally:
        if callback:
            callback.close()
        if env:
            env.close()
