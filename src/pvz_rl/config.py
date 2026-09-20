"""One resolved configuration shared by training, evaluation, and metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from importlib.resources import files
from pathlib import Path

from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_config(path: str | Path | None = None) -> dict:
    source = Path(path) if path else files("pvz_rl").joinpath("data/research.toml")
    cfg = tomllib.loads(source.read_text("utf-8"))
    validate_config(cfg)
    return copy.deepcopy(cfg)


def validate_config(cfg: dict) -> None:
    env, train = cfg["environment"], cfg["training"]
    if cfg["schema_version"] != 1 or (env["rows"], env["cols"], env["bins"]) != (5, 9, 20):
        raise ValueError("Unsupported research schema/board")
    if tuple(env["plants"]) != PLANT_TYPES or tuple(env["zombies"]) != ZOMBIE_TYPES:
        raise ValueError("Plant/zombie order is fixed by observation and action schema version 1")
    for key in ("decision_ticks", "cutoff_seconds"):
        if type(env[key]) is not int or env[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "total_steps",
        "n_envs",
        "rollout_size",
        "batch_size",
        "n_epochs",
        "eval_interval",
        "torch_threads",
    ):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if train["rollout_size"] % train["n_envs"] or train["rollout_size"] % train["batch_size"]:
        raise ValueError("Rollout size must divide into complete workers and minibatches")
    if train["rollout_size"] < 2 or not 0 < cfg["reward"]["gamma"] <= 1:
        raise ValueError("Invalid rollout size or discount")
    if train["device"] not in ("cpu", "cuda"):
        raise ValueError("Training device must be cpu or cuda")
    seeds = train["learner_seeds"]
    if (
        not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Learner seeds must be nonnegative and distinct")
    if not train["hidden_sizes"] or any(
        type(n) is not int or n <= 0 for n in train["hidden_sizes"]
    ):
        raise ValueError("Hidden layer sizes must be positive integers")
    for group, keys in (
        ("encoding", ("count_scale", "wave_scale")),
        ("reward", ("sun_target", "flower_target")),
        ("training", ("learning_rate", "gae_lambda", "clip_range")),
    ):
        for key in keys:
            if not math.isfinite(cfg[group][key]) or cfg[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be finite and positive")
    for key in ("defeated_weight", "sun_weight", "flower_weight"):
        if not math.isfinite(cfg["reward"][key]) or cfg["reward"][key] < 0:
            raise ValueError("Potential weights must be finite and nonnegative")
    if train["gae_lambda"] > 1 or not math.isfinite(train["ent_coef"]) or train["ent_coef"] < 0:
        raise ValueError("Invalid GAE or entropy coefficient")
    if (
        cfg["evaluation"]["bootstrap_replicates"] < 1
        or cfg["evaluation"]["replays_per_outcome"] < 0
    ):
        raise ValueError("Invalid bootstrap/replay count")
    intervals = list(cfg["splits"].values())
    for i, (lo, hi) in enumerate(intervals):
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi:
            raise ValueError("Invalid seed interval")
        for other_lo, other_hi in intervals[i + 1 :]:
            if max(lo, other_lo) <= min(hi, other_hi):
                raise ValueError("Seed splits must be disjoint")
    for key in ("easy_weights", "intermediate_weights", "final_weights"):
        weights = cfg["curriculum"][key]
        if len(weights) != 3 or min(weights) < 0 or abs(sum(weights) - 1) > 1e-9:
            raise ValueError("Difficulty weights must sum to one")
    if not 0 < cfg["curriculum"]["boundaries"][0] < cfg["curriculum"]["boundaries"][1] < 1:
        raise ValueError("Curriculum boundaries must increase")


def seed_values(cfg: dict, split: str, limit: int | None = None) -> list[int]:
    lo, hi = cfg["splits"][split]
    if limit is not None and not 1 <= limit <= hi - lo + 1:
        raise ValueError("Seed count lies outside split")
    return list(range(lo, hi + 1 if limit is None else lo + limit))
