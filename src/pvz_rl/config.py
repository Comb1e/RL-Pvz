"""One resolved configuration shared by training, evaluation, and metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES

from .budget import uses_games


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def _output_defaults():
    bundled = tomllib.loads(files("pvz_rl").joinpath("data/research.toml").read_text("utf-8"))
    return {key: bundled[key] for key in ("logging", "visualization")}


def output_settings(cfg: dict) -> dict:
    """Optional output settings also work with pre-0.2.0 checkpoint configurations."""
    return {key: {**values, **cfg.get(key, {})} for key, values in _output_defaults().items()}


def research_config(cfg: dict) -> dict:
    """Output and data-transport preferences do not change the research protocol."""
    result = {
        key: value
        for key, value in cfg.items()
        if key not in ("logging", "visualization", "runtime")
    }
    # Optional timing instrumentation never changes compatibility. Backend does.
    result["simulation"] = {"backend": simulator(cfg)}
    return result


@lru_cache(maxsize=1)
def _runtime_defaults():
    return tomllib.loads(files("pvz_rl").joinpath("data/research.toml").read_text("utf-8"))[
        "runtime"
    ]


def runtime_settings(cfg: dict) -> dict:
    """Old configurations get current transport defaults without changing strategy."""
    return {**_runtime_defaults(), **cfg.get("runtime", {})}


def simulator(cfg):
    """Archived configurations without a backend field retain CPU simulation."""
    return cfg.get("simulation", {}).get("backend", "cpu")


@lru_cache(maxsize=1)
def gpu_defaults():
    return tomllib.loads(files("pvz_rl").joinpath("data/gpu-defaults.toml").read_text("utf-8"))


def resolve_rollout(cfg, *, per_env=None, total=None, new_cuda=False):
    """Resolve one explicit rollout contract and reject ambiguous overrides."""
    t = cfg["training"]
    steps = per_env if per_env is not None else t.get("rollout_steps_per_env")
    if new_cuda and steps is None:
        steps = 128
    if steps is not None:
        if type(steps) is not int or steps < 1:
            raise ValueError("rollout_steps_per_env must be a positive integer")
        derived = steps * t["n_envs"]
        if total is not None and total != derived:
            raise ValueError("--rollout-size conflicts with n_envs * rollout_steps_per_env")
        t.update(rollout_steps_per_env=steps, rollout_size=derived)
    elif total is not None:
        t["rollout_size"] = total


def load_config(path: str | Path | None = None) -> dict:
    source = Path(path) if path else files("pvz_rl").joinpath("data/research.toml")
    cfg = tomllib.loads(source.read_text("utf-8"))
    validate_config(cfg)
    return copy.deepcopy(cfg)


@lru_cache(maxsize=1)
def _teaching_defaults():
    return tomllib.loads(files("pvz_rl").joinpath("data/teaching.toml").read_text("utf-8"))


def lesson_settings(cfg=None):
    return (cfg or {}).get("curriculum", {}).get("lessons", _teaching_defaults()["lessons"])


def learning_profile(name, base=None):
    """Versioned recipes also available in installed wheels without a checkout."""
    if name not in ("baseline", "tactical", "grouped", "pure-rl"):
        raise ValueError("Unknown learning profile")
    cfg = copy.deepcopy(base or load_config())
    cfg["profile"] = name
    if name != "baseline":
        cfg["encoding"].update(
            version="tactical_v2", local_count_scale=5, firepower_scale=20, lane_count_scale=9
        )
    if name in ("grouped", "pure-rl"):
        cfg["policy"] = {"kind": "grouped_v1"}
    if name == "pure-rl":
        cfg["curriculum"].update(copy.deepcopy(_teaching_defaults()))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if simulator(cfg) not in ("cpu", "cuda"):
        raise ValueError("simulation.backend must be cpu or cuda")
    if "rollout_steps_per_env" in cfg["training"]:
        steps = cfg["training"]["rollout_steps_per_env"]
        if (
            type(steps) is not int
            or steps < 1
            or steps * cfg["training"]["n_envs"] != cfg["training"]["rollout_size"]
        ):
            raise ValueError("rollout_size must equal n_envs * rollout_steps_per_env")
    for key in (
        "plant_kill_weight",
        "mower_kill_weight",
        "damage_weight",
        "empty_mower_activation_penalty",
        "win_reward",
        "loss_penalty",
        "mower_sun_weight",
        "wall_nut_damage_weight",
        "empty_explosion_penalty",
        "offensive_plant_weight",
        "eaten_plant_penalty",
    ):
        value = cfg["reward"].get(key, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"reward.{key} must be finite and nonnegative")
    offensive = cfg["reward"].get("offensive_plants", [])
    if (
        not isinstance(offensive, list)
        or any(p not in PLANT_TYPES or p in ("sunflower", "wall_nut") for p in offensive)
        or len(set(offensive)) != len(offensive)
        or cfg["reward"].get("offensive_plant_weight", 0) > 0
        and not offensive
    ):
        raise ValueError("reward.offensive_plants must list distinct damaging plants")
    scale = cfg["reward"].get("mower_sun_scale", 300.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("reward.mower_sun_scale must be finite and positive")
    if type(cfg["reward"].get("normalize_kills", True)) is not bool:
        raise ValueError("reward.normalize_kills must be a boolean")
    potential_mode = cfg["reward"].get("potential_mode", "legacy")
    if potential_mode not in ("legacy", "plant_value"):
        raise ValueError("Unsupported reward.potential_mode")
    targets = (
        ("economy_scale",) if potential_mode == "plant_value" else ("sun_target", "flower_target")
    )
    weights = (
        ("economy_weight",) if potential_mode == "plant_value" else ("sun_weight", "flower_weight")
    )
    for key in targets + ("defeated_weight",) + weights:
        value = cfg["reward"].get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or key in targets
            and value == 0
        ):
            raise ValueError(
                f"Invalid reward.{key}: expected finite {'positive' if key in targets else 'nonnegative'} value"
            )
    encoding = cfg["encoding"]
    if encoding["version"] not in (1, "tactical_v2"):
        raise ValueError("Unsupported observation version; start a fresh compatible profile")
    if encoding["version"] == "tactical_v2":
        for key in ("local_count_scale", "firepower_scale", "lane_count_scale"):
            if not math.isfinite(encoding[key]) or encoding[key] <= 0:
                raise ValueError(f"encoding.{key} must be finite and positive")
    if cfg.get("policy", {}).get("kind", "flat") not in (
        "flat",
        "grouped_v1",
        "spatial_grouped_v2",
    ):
        raise ValueError("Unsupported policy kind")
    policy = cfg.get("policy", {})
    if "initial_dig_logit" in policy:
        value = policy["initial_dig_logit"]
        if (
            policy.get("kind") not in ("grouped_v1", "spatial_grouped_v2")
            or type(value) not in (int, float)
            or not math.isfinite(value)
        ):
            raise ValueError("policy.initial_dig_logit requires a grouped policy and finite value")
    if policy.get("kind") == "spatial_grouped_v2":
        if (
            encoding["version"] != "tactical_v2"
            or type(policy.get("channels")) is not int
            or policy["channels"] < 1
        ):
            raise ValueError("Spatial policy requires tactical_v2 and positive channels")
    exploration = cfg["training"].get("exploration", {"objective": "joint"})
    if cfg["training"].get("actor_objective", "all_steps") not in ("all_steps", "choice_points_v1"):
        raise ValueError("Unsupported training.actor_objective")
    if exploration.get("objective") not in ("joint", "balanced_heads_v1"):
        raise ValueError("Unsupported exploration objective")
    if exploration["objective"] == "balanced_heads_v1":
        if policy.get("kind") not in ("grouped_v1", "spatial_grouped_v2"):
            raise ValueError("Balanced exploration requires a grouped policy")
        for key in ("type_coef", "tile_coef"):
            value = exploration.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid exploration.{key}")
    minutes = cfg["training"].get("max_minutes")
    reserve = cfg["training"].get("finalization_minutes", 15)
    if minutes is not None and (
        type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0
    ):
        raise ValueError("max_minutes must be finite and positive")
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("finalization_minutes must be finite and nonnegative")
    c = cfg["curriculum"]
    if c.get("mode", "fixed") not in ("fixed", "teaching"):
        raise ValueError("Unsupported curriculum mode")
    if c.get("run_stage") is not None:
        from .curriculum import STAGES

        if c.get("mode") != "teaching" or c["run_stage"] not in STAGES:
            raise ValueError("curriculum.run_stage requires a teaching recipe and a valid stage")
    if c.get("mode") == "teaching":
        from .curriculum import STAGES

        schedule = (
            ("probe_interval_games", "minimum_stage_games")
            if uses_games(cfg)
            else ("probe_interval", "minimum_stage_steps")
        )
        for key in (*schedule, "probe_cases", "consecutive_passes"):
            if type(c[key]) is not int or c[key] < 1:
                raise ValueError(f"curriculum.{key} must be a positive integer")
        if c["probe_cases"] > cfg["splits"]["validation"][1] - cfg["splits"]["validation"][0] + 1:
            raise ValueError("Curriculum probes must fit the validation split")
        for name in STAGES:
            stage = c["stages"][name]
            if (
                len(stage["tasks"]) != len(stage["weights"])
                or min(stage["weights"]) < 0
                or abs(sum(stage["weights"]) - 1) > 1e-9
                or any(
                    task not in (*STAGES[:2], "easy", "standard", "hard") for task in stage["tasks"]
                )
                or any(
                    type(n) is not int or not 1 <= n <= c["probe_cases"]
                    for n in stage["requirements"].values()
                )
            ):
                raise ValueError("Invalid teaching distribution or probe threshold")
    runtime = runtime_settings(cfg)
    if not set(runtime) <= {
        "cache_legal_actions",
        "coalesce_masks",
        "cache_rollout_on_device",
        "refill_evaluation",
    } or any(type(value) is not bool for value in runtime.values()):
        raise ValueError(
            "Runtime settings must be known booleans (legacy transport flags are read-only compatibility settings)"
        )
    output = output_settings(cfg)
    log, visual = output["logging"], output["visualization"]
    if not math.isfinite(log["progress_seconds"]) or log["progress_seconds"] <= 0:
        raise ValueError("Logging progress_seconds must be finite and positive")
    if type(log["rolling_window"]) is not int or log["rolling_window"] < 1:
        raise ValueError("Logging rolling_window must be a positive integer")
    if any(type(visual[key]) is not bool for key in ("enabled", "demos", "videos")):
        raise ValueError("Visualization enabled/demos/videos must be booleans")
    size = visual["video_size"]
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(type(n) is not int or n < 2 or n % 2 for n in size)
    ):
        raise ValueError("Visualization video_size must contain two positive even dimensions")
    if not isinstance(visual["ffmpeg"], str):
        raise ValueError("Visualization ffmpeg must be an executable path or empty")
    if type(visual["crf"]) is not int or not 0 <= visual["crf"] <= 51:
        raise ValueError("Visualization crf must be an integer from 0 to 51")
    if not math.isfinite(visual["final_hold_seconds"]) or visual["final_hold_seconds"] < 0:
        raise ValueError("Visualization final_hold_seconds must be finite and nonnegative")
    env, train = cfg["environment"], cfg["training"]
    if train.get("budget_unit", "decisions") not in ("games", "decisions"):
        raise ValueError("training.budget_unit must be games or decisions")
    if env.get("action_timing", "fixed") not in ("fixed", "per_tick"):
        raise ValueError("Unsupported environment.action_timing")
    if env.get("action_timing") == "per_tick" and env["decision_ticks"] != 1:
        raise ValueError("per_tick action timing requires decision_ticks = 1")
    if cfg["schema_version"] != 1 or (env["rows"], env["cols"], env["bins"]) != (5, 9, 20):
        raise ValueError("Unsupported research schema/board")
    if tuple(env["plants"]) != PLANT_TYPES or tuple(env["zombies"]) != ZOMBIE_TYPES:
        raise ValueError("Plant/zombie order is fixed by observation and action schema version 1")
    for key in ("decision_ticks", "cutoff_seconds"):
        if type(env[key]) is not int or env[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "total_games" if uses_games(cfg) else "total_steps",
        "n_envs",
        "rollout_size",
        "batch_size",
        "n_epochs",
        "eval_interval_games" if uses_games(cfg) else "eval_interval",
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
        ("training", ("learning_rate", "gae_lambda", "clip_range")),
    ):
        for key in keys:
            if not math.isfinite(cfg[group][key]) or cfg[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be finite and positive")
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
