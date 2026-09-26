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
from pvz_game.cuda.schema import MOWER_STATES, ZOMBIE_STATES

from pvz_rl.learning.budget import uses_games


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
    result["training"] = copy.deepcopy(result["training"])
    result["training"].get("performance", {}).pop("token_cache_gib", None)
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


def validate_config(cfg: dict) -> None:
    if cfg["training"].get("validation_schedule", "periodic") not in (
        "periodic",
        "stage_success",
    ):
        raise ValueError("training.validation_schedule must be periodic or stage_success")
    if simulator(cfg) not in ("cpu", "cuda"):
        raise ValueError("simulation.backend must be cpu or cuda")
    if cfg["encoding"].get("version") != "event_v7" or cfg.get("policy", {}).get("kind") not in (
        "event_sequential_q_v1",
    ):
        raise ValueError(
            "Retired observation/policy format. Start fresh with configs/train.toml; archived reports remain readable; recordings must use the 100 Hz engine."
        )
    if cfg["reward"].get("version") != "net_value_v1":
        raise ValueError(
            "Retired reward format; 0.11.0 requires fresh training with configs/train.toml"
        )
    reward_keys = {
        "version",
        "win_reward",
        "loss_penalty",
        "mower_value",
        "basic_zombie_value",
        "progress_weight",
        "value_scale",
    }
    if set(cfg["reward"]) != reward_keys:
        raise ValueError("reward must contain only the outcome and net-value settings")
    if "actor_objective" in cfg["training"] or "ent_coef" in cfg["training"]:
        raise ValueError("Retired PPO objective; use sequential Q fitting")
    if cfg["training"]["exploration"].get("objective") != "sequential_plant_epsilon_v1":
        raise ValueError("Only conditional planting exploration is supported")
    for key in (
        "win_reward",
        "loss_penalty",
        "mower_value",
        "basic_zombie_value",
        "progress_weight",
    ):
        value = cfg["reward"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"reward.{key} must be finite and nonnegative")
    for group, keys in (
        ("reward", ("value_scale",)),
        ("encoding", ("local_count_scale",)),
        ("training", ("max_grad_norm",)),
    ):
        for key in keys:
            value = cfg[group][key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{group}.{key} must be finite and positive")
    policy = cfg["policy"]
    memory = policy.get("memory", {})
    for key in (
        "model_width",
        "layers",
        "heads",
        "feedforward_width",
        "local_tokens",
        "event_tokens",
        "summary_tokens",
        "summary_stride",
    ):
        if type(memory.get(key)) is not int or memory[key] < 1:
            raise ValueError(f"policy.memory.{key} must be a positive integer")
    if memory["model_width"] % memory["heads"]:
        raise ValueError("Memory model_width must be divisible by heads")
    if not math.isfinite(memory.get("gate_bias", float("nan"))):
        raise ValueError("Memory gate_bias must be finite")
    for key in ("plant_embedding", "state_embedding"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f"policy.{key} must be a positive integer")
    for key in ("scalar_sizes", "channels"):
        if not policy[key] or any(type(n) is not int or n < 1 for n in policy[key]):
            raise ValueError(f"policy.{key} must contain positive integers")
    sample_seconds = output_settings(cfg)["logging"]["hardware_sample_seconds"]
    if (
        type(sample_seconds) not in (int, float)
        or not math.isfinite(sample_seconds)
        or sample_seconds < 0.1
    ):
        raise ValueError("hardware_sample_seconds must be finite and at least 0.1")
    exploration = cfg["training"]["exploration"]
    for key in ("epsilon_start", "epsilon_floor"):
        value = exploration.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"exploration.{key} must be finite and in (0, 1]")
    if exploration["epsilon_floor"] > exploration["epsilon_start"]:
        raise ValueError("exploration floor exceeds start")
    if type(exploration.get("decay_games")) is not int or exploration["decay_games"] < 1:
        raise ValueError("exploration.decay_games must be a positive integer")
    minutes = cfg["training"].get("max_minutes")
    reserve = cfg["training"].get("finalization_minutes", 15)
    if minutes is not None and (
        type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0
    ):
        raise ValueError("max_minutes must be finite and positive")
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("finalization_minutes must be finite and nonnegative")
    c = cfg["curriculum"]
    unlimited = cfg["training"].get("until_stage_complete", False)
    if type(unlimited) is not bool:
        raise ValueError("training.until_stage_complete must be a boolean")
    if unlimited and (not uses_games(cfg) or c.get("mode") != "teaching" or not c.get("run_stage")):
        raise ValueError(
            "until_stage_complete requires a selected teaching stage and game-based scheduling"
        )
    if unlimited and not c.get("stages", {}).get(c.get("run_stage"), {}).get("requirements"):
        raise ValueError("until_stage_complete requires a stage with mastery requirements")
    if c.get("mode", "fixed") not in ("fixed", "teaching"):
        raise ValueError("Unsupported curriculum mode")
    if c.get("run_stage") is not None:
        from pvz_rl.learning.curriculum import STAGES

        if c.get("mode") != "teaching" or c["run_stage"] not in STAGES:
            raise ValueError("curriculum.run_stage requires a teaching recipe and a valid stage")
    if c.get("mode") == "teaching":
        from pvz_rl.learning.curriculum import STAGES

        for lesson in lesson_settings(cfg).values():
            if type(lesson.get("natural_sun")) is not bool:
                raise ValueError(
                    "lesson.natural_sun must be an explicit boolean; use configs/train.toml "
                    "with --init-from to adopt the current lesson trial"
                )
            lanes = lesson.get("lanes_per_spawn", 1)
            if type(lanes) is not int or not 1 <= lanes <= cfg["environment"]["rows"]:
                raise ValueError("lesson.lanes_per_spawn must be an integer within the board")
            if (
                type(lesson["initial_sun"]) is not int
                or lesson["initial_sun"] < 0
                or not lesson["spawn_ticks"]
                or any(type(t) is not int or t < 1 for t in lesson["spawn_ticks"])
                or not lesson["allowed_plants"]
                or not set(lesson["allowed_plants"]) <= set(PLANT_TYPES)
            ):
                raise ValueError("Invalid lesson sun, spawn ticks or allowed plants")
        schedule = (
            ("probe_interval_games", "minimum_stage_games")
            if uses_games(cfg)
            else ("probe_interval", "minimum_stage_steps")
        )
        for key in (*schedule, "probe_cases", "consecutive_passes"):
            if type(c[key]) is not int or c[key] < 1:
                raise ValueError(f"curriculum.{key} must be a positive integer")
        curriculum_probe_seeds(cfg)
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
                or not set(stage["requirements"]) <= set(stage["tasks"])
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
    if any(type(visual[key]) is not bool for key in ("enabled", "demos", "videos", "live_enabled")):
        raise ValueError("Visualization enabled/demos/videos/live_enabled must be booleans")
    if type(visual["live_fps"]) is not int or not 1 <= visual["live_fps"] <= 30:
        raise ValueError("live_fps must be an integer from 1 to 30")
    if type(visual["live_history_ram_mib"]) is not int or visual["live_history_ram_mib"] < 0:
        raise ValueError("live_history_ram_mib must be a nonnegative integer")
    size = visual["live_window_size"]
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(type(n) is not int or n < lower for n, lower in zip(size, (640, 480)))
    ):
        raise ValueError("live_window_size must be at least 640 by 480")
    if type(visual["video_fps"]) is not int or not 1 <= visual["video_fps"] <= 100:
        raise ValueError("video_fps must be an integer from 1 to 100")
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
    if train.get("method") != "sequential_q_mc_v1":
        raise ValueError("Training requires sequential_q_mc_v1 and fresh models")
    if any(
        k in train
        for k in (
            "role_phase_games",
            "critic_learning_rate",
            "clip_range",
            "normalize_advantage",
            "target_kl",
            "pipeline",
            "rollout_size",
            "rollout_steps_per_env",
            "gae_lambda",
            "critic_warmup_games",
        )
    ):
        raise ValueError("Short rollouts, GAE, periodic scheduling and actor warm-up were removed")
    storage = train.get("storage", {})
    for key in ("ram_gib", "block_rows"):
        if type(storage.get(key)) is not int or storage[key] < 1:
            raise ValueError(f"training.storage.{key} must be a positive integer")
    performance = train.get("performance", {})
    cache = performance.get("token_cache_gib", 2)
    if type(cache) not in (int, float) or not math.isfinite(cache) or cache < 0:
        raise ValueError("training.performance.token_cache_gib must be finite and nonnegative")
    for key in ("compile_kernels", "telemetry"):
        if type(performance.get(key, False)) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
    if train.get("budget_unit", "decisions") not in ("games", "decisions"):
        raise ValueError("training.budget_unit must be games or decisions")
    if env.get("action_timing", "fixed") not in ("fixed", "per_tick"):
        raise ValueError("Unsupported environment.action_timing")
    if env.get("action_timing") == "per_tick" and env["decision_ticks"] != 1:
        raise ValueError("per_tick action timing requires decision_ticks = 1")
    if cfg["schema_version"] != 1 or (env["rows"], env["cols"], env["bins"]) != (5, 9, 3):
        raise ValueError("Unsupported research schema/board")
    if tuple(env["plants"]) != PLANT_TYPES or tuple(env["zombies"]) != ZOMBIE_TYPES:
        raise ValueError("Plant/zombie order is fixed by observation and action schema version 1")
    for key, expected in (
        (
            "plant_states",
            ("ready", "arming", "armed", "fusing", "digesting", "exploding", "detonating"),
        ),
        ("zombie_states", ZOMBIE_STATES),
        ("mower_states", MOWER_STATES),
    ):
        if tuple(env[key]) != tuple(expected):
            raise ValueError(f"{key} must match the pinned categorical encoding order")
    for key in ("decision_ticks", "cutoff_seconds"):
        if type(env[key]) is not int or env[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "total_games" if uses_games(cfg) else "total_steps",
        "n_envs",
        "batch_size",
        "n_epochs",
        "eval_interval_games" if uses_games(cfg) else "eval_interval",
        "torch_threads",
    ):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if train.get("discount_clock") != "simulation_ticks" or train["gamma"] != 1:
        raise ValueError("Complete actual returns require gamma 1; start fresh")
    if train["batch_size"] > 1024:
        raise ValueError("Bounded minibatches must not exceed 1024")
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
        ("training", ("learning_rate",)),
    ):
        for key in keys:
            if not math.isfinite(cfg[group][key]) or cfg[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be finite and positive")
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


def curriculum_probe_seeds(cfg: dict) -> list[int]:
    """New mastery cases are separate; saved recipes retain their original seeds."""
    split = "curriculum" if "curriculum" in cfg["splits"] else "validation"
    return seed_values(cfg, split, cfg["curriculum"]["probe_cases"])
