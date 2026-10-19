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
from pvz_game.cuda.schema import MOWER_STATES, PLANT_STATES, ZOMBIE_STATES

from pvz_rl.learning.budget import uses_games

_CONFIG_RESOURCE = "data/train.toml"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def _bundled_config() -> dict:
    return tomllib.loads(files("pvz_rl").joinpath(_CONFIG_RESOURCE).read_text("utf-8"))


def output_settings(cfg: dict) -> dict:
    """Read the resolved current output configuration without version fallbacks."""
    return {key: cfg[key] for key in ("logging", "visualization")}


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
    return result


def runtime_settings(cfg: dict) -> dict:
    return cfg["runtime"]


def simulator(cfg):
    return cfg["simulation"]["backend"]


@lru_cache(maxsize=1)
def gpu_defaults():
    return copy.deepcopy(_bundled_config()["profiles"]["gpu_defaults"])


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _remove_path(target: dict, path: str) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        current = current.get(part, {})
    current.pop(parts[-1], None)


def _read_config(path: str | Path | None) -> dict:
    if path is None:
        return _bundled_config()
    return tomllib.loads(Path(path).read_text("utf-8"))


def _resolve_profile(raw: dict, profile: str) -> dict:
    raw = copy.deepcopy(raw)
    profiles = raw.pop("profiles", {})
    for catalogue in ("gpu_profiles", "learning_profiles", "tuning"):
        raw.pop(catalogue, None)
    selected = profiles.get(profile)
    if selected is None:
        if not profiles:
            return raw
        if profile != "train":
            raise ValueError(f"Configuration profile {profile!r} is not available")
        return raw
    selected = copy.deepcopy(selected)
    removals = selected.pop("remove", [])
    _deep_update(raw, selected)
    for path in removals:
        _remove_path(raw, path)
    return raw


def load_config(path: str | Path | None = None, *, profile: str = "train") -> dict:
    cfg = _resolve_profile(_read_config(path), profile)
    validate_config(cfg)
    return copy.deepcopy(cfg)


def load_demo_config(path: str | Path | None = None, *, profile: str = "demo") -> dict:
    """Recording verification profile; execution options are supplied separately."""
    cfg = load_config(path, profile=profile)
    if (
        cfg["encoding"]["version"] != "entity_v1"
        or cfg["policy"]["kind"] != "two_stage_transformer_q_v1"
    ):
        raise ValueError("Human demonstrations require the stateless two-stage Transformer profile")
    return copy.deepcopy(cfg)


def validate_config(cfg: dict) -> None:
    objective = cfg["training"].get("objective")
    if not isinstance(objective, dict):
        raise ValueError("training.objective must describe the stateless two-stage objective")
    if objective is not None:
        objective_keys = {
            "protocol",
            "accepted_outcome_share",
            "species_consistency_weight",
            "species_consistency_huber_delta",
            "invalid_action_target",
            "invalid_fit_group_size",
            "dense_multiplier",
            "ema_decay",
            "demo_rank_loss_weight",
            "demo_rank_margin",
        }
        if set(objective) != objective_keys:
            raise ValueError(
                "training.objective contains retired or missing settings, including "
                "species_consistency_huber_delta"
            )
        if objective.get("protocol") != "complete_return_selective_invalid_v3":
            raise ValueError(
                "Unsupported training objective protocol; fresh initialization is required"
            )
        if objective.get("accepted_outcome_share") != 0.5:
            raise ValueError("objective.accepted_outcome_share must be 0.5")
        if (
            type(objective.get("invalid_action_target")) not in (int, float)
            or objective["invalid_action_target"] != -1.0
        ):
            raise ValueError("objective.invalid_action_target must be -1.0")
        if (
            type(objective.get("invalid_fit_group_size")) is not int
            or objective["invalid_fit_group_size"] < 1
        ):
            raise ValueError("objective.invalid_fit_group_size must be a positive integer")
        consistency_weight = objective.get("species_consistency_weight")
        if (
            type(consistency_weight) not in (int, float)
            or not math.isfinite(consistency_weight)
            or consistency_weight < 0
        ):
            raise ValueError("objective.species_consistency_weight must be finite and nonnegative")
        consistency_delta = objective.get("species_consistency_huber_delta")
        if (
            type(consistency_delta) not in (int, float)
            or not math.isfinite(consistency_delta)
            or consistency_delta <= 0
        ):
            raise ValueError(
                "objective.species_consistency_huber_delta must be finite and positive"
            )
        for key in ("dense_multiplier", "demo_rank_margin"):
            if (
                not isinstance(objective.get(key), (int, float))
                or not math.isfinite(objective[key])
                or objective[key] <= 0
            ):
                raise ValueError(f"objective.{key} must be finite and positive")
        for key in ("demo_rank_loss_weight", "ema_decay"):
            if (
                not isinstance(objective.get(key), (int, float))
                or not math.isfinite(objective[key])
                or not 0 <= objective[key] < 1
            ):
                raise ValueError(f"objective.{key} must be finite and in [0, 1)")
    penalties = cfg["reward"].get("home_entry_penalties", [0, 0])
    if (
        len(penalties) != 2
        or any(not math.isfinite(p) or p < 0 for p in penalties)
        or penalties[1] < penalties[0]
    ):
        raise ValueError("home_entry_penalties require two increasing nonnegative magnitudes")
    weight = cfg["reward"].get("win_time_weight", 0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("win_time_weight must be finite and nonnegative")
    current_policy = (
        cfg.get("encoding", {}).get("version") == "entity_v1"
        and cfg.get("policy", {}).get("kind") == "two_stage_transformer_q_v1"
    )
    if cfg["training"].get("validation_schedule", "periodic") not in (
        "periodic",
        "stage_success",
    ):
        raise ValueError("training.validation_schedule must be periodic or stage_success")
    if simulator(cfg) not in ("cpu", "cuda"):
        raise ValueError("simulation.backend must be cpu or cuda")
    if not current_policy:
        raise ValueError(
            "Retired observation/policy format; entity_v1 requires fresh initialization"
        )
    if cfg["reward"].get("version") != "net_value_v1":
        raise ValueError(
            "Retired reward format; 0.11.0 requires fresh training with the bundled train profile"
        )
    reward_keys = {
        "version",
        "win_reward",
        "loss_penalty",
        "mower_value",
        "basic_zombie_value",
        "progress_weight",
        "value_scale",
        "early_sun_extra_multiplier",
        "early_sun_spawn_fraction",
    }
    if set(cfg["reward"]) not in (
        reward_keys,
        reward_keys | {"home_entry_penalties", "win_time_weight"},
    ):
        raise ValueError("reward must contain the outcome and net-value settings")
    if "actor_objective" in cfg["training"] or "ent_coef" in cfg["training"]:
        raise ValueError("Retired PPO objective; use sequential Q fitting")
    if cfg["training"]["exploration"].get("objective") != "committed_plant_tile_epsilon_v1":
        raise ValueError("Only committed plant and conditional tile exploration is supported")
    for key in (
        "win_reward",
        "loss_penalty",
        "mower_value",
        "basic_zombie_value",
        "progress_weight",
        "early_sun_extra_multiplier",
    ):
        value = cfg["reward"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"reward.{key} must be finite and nonnegative")
    fraction = cfg["reward"]["early_sun_spawn_fraction"]
    if (
        not isinstance(fraction, (list, tuple))
        or len(fraction) != 2
        or any(type(value) is not int or value < 1 for value in fraction)
        or fraction[0] > fraction[1]
    ):
        raise ValueError(
            "reward.early_sun_spawn_fraction must be positive numerator/denominator <= 1"
        )
    for group, keys in (
        ("reward", ("value_scale",)),
        ("training", ("max_grad_norm",)),
    ):
        for key in keys:
            value = cfg[group][key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{group}.{key} must be finite and positive")
    if type(cfg["encoding"].get("max_entities")) is not int or cfg["encoding"]["max_entities"] < 5:
        raise ValueError("encoding.max_entities must be an integer of at least 5")
    policy = cfg["policy"]
    model_settings = {
        "species_width": 128,
        "species_layers": 4,
        "species_heads": 8,
        "species_feedforward": 512,
        "tile_width": 192,
        "tile_layers": 6,
        "tile_heads": 12,
        "tile_feedforward": 768,
    }
    execution_settings = (
        "encoder_microbatch",
        "attention_query_chunk",
        "encoder_token_budget",
    )
    if set(policy) != set(model_settings) | set(execution_settings) | {
        "kind",
        "action_distribution",
    }:
        raise ValueError(
            "policy must contain only the stateless two-stage model and execution settings"
        )
    if policy["action_distribution"] != "sequential_q_unmasked_penalty_v1":
        raise ValueError("Unsupported sequential action-value distribution")
    for key in (
        *model_settings,
        *execution_settings,
    ):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError(f"policy.{key} must be a positive integer")
    for key, expected in model_settings.items():
        if policy[key] != expected:
            raise ValueError(
                f"policy.{key} must be {expected} for the two-stage Transformer contract"
            )
    sample_seconds = output_settings(cfg)["logging"]["hardware_sample_seconds"]
    if (
        type(sample_seconds) not in (int, float)
        or not math.isfinite(sample_seconds)
        or sample_seconds < 0.1
    ):
        raise ValueError("hardware_sample_seconds must be finite and at least 0.1")
    exploration = cfg["training"]["exploration"]
    for key in ("epsilon_start", "epsilon_floor", "plant_epsilon_start", "plant_epsilon_floor"):
        value = exploration.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"exploration.{key} must be finite and in (0, 1]")
    if exploration["epsilon_floor"] > exploration["epsilon_start"]:
        raise ValueError("exploration floor exceeds start")
    if exploration["plant_epsilon_floor"] > exploration["plant_epsilon_start"]:
        raise ValueError("plant exploration floor exceeds start")
    for key in ("decay_games", "plant_decay_games"):
        if type(exploration.get(key)) is not int or exploration[key] < 1:
            raise ValueError(f"exploration.{key} must be a positive integer")
    minutes = cfg["training"].get("max_minutes")
    reserve = cfg["training"].get("finalization_minutes", 15)
    if minutes is not None and (
        type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0
    ):
        raise ValueError("max_minutes must be finite and positive")
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("finalization_minutes must be finite and nonnegative")
    c = cfg["curriculum"]
    if c.get("lessons") or set(c.get("stages", {})) - {"easy", "standard", "shared"}:
        raise ValueError("Retired curriculum lesson; use easy, standard and shared stages")
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
                or any(task not in ("easy", "standard", "hard") for task in stage["tasks"])
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
        "refill_evaluation",
    } or any(type(value) is not bool for value in runtime.values()):
        raise ValueError("Runtime settings must be known booleans")
    output = output_settings(cfg)
    log, visual = output["logging"], output["visualization"]
    if not math.isfinite(log["progress_seconds"]) or log["progress_seconds"] <= 0:
        raise ValueError("Logging progress_seconds must be finite and positive")
    if not math.isfinite(log["terminal_progress_seconds"]) or log["terminal_progress_seconds"] <= 0:
        raise ValueError("Logging terminal_progress_seconds must be finite and positive")
    if log["terminal_mode"] != "compact":
        raise ValueError("Logging terminal_mode must be compact")
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
    expected_method = "complete_return_selective_invalid_v3"
    if train.get("method") != expected_method:
        raise ValueError("Training requires a supported complete-return method and fresh models")
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
    for key in ("compile_kernels", "telemetry"):
        if type(performance.get(key, False)) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
    if performance.get("fit_precision", "fp32") != "bf16_master":
        raise ValueError("training.performance.fit_precision must be bf16_master")
    if type(performance.get("prefetch", False)) is not bool:
        raise ValueError("training.performance.prefetch must be a boolean")
    if set(performance) - {
        "compile_kernels",
        "telemetry",
        "fit_precision",
        "prefetch",
        "fit_batch_groups",
        "fit_encoder_microbatch",
        "fit_species_microbatch",
        "fit_tile_microbatch",
        "fit_entity_buckets",
        "fit_token_budget",
        "collection_graph_cache_entries",
        "collection_vram_headroom_mib",
    }:
        raise ValueError("training.performance contains unsupported execution settings")
    fit_batch_groups = performance.get("fit_batch_groups", 4)
    if type(fit_batch_groups) is not int or not 1 <= fit_batch_groups <= 16:
        raise ValueError("training.performance.fit_batch_groups must be an integer from 1 to 16")
    for key, default in (
        ("collection_graph_cache_entries", 8),
        ("collection_vram_headroom_mib", 512),
    ):
        value = performance.get(key, default)
        if type(value) is not int or value < 0:
            raise ValueError(f"training.performance.{key} must be a nonnegative integer")
    for key in ("fit_encoder_microbatch", "fit_token_budget"):
        if type(performance.get(key)) is not int or performance[key] < 1:
            raise ValueError(f"training.performance.{key} must be a positive integer")
    for key in ("fit_species_microbatch", "fit_tile_microbatch"):
        if key in performance and (type(performance[key]) is not int or performance[key] < 1):
            raise ValueError(f"training.performance.{key} must be a positive integer")
    if performance.get("fit_entity_buckets", [8, 16, 32, 64, 128, 256]) != [
        8,
        16,
        32,
        64,
        128,
        256,
    ]:
        raise ValueError("Fitting buckets must cover 8,16,32,64,128,256 without truncation")
    if visual.get("viewer", "auto") not in ("auto", "desktop", "web", "off"):
        raise ValueError("visualization.viewer must be auto, desktop, web or off")
    viewer_port = visual.get("viewer_port", 0)
    if type(viewer_port) is not int or not 0 <= viewer_port <= 65535:
        raise ValueError("visualization.viewer_port must be an integer from 0 to 65535")
    if train.get("budget_unit") not in ("games", "decisions"):
        raise ValueError("training.budget_unit must be games or decisions")
    if env.get("action_timing") not in ("fixed", "per_tick"):
        raise ValueError("Unsupported environment.action_timing")
    if env.get("action_timing") == "per_tick" and env["decision_ticks"] != 1:
        raise ValueError("per_tick action timing requires decision_ticks = 1")
    if cfg["schema_version"] != 1 or (env["rows"], env["cols"]) != (5, 9):
        raise ValueError("Unsupported research schema/board")
    if tuple(env["plants"]) != PLANT_TYPES or tuple(env["zombies"]) != ZOMBIE_TYPES:
        raise ValueError("Plant/zombie order is fixed by observation and action schema version 1")
    for key, expected in (
        (
            "plant_states",
            PLANT_STATES,
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
    demo = train.get("demo", {})
    if not isinstance(demo, dict):
        raise ValueError("training.demo must be a table")
    for key in ("passes", "learner_seed"):
        if type(demo.get(key, 1)) is not int or demo.get(key, 1) < 1:
            raise ValueError(f"training.demo.{key} must be a positive integer")
    for key in ("learning_rate", "time_budget_minutes"):
        if (
            type(demo.get(key, 1.0)) not in (int, float)
            or not math.isfinite(demo.get(key, 1.0))
            or demo.get(key, 1.0) <= 0
        ):
            raise ValueError(f"training.demo.{key} must be finite and positive")
    annotation = train.get("annotation", {})
    if not isinstance(annotation, dict):
        raise ValueError("training.annotation must be a table")
    if set(annotation) - {
        "passes",
        "learning_rate",
        "batch_size",
        "margin",
        "learner_seed",
        "sampling_seed",
        "sampling_family",
        "sampling_max_steps",
        "sampling_max_attempts",
        "sampling_plant_interval_steps",
        "sampling_max_plants",
    }:
        raise ValueError("training.annotation contains unsupported settings")
    if type(annotation.get("passes", 0)) is not int or annotation.get("passes", 0) < 0:
        raise ValueError(
            "training.annotation.passes must be a nonnegative integer (0 means unlimited)"
        )
    if type(annotation.get("batch_size", 256)) is not int or annotation.get("batch_size", 256) < 1:
        raise ValueError("training.annotation.batch_size must be a positive integer")
    if (
        type(annotation.get("learner_seed", 101)) is not int
        or annotation.get("learner_seed", 101) < 0
    ):
        raise ValueError("training.annotation.learner_seed must be a nonnegative integer")
    sampling_seed = annotation.get("sampling_seed", "random")
    if sampling_seed != "random" and (type(sampling_seed) is not int or sampling_seed < 0):
        raise ValueError(
            "training.annotation.sampling_seed must be 'random' or a nonnegative integer"
        )
    if annotation.get("sampling_family", "preset") not in (
        "preset",
        "diagnostic",
        "redistributed",
        "faster",
        "concentrated",
    ):
        raise ValueError("training.annotation.sampling_family is unsupported")
    for key in ("sampling_max_steps", "sampling_max_attempts"):
        if (
            type(annotation.get(key, 20000 if key.endswith("steps") else 100)) is not int
            or annotation.get(key, 20000 if key.endswith("steps") else 100) < 1
        ):
            raise ValueError(f"training.annotation.{key} must be a positive integer")
    interval = annotation.get("sampling_plant_interval_steps", [500, 1500])
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(type(value) is not int or value < 1 for value in interval)
        or interval[0] > interval[1]
    ):
        raise ValueError(
            "training.annotation.sampling_plant_interval_steps must be ordered positive integers"
        )
    max_plants = annotation.get("sampling_max_plants", 8)
    if type(max_plants) is not int or not 1 <= max_plants < env["rows"] * env["cols"]:
        raise ValueError(
            "training.annotation.sampling_max_plants must leave at least one empty tile"
        )
    for key in ("learning_rate", "margin"):
        value = annotation.get(key, 3e-4 if key == "learning_rate" else 1.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.annotation.{key} must be finite and positive")
    if train["device"] not in ("cpu", "cuda"):
        raise ValueError("Training device must be cpu or cuda")
    seeds = train["learner_seeds"]
    if (
        not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Learner seeds must be nonnegative and distinct")
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
