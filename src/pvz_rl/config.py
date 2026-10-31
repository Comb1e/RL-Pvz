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

from pvz_rl.envs.tile_scenarios import is_bomb_skill
from pvz_rl.learning.budget import uses_games

_CONFIG_RESOURCE = "data/train.toml"

# User-facing stage tables resolve once into the checkpoint-compatible runtime layout.
_STAGE_PARAMETERS = {
    "normal": {
        **{key: f"training.{key}" for key in (
            "n_envs", "tile_only_cohorts", "exploration", "validation_schedule",
            "eval_interval_games",
        )},
        "search_query_probability": "training.search.normal_query_probability",
        "curriculum": "curriculum",
    },
    "cherry_bomb": {
        "n_envs": "training.tile_scenarios.cherry_bomb_n_envs",
        "initial_sun": "training.tile_scenarios.bomb_initial_sun",
        "plants": "training.tile_scenarios.bomb_plants",
        "obstacle_plants": "training.tile_scenarios.bomb_obstacle_plants",
        "mower_ready_probability": "training.tile_scenarios.bomb_mower_ready_probability",
        **{key: f"training.tile_scenarios.{key}" for key in (
            "ranking_objective", "ranking_margin", "distance_weight", "distance_temperature",
        )},
        **{key: f"training.performance.cherry_{key}" for key in (
            "prepare_workers", "worker_ram_mib", "prefetch", "oracle", "cuda_warmup",
            "oracle_lanes", "oracle_vram_mib",
        )},
        "probe_interval_games": "curriculum.cherry_bomb_probe_interval_games",
    },
    "peashooter": {
        "sky_sun_rate": "training.tile_scenarios.peashooter_sky_sun_rate",
        "search_query_probability": "training.search.peashooter_query_probability",
        "probe_interval_games": "curriculum.peashooter_probe_interval_games",
    },
}


def _resolve_training_sections(cfg):
    """Accept grouped tuning tables or legacy flat inputs, rejecting duplicate sources."""
    training = cfg.get("training", {})
    for name, fields in _STAGE_PARAMETERS.items():
        section = training.pop(name, None)
        if section is None:
            continue
        if not isinstance(section, dict) or set(section) - fields.keys():
            raise ValueError(f"Unsupported settings in training.{name}")
        for key, value in section.items():
            path = fields[key].split(".")
            target = cfg
            for part in path[:-1]:
                target = target.setdefault(part, {})
            if path[-1] in target:
                raise ValueError(f"Duplicate setting: training.{name}.{key} and {fields[key]}")
            target[path[-1]] = value


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


def cohort_envs(cfg, skill=None):
    """Configured real-episode cohort size; missing stage settings retain legacy size."""
    if is_bomb_skill(skill):
        return (
            cfg["training"]
            .get("tile_scenarios", {})
            .get("cherry_bomb_n_envs", cfg["training"]["n_envs"])
        )
    return cfg["training"]["n_envs"]


def bomb_ranking_margin(settings):
    """Saved zero-margin objectives stay zero, even with an unused old margin key."""
    if settings.get("ranking_objective", "top1_hinge_v1") == "zero_margin_distance_v1":
        return 0.0
    return settings["ranking_margin"]


def bomb_max_illegal_distance(settings):
    """Saved distance objectives keep their original occupied-cell guidance."""
    return settings.get("ranking_objective") == "margin_distance_illegal_v3"


def training_env_capacity(cfg, family="preset"):
    """Allocate enough lanes for selected stages without changing their cohort sizes."""
    stage = cfg["curriculum"].get("run_stage")
    tile = cfg["training"].get("tile_scenarios", {})
    if (
        family == "preset"
        and tile.get("enabled", False)
        and cfg["curriculum"].get("mode") == "teaching"
        and (stage in (None, "pre") or is_bomb_skill(stage))
    ):
        cherry = cohort_envs(cfg, "cherry_bomb")
        return cherry if is_bomb_skill(stage) else max(cherry, cohort_envs(cfg))
    return cohort_envs(cfg)


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
    _resolve_training_sections(raw)
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
    _resolve_training_sections(selected)
    removals = selected.pop("remove", [])
    _deep_update(raw, selected)
    for path in removals:
        _remove_path(raw, path)
    return raw


def _resolve_action_timing(cfg: dict) -> None:
    """Resolve the agent cadence into the engine's exact tick interval.

    The pinned simulator runs at 100 Hz. Keeping the user-facing frequency in
    the research TOML avoids making callers duplicate the corresponding tick
    count. A frequency that does not divide the engine rate is rejected rather
    than silently rounding the decision cadence.
    """
    environment = cfg.get("environment")
    if not isinstance(environment, dict) or "action_frequency_hz" not in environment:
        return
    frequency = environment["action_frequency_hz"]
    if (
        type(frequency) not in (int, float)
        or isinstance(frequency, bool)
        or not math.isfinite(frequency)
        or frequency <= 0
    ):
        raise ValueError("environment.action_frequency_hz must be finite and positive")
    from pvz_game import Rules

    tick_rate = Rules().game["tick_rate"]
    ticks = tick_rate / frequency
    if not ticks.is_integer() or ticks < 1:
        raise ValueError(
            "environment.action_frequency_hz must produce a positive whole simulator-tick interval"
        )
    environment["decision_ticks"] = int(ticks)
    environment["action_timing"] = "per_tick" if ticks == 1 else "fixed"


def plant_wait_check_ticks(cfg: dict) -> int | None:
    """Resolve the optional legacy-compatible wait interval to exact engine ticks."""
    settings = cfg["training"]["exploration"]
    if "plant_wait_check_seconds" not in settings:
        return None
    seconds = settings["plant_wait_check_seconds"]
    from pvz_game import Rules

    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("exploration.plant_wait_check_seconds must be finite and positive")
    ticks = float(seconds * Rules().game["tick_rate"])
    if not math.isfinite(ticks) or not ticks.is_integer() or not 1 <= ticks < 2**63:
        raise ValueError(
            "exploration.plant_wait_check_seconds must represent positive whole simulator ticks"
        )
    return int(ticks)


def passivity_settings(cfg: dict) -> dict:
    """Return residual-sun shaping settings, with legacy runs disabled."""
    settings = cfg.get("reward", {}).get("passivity")
    if settings is None:
        return {
            "win_max_reward": 0.0,
            "win_reserve": 0.0,
            "loss_max_penalty": 0.0,
            "loss_reserve": 0.0,
            "sun_reference": 1000.0,
        }
    return {
        "win_max_reward": float(settings["win_max_reward"]),
        "win_reserve": float(settings["win_reserve"]),
        "loss_max_penalty": float(settings["loss_max_penalty"]),
        "loss_reserve": float(settings["loss_reserve"]),
        "sun_reference": float(settings["sun_reference"]),
    }


def load_config(path: str | Path | None = None, *, profile: str = "train") -> dict:
    cfg = _resolve_profile(_read_config(path), profile)
    _resolve_action_timing(cfg)
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
    from pvz_rl.learning.search_settings import validate_search

    validate_search(cfg)
    objective = cfg["training"].get("objective")
    if not isinstance(objective, dict):
        raise ValueError("training.objective must describe the stateless two-stage objective")
    if objective is not None:
        if "invalid_action_target" in objective:
            raise ValueError(
                "objective.invalid_action_target was replaced by invalid_action_q_offset; "
                "use -0.1 to fit illegal proposals below the current head's minimum Q. "
                "Old optimizer states require fresh --init-from."
            )
        objective_keys = {
            "protocol",
            "accepted_outcome_share",
            "species_consistency_weight",
            "species_consistency_huber_delta",
            "invalid_action_q_offset",
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
        if objective.get("protocol") != "complete_return_action_global_v7":
            raise ValueError(
                "Unsupported training objective protocol; fresh initialization is required"
            )
        if objective.get("accepted_outcome_share") != 0.5:
            raise ValueError("objective.accepted_outcome_share must be 0.5")
        invalid_offset = objective.get("invalid_action_q_offset")
        if (
            type(invalid_offset) not in (int, float)
            or isinstance(invalid_offset, bool)
            or not math.isfinite(invalid_offset)
            or invalid_offset > 0
        ):
            raise ValueError(
                "objective.invalid_action_q_offset must be a finite non-positive number"
            )
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
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
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
    if cfg["reward"].get("version") != "action_global_value_v5":
        raise ValueError(
            "Retired reward format; ranked endgame action/global rewards require fresh training with the bundled train profile"
        )
    reward_keys = {
        "version",
        "win_reward",
        "loss_penalty",
        "mower_penalty",
        "basic_zombie_value",
        "progress_weight",
        "value_scale",
        "early_sun_extra_multiplier",
        "early_sun_spawn_fraction",
    }
    reward_keys_with_optional = (
        reward_keys,
        reward_keys | {"home_entry_penalties", "win_time_weight"},
        reward_keys | {"action"},
        reward_keys | {"action", "home_entry_penalties"},
        reward_keys | {"action", "win_time_weight"},
        reward_keys | {"action", "home_entry_penalties", "win_time_weight"},
        reward_keys | {"passivity"},
        reward_keys | {"passivity", "home_entry_penalties"},
        reward_keys | {"passivity", "win_time_weight"},
        reward_keys | {"passivity", "home_entry_penalties", "win_time_weight"},
        reward_keys | {"passivity", "action"},
        reward_keys | {"passivity", "action", "home_entry_penalties"},
        reward_keys | {"passivity", "action", "win_time_weight"},
        reward_keys | {"passivity", "action", "home_entry_penalties", "win_time_weight"},
    )
    if set(cfg["reward"]) - {"asset_loss_multiplier"} not in reward_keys_with_optional:
        raise ValueError("reward must contain the outcome, asset-value, and action settings")
    passivity = cfg["reward"].get("passivity")
    if passivity is not None:
        passivity_keys = {
            "win_max_reward",
            "win_reserve",
            "loss_max_penalty",
            "loss_reserve",
            "sun_reference",
        }
        if not isinstance(passivity, dict) or set(passivity) != passivity_keys:
            raise ValueError("reward.passivity contains an invalid setting")
        for key in passivity_keys:
            value = passivity[key]
            if (
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"reward.passivity.{key} must be finite and nonnegative")
        if passivity["sun_reference"] <= max(passivity["win_reserve"], passivity["loss_reserve"]):
            raise ValueError("reward.passivity.sun_reference must exceed both reserves")
    action_reward = cfg["reward"].get("action", {})
    if not isinstance(action_reward, dict):
        raise ValueError("reward.action must be a table")
    if set(action_reward) != {"sun_credit_multiplier", "block_value_by_plant"}:
        raise ValueError(
            "reward.action must contain sun_credit_multiplier and block_value_by_plant"
        )
    sun_credit_multiplier = action_reward["sun_credit_multiplier"]
    if (
        type(sun_credit_multiplier) not in (int, float)
        or not math.isfinite(sun_credit_multiplier)
        or sun_credit_multiplier < 0
    ):
        raise ValueError("reward.action.sun_credit_multiplier must be finite and nonnegative")
    block_values = action_reward["block_value_by_plant"]
    if not isinstance(block_values, dict) or set(block_values) != set(PLANT_TYPES):
        raise ValueError("reward.action.block_value_by_plant must cover every configured plant")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in block_values.values()
    ):
        raise ValueError("reward.action.block_value_by_plant values must be finite numbers")
    asset_loss_multiplier = cfg["reward"].get("asset_loss_multiplier", 1.0)
    if (
        type(asset_loss_multiplier) not in (int, float)
        or not math.isfinite(asset_loss_multiplier)
        or asset_loss_multiplier < 0
    ):
        raise ValueError("reward.asset_loss_multiplier must be finite and nonnegative")
    if "actor_objective" in cfg["training"] or "ent_coef" in cfg["training"]:
        raise ValueError("Retired PPO objective; use sequential Q fitting")
    if cfg["training"]["exploration"].get("objective") != "committed_plant_tile_epsilon_v1":
        raise ValueError("Only committed plant and conditional tile exploration is supported")
    for key in (
        "win_reward",
        "loss_penalty",
        "mower_penalty",
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
    if policy["action_distribution"] != "sequential_q_unmasked_tiles_v2":
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
    plant_wait_check_ticks(cfg)
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
    for key in ("cherry_bomb_probe_interval_games", "peashooter_probe_interval_games"):
        if key in c and (type(c[key]) is not int or c[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    trigger = cfg["training"].get("validation_trigger")
    if trigger is not None:
        threshold = trigger.get("cohort_success_threshold")
        if type(trigger.get("enabled")) is not bool:
            raise ValueError("training.validation_trigger.enabled must be a boolean")
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 <= threshold < 1
        ):
            raise ValueError("cohort_success_threshold must be finite and in [0, 1)")
        if trigger["enabled"] and (
            type(c.get("probe_interval_games")) is not int or c["probe_interval_games"] < 1
        ):
            raise ValueError("Validation triggers require a positive probe_interval_games")
    tile = cfg["training"].get("tile_scenarios")
    if tile is not None:
        if type(tile.get("enabled")) is not bool:
            raise ValueError("training.tile_scenarios.enabled must be a boolean")
        count = tile.get("cherry_bomb_n_envs", cfg["training"]["n_envs"])
        if type(count) is not int or count < 1:
            raise ValueError(
                "training.tile_scenarios.cherry_bomb_n_envs must be a positive integer"
            )
        sun, plants = (tile.get(k) for k in ("bomb_initial_sun", "bomb_plants"))
        if type(sun) is not int or not 150 <= sun <= 9990:
            raise ValueError("tile scenario bomb_initial_sun must be between 150 and 9990")
        ready = tile.get("bomb_mower_ready_probability", 0.0)
        if type(ready) not in (int, float) or not math.isfinite(ready) or not 0 <= ready <= 1:
            raise ValueError("tile scenario bomb_mower_ready_probability must be in [0, 1]")
        if (
            not isinstance(plants, list)
            or len(plants) != 2
            or any(type(n) is not int for n in plants)
            or not 1 <= plants[0] <= plants[1] <= 44
        ):
            raise ValueError("tile scenario bomb_plants must be an inclusive range in [1, 44]")
        objective = tile.get("ranking_objective", "top1_hinge_v1")
        if objective not in (
            "top1_hinge_v1", "zero_margin_distance_v1", "margin_distance_v2",
            "margin_distance_illegal_v3",
        ):
            raise ValueError("Unsupported tile scenario ranking_objective")
        if objective == "top1_hinge_v1":
            margin = tile.get("ranking_margin")
            if type(margin) not in (float, int) or not math.isfinite(margin) or margin <= 0:
                raise ValueError("legacy tile ranking_margin must be finite and positive")
        if objective in ("margin_distance_v2", "margin_distance_illegal_v3"):
            margin = tile.get("ranking_margin")
            if type(margin) not in (float, int) or not math.isfinite(margin) or margin < 0:
                raise ValueError("tile ranking_margin must be finite and nonnegative")
        if objective in (
            "zero_margin_distance_v1", "margin_distance_v2", "margin_distance_illegal_v3",
        ):
            weight, temperature = tile.get("distance_weight"), tile.get("distance_temperature")
            if (
                type(weight) not in (int, float) or not math.isfinite(weight)
                or type(temperature) not in (int, float) or not math.isfinite(temperature)
                or temperature <= 0 or not 0 <= weight < 4 * temperature
            ):
                raise ValueError("Bomb distance settings require finite tau > 0 and 0 <= weight < 4*tau")
        obstacle = tile.get("bomb_obstacle_plants", [1, 15])
        if (
            not isinstance(obstacle, list) or len(obstacle) != 2
            or any(type(n) is not int for n in obstacle)
            or not 1 <= obstacle[0] <= obstacle[1] <= 44
        ):
            raise ValueError("bomb_obstacle_plants must be an inclusive range in [1, 44]")
        if tile["enabled"] and not uses_games(cfg):
            raise ValueError("Tile prerequisites require game-based training")
        from pvz_game import Rules

        from pvz_rl.envs.sky_sun import validate_sky_sun_rate

        rules = Rules()
        validate_sky_sun_rate(
            tile.get("peashooter_sky_sun_rate", 1.0),
            rules,
            cfg["environment"]["cutoff_seconds"] * rules.game["tick_rate"]
            + cfg["environment"]["decision_ticks"],
        )
    from pvz_rl.learning.curriculum import TILE_STAGES

    if c.get("run_stage") in ("pre", *TILE_STAGES) and not (tile and tile["enabled"]):
        raise ValueError("Tile stages require training.tile_scenarios.enabled")
    if c.get("lessons") or set(c.get("stages", {})) - {"easy", "standard", "shared"}:
        raise ValueError("Retired curriculum lesson; use easy, standard and shared stages")
    unlimited = cfg["training"].get("until_stage_complete", False)
    if type(unlimited) is not bool:
        raise ValueError("training.until_stage_complete must be a boolean")
    if unlimited and (not uses_games(cfg) or c.get("mode") != "teaching" or not c.get("run_stage")):
        raise ValueError(
            "until_stage_complete requires a selected teaching stage and game-based scheduling"
        )
    if (
        unlimited
        and c.get("run_stage") not in ("pre", *TILE_STAGES)
        and not c.get("stages", {}).get(c.get("run_stage"), {}).get("requirements")
    ):
        raise ValueError("until_stage_complete requires a stage with mastery requirements")
    if c.get("mode", "fixed") not in ("fixed", "teaching"):
        raise ValueError("Unsupported curriculum mode")
    if c.get("run_stage") is not None:
        from pvz_rl.learning.curriculum import SELECTABLE_STAGES

        if c.get("mode") != "teaching" or c["run_stage"] not in (*SELECTABLE_STAGES, "pre"):
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
    expected_method = "complete_return_action_global_v7"
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
    if "compile_kernels" in performance:
        raise ValueError("compile_kernels was replaced by compile_collection and compile_fitting")
    for key in ("compile_collection", "compile_fitting", "telemetry"):
        if type(performance.get(key, False)) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
    if performance.get("fit_precision", "fp32") != "bf16_master":
        raise ValueError("training.performance.fit_precision must be bf16_master")
    if type(performance.get("prefetch", False)) is not bool:
        raise ValueError("training.performance.prefetch must be a boolean")
    if performance.get("annotation_input_cache", "auto") not in ("auto", "off"):
        raise ValueError("training.performance.annotation_input_cache must be auto or off")
    cache_mib = performance.get("annotation_cache_mib", 256)
    if type(cache_mib) is not int or cache_mib < 0:
        raise ValueError("training.performance.annotation_cache_mib must be a nonnegative integer")
    if type(performance.get("annotation_use_fit_limits", False)) is not bool:
        raise ValueError("training.performance.annotation_use_fit_limits must be a boolean")
    for key in ("cherry_prefetch", "cherry_cuda_warmup"):
        if type(performance.get(key, False)) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
    workers = performance.get("cherry_prepare_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError(
            "training.performance.cherry_prepare_workers must be a nonnegative integer"
        )
    if performance.get("cherry_oracle", "cpu") not in ("cpu", "cuda"):
        raise ValueError("training.performance.cherry_oracle must be cpu or cuda")
    for key, default in (
        ("cherry_oracle_lanes", 256),
        ("cherry_oracle_vram_mib", 256),
        ("cherry_worker_ram_mib", 96),
    ):
        value = performance.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"training.performance.{key} must be a positive integer")
    if set(performance) - {
        "compile_collection",
        "compile_fitting",
        "telemetry",
        "telemetry_span_limit",
        "fit_precision",
        "prefetch",
        "annotation_input_cache",
        "annotation_cache_mib",
        "annotation_use_fit_limits",
        "cherry_prepare_workers",
        "cherry_worker_ram_mib",
        "cherry_prefetch",
        "cherry_oracle",
        "cherry_cuda_warmup",
        "cherry_oracle_lanes",
        "cherry_oracle_vram_mib",
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
    if (
        type(performance.get("telemetry_span_limit", 1024)) is not int
        or performance.get("telemetry_span_limit", 1024) < 1
    ):
        raise ValueError("training.performance.telemetry_span_limit must be a positive integer")
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
    frequency = env.get("action_frequency_hz")
    if frequency is not None:
        if (
            type(frequency) not in (int, float)
            or isinstance(frequency, bool)
            or not math.isfinite(frequency)
            or frequency <= 0
        ):
            raise ValueError("environment.action_frequency_hz must be finite and positive")
        from pvz_game import Rules

        tick_rate = Rules().game["tick_rate"]
        if abs(frequency * env["decision_ticks"] - tick_rate) > 1e-9:
            raise ValueError(
                "environment.action_frequency_hz and decision_ticks disagree with the simulator tick rate"
            )
        expected_timing = "per_tick" if env["decision_ticks"] == 1 else "fixed"
        if env["action_timing"] != expected_timing:
            raise ValueError(
                "environment.action_timing must match action_frequency_hz and decision_ticks"
            )
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
    if "tile_warmup_passes" in train:
        raise ValueError(
            "training.tile_warmup_passes was replaced by training.tile_only_cohorts; "
            "n_epochs now sets all fitting passes in each cohort"
        )
    if type(train.get("tile_only_cohorts")) is not int or train["tile_only_cohorts"] < 0:
        raise ValueError("training.tile_only_cohorts must be a nonnegative integer")
    if train.get("discount_clock") != "simulation_ticks" or (
        train["gamma"] != 1 and not train.get("search", {}).get("enabled", False)
    ):
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
        "l2_strength",
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
    annotation_defaults = {"learning_rate": 3e-4, "margin": 1.0, "l2_strength": 1e-4}
    for key in ("learning_rate", "margin", "l2_strength"):
        value = annotation.get(key, annotation_defaults[key])
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
