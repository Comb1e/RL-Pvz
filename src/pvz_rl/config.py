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
from random import SystemRandom

from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES
from pvz_game.cuda.schema import MOWER_STATES, PLANT_STATES, ZOMBIE_STATES

from pvz_rl.envs.tile_scenarios import (
    ATTACKER_EXERCISES,
    BOMB_SKILLS,
    attacker_mix,
    is_bomb_skill,
    is_placement_skill,
    is_single_frame_skill,
    prediction_settings,
)
from pvz_rl.learning.budget import uses_games

_CONFIG_RESOURCE = "data/train.toml"

# User-facing stage tables resolve once into the checkpoint-compatible runtime layout.
_STAGE_PARAMETERS = {
    "normal": {
        **{
            key: f"training.{key}"
            for key in (
                "n_envs",
                "exploration",
                "validation_schedule",
                "eval_interval_games",
            )
        },
        "curriculum": "curriculum",
    },
    "cherry_bomb": {
        "n_envs": "training.tile_scenarios.cherry_bomb_n_envs",
        "initial_sun": "training.tile_scenarios.bomb_initial_sun",
        "plants": "training.tile_scenarios.bomb_plants",
        "obstacle_plants": "training.tile_scenarios.bomb_obstacle_plants",
        "mower_ready_probability": "training.tile_scenarios.bomb_mower_ready_probability",
        **{
            key: f"training.tile_scenarios.{key}"
            for key in (
                "ranking_objective",
                "softmax_temperature",
                "ranking_margin",
                "distance_weight",
                "distance_temperature",
            )
        },
        **{
            key: f"training.performance.cherry_{key}"
            for key in (
                "prepare_workers",
                "worker_ram_mib",
                "prefetch",
                "oracle",
                "cuda_warmup",
                "oracle_lanes",
                "oracle_vram_mib",
            )
        },
        "probe_interval_games": "curriculum.cherry_bomb_probe_interval_games",
        "validation_success_threshold": "curriculum.cherry_bomb_validation_success_threshold",
    },
    "cherry_bomb_obstacle": {
        "softmax_temperature": "training.tile_scenarios.obstacle_softmax_temperature",
    },
    "attacker_pre": {
        **{
            key: f"training.tile_scenarios.attacker.{key}"
            for key in (
                "n_envs",
                "plants",
                "zombie_count",
                "provided_plants",
                "initial_sun",
                "mower_ready_probability",
                "softmax_temperature",
                "ranking_objective",
            )
        },
        **{
            key: f"training.performance.attacker_{key}"
            for key in (
                "prepare_workers",
                "worker_ram_mib",
                "prefetch",
                "cuda_warmup",
                "oracle_lanes",
                "oracle_vram_mib",
            )
        },
        "probe_interval_games": "curriculum.attacker_pre_probe_interval_games",
    },
    "prediction": {
        **{
            key: f"training.tile_scenarios.prediction.{key}"
            for key in (
                "n_envs",
                "validation_cases",
                "validation_max_loss",
                "plants",
                "live_zombies",
                "defeated_zombies",
                "pending_zombies",
                "sun",
                "ready_probability",
            )
        },
        "probe_interval_games": "curriculum.prediction_probe_interval_games",
    },
    "attacker": {
        **{
            key: f"training.tile_scenarios.attacker_mix.{key}"
            for key in ("n_envs", "validation_cases", "ratios")
        },
        "probe_interval_games": "curriculum.attacker_probe_interval_games",
    },
    "peashooter": {
        "sky_sun_rate": "training.tile_scenarios.peashooter_sky_sun_rate",
        **{
            key: f"training.peashooter_rules.{key}"
            for key in (
                "fitting_objective",
                "column_decay",
                "softmax_temperature",
                "oracle_lanes",
                "oracle_vram_mib",
                "oracle_chunk_ticks",
                "prepare_workers",
                "n_envs",
                "snapshot_cache_mib",
            )
        },
        "probe_interval_games": "curriculum.peashooter_probe_interval_games",
        "validation_success_threshold": "curriculum.peashooter_validation_success_threshold",
    },
}

for _stage, _fields in _STAGE_PARAMETERS.items():
    _fields["forecast_mode"] = f"training.stage_forecast_modes.{_stage}"
    _fields["extra_actions"] = f"training.stage_extra_actions.{_stage}"
    for _outcome in ("static_field", "probe"):
        _fields[f"{_outcome}_prediction_weight"] = (
            f"training.stage_outcome_weights.{_stage}.{_outcome}"
        )


def absolute_outcomes(cfg):
    """The supported model uses public inputs and absolute action outcomes."""
    return cfg["encoding"]["version"] == "entity_public_v4"


def forecast_version(cfg):
    """Persisted outcome supervision identity; older versions are rejected."""
    return cfg["training"].get("forecasts", {}).get("supervision_version", 1)


def forecast_mode(cfg, stage=None):
    stage = stage or cfg.get("curriculum", {}).get("run_stage") or "normal"
    if stage not in _STAGE_PARAMETERS:
        stage = "normal"
    return cfg["training"]["stage_forecast_modes"][stage]


def forecast_forks_enabled(cfg, stage=None):
    stage = stage or cfg.get("curriculum", {}).get("run_stage")
    if stage == "prediction":
        return forecast_mode(cfg, stage) == "prediction_only"
    return cfg["training"].get("forecasts", {}).get("enabled", False) and forecast_mode(
        cfg, stage
    ) in ("placement_outcomes", "action_outcomes")


def outcome_prediction_weights(cfg, stage="normal"):
    """Static-field/probe coefficients; saved single-weight runs retain their loss."""
    training = cfg["training"]
    stage = stage if stage in _STAGE_PARAMETERS else "normal"
    if stage == "prediction":
        return 1.0, 1.0
    weights = training.get("stage_outcome_weights", {}).get(stage)
    if weights is not None:
        return weights["static_field"], weights["probe"]
    weight = (
        training.get("effect_prediction_weight", 0.1)
        if stage == "normal"
        else training.get("stage_effect_weights", {}).get(stage, 0.0)
    )
    return weight, weight


def effect_extra_actions(cfg, stage="normal"):
    """Stage overrides are collection-only; absent saved overrides retain sampling."""
    training = cfg["training"]
    stage = stage if stage in _STAGE_PARAMETERS else "normal"
    return training.get("stage_extra_actions", {}).get(
        stage, training["forecasts"]["extra_actions"]
    )


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
    if skill == "prediction":
        return prediction_settings(cfg)["n_envs"]
    if skill == "attacker":
        return attacker_mix(cfg)["n_envs"]
    if skill == "attacker_pre":
        return cfg["training"]["tile_scenarios"]["attacker"]["n_envs"]
    if skill == "peashooter":
        return cfg["training"].get("peashooter_rules", {}).get("n_envs", cfg["training"]["n_envs"])
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
    """Use maximum illegal distance for current diagnostics and saved v3 fitting."""
    return settings.get("ranking_objective") in (
        "margin_distance_illegal_v3",
        "set_cross_entropy_v1",
    )


def bomb_softmax_temperature(settings, success_rate, *, skill="cherry_bomb"):
    """Resolve a fixed temperature or a rising/falling cohort-success schedule."""
    value = settings.get("softmax_temperature")
    if skill == "cherry_bomb_obstacle":
        value = settings.get("obstacle_softmax_temperature", value)
    if isinstance(value, dict):
        if set(value) != {"zero_success", "full_success"}:
            raise ValueError("Bomb softmax_temperature requires zero_success and full_success")
        start, end = value["zero_success"], value["full_success"]
    else:
        start = end = value
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in (start, end)):
        raise ValueError("Bomb softmax_temperature must be finite and positive")
    if (
        type(success_rate) not in (int, float)
        or not math.isfinite(success_rate)
        or not 0 <= success_rate <= 1
    ):
        raise ValueError("Bomb cohort success rate must be finite and in [0, 1]")
    return (1 - success_rate) * start + success_rate * end


def training_env_capacity(cfg, family="preset"):
    """Allocate enough lanes for selected stages without changing their cohort sizes."""
    stage = cfg["curriculum"].get("run_stage")
    tile = cfg["training"].get("tile_scenarios", {})
    if (
        family == "preset"
        and tile.get("enabled", False)
        and cfg["curriculum"].get("mode") == "teaching"
        and (stage in (None, "pre") or is_single_frame_skill(stage))
    ):
        cherry = cohort_envs(cfg, "cherry_bomb")
        if is_single_frame_skill(stage):
            return cohort_envs(cfg, stage)
        attacker = tile.get("attacker", {}).get("n_envs", 0)
        mixed = (attacker_mix(cfg) or {}).get("n_envs", 0)
        prediction = (prediction_settings(cfg) or {}).get("n_envs", 0)
        return max(
            cherry, attacker, mixed, prediction, cohort_envs(cfg, "peashooter"), cohort_envs(cfg)
        )
    if stage == "peashooter":
        return cohort_envs(cfg, "peashooter")
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
    if (cfg["encoding"]["version"], cfg["policy"]["kind"]) not in {
        ("entity_public_v4", "shared_public_outcomes_q_v5"),
    }:
        raise ValueError("Human demonstrations require the stateless two-stage Transformer profile")
    return copy.deepcopy(cfg)


def validate_config(cfg: dict) -> None:
    removed = {"annotation", "counterfactual", "tile_only_cohorts"} & cfg["training"].keys()
    if removed:
        raise ValueError(
            f"Removed training settings: {sorted(removed)}; start from the shared-encoder train.toml"
        )
    from pvz_rl.learning.peashooter_rules import validate_rule_settings

    validate_rule_settings(cfg)
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
        if objective.get("protocol") != "nstep_action_outcomes_v4":
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
        cfg.get("encoding", {}).get("version"),
        cfg.get("policy", {}).get("kind"),
    ) in {
        ("entity_public_v4", "shared_public_outcomes_q_v5"),
    }
    if cfg["training"].get("validation_schedule", "periodic") not in (
        "periodic",
        "stage_success",
    ):
        raise ValueError("training.validation_schedule must be periodic or stage_success")
    if simulator(cfg) not in ("cpu", "cuda"):
        raise ValueError("simulation.backend must be cpu or cuda")
    if not current_policy:
        raise ValueError("Retired observation/policy format; use the bundled public model profile")
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
    if cfg["training"]["exploration"].get("objective") not in (
        "committed_plant_tile_epsilon_v1",
        "committed_species_epsilon_v1",
    ):
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
        "shared_width",
        "encoder_blocks",
        "tile_decoder_blocks",
        "attention_heads",
        "feedforward_width",
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
    if policy["shared_width"] % policy["attention_heads"]:
        raise ValueError("shared_width must be divisible by attention_heads")
    optional = {"attacker": attacker_mix(cfg), "prediction": prediction_settings(cfg)}
    stages = tuple(
        stage for stage in _STAGE_PARAMETERS if stage not in optional or optional[stage] is not None
    )
    if set(cfg["training"].get("stage_effect_weights", {})) - set(stages[1:]):
        raise ValueError("Unknown preliminary outcome stage")
    split_weights = cfg["training"].get("stage_outcome_weights", {})
    if not isinstance(split_weights, dict) or set(split_weights) - set(stages):
        raise ValueError("Unknown outcome weight stage")
    for stage in stages:
        weights = split_weights.get(stage)
        if stage == "prediction" and weights is not None:
            raise ValueError("Prediction-only fitting uses unweighted outcome loss")
        if stage in split_weights and (
            not isinstance(weights, dict) or set(weights) != {"static_field", "probe"}
        ):
            raise ValueError(
                f"training.{stage} requires both static_field_prediction_weight and "
                "probe_prediction_weight"
            )
        legacy = (
            cfg["training"].get("effect_prediction_weight")
            if stage == "normal"
            else cfg["training"].get("stage_effect_weights", {}).get(stage)
        )
        if weights is not None and legacy is not None:
            raise ValueError(f"Duplicate combined and separate outcome weights for {stage}")
        for name, weight in zip(
            ("static_field", "probe"), outcome_prediction_weights(cfg, stage), strict=True
        ):
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"training.{stage}.{name}_prediction_weight must be finite and nonnegative"
                )

    attempts = cfg["training"]["tile_scenarios"].get("sampling_max_attempts")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("training.tile_scenarios.sampling_max_attempts must be positive")
    sample_seconds = output_settings(cfg)["logging"]["hardware_sample_seconds"]
    if (
        type(sample_seconds) not in (int, float)
        or not math.isfinite(sample_seconds)
        or sample_seconds < 0.1
    ):
        raise ValueError("hardware_sample_seconds must be finite and at least 0.1")
    exploration = cfg["training"]["exploration"]
    plant_wait_check_ticks(cfg)
    legacy_tiles = exploration["objective"] == "committed_plant_tile_epsilon_v1"
    keys = ("plant_epsilon_start", "plant_epsilon_floor")
    if legacy_tiles:
        keys += ("epsilon_start", "epsilon_floor")
    for key in keys:
        value = exploration.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"exploration.{key} must be finite and in (0, 1]")
    if legacy_tiles and exploration["epsilon_floor"] > exploration["epsilon_start"]:
        raise ValueError("exploration floor exceeds start")
    if exploration["plant_epsilon_floor"] > exploration["plant_epsilon_start"]:
        raise ValueError("plant exploration floor exceeds start")
    for key in ("decay_games", "plant_decay_games") if legacy_tiles else ("plant_decay_games",):
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
    for key in (
        "cherry_bomb_probe_interval_games",
        "peashooter_probe_interval_games",
        "attacker_pre_probe_interval_games",
        "prediction_probe_interval_games",
        "attacker_probe_interval_games",
    ):
        if key in c and (type(c[key]) is not int or c[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "cherry_bomb_validation_success_threshold",
        "peashooter_validation_success_threshold",
    ):
        if key in c and (
            type(c[key]) not in (int, float) or not math.isfinite(c[key]) or not 0 <= c[key] < 1
        ):
            raise ValueError(f"{key} must be finite and in [0, 1)")
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
            "top1_hinge_v1",
            "zero_margin_distance_v1",
            "margin_distance_v2",
            "margin_distance_illegal_v3",
            "set_cross_entropy_v1",
        ):
            raise ValueError("Unsupported tile scenario ranking_objective")
        if objective == "set_cross_entropy_v1":
            for skill in BOMB_SKILLS:
                bomb_softmax_temperature(tile, 0.0, skill=skill)
        if objective == "top1_hinge_v1":
            margin = tile.get("ranking_margin")
            if type(margin) not in (float, int) or not math.isfinite(margin) or margin <= 0:
                raise ValueError("legacy tile ranking_margin must be finite and positive")
        if objective in ("margin_distance_v2", "margin_distance_illegal_v3"):
            margin = tile.get("ranking_margin")
            if type(margin) not in (float, int) or not math.isfinite(margin) or margin < 0:
                raise ValueError("tile ranking_margin must be finite and nonnegative")
        if objective in (
            "zero_margin_distance_v1",
            "margin_distance_v2",
            "margin_distance_illegal_v3",
        ):
            weight, temperature = tile.get("distance_weight"), tile.get("distance_temperature")
            if (
                type(weight) not in (int, float)
                or not math.isfinite(weight)
                or type(temperature) not in (int, float)
                or not math.isfinite(temperature)
                or temperature <= 0
                or not 0 <= weight < 4 * temperature
            ):
                raise ValueError(
                    "Bomb distance settings require finite tau > 0 and 0 <= weight < 4*tau"
                )
        obstacle = tile.get("bomb_obstacle_plants", [1, 15])
        if (
            not isinstance(obstacle, list)
            or len(obstacle) != 2
            or any(type(n) is not int for n in obstacle)
            or not 1 <= obstacle[0] <= obstacle[1] <= 44
        ):
            raise ValueError("bomb_obstacle_plants must be an inclusive range in [1, 44]")
        if tile["enabled"] and not uses_games(cfg):
            raise ValueError("Tile prerequisites require game-based training")
        from pvz_game import Rules

        from pvz_rl.envs.sky_sun import validate_sky_sun_rate

        rules = Rules()
        attacker = tile.get("attacker")
        if attacker is not None:
            from pvz_rl.envs.attacker_scenarios import ATTACKERS, attacker_bounds, warmup_ticks

            for key in ("n_envs", "initial_sun"):
                if type(attacker.get(key)) is not int or attacker[key] < 1:
                    raise ValueError(f"attacker_pre.{key} must be a positive integer")
            for key, lo, hi in (("plants", 4, 8), ("zombie_count", 1, 2)):
                value = attacker.get(key)
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(type(n) is not int for n in value)
                    or not lo <= value[0] <= value[1] <= hi
                ):
                    raise ValueError(f"attacker_pre.{key} must be a range in [{lo}, {hi}]")
            kinds = attacker.get("provided_plants")
            if (
                not isinstance(kinds, list)
                or not kinds
                or any(type(k) is not str for k in kinds)
                or len(set(kinds)) != len(kinds)
                or set(kinds) - set(ATTACKERS)
                or set(kinds) - set(cfg["environment"]["plants"])
            ):
                raise ValueError(
                    "attacker_pre.provided_plants must be distinct configured attackers"
                )
            if (
                not max(rules.plants[k]["cost"] for k in kinds)
                <= attacker["initial_sun"]
                <= rules.game["sun_cap"]
            ):
                raise ValueError("attacker_pre.initial_sun must afford every supplied plant")
            ready = attacker.get("mower_ready_probability")
            if type(ready) not in (int, float) or not math.isfinite(ready) or not 0 <= ready <= 1:
                raise ValueError("attacker_pre.mower_ready_probability must be in [0, 1]")
            if attacker.get("ranking_objective") != "set_cross_entropy_v1":
                raise ValueError("attacker_pre requires successful-set cross-entropy")
            bomb_softmax_temperature(attacker, 0.0)
            horizon = max(
                v["horizon"]
                for k, table in attacker_bounds(rules)["table"].items()
                if k in kinds
                for v in table.values()
            )
            # Slowest Basic must also reach a left-column mine within the cutoff.
            from pvz_rl.envs.attacker_scenarios import _arrival

            horizon = max(
                horizon,
                _arrival(rules, rules.game["cols"] * rules.game["units_per_tile"], latest=True),
            )
            if (
                tile["enabled"]
                and c.get("run_stage") in (None, "attacker_pre", "attacker")
                and warmup_ticks(rules, attacker) + horizon
                >= cfg["environment"]["cutoff_seconds"] * rules.game["tick_rate"]
            ):
                raise ValueError(
                    "attacker_pre cutoff is too short for cooldown and conservative completion"
                )
            if (
                tile["enabled"]
                and c.get("run_stage") in (None, "attacker_pre", "attacker")
                and "basic" not in cfg["environment"]["zombies"]
            ):
                raise ValueError("attacker_pre requires Basic zombies in the configured roster")
        validate_sky_sun_rate(
            tile.get("peashooter_sky_sun_rate", 1.0),
            rules,
            cfg["environment"]["cutoff_seconds"] * rules.game["tick_rate"]
            + cfg["environment"]["decision_ticks"],
        )
    from pvz_rl.learning.curriculum import TILE_STAGES

    if c.get("run_stage") in ("pre", *TILE_STAGES) and not (tile and tile["enabled"]):
        raise ValueError("Tile stages require training.tile_scenarios.enabled")
    if c.get("run_stage") == "attacker_pre" and "attacker" not in tile:
        raise ValueError("attacker_pre requires its training configuration section")
    prediction = prediction_settings(cfg)
    if prediction is not None:
        if type(prediction.get("n_envs")) is not int or prediction["n_envs"] < 1:
            raise ValueError("prediction.n_envs must be a positive integer")
        if (
            type(prediction.get("validation_cases")) is not int
            or prediction["validation_cases"] != 200
        ):
            raise ValueError("Prediction validation requires exactly 200 cases")
        for key, limit in (("ready_probability", 1), ("validation_max_loss", math.inf)):
            value = prediction.get(key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= limit
            ):
                raise ValueError(f"Invalid prediction.{key}")
        for key, limit in (
            ("plants", 45),
            ("live_zombies", math.inf),
            ("defeated_zombies", math.inf),
            ("pending_zombies", math.inf),
            ("sun", rules.game["sun_cap"]),
        ):
            values = prediction.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(type(n) is not int for n in values)
                or not 0 <= values[0] <= values[1] <= limit
            ):
                raise ValueError(f"Invalid prediction.{key} range")
        if not prediction["live_zombies"][1] and not prediction["pending_zombies"][1]:
            raise ValueError("Prediction scenes need live or pending zombies")
        if "prediction_probe_interval_games" not in c:
            raise ValueError("Prediction requires its own validation interval")
    elif c.get("run_stage") == "prediction":
        raise ValueError("prediction requires its training configuration section")
    mix = attacker_mix(cfg)
    if mix is not None:
        from fractions import Fraction

        if not isinstance(mix, dict) or "attacker" not in tile:
            raise ValueError("attacker requires its mix and the attacker_pre settings")
        if tile.get("ranking_objective") != "set_cross_entropy_v1":
            raise ValueError(
                "attacker requires successful-set cross-entropy for its bomb exercises"
            )
        ratios = mix.get("ratios")
        if (
            not isinstance(ratios, list)
            or len(ratios) != len(ATTACKER_EXERCISES)
            or any(type(r) not in (int, float) or not math.isfinite(r) or r <= 0 for r in ratios)
            or sum(Fraction(str(r)) for r in ratios) != 1
        ):
            raise ValueError("attacker.ratios must contain three positive shares summing to one")
        for key in ("n_envs", "validation_cases"):
            size = mix.get(key)
            if (
                type(size) is not int
                or size < 1
                or any((size * Fraction(str(r))).denominator != 1 for r in ratios)
            ):
                raise ValueError(f"attacker.{key} must give integral exercise quotas")
    elif c.get("run_stage") == "attacker":
        raise ValueError("attacker requires its training configuration section")
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
        split = "curriculum" if "curriculum" in cfg["splits"] else "validation"
        seed_values(cfg, split, c["probe_cases"])
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
    if type(train.get("random_cohort_seeds", False)) is not bool:
        raise ValueError("training.random_cohort_seeds must be boolean")
    expected_method = "nstep_action_outcomes_v4"
    if train.get("method") != expected_method:
        raise ValueError(
            "Training requires a supported n-step fork-forecast method and fresh models"
        )
    forecasts = train.get("forecasts", {})
    if forecast_version(cfg) != 4:
        raise ValueError(
            "Unsupported outcome supervision; old checkpoints are incompatible, start fresh"
        )
    if any(k.startswith("pretrain_") or k == "effect_conditional_weight" for k in forecasts):
        raise ValueError(
            "Removed private pretraining/legacy effect settings; use the current profile"
        )
    if "threat_prediction_weight" in train or "stage_threat_weights" in train:
        raise ValueError(
            "Removed factual prediction settings; use the current outcome-only profile"
        )
    for stage in stages:
        expected = (
            "prediction_only"
            if stage == "prediction"
            else "placement_outcomes"
            if is_placement_skill(stage)
            else "action_outcomes"
        )
        if train.get("stage_forecast_modes", {}).get(stage) != expected:
            raise ValueError(f"Unsupported outcome mode for {stage}; expected {expected}")
    if forecasts.get("effect_objective") != "absolute_outcome_v3":
        raise ValueError("Training requires absolute action outcomes")
    if forecasts.get("protocol") != "static-fork-action-outcomes-v2":
        raise ValueError("Training requires the outcome-only fork protocol")
    if type(forecasts.get("enabled")) is not bool:
        raise ValueError("training.forecasts.enabled must be boolean")
    if forecasts.get("probe") not in ("basic", "conehead", "buckethead"):
        raise ValueError("Static probe must be basic, conehead or buckethead")
    if forecasts.get("extra_actions") != 2:
        raise ValueError("training.forecasts.extra_actions must be 2")
    collection = forecasts.get("collection_protocol")
    if collection != "stage_action_outcomes_v4" or not absolute_outcomes(cfg):
        raise ValueError("Training requires public-only stage action outcomes")
    probability = forecasts.get("wait_effect_sample_probability", 1.0)
    if (
        type(probability) not in (int, float)
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("wait_effect_sample_probability must be between zero and one")
    for stage, count in train.get("stage_extra_actions", {}).items():
        if stage not in stages or type(count) is not int or count not in (0, 2):
            raise ValueError("Stage extra_actions must be zero or two")
    for stage in stages[1:]:
        if effect_extra_actions(cfg, stage) != 0:
            raise ValueError(f"{stage} requires zero extra actions")
    for key in ("lanes", "vram_mib", "chunk_ticks", "n_step", "injected_n_step"):
        if type(forecasts.get(key)) is not int or forecasts[key] <= 0:
            raise ValueError(f"training.forecasts.{key} must be a positive integer")
    for key in ("warmup_games", "injection_after_games"):
        if type(forecasts.get(key)) is not int or forecasts[key] < 0:
            raise ValueError(f"training.forecasts.{key} must be a nonnegative integer")
    for key in ("shaping_weight", "initial_discount_half_life_steps", "discount_half_life_steps"):
        value = forecasts.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or ("half_life" in key and value == 0)
        ):
            raise ValueError(f"training.forecasts.{key} is invalid")
    if type(forecasts.get("measure_gradient_norms")) is not bool:
        raise ValueError("training.forecasts.measure_gradient_norms must be boolean")
    weights = forecasts.get("shaping_weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 4
        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in weights)
    ):
        raise ValueError("training.forecasts.shaping_weights needs four finite nonnegative values")
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
    for key in ("fork_packed", "fork_suffix_cache"):
        if type(performance.get(key, True)) is not bool:
            raise ValueError(f"training.performance.{key} must be boolean")
    threads = performance.get("fork_threads", 32)
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("training.performance.fork_threads must be 32, 64, 128 or 256")
    for key, default in (("fork_cache_steps", 8), ("fork_cache_vram_mib", 128)):
        if type(performance.get(key, default)) is not int or performance.get(key, default) < 0:
            raise ValueError(f"training.performance.{key} must be a nonnegative integer")
    for key, default in (("fork_cache_zombies", 128), ("fork_cache_projectiles", 256)):
        if type(performance.get(key, default)) is not int or performance.get(key, default) < 1:
            raise ValueError(f"training.performance.{key} must be a positive integer")
    if "compile_kernels" in performance:
        raise ValueError("compile_kernels was replaced by compile_collection and compile_fitting")
    for key in ("compile_collection", "compile_fitting", "telemetry"):
        if type(performance.get(key, False)) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
    if performance.get("fit_precision", "fp32") != "bf16_master":
        raise ValueError("training.performance.fit_precision must be bf16_master")
    if type(performance.get("prefetch", False)) is not bool:
        raise ValueError("training.performance.prefetch must be a boolean")
    for key in (
        "cherry_prefetch",
        "cherry_cuda_warmup",
        "attacker_prefetch",
        "attacker_cuda_warmup",
    ):
        if type(performance.get(key, False)) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
    workers = performance.get("cherry_prepare_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError(
            "training.performance.cherry_prepare_workers must be a nonnegative integer"
        )
    for key in ("attacker_prepare_workers", "attacker_worker_ram_mib"):
        value = performance.get(key, 0 if key.endswith("workers") else 96)
        if type(value) is not int or value < (0 if key.endswith("workers") else 1):
            raise ValueError(f"Invalid training.performance.{key}")
    if performance.get("cherry_oracle", "cpu") not in ("cpu", "cuda"):
        raise ValueError("training.performance.cherry_oracle must be cpu or cuda")
    for key, default in (
        ("cherry_oracle_lanes", 256),
        ("cherry_oracle_vram_mib", 256),
        ("cherry_worker_ram_mib", 96),
        ("attacker_oracle_lanes", 256),
        ("attacker_oracle_vram_mib", 256),
    ):
        value = performance.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"training.performance.{key} must be a positive integer")
    from pvz_rl.learning.auxiliary_settings import validate_auxiliary_settings

    validate_auxiliary_settings(performance)
    if set(performance) - {
        "interval_backend",
        "auxiliary_hybrid",
        "auxiliary_cpu_workers",
        "auxiliary_worker_ram_mib",
        "auxiliary_cuda_target_ms",
        "auxiliary_vram_mib",
        "auxiliary_spill_gib",
        "auxiliary_interval_lanes",
        "auxiliary_teacher_lanes",
        "fork_packed",
        "fork_threads",
        "fork_suffix_cache",
        "fork_cache_steps",
        "fork_cache_vram_mib",
        "fork_cache_zombies",
        "fork_cache_projectiles",
        "compile_collection",
        "compile_fitting",
        "telemetry",
        "telemetry_span_limit",
        "fit_precision",
        "prefetch",
        "attacker_prepare_workers",
        "attacker_worker_ram_mib",
        "attacker_prefetch",
        "attacker_cuda_warmup",
        "attacker_oracle_lanes",
        "attacker_oracle_vram_mib",
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
        "collection_tile_buckets",
        "teacher_batched_transfers",
        "teacher_pipeline",
        "vectorized_effect_gather",
        "collection_vram_headroom_mib",
    }:
        raise ValueError("training.performance contains unsupported execution settings")
    for key in (
        "collection_tile_buckets",
        "teacher_batched_transfers",
        "teacher_pipeline",
        "vectorized_effect_gather",
    ):
        if key in performance and type(performance[key]) is not bool:
            raise ValueError(f"training.performance.{key} must be a boolean")
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
    if train.get("discount_clock") != "simulation_ticks" or not 0 < train["gamma"] <= 1:
        raise ValueError("Returns require tick-based discounting with 0 < gamma <= 1")
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


def curriculum_probe_seeds(cfg: dict, *, count: int | None = None) -> list[int]:
    """Draw a fresh mastery panel; callers retain it while validation is pending."""
    if count is not None:
        return _fresh_validation_seeds(cfg, count)
    split = "curriculum" if "curriculum" in cfg["splits"] else "validation"
    return random_validation_seeds(cfg, split, cfg["curriculum"]["probe_cases"])


def random_validation_seeds(cfg: dict, split="validation", limit: int | None = None) -> list[int]:
    """Independent fresh cases outside every reserved split; splits still set counts."""
    count = len(seed_values(cfg, split, limit))
    return _fresh_validation_seeds(cfg, count)


def _fresh_validation_seeds(cfg, count):
    start = max(hi for _, hi in cfg["splits"].values()) + 1
    population = range(start, 2**63)
    if len(population) < count:
        raise ValueError("Reserved seed splits leave insufficient space for random validation")
    return SystemRandom().sample(population, count)
