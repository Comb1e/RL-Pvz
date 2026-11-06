"""One training-progress unit shared by scheduling, logging, and reports."""


def validation_interval_games(cfg, domain="normal"):
    """Stage clocks; absent overrides retain the saved common interval."""
    curriculum = cfg["curriculum"]
    if domain in ("cherry_bomb", "cherry_bomb_obstacle"):
        return curriculum.get(
            "cherry_bomb_probe_interval_games", curriculum["probe_interval_games"]
        )
    if domain == "peashooter":
        return curriculum.get("peashooter_probe_interval_games", curriculum["probe_interval_games"])
    if domain in ("attacker_pre", "attacker"):
        return curriculum.get(f"{domain}_probe_interval_games", curriculum["probe_interval_games"])
    return curriculum["probe_interval_games"]


def refresh_validation_schedule(cfg, current):
    """Refresh the requested Cherry interval while preserving pending validation state."""
    from copy import deepcopy

    result = deepcopy(cfg)
    key = "cherry_bomb_probe_interval_games"
    if key in current["curriculum"]:
        result["curriculum"][key] = current["curriculum"][key]
    else:
        result["curriculum"].pop(key, None)
    return result


def until_stage_complete(cfg):
    return cfg["training"].get("until_stage_complete", False)


def uses_games(cfg):
    return cfg["training"]["budget_unit"] == "games"


def budget_target(cfg):
    if until_stage_complete(cfg):
        return None
    return cfg["training"]["total_games" if uses_games(cfg) else "total_steps"]


def effective_limits(cfg):
    target = budget_target(cfg)
    return {
        "games": target if uses_games(cfg) else None,
        "decisions": None if uses_games(cfg) else target,
        "minutes": None if until_stage_complete(cfg) else cfg["training"].get("max_minutes"),
    }


def evaluation_interval(cfg):
    if uses_games(cfg) and cfg["training"].get("validation_trigger", {}).get("enabled", False):
        return cfg["curriculum"]["probe_interval_games"]
    return cfg["training"]["eval_interval_games" if uses_games(cfg) else "eval_interval"]


def progress_value(cfg, model):
    from pvz_rl.learning.curriculum import TILE_STAGES

    stage = cfg["curriculum"].get("run_stage")
    if stage in TILE_STAGES:
        return model.tile_skills.episodes[stage]
    if stage == "pre":
        return model.tile_skills.total
    return getattr(model, "training_games", 0) if uses_games(cfg) else model.num_timesteps
