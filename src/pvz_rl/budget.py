"""One training-progress unit shared by scheduling, logging, and reports."""


def until_stage_complete(cfg):
    return cfg["training"].get("until_stage_complete", False)


def uses_games(cfg):
    # Old saved configurations keep their original decision schedule.
    return cfg["training"].get("budget_unit", "decisions") == "games"


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
    return cfg["training"]["eval_interval_games" if uses_games(cfg) else "eval_interval"]


def progress_value(cfg, model):
    return getattr(model, "training_games", 0) if uses_games(cfg) else model.num_timesteps


def curve_axis(cfg):
    return (
        ("training_games", "Completed training games")
        if uses_games(cfg)
        else ("training_steps", "Training decisions")
    )
