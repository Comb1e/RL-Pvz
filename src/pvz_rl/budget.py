"""One training-progress unit shared by scheduling, logging, and reports."""


def uses_games(cfg):
    # Old saved configurations keep their original decision schedule.
    return cfg["training"].get("budget_unit", "decisions") == "games"


def budget_target(cfg):
    return cfg["training"]["total_games" if uses_games(cfg) else "total_steps"]


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
