"""Episode-local sky income; combat and the installed engine stay pinned."""

from pvz_game import Rules

from pvz_rl.config import lesson_settings


def natural_sun(cfg, family):
    return family not in ("saving",) or lesson_settings(cfg)[family]["natural_sun"]


def sky_rules(rules, enabled):
    if enabled:
        return rules
    raw = rules.to_dict()
    raw["game"]["sky_sun_amount"] = 0
    return Rules(raw)
