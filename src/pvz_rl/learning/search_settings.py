"""Versioned search-target identity; absent settings always mean legacy learning."""

import math

import numpy as np

SEARCH_PROTOCOL = "bounded_public_search_v1"
TARGET_PROTOCOL = "confirmed_same_board_returns_v1"


def search_enabled(cfg):
    return cfg["training"].get("search", {}).get("enabled", False)


def validate_search(cfg):
    settings = cfg["training"].get("search")
    if settings is None:
        return
    if type(settings.get("enabled")) is not bool:
        raise ValueError("training.search.enabled must be boolean")
    if (
        settings.get("protocol") != SEARCH_PROTOCOL
        or settings.get("target_protocol") != TARGET_PROTOCOL
    ):
        raise ValueError("Unsupported search or teacher-target protocol")
    if settings["enabled"]:
        gamma = cfg["training"]["gamma"]
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("Search gamma must be finite and in (0, 1]")
        from pvz_game import Rules

        ticks = cfg["environment"]["cutoff_seconds"] * Rules().game["tick_rate"]
        # Prefix-dependent posthoc shaping is expressed at the query time. Its
        # inverse discount and squared feasibility term must fit the FP32 loss.
        if -ticks * math.log(gamma) > math.log(np.finfo(np.float32).max / 1e12) / 2:
            raise ValueError("Search gamma makes prefix shaping unrepresentable; increase gamma")
    for name in (
        "root_candidates",
        "node_candidates",
        "discovery_worlds",
        "confirmation_worlds",
        "rollouts",
        "max_nodes",
        "max_depth",
        "expansions_per_rollout",
        "queries_per_episode",
        "scratch_envs",
        "scratch_vram_mib",
    ):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f"training.search.{name} must be a positive integer")
    if settings["root_candidates"] < 2 or settings["node_candidates"] < 2:
        raise ValueError("Search needs at least two alternatives")
    if any(settings[k] < 2 or settings[k] % 2 for k in ("discovery_worlds", "confirmation_worlds")):
        raise ValueError(
            "World panels must have positive even size for the balanced controller mixture"
        )
    if settings["rollouts"] < settings["root_candidates"] * (
        settings["discovery_worlds"] + settings["confirmation_worlds"]
    ):
        raise ValueError("Search budget cannot cover balanced pilot and confirmation")
    for name in (
        "minimum_interval_seconds",
        "frontier_seconds",
        "uct_c",
        "value_weight",
        "difference_weight",
        "feasibility_weight",
        "huber_delta",
        "tie_tolerance",
    ):
        if (
            type(settings.get(name)) not in (int, float)
            or not math.isfinite(settings[name])
            or settings[name] <= 0
        ):
            raise ValueError(f"training.search.{name} must be finite and positive")
    for name in ("peashooter_query_probability", "normal_query_probability"):
        if type(settings.get(name)) not in (int, float) or not 0 <= settings[name] <= 1:
            raise ValueError(f"training.search.{name} must be in [0, 1]")


def validate_search_recovery(state, cfg, *, required=False):
    """Reject incompatible/incomplete teacher data before allocating a simulator."""
    if state is None:
        if required:
            raise ValueError(
                "Missing search recovery; change target protocols through fresh --init-from"
            )
        return
    if (
        not search_enabled(cfg)
        or state.get("protocol") != SEARCH_PROTOCOL
        or state.get("target_protocol") != TARGET_PROTOCOL
    ):
        raise ValueError("Search target/recovery protocol differs from saved workflow")
    if state.get("settings") != cfg["training"]["search"]:
        raise ValueError("Search settings differ from saved recovery")
    for name in ("rng", "records", "plans", "slots", "prefix", "behavior", "metrics"):
        if name not in state:
            raise ValueError(f"Incomplete search recovery: {name}")
    query = state.get("query")
    if query is None:
        return
    if any(key not in query for key in ("parked", "tree_bytes", "world_bytes")):
        raise ValueError("Incomplete search storage recovery")
    phase = query.get("phase")
    phases = (
        "queued",
        "pilot",
        "reference_frozen",
        "discovery",
        "confirmation",
        "ready",
        "action_committed",
    )
    if phase not in phases:
        raise ValueError("Invalid search recovery phase")
    candidates = query.get("candidates", [])
    if (
        len(candidates) < 2
        or len(set(candidates)) != len(candidates)
        or len(candidates) > cfg["training"]["search"]["root_candidates"]
    ):
        raise ValueError("Invalid search recovery candidate set")
    if phase in phases[3:]:
        reference = np.asarray(query.get("reference", []), np.float64)
        expected = len(candidates) * cfg["training"]["search"]["discovery_worlds"]
        if reference.shape != (expected,) or not np.isfinite(reference).all():
            raise ValueError("Incomplete frozen search duration reference")
    samples = np.asarray(query.get("samples"), np.float64)
    if (
        samples.shape != (len(candidates), cfg["training"]["search"]["confirmation_worlds"])
        or np.isinf(samples).any()
    ):
        raise ValueError("Invalid search confirmation panel")
    if phase == "ready" and (not np.isfinite(samples).all() or query.get("record") is None):
        raise ValueError("Unfinished confirmation cannot publish teacher labels")
    if len(query.get("results", [])) > cfg["training"]["search"]["rollouts"]:
        raise ValueError("Search recovery exceeds its rollout budget")
