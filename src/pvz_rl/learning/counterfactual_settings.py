"""Factual replay identities. Absent settings retain legacy return fitting."""

import math

PROTOCOL = "factual_tile_replay_v1"
OBJECTIVE = "paired_tile_cross_entropy_v1"


def counterfactual_settings(cfg, phase=None):
    settings = cfg["training"].get("counterfactual", {})
    stage = "peashooter" if phase == "peashooter" else "normal"
    result = {**settings, **settings.get(stage, {})}
    if stage == "peashooter" and "peashooter" not in settings and "enabled" not in settings:
        result.update(enabled=False, temperature=1.0)
    return result


def counterfactual_enabled(cfg, phase=None):
    from pvz_rl.envs.tile_scenarios import is_placement_skill

    return not is_placement_skill(phase) and counterfactual_settings(cfg, phase).get(
        "enabled", False
    )


def validate_counterfactual(cfg):
    settings = cfg["training"].get("counterfactual")
    if settings is None:
        return
    if settings.get("protocol") != PROTOCOL or settings.get("objective") != OBJECTIVE:
        raise ValueError("Unsupported factual counterfactual protocol/objective")
    for key in ("resident_envs", "scratch_vram_mib", "chunk_decisions"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"training.counterfactual.{key} must be a positive integer")
    for stage in ("peashooter", "normal"):
        resolved = counterfactual_settings(cfg, stage)
        if type(resolved.get("enabled")) is not bool:
            raise ValueError(f"{stage} counterfactual enablement must be boolean")
        temperature = resolved.get("temperature")
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError(f"{stage} comparison temperature must be finite and positive")
    gamma = cfg["training"]["gamma"]
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Counterfactual gamma must be in (0, 1]")
    # Prefix-dependent shaping uses inverse discounts at the query tick.
    from pvz_game import Rules

    ticks = cfg["environment"]["cutoff_seconds"] * Rules().game["tick_rate"]
    if -ticks * math.log(gamma) > 35:
        raise ValueError(
            "Counterfactual discount underflows over the episode cutoff; increase gamma"
        )


def reject_search_resume(state, cfg):
    """Old search archives remain readable only for portable weight transfer."""
    from pvz_rl.envs.tile_scenarios import is_placement_skill

    phase = (state.get("tile_scenarios") or {}).get("phase")
    if state.get("search_teacher") is not None or (
        cfg["training"].get("search", {}).get("enabled", False) and not is_placement_skill(phase)
    ):
        raise ValueError(
            "Slow search-teacher recovery is retired. Use --init-from CHECKPOINT "
            "to transfer its weights into a fresh factual-counterfactual run."
        )


def validate_counterfactual_recovery(state, cfg):
    reject_search_resume(state, cfg)
    payload = state.get("counterfactual")
    phase = (state.get("tile_scenarios") or {}).get("phase")
    required = state.get("buffer") is not None and counterfactual_enabled(cfg, phase)
    if payload is None:
        if required:
            raise ValueError("Missing factual counterfactual recovery; use fresh --init-from")
        return
    if payload.get("protocol") != PROTOCOL or payload.get("objective") != OBJECTIVE:
        raise ValueError("Unsupported factual counterfactual recovery")
    if payload.get("settings") != counterfactual_settings(cfg, phase):
        raise ValueError("Factual replay settings differ from saved objective")
    for key in (
        "roots",
        "jobs",
        "logs",
        "main_stats",
        "main_header",
        "main_ended",
        "identity",
        "seed",
        "records",
        "feasibility",
        "metrics",
    ):
        if key not in payload:
            raise ValueError(f"Incomplete factual replay recovery: {key}")
    import numpy as np

    logs, roots, jobs = payload["logs"], payload["roots"], payload["jobs"]
    stats, headers = np.asarray(payload["main_stats"]), np.asarray(payload["main_header"])
    if (
        stats.shape != (len(logs), 12)
        or headers.shape != (len(logs), 21)
        or not np.isfinite(stats).all()
    ):
        raise ValueError("Invalid factual main accounting dimensions")
    identities = set()
    for index, root in enumerate(roots):
        identity = tuple(root["identity"])
        if identity in identities or len(identity) != 3 or not 0 <= identity[0] < len(logs):
            raise ValueError("Duplicate or invalid factual root identity")
        identities.add(identity)
        if not 1 <= root["branch"] <= 8 or not 0 <= root["tile"] < 45:
            raise ValueError("Invalid factual root planting")
        for job_id in root["jobs"]:
            if not 0 <= job_id < len(jobs) or jobs[job_id]["root"] != index:
                raise ValueError("Factual branch/root identity mismatch")
    for job in jobs:
        if job["state"] not in (
            "queued",
            "running",
            "waiting_resources",
            "waiting_commands",
            "completed",
            "discarded",
        ):
            raise ValueError("Invalid factual branch state")
        if not 0 <= job["root"] < len(roots) or not 0 <= job["tile"] < 45:
            raise ValueError("Invalid factual alternative identity")
        slot = roots[job["root"]]["identity"][0]
        if not 0 <= job["cursor"] <= len(logs[slot]):
            raise ValueError("Invalid factual command cursor")
        if job["state"] == "completed":
            outcome = job.get("outcome")
            if not outcome or np.shape(outcome[0]) != (12,) or not np.isfinite(outcome[0]).all():
                raise ValueError("Missing completed factual accounting")
    recorded = set()
    for record in payload["records"]:
        identity = tuple(record["identity"])
        if (
            identity not in identities
            or identity in recorded
            or not np.isfinite(record["values"]).all()
        ):
            raise ValueError("Duplicate or invalid completed factual label")
        recorded.add(identity)
        if any(not (0 <= a < 45 and 0 <= b < 45 and a != b) for a, b in record["pairs"]):
            raise ValueError("Invalid factual preference pair")
