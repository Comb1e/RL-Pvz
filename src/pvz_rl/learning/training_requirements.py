"""Training eligibility, separate from reading historical experiment metadata."""

from dataclasses import asdict, dataclass
from functools import lru_cache

from pvz_rl.config import (
    cohorts_per_epoch,
    outcome_prediction_weights,
    research_config,
    simulator,
    stage_epochs,
    validate_config,
)
from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.tile_scenarios import attacker_mix
from pvz_rl.learning.curriculum import (
    COHORT_PERFECT_STAGES,
    STAGES,
    TILE_STAGES,
    selected_stage,
)
from pvz_rl.policy.sequential_q import ACTION_DISTRIBUTION
from pvz_rl.provenance import weight_engine_config

TRAINING_CONDITIONS = ("masked",)
CURRENT_OBJECTIVE = "nstep_action_outcomes_v5"
CURRENT_REWARD = "action_global_value_v5"
FIT_SCHEDULE_PROTOCOL = "joint_nstep_action_outcome_cohorts_v2"


@dataclass(frozen=True)
class CohortFitSchedule:
    cohort: int
    cycle_position: int
    cycle_cohorts: int
    stage: str
    passes: int | None
    until_cohort_perfect: bool = False

    def summary(self):
        return dict(protocol=FIT_SCHEDULE_PROTOCOL, **asdict(self))


def cohort_place(cfg, cohort):
    """Epoch, place, and whether this one-based cohort is the memorized one.

    Places ``1..cohorts_per_epoch`` take one pass. The next place fits until
    the cohort is fully correct. ``None`` when the count is absent: those
    cohorts take one pass.
    """
    per = cohorts_per_epoch(cfg)
    if per is None:
        return None
    if type(cohort) is not int or cohort < 1:
        raise ValueError("Fitting schedule requires a positive integer cohort number")
    cycle = per + 1
    place = (cohort - 1) % cycle + 1
    return (cohort - 1) // cycle + 1, place, place == cycle


def fixed_passes(schedule):
    """Pass count for a stage that still fits a fixed number of epochs."""
    if schedule.until_cohort_perfect or schedule.passes is None:
        raise ValueError(
            "prediction, cherry_bomb, cherry_bomb_obstacle, attacker_pre, and attacker "
            "fit until the cohort is fully correct"
        )
    return schedule.passes


def cohort_epoch_stage(cfg, tile_phase=None, curriculum_state=None):
    """Stage whose fitting passes apply to this cohort."""
    if tile_phase in TILE_STAGES:
        return tile_phase
    chosen = selected_stage(cfg)
    if chosen in STAGES:
        return chosen
    if isinstance(curriculum_state, dict):
        index = curriculum_state.get("stage")
        if type(index) is int and 0 <= index < len(STAGES):
            return STAGES[index]
    if chosen in TILE_STAGES:
        return chosen
    return "easy"


def fitting_schedule(cfg, cohort, stage=None):
    """Choose one fitting stage for an entire one-based collection cohort."""
    if type(cohort) is not int or cohort < 1:
        raise ValueError("Fitting schedule requires a positive integer cohort number")
    if stage is None:
        stage = cohort_epoch_stage(cfg)
    if stage in COHORT_PERFECT_STAGES:
        placed = cohort_place(cfg, cohort)
        if placed is None:
            return CohortFitSchedule(cohort, 1, 1, "joint", 1, False)
        _epoch, place, memorized = placed
        cycle = cohorts_per_epoch(cfg) + 1
        if memorized:
            return CohortFitSchedule(cohort, place, cycle, "joint", None, True)
        return CohortFitSchedule(cohort, place, cycle, "joint", 1, False)
    return CohortFitSchedule(cohort, 1, 1, "joint", stage_epochs(cfg, stage))


def current_model_config(cfg):
    identity = (
        cfg.get("policy", {}).get("kind"),
        cfg.get("encoding", {}).get("version"),
        cfg.get("training", {}).get("method"),
        cfg.get("training", {}).get("objective", {}).get("protocol"),
    )
    return (
        identity
        in {
            (
                "shared_public_outcomes_q_v11",
                "entity_public_v4",
                CURRENT_OBJECTIVE,
                CURRENT_OBJECTIVE,
            ),
        }
        and cfg.get("policy", {}).get("action_distribution") == ACTION_DISTRIBUTION
        and cfg.get("reward", {}).get("version") == CURRENT_REWARD
        and cfg.get("training", {}).get("discount_clock") == "simulation_ticks"
    )


def weight_model_config(cfg):
    """Only the shared-encoder family is compatible; no conversion is defined."""
    return current_model_config(cfg)


def inference_config(cfg):
    """Validate this architecture before adapting execution-only engine metadata."""
    if not current_model_config(cfg):
        raise ValueError(
            "Incompatible old model. Old checkpoints are incompatible with the outcome-only architecture; start fresh."
        )
    return weight_engine_config(cfg)


def require_supported_policy(cfg, condition="masked"):
    """Require the current complete-game CUDA collector protocol."""
    if (
        condition != "masked"
        or not cfg["conditions"].get(condition, {}).get("masked")
        or cfg["conditions"].get(condition, {}).get("hybrid")
        or not current_model_config(cfg)
    ):
        raise ValueError(
            "Retired policy or scheduler. Use the shared public encoder and conditional decoder, entity_public_v4, "
            "action_global_value_v5 and complete-game collection. "
            "Start fresh with the bundled train profile."
        )


@lru_cache(maxsize=1)
def _cuda_probe():
    from pvz_rl.monitoring.cuda_diagnostics import cuda_doctor

    result = cuda_doctor()
    if not result["available"]:
        raise RuntimeError(
            "CUDA training is unavailable: "
            + result["error"]
            + ". Run pvz-rl doctor and install the locked CUDA dependencies with tools/bootstrap.ps1."
        )


def require_cuda_training(cfg, condition="masked", *, runtime=True):
    """Reject unsupported runs before creating output or allocating collectors."""
    validate_config(cfg)
    if cfg["training"].get("objective", {}).get("protocol") != CURRENT_OBJECTIVE:
        raise ValueError(
            "Old-objective training cannot resume; use compatible BF16-master weights "
            "only through fresh --init-from."
        )
    if condition == "hybrid" or cfg["conditions"].get(condition, {}).get("hybrid"):
        raise ValueError("CPU/hybrid training was removed in 0.8.0; use a direct CUDA condition.")
    if condition not in cfg["conditions"]:
        raise ValueError(f"Missing training condition: {condition}")
    if selected_stage(cfg) and (
        not cfg["conditions"][condition]["curriculum"] or not cfg["conditions"][condition]["masked"]
    ):
        raise ValueError("Stage training requires a masked teaching-curriculum condition")
    if simulator(cfg) != "cuda" or cfg["training"]["device"] != "cuda":
        raise ValueError(
            "Training and resume require simulation.backend=cuda and training.device=cuda. "
            "CPU training is not supported. "
            "Start a fresh complete-game CUDA run with the bundled train profile."
        )
    if (
        cfg["conditions"][condition]
        != {"masked": True, "shaped": True, "curriculum": True, "hybrid": False}
        or cfg["curriculum"].get("mode") != "teaching"
    ):
        raise ValueError(
            "Use the single masked teaching method; adjust its reward coefficients and curriculum parameters instead of selecting retired training modes"
        )
    if runtime:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training is unavailable. Run pvz-rl doctor and install the locked CUDA "
                "dependencies with tools/bootstrap.ps1; CPU training is no longer supported."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 runtime inference and fitting are unavailable on this CUDA device"
            )
        _cuda_probe()
    require_supported_policy(cfg, condition)


def resume_protocol(cfg, condition):
    """Unused historical condition definitions do not alter an individual run."""
    from pvz_rl.learning.performance import without_performance

    result = without_performance(research_config(cfg))
    result.pop("profile", None)
    result["conditions"] = {condition: cfg["conditions"][condition]}
    # Equal split coefficients preserve saved outcome-only single-weight runs.
    result["training"]["stage_outcome_weights"] = {
        stage: dict(
            zip(("static_field", "probe"), outcome_prediction_weights(cfg, stage), strict=True)
        )
        for stage in ("normal", *TILE_STAGES)
        if stage != "attacker" or attacker_mix(cfg) is not None
    }
    result["training"].pop("effect_prediction_weight", None)
    result["training"].pop("stage_effect_weights", None)
    for key in ("total_games", "total_steps", "max_minutes", "finalization_minutes"):
        result["training"].pop(key, None)
    # Residual-sun passivity shaping is a cohort-scoped reward update. The
    # current settings are applied at the next collection boundary, so they do
    # not invalidate the optimizer or simulator state of an older run.
    result.get("reward", {}).pop("passivity", None)
    # This trigger changes only at a collection boundary; probability schedules
    # and every other learning setting remain part of strict resume identity.
    result["training"]["exploration"].pop("plant_wait_check_seconds", None)
    result["training"].get("forecasts", {}).pop("injection_active", None)
    # Only the Cherry probe interval may refresh; pending probes still run
    # against their saved weights before any further fitting.
    result["curriculum"].pop("cherry_bomb_probe_interval_games", None)
    return result


def transfer_protocol(cfg, condition="masked"):
    """Compare engine, model interface and objective identity after checkpoint validation."""
    env, p = cfg["environment"], cfg["policy"]
    return {
        "engine": [cfg[k] for k in ("engine_commit", "engine_version", "engine_package_version")],
        "encoding": cfg["encoding"],
        "timing": {
            k: env[k]
            for k in ("action_frequency_hz", "action_timing", "decision_ticks", "cutoff_seconds")
            if k in env
        },
        "actions": ActionSchema.version,
        "action_distribution": p.get("action_distribution"),
        "board": {
            k: env[k]
            for k in (
                "rows",
                "cols",
                "plants",
                "zombies",
                "plant_states",
                "zombie_states",
                "mower_states",
            )
        },
        "policy": {
            "kind": p["kind"],
            "head_dim": p["head_dim"],
            "board": dict(p["board"]),
            "action": dict(p["action"]),
            "tile": dict(p["tile"]),
            "effect": dict(p["effect"]),
        },
        "method": cfg["training"]["method"],
        "return": [cfg["training"]["gamma"], cfg["training"]["discount_clock"]],
    }


def weight_transfer_protocol(cfg):
    """Describe only the public model interface required by demo weights.

    Checkpoint validation separately requires the current objective. Its
    demonstration weights require the same entity/action representation and
    network dimensions, but do not require matching reward, return, optimizer,
    or execution batch settings.
    """
    result = transfer_protocol(weight_engine_config(cfg))
    for name in ("timing", "method", "return", "action_distribution"):
        result.pop(name)
    return result


def parameter_changes(old, new, prefix=""):
    """Flat, reviewable differences for explicit weights-only transfers."""
    changes = {}
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(a, dict) and isinstance(b, dict):
            changes.update(parameter_changes(a, b, name))
        elif a != b:
            changes[name] = {"before": a, "after": b}
    return changes
