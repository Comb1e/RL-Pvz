"""Training eligibility, separate from reading historical experiment metadata."""

from dataclasses import asdict, dataclass
from functools import lru_cache

from pvz_rl.config import research_config, simulator, validate_config
from pvz_rl.envs.actions import ActionSchema
from pvz_rl.learning.curriculum import selected_stage
from pvz_rl.policy.sequential_q import ACTION_DISTRIBUTION
from pvz_rl.provenance import weight_engine_config

TRAINING_CONDITIONS = ("masked",)
CURRENT_OBJECTIVE = "complete_return_reach_v1"
CURRENT_REWARD = "action_global_value_v5"
FIT_SCHEDULE_PROTOCOL = "joint_shared_reach_cohorts_v1"


@dataclass(frozen=True)
class CohortFitSchedule:
    cohort: int
    cycle_position: int
    cycle_cohorts: int
    stage: str
    passes: int

    def summary(self):
        return dict(protocol=FIT_SCHEDULE_PROTOCOL, **asdict(self))


def fitting_schedule(cfg, cohort):
    """Choose one fitting stage for an entire one-based collection cohort."""
    if type(cohort) is not int or cohort < 1:
        raise ValueError("Fitting schedule requires a positive integer cohort number")
    return CohortFitSchedule(cohort, 1, 1, "joint", cfg["training"]["n_epochs"])


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
                "shared_board_reach_q_v1",
                "entity_v1",
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
            "Incompatible old model. Start the shared board encoder fresh; no weight migration is supported."
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
            "Retired policy or scheduler. Use the shared board encoder and conditional decoder, entity_v1, "
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
    for key in ("total_games", "total_steps", "max_minutes", "finalization_minutes"):
        result["training"].pop(key, None)
    # Residual-sun passivity shaping is a cohort-scoped reward update. The
    # current settings are applied at the next collection boundary, so they do
    # not invalidate the optimizer or simulator state of an older run.
    result.get("reward", {}).pop("passivity", None)
    # This trigger changes only at a collection boundary; probability schedules
    # and every other learning setting remain part of strict resume identity.
    result["training"]["exploration"].pop("plant_wait_check_seconds", None)
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
            k: p[k]
            for k in (
                "kind",
                "shared_width",
                "encoder_blocks",
                "tile_decoder_blocks",
                "attention_heads",
                "feedforward_width",
            )
            if k in p
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
