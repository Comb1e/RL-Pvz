"""Training eligibility, separate from reading historical experiment metadata."""

from functools import lru_cache

from pvz_rl.config import research_config, simulator, validate_config
from pvz_rl.envs.actions import ActionSchema
from pvz_rl.learning.curriculum import selected_stage
from pvz_rl.policy.sequential_q import ACTION_DISTRIBUTION

TRAINING_CONDITIONS = ("masked",)


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
                "two_stage_transformer_q_v1",
                "entity_v1",
                "complete_return_selective_invalid_v3",
                "complete_return_selective_invalid_v3",
            ),
        }
        and cfg.get("policy", {}).get("action_distribution") == ACTION_DISTRIBUTION
        and cfg.get("reward", {}).get("version") == "net_value_v1"
        and cfg.get("training", {}).get("discount_clock") == "simulation_ticks"
    )


def weight_model_config(cfg):
    """The previous objective shares the exact architecture and weight contract."""
    policy, train = cfg.get("policy", {}), cfg.get("training", {})
    return (
        policy.get("kind") == "two_stage_transformer_q_v1"
        and cfg.get("encoding", {}).get("version") == "entity_v1"
        and policy.get("action_distribution") == ACTION_DISTRIBUTION
        and train.get("method")
        in (
            "complete_return_sequential_no_probe_v1",
            "complete_return_selective_invalid_v2",
            "complete_return_selective_invalid_v3",
        )
        and train.get("objective", {}).get("protocol") == train.get("method")
        and cfg.get("reward", {}).get("version") == "net_value_v1"
    )


def inference_config(cfg):
    """Refresh retired learning metadata only for inference/weight validation.

    Resume never uses this adapter. It does not change network or encoding fields.
    """
    import copy

    if not weight_model_config(cfg):
        raise ValueError("Incompatible model/encoding or objective protocol weight contract")
    result = copy.deepcopy(cfg)
    if current_model_config(cfg):
        return result
    result["training"]["method"] = "complete_return_selective_invalid_v3"
    result["training"]["objective"]["protocol"] = "complete_return_selective_invalid_v3"
    result["training"]["objective"].setdefault("invalid_action_target", -1.0)
    result["training"]["objective"].setdefault("invalid_fit_group_size", 100)
    result["reward"].pop("invalid_plant_penalty", None)
    result["reward"].pop("empty_dig_penalty", None)
    return result


def require_supported_policy(cfg, condition="masked"):
    """Require the current complete-game CUDA collector protocol."""
    if (
        condition != "masked"
        or not cfg["conditions"].get(condition, {}).get("masked")
        or cfg["conditions"].get(condition, {}).get("hybrid")
        or not current_model_config(cfg)
    ):
        raise ValueError(
            "Retired policy or scheduler. Use the stateless two-stage Transformer, entity_v1, "
            "net_value_v1, sequential_q_unmasked_penalty_v1 and complete-game collection. "
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
    if (
        cfg["training"].get("objective", {}).get("protocol")
        != "complete_return_selective_invalid_v3"
    ):
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
    if cfg["environment"].get("action_timing") != "per_tick":
        raise ValueError(
            "CUDA training requires per_tick actions; legacy timing is inference-only."
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
    result["training"].pop("annotation", None)
    return result


def transfer_protocol(cfg, condition="masked"):
    """Compare engine, model interface and objective identity after checkpoint validation."""
    env, p = cfg["environment"], cfg["policy"]
    return {
        "engine": [cfg[k] for k in ("engine_commit", "engine_version", "engine_package_version")],
        "encoding": cfg["encoding"],
        "timing": {k: env[k] for k in ("action_timing", "decision_ticks", "cutoff_seconds")},
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
                "species_width",
                "species_layers",
                "species_heads",
                "species_feedforward",
                "tile_width",
                "tile_layers",
                "tile_heads",
                "tile_feedforward",
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
    result = transfer_protocol(cfg)
    for name in ("timing", "method", "return"):
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
