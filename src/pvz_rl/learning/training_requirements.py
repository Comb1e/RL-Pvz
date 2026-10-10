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
    )
    return (
        identity
        in {
            ("transformer_lstm_q_v4", "entity_v1", "complete_return_joint_tile_probe_v1"),
        }
        and cfg.get("policy", {}).get("action_distribution") == ACTION_DISTRIBUTION
        and cfg.get("reward", {}).get("version") == "net_value_v1"
        and cfg.get("training", {}).get("discount_clock") == "simulation_ticks"
    )


def require_supported_policy(cfg, condition="masked"):
    """Require the current complete-game CUDA collector protocol."""
    if (
        condition != "masked"
        or not cfg["conditions"].get(condition, {}).get("masked")
        or cfg["conditions"].get(condition, {}).get("hybrid")
        or not current_model_config(cfg)
    ):
        raise ValueError(
            "Retired policy or scheduler. Models require the entity_v1 recurrent protocol, net_value_v1, the joint_q_unmasked_penalty_v1 distribution and complete-game collection. Start fresh with the bundled train profile."
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
        != "complete_return_joint_tile_probe_v1"
    ):
        raise ValueError(
            "Old-objective checkpoints cannot resume or initialize weights. Refit a verified demonstration in a new directory."
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
        if (
            cfg["training"].get("performance", {}).get("fit_precision", "fp32") == "features_bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 fitting is unavailable; set training.performance.fit_precision=fp32"
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
    return result


def transfer_protocol(cfg, condition="masked"):
    """Only the engine and input/action/network structure constrain weight reuse."""
    env, p = cfg["environment"], cfg["policy"]
    return {
        "engine": [cfg[k] for k in ("engine_commit", "engine_version", "engine_package_version")],
        "encoding": cfg["encoding"],
        "timing": {k: env[k] for k in ("action_timing", "decision_ticks", "cutoff_seconds")},
        "actions": ActionSchema.version,
        "action_distribution": p.get("action_distribution"),
        "event_history": p["history"],
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
                "entity_width",
                "transformer_layers",
                "transformer_heads",
                "transformer_feedforward",
                "scalar_width",
                "lstm_hidden",
                "event_width",
                "chunk_length",
            )
            if k in p
        },
        "method": cfg["training"]["method"],
        "return": [cfg["training"]["gamma"], cfg["training"]["discount_clock"]],
    }


def weight_transfer_protocol(cfg):
    """Describe only the public model interface required by demo weights.

    A demonstration is re-fit under the current objective and execution
    profile. Its weights require the same entity/action representation and
    network dimensions, but do not require matching reward, return, optimizer,
    or recurrent chunk settings.
    """
    result = transfer_protocol(cfg)
    for name in ("timing", "method", "return"):
        result.pop(name)
    result["policy"].pop("chunk_length", None)
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
