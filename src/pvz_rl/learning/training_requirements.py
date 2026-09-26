"""Training eligibility, separate from reading historical experiment metadata."""

from functools import lru_cache

from pvz_rl.config import research_config, simulator, validate_config
from pvz_rl.envs.actions import ActionSchema
from pvz_rl.learning.curriculum import selected_stage
from pvz_rl.policy.sequential_q import ACTION_DISTRIBUTION

TRAINING_CONDITIONS = ("masked",)


def current_model_config(cfg):
    return (
        cfg.get("policy", {}).get("kind") == "event_sequential_q_v1"
        and cfg.get("policy", {}).get("action_distribution") == ACTION_DISTRIBUTION
        and cfg.get("encoding", {}).get("version") == "event_v7"
        and cfg.get("reward", {}).get("version") == "net_value_v1"
        and cfg.get("training", {}).get("discount_clock") == "simulation_ticks"
        and cfg.get("training", {}).get("method") == "sequential_q_mc_v1"
    )


def require_supported_policy(cfg, condition="masked"):
    """One supported method for both model loading and learning."""
    if (
        condition != "masked"
        or not cfg["conditions"].get(condition, {}).get("masked")
        or cfg["conditions"].get(condition, {}).get("hybrid")
        or not current_model_config(cfg)
    ):
        raise ValueError(
            "Retired policy or scheduler. Models require event_sequential_q_v1, event_v7, net_value_v1, the sequential_plant_epsilon_v1 distribution and complete-game collection. Start fresh with configs/train.toml."
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
            "Start a fresh complete-game CUDA run with configs/train.toml."
        )
    if cfg["environment"].get("action_timing") != "per_tick":
        raise ValueError(
            "CUDA training requires per_tick actions; legacy timing is inference-only."
        )
    require_supported_policy(cfg, condition)
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
        _cuda_probe()


def resume_protocol(cfg, condition):
    """Unused historical condition definitions do not alter an individual run."""
    result = research_config(cfg)
    result.pop("profile", None)
    result["conditions"] = {condition: cfg["conditions"][condition]}
    return result


def transfer_protocol(cfg, condition="masked"):
    """Only the engine and input/action/network structure constrain weight reuse."""
    env, p = cfg["environment"], cfg["policy"]
    return {
        "engine": [cfg[k] for k in ("engine_commit", "engine_version", "engine_package_version")],
        "encoding": cfg["encoding"]["version"],
        "actions": ActionSchema.version,
        "action_distribution": p.get("action_distribution"),
        "action_history": "joint_embedding_v1",
        "board": {
            k: env[k]
            for k in (
                "rows",
                "cols",
                "bins",
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
                "plant_embedding",
                "state_embedding",
                "scalar_sizes",
                "channels",
                "memory",
            )
        },
        "heads": cfg["training"]["hidden_sizes"],
        "method": cfg["training"]["method"],
    }


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
