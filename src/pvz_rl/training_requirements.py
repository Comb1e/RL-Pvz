"""Training eligibility, separate from reading historical experiment metadata."""

from functools import lru_cache

from .config import research_config, simulator, validate_config

TRAINING_CONDITIONS = ("masked", "unmasked", "sparse", "mixed")


@lru_cache(maxsize=1)
def _cuda_probe():
    from .cuda_diagnostics import cuda_doctor

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
    if condition not in TRAINING_CONDITIONS or cfg["conditions"].get(condition, {}).get("hybrid"):
        raise ValueError("CPU/hybrid training was removed in 0.8.0; use a direct CUDA condition.")
    if condition not in cfg["conditions"]:
        raise ValueError(f"Missing training condition: {condition}")
    if simulator(cfg) != "cuda" or cfg["training"]["device"] != "cuda":
        raise ValueError(
            "Training and resume require simulation.backend=cuda and training.device=cuda. "
            "CPU training was removed in 0.8.0; start a fresh run with a retained CUDA recipe. "
            "Historical reports, recordings and compatible inference remain available."
        )
    if cfg["environment"].get("action_timing") != "per_tick":
        raise ValueError(
            "CUDA training requires per_tick actions; legacy timing is inference-only."
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
    result["conditions"] = {condition: cfg["conditions"][condition]}
    return result
