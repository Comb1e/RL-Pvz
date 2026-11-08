"""Execution-only hybrid settings; no framework or CUDA imports."""

import math

DEFAULTS = dict(
    interval_backend="legacy",
    auxiliary_hybrid=False,
    auxiliary_cpu_workers=4,
    auxiliary_worker_ram_mib=192,
    auxiliary_cuda_target_ms=2.0,
    auxiliary_vram_mib=4096,
    auxiliary_spill_gib=8,
    auxiliary_interval_lanes=32,
    auxiliary_teacher_lanes=128,
)


def enabled(cfg):
    from pvz_rl.config import absolute_outcomes

    settings = cfg["training"]["performance"]
    return absolute_outcomes(cfg) and (
        settings.get("auxiliary_hybrid", False)
        or settings.get("interval_backend", "legacy") == "packed_async"
    )


def validate_auxiliary_settings(settings):
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        if key == "interval_backend":
            valid = value in ("legacy", "packed_async")
        elif isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, float):
            valid = type(value) in (float, int) and math.isfinite(value) and value > 0
        else:
            valid = type(value) is int and value >= (0 if key == "auxiliary_cpu_workers" else 1)
        if not valid:
            raise ValueError(f"Invalid training.performance.{key}")
