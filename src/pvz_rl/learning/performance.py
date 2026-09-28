"""Execution-only settings; never replace a saved learning parameter."""

import copy

ENCODER_SETTINGS = ("encoder_microbatch", "encoder_token_budget", "attention_query_chunk")
PERFORMANCE_SETTINGS = ("fit_precision", "prefetch", "compile_kernels", "telemetry")
PERFORMANCE_DEFAULTS = dict(
    fit_precision="fp32", prefetch=False, compile_kernels=False, telemetry=False
)


def refresh_performance(cfg, current):
    result = copy.deepcopy(cfg)
    for key in ENCODER_SETTINGS:
        result["policy"][key] = current["policy"][key]
    target = result["training"].setdefault("performance", {})
    for key in PERFORMANCE_SETTINGS:
        target[key] = current["training"].get("performance", {}).get(key, PERFORMANCE_DEFAULTS[key])
    return result


def without_performance(cfg):
    result = copy.deepcopy(cfg)
    for key in ENCODER_SETTINGS:
        result["policy"].pop(key, None)
    for key in PERFORMANCE_SETTINGS:
        result["training"].get("performance", {}).pop(key, None)
    return result
