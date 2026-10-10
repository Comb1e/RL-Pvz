"""Execution-only settings; never replace a saved learning parameter."""

import copy

ENCODER_SETTINGS = ("encoder_microbatch", "encoder_token_budget", "attention_query_chunk")
PERFORMANCE_SETTINGS = (
    "fit_precision",
    "prefetch",
    "compile_kernels",
    "telemetry",
    "fit_sequence_groups",
    "collection_graph_cache_entries",
    "collection_vram_headroom_mib",
    "probe_active_capacity",
    "probe_pending_capacity",
    "probe_steps_per_round",
)


def refresh_performance(cfg, current):
    result = copy.deepcopy(cfg)
    for key in ENCODER_SETTINGS:
        result["policy"][key] = current["policy"][key]
    target = result["training"].setdefault("performance", {})
    for key in PERFORMANCE_SETTINGS:
        target[key] = current["training"]["performance"][key]
    return result


def without_performance(cfg):
    result = copy.deepcopy(cfg)
    for key in ENCODER_SETTINGS:
        result["policy"].pop(key, None)
    result["training"].pop("performance", None)
    return result


def phase_memory(device, *, empty_cache=False):
    """Sample memory at a quiescent boundary after releasing every phase owner."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    with torch.cuda.device(device):
        if empty_cache:
            torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        return dict(
            allocated_mib=torch.cuda.memory_allocated(device) / 1024**2,
            reserved_mib=torch.cuda.memory_reserved(device) / 1024**2,
            free_mib=free / 1024**2,
            total_mib=total / 1024**2,
        )
