"""Execution-only settings; never replace a saved learning parameter."""

import copy

POLICY_EXECUTION_GROUP = "execution"
PERFORMANCE_SETTINGS = (
    "interval_backend",
    "auxiliary_hybrid",
    "auxiliary_cpu_workers",
    "auxiliary_worker_ram_mib",
    "auxiliary_cuda_target_ms",
    "auxiliary_vram_mib",
    "auxiliary_spill_gib",
    "auxiliary_interval_lanes",
    "auxiliary_teacher_lanes",
    "fork_packed",
    "fork_threads",
    "fork_suffix_cache",
    "fork_cache_steps",
    "fork_cache_zombies",
    "fork_cache_projectiles",
    "fork_cache_vram_mib",
    "fit_precision",
    "prefetch",
    "attacker_prepare_workers",
    "attacker_worker_ram_mib",
    "attacker_prefetch",
    "attacker_cuda_warmup",
    "attacker_oracle_lanes",
    "attacker_oracle_vram_mib",
    "cherry_prepare_workers",
    "cherry_worker_ram_mib",
    "cherry_prefetch",
    "cherry_oracle",
    "cherry_cuda_warmup",
    "cherry_oracle_lanes",
    "cherry_oracle_vram_mib",
    "compile_collection",
    "compile_fitting",
    "telemetry",
    "telemetry_span_limit",
    "fit_batch_groups",
    "collection_graph_cache_entries",
    "collection_tile_buckets",
    "teacher_batched_transfers",
    "teacher_pipeline",
    "vectorized_effect_gather",
    "collection_vram_headroom_mib",
    "fit_encoder_microbatch",
    "fit_token_budget",
    "fit_species_microbatch",
    "fit_tile_microbatch",
    "fit_entity_buckets",
)


def refresh_performance(cfg, current):
    result = copy.deepcopy(cfg)
    result["policy"][POLICY_EXECUTION_GROUP] = copy.deepcopy(
        current["policy"][POLICY_EXECUTION_GROUP]
    )
    target = result["training"].setdefault("performance", {})
    for key in PERFORMANCE_SETTINGS:
        if key == "interval_backend" and key not in target:
            continue  # Saved runs without routing metadata retain legacy execution.
        if key in current["training"]["performance"]:
            target[key] = current["training"]["performance"][key]
        else:
            target.pop(key, None)
    result["training"]["torch_threads"] = current["training"]["torch_threads"]
    return result


def without_performance(cfg):
    result = copy.deepcopy(cfg)
    result["policy"].pop(POLICY_EXECUTION_GROUP, None)
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


def compilation_requested(performance):
    return performance.get("compile_collection", False) or performance.get("compile_fitting", False)
