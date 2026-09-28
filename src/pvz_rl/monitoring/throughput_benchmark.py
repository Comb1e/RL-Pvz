"""Bounded four-pass throughput controls; run against either source revision."""

import argparse
import gc
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import psutil
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.monitoring.benchmark import measure
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.monitoring.hardware import HardwareMonitor
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy
from pvz_rl.provenance import write_json


def synthetic(cfg, counts):
    torch.manual_seed(101)
    model = TransformerLSTMPolicy(cfg).cuda().train()
    raw = [observation(cfg, count)[0] for count in counts]
    obs = collate_observations((raw * (1024 // len(raw))), "cuda").reshape(4, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    branches = torch.arange(1024, device="cuda") % 9 + 1

    def passes():
        for _ in range(cfg["training"]["n_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            precision = cfg["training"]["performance"].get("fit_precision", "fp32")
            context = (
                model.fitting_precision(precision)
                if hasattr(model, "fitting_precision")
                else nullcontext()
            )
            with context:
                q, tiles, contexts, _ = model.forward_sequence(obs, return_context=True)
                tile_q = model.tile_values(tiles.flatten(0, 1), contexts.flatten(0, 1), branches)
                loss = (q - 0.5).square().mean() + (tile_q + 0.5).square().mean()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()

    passes()
    torch.cuda.synchronize()
    rows = []
    for repeat in range(3):
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        passes()
        torch.cuda.synchronize()
        seconds = perf_counter() - started
        rows.append(
            dict(
                repeat=repeat,
                entities=counts,
                seconds=seconds,
                passes=4,
                frames=4096,
                frames_per_second=4096 / seconds,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                host_rss_mib=psutil.Process().memory_info().rss / 2**20,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("synthetic", "collection"), required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    result = dict(
        label=args.label,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        scope=args.scope,
        n_envs=128,
        batch_size=1024,
        chunk_length=256,
        n_epochs=4,
        formal_training=False,
        measurements=[],
    )
    if args.scope == "synthetic":
        for counts in ([5], [40], [5, 40, 128, 256], [256]):
            result["measurements"].extend(synthetic(cfg, counts))
            gc.collect()
            torch.cuda.empty_cache()
            write_json(args.output, result)
    else:
        cfg["environment"]["cutoff_seconds"] = 1
        cfg["training"].update(total_games=100000, until_stage_complete=False)
        cfg["simulation"]["profile"] = True
        cfg["visualization"].update(enabled=False, live_enabled=True)
        for live in (False, True):
            for repeat in range(3):
                directory = args.output.parent / f"{args.label}-viewer-{live}-{repeat}"
                directory.mkdir(exist_ok=True, parents=True)
                with HardwareMonitor(directory / "hardware.jsonl", device="cuda") as monitor:
                    row = measure(cfg, 101, 12800, directory, live_view=live, load_monitor=monitor)
                samples = list(monitor.samples)
                utilization = [
                    s["gpu_percent"] for s in samples if s.get("gpu_percent") is not None
                ]
                import cupy

                row.update(
                    gpu_utilization_mean=sum(utilization) / len(utilization)
                    if utilization
                    else None,
                    sampled_host_peak_mib=max(
                        (s.get("process_ram_mib") or 0 for s in samples), default=0
                    ),
                    cupy_reserved_mib=cupy.get_default_memory_pool().total_bytes() / 2**20,
                )
                row.update(
                    repeat=repeat,
                    viewer=live,
                    host_rss_mib=psutil.Process().memory_info().rss / 2**20,
                )
                result["measurements"].append(row)
                write_json(args.output, result)
                print(
                    f"{args.label}: viewer={live}, repeat={repeat}, {row['decisions_per_second']:.1f} decisions/s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
