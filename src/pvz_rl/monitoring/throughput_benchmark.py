"""Bounded four-pass fitting controls for the current model and execution settings."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import psutil
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.monitoring.entity_benchmark import observation
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
            with model.fitting_precision(precision):
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
    args = parser.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        n_envs=128,
        batch_size=1024,
        chunk_length=256,
        n_epochs=4,
        formal_training=False,
        measurements=[],
    )
    for counts in ([5], [40], [5, 40, 128, 256], [256]):
        result["measurements"].extend(synthetic(cfg, counts))
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.output, result)


if __name__ == "__main__":
    main()
