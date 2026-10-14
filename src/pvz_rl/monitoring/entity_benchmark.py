"""Bounded entity-cap measurements; synthetic inputs, no learning-quality claim.

Run: python -m pvz_rl.monitoring.entity_benchmark --output artifacts/entity-benchmark.json
"""

import argparse
import gc
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules
from pvz_game.types import ProjectileView, ZombieView

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder, collate_observations
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.provenance import write_json


def observation(cfg, count):
    """Public records with varied positions, states and health; never private slots."""
    plants = min(40, max(0, count - 5))
    zombies = min(75, max(0, count - 5 - plants))
    base = Game().reset(
        LevelSpec(
            "benchmark",
            plants=tuple(InitialPlant("peashooter", i // 9, i % 9) for i in range(plants)),
        )
    )
    public = replace(
        base,
        zombies=tuple(
            ZombieView(i, "basic", i % 5, 2000 + 83 * i, 270 - i, 0, "walking", 0, False, 0, False)
            for i in range(zombies)
        ),
        projectiles=tuple(
            ProjectileView(i, i % 5, 500 + i * 13, 20, bool(i % 2))
            for i in range(max(0, count - 5 - plants - zombies))
        ),
    )
    encoder = ObservationEncoder(cfg, Rules())
    encoded = encoder.encode(public)
    return encoded, encoder.last_truncation


def measure(cfg, count, repeats=3):
    torch.manual_seed(101)
    model = TwoStageTransformerPolicy(cfg).cuda()
    encoded, omitted = observation(cfg, count)
    batch = collate_observations([encoded] * 128, "cuda")
    optimizer = torch.optim.Adam(model.master_parameters(), lr=cfg["training"]["learning_rate"])
    branches = torch.ones(128, device="cuda", dtype=torch.long)

    def stage_values(stage):
        return (
            model.species_values(batch)
            if stage == "species"
            else model.conditional_tiles(batch, branches)
        )

    def stage_inference(stage):
        model.eval()
        with torch.no_grad():
            stage_values(stage)

    def stage_fitting(stage):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = (stage_values(stage)[:, 0] - 1).square().mean()
        loss.backward()
        model.accumulate_master_gradients()
        model.clear_runtime_grad(set_to_none=True)
        torch.nn.utils.clip_grad_norm_(
            model.master_parameters(), cfg["training"]["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        model.sync_runtime_from_masters()

    def inference():
        model.eval()
        with torch.no_grad():
            model.species_values(batch)
            model.conditional_tiles(batch, torch.ones(128, device="cuda", dtype=torch.long))

    def fitting():
        model.train()
        model.zero_grad(set_to_none=True)
        q = model.species_values(batch)
        tile_q = model.conditional_tiles(batch, torch.ones(128, device="cuda", dtype=torch.long))
        loss = (q[:, 0] - 1).square().mean() + (tile_q[:, 0] - 1).square().mean()
        loss.backward()
        model.accumulate_master_gradients()
        model.clear_runtime_grad(set_to_none=True)
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise FloatingPointError("Nonfinite crowded-entity fitting")
        optimizer.step()
        model.sync_runtime_from_masters()

    inference()
    fitting()
    result = dict(
        cap=cfg["encoding"]["max_entities"],
        present=count,
        retained=len(encoded["entities"]),
        omitted=omitted,
    )
    for name, operation in (
        ("inference", inference),
        ("species_inference", lambda: stage_inference("species")),
        ("tile_inference", lambda: stage_inference("tile")),
        ("species_fitting", lambda: stage_fitting("species")),
        ("tile_fitting", lambda: stage_fitting("tile")),
        ("fitting", fitting),
    ):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for _ in range(repeats):
            operation()
        torch.cuda.synchronize()
        result[name] = dict(
            seconds=perf_counter() - start,
            observations=repeats * 128,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
        result[name]["observations_per_second"] = repeats * 128 / result[name]["seconds"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        total_vram_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
        scope="Warmed synthetic inference and short fitting; excludes simulation; no quality estimate",
        rows=[],
    )
    for cap in (256, 512):
        for count in (5, 64, 256, 512, 1024):
            gc.collect()
            torch.cuda.empty_cache()
            cfg = load_config()
            cfg["encoding"]["max_entities"] = cap
            row = measure(cfg, count)
            report["rows"].append(row)
            print(row, flush=True)
            write_json(args.output, report)


if __name__ == "__main__":
    main()
