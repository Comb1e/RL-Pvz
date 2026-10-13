"""Bounded two-stage current-board sensitivity diagnostic; never a mastery run."""

import argparse
import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.demo_initialization import load_demo_checkpoint
from pvz_rl.learning.training import build_model, vector_env
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.provenance import write_json


@torch.no_grad()
def sensitivity(policy, cfg):
    """Change one current public fact at a time without any historical inputs."""
    base, _ = observation(cfg, 85)
    variants = {"original": base}
    for name in (
        "zombie_position",
        "zombie_health",
        "projectiles",
        "sun",
        "cooldowns",
        "occupancy",
    ):
        changed = copy.deepcopy(base)
        entities = changed["entities"]
        zombie = np.flatnonzero((entities[:, 0] >= 9) & (entities[:, 0] <= 13))[0]
        if name == "zombie_position":
            entities[zombie, 3] = 999
        elif name == "zombie_health":
            entities[zombie, 4] = 80
        elif name == "projectiles":
            changed["entities"] = np.vstack((entities, [[14, 0, 0, 1900, 0, 0, 0, 0, 0, 0, 20]]))
            changed["globals"][16] += 1 / cfg["encoding"]["count_scale"]
        elif name == "sun":
            changed["globals"][0] += 100 / policy.layout.cost_scale
        elif name == "cooldowns":
            changed["globals"][6:14] = 1
        else:
            plant = np.flatnonzero((entities[:, 0] >= 1) & (entities[:, 0] <= 8))[0]
            changed["entities"] = np.delete(entities, plant, axis=0)
            changed["globals"][14] -= 1 / cfg["encoding"]["count_scale"]
        variants[name] = changed
    batch = collate_observations(variants.values(), next(policy.parameters()).device)
    policy.eval()
    tiles = policy.conditional_tiles(
        batch,
        torch.full((len(variants),), 2, device=batch.device, dtype=torch.long),
    )
    q, tile = (
        policy.species_values(batch).cpu().numpy(),
        tiles.cpu().numpy(),
    )
    return dict(
        branch_q={name: values.tolist() for name, values in zip(variants, q)},
        max_branch_delta={
            name: float(np.max(np.abs(values - q[0]))) for name, values in zip(variants, q)
        },
        max_tile_delta={
            name: float(np.max(np.abs(values - tile[0]))) for name, values in zip(variants, tile)
        },
    )


class DiagnosticCallback(BaseCallback):
    def __init__(self, cfg, output):
        super().__init__()
        self.cfg_diagnostic, self.output = cfg, output
        self.rows, self.passes = [], []
        self.last_report = perf_counter()
        self.last_pass = (0, 0)
        self.peak_host_bytes = 0

    def _on_step(self):
        self.log_progress()
        return True

    def log_progress(self):
        key = (self.model._n_updates, self.model._fit_epoch)
        if key != self.last_pass and self.model._fit_epoch:
            self.passes.append(
                dict(
                    cohort=key[0] + 1,
                    pass_index=key[1],
                    metrics={
                        k: float(v)
                        for k, v in self.model.logger.name_to_value.items()
                        if k.startswith("train/") and isinstance(v, (int, float, np.number))
                    },
                )
            )
            self.last_pass = key
        now = perf_counter()
        if now - self.last_report >= 30:
            self.peak_host_bytes = max(self.peak_host_bytes, psutil.Process().memory_info().rss)
            print(
                f"{self.output.name}: phase={self.model.phase}, games={self.model.training_games}, "
                f"decisions={self.model.num_timesteps}, pass={self.model._fit_epoch}",
                flush=True,
            )
            self.last_report = now

    def _on_rollout_end(self):
        self.rows.append(
            dict(
                cohort=self.model._n_updates,
                metrics=copy.deepcopy(self.model.cohort_metrics),
                current_board_sensitivity=sensitivity(self.model.policy, self.cfg_diagnostic),
                host_rss_bytes=psutil.Process().memory_info().rss,
                cupy_pool_bytes=self.model.env.cp.get_default_memory_pool().used_bytes(),
            )
        )
        write_json(self.output / "cohorts.json", dict(cohorts=self.rows, passes=self.passes))


def run(checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg["training"].update(n_envs=16, total_games=16, n_epochs=4)
    cfg["environment"]["cutoff_seconds"] = 30
    cfg["visualization"].update(enabled=False, live_enabled=False)
    cfg["simulation"]["profile"] = True
    torch.set_num_threads(cfg["training"]["torch_threads"])
    weights = None
    if checkpoint:
        source, _ = load_demo_checkpoint(checkpoint, cfg)
        weights = copy.deepcopy(source.state_dict())
        del source
    results = {}
    for name in ("two_stage_transformer",):
        local = copy.deepcopy(cfg)
        destination = output / name
        destination.mkdir()
        write_json(destination / "config.json", local)
        env = vector_env(local, "masked", 101)
        model = build_model(local, "masked", env, 101)
        if weights is not None:
            model.policy.load_state_dict(weights)
        model.trajectory_root = destination
        callback = DiagnosticCallback(local, destination)
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        try:
            initial = sensitivity(model.policy, local)
            model.learn(16, callback=callback)
            results[name] = dict(
                seconds=perf_counter() - started,
                games=model.training_games,
                cohorts=callback.rows,
                passes=callback.passes,
                initial_sensitivity=initial,
                peak_torch_bytes=torch.cuda.max_memory_allocated(),
                peak_host_bytes=callback.peak_host_bytes,
                compilation_status={
                    stage: encoder.compilation_status
                    for stage, encoder in zip(
                        ("species", "tile"), model.policy.encoders, strict=True
                    )
                },
            )
            write_json(output / "diagnostic.json", results)
        finally:
            if model._buffer is not None:
                model._buffer.close()
            env.close()
        del callback, model, env
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
