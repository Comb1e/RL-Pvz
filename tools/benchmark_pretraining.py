"""Cold private-pretraining sweeps, including setup, exact work and fitting order."""

import argparse
import gc
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter


def measure(base, setting, args):
    import numpy as np
    import torch

    from pvz_rl.config import digest
    from pvz_rl.envs.cuda_snapshots import download_snapshots
    from pvz_rl.learning.forecast_pretraining import ForecastPretraining
    from pvz_rl.learning.training import build_model, initial_weights, vector_env

    cfg = deepcopy(base)
    cfg["training"]["forecasts"].update(setting)
    cfg["training"]["performance"].update(
        {k: v for k, v in setting.items() if k.startswith("fork_")}
    )
    cfg["training"]["forecasts"]["pretrain_roots"] = args.roots
    cfg["training"].update(n_envs=args.envs, random_cohort_seeds=False)
    cfg["training"]["peashooter_rules"]["n_envs"] = args.envs
    if args.eager:
        cfg["training"]["performance"].update(compile_collection=False, compile_fitting=False)
    torch.manual_seed(args.seed)
    env = vector_env(cfg, "masked", args.seed)
    model = build_model(cfg, "masked", env, args.seed)
    owner = None
    try:
        if args.weights:
            weights, _ = initial_weights(args.weights, cfg)
            model.policy.load_state_dict(weights)
        model._ensure_teacher()
        torch.cuda.synchronize()
        started = perf_counter()
        owner = ForecastPretraining(model)
        signatures = []
        last_fit = None
        while True:
            if args.audit and owner.phase == "fit":
                key = (owner.cursor, owner.fit_pass)
                if key != last_fit:
                    if owner.fit_pass == 0:
                        if hasattr(owner, "labels"):
                            at = owner.fit_offset
                            count = min(32, owner.count - at)
                            raw = owner.labels[at : at + count].get()
                            signatures.append(
                                digest(
                                    dict(
                                        roots=download_snapshots(
                                            owner.source, indices=range(at, at + count)
                                        ),
                                        actions=owner.rows["actions"][at : at + count].tolist(),
                                        valid=owner.rows["valid"][at : at + count].tolist(),
                                        targets=raw.tolist(),
                                    )
                                )
                            )
                        else:
                            from pvz_rl.envs.static_fork import reference_tau
                            from pvz_rl.learning.forecast_labels import outcome_codes

                            labels = np.concatenate(
                                (
                                    outcome_codes(
                                        owner.baseline, reference_tau(owner.source.rules)
                                    )[:, None],
                                    owner.rows["effect_labels"],
                                ),
                                axis=1,
                            ).astype(np.int64)
                            signatures.append(
                                digest(
                                    dict(
                                        roots=download_snapshots(
                                            owner.source, indices=range(owner.count)
                                        ),
                                        actions=owner.rows["effect_actions"].tolist(),
                                        valid=owner.rows["effect_valid"].tolist(),
                                        targets=labels.tolist(),
                                    )
                                )
                            )
                    last_fit = key
            if owner.advance():
                break
        torch.cuda.synchronize()
        metrics = dict(owner.metrics)
        cleanup_started = perf_counter()
        owner.close()
        elapsed = perf_counter() - started
        cleanup_seconds = perf_counter() - cleanup_started
        state_hash = hashlib.sha256()
        for name, tensor in model.policy.state_dict().items():
            state_hash.update(name.encode())
            state_hash.update(tensor.float().cpu().numpy().tobytes())
        result = dict(
            settings=setting,
            resolved_settings={
                k: v for k, v in cfg["training"]["forecasts"].items() if k.startswith("pretrain_")
            },
            retained_main_envs=args.envs,
            seed=args.seed,
            initial_weights=str(args.weights) if args.weights else "random",
            roots=args.roots,
            seconds=elapsed,
            seconds_per_root=elapsed / max(1, args.roots),
            target_seconds=10,
            target_met=elapsed <= 10,
            metrics=metrics,
            signatures=signatures,
            weight_signature=state_hash.hexdigest(),
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            audit_overhead_included=args.audit,
            cleanup_seconds=cleanup_seconds,
        )
        if args.save_weights:
            path = args.output.with_suffix(
                f".{cfg['training']['forecasts'].get('pretrain_execution', 'legacy')}-{digest(setting)[:8]}.weights.pt"
            )
            torch.save(model.policy.state_dict(), path)
            result["weights"] = str(path)
        return result
    finally:
        if owner is not None:
            owner.close()
        env.close()
        del owner, model, env
        gc.collect()
        torch.cuda.empty_cache()
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, help="JSON list of training.forecasts overrides")
    parser.add_argument("--roots", type=int, default=2048)
    parser.add_argument(
        "--envs", type=int, default=128, help="Retained main-environment allocation"
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--gpu-profile", default="rtx4070-laptop")
    parser.add_argument(
        "--audit", action="store_true", help="Download and hash each panel; included in time"
    )
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    import torch

    from pvz_rl.config import load_config
    from pvz_rl.gpu import apply_gpu_profile, select_gpu

    cfg, _ = apply_gpu_profile(load_config(), args.gpu_profile, device=select_gpu("0"))
    cfg["curriculum"]["run_stage"] = "peashooter"
    cfg["visualization"].update(enabled=False, live_enabled=False, demos=False, videos=False)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    settings = json.loads(args.settings.read_text()) if args.settings else [{}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for setting in settings:
        torch.cuda.reset_peak_memory_stats()
        result = measure(cfg, setting, args)
        results.append(result)
        args.output.write_text(json.dumps(dict(samples=results), indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
