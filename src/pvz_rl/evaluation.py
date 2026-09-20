"""Common paired evaluation for learning and non-learning policies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from pvz_game.replay import verify_replay

from .config import digest, output_settings
from .env import PvZEnv
from .frozen_baseline import choose_action
from .progress import Phase, ProgressReporter
from .provenance import append_jsonl, verify_engine, write_json
from .scenarios import namespace_seed

BASELINES = ("wait", "random_legal", "heuristic", "random_strategy")


def select_action(env: PvZEnv, obs, *, policy=None, baseline=None, rng=None, masked=True):
    if baseline == "wait":
        return 0
    if baseline in ("random_legal", "random_strategy"):
        return int(rng.choice(np.flatnonzero(env.action_masks())))
    if baseline == "heuristic":
        return env.codec.encode(choose_action(env.public_board()))
    if policy is None:
        raise ValueError("Specify a baseline or trained policy")
    kwargs = {"action_masks": env.action_masks()} if masked else {}
    action, _ = policy.predict(obs, deterministic=True, **kwargs)
    return int(action)


def summarize(rows: list[dict]) -> dict:
    result = {}
    for family, level in sorted({(r["family"], r["level"]) for r in rows}):
        group = [r for r in rows if (r["family"], r["level"]) == (family, level)]
        n = len(group)
        wins = [r for r in group if r["win"]]
        losses = [r for r in group if r["status"] == "lost"]
        decisions = sum(r["decisions"] for r in group)
        result[f"{family}/{level}"] = {
            "episodes": n,
            "win_rate": len(wins) / n,
            "truncation_rate": sum(r["status"] == "truncated" for r in group) / n,
            "mean_mowers_used": float(np.mean([r["mowers_used"] for r in group])),
            "fraction_wins_without_mowers": (
                sum(r["mowers_used"] == 0 for r in wins) / len(wins) if wins else None
            ),
            "invalid_action_rate": sum(r["invalid_actions"] for r in group) / max(1, decisions),
            "waiting_rate": sum(r["wait_actions"] for r in group) / max(1, decisions),
            "win_seconds": float(np.mean([r["simulated_seconds"] for r in wins])) if wins else None,
            "loss_seconds": float(np.mean([r["simulated_seconds"] for r in losses]))
            if losses
            else None,
            "inference_ms": 1000 * sum(r["inference_seconds"] for r in group) / max(1, decisions),
        }
    return result


def evaluate(
    cfg: dict,
    *,
    seeds: list[int],
    levels: list[str],
    output: str | Path,
    policy=None,
    condition="masked",
    baseline=None,
    learner_seed=None,
    family="preset",
    split="validation",
    training_steps=0,
    record=False,
    progress=None,
    checkpoint_hash=None,
    replay_limit=None,
) -> list[dict]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if baseline is not None and baseline not in BASELINES:
        raise ValueError("Unknown baseline")
    env_condition = (
        "hybrid" if baseline == "random_strategy" else "masked" if baseline else condition
    )
    label = baseline or condition
    protocol_hash = digest(
        {
            "engine": verify_engine(cfg),
            "environment": cfg["environment"],
            "encoding": cfg["encoding"],
            "reward": cfg["reward"],
        }
    )
    training_hash = (
        None
        if baseline
        else digest(
            {
                "training": cfg["training"],
                "curriculum": cfg["curriculum"],
                "condition": cfg["conditions"][condition],
            }
        )
    )
    rows = []
    owns_progress = progress is None
    progress = progress or ProgressReporter(
        output / "evaluation.log", output_settings(cfg)["logging"]["progress_seconds"]
    )
    if owns_progress:
        progress.phase(Phase.VALIDATING)
    total = len(levels) * len(seeds)
    progress.emit(f"Evaluation: {label}, {family}/{split}, {total} games", force=True)
    limit = cfg["evaluation"]["replays_per_outcome"] if replay_limit is None else replay_limit
    try:
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for level in levels:
                quotas = Counter()
                env = PvZEnv(cfg, condition=env_condition, level=level, family=family)
                try:
                    for seed in sorted(seeds):
                        env.record = record and any(
                            quotas[k] < limit for k in ("won", "lost", "truncated")
                        )
                        obs, _ = env.reset(seed=seed)
                        rng = np.random.default_rng(namespace_seed(f"baseline/{label}", seed))
                        inference_seconds, started = 0.0, perf_counter()
                        while True:
                            t = perf_counter()
                            action = select_action(
                                env,
                                obs,
                                policy=policy,
                                baseline=baseline,
                                rng=rng,
                                masked=cfg["conditions"][condition]["masked"],
                            )
                            inference_seconds += perf_counter() - t
                            obs, _, terminated, truncated, info = env.step(action)
                            progress.emit(
                                f"Evaluation {len(rows)}/{total} complete; {level}, seed {seed}, "
                                f"game time {env.public.elapsed_seconds:.1f}s"
                            )
                            if terminated or truncated:
                                break
                        row = {
                            **info["episode_metrics"],
                            "policy": label,
                            "learner_seed": learner_seed,
                            "protocol_hash": protocol_hash,
                            "training_config_hash": training_hash,
                            "split": split,
                            "training_steps": training_steps,
                            "inference_seconds": inference_seconds,
                            "wall_seconds": perf_counter() - started,
                            "state_hash": env.game.state_hash(),
                            "checkpoint_hash": checkpoint_hash,
                        }
                        if env.recorder and quotas[row["status"]] < limit:
                            replay_path = (
                                output / "replays" / f"{level}-{seed}-{row['status']}.json"
                            )
                            env.recorder.save(replay_path)
                            verified = verify_replay(replay_path)
                            if verified.state_hash() != row["state_hash"]:
                                raise RuntimeError("Evaluation replay diverged")
                            row["replay"] = str(replay_path.resolve())
                            row["replay_verified"] = True
                            quotas[row["status"]] += 1
                        append_jsonl(stream, row)
                        rows.append(row)
                finally:
                    env.close()
        write_json(output / "summary.json", summarize(rows))
        write_json(output / "status.json", {"state": "complete", "episodes": len(rows)})
        progress.emit(f"Evaluation complete: {len(rows)}/{total} games", force=True)
        return rows
    except BaseException as exc:
        write_json(
            output / "status.json",
            {"state": "failed", "error": repr(exc), "completed_episodes": len(rows)},
        )
        raise
    finally:
        if owns_progress:
            progress.close()


def read_rows(paths) -> list[dict]:
    result = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            result.extend(json.loads(line) for line in stream if line.strip())
    return result
