"""Common paired evaluation for learning and non-learning policies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from pvz_rl.config import digest, output_settings, simulator
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.rewards import LEDGER_METRICS, REWARD_METRICS
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.evaluation.frozen_baseline import choose_action
from pvz_rl.learning.deadline import BudgetExpired, check_deadline
from pvz_rl.monitoring.progress import Phase, ProgressReporter
from pvz_rl.presentation.recordings import verify_replay
from pvz_rl.provenance import append_jsonl, file_hash, verify_engine, write_json

BASELINES = ("wait", "random_legal", "heuristic", "random_strategy")


def select_action(env: PvZEnv, obs, *, policy=None, baseline=None, rng=None, masked=True):
    if baseline == "wait":
        return 0
    if baseline in ("random_legal", "random_strategy"):
        mask = env.engine_action_masks() if baseline == "random_legal" else env.action_masks()
        return int(rng.choice(np.flatnonzero(mask)))
    if baseline == "heuristic":
        return env.codec.encode(choose_action(env.public_board()))
    if policy is None:
        raise ValueError("Specify a baseline or trained policy")
    kwargs = {"action_masks": env.action_masks()} if masked else {}
    state = None if env.metrics["decisions"] == 0 else getattr(env, "_policy_memory", None)
    action, env._policy_memory = policy.predict(obs, state=state, deterministic=True, **kwargs)
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
            **{
                f"mean_{key}": float(np.mean([r[key] for r in group]))
                if all(key in r for r in group)
                else None
                for key in (*REWARD_METRICS, *LEDGER_METRICS)
            },
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
    deadline=None,
) -> list[dict]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if baseline is not None and baseline not in BASELINES:
        raise ValueError("Unknown baseline")
    env_condition = (
        "hybrid" if baseline == "random_strategy" else "masked" if baseline else condition
    )
    profile = cfg.get("profile", "baseline")
    training_games = getattr(policy, "training_games", 0) if policy is not None else 0
    label = baseline or (condition if profile == "baseline" else f"{profile}/{condition}")
    engine = verify_engine(cfg)
    protocol_hash = digest(
        {
            "engine": engine,
            "environment": cfg["environment"],
            "encoding": cfg["encoding"],
            "reward": cfg["reward"],
            "discount_clock": cfg["training"]["discount_clock"],
            "gamma": cfg["training"]["gamma"],
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
                "encoding": cfg["encoding"],
                "policy": cfg.get("policy", {"kind": "flat"}),
                "simulator": simulator(cfg),
            }
        )
    )
    game_protocol_hash = digest(
        {"engine": engine, "environment": cfg["environment"], "reward": cfg["reward"]}
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
        gpu_rows = None
        if policy is not None and simulator(cfg) == "cuda":
            from pvz_rl.evaluation.cuda_evaluation import batched_games

            gpu_rows = iter(
                batched_games(
                    cfg,
                    policy,
                    condition,
                    seeds,
                    levels,
                    family,
                    record=record,
                    progress=progress,
                    deadline=deadline,
                )
            )
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for level in levels:
                quotas = Counter()
                replay_metadata = {
                    "policy_id": label,
                    "experiment": {
                        "learner_seed": learner_seed,
                        "training_steps": training_steps,
                        "training_games": training_games,
                        "split": split,
                        "engine": engine,
                        "protocol_hash": protocol_hash,
                        "game_protocol_hash": game_protocol_hash,
                        "profile": profile,
                        "training_config_hash": training_hash,
                    },
                }
                if checkpoint_hash:
                    replay_metadata["checkpoint_sha256"] = checkpoint_hash
                env = PvZEnv(
                    cfg,
                    condition=env_condition,
                    level=level,
                    family=family,
                    replay_metadata=replay_metadata,
                )
                try:
                    for seed in sorted(seeds):
                        check_deadline(deadline)
                        env.record = record and any(
                            quotas[k] < limit for k in ("won", "lost", "truncated")
                        )
                        gpu = next(gpu_rows) if gpu_rows is not None else None
                        obs, _ = env.reset(seed=seed)
                        rng = np.random.default_rng(namespace_seed(f"baseline/{label}", seed))
                        inference_seconds, started = 0.0, perf_counter()
                        trace_index = 0
                        while gpu is None or env.record:
                            check_deadline(deadline)
                            t = perf_counter()
                            action = (
                                gpu["action_trace"][trace_index]
                                if gpu is not None
                                else select_action(
                                    env,
                                    obs,
                                    policy=policy,
                                    baseline=baseline,
                                    rng=rng,
                                    masked=cfg["conditions"][condition]["masked"],
                                )
                            )
                            trace_index += 1
                            inference_seconds += perf_counter() - t
                            obs, _, terminated, truncated, info = env.step(action)
                            if (
                                not info["accepted"]
                                and getattr(env, "_policy_memory", None) is not None
                            ):
                                env._policy_memory.previous_actions.zero_()
                            progress.emit(
                                f"Evaluation {len(rows)}/{total} complete; {level}, seed {seed}, "
                                f"game time {env.public.elapsed_seconds:.1f}s"
                            )
                            if terminated or truncated:
                                break
                        if gpu is not None:
                            if env.record and (
                                env.game.state_hash() != gpu["state_hash"]
                                or info["episode_metrics"]["status"] != gpu["status"]
                                or trace_index != gpu["decisions"]
                            ):
                                raise RuntimeError(
                                    "CUDA demonstration diverged from CPU reference; recording not saved"
                                )
                            info = {
                                "episode_metrics": {
                                    k: v
                                    for k, v in gpu.items()
                                    if k
                                    not in (
                                        "action_trace",
                                        "state_hash",
                                        "inference_seconds",
                                        "wall_seconds",
                                    )
                                }
                            }
                            inference_seconds = gpu["inference_seconds"]
                        row = {
                            **info["episode_metrics"],
                            "policy": label,
                            "learner_seed": learner_seed,
                            "protocol_hash": protocol_hash,
                            "game_protocol_hash": game_protocol_hash,
                            "profile": profile,
                            "training_config_hash": training_hash,
                            "split": split,
                            "training_steps": training_steps,
                            "training_games": training_games,
                            "inference_seconds": inference_seconds,
                            "wall_seconds": gpu["wall_seconds"]
                            if gpu is not None
                            else perf_counter() - started,
                            "simulator": simulator(cfg) if gpu is not None else "cpu",
                            "state_hash": gpu["state_hash"]
                            if gpu is not None
                            else env.game.state_hash(),
                            "checkpoint_hash": checkpoint_hash,
                        }
                        if env.recorder and quotas[row["status"]] < limit:
                            replay_path = (
                                output / "replays" / f"{level}-{seed}-{row['status']}.pvzdemo"
                            )
                            check_deadline(deadline)
                            env.recorder.save(replay_path)
                            verified = verify_replay(replay_path)
                            if verified.state_hash() != row["state_hash"]:
                                raise RuntimeError("Evaluation replay diverged")
                            row["replay"] = str(replay_path.resolve())
                            row["replay_verified"] = True
                            row["replay_hash"] = file_hash(replay_path)
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
            {
                "state": "pending" if isinstance(exc, BudgetExpired) else "failed",
                "error": repr(exc),
                "completed_episodes": len(rows),
            },
        )
        raise
    finally:
        if gpu_rows is not None:
            gpu_rows.close()
        if owns_progress:
            progress.close()


def read_rows(paths) -> list[dict]:
    result = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            result.extend(json.loads(line) for line in stream if line.strip())
    return result
