"""Fixed opening panels and exact scoring for the pre-Prediction capacity control."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

PROTOCOL = "fixed-pre-prediction-v1"


def fixed_seeds(cfg):
    return (
        cfg["training"].get("tile_scenarios", {}).get("pre_prediction", {}).get("fixed_scene_seeds")
    )


def validate_fixed_seeds(settings, stage):
    seeds = settings.get("fixed_scene_seeds")
    if seeds is None:
        return
    if (
        stage != "pre_prediction"
        or not isinstance(seeds, list)
        or len(seeds) != settings["n_envs"]
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Fixed pre_prediction needs one distinct nonnegative scene seed per env")


def opening_hash(snapshot, motion):
    value = json.dumps(
        [snapshot, motion],
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.tolist(),
        allow_nan=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()


def note_openings(model):
    if fixed_seeds(model.cfg) is None:
        return
    hashes = model.env.queue.fixed_opening_hashes
    expected = len(fixed_seeds(model.cfg))
    if sorted(hashes) != list(range(expected)):
        raise ValueError("Fixed cohort did not prepare its complete opening panel")
    hashes = [hashes[index] for index in range(expected)]
    state = getattr(model, "fixed_prediction_state", None)
    if state is None:
        model.fixed_prediction_state = dict(
            protocol=PROTOCOL,
            opening_hashes=hashes,
            target_hashes=None,
            collections=0,
            passes=0,
            passed=False,
        )
    elif state.get("protocol") != PROTOCOL or state.get("opening_hashes") != hashes:
        raise ValueError("Fixed cohort physical openings or motion history changed")
    if model._lr_restart_active or model._lr_restart_requested:
        raise ValueError("Fixed pre_prediction cannot use a learning-rate restart")


def start_check(model):
    if fixed_seeds(model.cfg) is None:
        return
    count = len(fixed_seeds(model.cfg))
    model._fixed_check_started = perf_counter()
    model._fixed_check_losses = np.full((count, 3), np.nan)
    model._fixed_check_targets = [None] * count


def score_batch(model, rows, logits, targets):
    if fixed_seeds(model.cfg) is None:
        return
    import torch

    from pvz_rl.learning.forecast_labels import categorical_loss, effect_blocks

    envs = np.asarray(rows["env"], dtype=np.int64)
    if len(set(envs.tolist())) != len(envs) or any(
        model._fixed_check_targets[index] is not None for index in envs
    ):
        raise RuntimeError("Fixed pre_prediction recorded more than one forecast per env")
    losses = (
        torch.stack(
            [
                categorical_loss(block, targets[..., term])[0].mean(-1)
                for term, block in enumerate(effect_blocks(logits))
            ],
            -1,
        )
        .cpu()
        .numpy()
    )
    model._fixed_check_losses[envs] = losses
    for index, outcome in zip(envs, rows["baseline_outcomes"], strict=True):
        model._fixed_check_targets[index] = hashlib.sha256(
            np.asarray(outcome, dtype="<i8").tobytes()
        ).hexdigest()


def _publish(model, event):
    from pvz_rl.config import jsonl_batch_bytes
    from pvz_rl.jsonl_log import append_jsonl_file
    from pvz_rl.provenance import write_json

    root = Path(model.trajectory_root)
    state = model.fixed_prediction_state
    write_json(root / "overfit-state.json", state)
    append_jsonl_file(root / "overfit-metrics.jsonl", event, jsonl_batch_bytes(model.cfg))


def finish_check(model, callback):
    """Return true only for a complete, perfectly scored fixed panel."""
    if fixed_seeds(model.cfg) is None:
        return False
    state = model.fixed_prediction_state
    targets = model._fixed_check_targets
    missing = sum(target is None for target in targets)
    previous = state["target_hashes"]
    if previous is not None and any(
        old is not None and new is not None and old != new
        for old, new in zip(previous, targets, strict=True)
    ):
        raise ValueError("Exact target changed for a fixed physical opening")
    state["target_hashes"] = (
        targets
        if previous is None
        else [old or new for old, new in zip(previous, targets, strict=True)]
    )
    scored = int(model._env_forecast_scored.sum())
    correct = int(model._env_forecast_correct.sum())
    passed = missing == 0 and correct == scored == len(targets)
    losses = model._fixed_check_losses
    known = np.isfinite(losses).all(-1)
    means = losses[known].mean(0).tolist() if known.any() else [None] * 3
    row = dict(
        event="check",
        cohort=model.cohort_number,
        correct=correct,
        scored=scored,
        expected=len(targets),
        missing_labels=missing,
        accuracy=correct / len(targets),
        field_depth_loss=means[0],
        breach_loss=means[1],
        probe_loss=means[2],
        loss=float(losses[known].mean()) if known.any() else None,
        learning_rate=model.policy.optimizer.param_groups[0]["lr"],
        optimizer_commits=model.optimizer_commits,
        passed=passed,
        collection_seconds=model._phase_times["collect"],
        preparation_seconds=model.env.last_preparation_seconds,
        scoring_seconds=perf_counter() - model._fixed_check_started,
    )
    state.update(collections=model.cohort_number, passed=passed, last_check=row)
    _publish(model, row)
    if hasattr(callback, "progress"):
        callback.progress.emit(
            f"Fixed pre_prediction {correct}/{len(targets)}; missing {missing}; "
            f"loss {row['loss']}; lr {row['learning_rate']:g}",
            force=True,
        )
    return passed


def record_fit(model, norm):
    if fixed_seeds(model.cfg) is None:
        return
    state = model.fixed_prediction_state
    state["passes"] += 1
    row = dict(
        event="fit",
        cohort=model.cohort_number,
        passes=1,
        gradient_norm=norm,
        learning_rate=model.policy.optimizer.param_groups[0]["lr"],
        optimizer_commits=model.optimizer_commits,
        fitting_seconds=perf_counter() - model._fixed_fit_started,
    )
    state["last_fit"] = row
    _publish(model, row)
