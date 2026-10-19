"""Convergent top-1 feasible-tile warm starts with atomic pass recovery."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from pvz_rl.annotation.dataset import (
    ANNOTATION_PROTOCOL,
    annotation_conflict_groups,
    dataset_digest,
    load_records,
    write_manifest,
)
from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.batch_transport import collate_fitting_observations
from pvz_rl.learning.checkpoints import MODEL_STATE_PROTOCOL
from pvz_rl.learning.cohort import atomic_transition
from pvz_rl.policy.sequential_q import observation_tile_masks
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.provenance import write_json

ANNOTATION_CHECKPOINT_PROTOCOL = "pvz-rl/tile-annotation-initialization-v1"
ANNOTATION_PROGRESS_PROTOCOL = "pvz-rl/tile-annotation-progress-v1"


def set_ranking_loss(values, feasible_tiles, *, margin=1.0, legal_masks=None):
    """Hinge between the best acceptable tile and the best legal negative.

    Each record has equal weight, including automatically correct all-positive
    boards. Argmax selects the lowest index on ties, as in the normal policy.
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    legal = torch.ones_like(values, dtype=torch.bool) if legal_masks is None else legal_masks
    losses = []
    for row, tiles, mask in zip(values, feasible_tiles, legal, strict=True):
        positives = sorted(set(int(tile) for tile in tiles))
        if not positives:
            raise ValueError("empty labeled feasible set")
        positive = torch.zeros_like(mask)
        positive[positives] = True
        if bool((positive & ~mask).any()):
            raise ValueError("annotated tiles must be legal")
        negative = mask & ~positive
        if not bool(negative.any()):
            losses.append(row.sum() * 0)
        else:
            best_positive = row.masked_fill(~positive, -torch.inf).max(dim=0).values
            best_negative = row.masked_fill(~negative, -torch.inf).max(dim=0).values
            losses.append(torch.relu(row.new_tensor(margin) + best_negative - best_positive))
    return torch.stack(losses).mean() if losses else values.sum() * 0


def _metrics(values, feasible_tiles, legal_masks=None):
    values = values.detach().float().cpu()
    legal = torch.ones_like(values, dtype=torch.bool) if legal_masks is None else legal_masks.cpu()
    hits = count = all_positive = 0
    precision = recall = 0.0
    for row, tiles, mask in zip(values, feasible_tiles, legal, strict=True):
        positive = set(tiles)
        if not positive:
            raise ValueError("empty labeled feasible set")
        masked = row.masked_fill(~mask, -torch.inf)
        hits += int(int(masked.argmax()) in positive)
        count += len(positive)
        all_positive += int(len(positive) == int(mask.sum()))
        # Set recovery is diagnostic only; convergence is top-1 membership.
        predicted = set(masked.argsort(descending=True, stable=True)[: len(positive)].tolist())
        overlap = len(predicted & positive)
        precision += overlap / len(positive)
        recall += overlap / len(positive)
    total = len(feasible_tiles)
    return dict(
        records=total,
        hits=hits,
        failures=total - hits,
        top1_feasible_rate=hits / total if total else None,
        top1_feasible_hit_rate=hits / total if total else None,
        feasible_set_precision=precision / total if total else None,
        feasible_set_recall=recall / total if total else None,
        feasible_set_size=count / total if total else None,
        all_positive_records=all_positive,
        empty_records=0,
        numerically_stable=bool(torch.isfinite(values).all()),
    )


def _validate_labels(records):
    for row in records:
        if not row["feasible_tiles"]:
            raise ValueError(f"empty labeled feasible set: {row['id']}")
    conflicts = [
        ", ".join(row["id"] for row in group["records"])
        for group in annotation_conflict_groups(records)
    ]
    if conflicts:
        raise ValueError("contradictory identical board/plant inputs: " + "; ".join(conflicts))


def _groups(records):
    groups = sorted({row["group_id"] for row in records})
    rng = random.Random(101)
    rng.shuffle(groups)
    n = len(groups)
    if n >= 3:
        train_end = max(1, min(n - 2, int(n * 0.8)))
        validation_end = max(train_end + 1, min(n - 1, int(n * 0.9)))
    elif n == 2:
        train_end, validation_end = 1, 2
    else:
        train_end = validation_end = 1
    assignment = {}
    for group in groups[:train_end]:
        assignment[group] = "train"
    for group in groups[train_end:validation_end]:
        assignment[group] = "validation"
    for group in groups[validation_end:]:
        assignment[group] = "test"
    return assignment


def _load_initial_model(cfg, device, init_from):
    model = TwoStageTransformerPolicy(cfg).to(device)
    if init_from is not None:
        from pvz_rl.learning.training import initial_weights

        weights, _ = initial_weights(init_from, cfg)
        model.load_state_dict(weights, strict=True)
    return model


def _batch(model, rows, cfg, device):
    observations = collate_fitting_observations(
        [row["encoded_observation"] for row in rows], cfg, device
    )
    plants = cfg["environment"]["plants"]
    branches = torch.as_tensor(
        [plants.index(row["plant_type"]) + 1 for row in rows], device=device, dtype=torch.long
    )
    values = model.fitting_tiles(
        observations, branches, rows=torch.arange(len(rows), device=device)
    )
    feasible = [row["feasible_tiles"] for row in rows]
    return values, feasible


def _legal_masks(rows, device):
    observations = collate_observations([row["encoded_observation"] for row in rows], device)
    geometry = observation_tile_masks(observations)[:, 1:46]
    public = torch.zeros(len(rows), 45, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        public[index, row["availability"]["legal_tiles"]] = True
    return geometry & public


def _run_split(model, records, cfg, device, *, batch_size, margin, train=False, optimizer=None):
    total_loss, hits, all_positive = 0.0, 0, 0
    precision = recall = size = 0.0
    if train:
        if optimizer is None:
            raise ValueError("training split requires an optimizer")
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for start in range(0, len(records), batch_size):
            rows = records[start : start + batch_size]
            values, feasible = _batch(model, rows, cfg, device)
            legal = _legal_masks(rows, device)
            loss = set_ranking_loss(values.float(), feasible, margin=margin, legal_masks=legal)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(values).all()):
                raise FloatingPointError("non-finite annotation values/loss")
            if train:
                (loss * (len(rows) / len(records))).backward()
                model.accumulate_master_gradients()
                model.clear_runtime_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(rows)
            metrics = _metrics(values, feasible, legal)
            hits += metrics["hits"]
            all_positive += metrics["all_positive_records"]
            precision += metrics["feasible_set_precision"] * len(rows)
            recall += metrics["feasible_set_recall"] * len(rows)
            size += metrics["feasible_set_size"] * len(rows)
    n = len(records)
    return dict(
        rows=n,
        records=n,
        hits=hits,
        failures=n - hits,
        loss=total_loss / n if n else None,
        top1_feasible_hit_rate=hits / n if n else None,
        feasible_set_precision=precision / n if n else None,
        feasible_set_recall=recall / n if n else None,
        feasible_set_size=size / n if n else None,
        all_positive_records=all_positive,
        empty_records=0,
        numerically_stable=True,
    )


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _rng_state(device):
    state = np.random.get_state()
    return dict(
        python=random.getstate(),
        torch=torch.get_rng_state(),
        numpy=[state[0], state[1].tolist(), *state[2:]],
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def _restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], np.uint32), *numpy_state[2:]))
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def fit_annotations(
    annotations,
    output,
    *,
    cfg=None,
    init_from=None,
    resume_from=None,
    device="cuda",
    passes=None,
    learning_rate=None,
    batch_size=None,
    margin=None,
    seed=None,
):
    """Publish only freshly verified convergence; incomplete progress is separate."""
    if init_from is not None and resume_from is not None:
        raise ValueError("--resume-from and --init-from are mutually exclusive")
    cfg = copy.deepcopy(cfg or load_config())
    validate_config(cfg)
    settings = cfg["training"]["annotation"]
    passes = settings["passes"] if passes is None else passes
    learning_rate = settings["learning_rate"] if learning_rate is None else learning_rate
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    margin = settings["margin"] if margin is None else margin
    seed = settings["learner_seed"] if seed is None else seed
    if type(passes) is not int or passes < 0 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("passes must be nonnegative (0 unlimited); batch size must be positive")
    if type(seed) is not int or seed < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid annotation learning rate/seed")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("annotation margin must be finite and positive")
    fit_settings = dict(
        passes=passes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        margin=margin,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        learner_seed=seed,
        objective="legal-top1-hinge-v1",
        reward_protocol=cfg["reward"]["version"],
        objective_protocol=cfg["training"]["objective"]["protocol"],
    )
    all_records = load_records(
        annotations, annotation=True, plant_types=cfg["environment"]["plants"]
    )
    records = [row for row in all_records if row["status"] == "labeled"]
    if not records:
        raise ValueError("annotation file contains no labeled records")
    _validate_labels(records)
    split = _groups(records)
    partitions = {
        name: [row for row in records if split[row["group_id"]] == name]
        for name in ("train", "validation", "test")
    }
    identity = dict(
        dataset_digest=dataset_digest(records), split_digest=digest(split), settings=fit_settings
    )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for annotation fitting")
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = _load_initial_model(cfg, device, init_from)
    schema = model.layout.schema()
    manifest_path = Path(str(annotations) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    for row in [*all_records, manifest]:
        if row.get("observation_schema") not in (None, schema):
            raise ValueError("annotation observation schema disagrees with current model")
        if row.get("rules_hash") not in (None, schema["rules_hash"]):
            raise ValueError("annotation rules hash disagrees with current game")
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("tile_encoder.", "tile_head.")))
    trainable = []
    for name, parameter in zip(model._master_names, model.master_parameters(), strict=True):
        parameter.requires_grad_(name.startswith(("tile_encoder.", "tile_head.")))
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    output = Path(output).with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(".progress.pt")
    history_path = output.with_suffix(".history.jsonl")
    report_path = output.with_suffix(".report.json")
    completed = 0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if (
            saved.get("protocol") != ANNOTATION_PROGRESS_PROTOCOL
            or saved.get("identity") != identity
            or saved.get("observation_schema") != schema
            or saved.get("model_storage_protocol") != MODEL_STATE_PROTOCOL
        ):
            raise ValueError(
                "annotation progress dataset, split, fitting settings or schema mismatch"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["passes_completed"]
        _restore_rng(saved["rng"], device)
    model.eval()

    def evaluate(name):
        return _run_split(
            model, partitions[name], cfg, device, batch_size=batch_size, margin=margin
        )

    def save_progress(training, validation):
        _atomic_save(
            dict(
                protocol=ANNOTATION_PROGRESS_PROTOCOL,
                identity=identity,
                model_storage_protocol=MODEL_STATE_PROTOCOL,
                reward_protocol=cfg["reward"]["version"],
                objective_protocol=cfg["training"]["objective"]["protocol"],
                observation_schema=schema,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=_rng_state(device),
                passes_completed=completed,
                optimizer_updates=completed,
                status="incomplete",
                train=training,
                validation=validation,
            ),
            progress_path,
        )

    training = saved.get("train") if resume_from is not None else None
    validation = saved.get("validation") if resume_from is not None else None
    save_progress(training, validation)
    if resume_from is None:
        history_path.write_text("", encoding="utf-8")
    interrupted = False
    try:
        training, validation = evaluate("train"), evaluate("validation")
        save_progress(training, validation)
        print(
            f"pass={completed} updates={completed} train={training['hits']}/{training['rows']} "
            f"remaining={training['failures']} loss={training['loss']:.6g} "
            f"validation={validation['hits']}/{validation['rows']} validation_loss={validation['loss']}",
            flush=True,
        )
        while training["hits"] != len(partitions["train"]) and (passes == 0 or completed < passes):
            model.train()
            _run_split(
                model,
                partitions["train"],
                cfg,
                device,
                batch_size=batch_size,
                margin=margin,
                train=True,
                optimizer=optimizer,
            )
            torch.nn.utils.clip_grad_norm_(
                trainable, fit_settings["max_grad_norm"], error_if_nonfinite=True
            )
            # An interrupt during commit is honored only after its evaluation and
            # durable progress are complete. Earlier gradients are disposable.
            with atomic_transition():
                optimizer.step()
                model.sync_runtime_from_masters()
                model.eval()
                completed += 1
                training, validation = evaluate("train"), evaluate("validation")
                save_progress(training, validation)
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                pass_number=completed,
                                optimizer_updates=completed,
                                train=training,
                                validation=validation,
                            )
                        )
                        + "\n"
                    )
                    stream.flush()
                print(
                    f"pass={completed} updates={completed} train={training['hits']}/{training['rows']} "
                    f"remaining={training['failures']} loss={training['loss']:.6g} "
                    f"validation={validation['hits']}/{validation['rows']} "
                    f"validation_loss={validation['loss']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["passes_completed"]
        training, validation = saved["train"], saved["validation"]
        _restore_rng(saved["rng"], device)

    converged = (
        training is not None and training["hits"] == len(partitions["train"]) and not interrupted
    )
    test = None
    if converged:
        # Reload the durable FP32 weights; their BF16 runtime must qualify again.
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.eval()
        training = evaluate("train")
        if training["hits"] != len(partitions["train"]):
            raise RuntimeError("saved annotation weights failed fresh training-set verification")
        test = evaluate("test")
        checkpoint_cfg = copy.deepcopy(cfg)
        checkpoint_cfg["training"].pop("annotation", None)
        payload = dict(
            protocol=ANNOTATION_CHECKPOINT_PROTOCOL,
            model_storage_protocol=MODEL_STATE_PROTOCOL,
            precision_contract="bf16_master",
            observation_schema=schema,
            rules_hash=schema["rules_hash"],
            config=checkpoint_cfg,
            config_digest=digest(checkpoint_cfg),
            model=model.state_dict(),
            passes_completed=completed,
            learner_seed=seed,
            convergence=dict(verified=True, hits=training["hits"], records=training["rows"]),
            annotation=dict(
                protocol=ANNOTATION_PROTOCOL,
                reward_protocol=cfg["reward"]["version"],
                objective_protocol=cfg["training"]["objective"]["protocol"],
                **identity,
                split={r["id"]: split[r["group_id"]] for r in records},
                source_digest=manifest.get("source_digest"),
            ),
        )
        _atomic_save(payload, output)
    manifest.update(
        protocol=manifest.get("protocol", ANNOTATION_PROTOCOL),
        config_digest=manifest.get("config_digest") or digest(cfg),
        dataset_digest=identity["dataset_digest"],
        counts=dict(
            generated=len(all_records),
            pending=sum(r["status"] == "pending" for r in all_records),
            labeled=len(records),
            discarded=sum(r["status"] == "discarded" for r in all_records),
            usable_training=len(partitions["train"]),
        ),
        observation_schema=schema,
        rules_hash=schema["rules_hash"],
        split={r["id"]: split[r["group_id"]] for r in records},
        split_digest=digest(split),
    )
    write_manifest(manifest_path, manifest)
    report = dict(
        status="complete" if converged else "incomplete",
        stop_reason="converged" if converged else "interrupted" if interrupted else "pass_cap",
        checkpoint=str(output.resolve()) if converged else None,
        progress=str(progress_path.resolve()),
        history=str(history_path.resolve()),
        records=len(records),
        labeled_records=len(records),
        pending_records=sum(r["status"] == "pending" for r in all_records),
        discarded_records=sum(r["status"] == "discarded" for r in all_records),
        **identity,
        passes_completed=completed,
        optimizer_updates=completed,
        partitions={name: len(rows) for name, rows in partitions.items()},
        train=training,
        validation=validation,
        test=test,
        tile_parameters_only=True,
    )
    write_json(report_path, report)
    return report
