"""Convergent top-1 feasible-tile warm starts with atomic pass recovery."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np
import torch

from pvz_rl.annotation.dataset import (
    ANNOTATION_PROTOCOL,
    action_names,
    annotation_conflict_groups,
    annotation_kind,
    dataset_digest,
    load_records,
    write_manifest,
)
from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.learning.batch_transport import PreparedDataSource, collate_fitting_observations
from pvz_rl.learning.checkpoints import MODEL_STATE_PROTOCOL
from pvz_rl.learning.cohort import atomic_transition
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.provenance import write_json

ANNOTATION_CHECKPOINT_PROTOCOL = "pvz-rl/annotation-initialization-v2"
ANNOTATION_PROGRESS_PROTOCOL = "pvz-rl/annotation-progress-v2"


def _label_masks(feasible, absolute, width):
    positive = torch.zeros(len(feasible), width, dtype=torch.bool)
    negative = torch.zeros_like(positive)
    if absolute is None:
        absolute = [[] for _ in feasible]
    if len(absolute) != len(feasible):
        raise ValueError("label batch dimensions disagree")
    for index, (yes, no) in enumerate(zip(feasible, absolute, strict=True)):
        if not yes:
            raise ValueError("empty labeled feasible set")
        for labels, destination in ((yes, positive), (no, negative)):
            if any(type(label) is not int or not 0 <= label < width for label in labels):
                raise ValueError("annotation labels must be in range")
            destination[index, list(set(labels))] = True
    if bool((positive & negative).any()):
        raise ValueError("feasible and absolute labels overlap")
    return positive, negative


def _ranking_terms(values, positive, absolute, margin, legal=None):
    legal = torch.ones_like(positive) if legal is None else legal
    other = legal & ~positive
    has_other, has_absolute = other.any(-1), absolute.any(-1)
    # Indexed reductions preserve the original lowest-index tie subgradient.
    best = values.masked_fill(~positive, -torch.inf).max(-1).values
    competitor = values.masked_fill(~other, -torch.inf).max(-1).values
    worst_absolute = values.masked_fill(~absolute, -torch.inf).max(-1).values
    remaining = values.masked_fill(~(legal & ~absolute), torch.inf).min(-1).values
    top = torch.relu(margin + torch.where(has_other, competitor, best) - best)
    bottom = torch.relu(margin + torch.where(has_absolute, worst_absolute, remaining) - remaining)
    return torch.where(has_other, top, 0), torch.where(has_absolute, bottom, 0)


def set_ranking_loss(values, feasible_tiles, absolute_labels=None, *, margin=1.0, legal_masks=None):
    """Batched two-set hinge; labels are validated once on the CPU."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    positive, absolute = _label_masks(feasible_tiles, absolute_labels, values.shape[-1])
    if legal_masks is not None:
        legal_cpu = legal_masks.detach().cpu()
        if bool((positive & ~legal_cpu).any()):
            raise ValueError("annotated tiles must be legal")
        if bool((absolute.any(-1) & ~(legal_cpu & ~absolute).any(-1)).any()):
            raise ValueError("absolute labels leave no non-absolute outputs")
    positive, absolute = positive.to(values.device), absolute.to(values.device)
    top, bottom = _ranking_terms(values, positive, absolute, margin, legal_masks)
    return (top + bottom).mean() if len(values) else values.sum() * 0


def _metric_totals(values, positive, absolute, legal=None):
    legal = torch.ones_like(positive) if legal is None else legal
    masked = values.detach().float().masked_fill(~legal, -torch.inf)
    ranked = masked.argsort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(ranked).scatter_(
        1, ranked, torch.arange(values.shape[1], device=values.device)[None].expand_as(ranked)
    )
    size, bottom_size = positive.sum(-1), absolute.sum(-1)
    top = positive.gather(1, masked.argmax(-1)[:, None]).flatten()
    bottom = (~absolute | (ranks >= values.shape[1] - bottom_size[:, None])).all(-1)
    overlap = (positive & (ranks < size[:, None])).sum(-1) / size.clamp_min(1)
    return torch.stack(
        (
            top.sum(),
            bottom.sum(),
            (size == legal.sum(-1)).sum(),
            overlap.sum(),
            overlap.sum(),
            size.sum(),
            bottom_size.sum(),
            (bottom_size == 0).sum(),
        )
    ).double()


def _metric_report(totals, n):
    hits, bottom, all_positive, precision, recall, size, absolute, no_absolute = totals
    return dict(
        records=n,
        rows=n,
        hits=int(hits),
        failures=n - int(hits),
        top1_feasible_rate=hits / n if n else None,
        top1_feasible_hit_rate=hits / n if n else None,
        top1_feasible_accuracy=hits / n if n else None,
        absolute_bottom_hits=int(bottom),
        absolute_bottom_rate=bottom / n if n else None,
        absolute_bottom_accuracy=bottom / n if n else None,
        feasible_set_precision=precision / n if n else None,
        feasible_set_recall=recall / n if n else None,
        feasible_set_size=size / n if n else None,
        absolute_set_size=absolute / n if n else None,
        all_positive_records=int(all_positive),
        no_absolute_records=int(no_absolute),
        empty_records=0,
        numerically_stable=True,
    )


def _metrics(values, feasible_tiles, absolute_labels=None, legal_masks=None):
    if torch.is_tensor(absolute_labels):
        legal_masks, absolute_labels = absolute_labels, None
    positive, absolute = _label_masks(feasible_tiles, absolute_labels, values.shape[1])
    totals = _metric_totals(
        values, positive.to(values.device), absolute.to(values.device), legal_masks
    )
    result = _metric_report(totals.cpu().tolist(), len(values))
    result["numerically_stable"] = bool(torch.isfinite(values).all())
    return result


def _validate_labels(records, kind="tile"):
    for row in records:
        if kind == "tile":
            if not row.get("feasible_tiles"):
                raise ValueError(f"empty labeled feasible set: {row['id']}")
            feasible, absolute = (
                row.get("feasible_tiles", []),
                row.get("absolute_infeasible_tiles", []),
            )
        else:
            if not row.get("feasible_actions"):
                raise ValueError(f"empty labeled feasible action set: {row['id']}")
            feasible, absolute = (
                row.get("feasible_actions", []),
                row.get("absolute_infeasible_actions", []),
            )
        if set(feasible) & set(absolute):
            raise ValueError(f"feasible and absolute labels overlap: {row['id']}")
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


def _batch(model, rows, cfg, device, *, kind="tile", include_absolute=False):
    observations = collate_fitting_observations(
        [row["encoded_observation"] for row in rows], cfg, device
    )
    plants = cfg["environment"]["plants"]
    if kind == "action":
        values = model.species_values(observations)
        feasible = [
            [action_names(plants).index(name) for name in row["feasible_actions"]] for row in rows
        ]
        absolute = [
            [action_names(plants).index(name) for name in row["absolute_infeasible_actions"]]
            for row in rows
        ]
    else:
        branches = torch.as_tensor(
            [plants.index(row["plant_type"]) + 1 for row in rows], device=device, dtype=torch.long
        )
        values = model.fitting_tiles(
            observations, branches, rows=torch.arange(len(rows), device=device)
        )
        feasible = [row["feasible_tiles"] for row in rows]
        absolute = [row.get("absolute_infeasible_tiles", []) for row in rows]
    return (values, feasible, absolute) if include_absolute else (values, feasible)


class PreparedAnnotations:
    def __init__(self, records, cfg, kind):
        names = action_names(cfg["environment"]["plants"])
        if kind == "action":
            feasible = [[names.index(x) for x in r["feasible_actions"]] for r in records]
            absolute = [[names.index(x) for x in r["absolute_infeasible_actions"]] for r in records]
            self.branches = None
        else:
            feasible = [r["feasible_tiles"] for r in records]
            absolute = [r.get("absolute_infeasible_tiles", []) for r in records]
            self.branches = torch.tensor(
                [cfg["environment"]["plants"].index(r["plant_type"]) + 1 for r in records],
                dtype=torch.long,
            )
        self.positive, self.absolute = _label_masks(
            feasible, absolute, 10 if kind == "action" else 45
        )
        self.source = PreparedDataSource([r["encoded_observation"] for r in records], cfg)

    def close(self):
        self.source.close()


def _run_split(
    model,
    records,
    cfg,
    device,
    *,
    batch_size,
    margin,
    train=False,
    optimizer=None,
    kind="tile",
    prepared=None,
):
    with ExitStack() as stack:
        if prepared is None:
            prepared = PreparedAnnotations(records, cfg, kind)
            stack.callback(prepared.close)
        totals = torch.zeros(12, device=device, dtype=torch.float64)
        if train:
            if optimizer is None:
                raise ValueError("training split requires an optimizer")
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
        batches = stack.enter_context(closing(prepared.source.batches(batch_size, device)))
        start = 0
        with torch.set_grad_enabled(train):
            for rows, observations, _ in batches:
                stop = start + len(rows)
                if kind == "action":
                    values = model.species_values(observations)
                else:
                    branches = prepared.branches[start:stop].to(device)
                    values = model.fitting_tiles(
                        observations, branches, rows=torch.arange(len(rows), device=device)
                    )
                positive = prepared.positive[start:stop].to(device)
                absolute = prepared.absolute[start:stop].to(device)
                top, bottom = _ranking_terms(values.float(), positive, absolute, margin)
                loss = (top + bottom).mean()
                if train:
                    (loss * (len(rows) / len(records))).backward()
                    model.accumulate_master_gradients()
                    model.clear_runtime_grad(set_to_none=True)
                else:
                    totals[:8] += _metric_totals(values, positive, absolute)
                totals[8] += loss.detach().double() * len(rows)
                totals[9] += (~torch.isfinite(values)).sum() + (~torch.isfinite(loss)).sum()
                totals[10] += top.detach().double().sum()
                totals[11] += bottom.detach().double().sum()
                start = stop
        summary = totals.cpu().tolist()
        if summary[9]:
            raise FloatingPointError("non-finite annotation values/loss")
        result = _metric_report(summary[:8], len(records))
        result["loss"] = summary[8] / len(records) if records else None
        result["top_loss"] = summary[10] / len(records) if records else None
        result["bottom_loss"] = summary[11] / len(records) if records else None
        return result


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


def _fit_annotations(
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
    kind="tile",
    _resources=None,
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
    if kind not in ("tile", "action"):
        raise ValueError("annotation kind must be tile or action")
    fit_settings = dict(
        passes=passes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        margin=margin,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        learner_seed=seed,
        objective="top1-bottomk-hinge-v2",
        annotation_kind=kind,
        head_scope="tile" if kind == "tile" else "species",
        reward_protocol=cfg["reward"]["version"],
        objective_protocol=cfg["training"]["objective"]["protocol"],
    )
    all_records = load_records(
        annotations, annotation=True, plant_types=cfg["environment"]["plants"]
    )
    records = [row for row in all_records if row["status"] == "labeled"]
    if not records:
        raise ValueError("annotation file contains no labeled records")
    if any(annotation_kind(row) != kind for row in records):
        raise ValueError("annotation dataset kind does not match fitter")
    _validate_labels(records, kind)
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
    train_prefix = (
        ("tile_encoder.", "tile_head.") if kind == "tile" else ("species_encoder.", "species_head.")
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(train_prefix))
    trainable = []
    for name, parameter in zip(model._master_names, model.master_parameters(), strict=True):
        parameter.requires_grad_(name.startswith(train_prefix))
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

    prepared = {}
    for name, rows in partitions.items():
        prepared[name] = PreparedAnnotations(rows, cfg, kind)
        _resources.callback(prepared[name].close)

    def evaluate(name):
        return _run_split(
            model,
            partitions[name],
            cfg,
            device,
            batch_size=batch_size,
            margin=margin,
            kind=kind,
            prepared=prepared[name],
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
            f"remaining={training['failures']} bottom={training.get('absolute_bottom_hits', 0)}/{training['rows']} loss={training['loss']:.6g} "
            f"validation={validation['hits']}/{validation['rows']} validation_loss={validation['loss']}",
            flush=True,
        )
        while (
            training["hits"] != len(partitions["train"])
            or training.get("absolute_bottom_hits", len(partitions["train"]))
            != len(partitions["train"])
        ) and (passes == 0 or completed < passes):
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
                kind=kind,
                prepared=prepared["train"],
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
                    f"remaining={training['failures']} bottom={training.get('absolute_bottom_hits', 0)}/{training['rows']} loss={training['loss']:.6g} "
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
        training is not None
        and training["hits"] == len(partitions["train"])
        and training.get("absolute_bottom_hits", len(partitions["train"]))
        == len(partitions["train"])
        and not interrupted
    )
    test = None
    if converged:
        # Reload the durable FP32 weights; their BF16 runtime must qualify again.
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.eval()
        training = evaluate("train")
        if training["hits"] != len(partitions["train"]) or training["absolute_bottom_hits"] != len(
            partitions["train"]
        ):
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
                annotation_kind=kind,
                head_scope=kind,
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
        tile_parameters_only=kind == "tile",
        species_parameters_only=kind == "action",
        annotation_kind=kind,
    )
    write_json(report_path, report)
    return report


def fit_annotations(annotations, output, **kwargs):
    with ExitStack() as resources:
        return _fit_annotations(annotations, output, _resources=resources, **kwargs)


def fit_action_annotations(annotations, output, **kwargs):
    """Fit the species/action head from action-type annotations."""
    kwargs["kind"] = "action"
    return fit_annotations(annotations, output, **kwargs)
