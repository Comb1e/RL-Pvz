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
from pvz_rl.learning.checkpoints import (
    ANNOTATION_CHECKPOINT_PROTOCOL,
    ANNOTATION_PROGRESS_PROTOCOL,
    LEGACY_ANNOTATION_PROGRESS_PROTOCOL,
    MODEL_STATE_PROTOCOL,
    SnapshotStager,
    compatible_config,
    load_torch_snapshot,
    retry_file_sharing,
)
from pvz_rl.learning.cohort import atomic_transition
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.provenance import write_json

ANNOTATION_OBJECTIVE = "top1-bottomk-hinge-v3-l2-validation-grokking"


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


def _split_converged(metrics, rows):
    """Return whether both exact annotation metrics pass for one split."""
    return (
        metrics is not None
        and metrics["hits"] == rows
        and metrics.get("absolute_bottom_hits", rows) == rows
    )


def _l2_penalty(trainable, strength):
    """Return the explicit 0.5 * strength * ||theta||^2 master-weight penalty."""
    if not trainable:
        raise ValueError("annotation fitting has no trainable parameters")
    return 0.5 * strength * torch.stack([parameter.square().sum() for parameter in trainable]).sum()


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
    initialization = None
    if init_from is not None:
        from pvz_rl.learning.training import initial_weights

        weights, initialization = initial_weights(init_from, cfg)
        model.load_state_dict(weights, strict=True)
    return model, initialization


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
        retry_file_sharing(lambda: os.replace(temporary, path))
    finally:
        Path(temporary).unlink(missing_ok=True)


def _history_entry(completed, training, validation, active, penalty):
    return dict(
        pass_number=completed,
        optimizer_updates=completed,
        train=training,
        validation=validation,
        regularization_active=active,
        regularization_loss=penalty,
    )


def _last_history_line(stream):
    """Read one trailing JSONL line without scanning a long fitting history."""
    end = stream.seek(0, os.SEEK_END)
    position, tail = end, b""
    while position and tail.count(b"\n") < 2:
        start = max(0, position - 65536)
        stream.seek(start)
        tail = stream.read(position - start) + tail
        position = start
    separator = tail.rfind(b"\n", 0, len(tail) - int(tail.endswith(b"\n")))
    return position + separator + 1, tail[separator + 1 :]


def _reconcile_history(path, entry):
    """Discard an uncommitted/torn tail and recover the durable pass's metrics."""
    with path.open("a+b") as stream:
        while stream.seek(0, os.SEEK_END):
            offset, line = _last_history_line(stream)
            try:
                previous = json.loads(line)
                number = previous["pass_number"]
                valid = type(number) is int and 0 < number <= entry["pass_number"]
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                stream.truncate(offset)
                continue
            if number == entry["pass_number"]:
                # Replace this row from the checkpoint; also remove tail duplicates.
                stream.truncate(offset)
                continue
            if not line.endswith(b"\n"):
                stream.write(b"\n")
            break
        if entry["pass_number"] and entry["train"] is not None:
            stream.write((json.dumps(entry) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _convergence(training, validation):
    train, valid = training or {}, validation or {}
    return dict(
        verified=bool(train.get("rows", 0))
        and _split_converged(training, train["rows"])
        and _split_converged(validation, valid.get("rows", 0)),
        hits=train.get("hits"),
        records=train.get("rows", 0),
        absolute_bottom_hits=train.get("absolute_bottom_hits"),
        validation_hits=valid.get("hits"),
        validation_records=valid.get("rows", 0),
        validation_absolute_bottom_hits=valid.get("absolute_bottom_hits"),
    )


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
    l2_strength=None,
    seed=None,
    kind="tile",
    _resources=None,
):
    """Resume whole passes and expose saved weights independently of convergence."""
    if init_from is not None and resume_from is not None:
        raise ValueError("--resume-from and --init-from are mutually exclusive")
    cfg = copy.deepcopy(cfg or load_config())
    validate_config(cfg)
    settings = cfg["training"]["annotation"]
    saved = None
    saved_identity = None
    if resume_from is not None:
        saved, _ = load_torch_snapshot(resume_from)
        if saved.get("protocol") not in (
            ANNOTATION_PROGRESS_PROTOCOL,
            LEGACY_ANNOTATION_PROGRESS_PROTOCOL,
        ):
            raise ValueError(
                "Unsupported annotation progress protocol; use --init-from for weights"
            )
        saved_identity = copy.deepcopy(saved.get("identity"))
        if not isinstance(saved_identity, dict) or not isinstance(
            saved_identity.get("settings"), dict
        ):
            raise ValueError("Annotation progress is missing fitting identity/settings")
        saved_settings = saved_identity["settings"]
        # v3 included the total pass cap in objective identity. It was never a loss setting.
        saved_settings.pop("passes", None)
        overrides = dict(
            learning_rate=learning_rate,
            batch_size=batch_size,
            margin=margin,
            l2_strength=l2_strength,
            learner_seed=seed,
        )
        for key, value in overrides.items():
            if key not in saved_settings:
                raise ValueError(f"Annotation progress is missing fitting setting {key}")
            if value is not None and value != saved_settings[key]:
                raise ValueError(
                    f"Resume {key} differs from saved fitting settings; "
                    "use --init-from to start a fresh experiment with changed settings"
                )
            settings[key] = saved_settings[key]
        cfg["training"]["max_grad_norm"] = saved_settings.get("max_grad_norm")
    passes = settings["passes"] if passes is None else passes
    learning_rate = settings["learning_rate"] if learning_rate is None else learning_rate
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    margin = settings["margin"] if margin is None else margin
    l2_strength = settings.get("l2_strength", 1e-4) if l2_strength is None else l2_strength
    seed = settings["learner_seed"] if seed is None else seed
    if type(passes) is not int or passes < 0 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("passes must be nonnegative (0 unlimited); batch size must be positive")
    if type(seed) is not int or seed < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid annotation learning rate/seed")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("annotation margin must be finite and positive")
    if not math.isfinite(l2_strength) or l2_strength <= 0:
        raise ValueError("annotation L2 strength must be finite and positive")
    if kind not in ("tile", "action"):
        raise ValueError("annotation kind must be tile or action")
    fit_settings = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        margin=margin,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        learner_seed=seed,
        objective=ANNOTATION_OBJECTIVE,
        annotation_kind=kind,
        head_scope="tile" if kind == "tile" else "species",
        l2_strength=l2_strength,
        convergence="train-and-validation-exact-v1",
        reward_protocol=cfg["reward"]["version"],
        objective_protocol=cfg["training"]["objective"]["protocol"],
    )
    settings.update(
        passes=passes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        margin=margin,
        l2_strength=l2_strength,
        learner_seed=seed,
    )
    validate_config(cfg)
    if saved is not None and saved["protocol"] == ANNOTATION_PROGRESS_PROTOCOL:
        if not isinstance(saved.get("config"), dict) or saved.get("config_digest") != digest(
            saved["config"]
        ):
            raise ValueError("Annotation progress configuration digest does not match its contents")
        compatible_config(saved["config"], cfg)
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
    model, initialization = _load_initial_model(cfg, device, init_from)
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
    snapshot_stager = SnapshotStager(cfg["training"]["storage"]["ram_gib"] * 1024**3)
    completed = 0
    if saved is not None:
        if (
            saved_identity != identity
            or saved.get("observation_schema") != schema
            or saved.get("model_storage_protocol") != MODEL_STATE_PROTOCOL
            or (
                saved["protocol"] == ANNOTATION_PROGRESS_PROTOCOL
                and saved.get("precision_contract") != "bf16_master"
            )
        ):
            raise ValueError(
                "annotation progress dataset, split, fitting settings or schema mismatch"
            )
        if (
            type(saved.get("passes_completed")) is not int
            or saved["passes_completed"] < 0
            or saved.get("optimizer_updates") != saved["passes_completed"]
            or not isinstance(saved.get("model"), dict)
            or not saved["model"]
            or any(
                not torch.is_tensor(value) or not torch.isfinite(value).all()
                for value in saved["model"].values()
            )
        ):
            raise ValueError("Annotation progress has invalid pass counters or model weights")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["passes_completed"]
        initialization = saved.get("initialization")
        _restore_rng(saved["rng"], device)
    starting_pass = completed
    pass_limit = starting_pass + passes if passes else None
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

    def save_progress(training, validation, regularization_active=False, regularization_loss=0.0):
        convergence = _convergence(training, validation)
        _atomic_save(
            snapshot_stager.stage(dict(
                protocol=ANNOTATION_PROGRESS_PROTOCOL,
                identity=identity,
                model_storage_protocol=MODEL_STATE_PROTOCOL,
                precision_contract="bf16_master",
                config=cfg,
                config_digest=digest(cfg),
                annotation_kind=kind,
                learner_seed=seed,
                rules_hash=schema["rules_hash"],
                initialization=initialization,
                reward_protocol=cfg["reward"]["version"],
                objective_protocol=cfg["training"]["objective"]["protocol"],
                observation_schema=schema,
                model=model.checkpoint_state_dict(),
                optimizer=optimizer.state_dict(),
                rng=_rng_state(device),
                passes_completed=completed,
                optimizer_updates=completed,
                starting_pass=starting_pass,
                passes_requested=passes,
                passes_this_invocation=completed - starting_pass,
                pass_limit=pass_limit,
                status="complete" if convergence["verified"] else "incomplete",
                convergence=convergence,
                train=training,
                validation=validation,
                regularization_active=regularization_active,
                regularization_loss=regularization_loss,
                l2_strength=l2_strength,
            )),
            progress_path,
        )

    regularization_active = bool(
        resume_from is not None and saved.get("regularization_active", False)
    )
    regularization_loss = float(
        saved.get("regularization_loss", 0.0) if resume_from is not None else 0.0
    )
    training = saved.get("train") if resume_from is not None else None
    validation = saved.get("validation") if resume_from is not None else None
    interrupted = False

    def exact(name, metrics):
        return _split_converged(metrics, len(partitions[name]))

    try:
        with atomic_transition():
            if resume_from is None:
                history_path.write_text("", encoding="utf-8")
            else:
                _reconcile_history(
                    history_path,
                    _history_entry(
                        completed, training, validation, regularization_active, regularization_loss
                    ),
                )
            save_progress(training, validation, regularization_active, regularization_loss)
        training, validation = evaluate("train"), evaluate("validation")
        if exact("train", training) and exact("validation", validation):
            regularization_active = False
        elif exact("train", training):
            regularization_active = True
        save_progress(training, validation, regularization_active, regularization_loss)
        print(
            f"pass={completed} updates={completed} train={training['hits']}/{training['rows']} "
            f"remaining={training['failures']} bottom={training.get('absolute_bottom_hits', 0)}/{training['rows']} loss={training['loss']:.6g} "
            f"validation={validation['hits']}/{validation['rows']} validation_loss={validation['loss']}",
            flush=True,
        )
        while not (exact("train", training) and exact("validation", validation)) and (
            pass_limit is None or completed < pass_limit
        ):
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
            regularization_loss = 0.0
            if regularization_active:
                penalty = _l2_penalty(trainable, l2_strength)
                if not bool(torch.isfinite(penalty)):
                    raise FloatingPointError("non-finite annotation L2 penalty")
                regularization_loss = float(penalty.detach().cpu())
                penalty.backward()
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
                if exact("train", training) and exact("validation", validation):
                    regularization_active = False
                elif exact("train", training):
                    regularization_active = True
                save_progress(training, validation, regularization_active, regularization_loss)
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            _history_entry(
                                completed,
                                training,
                                validation,
                                regularization_active,
                                regularization_loss,
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
        regularization_active = bool(saved.get("regularization_active", False))
        regularization_loss = float(saved.get("regularization_loss", 0.0))
        _restore_rng(saved["rng"], device)

    converged = exact("train", training) and exact("validation", validation) and not interrupted
    test = None
    if converged:
        # Reload the durable FP32 weights; their BF16 runtime must qualify again.
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.eval()
        training = evaluate("train")
        validation = evaluate("validation")
        if not exact("train", training) or not exact("validation", validation):
            raise RuntimeError("saved annotation weights failed fresh convergence verification")
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
            model=model.checkpoint_state_dict(),
            passes_completed=completed,
            learner_seed=seed,
            initialization=initialization,
            convergence=dict(
                verified=True,
                hits=training["hits"],
                records=training["rows"],
                absolute_bottom_hits=training["absolute_bottom_hits"],
                validation_hits=validation["hits"],
                validation_records=validation["rows"],
                validation_absolute_bottom_hits=validation["absolute_bottom_hits"],
            ),
            annotation=dict(
                protocol=ANNOTATION_PROTOCOL,
                annotation_kind=kind,
                head_scope=kind,
                reward_protocol=cfg["reward"]["version"],
                objective_protocol=cfg["training"]["objective"]["protocol"],
                objective=ANNOTATION_OBJECTIVE,
                l2_strength=l2_strength,
                validation_converged=True,
                **identity,
                split={r["id"]: split[r["group_id"]] for r in records},
                source_digest=manifest.get("source_digest"),
            ),
        )
        _atomic_save(snapshot_stager.stage(payload), output)
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
        starting_pass=starting_pass,
        passes_requested=passes,
        passes_this_invocation=completed - starting_pass,
        pass_limit=pass_limit,
        initialization=initialization,
        partitions={name: len(rows) for name, rows in partitions.items()},
        train=training,
        validation=validation,
        test=test,
        regularization_active=regularization_active,
        regularization_loss=regularization_loss,
        l2_strength=l2_strength,
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
