"""Portable model exports and training artifact retention."""

import io
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import torch

from pvz_rl.config import digest
from pvz_rl.jsonl_log import series_dir
from pvz_rl.learning.checkpoints import (
    has_recovery_state,
    inspect_checkpoint,
    protocol_for,
    retry_file_sharing,
)

WEIGHTS_PROTOCOL = "pvz-rl/autonomous-weights-v1"
RECOVERY_ARCHIVE = "recovery.zip"
AUTONOMOUS_OUTPUTS = (
    "config.json",
    "metadata.json",
    "status.json",
    "train.log",
    "visualize.log",
    "hardware-metrics.jsonl",
    "training-episodes.jsonl",
    "training-metrics.jsonl",
    "validation-checks.jsonl",
    "tile-skill-probes.jsonl",
    "curriculum.json",
    "curriculum-probes.jsonl",
    "learning-curve.jsonl",
    "best.json",
    "best.zip",
    "latest.zip",
    "interrupted.zip",
    RECOVERY_ARCHIVE,
    "tensorboard",
    "curriculum-probes",
    "validation",
)


def _write_weights_archive(path, metadata, entries, *, preserve_recovery=False):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w") as target:
            for name, contents in entries.items():
                target.writestr(name, contents)
            target.writestr("run.json", json.dumps(metadata))
            target.writestr(
                "weights-only.json",
                json.dumps(
                    dict(protocol=WEIGHTS_PROTOCOL, config_digest=digest(metadata["config"]))
                ),
            )
        inspect_checkpoint(temporary, weights_only=True)
        if preserve_recovery:
            recovery = path.with_name(RECOVERY_ARCHIVE)
            if not recovery.exists() and has_recovery_state(path):
                retry_file_sharing(lambda: path.replace(recovery))
        retry_file_sharing(lambda: temporary.replace(path))
    finally:
        temporary.unlink(missing_ok=True)


def save_latest_model(output, model):
    """Serialize only authoritative FP32 policy weights and their load metadata."""
    source = model.research_metadata
    metadata = dict(
        config=model.cfg,
        condition=source["condition"],
        family=source["family"],
        learner_seed=source["learner_seed"],
        source_counters=dict(
            steps=int(model.num_timesteps),
            games=int(model.training_games),
            updates=int(model._n_updates),
        ),
    )
    weights = io.BytesIO()
    torch.save(
        {name: tensor.cpu() for name, tensor in model.policy.checkpoint_state_dict().items()},
        weights,
    )
    _write_weights_archive(
        Path(output) / "latest.zip",
        metadata,
        {
            "policy.pth": weights.getvalue(),
            "protocol.json": json.dumps(protocol_for(model.policy_protocol)),
            "observation-schema.json": json.dumps(model.policy.layout.schema()),
        },
        preserve_recovery=True,
    )


def retain_files(root, keep, *, candidates=None):
    """Remove owned outputs only, never follow a link outside the output directory."""
    root = Path(root).resolve(strict=True)
    keep = {Path(path).absolute() for path in keep}
    paths = list(root.iterdir()) if candidates is None else [Path(p).absolute() for p in candidates]
    for path in paths:
        if path.parent != root or path in keep:
            continue
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            # Junctions must also be unlinked rather than traversed on Windows.
            if path.is_junction():
                path.rmdir()
            else:
                shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def finish_autonomous_run(output):
    """Compact final.zip and discard recovery while retaining generated visualizations."""
    output = Path(output).resolve(strict=True)
    final = output / "final.zip"
    status = json.loads((output / "status.json").read_text("utf-8"))
    if status["state"] != "complete" or status.get("budget_stopped"):
        raise ValueError("Only completed training can discard recovery artifacts")
    with ZipFile(final) as source:
        metadata = json.loads(source.read("run.json"))
        counters = json.loads(source.read("data"))
        metadata.update(
            completion=status,
            source_counters={
                key: counters.get(field, 0)
                for key, field in (
                    ("steps", "num_timesteps"),
                    ("games", "training_games"),
                    ("updates", "_n_updates"),
                )
            },
        )
        entries = {
            name: source.read(name)
            for name in ("policy.pth", "protocol.json", "observation-schema.json")
        }
    _write_weights_archive(final, metadata, entries)
    candidates = []
    for name in AUTONOMOUS_OUTPUTS:
        path = output / name
        if name.endswith(".jsonl"):
            folder = series_dir(path)
            if folder.is_dir() or folder.is_symlink():
                candidates.append(folder)
        candidates.append(path)
    retain_files(output, [final], candidates=candidates)
