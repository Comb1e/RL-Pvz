"""Completed-run retention; unfinished runs keep their full recovery artifacts."""

import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from pvz_rl.config import digest

WEIGHTS_PROTOCOL = "pvz-rl/autonomous-weights-v1"
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
    "tensorboard",
    "visualizations",
    "curriculum-probes",
    "validation",
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
    """Atomically replace final.zip with portable weights before discarding recovery."""
    output = Path(output).resolve(strict=True)
    final = output / "final.zip"
    status = json.loads((output / "status.json").read_text("utf-8"))
    if status["state"] != "complete" or status.get("budget_stopped"):
        raise ValueError("Only completed training can discard recovery artifacts")
    with tempfile.NamedTemporaryFile(dir=output, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(final) as source, ZipFile(temporary, "w") as target:
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
            for name in ("policy.pth", "protocol.json", "observation-schema.json"):
                with source.open(name) as inp, target.open(name, "w", force_zip64=True) as out:
                    shutil.copyfileobj(inp, out)
            target.writestr("run.json", json.dumps(metadata))
            target.writestr(
                "weights-only.json",
                json.dumps(
                    dict(protocol=WEIGHTS_PROTOCOL, config_digest=digest(metadata["config"]))
                ),
            )
        from pvz_rl.learning.checkpoints import inspect_checkpoint

        inspect_checkpoint(temporary, weights_only=True)
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    retain_files(output, [final], candidates=[output / name for name in AUTONOMOUS_OUTPUTS])
