"""Read native replay annotations with a fallback for research 0.2 sidecars."""

import json
from pathlib import Path

from pvz_game.replay import Playback, read_recording


def open_playback(source, *, fallback=None):
    source = Path(source)
    data = read_recording(source)
    legacy = dict(fallback or {})
    sidecar = source.with_suffix(".metadata.json")
    if sidecar.exists():
        legacy.update(json.loads(sidecar.read_text("utf-8")))
    annotations = {}
    for key in ("policy_id", "checkpoint_sha256", "outcome", "termination_reason"):
        value = legacy.get(key)
        if value is not None:
            annotations[key] = value
    for old, new in (
        ("policy", "policy_id"),
        ("checkpoint_hash", "checkpoint_sha256"),
        ("status", "outcome"),
    ):
        if legacy.get(old) is not None:
            annotations.setdefault(new, legacy[old])
    annotations["experiment"] = {
        key: legacy[key]
        for key in ("level", "family", "scenario_seed", "learner_seed")
        if key in legacy
    }
    # Embedded metadata is authoritative. Legacy annotations only fill absent fields.
    annotations.update(data.get("metadata", {}))
    if annotations:
        data["metadata"] = annotations
    return Playback(data)
