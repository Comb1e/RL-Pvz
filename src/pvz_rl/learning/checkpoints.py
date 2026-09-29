"""Inspect checkpoint identity and dispatch models before starting environments."""

import copy
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.learning.performance import refresh_performance
from pvz_rl.learning.training_requirements import transfer_protocol

DEMO_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v2"
STATE_PROTOCOL = "pvz-rl/lstm-state-v2"


def protocol_for(kind):
    methods = {
        "transformer_lstm_q_v2": "complete_return_lstm_v1",
    }
    if kind not in methods:
        raise ValueError(
            f"Incompatible checkpoint model family: {kind}; entity_v1 requires fresh initialization"
        )
    from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL

    return dict(policy=kind, optimizer=methods[kind], exploration=EXPLORATION_PROTOCOL)


def compatible_config(source, target):
    if transfer_protocol(source) != transfer_protocol(target):
        raise ValueError(
            "Checkpoint transfer requires matching engine, observation encoding, action "
            "semantics, reward definition and network structure; incompatible model transfer"
        )


def execution_config(cfg):
    """Fill optional execution settings; never replace recorded protocol parameters."""
    cfg = copy.deepcopy(cfg)
    defaults = load_config()
    cfg["training"].setdefault("performance", {"fit_precision": "fp32", "prefetch": False})
    for key in ("logging", "visualization", "runtime", "simulation"):
        cfg.setdefault(key, copy.deepcopy(defaults[key]))
    for key in ("storage", "performance", "batch_size", "n_epochs", "max_grad_norm"):
        cfg["training"].setdefault(key, copy.deepcopy(defaults["training"][key]))
    # Absence in a historical checkpoint means its original FP32 execution.
    saved_performance = cfg["training"]["performance"]
    saved_performance.setdefault("fit_precision", "fp32")
    saved_performance.setdefault("prefetch", False)
    saved_performance.setdefault("fit_sequence_groups", 1)
    return cfg


def inspect_checkpoint(path):
    path = Path(path).resolve()
    if not path.suffix:
        path = path.with_suffix(".zip")
    if not path.exists() and (path.parent / "metadata.json").exists():
        # Diagnose an obsolete protocol from its plain metadata before touching
        # model serialization; a valid sidecar never substitutes for a missing ZIP.
        from pvz_rl.provenance import verify_engine

        legacy = json.loads((path.parent / "metadata.json").read_text("utf-8"))
        validate_config(legacy["config"])
        from pvz_rl.learning.training_requirements import require_supported_policy

        require_supported_policy(legacy["config"], legacy.get("condition", "masked"))
        verify_engine(legacy["config"])
    if path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("protocol") != DEMO_PROTOCOL:
            raise ValueError(
                "Unsupported demonstration checkpoint protocol; record a fresh entity_v1 demonstration"
            )
        if saved.get("recurrent_storage_protocol") != STATE_PROTOCOL:
            raise ValueError("Unsupported recurrent storage protocol")
        if not isinstance(saved.get("config"), dict) or saved.get("config_digest") != digest(
            saved["config"]
        ):
            raise ValueError("Checkpoint configuration digest does not match its contents")
        if saved.get("passes_completed", 0) < 1 or not isinstance(saved.get("model"), dict):
            raise ValueError("Demonstration checkpoint has no completed fitting pass or weights")
        if any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
            for value in saved["model"].values()
        ):
            raise ValueError("Demonstration checkpoint contains invalid or non-finite weights")
        cfg = saved["config"]
        from pvz_game import Rules

        from pvz_rl.envs.encoding import ObservationEncoder

        if saved.get("observation_schema") != ObservationEncoder(cfg, Rules()).schema():
            raise ValueError("Checkpoint entity schema disagrees with configuration")
        if cfg["policy"]["kind"] != "transformer_lstm_q_v2":
            raise ValueError("Demonstration checkpoint requires Transformer-LSTM weights")
        metadata = dict(
            config=cfg,
            condition="masked",
            family="preset",
            learner_seed=saved["learner_seed"],
            validation_limit=None,
            initialization_type="demonstration",
            payload=saved,
        )
    else:
        from pvz_rl.learning.cuda_q import checkpoint_metadata

        protocol = checkpoint_metadata(path)
        with ZipFile(path) as archive:
            if not {"data", "policy.pth", "policy.optimizer.pth", "cohort-state.pt"} <= set(
                archive.namelist()
            ):
                raise ValueError(
                    "Autonomous checkpoint is missing model, optimizer or recovery state"
                )
            metadata = (
                json.loads(archive.read("run.json"))
                if "run.json" in archive.namelist()
                else json.loads((path.parent / "metadata.json").read_text("utf-8"))
            )
        cfg = metadata["config"]
        from pvz_game import Rules

        from pvz_rl.envs.encoding import ObservationEncoder

        with ZipFile(path) as archive:
            if "observation-schema.json" not in archive.namelist():
                raise ValueError(
                    "Checkpoint is missing entity schema; fresh initialization is required"
                )
            if (
                json.loads(archive.read("observation-schema.json"))
                != ObservationEncoder(cfg, Rules()).schema()
            ):
                raise ValueError("Checkpoint entity schema disagrees with configuration")
        if protocol != protocol_for(cfg["policy"]["kind"]):
            raise ValueError("Checkpoint model family disagrees with saved configuration")
        metadata["initialization_type"] = "autonomous"
    # Demonstration checkpoints are weights-only artifacts.  Their structural
    # configuration is retained for schema and transfer validation, but every
    # execution-only setting comes from the current training profile.  This
    # keeps an old initialization useful after a BF16, batching or prefetch
    # change without requiring a refresh flag or another recording.
    cfg = (
        refresh_performance(metadata["config"], load_config())
        if metadata["initialization_type"] == "demonstration"
        else execution_config(metadata["config"])
    )
    validate_config(cfg)
    metadata["config"] = cfg
    return metadata


def model_class(cfg):
    if cfg["policy"]["kind"] != "transformer_lstm_q_v2":
        raise ValueError("Retired model; entity_v1 requires fresh initialization")
    from pvz_rl.learning.recurrent_q import CudaRecurrentQ

    return CudaRecurrentQ
