"""Inspect checkpoint identity and dispatch models before starting environments."""

import copy
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.learning.training_requirements import transfer_protocol

DEMO_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v1"
STATE_PROTOCOL = "pvz-rl/lstm-state-v1"


def protocol_for(kind):
    methods = {
        "event_sequential_q_v2": "sequential_q_mc_v2",
        "transformer_lstm_q_v1": "complete_return_lstm_v1",
    }
    if kind not in methods:
        raise ValueError(f"Incompatible checkpoint model family: {kind}")
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
    for key in ("logging", "visualization", "runtime", "simulation"):
        cfg.setdefault(key, copy.deepcopy(defaults[key]))
    for key in ("storage", "performance", "batch_size", "n_epochs", "max_grad_norm"):
        cfg["training"].setdefault(key, copy.deepcopy(defaults["training"][key]))
    return cfg


def migrate_curriculum(metadata):
    """Read old model weights without reinstating removed lesson implementations."""
    from pvz_rl.learning.curriculum import STAGES

    cfg = copy.deepcopy(metadata["config"])
    curriculum = cfg["curriculum"]
    removed = set(curriculum.get("stages", {})) - set(STAGES)
    if not removed:
        return cfg
    if len(removed) != 1 or not set(STAGES) <= set(curriculum["stages"]):
        raise ValueError("Unsupported historical curriculum schema")
    metadata["previous_stage_order"] = [*sorted(removed), *STAGES]
    if curriculum.get("run_stage") in removed:
        metadata["resume_unsupported_reason"] = "the selected curriculum lesson was removed"
        curriculum.pop("run_stage")
        cfg["training"]["until_stage_complete"] = False
    curriculum.pop("lessons", None)
    for name in removed:
        curriculum["stages"].pop(name)
    for stage in curriculum["stages"].values():
        pairs = [
            (task, weight)
            for task, weight in zip(stage["tasks"], stage["weights"])
            if task not in removed
        ]
        total = sum(weight for _, weight in pairs)
        if not total:
            raise ValueError("Historical curriculum has no retained task weight")
        stage["tasks"] = [task for task, _ in pairs]
        stage["weights"] = [weight / total for _, weight in pairs]
        stage["requirements"] = {k: v for k, v in stage["requirements"].items() if k not in removed}
    return cfg


def migrate_resume_state(model, metadata):
    """Translate retained stage indices; never resume an unavailable lesson."""
    from pvz_rl.learning.curriculum import STAGES

    previous = metadata.get("previous_stage_order")
    if not previous:
        return
    if metadata.get("resume_unsupported_reason"):
        raise ValueError(
            "Cannot resume: " + metadata["resume_unsupported_reason"] + "; use --init-from"
        )

    def stage(index):
        name = previous[index]
        if name not in STAGES:
            raise ValueError("Cannot resume a removed curriculum lesson; use --init-from")
        return STAGES.index(name)

    curriculum = getattr(model, "curriculum_state", None)
    if curriculum:
        curriculum["stage"] = stage(curriculum["stage"])
    pending = getattr(model, "research_schedule", {}).get("pending_stage_validation")
    if pending and pending.get("stage") not in STAGES:
        raise ValueError("Cannot resume pending evaluation of a removed lesson; use --init-from")
    runtime = model.runtime_state or {}
    environment = runtime.get("environment")
    if environment:
        environment["queue_stage"] = stage(environment["queue_stage"])
        attributes = environment["attributes"]
        for episode in attributes["_episode"]:
            if episode and episode[1] not in ("preset", "diagnostic"):
                raise ValueError(
                    "Cannot resume unfinished removed-lesson trajectories; use --init-from"
                )
        attributes["_episode_stages"] = [stage(i) for i in attributes["_episode_stages"]]


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
            raise ValueError("Unsupported demonstration checkpoint protocol")
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
        if cfg["policy"]["kind"] != "transformer_lstm_q_v1":
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
        if protocol != protocol_for(cfg["policy"]["kind"]):
            raise ValueError("Checkpoint model family disagrees with saved configuration")
        metadata["initialization_type"] = "autonomous"
    cfg = execution_config(migrate_curriculum(metadata))
    validate_config(cfg)
    metadata["config"] = cfg
    return metadata


def model_class(cfg):
    if cfg["policy"]["kind"] == "transformer_lstm_q_v1":
        from pvz_rl.learning.recurrent_q import CudaRecurrentQ

        return CudaRecurrentQ
    from pvz_rl.learning.cuda_q import CudaSequentialQ

    return CudaSequentialQ
