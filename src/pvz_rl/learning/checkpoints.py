"""Inspect checkpoint identity and dispatch models before starting environments."""

import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.learning.performance import ENCODER_SETTINGS, refresh_performance
from pvz_rl.learning.training_requirements import (
    current_model_config,
    transfer_protocol,
    weight_transfer_protocol,
)

DEMO_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v9"
MODEL_STATE_PROTOCOL = "pvz-rl/stateless-two-stage-bf16-master-state-v2"
AUTONOMOUS_STATE_PROTOCOL = "pvz-rl/stateless-no-probe-state-v1"


def protocol_for(kind):
    methods = {
        "two_stage_transformer_q_v1": "complete_return_sequential_no_probe_v1",
    }
    if kind not in methods:
        raise ValueError(
            f"Incompatible checkpoint model family: {kind}; entity_v1 requires fresh initialization"
        )
    from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL

    return dict(policy=kind, optimizer=methods[kind], exploration=EXPLORATION_PROTOCOL)


def compatible_config(source, target, *, weights_only=False):
    if not current_model_config(source) or not current_model_config(target):
        raise ValueError(
            "Old-objective checkpoints cannot resume or initialize weights. "
            "Refit a verified demonstration in a new directory."
        )
    protocol = weight_transfer_protocol if weights_only else transfer_protocol
    if protocol(source) != protocol(target):
        raise ValueError(
            "Checkpoint transfer requires matching engine, observation encoding, action "
            "semantics and network structure; incompatible model transfer"
        )


def execution_config(cfg):
    """Fill optional execution settings; never replace recorded protocol parameters."""
    cfg = copy.deepcopy(cfg)
    defaults = load_config()
    for key in ENCODER_SETTINGS:
        cfg["policy"].setdefault(key, copy.deepcopy(defaults["policy"][key]))
    for key in ("logging", "visualization", "runtime", "simulation"):
        cfg.setdefault(key, copy.deepcopy(defaults[key]))
    for key in ("storage", "performance", "batch_size", "n_epochs", "max_grad_norm"):
        cfg["training"].setdefault(key, copy.deepcopy(defaults["training"][key]))
    for key, value in defaults["training"]["performance"].items():
        cfg["training"]["performance"].setdefault(key, copy.deepcopy(value))
    return cfg


def validate_runtime_identity(runtime):
    if (
        not isinstance(runtime, dict)
        or runtime.get("protocol") != AUTONOMOUS_STATE_PROTOCOL
        or runtime.get("model_storage_protocol") != MODEL_STATE_PROTOCOL
    ):
        raise ValueError(
            "Unsupported autonomous recovery protocol; fresh initialization is required"
        )
    if any(key in runtime for key in ("memory", "teacher_memory")):
        raise ValueError("Stateless recovery cannot contain learned-history state")
    if any(
        key in runtime for key in ("probe_schedule", "probe_pool", "probe_jobs", "pending_probes")
    ):
        raise ValueError("Counterfactual rollout recovery is no longer supported")
    if runtime.get("precision_contract") != "bf16_master":
        raise ValueError("Unsupported runtime precision contract; fresh initialization is required")


def inspect_checkpoint(path, *, weights_only=False):
    path = Path(path).resolve()
    if not path.suffix:
        path = path.with_suffix(".zip")
    if path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("protocol") != DEMO_PROTOCOL:
            raise ValueError(
                "Unsupported demonstration checkpoint protocol; refit a verified recording in a new directory"
            )
        if (
            saved.get("model_storage_protocol") != MODEL_STATE_PROTOCOL
            or saved.get("precision_contract") != "bf16_master"
        ):
            raise ValueError(
                "Unsupported stateless model storage protocol; fresh weights are required"
            )
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
        validate_config(execution_config(cfg))
        from pvz_game import Rules

        from pvz_rl.envs.encoding import ObservationEncoder

        if saved.get("observation_schema") != ObservationEncoder(cfg, Rules()).schema():
            raise ValueError("Checkpoint entity schema disagrees with configuration")
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

        protocol = checkpoint_metadata(path, weights_only=weights_only)
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
        validate_config(execution_config(cfg))
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
            stored = json.loads(archive.read("data"))
            if weights_only:
                metadata["source_counters"] = {
                    key: stored.get(field, 0)
                    for key, field in (
                        ("steps", "num_timesteps"),
                        ("games", "training_games"),
                        ("updates", "_n_updates"),
                    )
                }
            runtime = torch.load(
                io.BytesIO(archive.read("cohort-state.pt")),
                map_location="cpu",
                weights_only=False,
            )
            validate_runtime_identity(runtime)
            if not weights_only:
                if runtime.get("buffer") is not None:
                    from pvz_rl.learning.collection_scheduler import CollectionScheduler
                    from pvz_rl.learning.exploration import CommittedPlantExploration

                    if any(
                        runtime.get(key) is None
                        for key in (
                            "collection_scheduler",
                            "teacher",
                            "compilation",
                            "pending_episodes",
                        )
                    ):
                        raise ValueError("Incomplete no-probe recovery state")
                    CollectionScheduler(cfg["training"]["n_envs"], "cpu").restore(
                        runtime["collection_scheduler"]
                    )
                    controller = runtime.get("plant_exploration")
                    if not isinstance(controller, dict):
                        raise ValueError("Incomplete committed plant recovery state")
                    state = CommittedPlantExploration(
                        cfg["training"]["n_envs"],
                        "cpu",
                        stored.get("plant_exploration_rate", float("nan")),
                        stored.get("exploration_rate", float("nan")),
                    )
                    state.restore(controller)
                    from pvz_rl.learning.cuda_buffer import CompleteGameBuffer

                    CompleteGameBuffer.validate_metadata(runtime["buffer"])
                    if runtime["buffer"]["exploration"] != dict(
                        plant_epsilon=state.plant_epsilon, tile_epsilon=state.tile_epsilon
                    ):
                        raise ValueError("Trajectory exploration differs from recovery state")
        if not weights_only and protocol != protocol_for(cfg["policy"]["kind"]):
            raise ValueError("Checkpoint model family disagrees with saved configuration")
        metadata["initialization_type"] = "autonomous"
    if weights_only:
        return metadata
    if metadata["initialization_type"] == "demonstration":
        cfg = refresh_performance(metadata["config"], load_config())
    else:
        cfg = execution_config(metadata["config"])
    validate_config(cfg)
    metadata["config"] = cfg
    return metadata


def model_class(cfg):
    if cfg["policy"]["kind"] != "two_stage_transformer_q_v1":
        raise ValueError("Retired model; entity_v1 requires fresh initialization")
    from pvz_rl.learning.sequential_training import CudaSequentialQ

    return CudaSequentialQ
