"""Inspect checkpoint identity and dispatch models before starting environments."""

import copy
import errno
import hashlib
import io
import json
import os
import time
from pathlib import Path
from zipfile import ZipFile

import torch

from pvz_rl.config import digest, load_config, training_env_capacity, validate_config
from pvz_rl.learning.performance import ENCODER_SETTINGS, refresh_performance
from pvz_rl.learning.training_requirements import (
    FIT_SCHEDULE_PROTOCOL,
    current_model_config,
    inference_config,
    transfer_protocol,
    weight_model_config,
    weight_transfer_protocol,
)

DEMO_PROTOCOL = "pvz-rl/shared-demo-initialization-v1"
MODEL_STATE_PROTOCOL = "pvz-rl/shared-board-reach-bf16-master-state-v1"
AUTONOMOUS_STATE_PROTOCOL = "pvz-rl/shared-reach-state-v1"


class SnapshotStager:
    """Stage one immutable CPU snapshot, packing CUDA transfers by dtype/device.

    Only layout descriptions are reused. Returned CPU storage belongs to that
    snapshot and cannot be overwritten by a later save or optimizer commit.
    ``max_bytes`` bounds simultaneous packed device and host scratch storage;
    direct per-tensor copies remain available under allocation pressure.
    """

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self._signature = None
        self._groups = ()
        self.mode = "direct"

    @staticmethod
    def _map(value, transform):
        if isinstance(value, torch.Tensor):
            return transform(value)
        if isinstance(value, dict):
            result = copy.copy(value)
            for key, item in value.items():
                result[key] = SnapshotStager._map(item, transform)
            return result
        if isinstance(value, (list, tuple)):
            items = [SnapshotStager._map(item, transform) for item in value]
            return tuple(items) if isinstance(value, tuple) else items
        return copy.deepcopy(value)

    def _layout(self, tensors):
        signature = tuple((t.device, t.dtype, t.shape, t.layout) for t in tensors)
        if signature != self._signature:
            groups = {}
            for index, tensor in enumerate(tensors):
                if tensor.device.type == "cuda":
                    groups.setdefault((tensor.device, tensor.dtype), []).append(index)
            self._groups = tuple(tuple(indices) for indices in groups.values())
            self._signature = signature

    @torch.no_grad()
    def _packed(self, tensors):
        result = [None] * len(tensors)
        for indices in self._groups:
            packed = torch.cat([tensors[i].detach().reshape(-1) for i in indices])
            # Blocking D2H completes before serialization and device scratch reuse.
            host = packed.cpu()
            offset = 0
            for index in indices:
                tensor = tensors[index]
                result[index] = host[offset : offset + tensor.numel()].view(tensor.shape)
                offset += tensor.numel()
            del packed
        for index, tensor in enumerate(tensors):
            if result[index] is None:
                result[index] = tensor.detach().to("cpu", copy=True)
        return result

    def stage(self, payload):
        tensors = []
        self._map(payload, lambda tensor: tensors.append(tensor))
        self._layout(tensors)
        packed_bytes = 2 * sum(
            t.numel() * t.element_size() for t in tensors if t.device.type == "cuda"
        )
        self.mode = "direct"
        staged = None
        if packed_bytes <= self.max_bytes and all(t.layout == torch.strided for t in tensors):
            try:
                staged = self._packed(tensors)
                self.mode = "packed"
            except (torch.cuda.OutOfMemoryError, MemoryError):
                # _packed's partial storage is released before the direct retry.
                pass
        if staged is None:
            staged = [tensor.detach().to("cpu", copy=True) for tensor in tensors]
        copies = iter(staged)
        return self._map(payload, lambda _: next(copies))


def retry_file_sharing(operation):
    """Bound retries for Windows readers briefly preventing atomic replacement."""
    delays = iter((0.02, 0.05, 0.1, 0.2, 0.4, 0.8))
    while True:
        try:
            return operation()
        except PermissionError as error:
            delay = next(delays, None)
            sharing_conflict = getattr(error, "winerror", None) in (5, 32, 33) or (
                getattr(error, "winerror", None) is None and error.errno == errno.EACCES
            )
            if os.name != "nt" or not sharing_conflict or delay is None:
                raise
            time.sleep(delay)


def load_torch_snapshot(path):
    """Deserialize and hash the same immutable bytes, closing the live file promptly."""
    contents = retry_file_sharing(lambda: Path(path).read_bytes())
    saved = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    return saved, hashlib.sha256(contents).hexdigest()


def protocol_for(kind):
    if kind != "shared_board_reach_q_v1":
        raise ValueError(
            f"Incompatible checkpoint model family: {kind}; entity_v1 requires fresh initialization"
        )
    from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL

    return dict(policy=kind, optimizer=FIT_SCHEDULE_PROTOCOL, exploration=EXPLORATION_PROTOCOL)


def compatible_config(source, target, *, weights_only=False):
    check = weight_model_config if weights_only else current_model_config
    if not check(source) or not check(target):
        raise ValueError(
            "Checkpoint model or objective is incompatible with this transfer mode. "
            "Old model weights and recovery cannot migrate; initialize this architecture fresh."
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
    for key in (
        "storage",
        "performance",
        "batch_size",
        "n_epochs",
        "max_grad_norm",
    ):
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
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            portable = "weights-only.json" in archive.namelist()
        if portable:
            return _inspect_completed_weights(path, weights_only=weights_only)
    if path.suffix == ".pt":
        saved, snapshot_sha256 = load_torch_snapshot(path)
        if saved.get("protocol") != DEMO_PROTOCOL:
            raise ValueError(
                "Incompatible initialization: start the shared board encoder fresh or initialize from verified gameplay demos"
            )
        initialization_type = "demonstration"
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
        completed = saved.get("passes_completed")
        if type(completed) is not int or completed < 1 or not saved.get("model"):
            raise ValueError("Demo initialization has no committed fitting pass")
        if any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
            for value in saved["model"].values()
        ):
            raise ValueError("Weight-only checkpoint contains invalid or non-finite weights")
        cfg = saved["config"]
        validate_config(execution_config(inference_config(cfg) if weights_only else cfg))
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
            initialization_type=initialization_type,
            payload=saved,
            snapshot_sha256=snapshot_sha256,
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
        if not weights_only:
            from pvz_rl.provenance import verify_engine

            verify_engine(cfg)
        validate_config(execution_config(inference_config(cfg) if weights_only else cfg))
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
            if weights_only:
                if (
                    runtime.get("protocol") != AUTONOMOUS_STATE_PROTOCOL
                    or runtime.get("model_storage_protocol") != MODEL_STATE_PROTOCOL
                    or runtime.get("precision_contract") != "bf16_master"
                ):
                    raise ValueError("Unsupported autonomous recovery protocol weight contract")
            else:
                validate_runtime_identity(runtime)
            if not weights_only:
                from pvz_rl.learning.peashooter_teacher import validate_rule_recovery

                validate_rule_recovery(runtime, cfg)
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
                    capacity = training_env_capacity(cfg, metadata["family"])
                    CollectionScheduler(capacity, "cpu").restore(runtime["collection_scheduler"])
                    controller = runtime.get("plant_exploration")
                    if not isinstance(controller, dict):
                        raise ValueError("Incomplete committed plant recovery state")
                    state = CommittedPlantExploration(
                        capacity,
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
        metadata["fit_schedule_protocol"] = protocol["optimizer"]
    if weights_only:
        return metadata
    if metadata["initialization_type"] == "demonstration":
        cfg = refresh_performance(metadata["config"], load_config())
    else:
        cfg = execution_config(metadata["config"])
    validate_config(cfg)
    metadata["config"] = cfg
    return metadata


def _inspect_completed_weights(path, *, weights_only):
    from pvz_game import Rules

    from pvz_rl.envs.encoding import ObservationEncoder
    from pvz_rl.learning.cuda_q import checkpoint_metadata
    from pvz_rl.learning.training_artifacts import WEIGHTS_PROTOCOL

    if not weights_only:
        raise ValueError(
            "Completed runs retain weights only; use --init-from or inference. "
            "Resume requires an interrupted/latest recovery checkpoint."
        )
    checkpoint_metadata(path, weights_only=True)
    with ZipFile(path) as archive:
        required = {
            "policy.pth",
            "protocol.json",
            "observation-schema.json",
            "run.json",
            "weights-only.json",
        }
        if set(archive.namelist()) != required:
            raise ValueError("Completed weight archive has unexpected or missing entries")
        marker = json.loads(archive.read("weights-only.json"))
        metadata = json.loads(archive.read("run.json"))
        cfg = metadata["config"]
        if marker != dict(protocol=WEIGHTS_PROTOCOL, config_digest=digest(cfg)):
            raise ValueError("Completed weights configuration or protocol does not match")
        validate_config(execution_config(inference_config(cfg)))
        if (
            json.loads(archive.read("observation-schema.json"))
            != ObservationEncoder(cfg, Rules()).schema()
        ):
            raise ValueError("Completed weights entity schema disagrees with configuration")
        if metadata.get("completion", {}).get("state") != "complete":
            raise ValueError("Completed weights are missing completion metadata")
    metadata["initialization_type"] = "weights"
    return metadata


def model_class(cfg):
    if cfg["policy"]["kind"] != "shared_board_reach_q_v1":
        raise ValueError("Retired model; entity_v1 requires fresh initialization")
    from pvz_rl.learning.sequential_training import CudaSequentialQ

    return CudaSequentialQ
