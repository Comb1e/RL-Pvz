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

from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.learning.performance import ENCODER_SETTINGS, refresh_performance
from pvz_rl.learning.training_requirements import (
    current_model_config,
    inference_config,
    transfer_protocol,
    weight_model_config,
    weight_transfer_protocol,
)

DEMO_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v10"
LEGACY_DEMO_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v9"
ANNOTATION_CHECKPOINT_PROTOCOL = "pvz-rl/annotation-initialization-v3"
ANNOTATION_PROGRESS_PROTOCOL = "pvz-rl/annotation-progress-v4"
LEGACY_ANNOTATION_PROGRESS_PROTOCOL = "pvz-rl/annotation-progress-v3"
LEGACY_ANNOTATION_CHECKPOINT_PROTOCOL_V2 = "pvz-rl/annotation-initialization-v2"
LEGACY_ANNOTATION_CHECKPOINT_PROTOCOL = "pvz-rl/tile-annotation-initialization-v1"
MODEL_STATE_PROTOCOL = "pvz-rl/stateless-two-stage-bf16-master-state-v2"
AUTONOMOUS_STATE_PROTOCOL = "pvz-rl/action-global-state-v4"


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
    methods = {
        "two_stage_transformer_q_v1": "complete_return_action_global_v3",
    }
    if kind not in methods:
        raise ValueError(
            f"Incompatible checkpoint model family: {kind}; entity_v1 requires fresh initialization"
        )
    from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL

    return dict(policy=kind, optimizer=methods[kind], exploration=EXPLORATION_PROTOCOL)


def compatible_config(source, target, *, weights_only=False):
    check = weight_model_config if weights_only else current_model_config
    if not check(source) or not check(target):
        raise ValueError(
            "Checkpoint model or objective is incompatible with this transfer mode. "
            "Previous compatible BF16-master weights allow fresh initialization only; "
            "old-objective runs cannot resume."
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
        saved, snapshot_sha256 = load_torch_snapshot(path)
        progress = saved.get("protocol") == ANNOTATION_PROGRESS_PROTOCOL
        if saved.get("protocol") == LEGACY_ANNOTATION_PROGRESS_PROTOCOL:
            raise ValueError(
                "Annotation progress v3 must first be loaded with the annotation fitter's "
                "--resume-from to write v4 progress; then use that file with --init-from"
            )
        if progress and not weights_only:
            raise ValueError(
                "Annotation progress supports --init-from only here; train --resume "
                "requires an autonomous checkpoint. Use an annotation fitter's "
                "--resume-from to continue annotation optimization."
            )
        initialization_type = {
            DEMO_PROTOCOL: "demonstration",
            LEGACY_DEMO_PROTOCOL: "demonstration",
            ANNOTATION_CHECKPOINT_PROTOCOL: "annotation",
            ANNOTATION_PROGRESS_PROTOCOL: "annotation",
            LEGACY_ANNOTATION_CHECKPOINT_PROTOCOL_V2: "annotation",
            LEGACY_ANNOTATION_CHECKPOINT_PROTOCOL: "annotation",
        }.get(saved.get("protocol"))
        if initialization_type is None:
            raise ValueError(
                "Unsupported weight-only checkpoint protocol; create a current demonstration or tile annotation initialization"
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
        convergence = saved.get("convergence", {})
        if not isinstance(convergence, dict):
            raise ValueError("Checkpoint convergence metadata must be a dictionary")
        current_annotation_convergence = saved.get("protocol") == ANNOTATION_CHECKPOINT_PROTOCOL
        zero_pass_verified = (
            initialization_type == "annotation"
            and convergence.get("verified") is True
            and type(convergence.get("records")) is int
            and convergence["records"] > 0
            and convergence.get("hits") == convergence["records"]
            and (
                not current_annotation_convergence
                or (
                    convergence.get("absolute_bottom_hits") == convergence["records"]
                    and type(convergence.get("validation_records")) is int
                    and convergence["validation_records"] >= 0
                    and convergence.get("validation_hits") == convergence["validation_records"]
                    and convergence.get("validation_absolute_bottom_hits")
                    == convergence["validation_records"]
                )
            )
        )
        completed = saved.get("passes_completed")
        if (
            type(completed) is not int
            or completed < (0 if progress or zero_pass_verified else 1)
            or not isinstance(saved.get("model"), dict)
            or not saved["model"]
        ):
            raise ValueError("Weight-only checkpoint has no completed fitting pass or weights")
        if progress:
            identity = saved.get("identity")
            settings = identity.get("settings") if isinstance(identity, dict) else None
            if (
                saved.get("annotation_kind") not in ("tile", "action")
                or not isinstance(settings, dict)
                or settings.get("annotation_kind") != saved["annotation_kind"]
            ):
                raise ValueError("Annotation progress has invalid annotation kind metadata")
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
        if initialization_type == "annotation":
            metadata["source_annotation"] = dict(
                source_passes_completed=completed,
                source_annotation_kind=saved.get("annotation_kind")
                or saved.get("annotation", {}).get("annotation_kind", "tile"),
                source_converged=convergence.get("verified") is True,
                source_checkpoint_protocol=saved["protocol"],
            )
            metadata["source_counters"] = dict(steps=0, games=0, updates=completed)
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
                    runtime.get("protocol")
                    not in (
                        AUTONOMOUS_STATE_PROTOCOL,
                        "pvz-rl/action-global-state-v3",
                        "pvz-rl/action-global-state-v2",
                        "pvz-rl/action-global-state-v1",
                        "pvz-rl/stateless-no-probe-state-v1",
                        "pvz-rl/selective-invalid-state-v2",
                    )
                    or runtime.get("model_storage_protocol") != MODEL_STATE_PROTOCOL
                    or runtime.get("precision_contract") != "bf16_master"
                ):
                    raise ValueError("Unsupported autonomous recovery protocol weight contract")
            else:
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
