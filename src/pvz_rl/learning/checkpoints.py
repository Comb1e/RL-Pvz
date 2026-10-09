"""Inspect checkpoint identity and dispatch models before starting environments."""

import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from pvz_rl.config import digest, load_config, validate_config
from pvz_rl.learning.performance import refresh_performance
from pvz_rl.learning.training_requirements import transfer_protocol, weight_transfer_protocol

DEMO_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v3"
STATE_PROTOCOL = "pvz-rl/event-lstm-state-v1"
AUTONOMOUS_STATE_PROTOCOL = "pvz-rl/committed-event-lstm-state-v2"


def protocol_for(kind):
    methods = {
        "transformer_lstm_q_v3": "complete_return_event_lstm_v1",
    }
    if kind not in methods:
        raise ValueError(
            f"Incompatible checkpoint model family: {kind}; entity_v1 requires fresh initialization"
        )
    from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL

    return dict(policy=kind, optimizer=methods[kind], exploration=EXPLORATION_PROTOCOL)


def compatible_config(source, target, *, weights_only=False):
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
    for key in ("logging", "visualization", "runtime", "simulation"):
        cfg.setdefault(key, copy.deepcopy(defaults[key]))
    for key in ("storage", "performance", "batch_size", "n_epochs", "max_grad_norm"):
        cfg["training"].setdefault(key, copy.deepcopy(defaults["training"][key]))
    for key, value in defaults["training"]["performance"].items():
        cfg["training"]["performance"].setdefault(key, copy.deepcopy(value))
    return cfg


def inspect_checkpoint(path, *, weights_only=False):
    path = Path(path).resolve()
    if not path.suffix:
        path = path.with_suffix(".zip")
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
        if cfg["policy"]["kind"] != "transformer_lstm_q_v3":
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
            else:
                runtime = torch.load(
                    io.BytesIO(archive.read("cohort-state.pt")),
                    map_location="cpu",
                    weights_only=False,
                )
                if runtime.get("protocol") != AUTONOMOUS_STATE_PROTOCOL:
                    raise ValueError("Unsupported autonomous recovery protocol; use --init-from")
                if runtime.get("buffer") is not None:
                    from pvz_rl.learning.exploration import CommittedPlantExploration

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
    # Demonstration checkpoints are weights-only artifacts.  Their structural
    # configuration is retained for schema and transfer validation, but every
    # execution-only setting comes from the current training profile.  This
    # keeps an old initialization useful after a BF16, batching or prefetch
    # change without requiring a refresh flag or another recording.
    if metadata["initialization_type"] == "demonstration":
        current = load_config()
        source = copy.deepcopy(metadata["config"])
        # Demonstration weights are structural artifacts. Historical reward files
        # may predate the current optional shaping fields; hydrate those fields
        # from today's profile while retaining the source's architecture.
        source.setdefault("reward", {}).setdefault(
            "home_entry_penalties", current["reward"]["home_entry_penalties"]
        )
        source.setdefault("reward", {}).setdefault(
            "win_time_weight", current["reward"]["win_time_weight"]
        )
        for key in ("early_sun_extra_multiplier", "early_sun_spawn_fraction"):
            source["reward"].setdefault(key, copy.deepcopy(current["reward"][key]))
        source["training"]["exploration"] = copy.deepcopy(current["training"]["exploration"])
        source.setdefault("training", {})["objective"] = copy.deepcopy(
            current["training"]["objective"]
        )
        cfg = refresh_performance(source, current)
    else:
        cfg = execution_config(metadata["config"])
    validate_config(cfg)
    metadata["config"] = cfg
    return metadata


def model_class(cfg):
    if cfg["policy"]["kind"] != "transformer_lstm_q_v3":
        raise ValueError("Retired model; entity_v1 requires fresh initialization")
    from pvz_rl.learning.recurrent_q import CudaRecurrentQ

    return CudaRecurrentQ
