"""Hardware selection must not silently reinterpret an archived protocol."""

import argparse
import json

import pytest

from pvz_rl.cli import configured
from pvz_rl.config import load_config, research_config, simulator
from pvz_rl.monitoring.gpu_benchmark import recommend


def args(**kwargs):
    return argparse.Namespace(**{"command": "train", "config": None, **kwargs})


def test_new_shared_run_defaults_and_explicit_parallelism():
    cfg = configured(args())
    assert simulator(cfg) == "cuda"
    assert cfg["training"]["n_envs"] == 128
    assert cfg["training"]["method"] == "complete_return_joint_tile_probe_v1"
    for count in (3, 32, 64, 128, 256, 512, 1024):
        cfg = configured(args(n_envs=count))
        assert cfg["training"]["n_envs"] == count


@pytest.mark.parametrize("artifact", ["autonomous", "demonstration"])
@pytest.mark.parametrize("mode", ["init_from", "resume"])
def test_previous_plant_probe_checkpoints_reject_before_cuda(tmp_path, monkeypatch, artifact, mode):
    from zipfile import ZipFile

    import torch

    from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL

    def forbidden(*args, **kwargs):
        raise AssertionError("Retired checkpoint reached simulator allocation")

    monkeypatch.setattr("pvz_rl.learning.training_requirements._cuda_probe", forbidden)
    monkeypatch.setattr("pvz_rl.envs.cuda_env.CudaVecEnv.__init__", forbidden)
    if artifact == "autonomous":
        path = tmp_path / "previous.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr(
                "protocol.json",
                json.dumps(
                    dict(
                        policy="transformer_lstm_q_v4",
                        optimizer="complete_return_joint_probe_v1",
                        exploration=EXPLORATION_PROTOCOL,
                    )
                ),
            )
    else:
        path = tmp_path / "previous.pt"
        torch.save(dict(protocol="pvz-rl/demo-initialization-checkpoint-v4"), path)
    with pytest.raises(ValueError, match="Retired|Unsupported|fresh|refit"):
        if mode == "init_from":
            from pvz_rl.learning.training import initial_weights

            initial_weights(path, load_config())
        else:
            configured(args(resume=path))


@pytest.mark.parametrize("fault", ["missing_scheduler", "scheduler_counts", "pending_evidence"])
def test_incomplete_background_recovery_rejected_before_allocation(tmp_path, monkeypatch, fault):
    from io import BytesIO
    from zipfile import ZipFile

    import numpy as np
    import torch
    from pvz_game import Rules

    from pvz_rl.envs.encoding import ObservationEncoder
    from pvz_rl.learning.checkpoints import (
        AUTONOMOUS_STATE_PROTOCOL,
        inspect_checkpoint,
        protocol_for,
    )
    from pvz_rl.learning.collection_scheduler import CollectionScheduler
    from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
    from pvz_rl.learning.exploration import CommittedPlantExploration

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete checkpoint reached simulator allocation")

    monkeypatch.setattr("pvz_rl.learning.training_requirements._cuda_probe", forbidden)
    monkeypatch.setattr("pvz_rl.envs.cuda_env.CudaVecEnv.__init__", forbidden)
    cfg = load_config()
    count = cfg["training"]["n_envs"]
    controller = CommittedPlantExploration(count, "cpu", 0.1, 0.5)
    buffer = CompleteGameBuffer(tmp_path / "buffer", count, cfg=cfg)
    try:
        buffer.exploration = dict(plant_epsilon=0.1, tile_epsilon=0.5)
        metadata = buffer.save(tmp_path / "saved")
    finally:
        buffer.close()
    runtime = dict(
        protocol=AUTONOMOUS_STATE_PROTOCOL,
        buffer=metadata,
        collection_scheduler=CollectionScheduler(count, "cpu").snapshot(),
        teacher={},
        teacher_memory={},
        probe_schedule={},
        compilation={},
        pending_episodes={},
        plant_exploration=controller.snapshot(),
    )
    if fault == "missing_scheduler":
        runtime.pop("collection_scheduler")
    elif fault == "scheduler_counts":
        runtime["collection_scheduler"]["decisions"] = np.array([-1] * count)
    else:
        runtime["buffer"]["pending_probes"] = 1
    stream = BytesIO()
    torch.save(runtime, stream)
    path = tmp_path / "incomplete.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("protocol.json", json.dumps(protocol_for(cfg["policy"]["kind"])))
        archive.writestr("run.json", json.dumps(dict(config=cfg)))
        archive.writestr("data", json.dumps(dict(plant_exploration_rate=0.1, exploration_rate=0.5)))
        archive.writestr(
            "observation-schema.json", json.dumps(ObservationEncoder(cfg, Rules()).schema())
        )
        archive.writestr("cohort-state.pt", stream.getvalue())
        for name in ("policy.pth", "policy.optimizer.pth"):
            archive.writestr(name, b"unread weights")
    with pytest.raises(ValueError, match="Incomplete|Invalid collection"):
        inspect_checkpoint(path)


@pytest.mark.parametrize(
    "options", [{"simulator": "cpu"}, {"device": "cpu"}, {"condition": "hybrid"}]
)
def test_cpu_and_hybrid_requests_are_rejected(options):
    with pytest.raises(ValueError, match="removed|not supported"):
        configured(args(**options))


def test_hardware_recommendations_and_override_conflicts(tmp_path):
    path = tmp_path / "recommendation.json"
    path.write_text(json.dumps(dict(n_envs=64, device="cuda", simulator="cuda")))
    cfg = configured(args(hardware=path))
    assert simulator(cfg) == "cuda" and cfg["training"]["n_envs"] == 64
    cfg = configured(args(hardware=path, n_envs=32))
    assert cfg["training"]["n_envs"] == 32
    path.write_text(json.dumps(dict(n_envs=None, device="cuda")))
    with pytest.raises(ValueError, match="no usable"):
        configured(args(hardware=path))


def test_resume_uses_saved_protocol_and_instrumentation_is_optional(tmp_path, monkeypatch):
    saved = load_config()
    saved["training"].update(n_envs=1024)
    (tmp_path / "metadata.json").write_text(json.dumps({"config": saved}))
    monkeypatch.setattr(
        "pvz_rl.learning.checkpoints.inspect_checkpoint", lambda path: {"config": saved}
    )
    cfg = configured(args(resume=tmp_path / "latest.zip"))
    assert research_config(cfg) == research_config(saved)
    cfg["simulation"] = {"backend": "cuda", "profile": True}
    assert research_config(cfg) == research_config(saved)
    cfg["simulation"]["backend"] = "cpu"
    assert research_config(cfg) != research_config(saved)


def test_promotion_requires_complete_stable_trials_and_correctness():
    rows = [
        dict(profile=name, state="complete", decisions_per_second=rate)
        for name, rates in {
            "cuda-32": [150, 151, 152],
            "cuda-64": [155, 156, 157],
            "cuda-128": [100, 300, 500],
        }.items()
        for rate in rates
    ]
    # 32 is within 5% of stable 64; the unstable 128 profile cannot win.
    result = recommend(rows, parity_passed=True, provisional=False)
    assert result["n_envs"] == 32 and result["promote_default"]
    assert not recommend(rows)["promote_default"]
    assert not recommend(rows[:-1], parity_passed=True)["promote_default"]
    assert recommend([])["n_envs"] is None


def test_recommendation_handles_larger_batches_and_failed_trials():
    rows = [
        dict(profile=f"cuda-{n}", state="complete", decisions_per_second=rate)
        for n, rates in {
            128: [100, 102, 101],
            256: [149, 150, 151],
            512: [151, 154, 153],
            1024: [200, 201],
        }.items()
        for rate in rates
    ]
    rows.append(dict(profile="cuda-1024", state="failed", decisions_per_second=0))
    rows.extend(
        dict(profile="cuda-equivalent", state="complete", decisions_per_second=100)
        for _ in range(3)
    )
    # Incomplete fastest batch cannot win; 256 is within 5% of stable 512.
    result = recommend(rows)
    assert result["n_envs"] == 256
    assert "cuda-1024" not in result["median_decisions_per_second"]


@pytest.mark.parametrize("key", ["collection_graph_cache_entries", "collection_vram_headroom_mib"])
@pytest.mark.parametrize("value", [-1, True, 1.5, None])
def test_collection_graph_settings_reject_invalid_values(key, value):
    from pvz_rl.config import validate_config

    cfg = load_config()
    cfg["training"]["performance"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_collection_graph_settings_defaults_refresh_and_zero_boundaries():
    from pvz_rl.config import validate_config
    from pvz_rl.learning.performance import refresh_performance, without_performance

    current = load_config()
    settings = current["training"]["performance"]
    assert settings["collection_graph_cache_entries"] == 8
    assert settings["collection_vram_headroom_mib"] == 512
    saved = load_config()
    saved["training"]["performance"].update(
        collection_graph_cache_entries=0, collection_vram_headroom_mib=0
    )
    saved["training"]["learning_rate"] = 0.001
    validate_config(saved)
    refreshed = refresh_performance(saved, current)
    assert refreshed["training"]["performance"] == settings
    assert refreshed["training"]["learning_rate"] == 0.001
    assert saved["training"]["performance"]["collection_graph_cache_entries"] == 0
    assert "performance" not in without_performance(refreshed)["training"]
