import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from pvz_rl.config import validate_config
from pvz_rl.presentation.reporting import make_report


def test_hardware_sampler_flushes_sessions_and_keeps_null_measurements(tmp_path, monkeypatch):
    from pvz_rl.monitoring.hardware import HardwareMonitor
    from pvz_rl.presentation.visualization import read_series

    sampled = threading.Event()

    def query(identifier, timeout):
        assert identifier == "GPU-test" and timeout == 0.1
        sampled.set()
        return {"gpu_percent": None, "gpu_error": "Unavailable"}

    monkeypatch.setattr("pvz_rl.monitoring.hardware.gpu_sample", query)
    path = tmp_path / "hardware-metrics.jsonl"
    for phase in ("warmup", "formal"):
        sampled.clear()
        with HardwareMonitor(path, seconds=0.1, gpu_id="GPU-test") as monitor:
            monitor.update(stage="saving", phase=phase, activity="validation", training_steps=128)
            assert sampled.wait(3)
        assert not monitor.thread.is_alive()
        assert monitor.latest["gpu_percent"] is None
        assert monitor.latest["system_cpu_percent"] >= 0
        assert monitor.latest["process_ram_mib"] > 0
        assert monitor.samples.maxlen == 6000
    rows = read_series(path)
    assert len({row["session"] for row in rows}) == 2
    assert {row["phase"] for row in rows} == {"warmup", "formal"}
    assert all(row["activity"] == "validation" and row["stage"] == "saving" for row in rows)
    with path.open("a") as stream:
        stream.write('{"partial":')
    assert read_series(path, tolerate_partial_tail=True) == rows
    with pytest.raises(json.JSONDecodeError):
        read_series(path)


@pytest.mark.parametrize("response", ["partial", "timeout", "exit"])
def test_gpu_query_targets_uuid_and_exposes_missing_data(monkeypatch, response):
    from pvz_rl.monitoring.hardware import gpu_sample

    def run(command, **kwargs):
        assert "--id=GPU-selected" in command and kwargs["timeout"] == 0.1
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if response == "timeout":
            raise subprocess.TimeoutExpired(command, 0.1)
        return SimpleNamespace(
            returncode=1 if response == "exit" else 0,
            stderr="failed",
            stdout="40, N/A, 500, [Not Supported], 65",
        )

    monkeypatch.setattr("pvz_rl.monitoring.hardware.subprocess.run", run)
    row = gpu_sample("GPU-selected", 0.1)
    assert row["gpu_error"] and row["gpu_memory_percent"] is None
    assert row["gpu_percent"] == (40 if response == "partial" else None)


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("encoding", "count_scale", 0),
        ("reward", "value_scale", 0),
        ("training", "learner_seeds", [101, 101]),
        ("training", "hidden_sizes", [0]),
        ("training", "learning_rate", float("nan")),
        ("training", "gamma", float("nan")),
    ],
)
def test_invalid_research_config_rejected(cfg, group, key, value):
    cfg[group][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_report_rejects_incompatible_game_protocols(cfg, tmp_path):
    rows = [{"protocol_hash": "A"}, {"protocol_hash": "B"}]
    source = tmp_path / "bad.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="same verified"):
        make_report([source], tmp_path / "report", cfg)


def test_report_rejects_mixed_training_budgets(cfg, tmp_path):
    rows = [
        {
            "protocol_hash": "same",
            "training_config_hash": str(seed),
            "split": "test",
            "family": "preset",
            "level": "easy",
            "policy": "masked",
            "learner_seed": seed,
            "scenario_seed": 200000,
            "win": 1,
        }
        for seed in (101, 102)
    ]
    source = tmp_path / "bad.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="pool different"):
        make_report([source], tmp_path / "report", cfg)
