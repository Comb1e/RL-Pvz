"""CUDA eligibility, historical inference, and non-mutating pinned installation."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

from pvz_rl.cli import configured, main
from pvz_rl.config import load_config
from pvz_rl.learning.training import train
from pvz_rl.learning.training_requirements import require_cuda_training, resume_protocol


@pytest.mark.parametrize("setting", ["device", "simulator", "hybrid", "timing"])
def test_unsupported_direct_training_leaves_no_output(smoke_cfg, tmp_path, setting):
    if setting == "device":
        smoke_cfg["training"]["device"] = "cpu"
    elif setting == "simulator":
        smoke_cfg["simulation"]["backend"] = "cpu"
    elif setting == "timing":
        smoke_cfg["environment"].update(action_timing="fixed", decision_ticks=10)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        train(smoke_cfg, "hybrid" if setting == "hybrid" else "masked", 101, output)
    assert not output.exists()


@pytest.mark.parametrize("entry", ["train", "suite", "benchmark"])
def test_missing_cuda_fails_before_creating_output(smoke_cfg, tmp_path, monkeypatch, entry):
    from pvz_rl.learning.suite import run_suite
    from pvz_rl.monitoring.gpu_benchmark import benchmark_gpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / entry
    calls = {
        "train": lambda: train(smoke_cfg, "masked", 101, output),
        "suite": lambda: run_suite(smoke_cfg, output),
        "benchmark": lambda: benchmark_gpu(smoke_cfg, output),
    }
    with pytest.raises(RuntimeError, match="CUDA training is unavailable"):
        calls[entry]()
    assert not output.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--device", "cpu"],
        ["train", "--simulator", "cpu"],
        ["train", "--condition", "hybrid"],
        ["pilot"],
        ["benchmark"],
    ],
)
def test_retired_cli_options_are_not_executable(args, tmp_path):
    output = tmp_path / "absent"
    with pytest.raises(SystemExit):
        main([*args, "--output", str(output)])
    assert not output.exists()


def test_cpu_hardware_recommendation_is_rejected(tmp_path):
    path = tmp_path / "recommendation.json"
    path.write_text(json.dumps({"n_envs": 8, "device": "cuda", "simulator": "cpu"}))
    with pytest.raises(ValueError, match="removed"):
        configured(argparse.Namespace(command="train", config=None, hardware=path))


def test_cpu_resume_and_dormant_condition_compatibility(smoke_cfg, tmp_path):
    cfg = json.loads(json.dumps(smoke_cfg))
    original = resume_protocol(cfg, "masked")
    cfg["conditions"]["hybrid"] = dict(masked=True, shaped=True, curriculum=True, hybrid=True)
    assert resume_protocol(cfg, "masked") == original
    cfg["simulation"]["backend"] = "cpu"
    (tmp_path / "metadata.json").write_text(json.dumps({"config": cfg, "condition": "masked"}))
    with pytest.raises(ValueError, match="CPU training was removed"):
        train(smoke_cfg, "masked", 101, tmp_path / "new", resume=tmp_path / "old.zip")
    assert not (tmp_path / "new").exists()


@pytest.fixture
def staged_repository(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "tools/stage_game.py"
    spec = importlib.util.spec_from_file_location("staging_control", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project"
    lock = root / "src/pvz_rl/data/engine-lock.json"
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(root / "tools/stage_game.py"))
    repo = tmp_path / "game"
    repo.mkdir()

    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)
            .decode()
            .strip()
        )

    git("init")
    source = repo / "src/pvz_game/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("PINNED = True\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "pin")
    pin = git("rev-parse", "HEAD")
    manifest = {
        "__init__.py": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    }
    lock.write_text(json.dumps({"commit": pin, "files": manifest}))
    source.write_text("PINNED = False\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "new reader",
    )
    source.write_text("UNCOMMITTED = True\n")
    return module, repo, lock, git


def test_stage_pinned_archive_ignores_head_and_working_changes(staged_repository, tmp_path):
    module, repo, _, git = staged_repository
    before = git("rev-parse", "HEAD"), git("status", "--porcelain")
    output = module.stage_game(repo, tmp_path / "staged")
    assert (output / "src/pvz_game/__init__.py").read_text() == "PINNED = True\n"
    assert (git("rev-parse", "HEAD"), git("status", "--porcelain")) == before


@pytest.mark.parametrize("fault", ["missing_commit", "manifest"])
def test_stage_rejects_missing_or_mismatched_pin(staged_repository, tmp_path, fault):
    module, repo, lock, _ = staged_repository
    data = json.loads(lock.read_text())
    if fault == "missing_commit":
        data["commit"] = "1" * 40
    else:
        data["files"]["__init__.py"] = "bad"
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pinned commit|manifest"):
        module.stage_game(repo, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_only_one_recipe_is_shipped():
    root = Path(__file__).resolve().parents[1]
    expected = {"train"}
    assert {p.stem for p in (root / "configs").glob("*.toml")} == expected
    for name in expected:
        require_cuda_training(load_config(root / "configs" / f"{name}.toml"), runtime=False)


def test_benchmark_schedules_configurable_cuda_sizes(smoke_cfg, tmp_path, monkeypatch):
    import pvz_rl.monitoring.gpu_benchmark as benchmark

    seen = []

    class Monitor:
        samples = []

        def __init__(self, *args, **kwargs):
            pass

        def update(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def measure(cfg, seed, steps, output, **kwargs):
        require_cuda_training(cfg, runtime=False)
        seen.append((cfg["training"]["n_envs"], cfg["training"]["rollout_steps_per_env"], seed))
        return dict(state="complete", decisions_per_second=100, games_per_minute=2)

    monkeypatch.setattr(benchmark, "HardwareMonitor", Monitor)
    monkeypatch.setattr(benchmark, "measure", measure)
    result = benchmark.benchmark_gpu(smoke_cfg, tmp_path / "benchmark", minutes=1, steps=128)
    assert len(seen) == 12
    assert {(n, steps) for n, steps, _ in seen} == {(128, 128), (256, 128), (512, 128), (1024, 128)}
    assert {seed for _, _, seed in seen} == {800, 801, 802}
    assert result["n_envs"] == 128
    assert "speedup_over_current" not in result
    seen.clear()

    def limited(cfg, seed, steps, output, **kwargs):
        if cfg["training"]["n_envs"] == 512:
            raise MemoryError("test device memory exhausted")
        return measure(cfg, seed, steps, output, **kwargs)

    monkeypatch.setattr(benchmark, "measure", limited)
    result = benchmark.benchmark_gpu(
        smoke_cfg, tmp_path / "custom", minutes=1, steps=128, env_counts=[64, 512]
    )
    assert result["n_envs"] == 64
    assert len(seen) == 3
    rows = json.loads((tmp_path / "custom" / "measurements.json").read_text())
    assert sum(row["state"] == "failed" for row in rows) == 3
    assert all("memory exhausted" in row["error"] for row in rows if row["state"] == "failed")
    for counts in ([], [128, 128], [0], [1.5]):
        with pytest.raises(ValueError, match="env counts"):
            benchmark.benchmark_gpu(smoke_cfg, tmp_path / "bad", env_counts=counts)
    assert not (tmp_path / "bad").exists()


def test_compiler_failure_is_actionable_and_creates_no_run(smoke_cfg, tmp_path, monkeypatch):
    from pvz_rl.learning.training_requirements import _cuda_probe

    _cuda_probe.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "pvz_rl.monitoring.cuda_diagnostics.cuda_doctor",
        lambda: {
            "available": False,
            "error": "NVRTC compilation failed",
        },
    )
    with pytest.raises(RuntimeError, match="NVRTC.*bootstrap"):
        train(smoke_cfg, "masked", 101, tmp_path / "no-run")
    assert not (tmp_path / "no-run").exists()
    _cuda_probe.cache_clear()
