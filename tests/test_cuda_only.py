"""CUDA eligibility, historical inference, and non-mutating pinned installation."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pvz_rl.cli import configured, main
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
    with pytest.raises(ValueError, match="removed|not supported"):
        configured(argparse.Namespace(command="train", config=None, hardware=path))


def test_cpu_resume_and_dormant_condition_compatibility(smoke_cfg, tmp_path, monkeypatch):
    cfg = json.loads(json.dumps(smoke_cfg))
    original = resume_protocol(cfg, "masked")
    cfg["conditions"]["hybrid"] = dict(masked=True, shaped=True, curriculum=True, hybrid=True)
    assert resume_protocol(cfg, "masked") == original
    cfg["simulation"]["backend"] = "cpu"
    (tmp_path / "metadata.json").write_text(json.dumps({"config": cfg, "condition": "masked"}))
    monkeypatch.setattr(
        "pvz_rl.learning.checkpoints.inspect_checkpoint",
        lambda path: {
            "config": cfg,
            "condition": "masked",
            "initialization_type": "autonomous",
        },
    )
    with pytest.raises(ValueError, match="CPU training is not supported"):
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


@pytest.fixture
def bash_bootstrap(staged_repository, tmp_path):
    _, repo, lock, _ = staged_repository
    git_executable = shutil.which("git")
    if os.name == "nt":
        bash = Path(git_executable).resolve().parents[1] / "bin/bash.exe"
        if not bash.is_file():
            pytest.skip("Git Bash is required for installation controls on Windows")
    else:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("Bash is required for installation controls")

    def shell_path(path):
        if os.name == "nt":
            return subprocess.check_output(
                [str(bash), "-c", 'cygpath -u "$1"', "paths", str(path)], text=True
            ).strip()
        return str(path)

    project = tmp_path / "project with spaces"
    shutil.copytree(lock.parents[3], project)
    game = tmp_path / "game checkout"
    shutil.copytree(repo, game)
    tools = project / "tools"
    tools.mkdir()
    original = Path(__file__).resolve().parents[1]
    for name in ("bootstrap.sh", "stage_game.py"):
        shutil.copyfile(original / "tools" / name, tools / name)
    for name in ("requirements-lock.txt", "requirements-cuda-lock.txt"):
        shutil.copyfile(original / name, project / name)
    scratch = tmp_path / "temporary sources"
    scratch.mkdir()
    log = tmp_path / "commands.jsonl"
    recorder = tmp_path / "record_commands.py"
    recorder.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path(os.environ['BOOTSTRAP_LOG']).open('a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if '--force-reinstall' in args:\n"
        "    source = Path(args[-1]) / 'src/pvz_game/__init__.py'\n"
        "    assert source.read_text() == 'PINNED = True\\n'\n"
        "fault = os.environ.get('BOOTSTRAP_FAIL')\n"
        "if (fault == 'install' and args[:3] == ['-m', 'pip', 'install']\n"
        "    or fault == 'game' and '--force-reinstall' in args\n"
        "    or fault == 'doctor' and args == ['-m', 'pvz_rl', 'doctor']):\n"
        "    sys.exit(17)\n"
    )
    python = tmp_path / "fake python"
    python.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'if [[ "$1" == -m ]]; then\n'
        '    "$BOOTSTRAP_REAL_PYTHON" "$BOOTSTRAP_RECORDER" "$@"\n'
        '    if [[ "$2" == venv ]]; then\n'
        '        mkdir -p -- "$3/bin"\n'
        '        cp -- "$BOOTSTRAP_FAKE_PYTHON" "$3/bin/python"\n'
        '        chmod +x "$3/bin/python"\n'
        "    fi\n"
        "    exit 0\n"
        "fi\n"
        'exec "$BOOTSTRAP_REAL_PYTHON" "$@"\n'
    )
    python.chmod(0o755)
    environment = {
        **os.environ,
        "BOOTSTRAP_REAL_PYTHON": shell_path(sys.executable),
        "BOOTSTRAP_RECORDER": shell_path(recorder),
        "BOOTSTRAP_FAKE_PYTHON": shell_path(python),
        "BOOTSTRAP_LOG": str(log),
        "TMPDIR": shell_path(scratch),
    }
    arguments = ["--game-repo", shell_path(game), "--python", shell_path(python)]

    def run(args=None):
        return subprocess.run(
            [str(bash), shell_path(tools / "bootstrap.sh"), *(arguments if args is None else args)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def git(*args):
        return subprocess.check_output([git_executable, "-C", str(game), *args])

    def commands():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return dict(
        project=project,
        scratch=scratch,
        python=python,
        environment=environment,
        run=run,
        git=git,
        commands=commands,
    )


@pytest.mark.parametrize("existing_venv", [None, "bin/python", "Scripts/python.exe"])
def test_bash_bootstrap_installs_pinned_game_and_preserves_checkout(bash_bootstrap, existing_venv):
    control = bash_bootstrap
    if existing_venv:
        python = control["project"] / ".venv" / existing_venv
        python.parent.mkdir(parents=True)
        shutil.copyfile(control["python"], python)
        python.chmod(0o755)
    before = control["git"]("rev-parse", "HEAD"), control["git"]("status", "--porcelain")
    result = control["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    commands = control["commands"]()
    created = [command for command in commands if command[:2] == ["-m", "venv"]]
    assert len(created) == int(existing_venv is None)
    installs = [command for command in commands if command[:3] == ["-m", "pip", "install"]]
    assert len(installs) == 5
    assert installs[0][3:] == [
        "--no-cache-dir",
        "torch==2.8.0+cu128",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ]
    assert installs[1][-2] == "-r" and Path(installs[1][-1]).name == "requirements-lock.txt"
    assert installs[2][3:5] == ["--no-deps", "--force-reinstall"]
    assert not Path(installs[2][-1]).exists()
    assert installs[3][3:5] == ["--no-deps", "-e"]
    assert installs[3][-1].endswith("[dev,ui]")
    assert Path(installs[4][-1]).name == "requirements-cuda-lock.txt"
    assert commands[-2:] == [["-m", "pip", "check"], ["-m", "pvz_rl", "doctor"]]
    assert (control["git"]("rev-parse", "HEAD"), control["git"]("status", "--porcelain")) == before
    assert list(control["scratch"].iterdir()) == []


@pytest.mark.parametrize("fault", ["missing_commit", "manifest"])
def test_bash_bootstrap_rejects_invalid_pin_before_installation(bash_bootstrap, fault):
    control = bash_bootstrap
    lock = control["project"] / "src/pvz_rl/data/engine-lock.json"
    data = json.loads(lock.read_text())
    if fault == "missing_commit":
        data["commit"] = "1" * 40
    else:
        data["files"]["__init__.py"] = "bad"
    lock.write_text(json.dumps(data))
    result = control["run"]()
    assert result.returncode != 0
    assert "pinned commit" in result.stderr or "manifest" in result.stderr
    assert control["commands"]() == []
    assert not (control["project"] / ".venv").exists()
    assert list(control["scratch"].iterdir()) == []


@pytest.mark.parametrize("fault", ["install", "game", "doctor"])
def test_bash_bootstrap_stops_and_cleans_temporary_sources_on_failure(bash_bootstrap, fault):
    control = bash_bootstrap
    control["environment"]["BOOTSTRAP_FAIL"] = fault
    result = control["run"]()
    assert result.returncode == 17, result.stdout + result.stderr
    commands = control["commands"]()
    if fault == "install":
        assert len(commands) == 2
    elif fault == "game":
        assert "--force-reinstall" in commands[-1]
    else:
        assert commands[-1] == ["-m", "pvz_rl", "doctor"]
    assert list(control["scratch"].iterdir()) == []


@pytest.mark.parametrize("args", [[], ["--game-repo"], ["--python"], ["--unknown"]])
def test_bash_bootstrap_argument_errors_do_not_install(bash_bootstrap, args):
    result = bash_bootstrap["run"](args)
    assert result.returncode != 0 and "Error:" in result.stderr
    assert bash_bootstrap["commands"]() == []
    assert list(bash_bootstrap["scratch"].iterdir()) == []


def test_bash_bootstrap_help_needs_no_repository_or_python(bash_bootstrap):
    result = bash_bootstrap["run"](["--help"])
    assert result.returncode == 0 and "Usage:" in result.stdout
    assert bash_bootstrap["commands"]() == []


def test_bash_bootstrap_rejects_unusable_existing_virtual_environment(bash_bootstrap):
    control = bash_bootstrap
    (control["project"] / ".venv").mkdir()
    result = control["run"]()
    assert result.returncode != 0 and "no usable Python" in result.stderr
    assert control["commands"]() == []
    assert list(control["scratch"].iterdir()) == []


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
        seen.append((cfg["training"]["n_envs"], cfg["training"]["method"], seed))
        return dict(state="complete", decisions_per_second=100, games_per_minute=2)

    monkeypatch.setattr(benchmark, "HardwareMonitor", Monitor)
    monkeypatch.setattr(benchmark, "measure", measure)
    result = benchmark.benchmark_gpu(smoke_cfg, tmp_path / "benchmark", minutes=1, steps=128)
    assert len(seen) == 3
    assert {(n, steps) for n, steps, _ in seen} == {(128, "complete_return_selective_invalid_v2")}
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
