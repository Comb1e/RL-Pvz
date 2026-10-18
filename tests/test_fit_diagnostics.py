"""Collector transparency, bounded traces and independently checked resource deltas."""

import gzip
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pvz_rl.envs.encoding import EntityBatch
from pvz_rl.learning import sequential_training
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy


@pytest.fixture(scope="module")
def collector_module():
    path = Path(__file__).resolve().parents[1] / "tools/collect_fit_diagnostics.py"
    spec = importlib.util.spec_from_file_location("fit_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def diagnostics(collector_module, tmp_path, monkeypatch):
    collector = collector_module.FitDiagnostics(tmp_path / "diagnostics")
    monitor = SimpleNamespace(update=lambda **kwargs: None, close=lambda: None)
    monkeypatch.setattr(
        collector, "initialize", lambda model: setattr(collector, "hardware", monitor)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)

    def unexpected_wait(*args, **kwargs):
        pytest.fail("Lightweight diagnostics must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_wait)
    return collector


def learner(policy=None, teacher=None):
    return SimpleNamespace(
        policy=policy or SimpleNamespace(encoders=[]),
        _teacher=teacher,
        _buffer=SimpleNamespace(transport_metrics={"preparation_seconds": 0.0}),
        _batch_iterator=None,
        _fit_epoch=0,
        execution_state={"microbatch": 192},
        cohort_number=1,
        device=torch.device("cpu"),
    )


def events(collector):
    return [
        json.loads(line)
        for line in (collector.output / "fit-events.jsonl").read_text().splitlines()
    ]


def tiny_fit_case(cfg, device="cpu"):
    cfg = deepcopy(cfg)
    for stage in ("species", "tile"):
        cfg["policy"].update(
            {
                f"{stage}_width": 16,
                f"{stage}_layers": 1,
                f"{stage}_heads": 1,
                f"{stage}_feedforward": 32,
            }
        )
    policy = TwoStageTransformerPolicy(cfg).to(device)
    teacher = TwoStageTransformerPolicy(cfg).to(device).requires_grad_(False)
    teacher.load_state_dict(policy.runtime_state_dict())
    instrumented = TwoStageTransformerPolicy(cfg).to(device)
    instrumented.load_state_dict(policy.runtime_state_dict())
    rows = np.zeros(
        3,
        dtype=[
            ("active", "?"),
            ("action", "i8"),
            ("accepted", "?"),
            ("target", "f4"),
            ("route", "u1"),
            ("branch_usable", "?"),
            ("entity_count", "i4"),
        ],
    )
    rows["active"] = rows["accepted"] = True
    rows["action"], rows["target"] = [0, 1, 361], [1, -2, 3]
    from pvz_rl.learning.action_routing import route_outcomes

    rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    observations = EntityBatch(
        torch.zeros(3, 0, 11, dtype=torch.int32),
        torch.zeros(3, 0, dtype=torch.bool),
        torch.zeros(3, 18),
        True,
    ).to(device)

    def chunk(model):
        loss, _, _ = sequential_training.batch_loss(
            model.policy, rows, [[0, 1], [0, 1], [0, 1]], observations, teacher=model._teacher
        )
        loss.backward()
        model.policy.accumulate_master_gradients()
        model.policy.clear_runtime_grad()
        return loss.detach()

    return policy, teacher, instrumented, chunk


def test_real_loss_gradients_and_hooks_are_preserved(diagnostics, cfg):
    policy, teacher, instrumented, chunk = tiny_fit_case(cfg)
    expected = chunk(learner(policy, teacher))
    original = sequential_training.CudaSequentialQ._fit_chunk
    backward = torch.Tensor.backward
    with diagnostics:
        actual = diagnostics.fit_chunk(chunk, learner(instrumented, teacher))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for reference, measured in zip(policy.master_parameters(), instrumented.master_parameters()):
        torch.testing.assert_close(measured.grad, reference.grad, rtol=0, atol=0)
    assert sequential_training.CudaSequentialQ._fit_chunk is original
    assert torch.Tensor.backward is backward
    row = events(diagnostics)[-1]
    assert row["batch"]["active_rows"] == 3 and row["batch"]["nonwait_rows"] == 2
    assert row["host_spans"]["backward"]["calls"] == 1
    assert set(row["encoder_calls"]) == {"online.species:3x0", "online.tile:2x0", "ema.tile:2x0"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiled diagnostic control")
def test_cuda_compiled_fit_trace_preserves_eager_results(collector_module, cfg, tmp_path):
    policy, teacher, instrumented, chunk = tiny_fit_case(cfg, "cuda")
    expected = chunk(learner(policy, teacher))
    for network in (instrumented, teacher):
        network.enable_compilation()
        for encoder in network.encoders:
            encoder.microbatch = 16
    model = learner(instrumented, teacher)
    model.device, model.cfg = instrumented.device, instrumented.cfg
    model.batch_size, model.n_epochs = 1024, 4
    collector = collector_module.FitDiagnostics(
        tmp_path / "cuda-diagnostics", trace_batches=1, trace_skip_batches=0, sample_seconds=0.1
    )
    try:
        with collector:
            collector.fit_chunk(chunk, model)  # Warmup includes lazy compilation.
            instrumented.zero_grad()
            actual = collector.fit_chunk(chunk, model)  # Record compiled forward/backward.
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
        for reference, measured in zip(
            policy.master_parameters(), instrumented.master_parameters()
        ):
            torch.testing.assert_close(measured.grad, reference.grad, rtol=0.03, atol=0.01)
        assert collector.trace_finished and not collector.errors
        assert (collector.output / "metadata.json").is_file()
        with gzip.open(collector.output / "fit-trace.json.gz", "rt") as stream:
            trace = json.load(stream)
        assert any(row.get("cat") == "kernel" for row in trace["traceEvents"])
        row = [row for row in events(collector) if row["kind"] == "fit_chunk"][-1]
        assert row["microbatch_paths"]["species:fitting_compiled:16x0"] == 1
        assert row["microbatch_paths"]["tile:fitting_compiled:16x0"] == 1
    finally:
        for network in (instrumented, teacher):
            for encoder in network.encoders:
                encoder.release_fitting()
                encoder.release_collection()


def test_original_failure_survives_and_hooks_close(diagnostics):
    original = torch.Tensor.backward
    failure = RuntimeError("original fitting failure")

    def fail(model):
        raise failure

    with pytest.raises(RuntimeError) as caught, diagnostics:
        diagnostics.fit_chunk(fail, learner())
    assert caught.value is failure and torch.Tensor.backward is original
    assert "original fitting failure" in events(diagnostics)[-1]["error"]
    assert json.loads((diagnostics.output / "summary.json").read_text())["fit_batches"] == 0


def test_hardware_phase_follows_return_to_collection(diagnostics):
    context = {}
    diagnostics.hardware = SimpleNamespace(
        update=lambda **kwargs: context.update(kwargs), close=lambda: None
    )
    sentinel = object()
    phase = SimpleNamespace(value="collect")
    with diagnostics:
        actual = diagnostics.phase_change(lambda *args: sentinel, learner(), phase)
    assert actual is sentinel and context["phase"] == "collect"


def test_real_cpu_trace_has_exact_bounded_batch_window(diagnostics, monkeypatch):
    diagnostics.trace_batches, diagnostics.trace_skip_batches = 2, 1
    monkeypatch.setattr(
        torch.profiler, "supported_activities", lambda: {torch.profiler.ProfilerActivity.CPU}
    )

    def chunk(model):
        diagnostics.local.row["batch"] = {"active_rows": 1}
        return torch.ones(4).sum()

    with diagnostics:
        for _ in range(5):
            diagnostics.fit_chunk(chunk, learner())
    rows = [row for row in events(diagnostics) if row["kind"] == "fit_chunk"]
    assert [row["trace_enabled"] for row in rows] == [False, True, True, True, False]
    assert diagnostics.trace_steps == 3 and diagnostics.trace_finished
    with gzip.open(diagnostics.output / "fit-trace.json.gz", "rt") as stream:
        trace = json.load(stream)
    assert sum(row.get("name") == "fit/fit_chunk" for row in trace["traceEvents"]) == 2
    assert (diagnostics.output / "operators-cpu.txt").is_file()
    assert not any("trace export:" in error for error in diagnostics.errors)


def test_trace_export_failure_is_diagnostic_only(diagnostics):
    def fail(*args):
        raise RuntimeError("disk full")

    with diagnostics:
        diagnostics.trace_ready(SimpleNamespace(export_chrome_trace=fail))
    assert diagnostics.errors == ["trace export: RuntimeError('disk full')"]


def test_partial_trace_and_empty_session_are_reported(diagnostics, monkeypatch):
    diagnostics.trace_batches, diagnostics.trace_skip_batches = 3, 0
    monkeypatch.setattr(
        torch.profiler, "supported_activities", lambda: {torch.profiler.ProfilerActivity.CPU}
    )
    with diagnostics:
        diagnostics.fit_chunk(lambda model: torch.ones(1), learner())
    assert not diagnostics.trace_finished
    summary = json.loads((diagnostics.output / "summary.json").read_text())
    assert summary["fit_batches"] == 0 and summary["trace_steps_including_one_warmup"] == 0
    assert diagnostics.profiler is None


def test_existing_outputs_are_never_overwritten(collector_module, tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    saved = path / "summary.json"
    saved.write_text("preserved")
    with pytest.raises(FileExistsError), collector_module.FitDiagnostics(path):
        pytest.fail("Should not enter an existing directory")
    assert saved.read_text() == "preserved"


@pytest.mark.parametrize(
    "options",
    [
        {"trace_batches": -1},
        {"trace_skip_batches": True},
        {"sample_seconds": float("nan")},
        {"sample_seconds": 0.09},
    ],
)
def test_invalid_options_rejected_before_output_creation(collector_module, tmp_path, options):
    path = tmp_path / "bad"
    with pytest.raises(ValueError):
        collector_module.FitDiagnostics(path, **options)
    assert not path.exists()


@pytest.mark.parametrize("suffix", [[], ["--"], ["--", "evaluate"]])
def test_cli_requires_explicit_training_arguments(collector_module, tmp_path, suffix):
    path = tmp_path / "bad"
    with pytest.raises(SystemExit) as caught:
        collector_module.main(["--output", str(path), *suffix])
    assert caught.value.code == 2 and not path.exists()


def test_resource_rates_use_deltas_and_container_counters(collector_module, tmp_path, monkeypatch):
    sample = collector_module.ResourceDetails()
    sample.cgroup = tmp_path
    sample.last_time = 10
    sample.last_threads = {7: 3.0}
    sample.last_cpu_stat = {"nr_throttled": 10, "throttled_usec": 1_000_000}
    (tmp_path / "cpu.stat").write_text("nr_throttled 12\nthrottled_usec 1600000\n")
    monkeypatch.setattr(collector_module, "perf_counter", lambda: 12)
    process = SimpleNamespace(
        threads=lambda: [SimpleNamespace(id=7, user_time=4, system_time=0.5)],
        io_counters=lambda: SimpleNamespace(_asdict=lambda: {"read_bytes": 8192}),
    )
    row = sample(process)
    assert row["busiest_threads"][0]["cpu_percent"] == 75
    assert row["cgroup_cpu_delta"] == {"nr_throttled": 2, "throttled_usec": 600_000}
    assert row["process_io_counters"]["read_bytes"] == 8192
