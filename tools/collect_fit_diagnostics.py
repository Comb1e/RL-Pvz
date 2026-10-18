"""Collect fitting diagnostics while running the ordinary ``pvz-rl train`` command.

Example: python tools/collect_fit_diagnostics.py --output artifacts/fit-profile --
         train --output runs/easy --resume runs/easy/latest.zip --no-live-view
Add --trace-batches 6 for one bounded CPU/CUDA trace after 20 fitting batches.
"""

import argparse
import heapq
import json
import math
import platform
import subprocess
import sys
import threading
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import psutil
import torch

from pvz_rl.learning import sequential_training
from pvz_rl.learning.action_routing import workload_counts
from pvz_rl.learning.batch_transport import BatchPrefetch, FittingSpool
from pvz_rl.monitoring.hardware import HardwareMonitor
from pvz_rl.policy.cudagraph_backend import FittingGraphBackend, GraphCapture
from pvz_rl.policy.entity_attention import SpatialTransformer
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cgroup_directory():
    """Find this process's cgroup v2, including a container's namespaced root."""
    root = Path("/sys/fs/cgroup")
    for line in (read_text(Path("/proc/self/cgroup")) or "").splitlines():
        if line.startswith("0::"):
            candidate = root / line[3:].lstrip("/")
            if (candidate / "cpu.stat").is_file():
                return candidate
    return root if (root / "cpu.stat").is_file() else None


def counters(path):
    result = {}
    for line in (read_text(path) or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result


class ResourceDetails:
    """Extra host facts on HardwareMonitor's existing independent sampler thread."""

    def __init__(self):
        self.cgroup = cgroup_directory()
        self.last_time = None
        self.last_threads = {}
        self.last_cpu_stat = {}

    def __call__(self, process):
        now = perf_counter()
        elapsed = None if self.last_time is None else now - self.last_time
        threads = {row.id: row.user_time + row.system_time for row in process.threads()}
        names = {t.native_id: t.name for t in threading.enumerate()}
        busy = []
        if elapsed:
            for tid, seconds in threads.items():
                if tid in self.last_threads:
                    busy.append(
                        dict(
                            tid=tid,
                            name=names.get(tid),
                            cpu_percent=100 * max(0, seconds - self.last_threads[tid]) / elapsed,
                        )
                    )
        row = dict(
            thread_cpu_interval_seconds=elapsed,
            busiest_threads=sorted(busy, key=lambda r: r["cpu_percent"], reverse=True)[:8],
            process_threads=len(threads),
            available_ram_mib=psutil.virtual_memory().available / 2**20,
            swap_used_mib=psutil.swap_memory().used / 2**20,
            host_cpu_times=psutil.cpu_times()._asdict(),
        )
        try:
            row["process_io_counters"] = process.io_counters()._asdict()
        except (AttributeError, psutil.Error):
            row["process_io_counters"] = None
        stat = read_text(Path("/proc/self/stat"))
        if stat:
            fields = stat.rsplit(")", 1)[1].split()
            row["process_minor_faults"], row["process_major_faults"] = (
                int(fields[7]),
                int(fields[9]),
            )
        if self.cgroup:
            current = counters(self.cgroup / "cpu.stat")
            row["cgroup_cpu_stat"] = current
            row["cgroup_cpu_delta"] = {
                key: value - self.last_cpu_stat[key]
                for key, value in current.items()
                if key in self.last_cpu_stat
            }
            row["cgroup_memory_events"] = counters(self.cgroup / "memory.events")
            self.last_cpu_stat = current
        for kind in ("cpu", "io", "memory"):
            row["host_" + kind + "_pressure"] = read_text(Path("/proc/pressure") / kind)
        self.last_time, self.last_threads = now, threads
        return row


class FitDiagnostics:
    """Temporary timing wrappers; the original learner owns every training operation.

    Host spans are inclusive submission/wait times, not CUDA kernel durations.
    Overlapping/nested spans must not be added together. No CUDA events, tensor
    readbacks or synchronization are added by the lightweight collector.
    """

    def __init__(self, output, *, sample_seconds=1.0, trace_batches=0, trace_skip_batches=20):
        if not math.isfinite(sample_seconds) or sample_seconds < 0.1:
            raise ValueError("sample_seconds must be finite and at least 0.1")
        if any(type(n) is not int or n < 0 for n in (trace_batches, trace_skip_batches)):
            raise ValueError("Trace batch counts must be nonnegative integers")
        self.output = Path(output)
        self.sample_seconds = sample_seconds
        self.trace_batches, self.trace_skip_batches = trace_batches, trace_skip_batches
        self.resources = ResourceDetails()
        self.local = threading.local()
        self.main_thread = threading.get_ident()
        self.lock = threading.Lock()
        self.hardware = self.profiler = self.stream = None
        self.trace_started = self.trace_finished = False
        self.trace_steps = self.batches = self.chunks = 0
        self.fit_wall_seconds = 0.0
        self.spans = defaultdict(lambda: dict(calls=0, host_seconds=0.0, max_host_seconds=0.0))
        self.slowest = []
        self.errors = []
        self.last_summary = 0.0

    def write_json(self, filename, value):
        (self.output / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def event(self, kind, **data):
        with self.lock:
            if self.stream.closed:
                return
            self.stream.write(json.dumps(dict(kind=kind, utc=utc_now(), **data)) + "\n")
            self.stream.flush()

    def summary(self):
        self.write_json(
            "summary.json",
            dict(
                utc=utc_now(),
                fit_chunks=self.chunks,
                fit_batches=self.batches,
                fit_host_wall_seconds=self.fit_wall_seconds,
                inclusive_host_spans=dict(self.spans),
                slowest_chunks=[entry[2] for entry in sorted(self.slowest, reverse=True)],
                trace_requested_batches=self.trace_batches,
                trace_steps_including_one_warmup=self.trace_steps,
                trace_finished=self.trace_finished,
                diagnostic_errors=self.errors,
                timing_note="Nested host spans overlap. CUDA submission time is not GPU busy time. "
                "Existing whole-pass CUDA-event timing also includes idle gaps on that stream.",
            ),
        )
        self.last_summary = perf_counter()

    @contextmanager
    def span(self, name):
        row = getattr(self.local, "row", None)
        if row is None:
            yield
            return
        previous = getattr(self.local, "operation", "fit_chunk")
        self.local.operation = name
        if self.hardware:
            self.hardware.update(operation=name)
        started = perf_counter()
        scope = torch.profiler.record_function("fit/" + name) if self.profiler else nullcontext()
        try:
            with scope:
                yield
        finally:
            seconds = perf_counter() - started
            for destination in (row["host_spans"], self.spans):
                stat = destination.setdefault(
                    name, dict(calls=0, host_seconds=0.0, max_host_seconds=0.0)
                )
                stat["calls"] += 1
                stat["host_seconds"] += seconds
                stat["max_host_seconds"] = max(stat["max_host_seconds"], seconds)
            self.local.operation = previous
            if self.hardware:
                self.hardware.update(operation=previous)

    def wrapped(self, function, name):
        @wraps(function)
        def run(*args, **kwargs):
            with self.span(name):
                return function(*args, **kwargs)

        return run

    def initialize(self, model):
        self.hardware = HardwareMonitor(
            self.output / "hardware-metrics.jsonl",
            seconds=self.sample_seconds,
            device=model.device,
            sample_details=self.resources,
        ).__enter__()
        self.hardware.update(phase="diagnostic_startup", activity="diagnostics")
        process = psutil.Process()
        cfg = model.cfg
        metadata = dict(
            utc=utc_now(),
            python=sys.version,
            platform=platform.platform(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            torch_threads=torch.get_num_threads(),
            cpu_logical=psutil.cpu_count(),
            cpu_physical=psutil.cpu_count(logical=False),
            gpu_id=self.hardware.gpu_id,
            gpu_name=torch.cuda.get_device_name(model.device),
            policy=cfg["policy"],
            performance=cfg["training"]["performance"],
            storage=cfg["training"]["storage"],
            batch_size=model.batch_size,
            n_epochs=model.n_epochs,
            encoding=cfg["encoding"],
            diagnostics=dict(
                sample_seconds=self.sample_seconds,
                trace_batches=self.trace_batches,
                trace_skip_batches=self.trace_skip_batches,
            ),
        )
        if hasattr(process, "cpu_affinity"):
            try:
                metadata["cpu_affinity"] = process.cpu_affinity()
            except psutil.Error:
                pass
        if self.resources.cgroup:
            metadata["cgroup"] = {
                name: read_text(self.resources.cgroup / name)
                for name in ("cpu.max", "cpuset.cpus.effective", "memory.max")
            }
        for filename, query in (
            ("gpu-status.xml", ["-q", "-x"]),
            (
                "gpu-processes.csv",
                ["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
            ),
        ):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--id=" + self.hardware.gpu_id, *query],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                (self.output / filename).write_text(result.stdout, encoding="utf-8")
                if result.returncode:
                    self.errors.append(result.stderr.strip())
            except (OSError, subprocess.SubprocessError) as exc:
                self.errors.append(repr(exc))
        self.write_json("metadata.json", metadata)

    def trace_ready(self, profiler):
        if self.hardware:
            self.hardware.update(operation="trace_export")
        try:
            profiler.export_chrome_trace(str(self.output / "fit-trace.json.gz"))
            averages = profiler.key_averages()
            for device in ("cpu", "device"):
                (self.output / f"operators-{device}.txt").write_text(
                    averages.table(sort_by=f"self_{device}_time_total", row_limit=60),
                    encoding="utf-8",
                )
            self.event("trace_export", steps=self.trace_steps)
        except Exception as exc:
            self.errors.append("trace export: " + repr(exc))

    def start_trace(self):
        if not self.trace_batches or self.trace_started or self.batches < self.trace_skip_batches:
            return
        self.trace_started = True
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            else:
                self.errors.append("CUDA profiler unavailable; trace contains CPU activities only")
            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0, warmup=1, active=self.trace_batches, repeat=1
                ),
                on_trace_ready=self.trace_ready,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.__enter__()
        except Exception as exc:
            self.errors.append("trace start: " + repr(exc))
            self.profiler = None

    def stop_trace(self):
        profiler, self.profiler = self.profiler, None
        if profiler is not None:
            try:
                profiler.__exit__(None, None, None)
            except Exception as exc:
                self.errors.append("trace stop: " + repr(exc))

    def fit_chunk(self, original, model, *args, **kwargs):
        if self.hardware is None:
            self.initialize(model)
        self.start_trace()
        row = dict(
            chunk=self.chunks + 1,
            cohort=getattr(model, "cohort_number", None),
            pass_index=model._fit_epoch,
            trace_enabled=self.profiler is not None,
            host_spans={},
            encoder_calls={},
            microbatch_paths={},
            execution=deepcopy(model.execution_state),
        )
        before = dict(model._buffer.transport_metrics)
        self.local.row = row
        self.hardware.update(
            phase="fit", activity="training", chunk=row["chunk"], pass_index=row["pass_index"]
        )
        started = perf_counter()
        try:
            with self.span("fit_chunk"):
                return original(model, *args, **kwargs)
        except BaseException as exc:
            row["error"] = repr(exc)
            raise
        finally:
            row["host_seconds"] = perf_counter() - started
            self.local.row = None
            self.hardware.update(operation="diagnostic_reporting")
            self.chunks += 1
            self.fit_wall_seconds += row["host_seconds"]
            row["transport_delta"] = {
                key: value - before.get(key, 0)
                for key, value in model._buffer.transport_metrics.items()
            }
            row["pass_index_after"] = model._fit_epoch
            row["prefetch_active"] = isinstance(model._batch_iterator, BatchPrefetch)
            row["encoder_state"] = {
                name + "." + encoder.stage: dict(
                    status=encoder.compilation_status,
                    error=encoder.compilation_error,
                    compiled_shapes=list(encoder._compiled_shapes),
                    collection_graphs=encoder.collection_metrics,
                    microbatch=encoder.microbatch,
                    token_budget=encoder.token_budget,
                    checkpointing=encoder.activation_checkpointing,
                )
                for name, network in (("online", model.policy), ("ema", model._teacher))
                if network is not None
                for encoder in network.encoders
            }
            row["torch_allocated_mib"] = torch.cuda.memory_allocated(model.device) / 2**20
            row["torch_reserved_mib"] = torch.cuda.memory_reserved(model.device) / 2**20
            spool = getattr(model, "_fitting_spool", None)
            row["storage"] = {
                name + "_mib": (
                    None if (value := getattr(model._buffer, name, None)) is None else value / 2**20
                )
                for name in ("ram_used", "ram_bytes", "staging_bytes")
            }
            row["storage"].update(
                spool_rows_on_disk=isinstance(getattr(spool, "rows", None), np.memmap),
                spool_entities_on_disk=isinstance(getattr(spool, "entities", None), np.memmap),
            )
            self.event("fit_chunk", **row)
            heapq.heappush(self.slowest, (row["host_seconds"], row["chunk"], row))
            if len(self.slowest) > 12:
                heapq.heappop(self.slowest)
            if "batch" in row:
                self.batches += 1
                if self.profiler:
                    self.trace_steps += 1
                    try:
                        self.profiler.step()
                    except Exception as exc:
                        self.errors.append("trace step: " + repr(exc))
                        self.stop_trace()
                    if self.trace_steps >= self.trace_batches + 1:
                        self.trace_finished = True
                        self.stop_trace()
            if (
                perf_counter() - self.last_summary >= 15
                or row["pass_index_after"] != row["pass_index"]
            ):
                self.summary()
            self.hardware.update(operation="between_chunks")

    def phase_change(self, original, model, phase, *args, **kwargs):
        result = original(model, phase, *args, **kwargs)
        if self.hardware:
            self.hardware.update(
                phase=getattr(phase, "value", str(phase)), operation="phase_transition"
            )
        return result

    def batch_loss(self, original, policy, rows, counts, observations, *args, **kwargs):
        row = getattr(self.local, "row", None)
        if row is not None:
            active = rows["active"]
            actions = rows["action"][active]
            widths = rows["entity_count"][active]
            row["batch"] = dict(
                rows=len(rows),
                active_rows=int(active.sum()),
                nonwait_rows=int(np.count_nonzero(actions)),
                routed_workload=workload_counts(rows[active]),
                entity_width=int(observations.entities.shape[-2]),
                entity_count_min=int(widths.min()) if len(widths) else None,
                entity_count_max=int(widths.max()) if len(widths) else None,
                entity_count_mean=float(widths.mean()) if len(widths) else None,
            )
        with self.span("loss_forward"):
            return original(policy, rows, counts, observations, *args, **kwargs)

    def encoder_forward(self, original, encoder, observations, *args, **kwargs):
        row = getattr(self.local, "row", None)
        if row is not None:
            network = "online" if torch.is_grad_enabled() else "ema"
            key = f"{network}.{encoder.stage}:{len(observations)}x{observations.entities.shape[-2]}"
            row["encoder_calls"][key] = row["encoder_calls"].get(key, 0) + 1
        network = "online" if torch.is_grad_enabled() else "ema"
        with self.span(network + "." + encoder.stage + "_encoder"):
            return original(encoder, observations, *args, **kwargs)

    def graph_capture(self, original, capture, inputs):
        if capture.graph is None:
            with self.span("graph_setup"):
                return original(capture, inputs)
        return original(capture, inputs)

    def encoder_execution(self, original, encoder, inputs):
        row = getattr(self.local, "row", None)
        if row is not None:
            shape = tuple(inputs[0].shape[:2])
            if encoder._compiled_encode is None:
                path = "disabled_eager"
            elif not torch.is_grad_enabled():
                path = "collection_graph_attempt"
            elif shape in encoder._compiled_shapes:
                path = "fitting_compiled"
            elif len(encoder._compiled_shapes) >= 4:
                path = "extra_shape_eager"
            else:
                path = "new_shape_compile_attempt"
            key = f"{encoder.stage}:{path}:{shape[0]}x{shape[1]}"
            row["microbatch_paths"][key] = row["microbatch_paths"].get(key, 0) + 1
        return original(encoder, inputs)

    def prepare(self, original, prefetch, *args, **kwargs):
        started = perf_counter()
        before = prefetch.buffer.transport_metrics["preparation_seconds"]
        try:
            return original(prefetch, *args, **kwargs)
        finally:
            self.event(
                "prepare",
                host_seconds=perf_counter() - started,
                preparation_seconds=prefetch.buffer.transport_metrics["preparation_seconds"]
                - before,
            )

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / "fit-events.jsonl").open("w", encoding="utf-8")
        self.patches = ExitStack()
        for owner, method, name in (
            (sequential_training.CudaSequentialQ, "_start_pass", "start_pass"),
            (sequential_training.CudaSequentialQ, "_publish_fit_metrics", "pass_metrics"),
            (sequential_training.CudaSequentialQ, "_close_batches", "close_batches"),
            (sequential_training.CudaSequentialQ, "_update_teacher", "ema_update"),
            (FittingSpool, "__init__", "spool_build"),
            (BatchPrefetch, "__init__", "prefetch_start"),
            (BatchPrefetch, "__next__", "next_batch"),
            (FittingGraphBackend, "__call__", "compile_backend"),
            (torch.Tensor, "backward", "backward"),
            (TwoStageTransformerPolicy, "accumulate_master_gradients", "master_gradients"),
            (TwoStageTransformerPolicy, "sync_runtime_from_masters", "runtime_weights"),
        ):
            self.patches.enter_context(
                patch.object(owner, method, self.wrapped(getattr(owner, method), name))
            )
        for owner, method, handler in (
            (sequential_training.CudaSequentialQ, "_fit_chunk", self.fit_chunk),
            (sequential_training.CudaSequentialQ, "_phase", self.phase_change),
            (sequential_training, "batch_loss", self.batch_loss),
            (SpatialTransformer, "forward", self.encoder_forward),
            (SpatialTransformer, "_encode_with_optional_compile", self.encoder_execution),
            (GraphCapture, "__call__", self.graph_capture),
            (BatchPrefetch, "_prepare", self.prepare),
        ):
            original = getattr(owner, method)

            def run(*args, _original=original, _handler=handler, **kwargs):
                return _handler(_original, *args, **kwargs)

            self.patches.enter_context(patch.object(owner, method, run))
        self.summary()
        return self

    def __exit__(self, *exc):
        self.stop_trace()
        self.patches.close()
        if self.hardware:
            self.hardware.close()
        self.summary()
        with self.lock:
            self.stream.close()


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="unused diagnostic directory")
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--trace-batches", type=int, default=0, help="0 disables the bounded trace")
    parser.add_argument("--trace-skip-batches", type=int, default=20)
    if "--" not in arguments:
        parser.parse_args(arguments)  # Preserve --help without starting training.
        parser.error("append -- followed by your ordinary pvz-rl train arguments")
    separator = arguments.index("--")
    options = parser.parse_args(arguments[:separator])
    training_args = arguments[separator + 1 :]
    if not training_args or training_args[0] != "train":
        parser.error("only the existing train command is supported after --")
    try:
        diagnostics = FitDiagnostics(**vars(options))
    except ValueError as exc:
        parser.error(str(exc))
    from pvz_rl.cli import main as training_main

    with diagnostics:
        print(f"Fitting diagnostics: {diagnostics.output.resolve()}", flush=True)
        training_main(training_args)


if __name__ == "__main__":
    main()
