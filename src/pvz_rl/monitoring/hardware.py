"""Independent, bounded hardware sampling shared by training and benchmarks."""

import json
import math
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import psutil

GPU_FIELDS = {
    "gpu_percent": "utilization.gpu",
    "gpu_memory_percent": "utilization.memory",
    "gpu_memory_mib": "memory.used",
    "gpu_watts": "power.draw",
    "gpu_temperature_c": "temperature.gpu",
}


def training_gpu_id(device=None):
    """Use CUDA's UUID, not nvidia-smi's possibly different device ordering."""
    import torch

    if not torch.cuda.is_available():
        return None
    identifier = str(torch.cuda.get_device_properties(device or torch.cuda.current_device()).uuid)
    return identifier if identifier.startswith(("GPU-", "MIG-")) else "GPU-" + identifier


def gpu_sample(identifier, timeout):
    row = dict.fromkeys(GPU_FIELDS)
    if identifier is None:
        return {**row, "gpu_error": "Training CUDA device unavailable"}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=" + identifier,
                "--query-gpu=" + ",".join(GPU_FIELDS.values()),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"nvidia-smi exit {result.returncode}")
        values = result.stdout.strip().split(",")
        if len(values) != len(GPU_FIELDS):
            raise ValueError("Unexpected nvidia-smi field count")
        missing = []
        for key, value in zip(GPU_FIELDS, values):
            try:
                number = float(value.strip())
                if not math.isfinite(number):
                    raise ValueError("nonfinite")
                row[key] = number
            except ValueError:
                missing.append(key)
        if missing:
            row["gpu_error"] = "Unavailable fields: " + ", ".join(missing)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        row["gpu_error"] = str(exc)
    return row


class HardwareMonitor:
    """Append/flush raw samples; keep only ten minutes of recent data in RAM.

    System/per-core CPU are 0..100%; process CPU uses psutil's one-core units
    and can exceed 100%. GPU fields are device-wide, not process attribution.
    No CUDA work is issued by the sampler.
    """

    def __init__(self, path, *, seconds=1.0, device=None, gpu_id=None):
        if not math.isfinite(seconds) or seconds < 0.1:
            raise ValueError("Hardware sampling interval must be at least 0.1 seconds")
        self.path, self.interval = Path(path), seconds
        self.gpu_id = training_gpu_id(device) if gpu_id is None else gpu_id
        self.session = uuid.uuid4().hex
        self.samples = deque(maxlen=max(1, math.ceil(600 / seconds)))
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.context = {"stage": "starting", "phase": "starting", "activity": "training"}
        self.thread = threading.Thread(target=self._run, name="hardware-sampler", daemon=True)
        self.error = None

    def update(self, **context):
        with self.lock:
            self.context.update(context)

    def record_window(self, metrics):
        self.update(
            transitions_per_second=metrics.get("transitions_per_second"),
            collection_seconds=sum(metrics["slot_collection_seconds"]),
            update_seconds=sum(metrics["slot_optimization_seconds"]),
            overlap_seconds=metrics.get("overlap_seconds"),
            window_seconds=metrics.get("window_seconds"),
        )

    @property
    def phase(self):
        with self.lock:
            return self.context["phase"]

    @phase.setter
    def phase(self, value):
        self.update(phase=value)

    @property
    def latest(self):
        with self.lock:
            row = dict(self.samples[-1]) if self.samples else {}
        if row:
            row["age_seconds"] = perf_counter() - self.started - row["seconds"]
        if self.error:
            row["sampler_error"] = self.error
        return row

    def _run(self):
        try:
            process = psutil.Process()
            # Prime nonblocking interval measurements in the sampling thread.
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
            process.cpu_percent(interval=None)
            with self.path.open("a", encoding="utf-8") as stream:
                delay = self.interval
                while not self.stop.wait(delay):
                    started = perf_counter()
                    with self.lock:
                        row = dict(self.context)
                    row.update(
                        utc=datetime.now(timezone.utc).isoformat(),
                        seconds=started - self.started,
                        session=self.session,
                        gpu_id=self.gpu_id,
                        system_cpu_percent=None,
                        busiest_cpu_percent=None,
                        process_cpu_percent=None,
                        process_ram_mib=None,
                    )
                    try:
                        row.update(
                            system_cpu_percent=psutil.cpu_percent(interval=None),
                            busiest_cpu_percent=max(psutil.cpu_percent(interval=None, percpu=True)),
                            process_cpu_percent=process.cpu_percent(interval=None),
                            process_ram_mib=process.memory_info().rss / 2**20,
                        )
                    except (OSError, psutil.Error) as exc:
                        row["cpu_error"] = str(exc)
                    row.update(gpu_sample(self.gpu_id, min(4.0, self.interval)))
                    row["sampling_seconds"] = perf_counter() - started
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    with self.lock:
                        self.samples.append(row)
                    delay = max(0.0, self.interval - (perf_counter() - started))
        except Exception as exc:
            self.error = repr(exc)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = perf_counter()
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=min(4.0, self.interval) + 1)

    def __exit__(self, *_):
        self.close()
