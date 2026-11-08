"""Diagnostic-only exclusive host attribution and bounded CUPTI kernel windows."""

import cProfile
import gc
import heapq
import json
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from time import perf_counter


def union_seconds(intervals):
    end, total = float("-inf"), 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def host_attribution(events, *, start=None, end=None):
    """Self = span minus the union of contained spans on the same thread.

    Duplicate equal-boundary wrappers are ordered stably. Crossing spans give
    overlapping time to the most recently started active span, never twice.
    """
    threads = defaultdict(list)
    for event in events:
        lower, upper = event["start"], event["start"] + event["seconds"]
        lower = max(lower, start) if start is not None else lower
        upper = min(upper, end) if end is not None else upper
        if upper > lower:
            threads[(event.get("thread", "main"), event.get("thread_id"))].append(
                (lower, upper, event["name"])
            )
    totals = defaultdict(lambda: dict(calls=0, inclusive_seconds=0.0, self_seconds=0.0))
    covered = {}
    for thread, spans in threads.items():
        boundaries, active, priority = [], set(), []
        for identity, (lower, upper, name) in enumerate(spans):
            own = totals[name]
            own["calls"] += 1
            own["inclusive_seconds"] += upper - lower
            boundaries.extend(((lower, 1, identity), (upper, 0, identity)))
        previous, busy = None, 0.0
        for point, entering, identity in sorted(boundaries):
            while priority and priority[0][3] not in active:
                heapq.heappop(priority)
            if priority and previous is not None:
                elapsed = point - previous
                totals[priority[0][4]]["self_seconds"] += elapsed
                busy += elapsed
            if entering:
                lower, upper, name = spans[identity]
                active.add(identity)
                heapq.heappush(priority, (-lower, upper, -identity, identity, name))
            else:
                active.remove(identity)
            previous = point
        covered[thread[0]] = covered.get(thread[0], 0) + busy
    return dict(operations=dict(totals), covered_seconds_by_thread=covered)


def kernel_summary(trace, window_name="pvz_kernel_window"):
    spans = trace["traceEvents"]
    markers = [e for e in spans if e.get("name") == window_name and e.get("ph") == "X"]
    if not markers:
        raise ValueError("Missing CUPTI window marker")
    marker = markers[0]
    lower, upper = marker["ts"], marker["ts"] + marker["dur"]
    kernels, transfers, synchronizations, allocations = [], [], [], []
    for event in spans:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        a, b = max(lower, event["ts"]), min(upper, event["ts"] + event["dur"])
        if b <= a:
            continue
        category, name = event.get("cat", ""), event.get("name", "")
        if category == "kernel":
            kernels.append((a, b, name))
        if "memcpy" in category.lower():
            transfers.append(event)
        if category in ("cuda_runtime", "cuda_driver"):
            if "Synchronize" in name:
                synchronizations.append(event)
            if any(key in name for key in ("Malloc", "Free", "Alloc")):
                allocations.append(event)
    wall = marker["dur"] / 1e6
    summed = sum(b - a for a, b, _ in kernels) / 1e6
    busy = union_seconds((a / 1e6, b / 1e6) for a, b, _ in kernels)
    groups = defaultdict(lambda: dict(calls=0, seconds=0.0))
    for a, b, name in kernels:
        groups[name]["calls"] += 1
        groups[name]["seconds"] += (b - a) / 1e6
    return dict(
        wall_seconds=wall,
        kernel_count=len(kernels),
        kernel_sum_seconds=summed,
        kernel_busy_seconds=busy,
        kernel_sum_ratio=summed / wall,
        kernel_busy_ratio=busy / wall,
        transfers=transfers,
        synchronizations=synchronizations,
        allocations=allocations,
        kernels=dict(sorted(groups.items(), key=lambda item: -item[1]["seconds"])),
        note="CUPTI kernel intervals, not CUDA event spans; overlapping kernels are unioned for busy ratio. Profiling overhead is included.",
    )


class KernelWindows:
    """Bounded sequential windows; no model/stream references survive restore."""

    def __init__(self, path, calls=20, *, origin=None):
        self.path, self.calls = Path(path), calls
        self.originals, self.results = [], {}
        self.active = None
        self.origin = perf_counter() if origin is None else origin

    def wrap(self, owner, name, phase, *, skip=0, calls=None):
        original = getattr(owner, name)
        owned = name in vars(owner)
        self.originals.append((owner, name, original, owned))
        seen, completed = 0, 0

        @wraps(original)
        def measured(*args, **kwargs):
            nonlocal seen, completed
            seen += 1
            if phase not in self.results and seen > skip and self.active in (None, phase):
                if self.active is None:
                    self._start(phase)
                completed += 1
            try:
                return original(*args, **kwargs)
            finally:
                if self.active == phase and completed >= (calls or self.calls):
                    self._stop(completed)

        setattr(owner, name, measured)

    def _start(self, phase):
        import torch

        self.active = phase
        self.window_started = perf_counter()
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        self.profiler.start()
        self.python = cProfile.Profile()
        self.python.enable()
        self.marker = torch.profiler.record_function("pvz_kernel_window")
        self.marker.__enter__()

    def _stop(self, calls):
        import torch

        # Drain inside the window: kernel tails must count against its wall.
        torch.cuda.synchronize()
        self.marker.__exit__(None, None, None)
        self.python.disable()
        self.profiler.stop()
        stem = self.path.with_suffix(f".{self.active}")
        chrome = stem.with_suffix(stem.suffix + ".chrome.json")
        self.profiler.export_chrome_trace(str(chrome))
        self.python.dump_stats(str(stem.with_suffix(stem.suffix + ".pstats")))
        result = kernel_summary(json.loads(chrome.read_text(encoding="utf-8")))
        result.update(
            calls=calls, chrome_trace=str(chrome), start_seconds=self.window_started - self.origin
        )
        self.results[self.active] = result
        self.path.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        self.active = self.profiler = self.python = self.marker = None

    def close(self):
        if self.active is not None:
            self._stop(0)
        for owner, name, original, owned in reversed(self.originals):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.originals.clear()


class GarbageCollectionTrace:
    def __init__(self, events, origin):
        self.events, self.origin, self.pending = events, origin, {}
        self.main_thread = threading.get_ident()
        gc.callbacks.append(self.callback)

    def callback(self, phase, info):
        key = threading.get_ident(), info["generation"]
        if phase == "start":
            self.pending[key] = perf_counter()
        else:
            started = self.pending.pop(key, None)
            if started is not None:
                self.events.append(
                    dict(
                        name="Python GC",
                        start=started - self.origin,
                        seconds=perf_counter() - started,
                        thread="main" if key[0] == self.main_thread else "coordinator",
                        thread_id=key[0],
                    )
                )

    def close(self):
        gc.callbacks.remove(self.callback)


class DeviceStageTrace:
    """Calibrated CUDA event coverage outside the collection tracer's lifetime."""

    def __init__(self, env, origin):
        self.cp, self.origin = env.cp, origin
        self.epoch = self.cp.cuda.Event()
        with env.device_context():
            self.epoch.record()
        self.epoch.synchronize()
        self.offset = perf_counter() - origin
        self.events = []

    def begin(self, label, stream):
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record(stream)
        self.events.append((label, start, end, stream))
        return end

    def finish(self):
        return [
            dict(
                name=label,
                start=self.offset + self.cp.cuda.get_elapsed_time(self.epoch, start) / 1000,
                seconds=self.cp.cuda.get_elapsed_time(start, end) / 1000,
                stream=int(stream.ptr),
            )
            for label, start, end, stream in self.events
            if end.done
        ]
