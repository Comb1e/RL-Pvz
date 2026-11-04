"""Draw chronological CPU/GPU timelines; collapse loops to their start/end markers.

Record inputs with benchmark_peashooter.py --process-trace or
profile_training_workflow.py. Older host-only recordings remain supported.
"""

import argparse
import json
import threading
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from time import perf_counter


class CollectionTrace:
    """Optional wrappers; one epoch calibration, event queries otherwise nonblocking."""

    def __init__(self, model):
        from pvz_rl.learning import effect_collection

        self.model, self.env = model, model.env
        self.events, self.originals = [], []
        self.device_events = []
        self.epoch = self.env.cp.cuda.Event()
        # One diagnostic calibration before collection; never synchronize per span.
        with self.env.device_context():
            self.epoch.record()
        self.epoch.synchronize()
        self.origin = perf_counter()
        self.main_thread = threading.get_ident()
        self.first_drain = None
        self.gpu_before = dict(self.env.profiler.flush())
        self.trace_profiler()
        self.wrap(model, "_prepare_collection", "prepare / label publication")
        self.wrap(model, "_collect_step", "decision")
        self.wrap(model._runner, "decide", "policy inference")
        self.wrap(effect_collection, "begin_effects", "candidate / snapshot preparation")
        self.wrap(self.env, "step_device", "main environment submission", gpu=True)
        self.wrap(self.env, "step_host", "interpret physical results")
        self.wrap(self.env.handoff, "wait", "wait: main stream + result copies")
        self.wrap(model._buffer, "append", "trajectory storage")
        for owner, prefix in (
            (getattr(model, "_wait_outcomes", None), "interval"),
            (getattr(model, "_peashooter_teacher", None), "teacher"),
        ):
            if owner:
                for name in ("before", "arm", "observe", "advance", "finalize"):
                    if hasattr(owner, name):
                        self.wrap(owner, name, f"{prefix}: {name}")
        reach = getattr(model, "_reach", None)
        if reach is not None:
            self.wrap(reach, "prepare", "factual: prepare")
        teacher = getattr(model, "_peashooter_teacher", None)
        if teacher is not None and hasattr(teacher, "_run_chunk"):
            self.wrap(teacher, "_initialize", "teacher: initialize")
            self.wrap(teacher, "_fill", "teacher: fill")
            traced_batches = set()

            def trace_teacher_batch():
                if teacher.batch is not None and id(teacher.batch) not in traced_batches:
                    self.wrap(
                        teacher.batch,
                        "step_device",
                        "teacher simulation submission",
                        device_label="teacher simulation",
                    )
                    traced_batches.add(id(teacher.batch))

            self.wrap(teacher, "_run_chunk", "teacher: run chunk", before=trace_teacher_batch)
        service = getattr(self.env, "auxiliary", None)
        if service:
            self.wrap(service, "capture", "freeze snapshot", device_label="freeze snapshot")
            for family, backend in service.gpu.items():
                self.wrap(
                    backend,
                    "launch",
                    f"auxiliary: {family} submission",
                    device_label=f"auxiliary: {family}",
                    stream=backend.stream,
                )
            for name in (
                "_prepare_boards",
                "_dispatch_cpu",
                "_dispatch_cuda",
                "_harvest_cpu",
                "_harvest_cuda",
                "_migrate_drain",
            ):
                self.wrap(service, name, f"coordinator: {name[1:]}")
            for name in ("materialize", "spill"):
                self.wrap(service.store, name, f"coordinator: {name}")

    @contextmanager
    def device_span(self, label, stream=None):
        cp = self.env.cp
        stream = stream if stream is not None else cp.cuda.get_current_stream()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self.device_events.append((label, start, end, int(stream.ptr)))

    def trace_profiler(self):
        profiler = self.env.profiler
        original = profiler.track
        self.originals.append((profiler, "track", original, "track" in vars(profiler)))

        @contextmanager
        def track(phase):
            with original(phase), self.device_span(phase):
                yield

        profiler.track = track

    def wrap(self, owner, name, label, *, gpu=False, device_label=None, stream=None, before=None):
        original = getattr(owner, name)
        self.originals.append((owner, name, original, name in vars(owner)))

        @wraps(original)
        def measured(*args, **kwargs):
            started = perf_counter()
            if before:
                before()
            if name == "_prepare_collection" and not self.env.enabled_envs.any():
                if self.first_drain is None:
                    self.first_drain = started
            try:
                with self.device_span(device_label, stream) if device_label else nullcontext():
                    if gpu:
                        with self.env.profiler.track("profile_main_environment"):
                            return original(*args, **kwargs)
                    else:
                        return original(*args, **kwargs)
            finally:
                self.events.append(
                    dict(
                        name=label,
                        start=started - self.origin,
                        seconds=perf_counter() - started,
                        thread="main"
                        if threading.get_ident() == self.main_thread
                        else "coordinator",
                    )
                )

        setattr(owner, name, measured)

    def restore(self):
        """Remove inherited-method overrides so checkpoint data cannot retain this tracer."""
        for owner, name, original, owned in reversed(self.originals):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.originals.clear()

    def finish(
        self, path, *, settings, auxiliary, games, ended=None, setup_seconds=0, workload=None
    ):
        finished = ended if ended is not None else perf_counter()
        self.restore()
        totals, counts = defaultdict(float), defaultdict(int)
        events = list(self.events)
        for event in events:
            totals[event["name"]] += event["seconds"]
            counts[event["name"]] += 1
        gpu = {k: v - self.gpu_before.get(k, 0) for k, v in self.env.profiler.flush().items()}
        device_events = [
            dict(
                name=label,
                start=self.env.cp.cuda.get_elapsed_time(self.epoch, start) / 1000,
                seconds=self.env.cp.cuda.get_elapsed_time(start, end) / 1000,
                stream=stream,
            )
            for label, start, end, stream in self.device_events
            if end.done
        ]
        data = dict(
            protocol="measured-collection-process-v2",
            games=games,
            settings=settings,
            workload=workload,
            collection_seconds=finished - self.origin,
            setup_seconds=setup_seconds,
            main_trajectory_seconds=(self.first_drain or finished) - self.origin,
            label_drain_seconds=finished - self.first_drain if self.first_drain else 0,
            cpu_wall_seconds=dict(totals),
            calls=dict(counts),
            gpu_stream_seconds=gpu,
            auxiliary=auxiliary,
            events=events,
            gpu_events=device_events,
            pending_gpu_events=len(self.device_events) - len(device_events),
            timing_notes=[
                "CPU call spans are wall time and include thread scheduling; nested spans overlap.",
                "GPU main-stream event spans include launch gaps and stream dependencies.",
                "Auxiliary CUDA seconds measure kernel time; CPU seconds sum dispatch-to-result spans.",
                "Queue wait is summed per job; it overlaps collection and is not added to wall time.",
                "Drain starts after the last real decision and ends when all labels are published.",
                "GPU event timestamps share one calibrated epoch; no per-operation synchronization is added.",
            ],
        )
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data


def _seconds(value):
    return f"{value:.3f} s" if value < 1 else f"{value:.2f} s"


def _events(data, *, workflow=False, offset=0):
    """Normalize recorded spans; never fabricate positions from aggregate totals."""
    return [
        dict(
            name=e.get("operation") if workflow else e["name"],
            start=e.get("start_seconds", 0) if workflow else e["start"] + offset,
            seconds=e.get("duration_seconds", 0) if workflow else e["seconds"],
            thread=e.get("thread", "main"),
        )
        for e in data.get("events", [])
    ]


def _total(events, name, start, end):
    return sum(
        max(0, min(e["start"] + e["seconds"], end) - max(e["start"], start))
        for e in events
        if e["name"] == name
    )


def timeline(data):
    """One ordered phase frontier shared by CPU and GPU; loop interiors collapse."""
    if data.get("process_trace") is not None:
        source = dict(data)
        detailed = timeline(source.pop("process_trace"))
        prefix = timeline(source)
        phases = [p for p in prefix["phases"] if p["name"] in ("Startup", "Private pretraining")]
        offset = data["collection_started_seconds"]
        for phase in detailed["phases"]:
            if phase["name"] == "Decision loop":
                graphs = (
                    (data.get("final_collection") or {})
                    .get("collection_execution", {})
                    .get("graph_cache", {})
                )
                captures = sum(
                    value.get("capture_setup_seconds", 0)
                    for value in graphs.get("online", {}).values()
                )
                if captures:
                    phase["cpu"].append(
                        f"Graph capture setup (within inference): {_seconds(captures)}"
                    )
            phases.append(dict(phase, start=phase["start"] + offset, end=phase["end"] + offset))
        return dict(phases=phases, gpu_timestamps=detailed["gpu_timestamps"], workflow=True)
    workflow = "command" in data and "timings" in data
    setup = 0 if workflow else data.get("setup_seconds", 0)
    events = _events(data, workflow=workflow, offset=setup)
    gpu_events = [dict(e, start=e["start"] + setup) for e in data.get("gpu_events", [])]
    phases = []

    def add(start, end, name, cpu, gpu, *, loop=False, complete=True, wait=None):
        if end <= start:
            return

        def measured(choices, source):
            return [
                f"{label}: {_seconds(value)}"
                for key, label in choices
                if (value := _total(source, key, start, end)) > 0
            ]

        phases.append(
            dict(
                name=name,
                start=start,
                end=end,
                loop=loop,
                complete=complete,
                cpu=measured(cpu, events),
                gpu=measured(gpu, gpu_events),
                wait=wait,
            )
        )
        return phases[-1]

    main_cpu = [
        ("interval: advance", "Poll / finish intervals (nested)"),
        ("policy inference", "Choose action"),
        ("candidate / snapshot preparation", "Prepare candidates"),
        ("main environment submission", "Submit real step"),
        ("wait: main stream + result copies", "WAIT for step / D2H"),
        ("interpret physical results", "Interpret results"),
        ("trajectory storage", "Retain trajectory"),
        ("teacher: observe", "Retain teacher roots"),
    ]
    gpu_ops = [
        ("online_inference", "Policy inference"),
        ("freeze snapshot", "Freeze packed roots"),
        ("profile_main_environment", "Real environment step"),
        ("auxiliary: outcomes", "Interval chunks + D2H"),
        ("auxiliary: teacher", "Teacher chunks + D2H"),
        ("teacher simulation", "Teacher simulation"),
    ]
    if workflow:
        collection_start = data.get("collection_started_seconds")
        if collection_start is None:
            collection_start = data["elapsed_seconds"]
        advances = [
            e
            for e in events
            if e["name"] == "ForecastPretraining.advance" and e["start"] < collection_start
        ]
        pretrain_start = min((e["start"] for e in advances), default=collection_start)
        add(
            0,
            pretrain_start,
            "Startup",
            [
                ("imports", "Import packages"),
                ("pvz_rl.learning.training.initial_weights", "Load weights"),
                ("pvz_rl.learning.training.vector_env", "Create environment"),
                ("pvz_rl.learning.training.build_model", "Build model"),
            ],
            [],
        )
        pretraining = add(
            pretrain_start, collection_start, "Private pretraining", [], [], loop=True
        )
        if advances and pretraining is not None:
            startup = data.get("timings_by_phase", {}).get("startup", data["timings"])
            seconds = startup.get("ForecastPretraining.advance", {}).get("seconds", 0)
            pretraining["cpu"] = [
                f"Prepare roots / submit / wait / fit: {_seconds(seconds)} total",
                "Repeated private panels; inner iterations collapsed",
            ]
            pretraining["gpu_activity"] = "Simulate private roots, then fit auxiliary heads"
        window = data.get("collection_window_seconds")
        if window is None:
            final = data.get("final_collection") or (data.get("samples") or [{}])[-1]
            window = max(0, final.get("elapsed_seconds", collection_start) - collection_start)
        end = collection_start + window
        drain = data.get("label_drain_started_seconds")
        main_end = min(end, drain) if drain is not None else end
        main_cpu = [
            ("CudaSequentialQ._prepare_collection", "Prepare / label work"),
            ("CudaSequentialQ._collect_step", "Choose / step / retain"),
            ("WaitOutcomes.advance", "Interval labeling (nested)"),
            ("CudaVecEnv.step_device", "Submit real step (nested)"),
            ("CudaVecEnv.step_host", "Interpret results (nested)"),
        ]
        decision = add(
            collection_start,
            main_end,
            "Decision loop",
            main_cpu,
            [],
            loop=True,
            complete=drain is not None,
            wait="Legacy synchronous labels / result copies are included in host calls",
        )
        # Workflow v1 records selected spans only; cumulative totals are authoritative.
        stats = data.get("timings_by_phase", {}).get("collection", data["timings"])
        if decision is not None:
            # Selected v1 events cannot partition cumulative totals across a drain.
            suffix = "whole collection" if drain is not None and drain < end else "total"
            decision["cpu"] = [
                f"{label}: {_seconds(stats[key]['seconds'])} {suffix}"
                for key, label in main_cpu
                if key in stats
            ]
        if drain is not None and drain < end:
            add(
                drain,
                end,
                "Label drain",
                [],
                [],
                loop=True,
                complete=False,
                wait="Field / probe continuations and teacher descendants",
            )
        if phases:
            phases[-1]["end_note"] = "Recording stops here; pending work is not assigned a duration"
    else:
        total = setup + data["collection_seconds"]
        main_end = setup + data["main_trajectory_seconds"]
        add(0, setup, "Cohort preparation", [], [])
        if setup:
            phases[-1]["cpu"] = ["Prepare boards / interval owners / worker resources"]
            phases[-1]["gpu_activity"] = "Initialize boards and prewarm scratch / kernels"
        add(
            setup,
            main_end,
            "Decision loop",
            main_cpu,
            gpu_ops,
            loop=True,
            wait="Real-step completion / D2H; interval work is included in this loop",
        )
        # Only introduce boundaries actually present in the recording.
        factual = min(
            (
                e["start"]
                for e in events
                if e["name"] == "factual: prepare" and e["start"] >= main_end
            ),
            default=total,
        )
        teacher = min(
            (
                e["start"]
                for e in events
                if e["name"] in ("teacher: run chunk", "teacher: advance")
                and e["start"] >= main_end
            ),
            default=total,
        )
        split = min(factual, teacher)
        add(
            main_end,
            split,
            "Interval drain / publish",
            [
                ("interval: advance", "Poll / finish intervals"),
                ("interval: finalize", "Publish interval targets"),
            ],
            gpu_ops,
            loop=True,
            wait="Remaining interval field / probe jobs; publish all exact targets",
        )
        run = min(
            (
                e["start"]
                for e in events
                if e["name"] == "teacher: run chunk" and e["start"] >= split
            ),
            default=split,
        )
        add(
            split,
            run,
            "Factual targets / teacher preparation",
            [
                ("factual: prepare", "Build factual targets"),
                ("teacher: initialize", "Project / prepare teacher roots"),
            ],
            gpu_ops,
            loop=True,
            wait="CPU lane-preparation replies (included in preparation calls)"
            if data.get("settings", {}).get("prepare_workers", 0)
            else None,
        )
        add(
            run,
            total,
            "Teacher / remaining label drain",
            [
                ("factual: prepare", "Build remaining factual targets"),
                ("teacher: initialize", "Prepare remaining teacher roots"),
                ("teacher: fill", "Refill teacher workers"),
                ("teacher: run chunk", "Submit / download / publish (nested)"),
                ("teacher: advance", "Teacher total (includes nested calls)"),
            ],
            gpu_ops,
            loop=True,
            wait="Teacher coverage / candidate results; synchronous header copies included",
        )
        phases[-1]["end_note"] = "All required labels ready; fitting may start (not timed here)"
        if data.get("cohort_finished_seconds") is not None:
            phases[-1].pop("end_note", None)
            fitting_start = setup + data["fitting_started_seconds"]
            fitting_end = max(
                (
                    e["start"] + e["seconds"]
                    for e in events
                    if e["name"] == "fit pass / forward backward optimizer"
                ),
                default=setup + data["cohort_finished_seconds"],
            )
            preparation = add(
                total,
                fitting_start,
                "Prepare fitting targets / release simulation scratch",
                [
                    ("finalize rewards", "Finalize rewards"),
                    ("assemble fitting targets", "Assemble targets"),
                ],
                [],
            )
            if preparation is not None:
                preparation["gpu_activity"] = "Release simulation scratch / stage fitting tensors"
            fitting = add(
                fitting_start,
                fitting_end,
                "Four fitting passes",
                [
                    (
                        "fit pass / forward backward optimizer",
                        "Forward / backward / commit (host wall)",
                    )
                ],
                [("fit forward backward optimizer", "Forward / backward / optimizer event spans")],
                loop=True,
                wait="Batch staging / GPU loss and gradient checks before each optimizer commit",
            )
            fitting["end_note"] = "Full cohort fitted; checkpoint / report cleanup excluded"

    for phase in phases:
        if not phase["cpu"]:
            phase["cpu"] = ["Host activity not separately instrumented"]
        if not phase["gpu"]:
            if phase["name"] == "Startup":
                phase["gpu_activity"] = "Initialize CUDA / load weights / allocate model"
            if phase["name"] == "Decision loop":
                phase["gpu_activity"] = "Policy inference → real step; interval jobs when admitted"
            phase["gpu"] = [
                phase.get("gpu_activity", "Private simulations / transfers when submitted"),
                "Device timestamps unavailable for this phase",
            ]
    return dict(phases=phases, gpu_timestamps=bool(gpu_events), workflow=workflow)


def draw(data, output):
    """Two timelines, top to bottom in wall-clock order, with no unrolled loops."""
    import matplotlib

    matplotlib.use("Agg")
    from textwrap import wrap

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    result = timeline(data)
    phases = result["phases"]
    heights = [
        max(2.0, 0.30 * (max(len(p["cpu"]), len(p["gpu"])) + 4)) + (0.45 if p["wait"] else 0)
        for p in phases
    ]
    gaps = [0.58 if p.get("end_note") else 0.35 for p in phases]
    height = sum(heights) + sum(gaps) + 2.5
    fig, ax = plt.subplots(figsize=(15, height * 0.82))
    ax.set(xlim=(0, 15), ylim=(height, 0))
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.02)
    ax.text(7.5, 0.20, "CPU / GPU process timelines", ha="center", fontsize=18, weight="bold")
    ax.text(
        7.5,
        0.58,
        "Time flows downward · shared elapsed timestamps · loop interiors collapsed · spacing is schematic",
        ha="center",
        fontsize=10,
        color="#526170",
    )
    ax.text(
        3.8,
        1.02,
        "CPU timeline · main thread + coordinator / workers",
        ha="center",
        fontsize=11,
        weight="bold",
        color="#275d93",
    )
    ax.text(
        11.2,
        1.02,
        "GPU timeline · main + auxiliary streams",
        ha="center",
        fontsize=11,
        weight="bold",
        color="#25724b",
    )
    for x, color in ((0.45, "#4b7aab"), (14.55, "#43966b")):
        ax.annotate(
            "",
            xy=(x, height - 1.15),
            xytext=(x, 1.2),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

    y = 1.35
    for phase, span, gap in zip(phases, heights, gaps, strict=True):
        top, bottom = y + 0.28, y + span - 0.25
        duration = phase["end"] - phase["start"]
        marker = "LOOP START · " if phase["loop"] else ""
        ax.text(
            7.5,
            y,
            f"{marker}{phase['name']} · wall {_seconds(duration)}",
            fontsize=11,
            ha="center",
            va="center",
            weight="bold",
        )
        for x, lane, color in ((0.8, "cpu", "#e9f1fa"), (8.1, "gpu", "#e5f4eb")):
            unknown = lane == "gpu" and phase[lane][-1].startswith("Device timestamps")
            ax.add_patch(
                FancyBboxPatch(
                    (x, top),
                    6.1,
                    bottom - top,
                    boxstyle="round,pad=0.06",
                    facecolor=color,
                    edgecolor="#7a8c9b",
                    hatch="/" if unknown else None,
                    lw=0.7,
                )
            )
            lines = phase[lane]
            ax.text(
                x + 0.15,
                top + 0.12,
                "\n".join(lines),
                va="top",
                fontsize=10.2,
                linespacing=1.45,
                color="#173044",
            )
            if lane == "cpu" and phase["wait"]:
                wait = "\n".join(wrap("WAIT FOR: " + phase["wait"], width=78))
                ax.text(
                    x + 0.15,
                    bottom - 0.13,
                    wait,
                    va="bottom",
                    fontsize=9,
                    color="#925000",
                    bbox=dict(facecolor="#fff0ce", edgecolor="none", pad=3),
                )
        for stamp, at in ((phase["start"], top + 0.06), (phase["end"], bottom - 0.04)):
            ax.text(
                7.5,
                at,
                f"{stamp:.2f} s",
                ha="center",
                va="center",
                fontsize=9,
                color="#526170",
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
            )
        if phase["loop"]:
            end = "LOOP END" if phase["complete"] else "LOOP STILL OPEN · window ends"
            ax.text(7.5, bottom + 0.18, end, ha="center", fontsize=9.5, weight="bold")
        if phase.get("end_note"):
            ax.text(7.5, bottom + 0.42, phase["end_note"], ha="center", fontsize=9, color="#526170")
        y += span + gap
    notes = [
        "Call durations are cumulative within each phase; nested CPU spans overlap. "
        "GPU event spans include launch gaps / stream dependencies, not GPU utilization.",
        "Each box spans its phase's wall window. Unmeasured GPU placement is hatched; "
        "missing timing never means idle or zero work.",
    ]
    if not result["gpu_timestamps"]:
        gpu = data.get("gpu_stream_seconds", {})
        parts = [
            f"{name} {_seconds(gpu[key])}"
            for key, name in (("online_inference", "inference"), ("simulation", "real simulation"))
            if key in gpu
        ]
        notes.append(
            "This older recording has aggregate GPU timing only: " + "; ".join(parts)
            if parts
            else "Host-only recording; no device timestamp trace was captured."
        )
    notes.append(
        "Benchmark traces exclude CLI startup / private pretraining. Actual-CLI audits "
        "retain startup and explicitly mark unfinished collection windows."
    )
    ax.text(
        0.8,
        height - 0.8,
        "\n".join(notes),
        va="top",
        fontsize=8.8,
        color="#526170",
        linespacing=1.45,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    output.with_suffix(".timeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plt.close(fig)


def draw_workflow(data, output):
    """Compatibility entry point; actual-CLI and benchmark inputs share two lanes."""
    draw(data, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if "command" in data and "timings" in data:
        draw_workflow(data, args.output)
    else:
        draw(data, args.output)


if __name__ == "__main__":
    main()
