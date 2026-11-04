"""Draw measured collection states: benchmark_peashooter.py --process-trace records them."""

import argparse
import json
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from time import perf_counter


class CollectionTrace:
    """Optional diagnostic wrappers; no synchronization or production-path hooks."""

    def __init__(self, model):
        from pvz_rl.learning import effect_collection

        self.model, self.env = model, model.env
        self.events, self.originals = [], []
        self.origin = perf_counter()
        self.main_thread = threading.get_ident()
        self.first_drain = None
        self.gpu_before = dict(self.env.profiler.flush())
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
        service = getattr(self.env, "auxiliary", None)
        if service:
            self.wrap(service, "capture", "freeze snapshot")
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

    def wrap(self, owner, name, label, *, gpu=False):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        @wraps(original)
        def measured(*args, **kwargs):
            started = perf_counter()
            if name == "_prepare_collection" and not self.env.enabled_envs.any():
                if self.first_drain is None:
                    self.first_drain = started
            try:
                if gpu:
                    with self.env.profiler.track("profile_main_environment"):
                        return original(*args, **kwargs)
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

    def finish(self, path, *, settings, auxiliary, games, ended=None, setup_seconds=0):
        finished = ended if ended is not None else perf_counter()
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        totals, counts = defaultdict(float), defaultdict(int)
        events = list(self.events)
        for event in events:
            totals[event["name"]] += event["seconds"]
            counts[event["name"]] += 1
        gpu = {k: v - self.gpu_before.get(k, 0) for k, v in self.env.profiler.flush().items()}
        data = dict(
            protocol="measured-collection-process-v1",
            games=games,
            settings=settings,
            collection_seconds=finished - self.origin,
            setup_seconds=setup_seconds,
            main_trajectory_seconds=(self.first_drain or finished) - self.origin,
            label_drain_seconds=finished - self.first_drain if self.first_drain else 0,
            cpu_wall_seconds=dict(totals),
            calls=dict(counts),
            gpu_stream_seconds=gpu,
            auxiliary=auxiliary,
            events=events,
            timing_notes=[
                "CPU call spans are wall time and include thread scheduling; nested spans overlap.",
                "GPU main-stream event spans include launch gaps and stream dependencies.",
                "Auxiliary CUDA seconds measure kernel time; CPU seconds sum dispatch-to-result spans.",
                "Queue wait is summed per job; it overlaps collection and is not added to wall time.",
                "Drain starts after the last real decision and ends when all labels are published.",
            ],
        )
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data


def draw(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    cpu, gpu, aux = (data[k] for k in ("cpu_wall_seconds", "gpu_stream_seconds", "auxiliary"))

    def sec(value):
        return f"{value:.2f} s"

    total, drain = data["collection_seconds"], data["label_drain_seconds"]
    setup = data.get("setup_seconds", 0)
    fig, ax = plt.subplots(figsize=(15, 11), layout="constrained")
    ax.set(xlim=(0, 15), ylim=(0, 12))
    ax.axis("off")
    fig.suptitle(
        f"Real Peashooter collection · {data['games']} environments · {total + setup:.2f} s "
        f"({(total + setup) / data['games']:.3f} s/env)",
        fontsize=17,
        fontweight="bold",
    )
    colors = {"CPU": "#e8f0fa", "GPU": "#e6f4ec", "WAIT": "#fff1d6"}

    def box(x, y, title, body, kind, width=4.3, height=1.05):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.09",
                facecolor=colors[kind],
                edgecolor="#78848f",
                linewidth=0.8,
            )
        )
        ax.text(x + 0.13, y + height - 0.16, title, fontsize=11, fontweight="bold", va="top")
        ax.text(x + 0.13, y + height - 0.44, body, fontsize=10, va="top", linespacing=1.4)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="->",
                mutation_scale=13,
                color="#53616d",
                linewidth=1.1,
            )
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.06,
                label,
                fontsize=9,
                ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            )

    ax.text(0.2, 11.5, "MAIN COLLECTION THREAD", fontsize=11, fontweight="bold")
    ax.text(5.3, 11.5, "GPU STREAMS", fontsize=11, fontweight="bold")
    ax.text(10.3, 11.5, "BACKGROUND CPU", fontsize=11, fontweight="bold")
    box(
        0.2,
        9.95,
        "CPU → GPU · Choose action",
        f"Inference call wall: {sec(cpu.get('policy inference', 0))}\n"
        f"GPU stream: {sec(gpu.get('online_inference', 0))}",
        "CPU",
    )
    box(
        0.2,
        8.4,
        "CPU / GPU copy · Freeze owned roots",
        f"Capture call wall: {sec(cpu.get('freeze snapshot', 0))}\n"
        f"Storage backpressure: {sec(aux.get('host_wait_seconds', 0))}",
        "CPU",
    )
    box(
        5.3,
        8.4,
        "GPU · Step real environments",
        f"Stream: {sec(gpu.get('profile_main_environment', 0))} · simulation: {sec(gpu.get('simulation', 0))}\n"
        "Physical state / RNG; record Wait prefixes",
        "GPU",
    )
    box(
        0.2,
        6.85,
        "CPU WAIT · Main result handoff",
        f"Blocked: {sec(cpu.get('wait: main stream + result copies', 0))}\n"
        "Waits for policy + step + pinned D2H copies",
        "WAIT",
    )
    book = sum(
        cpu.get(k, 0)
        for k in (
            "interpret physical results",
            "trajectory storage",
            "interval: observe",
            "teacher: observe",
        )
    )
    box(
        0.2,
        5.3,
        "CPU · Commit decision / publish labels",
        f"Physical + row/owner calls: {sec(book)}\nNext action needs no simulation labels",
        "CPU",
    )
    box(
        10.3,
        9.95,
        "CPU · Decode / expand dependencies",
        f"Decode: {sec(aux.get('decode_seconds', 0))} · expand: {sec(cpu.get('coordinator: prepare_boards', 0))}\n"
        f"Spill: {sec(cpu.get('coordinator: spill', 0))} · {aux.get('root_spill_bytes', 0) / 2**20:.1f} MiB",
        "CPU",
    )
    box(
        10.3,
        8.4,
        "CPU · Shared queue / admission",
        f"Mean queued/job: {sec(aux.get('queue_wait_seconds', 0) / max(1, aux.get('submitted', 0)))} · max: {sec(aux.get('peak_queue_age_seconds', 0))}\n"
        f"Dispatch calls: GPU {sec(cpu.get('coordinator: dispatch_cuda', 0))} / CPU {sec(cpu.get('coordinator: dispatch_cpu', 0))}",
        "WAIT",
    )
    box(
        10.3,
        6.85,
        "CPU · Persistent simulation processes",
        f"Dispatch-to-result sum: {sec(aux.get('cpu_seconds', 0))}\n"
        f"{aux.get('cpu_jobs', 0):,} jobs · exact physical simulator",
        "CPU",
    )
    box(
        5.3,
        6.85,
        "GPU · Auxiliary low-priority chunks",
        f"Kernel sum: {sec(aux.get('cuda_seconds', 0))}\n"
        f"{aux.get('cuda_jobs', 0):,} jobs · field / probe / teacher",
        "GPU",
    )
    box(
        5.3,
        5.3,
        "CPU polling · Consume ready results",
        f"Result handling: {sec(cpu.get('coordinator: harvest_cuda', 0) + cpu.get('coordinator: harvest_cpu', 0))}\n"
        "Polls CUDA copy events / worker replies",
        "CPU",
    )
    box(
        0.2,
        3.5,
        "CPU WAIT · Cohort label drain",
        f"Wall: {sec(drain)} after all real games finish\n"
        "Waits for field/probe + teacher descendants",
        "WAIT",
        width=9.4,
    )
    box(
        10.3,
        3.5,
        "GPU · Fitting may now start",
        "Every required target published exactly once\nFitting excluded from this collection timing",
        "GPU",
    )
    for y in (9.95, 8.4, 6.85):
        arrow(2.35, y, 2.35, y - 0.5)
    arrow(4.5, 8.9, 5.3, 8.9)
    arrow(5.3, 8.9, 4.5, 7.35)
    arrow(2.35, 5.3, 2.35, 4.55, "all games ended")
    arrow(0.12, 5.75, 0.12, 10.4)
    ax.text(0.06, 8, "next decision", fontsize=9, rotation=90, ha="center", va="center")
    arrow(4.5, 9.3, 10.3, 10.55, "owned snapshot / no label barrier")
    arrow(12.45, 9.95, 12.45, 9.45)
    arrow(12.45, 8.4, 12.45, 7.9)
    arrow(10.3, 8.7, 9.6, 7.55, "GPU admitted")
    arrow(7.45, 6.85, 7.45, 6.35)
    arrow(12.45, 6.85, 9.6, 5.8)
    arrow(5.3, 5.8, 4.5, 5.8, "ready labels")
    arrow(9.6, 4.0, 10.3, 4.0, "labels ready")
    notes = [
        f"Setup: {sec(setup)}; main trajectory: {sec(data['main_trajectory_seconds'])}; label drain: {sec(drain)}.",
        "CUDA admission waits for main-stream readiness, an available auxiliary slot and memory headroom.",
        "CPU calls are wall spans; GPU stream spans include launch gaps. Nested / parallel spans must not be added.",
        "Queue residence and worker dispatch-to-result spans overlap across jobs / processes and the main trajectory.",
    ]
    ax.text(0.2, 2.95, "\n".join(notes), va="top", fontsize=10, linespacing=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.input.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
