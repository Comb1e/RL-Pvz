"""Summarize real workflow traces without adding nested spans or inventing GPU time."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from draw_collection_process import timeline
from profiling_metrics import host_attribution


def report(data, kernels=None):
    trace = data.get("process_trace", data)
    offset = data.get("collection_started_seconds", 0) + trace.get("setup_seconds", 0)
    if trace is data:
        offset = trace.get("setup_seconds", 0)
    events = [dict(e, start=e["start"] + offset) for e in trace.get("events", [])]
    phases = []
    names = {
        "Private pretraining": "private-pretraining",
        "Decision loop": "decisions",
        "Factual targets / teacher preparation": "factual-targets",
        "Teacher / remaining label drain": "teacher",
        "Four fitting passes": "fitting",
    }
    decisions = trace.get("decision_batches", trace.get("calls", {}).get("decision", 0))
    roots = trace.get("decision_roots", (data.get("final_collection") or {}).get("steps", 0))
    for phase in timeline(data)["phases"]:
        start, end = phase["start"], phase["end"]
        attribution = host_attribution(events, start=start, end=end)
        operations = attribution["operations"]
        for value in operations.values():
            value["ms_per_decision_batch"] = value["inclusive_seconds"] * 1000 / max(1, decisions)
            value["ms_per_board_decision"] = value["inclusive_seconds"] * 1000 / max(1, roots)
        phases.append(
            dict(
                name=phase["name"],
                start_seconds=start,
                end_seconds=end,
                wall_seconds=end - start,
                main_thread_uncovered_seconds=max(
                    0, end - start - attribution["covered_seconds_by_thread"].get("main", 0)
                ),
                operations=operations,
                kernel_window=(kernels or {}).get(names.get(phase["name"])),
            )
        )
    passes = defaultdict(lambda: dict(start=float("inf"), end=0.0, calls=0))
    for event in events:
        if "fitting_pass" in event:
            item = passes[event["fitting_pass"]]
            item["start"] = min(item["start"], event["start"])
            item["end"] = max(item["end"], event["start"] + event["seconds"])
            item["calls"] += 1
    for item in passes.values():
        item["wall_seconds"] = item["end"] - item["start"]
    return dict(
        protocol="training-hotspots-v1",
        decision_batches=decisions,
        board_decisions=roots,
        phases=phases,
        fitting_passes=dict(passes),
        loss_curve=data.get("fitting_passes"),
        label_signature=data.get("label_signature"),
        notes=[
            "Host self time excludes contained spans on the same thread; it includes OS scheduling and GPU waits, not just CPU instructions.",
            "Normalize a batched decision separately from an individual board decision.",
            "Kernel busy ratios describe only their bounded CUPTI windows, with profiler overhead; do not extrapolate to the whole phase.",
            "CUDA event spans include launch gaps and dependencies; they are not utilization. Missing timing is unknown, never idle.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", type=Path)
    args = parser.parse_args()
    kernels = json.loads(args.kernels.read_text(encoding="utf-8")) if args.kernels else None
    result = report(json.loads(args.input.read_text(encoding="utf-8")), kernels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        f"{result['decision_batches']:,} batch calls; {result['board_decisions']:,} board decisions.",
        "",
        "| Phase | Wall s | CUPTI window busy | Uncovered main-thread s |",
        "|---|---:|---:|---:|",
    ]
    for phase in result["phases"]:
        kernel = phase["kernel_window"]
        busy = (
            f"{kernel['kernel_busy_ratio']:.1%} ({kernel['wall_seconds']:.2f} s window)"
            if kernel
            else "Unmeasured"
        )
        lines.append(
            f"| {phase['name']} | {phase['wall_seconds']:.3f} | {busy} | {phase['main_thread_uncovered_seconds']:.3f} |"
        )
    for phase in result["phases"]:
        lines.extend(
            [
                "",
                phase["name"],
                "",
                "| Operation | Calls | Inclusive s | Self s | ms/batch | ms/board decision |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, value in sorted(
            phase["operations"].items(), key=lambda item: -item[1]["self_seconds"]
        )[:15]:
            lines.append(
                f"| {name} | {value['calls']} | {value['inclusive_seconds']:.3f} | {value['self_seconds']:.3f} | {value['ms_per_decision_batch']:.3f} | {value['ms_per_board_decision']:.5f} |"
            )
    lines.extend(["", *result["notes"]])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
