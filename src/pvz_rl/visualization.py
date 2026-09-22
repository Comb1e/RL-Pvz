"""Offline run reports and fixed-case demos from one shared best checkpoint."""

from __future__ import annotations

import html
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .budget import curve_axis
from .config import output_settings
from .deadline import BudgetExpired, check_deadline
from .evaluation import evaluate
from .progress import Phase, ProgressReporter
from .provenance import current_engine_config, file_hash, write_json
from .video import export_replay, video_settings


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def read_series(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_segments(run):
    """Only link ancestors with a recorded resume boundary; label times separately."""
    result, seen, cutoff = [], set(), None
    current = Path(run).resolve()
    while current not in seen:
        seen.add(current)
        meta = read_json(current / "metadata.json", {})
        series = {}
        for name in ("learning-curve", "training-metrics"):
            rows = read_series(current / f"{name}.jsonl")
            # Updates at a repeated step supersede progress-only rows at that step.
            series[name] = list(
                {
                    row["training_steps"]: row
                    for row in rows
                    if cutoff is None or row["training_steps"] <= cutoff
                }.values()
            )
        result.append((current.name, series))
        boundary = meta.get("resume_steps")
        if not meta.get("resume") or boundary is None:
            break
        cutoff = boundary if cutoff is None else min(boundary, cutoff)
        current = Path(meta["resume"]).resolve().parent
        if not (current / "metadata.json").exists():
            break
    return list(reversed(result))


def _save(fig, output, name):
    fig.tight_layout()
    temporary = output / (name + ".tmp.png")
    try:
        fig.savefig(temporary, dpi=150)
        temporary.replace(output / (name + ".png"))
    finally:
        plt.close(fig)


def _empty(ax, message="Not recorded yet"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def build_run_report(run, cfg=None):
    run = Path(run).resolve()
    meta = read_json(run / "metadata.json", {})
    cfg = cfg or meta["config"]
    output = run / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    segments = run_segments(run)
    progress_key, progress_label = curve_axis(cfg)
    images = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"macro": "#233d37", "easy": "#399471", "standard": "#d69536", "hard": "#b54e59"}
    for ax, xkey, xlabel in zip(
        axes,
        (progress_key, "wall_seconds"),
        (progress_label, "Wall time within each run segment (hours)"),
    ):
        plotted = False
        for label, series in segments:
            rows = series["learning-curve"]
            for level in colors:
                points = []
                for row in rows:
                    value = (
                        row.get("macro_win_rate")
                        if level == "macro"
                        else next(
                            (
                                score["win_rate"]
                                for key, score in row.get("levels", {}).items()
                                if key.endswith("/" + level)
                            ),
                            None,
                        )
                    )
                    if value is not None and row.get(xkey) is not None:
                        points.append((row[xkey] / (3600 if xkey == "wall_seconds" else 1), value))
                if points:
                    x, y = zip(*points)
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        markersize=3,
                        color=colors[level],
                        label=f"{label}: {level}",
                        alpha=0.85,
                    )
                    plotted = True
        ax.set(xlabel=xlabel, ylabel="Validation win rate", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=7)
            ax.set_xlim(left=0)
        else:
            _empty(ax, "Validation has not finished yet")
    _save(fig, output, "validation-curves")
    images.append(("Validation performance", "validation-curves.png"))

    panels = [
        ("rolling_win_rate", "Rolling training win rate"),
        ("rolling_return", "Rolling episode reward"),
        ("rolling_seconds", "Rolling episode duration (simulated seconds)"),
        ("invalid_action_rate", "Rolling invalid-action rate"),
        ("decisions_per_second", "Training decisions / second (excludes validation/reporting)"),
        ("end_to_end_decisions_per_second", "Training decisions / wall second"),
        ("rolling_plant_kills", "Plant kills / training episode"),
        ("rolling_mower_kills", "Mower kills / training episode"),
        ("rolling_nonlethal_health_damage", "Nonlethal damage / training episode"),
        ("rolling_empty_mower_activations", "Empty mower activations / training episode"),
        ("rolling_mower_activation_penalty", "Mower activation cost / game"),
        ("rolling_wall_nut_damage", "Wall-nut damage / training episode"),
        ("rolling_empty_explosions", "Empty explosions / training episode"),
        ("simulation_ticks_per_second", "Simulation ticks / training second"),
        ("instant_action_fraction", "Fraction of decisions without advancing time"),
        ("rolling_early_voluntary_digs", "Early voluntary digs / game (configured window)"),
        ("early_digs_per_planting", "Early voluntary digs / accepted planting"),
        ("rolling_attacker_purchases", "Sustained attackers purchased / game"),
        ("rolling_maximum_sun", "Maximum sun / game"),
        ("rolling_first_attacker_seconds", "First sustained attacker (seconds; purchasing games)"),
    ]
    fig, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(12, 3 * ((len(panels) + 1) // 2)))
    for ax, (key, title) in zip(axes.flat, panels):
        plotted = False
        for label, series in segments:
            rows = [
                r
                for r in series["training-metrics"]
                if r.get(key) is not None and r.get(progress_key) is not None
            ]
            if rows:
                ax.plot(
                    [r[progress_key] for r in rows],
                    [r[key] for r in rows],
                    marker=".",
                    label=label,
                )
                plotted = True
        ax.set(title=title, xlabel=progress_label)
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=7)
        else:
            has_metrics = any(s["training-metrics"] for _, s in segments)
            has_episodes = any(
                row.get("rolling_episodes", 0) > 0
                for _, series in segments
                for row in series["training-metrics"]
            )
            _empty(
                ax,
                "Metric unavailable in this run"
                if has_episodes
                else "No completed training episodes yet"
                if has_metrics
                else "No aggregated metrics available",
            )
    _save(fig, output, "training-curves")
    images.append(("Training behavior and throughput", "training-curves.png"))

    optimizer_panels = [
        ("policy_gradient_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("entropy_loss", "Entropy loss (negative entropy)"),
        ("approx_kl", "Approximate KL"),
        ("clip_fraction", "Clipped fraction"),
        ("explained_variance", "Explained variance"),
        ("joint_entropy", "True joint action entropy"),
        ("exploration_bonus", "Exploration bonus in optimizer objective"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for ax, (key, title) in zip(axes.flat, optimizer_panels):
        plotted = False
        for label, series in segments:
            rows = [
                r
                for r in series["training-metrics"]
                if r.get("optimization", {}).get(key) is not None
                and r.get(progress_key) is not None
            ]
            if rows:
                ax.plot(
                    [r[progress_key] for r in rows],
                    [r["optimization"][key] for r in rows],
                    marker=".",
                    label=label,
                )
                plotted = True
        ax.set(title=title, xlabel=progress_label + " at completed PPO update")
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=7)
        else:
            _empty(ax, "Optimizer metrics unavailable")
    _save(fig, output, "optimization-curves")
    images.append(("PPO optimization", "optimization-curves.png"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, key, title in zip(
        axes.flat,
        (
            "rolling_attacker_purchases",
            "rolling_maximum_sun",
            "type_entropy",
            "conditional_tile_entropy",
        ),
        (
            "Sustained attackers purchased / episode",
            "Maximum sun / episode",
            "Action-type entropy",
            "Type-weighted tile entropy",
        ),
    ):
        plotted = False
        for label, series in segments:
            points = [
                (r.get(progress_key), r.get(key, r.get("optimization", {}).get(key)))
                for r in series["training-metrics"]
            ]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, marker="o", markersize=3, label=label)
                plotted = True
        ax.set(title=title, xlabel=progress_label)
        if plotted:
            ax.legend(fontsize=7)
        else:
            _empty(ax)
    _save(fig, output, "learning-diagnostics")
    images.append(("Economy and exploration", "learning-diagnostics.png"))

    status = read_json(run / "status.json", {})
    best = read_json(run / "best.json", {})
    visual_status = read_json(output / "status.json", {})
    demos = read_json(output / "demos.json", {}).get("demos", [])
    # Never display stale videos as demonstrations of a newly selected checkpoint.
    demos = [d for d in demos if d.get("checkpoint_hash") == best.get("checkpoint_hash")]

    def escape(value):
        return html.escape(str(value), quote=True)

    cards = []
    for demo in demos:
        video = demo.get("video")
        player = (
            f'<video controls preload="metadata" src="{escape(video)}"></video>'
            '<label>Playback speed <select onchange="this.parentElement.previousElementSibling'
            '.playbackRate=Number(this.value)"><option>0.5</option><option selected>1</option>'
            "<option>2</option><option>4</option></select>×</label>"
            if video and (output / video).exists()
            else "<p>Recording available for the game viewer. MP4 export is optional.</p>"
        )
        cards.append(
            f"<article><h3>{escape(demo['level'])} — {escape(demo['outcome'])}</h3>"
            f"<p>Validation seed {demo['scenario_seed']}; "
            f"{demo['simulated_seconds']:.1f} simulated seconds.</p>{player}"
            f'<p><a href="{escape(demo["replay"])}">Verified '
            f"{'compact demo' if demo['replay'].endswith('.pvzdemo') else 'legacy replay'}</a></p>"
            f'<pre>pvz-rl replay "{escape((output / demo["replay"]).resolve())}" --watch --speed 2</pre>'
            f'<p class="hash">Shared checkpoint SHA-256: {escape(demo["checkpoint_hash"])}</p>'
            "</article>"
        )
    details = {
        "training_state": status.get("state", "unknown"),
        "condition": meta.get("condition"),
        "profile": cfg.get("profile", "baseline"),
        "observation_version": cfg["encoding"]["version"],
        "policy": cfg.get("policy", {"kind": "flat"}),
        "learner_seed": meta.get("learner_seed"),
        "family": meta.get("family"),
        "training_settings": cfg["training"],
        "curriculum": cfg["curriculum"],
        "initialization": meta.get("initialization"),
        "progress": status,
        "selected_shared_checkpoint": best or "No validated checkpoint yet",
        "visualization_status": visual_status,
    }
    charts = "".join(
        f'<section><h2>{title}</h2><a href="{name}"><img src="{name}" alt="{title}"></a></section>'
        for title, name in images
    )
    policy_description = (
        "Restricted diagnostic policy; this run does not evaluate the full game"
        if meta.get("family") in ("diagnostic", "placement", "saving")
        else "One shared policy for easy, standard, and hard"
    )
    completion = (
        "Curriculum incomplete" if status.get("curriculum_incomplete") else "Curriculum status"
    )
    run_progress = (
        f"<p>Completed games: {escape(status.get('training_games', 'not recorded'))}. "
        f"{completion}: {escape(status.get('curriculum_stage', 'not recorded'))}. "
        f"Stop reason: {escape(status.get('stop_reason') or 'not recorded / in progress')}. "
        f"Game target reached: {escape(status.get('budget_complete', 'in progress'))}.</p>"
    )
    if status.get("selected_stage"):
        run_progress += (
            f"<p>Single stage: {escape(status['selected_stage'])}. "
            f"Mastery reached: {escape(status.get('stage_mastered', False))}. "
            "Use final.zip to initialize another stage with a fresh budget.</p>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PVZ training — {escape(run.name)}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;background:#edf2ed;color:#203b33;
max-width:1200px;margin:auto;padding:24px}}section,article,header{{background:white;
border-radius:12px;padding:22px;margin:18px 0}}h1,h2,h3{{line-height:1.2}}img,video{{width:100%;
height:auto;border-radius:6px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}
.hash{{font:12px monospace;overflow-wrap:anywhere}}a{{color:#26684f}}select{{margin:12px}}
</style></head><body><header><h1>PVZ / {escape(run.name)}</h1>
<p>{policy_description}. Training: <b>{escape(status.get("state", "unknown"))}</b>.</p>
{run_progress}
<p>Checkpoint selection uses equal-weight validation win rates. These development curves and
fixed validation demonstrations do not establish held-out performance.</p>
<p>Training curves use up to {output_settings(cfg)["logging"]["rolling_window"]} completed episodes;
difficulty composition changes with the curriculum. Reward is not comparable across reward settings.
Resumed segments have separate wall-time axes and lines. Missing older metrics are left blank.</p>
<details><summary>Settings, checkpoint, and export status</summary>
<pre>{escape(json.dumps(details, indent=2))}</pre></details></header>{charts}
<section><h2>Shared checkpoint demonstrations</h2>
<p>Each full recording uses identical weights on a predetermined validation case.
Diagnostic runs show only the diagnostic task. A short smoke run may end at its configured cutoff.</p>
{"".join(cards) or "<p>No demonstrations generated yet.</p>"}</section></body></html>"""
    temporary = output / "index.tmp.html"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output / "index.html")
    return output / "index.html"


def create_demonstrations(run, cfg, progress, deadline=None):
    from .training import load_policy
    from .training_requirements import current_model_config

    run = Path(run).resolve()
    output = run / "visualizations"
    check_deadline(deadline)
    checkpoint = run / "best.zip"
    checkpoint_hash = file_hash(checkpoint)
    best = read_json(run / "best.json", {})
    if best.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Selected checkpoint hash does not match best.json")
    meta = read_json(run / "metadata.json")
    archive = not current_engine_config(meta["config"]) or not current_model_config(meta["config"])
    levels = (
        ["easy"]
        if meta["family"] in ("diagnostic", "placement", "saving")
        else cfg["evaluation"]["levels"]
    )
    seed = cfg["splits"]["validation"][0]
    existing = read_json(output / "demos.json", {})
    demos = existing.get("demos", [])
    reusable = (
        existing.get("checkpoint_hash") == checkpoint_hash
        and bool(demos)
        and all(
            (output / d["replay"]).is_file()
            and (not d.get("replay_hash") or d["replay_hash"] == file_hash(output / d["replay"]))
            for d in demos
        )
    )
    if archive:
        # Historical reports and recordings are readable without deserializing a model.
        if reusable:
            return demos
        raise RuntimeError(
            "Archived checkpoint cannot regenerate gameplay. Start a fresh run; "
            "existing replay files can still be watched/exported individually."
        )
    if (
        not reusable
        or {d["level"] for d in demos} != set(levels)
        or any(
            d.get("scenario_seed") != seed
            or not (output / d["replay"]).exists()
            or d.get("replay_hash") != file_hash(output / d["replay"])
            for d in demos
        )
    ):
        model, _ = load_policy(checkpoint)  # Exactly one model for every difficulty.
        root = output / "games"
        root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (root / f"attempt-{attempt}").exists():
            attempt += 1
        rows = evaluate(
            cfg,
            seeds=[seed],
            levels=levels,
            output=root / f"attempt-{attempt}",
            policy=model,
            condition=meta["condition"],
            learner_seed=meta["learner_seed"],
            family=meta["family"],
            record=True,
            replay_limit=1,
            training_steps=model.num_timesteps,
            progress=progress,
            checkpoint_hash=checkpoint_hash,
            deadline=deadline,
        )
        demos = []
        for row in rows:
            replay = Path(row["replay"])
            record = {
                **row,
                "outcome": row["status"],
                "seed": seed,
                "replay_hash": file_hash(replay),
            }
            demos.append({**record, "replay": replay.relative_to(output).as_posix()})
        write_json(output / "demos.json", {"checkpoint_hash": checkpoint_hash, "demos": demos})
    return demos


def export_demonstrations(run, demos, cfg, progress, deadline=None):
    output = Path(run).resolve() / "visualizations"
    for demo in demos:
        check_deadline(deadline)
        checkpoint_hash = demo["checkpoint_hash"]
        replay = output / demo["replay"]
        replay_hash = file_hash(replay)
        if demo.get("replay_hash") and demo["replay_hash"] != replay_hash:
            raise ValueError("Replay checksum differs from the saved demo manifest")
        destination = output / "videos" / f"{demo['level']}-{demo['scenario_seed']}.mp4"
        video_meta = read_json(destination.with_suffix(".video.json"), {})
        if (
            not destination.exists()
            or video_meta.get("checkpoint_hash") != checkpoint_hash
            or video_meta.get("video_hash") != file_hash(destination)
            or video_meta.get("replay_hash") != replay_hash
            or video_meta.get("settings") != video_settings(cfg)
        ):
            export_replay(
                output / demo["replay"],
                destination,
                cfg,
                context=demo,
                progress=progress,
                deadline=deadline,
            )
        demo["video"] = destination.relative_to(output).as_posix()
        write_json(output / "demos.json", {"checkpoint_hash": checkpoint_hash, "demos": demos})
    return demos


def visualize_run(run, *, cfg=None, videos=None, progress=None, deadline=None):
    """Rebuild derived artifacts. Failures are recorded separately from training."""
    from .training_requirements import current_model_config

    run = Path(run).resolve()
    cfg = cfg or read_json(run / "metadata.json")["config"]
    settings = output_settings(cfg)
    videos = settings["visualization"]["videos"] if videos is None else videos
    output = run / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    owns_progress = progress is None
    progress = progress or ProgressReporter(
        run / "visualize.log", settings["logging"]["progress_seconds"]
    )
    started = perf_counter()
    status = {"state": "exporting", "videos_requested": videos}
    write_json(output / "status.json", status)
    try:
        progress.phase(Phase.EXPORTING, "Generating training report")
        check_deadline(deadline)
        build_run_report(run, cfg)
        original = read_json(run / "metadata.json")["config"]
        archived = not current_engine_config(original) or not current_model_config(original)
        demos = []
        if archived:
            best = read_json(run / "best.json", {})
            demos = [
                d
                for d in read_json(output / "demos.json", {}).get("demos", [])
                if d.get("checkpoint_hash") == best.get("checkpoint_hash")
            ]
            status["note"] = (
                "Archived run: report and existing recordings only; fresh training required"
            )
            if videos and not demos:
                raise RuntimeError(
                    "No archived demonstrations to export; old checkpoints cannot regenerate gameplay"
                )
        elif (settings["visualization"]["demos"] or videos) and (run / "best.zip").exists():
            demos = create_demonstrations(run, cfg, progress, deadline=deadline)
        else:
            status["note"] = "Report only: no validated checkpoint or demos disabled"
        if videos and demos:
            export_demonstrations(run, demos, cfg, progress, deadline=deadline)
        status.update(state="complete", export_seconds=perf_counter() - started)
    except BudgetExpired as exc:
        status.update(state="pending", error=str(exc), export_seconds=perf_counter() - started)
        progress.emit("Presentation pending; regenerate with pvz-rl visualize", force=True)
    except Exception as exc:
        status.update(state="failed", error=repr(exc), export_seconds=perf_counter() - started)
        progress.emit(f"Visualization failed: {exc}; regenerate with pvz-rl visualize", force=True)
    except KeyboardInterrupt:
        status.update(state="interrupted", export_seconds=perf_counter() - started)
        raise
    finally:
        write_json(output / "status.json", status)
        try:
            if deadline is None or perf_counter() < deadline:
                build_run_report(run, cfg)
            else:
                pending_report(run, status)
        except Exception as exc:
            status.update(state="failed", report_error=repr(exc))
            write_json(output / "status.json", status)
            progress.emit(f"Report could not be written: {exc}", force=True)
        if owns_progress:
            progress.close()
    return status


def pending_report(run, status):
    """Update a lightweight status notice when plotting would exceed the budget."""
    output = Path(run) / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "index.html"
    notice = (
        '<aside id="pending-export"><p>Presentation pending: time allowance exhausted. '
        "Checkpoints and completed recordings are preserved.</p><pre>pvz-rl visualize --run "
        + html.escape(str(Path(run).resolve()))
        + "</pre></aside>"
    )
    if destination.exists():
        page = destination.read_text("utf-8")
        if 'id="pending-export"' not in page:
            page = page.replace("</body>", notice + "</body>")
    else:
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>PVZ results</title>'
            "<body><h1>PVZ training results</h1>" + notice + "</body></html>"
        )
    destination.write_text(page, encoding="utf-8")
