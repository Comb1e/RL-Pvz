"""Offline run reports and fixed-case demos from one shared best checkpoint."""

from __future__ import annotations

import html
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pvz_rl.config import output_settings
from pvz_rl.evaluation.runner import evaluate
from pvz_rl.learning.curriculum import TILE_STAGES, pre_stage
from pvz_rl.learning.deadline import BudgetExpired, check_deadline
from pvz_rl.learning.tile_curriculum import required_validation_wins, tile_validation_count
from pvz_rl.monitoring.progress import Phase, ProgressReporter
from pvz_rl.presentation.forecast_display import fully_correct_text
from pvz_rl.presentation.training_charts import domain, episodes, report_charts
from pvz_rl.presentation.video import export_replay, video_settings
from pvz_rl.provenance import current_engine_config, file_hash, write_json


def score_text(scores, cfg):
    if not scores:
        return "not recorded"
    parts = []
    for skill, wins in scores.items():
        total = tile_validation_count(cfg, skill)
        if skill == "prediction":
            parts.append(fully_correct_text(wins, total) or f"{wins}/{total}")
        else:
            parts.append(f"{skill} {wins}/{total}")
    return ", ".join(parts)


def cohort_result_text(check):
    cohort = check.get("cohort") or {}
    domain_name = cohort.get("domain") or check.get("domain")
    if domain_name == "prediction" or cohort.get("skill") == "prediction":
        rate = fully_correct_text(
            cohort.get("fully_correct_scenes"), cohort.get("forecast_scene_count")
        )
        if rate:
            return rate
        return f"{cohort.get('games', 0)} scenes"
    return f"{cohort.get('wins', 0)}/{cohort.get('games', 0)}"


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def read_series(path, *, tolerate_partial_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        lines = stream.readlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not (tolerate_partial_tail and index == len(lines) - 1 and not line.endswith("\n")):
                raise
    return rows


def run_segments(run):
    """Only link ancestors with a recorded resume boundary; label times separately."""
    result, seen, cutoff, recorded = [], set(), None, {}
    current = Path(run).resolve()
    while current not in seen:
        seen.add(current)
        meta = read_json(current / "metadata.json", {})
        series = {}
        for name in (
            "training-metrics",
            "learning-curve",
            "tile-skill-probes",
            "curriculum-probes",
            "validation-checks",
        ):
            rows = read_series(current / f"{name}.jsonl", tolerate_partial_tail=True)
            unique = {}
            for row in rows:
                if "training_steps" not in row:
                    # Historical tile probes recorded scenario episodes and optimizer commits.
                    match = next(
                        (
                            r
                            for r in series.get("training-metrics", [])
                            if episodes(r) == row.get("episodes")
                        ),
                        None,
                    )
                    if match:
                        row = dict(row, training_steps=match["training_steps"])
                step = row.get("training_steps")
                if cutoff is not None and (step is None or step > cutoff):
                    continue
                identity = (
                    step,
                    row.get("completed_updates"),
                    row.get("optimizer_step"),
                    row.get("episodes"),
                    row.get("stage"),
                    row.get("kind"),
                )
                unique[identity] = row
            used = recorded.setdefault(name, set())
            series[name] = [row for key, row in unique.items() if key not in used]
            used.update(unique)
        boundary = meta.get("resume_steps")
        if (
            meta.get("resume")
            and boundary is not None
            and Path(meta["resume"]).resolve().parent == current
        ):
            for newer in (True, False):
                part = {
                    name: [
                        row for row in rows if (row.get("training_steps", 0) > boundary) == newer
                    ]
                    for name, rows in series.items()
                }
                if any(part.values()):
                    result.append(
                        (current.name + (" / resumed" if newer else " / before resume"), part)
                    )
            break
        result.append((current.name, series))
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


def build_run_report(run, cfg=None):
    run = Path(run).resolve()
    meta = read_json(run / "metadata.json", {})
    cfg = cfg or meta["config"]
    output = run / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    segments = run_segments(run)
    images = report_charts(segments, cfg, output, _save)
    metrics = [row for _, series in segments for row in series["training-metrics"]]

    status = {**(metrics[-1] if metrics else {}), **read_json(run / "status.json", {})}
    best = read_json(run / "best.json", {})
    visual_status = read_json(output / "status.json", {})
    demos = read_json(output / "demos.json", {}).get("demos", [])
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
            f"{'compact demo' if demo['replay'].endswith('.pvzdemo') else 'native replay'}</a></p>"
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
    charts = ""
    for title, name in images:
        plot = f'<a href="{name}"><img src="{name}" alt="{title}" loading="lazy"></a>'
        if any(group in name for group in ("accounting", "exploration", "diagnostics", "tasks")):
            charts += f"<section><details><summary>{title}</summary>{plot}</details></section>"
        else:
            charts += f"<section><h2>{title}</h2>{plot}</section>"
    policy_description = (
        "Restricted diagnostic policy; this run does not evaluate the full game"
        if meta.get("family") in ("diagnostic",)
        else "Tile-skill training stage"
        if pre_stage(cfg)
        else "One shared board encoder and conditional policy across all training stages"
    )
    completion = (
        "Curriculum incomplete" if status.get("curriculum_incomplete") else "Curriculum status"
    )
    normal_progress = (
        f"<p>Completed games: {escape(status.get('training_games', 'not recorded'))}. "
        f"{completion}: {escape(status.get('curriculum_stage', 'not recorded'))}. "
        f"Stop reason: {escape(status.get('stop_reason') or 'not recorded / in progress')}. "
        + (
            "Until selected-stage mastery; no training time or game ceiling.</p>"
            if cfg["training"].get("until_stage_complete", False)
            else f"Game target reached: {escape(status.get('budget_complete', 'in progress'))}.</p>"
        )
    )
    skills = status.get("tile_scenarios") or {}
    has_pre = (
        any(domain(row) != "normal" for row in metrics)
        or pre_stage(cfg)
        or skills.get("phase") in TILE_STAGES
    )
    mastered = ", ".join(
        skill
        for skill in skills.get("stage_order") or []
        if skills.get("scores", {}).get(skill, -1) >= required_validation_wins(cfg, skill)
    )
    mastery = (
        f"Joint mastery: {escape(skills.get('phase') == 'complete')}. "
        if skills.get("stage_order") is None
        else f"Completed stages: {escape(mastered or 'none')}. "
    )
    pre_progress = (
        (
            f"<p>Tile stage: <b>{escape(status.get('curriculum_stage', skills.get('phase', 'not recorded')))}</b>. "
            f"Completed scenario episodes: {escape(sum(skills.get('episodes', {}).values()))}. "
            f"Peashooter: {escape(skills.get('episodes', {}).get('peashooter', 0))}; "
            f"Cherry Bomb: {escape(skills.get('episodes', {}).get('cherry_bomb', 0))}; "
            f"Obstacle: {escape(skills.get('episodes', {}).get('cherry_bomb_obstacle', 0))}; "
            f"Attacker pre: {escape(skills.get('episodes', {}).get('attacker_pre', 0))}; "
            f"Prediction: {escape(skills.get('episodes', {}).get('prediction', 0))}; "
            f"Attacker: {escape(skills.get('episodes', {}).get('attacker', 0))}. "
            f"Generation failures: {escape(skills.get('generation_failures', 0))}. "
            + mastery
            + f"Validation: {escape(score_text(skills.get('scores'), cfg))}. "
            f"Stop reason: {escape(status.get('stop_reason') or status.get('state', 'in progress'))}.</p>"
        )
        if has_pre
        else ""
    )
    run_progress = pre_progress + (
        normal_progress if any(domain(r) == "normal" for r in metrics) or not has_pre else ""
    )
    trigger = status.get("validation_trigger", {})
    if trigger.get("enabled"):
        pending = trigger.get("pending") or "—"
        threshold = (
            ""
            if skills.get("phase") == "prediction" or status.get("curriculum_stage") == "prediction"
            else f"{trigger['cohort_success_threshold']:.0%} · "
        )
        run_progress += (
            f"<p>{threshold}next {escape(trigger.get('next_check'))} · {escape(pending)}</p>"
        )
    checks = [row for _, series in segments for row in series.get("validation-checks", [])]
    if checks:
        run_progress += "<details><summary>Validation trigger history</summary><table><tr><th>Stage / skill</th><th>Episodes</th><th>Cohort success</th><th>Decision</th></tr>"
        for check in checks:
            cohort = check["cohort"]
            run_progress += (
                f"<tr><td>{escape(cohort.get('skill') or check['domain'])}</td>"
                f"<td>{check['progress']}</td><td>{escape(cohort_result_text(check))}</td>"
                f"<td>{'Launch' if check['eligible'] else 'Skip'} ({escape(check['reason'])})</td></tr>"
            )
        run_progress += "</table></details>"
    probes = [row for _, series in segments for row in series.get("tile-skill-probes", [])]
    if probes:
        run_progress += (
            "<details><summary>Tile-stage validation and transitions</summary><table><tr>"
            "<th>Stage</th><th>Completed episodes</th>"
            + "".join(f"<th>{escape(skill)}</th>" for skill in TILE_STAGES)
            + "<th>Training stage</th></tr>"
        )
        for probe in probes:
            counts = {skill: len(rows) for skill, rows in probe.get("cases", {}).items()}
            breakdown = "".join(
                f"<br>{escape(skill)} {score['wins']}/{score['games']}"
                for skill, score in probe.get("exercise_scores", {}).items()
            )
            run_progress += (
                f"<tr><td>{escape(probe.get('stage', 'pre'))}</td>"
                f"<td>{probe.get('stage_episodes', probe['episodes'])}</td>"
                + "".join(
                    (
                        f"<td>{escape(fully_correct_text(probe['scores'][skill], counts.get(skill) or tile_validation_count(cfg, skill)) or '—')}</td>"
                        if skill == "prediction"
                        else f"<td>{probe['scores'][skill]}/{counts.get(skill, cfg['curriculum']['probe_cases'])}"
                        f"{breakdown if skill == 'attacker' else ''}</td>"
                    )
                    if skill in probe["scores"]
                    else "<td>—</td>"
                    for skill in TILE_STAGES
                )
                + f"<td>{escape(probe.get('previous'))} → {escape(probe.get('next'))}</td></tr>"
            )
        run_progress += "</table></details>"
    if status.get("selected_stage"):
        run_progress += (
            f"<p>Single stage: {escape(status['selected_stage'])}. "
            f"Mastery reached: {escape(status.get('stage_mastered', False))}. "
            "Use final.zip to initialize another stage with a fresh budget.</p>"
        )
    if status.get("validation_schedule") == "stage_success":
        run_progress += (
            "<p>Normal-game checkpoint evaluation runs only after each curriculum stage passes. "
            "A game/time limit does not trigger an extra evaluation. "
            "Mastery probes remain separate; final.zip is saved even without a validated best.zip.</p>"
        )
        if status.get("pending_stage_validation"):
            run_progress += (
                "<p>Stage-success evaluation pending: "
                + escape(status["pending_stage_validation"]["stage"])
                + ". Resume its checkpoint to finish evaluation.</p>"
            )
    counts = status.get("task_counts", {})
    task_table = ""
    if counts:
        task_table = "<h2>Training data composition</h2><table><tr><th>Task</th><th>Started</th><th>Active</th><th>Completed</th><th>Transitions</th></tr>"
        for task, counts in sorted(counts.items()):
            task_table += (
                f"<tr><td>{escape(task)}</td>"
                + "".join(
                    f"<td>{escape(counts.get(key, 'unavailable'))}</td>"
                    for key in ("started_games", "active_games", "completed_games", "transitions")
                )
                + "</tr>"
            )
        task_table += "</table><p>Configured mixtures select episodes, not equal numbers of transitions. Restarts add new starts.</p>"
        task_table += (
            "<details><summary>Accepted plant purchases within task windows</summary><pre>"
            + escape(
                json.dumps(
                    {
                        k: v.get("plant_usage", {})
                        for k, v in status.get("rolling_by_task", {}).items()
                    },
                    indent=2,
                )
            )
            + "</pre></details>"
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
{task_table}
<details><summary>Settings, checkpoint, and export status</summary>
<pre>{escape(json.dumps(details, indent=2))}</pre></details></header>{charts}
<section><h2>Shared checkpoint demonstrations</h2>
<p>Each full recording uses identical weights on a predetermined validation case.
Diagnostic runs show only the diagnostic task. A short smoke run may end at its configured cutoff.</p>
{"".join(cards) or "<p>No demonstrations generated yet.</p>"}</section></body></html>"""
    temporary = output / "index.tmp.html"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output / "index.html")
    for retired in (
        "validation-curves",
        "training-curves",
        "accounting-curves",
        "optimization-curves",
        "learning-diagnostics",
        "task-curves",
    ):
        (output / f"{retired}.png").unlink(missing_ok=True)
    return output / "index.html"


def create_demonstrations(run, cfg, progress, deadline=None):
    from pvz_rl.learning.training import load_policy
    from pvz_rl.learning.training_requirements import current_model_config

    run = Path(run).resolve()
    output = run / "visualizations"
    check_deadline(deadline)
    checkpoint = run / "best.zip"
    checkpoint_hash = file_hash(checkpoint)
    best = read_json(run / "best.json", {})
    if best.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Selected checkpoint hash does not match best.json")
    meta = read_json(run / "metadata.json")
    if not current_engine_config(meta["config"]) or not current_model_config(meta["config"]):
        raise ValueError("Demonstration generation requires the current engine and model protocol")
    levels = ["easy"] if meta["family"] in ("diagnostic",) else cfg["evaluation"]["levels"]
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


def visualize_run(run, *, cfg=None, videos=None, progress=None, deadline=None, report_only=False):
    """Rebuild derived artifacts. Failures are recorded separately from training."""
    from pvz_rl.learning.training_requirements import current_model_config

    run = Path(run).resolve()
    cfg = cfg or read_json(run / "metadata.json")["config"]
    settings = output_settings(cfg)
    videos = settings["visualization"]["videos"] if videos is None else videos
    output = run / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    owns_progress = progress is None
    progress = progress or ProgressReporter.from_settings(run / "visualize.log", cfg)
    started = perf_counter()
    status = {"state": "exporting", "videos_requested": videos}
    write_json(output / "status.json", status)
    try:
        progress.phase(Phase.EXPORTING, "Generating training report")
        check_deadline(deadline)
        build_run_report(run, cfg)
        original = read_json(run / "metadata.json")["config"]
        if not current_engine_config(original) or not current_model_config(original):
            raise ValueError("Run visualization requires the current engine and model protocol")
        demos = []
        if report_only:
            status["note"] = "Report only; checkpoints and demonstrations were not loaded"
        elif videos and not (run / "best.zip").exists():
            best = read_json(run / "best.json", {})
            demos = [
                d
                for d in read_json(output / "demos.json", {}).get("demos", [])
                if d.get("checkpoint_hash") == best.get("checkpoint_hash")
            ]
            status["note"] = (
                "Exporting existing current-protocol recordings without loading a model"
            )
            if not demos:
                raise RuntimeError("No validated checkpoint or matching recordings to export")
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
