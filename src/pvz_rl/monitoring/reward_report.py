"""Read-only, cohort-scoped reward accounting from a run's saved configuration."""

import argparse
import json
import math
import os
import re
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import numpy as np

from pvz_rl.learning.objective import COMPONENTS, raw_rewards, training_rewards

LABELS = ("Terminal + time", "Asset changes/losses", "Damage", "Blocking", "Produced sun", "Mowers")


def read_json(path):
    """Read one atomically published version, retrying Windows sharing conflicts."""
    for attempt in range(4):
        try:
            return json.loads(path.read_bytes())
        except PermissionError:
            if attempt == 3:
                raise
            sleep(0.05 * (attempt + 1))


def snapshot_lines(path, notes):
    """Bound reads to the opened file's initial size; ignore an unfinished tail."""
    if not path.exists():
        notes.append(f"{path.name} is not available yet.")
        return
    with path.open("rb") as stream:
        limit = os.fstat(stream.fileno()).st_size
        line_number = 0
        while stream.tell() < limit:
            line = stream.readline(limit - stream.tell())
            line_number += 1
            if not line.endswith(b"\n"):
                notes.append(f"Ignored unfinished final line in {path.name}:{line_number}.")
                break
            if line.strip():
                yield line_number, line


def snapshot_rows(path, notes):
    for number, line in snapshot_lines(path, notes):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid complete JSON record at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{number}")
        yield number, row


def metric_cohort(row):
    return row.get("cohort", {}).get("objective", {}).get("cohort")


def select_records(folder, cohort, notes):
    metric, metric_line, recent = None, None, deque(maxlen=5)
    for number, row in snapshot_rows(folder / "training-metrics.jsonl", notes):
        identity = metric_cohort(row)
        if not isinstance(identity, int) or identity > cohort:
            continue
        if recent and metric_cohort(recent[-1]) == identity:
            recent.pop()
        recent.append(row)
        if identity == cohort:
            metric, metric_line = row, number
    episodes, duplicate_count = {}, 0
    for number, row in snapshot_rows(folder / "training-episodes.jsonl", notes):
        match = re.fullmatch(r"cohort:(\d+):slot:(\d+)", str(row.get("record_id", "")))
        if match is None:
            raise ValueError(f"Unrecognized episode record_id at training-episodes.jsonl:{number}")
        # Episode IDs use the pre-commit, zero-based cohort counter.
        if int(match[1]) + 1 != cohort:
            continue
        key = row["record_id"]
        if key in episodes:
            duplicate_count += 1
            if not episodes[key].get("time_reward_pending", True) and row.get(
                "time_reward_pending", True
            ):
                continue
        episodes[key] = row
    if duplicate_count:
        notes.append(
            f"Deduplicated {duplicate_count} episode IDs; finalized revisions take priority."
        )
    return metric, metric_line, list(episodes.values()), list(recent)


def warning_events(folder, cohort, notes):
    current, events = None, []
    for number, line in snapshot_lines(folder / "train.log", notes):
        text = line.decode("utf-8")
        match = re.search(r"\bcohort\s+(\d+)\b", text, re.IGNORECASE)
        if match:
            current = int(match[1])
        if current == cohort and "Warning:" in text:
            events.append(f"train.log:{number}: {text.strip()}")
    return events


def event_values(cfg, fitted_factor):
    reward = cfg["reward"]
    sun = reward.get("action", {}).get("sun_credit_multiplier", 1.0)
    extra = reward["early_sun_extra_multiplier"]
    return {
        "Plant damage equal to one basic zombie's body HP": reward["basic_zombie_value"]
        * fitted_factor,
        "25 ordinary Sunflower sun (asset + production)": 25 * (1 + sun) * fitted_factor,
        "25 eligible early Sunflower sun (both contributions)": 25
        * (1 + sun + extra)
        * fitted_factor,
        "Lose/remove a healthy 50-sun plant": -50
        * reward.get("asset_loss_multiplier", 1.0)
        * fitted_factor,
        "Lose/remove a healthy 100-sun plant": -100
        * reward.get("asset_loss_multiplier", 1.0)
        * fitted_factor,
        "One mower activates": -reward["mower_penalty"],
    }


def episode_statistics(episodes, cfg):
    missing = sorted({key for row in episodes for key in COMPONENTS if key not in row})
    if missing:
        raise ValueError(f"Episode records lack reward components: {', '.join(missing)}")
    values = np.asarray([[row[key] for key in COMPONENTS] for row in episodes], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Episode reward components contain non-finite values")
    # Use the production conversion on basis vectors instead of duplicating its economics.
    basis = np.eye(len(COMPONENTS), dtype=np.float64)
    raw = values * raw_rewards(basis, cfg)
    fitted = values * training_rewards(basis, cfg)
    total = training_rewards(values, cfg)
    action = fitted[:, 2:].sum(-1)  # The warning excludes global asset, but includes mowers.
    mean_abs_action = float(np.abs(action).mean())
    mean_abs_terminal = float(np.abs(fitted[:, 0]).mean())
    known_returns = [(i, row["return"]) for i, row in enumerate(episodes) if "return" in row]
    raw_error = (
        max(abs(raw[i].sum() - value) for i, value in known_returns) if known_returns else None
    )
    terminal = fitted[:, 0]
    return dict(
        ledger=values,
        raw=raw,
        fitted=fitted,
        totals=total,
        mean_absolute_action=mean_abs_action,
        mean_absolute_terminal=mean_abs_terminal,
        action_ratio=mean_abs_action / mean_abs_terminal if mean_abs_terminal else None,
        action_warning=mean_abs_action > 0.25 * mean_abs_terminal,
        net_physical_mean=float(fitted[:, 1:-1].sum(-1).mean()),
        raw_accounting_error=raw_error,
        raw_checked_rows=len(known_returns),
        time_adjustment=terminal
        - np.where(
            [bool(row["win"]) for row in episodes],
            cfg["reward"]["win_reward"],
            -cfg["reward"]["loss_penalty"],
        ),
    )


def build_report(folder, cohort):
    folder = Path(folder).resolve()
    if cohort < 1:
        raise ValueError("Cohort number must be positive and match the one-based training log.")
    cfg = read_json(folder / "config.json")
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain the saved configuration object.")
    if cfg.get("reward", {}).get("version") != "action_global_value_v5":
        raise ValueError(
            "This report requires the six-component action_global_value_v5 reward ledger."
        )
    if cfg["reward"]["value_scale"] <= 0:
        raise ValueError("Saved reward.value_scale must be positive.")
    scale = cfg["reward"]["progress_weight"] / cfg["reward"]["value_scale"]
    multiplier = cfg["training"].get("objective", {}).get("dense_multiplier", 1.0)
    if not math.isfinite(scale * multiplier) or scale < 0 or multiplier < 0:
        raise ValueError("Saved physical conversion factors must be finite and nonnegative.")
    notes = []
    metric, metric_line, episodes, recent = select_records(folder, cohort, notes)
    status_path = folder / "status.json"
    status = read_json(status_path) if status_path.exists() else {}
    summary = metric.get("cohort", {}) if metric else {}
    source = f"training-metrics.jsonl:{metric_line}" if metric else None
    if not metric and metric_cohort(status) == cohort:
        summary, source = status["cohort"], "status.json (matching cohort summary)"
    if not episodes and not summary:
        raise ValueError(
            f"Cohort {cohort} has no saved episode records or matching summary in {folder}."
        )
    expected = cfg["training"]["n_envs"]
    pending = sum(bool(row.get("time_reward_pending", True)) for row in episodes)
    complete = len(episodes) == expected and pending == 0
    if not complete:
        notes.append(
            f"Episode accounting is provisional: {len(episodes)}/{expected} records, "
            f"{pending} pending time ranking. Rerun after cohort finalization."
        )
    if source is None:
        notes.append("Return-variation/route/compaction metrics are not published yet.")
    return dict(
        folder=folder,
        cohort=cohort,
        cfg=cfg,
        scale=scale,
        multiplier=multiplier,
        events=event_values(cfg, scale * multiplier),
        episodes=episodes,
        complete=complete,
        expected=expected,
        statistics=episode_statistics(episodes, cfg) if episodes else None,
        summary=summary,
        metric=metric,
        source=source,
        recent=recent,
        warnings=warning_events(folder, cohort, notes),
        status=status,
        notes=notes,
    )


def number(value, *, signed=False):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:+.8g}" if signed else f"{value:.8g}"


def percent(value):
    return "n/a" if value is None else f"{100 * value:.4f}%"


def table(headers, rows):
    rows = [[str(value) for value in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def format_row(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))

    return "\n".join(
        [format_row(rows[0]), "-+-".join("-" * w for w in widths)]
        + [format_row(row) for row in rows[1:]]
    )


def render_report(report):
    cfg, episodes, stats = report["cfg"], report["episodes"], report["statistics"]
    summary, metric = report["summary"], report["metric"]
    objective = summary.get("objective", {})
    reward, training = cfg["reward"], cfg["training"]
    settings = training.get("objective", {})
    sections = [
        f"REWARD REPORT - COHORT {report['cohort']}",
        f"Run: {report['folder']}\nRead at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "Snapshot: saved config.json + bounded file reads; rerun to refresh an active run.\n"
        f"Episode records: {len(episodes)}/{report['expected']} "
        f"({'finalized' if report['complete'] else 'PROVISIONAL'})\n"
        f"Cohort metrics: {report['source'] or 'not published'}\n"
        f"Run state at read: {report['status'].get('state', 'unknown')}",
        "SAVED SETTINGS (current train.toml is not used)\n"
        + table(
            ("Setting", "Value"),
            [
                ("Reward / objective", f"{reward['version']} / {training.get('method')}"),
                (
                    "progress_weight / value_scale",
                    f"{reward['progress_weight']} / {reward['value_scale']}",
                ),
                ("Raw physical factor", number(report["scale"])),
                ("Dense multiplier", number(report["multiplier"])),
                ("Fitted physical factor", number(report["scale"] * report["multiplier"])),
                ("Basic zombie damage value", number(reward["basic_zombie_value"])),
                ("Plant asset-loss multiplier", number(reward.get("asset_loss_multiplier", 1.0))),
                ("Mower penalty (final units)", number(-reward["mower_penalty"], signed=True)),
                (
                    "Early sun extra multiplier / spawn fraction",
                    f"{reward['early_sun_extra_multiplier']} / {reward['early_sun_spawn_fraction']}",
                ),
                (
                    "Win / loss base / max time adjustment",
                    f"+{reward['win_reward']} / -{reward['loss_penalty']} / {reward['win_time_weight']}",
                ),
                (
                    "Action Hz / decision ticks",
                    f"{cfg['environment'].get('action_frequency_hz', 'n/a')} / {cfg['environment'].get('decision_ticks', 'n/a')}",
                ),
                ("Compaction window", str(settings.get("invalid_fit_group_size", "n/a"))),
                (
                    "Invalid minimum-Q offset (fitting only)",
                    str(settings.get("invalid_action_q_offset", "n/a")),
                ),
            ],
        ),
        "EVENT VALUES (fitted units; examples, not observed event counts)\n"
        + table(
            ("Event", "Reward"), [(k, number(v, signed=True)) for k, v in report["events"].items()]
        )
        + "\nSun examples assume no simultaneous plant loss; sky sun is excluded.\n"
        "Placement exchanges sun for plant assets and is asset-neutral. Early action income\n"
        "uses (sun_credit_multiplier + extra); its separate asset contribution remains +1.",
    ]
    if stats is not None:
        sections.insert(
            3,
            "COHORT SUMMARY\n"
            f"Wins: {sum(bool(row['win']) for row in episodes)}/{len(episodes)}; "
            f"mean fitted reward: {number(stats['totals'].mean(), signed=True)}\n"
            f"Action-shaping ratio: {percent(stats['action_ratio'])} "
            f"({'WARNING' if stats['action_warning'] else 'below/equal 25%'}); "
            f"dense-return ratio: {percent(objective.get('dense_to_target_std_ratio'))}\n"
            "The action warning includes mowers; the dense warning excludes them.",
        )
        values, raw, fitted = stats["ledger"], stats["raw"], stats["fitted"]
        rows = []
        for i, label in enumerate(LABELS):
            rows.append(
                (
                    label,
                    number(values[:, i].mean(), signed=True),
                    number(raw[:, i].mean(), signed=True),
                    number(fitted[:, i].mean(), signed=True),
                    number(fitted[:, i].std()),
                    number(fitted[:, i].min(), signed=True),
                    number(fitted[:, i].max(), signed=True),
                )
            )
        rows.append(
            (
                "TOTAL",
                "mixed units",
                number(raw.sum(-1).mean(), signed=True),
                number(stats["totals"].mean(), signed=True),
                number(stats["totals"].std()),
                number(stats["totals"].min(), signed=True),
                number(stats["totals"].max(), signed=True),
            )
        )
        sections.append(
            "EPISODE REWARD BREAKDOWN\n"
            + table(
                (
                    "Component",
                    "Ledger mean",
                    "Raw mean",
                    "Fitted mean",
                    "Fitted std",
                    "Fitted min",
                    "Fitted max",
                ),
                rows,
            )
            + "\nLedger physical terms use sun-equivalent units; terminal/mowers already use final units.\n"
            "Raw/fitted columns are reward units. Std/min/max describe EPISODE TOTALS,\n"
            "not the per-decision return distribution used by the dense warning."
        )
        sections.append(
            "ACTION-SHAPING WARNING (strictly greater than 25%)\n"
            "A = fitted damage + blocking + sun + mower, excluding asset changes.\n"
            "Compare mean(abs(A)) against 0.25 * mean(abs(finalized terminal)).\n"
            + table(
                ("Quantity", "Value"),
                [
                    ("Mean absolute action shaping", number(stats["mean_absolute_action"])),
                    ("Mean absolute terminal", number(stats["mean_absolute_terminal"])),
                    ("25% terminal threshold", number(0.25 * stats["mean_absolute_terminal"])),
                    ("Action / terminal ratio", percent(stats["action_ratio"])),
                    (
                        "Warning reconstructed"
                        if report["complete"]
                        else "Provisional sample warning",
                        str(stats["action_warning"]),
                    ),
                    ("Mean mower contribution", number(fitted[:, -1].mean(), signed=True)),
                    (
                        "Mean damage + blocking + sun",
                        number(fitted[:, 2:-1].sum(-1).mean(), signed=True),
                    ),
                    (
                        "Net physical mean (asset included, mower excluded)",
                        number(stats["net_physical_mean"], signed=True),
                    ),
                ],
            )
            + "\nAbsolute value is applied PER EPISODE before averaging; abs(mean(A)) differs\n"
            "when signs vary. A zero terminal denominator has an undefined ratio; the\n"
            "warning still fires when mean(abs(A)) > 0. Neither warning changes fitting."
        )
        outcomes = []
        for won, label in ((True, "Win"), (False, "Loss")):
            ix = np.asarray([bool(row["win"]) == won for row in episodes])
            if ix.any():
                outcomes.append(
                    (
                        label,
                        int(ix.sum()),
                        number(stats["totals"][ix].mean(), signed=True),
                        number(fitted[ix, 0].mean(), signed=True),
                        number(fitted[ix, -1].mean(), signed=True),
                    )
                )
        sections.append(
            "OUTCOMES\n"
            + table(("Outcome", "Games", "Mean fitted", "Mean terminal", "Mean mower"), outcomes)
        )
        checks = [
            ("Raw reward accounting max abs error", number(stats["raw_accounting_error"])),
            ("Raw returns checked", str(stats["raw_checked_rows"])),
        ]
        if metric and report["complete"]:
            recorded = metric.get("current_cohort", {}).get("training_return")
            checks.append(
                (
                    "Saved vs reconstructed fitted mean abs error",
                    number(
                        abs(recorded - stats["totals"].mean()) if recorded is not None else None
                    ),
                )
            )
        sections.append("ACCOUNTING CHECKS\n" + table(("Check", "Value"), checks))
        facts = []
        for key, label in (
            ("decisions", "Executed decisions"),
            ("agent_actions", "Accepted non-wait actions"),
            ("wait_actions", "Executed waits (includes rejected proposals)"),
            ("invalid_actions", "Rejected proposals"),
            ("produced_sun", "Actual produced sun"),
            ("early_sun", "Eligible early sun"),
            ("effective_damage", "Effective plant/projectile damage"),
            ("mowers_used", "Mowers used"),
            ("simulated_seconds", "Game duration (seconds)"),
        ):
            if all(key in row for row in episodes):
                sample = np.asarray([row[key] for row in episodes], dtype=np.float64)
                facts.append(
                    (
                        label,
                        number(sample.sum()),
                        number(sample.mean()),
                        number(sample.min()),
                        number(sample.max()),
                    )
                )
        facts.append(
            (
                "Applied time adjustment",
                number(stats["time_adjustment"].sum(), signed=True),
                number(stats["time_adjustment"].mean(), signed=True),
                number(stats["time_adjustment"].min(), signed=True),
                number(stats["time_adjustment"].max(), signed=True),
            )
        )
        sections.append(
            "OBSERVED EPISODE FACTS\n" + table(("Fact", "Total", "Mean", "Min", "Max"), facts)
        )
        plants = Counter()
        for row in episodes:
            plants.update(row.get("plant_usage", {}))
        sections.append(
            "ACCEPTED PLANT USAGE\n"
            + (
                table(("Plant", "Count"), [(k, number(v)) for k, v in plants.most_common()])
                if plants
                else "No plant purchases recorded."
            )
        )
    sections.append(
        "DENSE-VARIATION WARNING (less than 5%; all original decisions before compaction)\n"
        "Dense = asset + damage + blocking + sun in fitted units, excluding mowers/terminal.\n"
        + table(
            ("Quantity", "Value"),
            [
                ("Dense return std", number(objective.get("dense_return_std"))),
                ("Complete-return std", number(objective.get("target_std"))),
                ("Dense / complete std", percent(objective.get("dense_to_target_std_ratio"))),
                (
                    "Recorded low-variation status",
                    str(objective.get("low_dense_return_variation", "n/a")),
                ),
            ],
        )
        + "\nThese are logged per-decision statistics, not recomputed from episode totals.\n"
        "They include covariance and do not measure Q-value or fitting-loss variation."
    )
    counts = objective.get("outcome_counts")
    if counts is not None:
        sections.append(
            "PROPOSED ACTIONS (rejected proposals execute Wait)\n"
            + table(
                ("Kind", "Accepted", "Rejected", "Total"),
                [
                    (label, number(pair[1]), number(pair[0]), number(sum(pair)))
                    for label, pair in zip(("Wait", "Plant", "Dig"), counts, strict=True)
                ],
            )
        )
    workload = objective.get("routed_workload", {})
    if workload:
        sections.append(
            "RAW FITTING ROUTES"
            + "\n"
            + table(
                ("Route / forward rows", "Count"),
                [
                    (k, number(v))
                    for k, v in {
                        **workload.get("routes", {}),
                        **{k: v for k, v in workload.items() if k != "routes"},
                    }.items()
                ],
            )
        )
    for key, title in (
        ("fitting_compaction", "FITTING COMPACTION"),
        ("fitting_workloads", "RETAINED FITTING WORKLOAD"),
    ):
        data = summary.get(key, {})
        if data:
            flat = {**data.get("routes", {}), **{k: v for k, v in data.items() if k != "routes"}}
            sections.append(
                title
                + "\n"
                + table(
                    ("Quantity", "Value"),
                    [
                        (k, percent(v) if k.endswith("reduction") else number(v))
                        for k, v in flat.items()
                    ],
                )
            )
    if report["recent"]:
        rows = []
        for row in report["recent"]:
            c, current = row["cohort"], row.get("current_cohort", {})
            obj = c["objective"]
            rows.append(
                (
                    obj["cohort"],
                    f"{current.get('wins', '?')}/{current.get('games', '?')}",
                    number(current.get("training_return"), signed=True),
                    percent(obj.get("dense_to_target_std_ratio")),
                    number(c.get("collection_seconds")),
                    number(c.get("fit_seconds")),
                )
            )
        sections.append(
            "RECENT COMMITTED COHORTS THROUGH SELECTION\n"
            + table(("Cohort", "Wins", "Mean fitted", "Dense ratio", "Collect s", "Fit s"), rows)
        )
    sections.append(
        "LOGGED WARNING EVENTS FOR THIS COHORT\n"
        + ("\n".join(report["warnings"]) or "None found in the log snapshot.")
    )
    sections.append(
        "LIMITS\n"
        + "\n".join(f"- {note}" for note in report["notes"])
        + ("\n" if report["notes"] else "")
        + "- Files are individually snapshotted; they are not one atomic multi-file snapshot.\n- Empty/in-progress metrics remain n/a; episode means cannot reconstruct return std.\n- Invalid minimum-Q supervision is fitting-only and is absent from reward totals.\n- This report reads logs only; it does not load checkpoints or start a simulator/GPU run."
    )
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_folder", type=Path, help="Run directory containing config.json and training logs"
    )
    parser.add_argument(
        "cohort",
        type=int,
        help="One-based cohort number, as displayed in train.log (not a count of recent cohorts)",
    )
    args = parser.parse_args()
    try:
        print(render_report(build_report(args.run_folder, args.cohort)), end="")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        parser.exit(2, f"Reward report failed: {exc}\n")


if __name__ == "__main__":
    main()
