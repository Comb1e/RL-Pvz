"""Stage-aware report series; each metric has one display owner."""

from math import isfinite

import matplotlib.pyplot as plt

from pvz_rl.envs.tile_scenarios import LEGACY_SKILLS, is_bomb_skill
from pvz_rl.learning.curriculum import TILE_STAGES


def domain(row):
    skills = row.get("tile_scenarios") or {}
    stage = row.get("curriculum_stage")
    if stage in TILE_STAGES:
        return stage
    if skills.get("stage_order") is not None and skills.get("phase") in TILE_STAGES:
        return skills["phase"]
    return (
        "pre"
        if (
            row.get("curriculum_stage") == "pre"
            or row.get("active_skill") in TILE_STAGES
            or skills.get("phase") in TILE_STAGES
        )
        else "normal"
    )


def episodes(row):
    counts = (row.get("tile_scenarios") or {}).get("episodes", {})
    scope = domain(row)
    return counts.get(scope, 0) if scope in TILE_STAGES else sum(counts.values())


def phase(row):
    if domain(row) in TILE_STAGES:
        return domain(row)
    return (
        row.get("active_skill") or (row.get("tile_scenarios") or {}).get("phase")
        if domain(row) != "normal"
        else row.get("curriculum_stage", "normal")
    )


def value(row, path):
    for key in path.split("."):
        row = row.get(key) if isinstance(row, dict) else None
    return row if isinstance(row, (int, float)) and isfinite(row) else None


def plot_panels(
    segments, fields, xvalue, xlabel, output, name, save, *, rate_fields=(), legend_prefix=""
):
    """Omit absent panels and break lines at phase changes and missing values."""
    fields = [
        (key, title, get)
        for key, title, get in fields
        if any(
            get(row) is not None and xvalue(row) is not None for _, rows in segments for row in rows
        )
    ]
    if not fields:
        return False
    fig, axes = plt.subplots(
        (len(fields) + 1) // 2, 2, figsize=(12, 3 * ((len(fields) + 1) // 2)), squeeze=False
    )
    colors = dict(
        peashooter="#399471",
        cherry_bomb="#b54e59",
        cherry_bomb_obstacle="#9b69ad",
        easy="#399471",
        standard="#d69536",
        shared="#5379ac",
    )
    for ax, (key, title, get) in zip(axes.flat, fields):
        labels = set()
        for label, rows in segments:
            chunks, current, previous = [], [], None
            wall, stage = None, None
            for row in rows:
                kind, x, y = phase(row), xvalue(row), get(row)
                current_wall = value(row, "wall_seconds")
                resumed = wall is not None and current_wall is not None and current_wall < wall
                if (
                    kind != previous
                    or row.get("report_stage") != stage
                    or x is None
                    or y is None
                    or resumed
                ):
                    if current:
                        chunks.append((previous, current))
                    current = []
                if x is not None and y is not None:
                    current.append((x, y))
                previous = kind
                wall, stage = current_wall, row.get("report_stage")
            if current:
                chunks.append((previous, current))
            for kind, points in chunks:
                legend = legend_prefix + str(kind).replace("_", " ").title()
                if len(segments) > 1:
                    legend += f" ({label})"
                ax.plot(
                    *zip(*points),
                    marker=".",
                    markersize=4,
                    color=colors.get(str(kind).rsplit("/", 1)[-1]),
                    label=legend if legend not in labels else "_nolegend_",
                )
                labels.add(legend)
        ax.set(title=title, xlabel=xlabel)
        if key in rate_fields:
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
    for ax in list(axes.flat)[len(fields) :]:
        ax.set_visible(False)
    save(fig, output, name)
    return True


def report_charts(segments, cfg, output, save):
    images = []
    for scope in (*TILE_STAGES, "pre", "normal"):
        records = [
            (label, [r for r in series["training-metrics"] if domain(r) == scope])
            for label, series in segments
        ]
        if not any(rows for _, rows in records):
            continue
        pre = scope != "normal"
        title = (
            scope.replace("_", " ").title() + " stage" if scope in TILE_STAGES
            else "Legacy pre stage" if pre else "Normal training"
        )
        xvalue = (
            episodes
            if pre
            else lambda r: (
                r.get("training_games")
                if cfg["training"]["budget_unit"] == "games"
                else r.get("training_steps")
            )
        )
        xlabel = (
            "Completed scenario episodes"
            if pre
            else "Completed training games"
            if cfg["training"]["budget_unit"] == "games"
            else "Training decisions"
        )

        def fields(spec):
            return [(key, text, lambda row, key=key: value(row, key)) for key, text in spec]

        def add(group, heading, spec, rates=()):
            name = f"{scope}-{group}"
            if plot_panels(records, spec, xvalue, xlabel, output, name, save, rate_fields=rates):
                images.append((f"{title}: {heading}", name + ".png"))

        behavior = [
            (
                "rolling_win_rate",
                "Collection success (search-assisted where enabled)"
                if cfg["training"].get("search", {}).get("enabled", False)
                else "Rolling training success rate",
            ),
            ("rolling_return", "Episode reward"),
            ("rolling_seconds", "Episode duration (simulated seconds)"),
            ("invalid_action_rate", "Invalid-action rate"),
            ("rolling_agent_actions", "Accepted plant + dig actions / episode"),
            ("rolling_first_attacker_seconds", "First sustained attacker (seconds)"),
            ("rolling_maximum_sun", "Maximum sun / episode"),
        ]
        if cfg["training"]["gamma"] != 1:
            behavior.insert(
                2, ("rolling_discounted_return", "Simulation-time-discounted episode return")
            )
        if not pre:
            behavior += [
                ("rolling_attacker_purchases", "Sustained attackers purchased / episode"),
                ("rolling_plant_kills", "Plant kills / episode"),
                ("rolling_mower_kills", "Mower kills / episode"),
                ("early_digs_per_planting", "Early digs / accepted planting"),
            ]
        behavior_fields = fields(behavior)
        if pre:
            # A one-proposal bomb board has ranking supervision, not an episode-return objective.
            behavior_fields = [
                (
                    key,
                    text,
                    get
                    if key in ("rolling_win_rate", "invalid_action_rate")
                    else lambda row, get=get: get(row) if phase(row) == "peashooter" else None,
                )
                for key, text, get in behavior_fields
            ]
        add(
            "training",
            "training behavior",
            behavior_fields,
            ("rolling_win_rate", "invalid_action_rate"),
        )

        fitting = fields(
            [
                (
                    "optimization.q_loss",
                    "Teacher composite loss"
                    if cfg["training"].get("search", {}).get("enabled", False)
                    else "Balanced Q loss",
                ),
                ("optimization.selected_action_error", "Selected-action fitting error"),
                ("optimization.placement_gap", "Best-tile minus sampled-tile Q"),
                ("optimization.q_grad_norm", "Gradient norm before clipping"),
            ]
        )
        search_fields = fields(
            [
                ("search.queries", "Completed search queries"),
                ("search.rollouts", "Completed oracle rollouts (excluded from games)"),
                ("search.simulated_ticks", "Oracle simulator ticks"),
                ("search.wall_seconds", "Search wall time (seconds)"),
                ("search.scratch_reserved_bytes", "Scratch storage reservation (bytes)"),
                ("search.rollout_budget_fraction", "Current query rollout-budget utilization"),
                ("search.generation_failures", "World generation failures"),
                ("search.overrides", "Root choices differing from network"),
                ("search.continuation_actions", "Public-plan actions"),
                ("search.fallback_actions", "Plan fallback actions"),
                ("search.discovery_score", "Best discovery score"),
                ("search.confirmation_score", "Fresh confirmation score"),
                ("search.confirmation_duration_adjustment", "Chosen duration adjustment"),
                ("search.confirmation_standard_error", "Confirmation standard error (diagnostic)"),
                ("search.duration_reversals", "Preferences reversed by duration shaping"),
                ("optimization.teacher_value_loss", "Confirmed value calibration"),
                ("optimization.teacher_difference_loss", "Same-board difference loss"),
                ("optimization.teacher_feasibility_loss", "Finite-reference feasibility loss"),
            ]
        )
        search_fields = [
            (
                key,
                text,
                lambda row, get=get: (
                    get(row) if row.get("collection_policy") == "search_assisted" else None
                ),
            )
            for key, text, get in search_fields
        ]
        add("search", "search-assisted collection and confirmed teacher fitting", search_fields)
        if pre:
            fitting = [
                (
                    key,
                    text,
                    get
                    if key.endswith("q_grad_norm")
                    else lambda row, get=get: get(row) if phase(row) == "peashooter" else None,
                )
                for key, text, get in fitting
            ]
            fitting.insert(
                1,
                (
                    "ranking",
                    "Bomb total fitting loss",
                    lambda r: (
                        (
                            value(r, "optimization.tile_ranking_loss")
                            if value(r, "optimization.tile_ranking_loss") is not None
                            else value(r, "optimization.q_loss")
                        )
                        if is_bomb_skill(phase(r))
                        else None
                    ),
                ),
            )
            fitting += fields([
                ("optimization.bomb_hinge_loss", "Bomb ranking hinge"),
                ("optimization.bomb_distance_loss", "Bomb weighted expected distance"),
                ("cohort.bomb_selected_tile_distance", "Selected tile Manhattan distance"),
                ("tile_scenarios.generation_failures", "Scenario generation failures"),
            ])
        else:
            consistency_fields = fields(
                [
                    ("optimization.species_consistency_loss", "Species consistency loss"),
                    ("optimization.species_consistency_mae", "Species consistency absolute error"),
                    (
                        "optimization.species_consistency_saturation_fraction",
                        "Species consistency Huber saturation",
                    ),
                    (
                        "optimization.species_selection_evaluation_gap",
                        "Online/EMA tile-selection gap",
                    ),
                ]
            )
            fitting += [
                (
                    key,
                    text,
                    lambda row, get=get: (
                        get(row) if row.get("collection_policy") != "search_assisted" else None
                    ),
                )
                for key, text, get in consistency_fields
            ]
        for action in ("plant",) if pre else ("wait", "plant", "dig"):
            if any(
                (value(r, f"optimization.action_count_{action}") or 0) > 0
                for _, rows in records
                for r in rows
            ):
                fitting += fields(
                    [
                        (
                            f"optimization.value_target_error_{action}",
                            f"{action.title()}: pre-fit target error",
                        )
                    ]
                )
        add("fitting", "fitting", fitting)

        accounting = fields(
            [
                ("rolling_sky_income", "Sky sunlight / episode"),
                ("rolling_effective_damage", "Plant-caused HP + armor removed"),
                ("rolling_plant_value_loss", "Lost plant value (sun equivalents)"),
                ("rolling_action_blocking", "Action blocking value"),
            ]
            + (
                []
                if pre
                else [
                    ("rolling_produced_sun", "Sunflower income (sun)"),
                    ("rolling_mower_activation_penalty", "Mower activation cost / episode"),
                    ("rolling_development", "Net development reward"),
                    ("rolling_terminal", "Outcome reward"),
                ]
            )
        )
        if pre:
            accounting = [
                (key, text, lambda row, get=get: get(row) if phase(row) == "peashooter" else None)
                for key, text, get in accounting
            ]
        add("accounting", "reward accounting", accounting)
        add(
            "exploration",
            "exploration",
            [] if is_bomb_skill(scope) else fields(
                [
                    ("tile_exploration_epsilon", "Tile exploration probability"),
                    ("optimization.tile_exploration_coins", "Tile exploration draws / cohort"),
                    (
                        "optimization.exploratory_changes",
                        "Commands changed by exploration / cohort",
                    ),
                ]
                + (
                    []
                    if pre
                    else [("plant_exploration_epsilon", "Species exploration probability")]
                )
            ),
        )
        if not pre:
            add(
                "diagnostics",
                "additional diagnostics",
                fields(
                    [
                        ("rolling_nonlethal_health_damage", "Nonlethal damage / episode"),
                        ("rolling_empty_mower_activations", "Empty mower activations / episode"),
                        ("rolling_wall_nut_damage", "Wall-nut damage / episode"),
                        ("rolling_empty_explosions", "Empty explosions / episode"),
                        ("rolling_early_voluntary_digs", "Early voluntary digs / episode"),
                        ("rolling_cumulative_net_value", "Cumulative net value / episode"),
                        ("rolling_maximum_net_value", "Historical maximum net value / episode"),
                        ("rolling_value_drawdown", "Drawdown from maximum (diagnostic)"),
                        ("rolling_mower_expenditure", "Mower expenditure (sun equivalents)"),
                        ("simulation_ticks_per_second", "Simulation ticks / training second"),
                        (
                            "end_to_end_decisions_per_second",
                            "Transitions / wall second (includes reporting and validation)",
                        ),
                    ]
                ),
            )
            task_records = []
            tasks = {
                task
                for _, rows in records
                for row in rows
                for task in row.get("rolling_by_task", {})
            }
            if len(tasks) > 1:
                for label, rows in records:
                    for task in sorted(tasks):
                        task_records.append(
                            (
                                label,
                                [
                                    dict(
                                        row,
                                        curriculum_stage=task,
                                        report_stage=row.get("curriculum_stage"),
                                        task_values=row["rolling_by_task"][task],
                                    )
                                    for row in rows
                                    if task in row.get("rolling_by_task", {})
                                ],
                            )
                        )
                name = "normal-tasks"
                if plot_panels(
                    task_records,
                    fields(
                        [
                            ("task_values.win_rate", "Training success by task"),
                            ("task_values.simulated_seconds", "Episode seconds by task"),
                            (
                                "task_values.attacker_purchases_per_game",
                                "Attacker purchases by task",
                            ),
                            (
                                "task_values.early_digs_per_planting",
                                "Early digs / planting by task",
                            ),
                        ]
                    ),
                    xvalue,
                    xlabel,
                    output,
                    name,
                    save,
                    rate_fields=("task_values.win_rate",),
                ):
                    images.append(("Normal training: task-specific diagnostics", name + ".png"))

        validation = []
        if pre:
            for label, series in segments:
                rows = []
                for probe in series.get("tile-skill-probes", []):
                    stage = probe.get("stage", "pre")
                    if stage != scope:
                        continue
                    row = dict(
                        probe,
                        active_skill=probe.get("previous", "peashooter"),
                        curriculum_stage=stage,
                    )
                    for skill, wins in probe["scores"].items():
                        count = (
                            len(probe.get("cases", {}).get(skill, []))
                            or cfg["curriculum"]["probe_cases"]
                        )
                        row[skill] = wins / count
                    rows.append(row)
                validation.append((label, rows))
            spec = fields(
                [
                    (skill, skill.replace("_", " ").title() + " validation success")
                    for skill in (LEGACY_SKILLS if scope == "pre" else (scope,))
                ]
            )

            def vx(row):
                return row.get("stage_episodes", row["episodes"])
        else:
            for label, series in segments:
                rows = []
                for probe in series.get("curriculum-probes", []):
                    row = dict(probe, curriculum_stage=probe.get("stage", "normal"))
                    row.update(
                        {
                            task: wins / cfg["curriculum"]["probe_cases"]
                            for task, wins in probe["wins"].items()
                        }
                    )
                    rows.append(row)
                for evaluation in series["learning-curve"]:
                    row = dict(evaluation, curriculum_stage="reporting")
                    for task, score in evaluation.get("levels", {}).items():
                        row["evaluation_" + task.rsplit("/", 1)[-1]] = score["win_rate"]
                    rows.append(row)
                validation.append((label, sorted(rows, key=lambda r: r.get("training_steps", 0))))
            spec = fields(
                [
                    (key, heading)
                    for key, heading in (
                        ("easy", "Easy mastery probe"),
                        ("standard", "Standard mastery probe"),
                        ("hard", "Hard mastery probe"),
                        ("evaluation_easy", "Separate evaluation: easy"),
                        ("evaluation_standard", "Separate evaluation: standard"),
                        ("evaluation_hard", "Separate evaluation: hard"),
                    )
                ]
            )
            vx = xvalue
        name = f"{scope}-validation"
        if plot_panels(
            validation,
            spec,
            vx,
            xlabel,
            output,
            name,
            save,
            rate_fields=[k for k, _, _ in spec],
            legend_prefix="Training: " if pre else "",
        ):
            images.insert(0, (f"{title}: validation", name + ".png"))
    return images
