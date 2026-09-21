"""Comparison integrity with mismatched actual counts and shared case pairing."""

import copy

import pytest

from pvz_rl.config import digest
from pvz_rl.provenance import append_jsonl, write_json
from pvz_rl.sc2_experiments import comparison_report, sc2_profile


def evidence(root, *, second_seed_wins=True):
    for profile in ("A", "E"):
        for seed in (101, 102):
            run = root / f"{profile}-{seed}"
            cfg = sc2_profile(profile)
            engine = {"fixture": "same source and rules"}
            game_hash = digest(
                {"engine": engine, "environment": cfg["environment"], "reward": cfg["reward"]}
            )
            write_json(
                run / "metadata.json", {"config": cfg, "engine": engine, "learner_seed": seed}
            )
            steps = 10000 + seed
            games = 10000 + (8 if profile == "A" else 17)
            rows = [
                {
                    "family": "preset",
                    "level": level,
                    "scenario_seed": case,
                    "win": int(
                        profile == "E" and case % 2 == 0 and (seed == 101 or second_seed_wins)
                    ),
                    "profile": cfg["profile"],
                    "learner_seed": seed,
                    "split": "validation",
                    "game_protocol_hash": game_hash,
                }
                for level in cfg["evaluation"]["levels"]
                for case in range(100000, 100050)
            ]
            folder = run / "validation" / str(steps)
            folder.mkdir(parents=True)
            with (folder / "episodes.jsonl").open("w") as stream:
                for row in rows:
                    append_jsonl(stream, row)
            with (run / "learning-curve.jsonl").open("w") as stream:
                append_jsonl(
                    stream,
                    {
                        "training_steps": steps,
                        "training_games": games,
                        "macro_win_rate": sum(r["win"] for r in rows) / len(rows),
                        "levels": {},
                        "wall_seconds": 6000,
                        "nominal_games": 10000,
                    },
                )
            write_json(
                run / "status.json",
                {
                    "state": "complete",
                    "steps": steps,
                    "wall_seconds": 6010,
                    "stop_reason": "time_budget",
                    "budget_complete": False,
                },
            )
    write_json(root / "diagnostics.json", {str(seed): {"passed": True} for seed in (101, 102)})


def test_comparison_accepts_real_rollout_overshoot_and_reports_missing(tmp_path):
    evidence(tmp_path)
    result = comparison_report(tmp_path)
    assert len(result["rows"]) == 4
    assert len(result["missing"]) == 8
    assert result["comparisons"]["E"]["recommended_for_further_study"]
    interval = result["comparisons"]["E"]["paired_95_percent_interval"]
    assert 0 < interval[0] <= 0.5 <= interval[1] < 1
    assert (tmp_path / "comparison-curves.png").is_file()


def test_one_seed_without_improvement_is_not_recommended(tmp_path):
    evidence(tmp_path, second_seed_wins=False)
    result = comparison_report(tmp_path)
    assert not result["comparisons"]["E"]["recommended_for_further_study"]


def test_different_engine_pins_and_same_name_different_profiles_are_rejected(tmp_path):
    import json

    evidence(tmp_path)
    path = tmp_path / "E-102/metadata.json"
    original = json.loads(path.read_text())
    altered = copy.deepcopy(original)
    altered["config"]["engine_commit"] = "different"
    write_json(path, altered)
    with pytest.raises(ValueError, match="same engine"):
        comparison_report(tmp_path)
    altered = copy.deepcopy(original)
    altered["config"]["policy"]["channels"] = 32
    write_json(path, altered)
    with pytest.raises(ValueError, match="different agent profiles"):
        comparison_report(tmp_path)
