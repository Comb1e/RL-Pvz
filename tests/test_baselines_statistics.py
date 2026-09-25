import numpy as np
import pytest

from pvz_rl.envs.env import PvZEnv
from pvz_rl.evaluation.frozen_baseline import choose_action
from pvz_rl.evaluation.runner import evaluate
from pvz_rl.evaluation.statistics import bootstrap_interval, paired_difference, result_matrix


@pytest.mark.parametrize(
    "level,seed,outcome,tick",
    [
        ("easy", 0, "won", 15164),
        ("standard", 0, "won", 29471),
        ("standard", 1, "won", 28698),
        ("hard", 0, "won", 42278),
        ("hard", 4, "won", 42618),
        ("hard", 6, "won", 46546),
        ("easy", 42, "won", 14799),
        ("standard", 42, "won", 29363),
        ("hard", 42, "won", 47581),
    ],
)
def test_frozen_baseline_versioned_outcomes(cfg, level, seed, outcome, tick):
    env = PvZEnv(cfg, level=level)
    env.reset(seed=seed)
    while env.state == "running":
        action = choose_action(env.public_board())
        env.step(env.codec.encode(action))
    assert env.state == outcome and env.public.tick == tick
    assert env.metrics["invalid_actions"] == 0


def test_replay_capture_for_truncated_and_winning_cases(cfg, tmp_path):
    cfg["environment"]["cutoff_seconds"] = 1
    rows = evaluate(
        cfg,
        seeds=[0, 1],
        levels=["easy"],
        output=tmp_path / "cutoff",
        baseline="wait",
        split="development",
        record=True,
    )
    assert len(rows) == 2 and all(r["replay_verified"] for r in rows)
    assert all(r["status"] == "truncated" for r in rows)


def test_crossed_bootstrap_and_paired_difference_against_independent_control():
    rows = [
        {"learner_seed": run, "scenario_seed": case, "win": int(case >= run)}
        for run in range(3)
        for case in range(5)
    ]
    control = [{"learner_seed": None, "scenario_seed": case, "win": 0} for case in range(5)]
    matrix, _, _ = result_matrix(rows)
    expected = (5 + 4 + 3) / 15
    interval = bootstrap_interval(matrix, replicates=300, seed=42)
    assert interval["mean"] == pytest.approx(expected)
    assert interval["low"] <= expected <= interval["high"]
    assert paired_difference(rows, control, replicates=300, seed=42) == interval
    same = paired_difference(rows, rows, replicates=100)
    assert same["mean"] == same["low"] == same["high"] == 0
    assert bootstrap_interval(matrix, replicates=300, seed=42) == interval


def test_incomplete_duplicate_or_unpaired_results_fail():
    rows = [
        {"learner_seed": run, "scenario_seed": case, "win": 1} for run in (1, 2) for case in (5, 6)
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        result_matrix(rows + rows[:1])
    with pytest.raises(ValueError, match="Incomplete"):
        result_matrix(rows[:-1])
    reference = [{"learner_seed": None, "scenario_seed": 7, "win": 0}]
    with pytest.raises(ValueError, match="identical"):
        paired_difference(rows, reference)
    with pytest.raises(ValueError):
        bootstrap_interval(np.array([[np.nan]]))
