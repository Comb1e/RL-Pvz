import json

import pytest

from pvz_rl.config import validate_config
from pvz_rl.reporting import make_report


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("encoding", "count_scale", 0),
        ("reward", "economy_scale", 0),
        ("training", "learner_seeds", [101, 101]),
        ("training", "hidden_sizes", [0]),
        ("training", "learning_rate", float("nan")),
        ("reward", "gamma", float("nan")),
    ],
)
def test_invalid_research_config_rejected(cfg, group, key, value):
    cfg[group][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_report_rejects_incompatible_game_protocols(cfg, tmp_path):
    rows = [{"protocol_hash": "A"}, {"protocol_hash": "B"}]
    source = tmp_path / "bad.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="same verified"):
        make_report([source], tmp_path / "report", cfg)


def test_report_rejects_mixed_training_budgets(cfg, tmp_path):
    rows = [
        {
            "protocol_hash": "same",
            "training_config_hash": str(seed),
            "split": "test",
            "family": "preset",
            "level": "easy",
            "policy": "masked",
            "learner_seed": seed,
            "scenario_seed": 200000,
            "win": 1,
        }
        for seed in (101, 102)
    ]
    source = tmp_path / "bad.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="pool different"):
        make_report([source], tmp_path / "report", cfg)
