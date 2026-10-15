"""Bounded tuner candidate selection controls."""

from pvz_rl.config import _read_config
from tools.tune_gpu import tuning_candidates


def test_tuning_cases_include_unique_larger_stress_cases():
    tuning = _read_config(None)["tuning"]
    base = tuning_candidates(tuning)

    assert len(base) == 18
    assert len({tuple(sorted(candidate.items())) for candidate in base}) == len(base)
    assert {candidate["groups"] for candidate in base} >= {1, 2, 4, 8}
    assert {candidate["threads"] for candidate in base} >= {1, 2, 4}
