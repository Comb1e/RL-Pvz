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


def test_shared_benchmark_routes_preserve_all_width_cases():
    import numpy as np

    from pvz_rl.config import load_config
    from pvz_rl.learning.action_routing import route_indices, workload_counts
    from pvz_rl.learning.cuda_buffer import trajectory_dtype
    from pvz_rl.monitoring.fit_benchmark import ACTION_MIXES, WORKLOAD_WIDTHS, benchmark_rows

    cfg = load_config()
    assert set(ACTION_MIXES) >= {
        "wait",
        "mixed",
        "tile",
        "species_invalid",
        "tile_invalid",
        "rejected_heavy",
    }
    for widths in WORKLOAD_WIDTHS.values():
        for mix in ACTION_MIXES:
            rows, boards = benchmark_rows(cfg, widths, 2, mix, trajectory_dtype())
            counts = workload_counts(rows)
            plans = route_indices(rows)
            assert len(rows) == len(boards) == 2 * len(widths)
            assert counts["ema_tile_rows"] == np.count_nonzero(
                rows["accepted"] & (rows["action"] != 0)
            )
            assert counts["online_tile_rows"] == len(plans["tiles"])
            if mix == "species_invalid":
                assert counts["online_tile_rows"] == counts["ema_tile_rows"] == 0
            if mix == "tile_invalid":
                assert counts["online_species_rows"] == counts["ema_tile_rows"] == 0
