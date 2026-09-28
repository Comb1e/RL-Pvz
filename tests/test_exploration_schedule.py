"""Tile-only exponential schedule, frozen cohort values and stage reset."""

import pytest

from pvz_rl.config import load_config
from pvz_rl.learning.exploration import exploration_state


@pytest.mark.parametrize(
    "games,epsilon",
    [(0, 0.5), (2500, (0.5 * 0.01) ** 0.5), (5000, 0.01), (9000, 0.01)],
)
def test_schedule_values_and_floor(games, epsilon):
    cfg = load_config()
    s = exploration_state(cfg, games)
    assert s.epsilon == pytest.approx(epsilon)
    assert s.tile_epsilon == pytest.approx(epsilon)
    assert s.at_floor == (games >= 5000)
    assert exploration_state(cfg, games, staged=False) == s
    assert exploration_state(cfg, 0).epsilon == 0.5
    assert s.epsilon == pytest.approx(epsilon)  # Resolving another cohort cannot mutate this one.
