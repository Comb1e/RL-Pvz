"""Combined schedules and committed species exploration controls."""

from types import SimpleNamespace

import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.learning.exploration import CommittedPlantExploration, exploration_state
from pvz_rl.policy.sequential_q import assemble


@pytest.mark.parametrize(
    "games,epsilon",
    [(0, 0.5), (5000, (0.5 * 0.01) ** 0.5), (10000, 0.01), (19000, 0.01)],
)
def test_schedule_values_and_floor(games, epsilon):
    cfg = load_config()
    s = exploration_state(cfg, games)
    assert s.epsilon == pytest.approx(epsilon)
    assert s.tile_epsilon == pytest.approx(epsilon)
    assert s.at_floor == (games >= 10000)
    assert s.plant_epsilon == pytest.approx(0.1 * (0.01 / 0.1) ** min(games / 10000, 1))
    assert s.plant_at_floor == (games >= 10000)
    assert exploration_state(cfg, 0).epsilon == 0.5
    assert s.epsilon == pytest.approx(epsilon)  # Resolving another cohort cannot mutate this one.


def test_accepted_normal_trigger_uniform_other_species_and_reset():
    torch.manual_seed(7)
    count = 14000
    controller = CommittedPlantExploration(count, "cpu", 1, 0)
    actions = assemble(torch.full((count,), 2), torch.zeros(count, dtype=torch.long))
    active = torch.ones(count, dtype=torch.bool)
    accepted = active.clone()
    mode = torch.zeros(count, dtype=torch.long)
    done = torch.zeros(count, dtype=torch.bool)
    accepted[0] = False
    active[1] = False
    mode[2] = 2
    actions[3] = 0
    actions[4] = A.dig_start
    done[5] = True
    fired = controller.observe(actions, accepted, mode, active, done)
    assert not fired[:6].any() and fired[6:].all()
    counts = torch.bincount(controller.pending[6:], minlength=9)
    assert counts[0] == counts[2] == 0
    for species in (1, 3, 4, 5, 6, 7, 8):
        assert abs(int(counts[species]) - (count - 6) / 7) < 160
    saved = controller.snapshot()
    restored = CommittedPlantExploration(count, "cpu", 1, 0)
    restored.restore(saved)
    torch.testing.assert_close(restored.pending, controller.pending)
    saved["pending"][0] = 9
    with pytest.raises(ValueError, match="recovery"):
        restored.restore(saved)


@pytest.mark.parametrize("blocker", ["sun", "cooldown", "full", "ready"])
def test_commitment_waits_or_uses_current_tile_and_actual_q(blocker):
    controller = CommittedPlantExploration(2, "cpu", 1, 0)
    controller.pending[:] = 1
    masks = torch.ones(2, A.size, dtype=torch.bool)
    if blocker == "full":
        masks[:, 1 : A.dig_start] = False
    sun = torch.tensor([49 if blocker == "sun" else 50, 500])
    cooldowns = torch.zeros(2, 8, dtype=torch.long)
    cooldowns[0, 0] = int(blocker == "cooldown")
    costs = torch.full((8,), 50)
    active = torch.tensor([True, False])
    branch_values = torch.arange(20).reshape(2, 10).float()
    tiles = torch.arange(45).float().expand(2, -1)
    policy = SimpleNamespace(tile_values=lambda board, context, branches: board + branches[:, None])
    original = torch.tensor([46, 46])

    def evidence():
        return dict(
            tile_features=tiles,
            context=torch.zeros(2, 1),
            branch_q=branch_values,
            branch_value=branch_values[:, 2],
            tile_value=torch.zeros(2),
            coins=torch.zeros(2, 2, dtype=torch.bool),
            valid=torch.tensor(True),
        )

    action, branch, tile, details = controller.apply(
        policy, original, evidence(), masks, sun, cooldowns, costs, active
    )
    ready = blocker == "ready"
    assert action.tolist() == [45 if ready else 0, 46]
    assert branch[0] == branch_values[0, int(ready)] and tile[0] == (45 if ready else 0)
    assert details["policy_action"].tolist() == original.tolist()
    assert details["exploration_mode"].tolist() == [2 if ready else 1, 0]
    controller.observe(
        action,
        torch.tensor([False, False]),
        details["exploration_mode"],
        active,
        torch.zeros(2, dtype=torch.bool),
    )
    assert controller.pending.tolist() == [1, 1]
    for _ in range(65):
        controller.observe(
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([1, 0]),
            active,
            torch.zeros(2, dtype=torch.bool),
        )
    assert controller.pending.tolist() == [1, 1]
    fired = controller.observe(
        torch.tensor([1, 0]),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([2, 0]),
        active,
        torch.zeros(2, dtype=torch.bool),
    )
    assert not fired.any() and controller.pending.tolist() == [0, 1]
    controller.observe(
        torch.zeros(2, dtype=torch.long),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.long),
        active,
        torch.tensor([False, True]),
    )
    assert not controller.pending.any()
