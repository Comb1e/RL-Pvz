"""Regression controls for retaining unavailable high-value plant proposals."""

import pytest
import torch
from pvz_game import LevelSpec, Place, Spawn
from pvz_game.cuda.schema import REASONS

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.env import PvZEnv
from pvz_rl.policy.sequential_q import select_q_actions


def _select(values, mask):
    board = torch.zeros(1, 1, 5, 9)
    pooled = torch.zeros(1, 1)

    def tile_values(_board, _pooled, branches):
        return torch.zeros(len(branches), A.tiles)

    return int(
        select_q_actions(
            torch.tensor([values], dtype=torch.float32),
            board,
            pooled,
            tile_values,
            torch.as_tensor(mask, dtype=torch.bool)[None],
            deterministic=True,
        )[0][0]
    )


def test_high_q_unaffordable_plant_stays_selected_and_is_rejected(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(
        seed=101,
        options={
            "scenario": LevelSpec("q-sun-boundary", (Spawn(1000, "basic", 0),), initial_sun=50)
        },
    )
    mask = env.action_masks()
    peashooter = env.codec.encode(Place("peashooter", 0, 0))
    sunflower = env.codec.encode(Place("sunflower", 0, 0))
    assert mask[peashooter] and mask[sunflower]
    assert not env.engine_action_masks()[peashooter]
    assert env.engine_action_masks()[sunflower]

    action = _select([0, 1, 5, 0, 0, 0, 0, 0, 0, 0], mask)
    assert action == peashooter
    executable_mask = mask.copy()
    executable_mask[1 + A.tiles : 1 + 2 * A.tiles] = False
    # A stale executable-card mask may hide every peashooter tile, but it
    # must never demote the independently selected first-level branch.
    assert _select([0, 1, 5, 0, 0, 0, 0, 0, 0, 0], executable_mask) == peashooter
    before = env.public
    _, reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert len(env.public.plants) == len(before.plants)
    assert {(p.plant_type, p.row, p.col) for p in env.public.plants} == {
        (p.plant_type, p.row, p.col) for p in before.plants
    }
    assert env.public.tick == before.tick + 1
    assert not info["accepted"] and info["rejection_reason"] == "insufficient_sun"
    assert info["proposal_action"] == peashooter and info["executed_action"] == 0
    assert info["ticks_advanced"] == 1
    assert reward == pytest.approx(-per_tick_cfg["reward"]["invalid_plant_penalty"])


def test_high_q_cooling_plant_stays_selected_and_is_rejected(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(
        seed=102,
        options={
            "scenario": LevelSpec(
                "q-cooldown-boundary", (Spawn(1000, "basic", 0),), initial_sun=1000
            )
        },
    )
    first = env.codec.encode(Place("sunflower", 0, 0))
    _, _, _, _, placed = env.step(first)
    assert placed["accepted"] and env.public.tick == 0

    mask = env.action_masks()
    cooling = env.codec.encode(Place("sunflower", 0, 1))
    assert mask[cooling] and not env.engine_action_masks()[cooling]
    action = _select([0, 5, 1, 0, 0, 0, 0, 0, 0, 0], mask)
    assert action == cooling
    before = env.public
    _, reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert len(env.public.plants) == len(before.plants)
    assert {(p.plant_type, p.row, p.col) for p in env.public.plants} == {
        (p.plant_type, p.row, p.col) for p in before.plants
    }
    assert env.public.tick == before.tick + 1
    assert not info["accepted"] and info["rejection_reason"] == "card_recharging"
    assert info["proposal_action"] == cooling and info["executed_action"] == 0
    assert info["ticks_advanced"] == 1
    assert reward == pytest.approx(-per_tick_cfg["reward"]["invalid_plant_penalty"])


@pytest.mark.parametrize(
    ("initial_sun", "actions", "reason"),
    [
        (50, ["peashooter"], "insufficient_sun"),
        (1000, ["sunflower", "sunflower"], "card_recharging"),
    ],
)
def test_cuda_rejection_preserves_proposal_and_matches_cpu(
    per_tick_cfg, initial_sun, actions, reason
):
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
    from pvz_rl.envs.cuda_features import CudaFeatures

    scenario = LevelSpec("q-cuda-rejection", (Spawn(1000, "basic", 0),), initial_sun=initial_sun)
    env = PvZEnv(per_tick_cfg)
    env.reset(seed=104, options={"scenario": scenario})
    batch = AccountingCudaBatch(1, zombie_capacity=1, max_step_ticks=1)
    batch.reset([scenario], [104])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, per_tick_cfg, "masked")
        features.encode()
        for index, kind in enumerate(actions):
            action = env.codec.encode(Place(kind, 0, index))
            cpu_before = env.public
            _, cpu_reward, _, _, cpu_info = env.step(action)
            features.step(batch.cp.asarray([action], dtype=batch.cp.int64))
            header = batch.header.get()[0]
            assert int(header[12]) == int(cpu_info["accepted"])
            assert int(header[13]) == (
                0 if cpu_info["accepted"] else REASONS.index(cpu_info["rejection_reason"])
            )
            assert int(header[14]) == cpu_info["ticks_advanced"]
            assert float(features.rewards.get()[0]) == pytest.approx(cpu_reward, abs=2e-6)
            if not cpu_info["accepted"]:
                assert cpu_info["rejection_reason"] == reason
                assert cpu_info["executed_action"] == 0
                assert cpu_info["ticks_advanced"] == 1
                assert len(env.public.plants) == len(cpu_before.plants)
                assert int(batch.header.get()[0, 9]) == len(cpu_before.plants)
                assert int(batch.header.get()[0, 14]) == 1


def test_counterfactual_parity_history_and_simulator_isolation(per_tick_cfg, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace

    import numpy as np

    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.encoding import observations_equal
    from pvz_rl.envs.probes import CounterfactualCollector, cpu_probe
    from pvz_rl.learning.objective import components
    from pvz_rl.policy.transformer_lstm import RecurrentState

    scenarios = [
        LevelSpec("probe-boundary", (Spawn(1, "basic", 0, x=x),), initial_sun=50)
        for x in (1999, 999)
    ]
    cpus = [PvZEnv(per_tick_cfg) for _ in scenarios]
    for env, scenario in zip(cpus, scenarios):
        env.reset(seed=104, options={"scenario": scenario})
    batch = AccountingCudaBatch(2, zombie_capacity=1, max_step_ticks=1)
    calls = []

    class Teacher:
        def tile_values(self, tiles, context, branches):
            return torch.arange(45, device="cuda").float().expand(len(branches), -1)

        def forward_step(self, obs, state, *, previous_action, execution_outcome):
            calls.append((state.clone(), previous_action.clone(), execution_outcome.clone()))
            return SimpleNamespace(
                state=RecurrentState(state.hidden + 1, state.cell + 1),
                branch_q=torch.full((2, 10), 0.5, device="cuda"),
            )

    teacher = Teacher()
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        batch.reset(scenarios, [104, 104])
        features = CudaFeatures(batch, per_tick_cfg, "masked")
        obs = features.encode()
        features.initialize_home([0, 1])
        probe = CounterfactualCollector(
            SimpleNamespace(batch=batch, features=features, cfg=per_tick_cfg, condition="masked")
        )
        active = torch.ones(2, dtype=torch.bool, device="cuda")
        actions = torch.zeros(2, dtype=torch.long, device="cuda")
        details = dict(tile_features=torch.zeros(2, 1, device="cuda"), context=None)
        # Nine successive pairs cover every nonselected branch exactly twice;
        # no behavior random draw or tile probe is made for waiting.
        rng = torch.cuda.get_rng_state().clone()
        branches = []
        for _ in range(9):
            proposed, valid = probe.schedule(
                teacher, actions, details, features.mask_tensor, active
            )
            branches.extend(((proposed[0, :2] - 1) // 45 + 1).cpu().tolist())
            assert not valid[:, 2:].any()
        assert np.bincount(branches, minlength=10).tolist() == [0] + [2] * 9
        torch.testing.assert_close(rng, torch.cuda.get_rng_state())

        proposals = torch.tensor([[0, 1, 46, 361]] * 2, device="cuda")
        monkeypatch.setattr(
            probe, "schedule", lambda *args: (proposals, active[:, None].expand(-1, 4))
        )
        history = SimpleNamespace(
            policy=teacher,
            state=RecurrentState(
                torch.zeros(1, 2, 1, device="cuda"), torch.zeros(1, 2, 1, device="cuda")
            ),
            previous_actions=actions.clone(),
            outcomes=torch.zeros(2, 2, device="cuda"),
            resets=active.clone(),
        )
        hashes = [batch.state_hash(i) for i in range(2)]
        snapshots = [env.game.snapshot() for env in cpus]
        ledgers = [deepcopy(env.proximity.stages) for env in cpus]
        home = features.home_ledger.copy()
        totals = features.totals.copy()
        host = probe.collect(teacher, history, obs, actions, details, features.mask_tensor, active)
        torch.cuda.synchronize()
        rows, observations = probe.materialize(host)
        assert hashes == [batch.state_hash(i) for i in range(2)]
        np.testing.assert_array_equal(home.get(), features.home_ledger.get())
        np.testing.assert_array_equal(totals.get(), features.totals.get())
        for i, env in enumerate(cpus):
            for j, action in enumerate((0, 1, 46, 361)):
                expected = cpu_probe(env.game, env.proximity, action, per_tick_cfg)
                row = rows[i, j]
                assert row["action"] == expected["proposal"]
                for key in ("executed_action", "accepted", "duration", "tick", "done", "won"):
                    assert row[key] == expected[key]
                assert row["reason"] == (
                    0 if expected["accepted"] else REASONS.index(expected["reason"])
                )
                np.testing.assert_allclose(
                    row["components"], components(expected["components"]), atol=1e-12
                )
                assert observations_equal(
                    observations[j][i : i + 1].observations()[0], expected["observation"]
                )
                assert row["bootstrap"] == 0.5
                if j in (0, 2, 3):
                    assert row["components"][4] == pytest.approx((-0.005, -0.020)[i])
            assert env.game.snapshot() == snapshots[i]
            assert env.proximity.stages == ledgers[i]
        # Only the actual current observation advances teacher history. Each
        # fork starts there, and rejected forks see executed wait + rejection.
        assert torch.all(history.state.hidden == 1)
        assert all(torch.all(state.hidden == 1) for state, _, _ in calls[1:])
        for _, previous, outcome in calls[3:]:
            assert torch.all(previous == 0)
            torch.testing.assert_close(outcome, torch.tensor([[0.0, 1.0]] * 2, device="cuda"))
