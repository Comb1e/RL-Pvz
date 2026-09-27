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


def test_occupancy_masks_are_shared_by_all_plant_branches(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(
        seed=103,
        options={"scenario": LevelSpec("q-occupied", (Spawn(1000, "basic", 0),), initial_sun=1000)},
    )
    occupied = env.codec.encode(Place("peashooter", 0, 0))
    assert env.action_masks()[occupied]
    _, _, terminated, truncated, _ = env.step(occupied)
    assert not terminated and not truncated
    masks = env.action_masks()
    for plant_index in range(A.plant_types):
        assert not masks[1 + plant_index * A.tiles]
        assert masks[1 + plant_index * A.tiles + 1]


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
