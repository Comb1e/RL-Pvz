"""Controls for the event_v8 entity/recurrent policy contract."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Game, Rules

from pvz_rl.config import load_demo_config
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy
from pvz_rl.presentation.demo_recording import CompactHistoryBuilder


def test_recurrent_geometry_isolated_per_board_and_full_board_falls_back_to_zero(monkeypatch):
    from pvz_rl.policy.transformer_lstm import RecurrentOutput

    cfg = load_demo_config()
    policy = TransformerLSTMPolicy(cfg)
    observations = torch.zeros(3, 294)
    q = torch.zeros(3, 10)
    q[:, 1] = 1
    result = RecurrentOutput(q, torch.zeros(3, 45, 1), torch.zeros(3, 1), policy.initial_state(3))
    monkeypatch.setattr(policy, "forward_step", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        policy,
        "tile_values",
        lambda tiles, context, branch: torch.arange(45).float().repeat(len(branch), 1),
    )
    masks = torch.zeros(3, 406, dtype=torch.bool)
    masks[:, 0] = True
    masks[0, 3] = masks[2, 8] = True
    assert policy.decide(observations, action_masks=masks)[0].tolist() == [3, 1, 8]
    # Default masks derive from occupancy; a full board cannot unmask another board.
    observations[0, 44 * 3] = 1
    observations[1, :135:3] = 1
    assert policy.decide(observations)[0].tolist() == [44, 1, 45]
    q[0, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        policy.decide(observations)


@pytest.mark.parametrize("demo,size", [(False, 286), (True, 294)])
def test_versioned_cpu_cuda_observation_width_and_cooldown_boundaries(demo, size):
    from pvz_game import LevelSpec

    from pvz_rl.config import load_config
    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch

    cfg = load_demo_config() if demo else load_config()
    batch = LessonCudaBatch(1, zombie_capacity=1, max_step_ticks=1)
    batch.reset([LevelSpec("cooldowns")], [0])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        assert features.observations.shape == (1, size)
        for divisor in (1, 2, 10000):
            ticks = [(p["recharge_ticks"] + 1) // divisor for p in batch.rules.plants.values()]
            batch.cooldowns[0] = batch.cp.asarray(ticks)
            features.encode()
            actual = features.observations.get()[0]
            np.testing.assert_array_equal(actual, features.encoder.encode(batch.observe(0)))
            if demo:
                expected = np.array(
                    [
                        t / (p["recharge_ticks"] + 1)
                        for t, p in zip(ticks, batch.rules.plants.values())
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_array_equal(actual[286:], expected)
            else:
                assert "cooldowns" not in features.encoder.slices


def test_event_v8_cooldowns_use_ready_and_recharge_boundaries():
    cfg = load_demo_config()
    encoder = ObservationEncoder(cfg, Rules())
    game = Game()
    observation = game.reset("easy", 1000)
    encoded = encoder.encode(observation)
    assert encoded.shape == (294,)
    assert np.all(encoded[286:] >= 0) and np.all(encoded[286:] <= 1)
    full = replace(
        observation,
        cards=tuple(
            replace(card, cooldown_ticks=card.recharge_ticks + 1) for card in observation.cards
        ),
    )
    np.testing.assert_allclose(encoder.encode(full)[286:], 1)
    ready = replace(
        observation, cards=tuple(replace(card, cooldown_ticks=0) for card in observation.cards)
    )
    np.testing.assert_allclose(encoder.encode(ready)[286:], 0)


def test_step_and_sequence_outputs_match_and_reset_isolated():
    cfg = load_demo_config()
    torch.manual_seed(101)
    policy = TransformerLSTMPolicy(cfg).eval()
    observations = torch.randn(2, 5, 294)
    previous = torch.randint(0, 406, (2, 5))
    outcomes = torch.rand(2, 5, 2)
    sequence_q, sequence_tiles, sequence_state = policy.forward_sequence(
        observations,
        previous_actions=previous,
        execution_outcomes=outcomes,
    )
    state = None
    step_q, step_tiles = [], []
    for index in range(observations.shape[1]):
        result = policy.forward_step(
            observations[:, index],
            state,
            previous_action=previous[:, index],
            execution_outcome=outcomes[:, index],
        )
        state = result.state
        step_q.append(result.branch_q)
        step_tiles.append(result.tile_features)
    torch.testing.assert_close(sequence_q, torch.stack(step_q, 1))
    torch.testing.assert_close(sequence_tiles, torch.stack(step_tiles, 1))
    torch.testing.assert_close(sequence_state.hidden, state.hidden)
    reset = policy.reset_state(state, torch.tensor([True, False]))
    fresh = policy.initial_state(1)
    reset_result = policy.forward_step(
        observations[:, 0], reset, previous_action=previous[:, 0], execution_outcome=outcomes[:, 0]
    )
    fresh_result = policy.forward_step(
        observations[:1, 0],
        fresh,
        previous_action=previous[:1, 0],
        execution_outcome=outcomes[:1, 0],
    )
    # The reset API zeros selected entries without altering the other batch lanes.
    assert torch.count_nonzero(reset.hidden[:, 0]) == 0
    assert torch.count_nonzero(reset.cell[:, 0]) == 0
    assert reset_result.branch_q.shape == (2, 10)
    torch.testing.assert_close(reset_result.branch_q[0], fresh_result.branch_q[0])
    assert fresh.hidden.shape == (1, 1, 256)


def test_future_observation_does_not_change_prefix():
    cfg = load_demo_config()
    policy = TransformerLSTMPolicy(cfg).eval()
    prefix = torch.zeros(1, 2, 294)
    first = torch.cat((prefix, torch.zeros(1, 2, 294)), 1)
    second = first.clone()
    second[:, 2:] = torch.randn_like(second[:, 2:])
    q1, _, _ = policy.forward_sequence(first)
    q2, _, _ = policy.forward_sequence(second)
    torch.testing.assert_close(q1[:, :2], q2[:, :2])


def test_compact_history_omits_elapsed_only_waits_and_keeps_cooldowns():
    base = np.zeros(294, dtype=np.float32)
    elapsed = base.copy()
    elapsed[271] = 0.1
    cooldown = elapsed.copy()
    cooldown[286] = 0.5
    rows = [
        {"decision_index": 0, "action": 0, "observation": base.tolist()},
        {"decision_index": 1, "action": 0, "observation": elapsed.tolist()},
        {"decision_index": 2, "action": 0, "observation": cooldown.tolist()},
    ]
    events = CompactHistoryBuilder().build(rows)
    assert events == [{"decision_index": 2, "reasons": ["cooldown_decrement"]}]
