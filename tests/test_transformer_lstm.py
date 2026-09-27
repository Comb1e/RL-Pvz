"""Controls for the event_v8 entity/recurrent policy contract."""

from dataclasses import replace

import numpy as np
import torch
from pvz_game import Game, Rules

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy
from pvz_rl.presentation.demo_recording import CompactHistoryBuilder


def test_event_v8_cooldowns_use_ready_and_recharge_boundaries():
    cfg = load_config()
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
    ready = replace(observation, cards=tuple(replace(card, cooldown_ticks=0) for card in observation.cards))
    np.testing.assert_allclose(encoder.encode(ready)[286:], 0)


def test_step_and_sequence_outputs_match_and_reset_isolated():
    cfg = load_config()
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
        observations[:1, 0], fresh, previous_action=previous[:1, 0], execution_outcome=outcomes[:1, 0]
    )
    # The reset API zeros selected entries without altering the other batch lanes.
    assert torch.count_nonzero(reset.hidden[:, 0]) == 0
    assert torch.count_nonzero(reset.cell[:, 0]) == 0
    assert reset_result.branch_q.shape == (2, 10)
    torch.testing.assert_close(reset_result.branch_q[0], fresh_result.branch_q[0])
    assert fresh.hidden.shape == (1, 1, 256)


def test_future_observation_does_not_change_prefix():
    cfg = load_config()
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
