"""Independent causal-history controls, including real CUDA rollout reconstruction."""

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules, Spawn

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.policy.event_memory import EventMemory


def inputs(n=1, device="cpu"):
    cfg = load_config()
    rules = Rules()
    encoder = ObservationEncoder(cfg, rules)
    game = Game()
    observation = game.reset(
        LevelSpec(
            "history", (Spawn(10000, "basic", 0),), plants=(InitialPlant("potato_mine", 1, 1),)
        )
    )
    obs = torch.tensor(np.tile(encoder.encode(observation), (n, 1)), device=device)
    masks = torch.ones(n, 406, dtype=torch.bool, device=device)
    memory = EventMemory(cfg, rules, n, device)
    return cfg, encoder, game, obs, masks, memory


def observe(memory, obs, masks, tick, reset=False, action=0):
    n, device = len(obs), obs.device
    return memory.observe(
        obs,
        masks,
        torch.full((n,), action, device=device),
        torch.full((n,), reset, device=device),
        torch.full((n,), tick, device=device),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pole_presence_events_ignore_movement_and_reset_history(device):
    _, encoder, _, obs, masks, memory = inputs(2, device)
    observe(memory, obs, masks, 0, reset=True)
    # One pole zombie in a region, plus a second spent-pole zombie. The counts
    # stay fixed when the unused pole disappears; only the new distance changes.
    region = encoder.slices["zombies"].start
    obs[:, region + 4] = 0.4
    obs[:, region + 5] = 0.4
    obs[:, region + 7] = 0.2
    obs[:, region + 8] = 0.3
    observe(memory, obs, masks, 1)
    assert memory.event_flags[:, -1].all()
    obs[:, region + 7 : region + 9] -= 0.01
    observe(memory, obs, masks, 2)
    assert not memory.event_flags[:, -1].any()
    # Another active carrier becomes nearest without exhausting the region's
    # active poles. This remains visible now, but is not a movement event.
    obs[:, region + 8] = 0.32
    observe(memory, obs, masks, 3)
    assert not memory.event_flags[:, -1].any()
    obs[0, region + 8] = -1
    observe(memory, obs, masks, 4)
    assert memory.event_flags[:, -1].tolist() == [True, False]
    obs[0, region + 8] = 1  # A pole at the spawn boundary is present.
    observe(memory, obs, masks, 5)
    assert memory.event_flags[:, -1].tolist() == [True, False]
    observe(memory, obs, masks, 0, reset=True)
    assert memory.valid.sum(-1).tolist() == [1, 1]
    observe(memory, obs, masks, 1)
    assert not memory.event_flags[:, -1].any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compression_events_reset_and_episode_isolation(device):
    _, _, _, obs, masks, memory = inputs(2, device)
    for tick in range(250):
        observe(memory, obs, masks, tick, reset=tick == 0)
    assert memory.valid.sum(-1).max() <= memory.capacity
    assert int(memory.admitted) == 2  # only initial public state
    assert memory.counts.sum(-1).tolist() == [250, 250]
    assert memory.valid[:, memory.local : memory.local + memory.events].sum() == 2
    obs[0, 1] = 0.5  # public health change
    observe(memory, obs, masks, 250)
    assert memory.event_flags[:, -1].tolist() == [True, False]
    masks[1, 45] = False
    observe(memory, obs, masks, 251)
    assert memory.event_flags[:, -1].tolist() == [False, True]
    memory.observe(
        obs,
        masks,
        torch.zeros(2, device=device),
        torch.tensor([True, False], device=device),
        torch.tensor([0, 252], device=device),
    )
    assert memory.valid[0].sum() == 1
    assert memory.counts[0].sum() == 1
    assert memory.valid[1].sum() > 1
    assert memory.tokens[0, memory.local - 1, -2] == 1


def test_event_fifo_retains_actions_and_never_reads_future():
    _, _, _, obs, masks, memory = inputs()
    for tick in range(100):
        observe(memory, obs, masks, tick, reset=tick == 0, action=1)
        assert (memory.tokens[..., -1][memory.valid] <= tick).all()
    times = memory.tokens[0, memory.local : memory.local + memory.events, -1]
    assert (times[1:] > times[:-1]).all()
    assert memory.counts.sum() == 100


def test_timer_replacement_probe_same_current_state_different_history():
    from gymnasium.spaces import Discrete

    from pvz_rl.policy.spatial_policy import SequentialQPolicy, SpatialFeatures

    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg, encoder, _, obs, masks, early = inputs()
    late = EventMemory(cfg, Rules(), 1, "cpu")
    empty = obs.clone()
    empty[:, (1 * 9 + 1) * 3 : (1 * 9 + 2) * 3] = 0
    observe(early, empty, masks, 0, reset=True)
    observe(late, empty, masks, 0, reset=True)
    observe(early, obs, masks, 10, action=190)
    observe(late, obs, masks, 900, action=190)
    a = observe(early, obs, masks, 1000).clone()
    b = observe(late, obs, masks, 1000).clone()
    policy = SequentialQPolicy(
        encoder.space,
        Discrete(406),
        lambda _: 3e-4,
        hidden_sizes=[128, 128],
        features_extractor_class=SpatialFeatures,
        features_extractor_kwargs={"layout_cfg": cfg},
    )
    # Same current public state; a random network must receive distinct usable
    # history features. This proves observability, not learned phase competence.
    with torch.no_grad():
        ea = policy.features_extractor(obs, a)
        eb = policy.features_extractor(obs, b)
        assert not torch.allclose(ea, eb, atol=1e-7)
        future = a.clone()
        future.tokens[:, -1, -1] = 1001
        future.valid[:, -1] = True
        future.tokens[:, -1, : encoder.size] = 0
        torch.testing.assert_close(policy.features_extractor(obs, future), ea)
        for extractor in (policy.features_extractor,):
            assert (extractor.temporal.blocks[0].attn_gate.bias == -2).all()
    loss = policy.predict_values(obs, a).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())


def test_real_mine_histories_affect_tile_preferences_without_countdown_inputs():
    """Identical current boards/masks, different actual arming ages.

    This is a representational control, not evidence that the agent learned timing.
    """
    from gymnasium.spaces import Discrete
    from pvz_game import Place, Wait

    from pvz_rl.envs.actions import ActionCodec
    from pvz_rl.envs.lesson_rules import sky_rules
    from pvz_rl.policy.spatial_policy import SequentialQPolicy, SpatialFeatures

    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = load_config()
    rules = sky_rules(Rules(), False)
    encoder, codec = ObservationEncoder(cfg, rules), ActionCodec(cfg)
    games = [Game(rules), Game(rules)]
    for game in games:
        game.reset(LevelSpec("mine-age", (Spawn(10000, "basic", 0),), initial_sun=50))
    memory = EventMemory(cfg, rules, 2, "cpu")
    plant = Place("potato_mine", 1, 1)
    for tick in range(3002):
        previous = []
        for index, game in enumerate(games):
            action = plant if tick == (2010, 2900)[index] else Wait()
            if tick:
                game.step(action)
            previous.append(codec.encode(action))
        observations = torch.tensor(np.stack([encoder.encode(g.observe()) for g in games]))
        masks = torch.tensor(np.stack([codec.mask(g) for g in games]))
        context = memory.observe(
            observations,
            masks,
            torch.tensor(previous),
            torch.full((2,), tick == 0),
            torch.full((2,), tick),
        )
    torch.testing.assert_close(observations[0], observations[1], atol=0, rtol=0)
    torch.testing.assert_close(masks[0], masks[1], atol=0, rtol=0)
    assert games[0].observe().plants[0].timer_ticks != games[1].observe().plants[0].timer_ticks
    policy = SequentialQPolicy(
        encoder.space,
        Discrete(406),
        lambda _: 3e-4,
        hidden_sizes=[128, 128],
        features_extractor_class=SpatialFeatures,
        features_extractor_kwargs={"layout_cfg": cfg},
    )
    with torch.no_grad():
        # Initialization intentionally has no learned tile preference. A nonzero
        # readout tests whether temporal features can inform such preferences.
        policy.tile_head[-1].weight.normal_(std=0.01)
        board, pooled = policy.encode(observations, context)
        logits = policy.tile_values(board, pooled, torch.full((2,), 5))
        # Temporal information reaches conditional tile preferences, not only
        # a uniform tile-map offset that would cancel under softmax.
        difference = logits[0] - logits[1]
        assert difference.std(-1).max() > 1e-7
        isolated = EventMemory(cfg, rules, 2, "cpu")
        reset = isolated.observe(
            observations,
            masks,
            torch.zeros(2),
            torch.ones(2, dtype=torch.bool),
            torch.full((2,), 3001),
        )
        torch.testing.assert_close(reset.tokens[0], reset.tokens[1], atol=0, rtol=0)
        # Compare identical batch positions: CPU GEMM can round a three-wide
        # output differently across rows even when their inputs are identical.
        reset_logits = [
            policy.tile_values(
                *policy.encode(observations[i : i + 1], reset.select(torch.tensor([i]))),
                torch.tensor([5]),
            ).clone()
            for i in range(2)
        ]
        torch.testing.assert_close(reset_logits[0], reset_logits[1], atol=0, rtol=0)
