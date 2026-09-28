"""Entity attention, recurrent causality and action geometry controls."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules
from pvz_game.types import ProjectileView, ZombieView

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import EntityBatch, ObservationEncoder, collate_observations
from pvz_rl.policy.entity_attention import AttentionBlock
from pvz_rl.policy.sequential_q import observation_tile_masks
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy
from pvz_rl.presentation.demo_recording import CompactHistoryBuilder


@pytest.fixture(autouse=True)
def full_precision_controls():
    # cuDNN TF32 changes with batch shape; measure invariance at full FP32.
    with torch.backends.cudnn.flags(allow_tf32=False):
        yield


def inputs(count=2, device="cpu"):
    cfg = load_config()
    encoder = ObservationEncoder(cfg, Rules())
    public = Game().reset(LevelSpec("mixed", plants=(InitialPlant("wall_nut", 2, 3),)))
    public = replace(
        public,
        zombies=(
            ZombieView(1, "buckethead", 2, 3650, 120, 700, "biting", 17, False, 12, False),
            ZombieView(2, "pole_vaulting", 0, 7000, 400, 0, "carrying_pole", 0, True, 0, False),
        ),
        projectiles=(ProjectileView(3, 2, 4500, 20, True),),
    )
    obs = encoder.encode(public)
    return cfg, collate_observations([obs] * count, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_permutation_padding_batching_and_empty_entities(device):
    cfg, batch = inputs(3, device)
    p = TransformerLSTMPolicy(cfg).to(device).eval()
    batch.globals[:, 0] = torch.arange(3, device=device)
    with torch.no_grad():
        reference = p.forward_step(batch)
        order = torch.randperm(batch.entities.shape[1], device=device)
        perm = EntityBatch(batch.entities[:, order], batch.entity_mask[:, order], batch.globals)
        shuffled = p.forward_step(perm)
        padding = torch.full((3, 7, 11), 999999, dtype=torch.int32, device=device)
        padded = EntityBatch(
            torch.cat((perm.entities, padding), 1),
            torch.cat((perm.entity_mask, torch.zeros(3, 7, dtype=torch.bool, device=device)), 1),
            batch.globals,
        )
        extra = p.forward_step(padded)
        single = [p.forward_step(batch[i : i + 1]) for i in range(3)]
        for other in (shuffled, extra):
            torch.testing.assert_close(reference.branch_q, other.branch_q, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(
                reference.tile_features, other.tile_features, atol=2e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                reference.state.hidden, other.state.hidden, atol=1e-6, rtol=1e-5
            )
            branches = torch.tensor([1, 8, 9], device=device)
            torch.testing.assert_close(
                p.tile_values(reference.tile_features, reference.context, branches),
                p.tile_values(other.tile_features, other.context, branches),
                atol=2e-6,
                rtol=1e-5,
            )
        torch.testing.assert_close(
            reference.branch_q, torch.cat([x.branch_q for x in single]), atol=1e-6, rtol=1e-5
        )
        empty = collate_observations({"entities": [], "globals": [0] * 18}, device)
        result = p.forward_step(empty)
        assert torch.isfinite(result.branch_q).all() and result.tile_features.shape == (1, 45, 32)
        assert p.entity.embed(batch).shape == (3, 9, 32)


def test_step_sequence_causality_and_reset():
    cfg, batch = inputs(10)
    observations = batch.reshape(2, 5)
    policy = TransformerLSTMPolicy(cfg).eval()
    previous = torch.randint(0, 406, (2, 5))
    outcomes = torch.rand(2, 5, 2)
    q, tiles, state = policy.forward_sequence(
        observations, previous_actions=previous, execution_outcomes=outcomes
    )
    carried = None
    outputs = []
    for t in range(5):
        out = policy.forward_step(
            observations[:, t],
            carried,
            previous_action=previous[:, t],
            execution_outcome=outcomes[:, t],
        )
        outputs.append(out.branch_q)
        carried = out.state
    torch.testing.assert_close(q, torch.stack(outputs, 1))
    torch.testing.assert_close(state.hidden, carried.hidden)
    changed = observations.clone()
    changed.globals[:, 3:] = 100
    q2 = policy.forward_sequence(changed, previous_actions=previous, execution_outcomes=outcomes)[0]
    torch.testing.assert_close(q[:, :3], q2[:, :3])
    reset = policy.reset_state(state, torch.tensor([True, False]))
    assert not reset.hidden[:, 0].any()
    torch.testing.assert_close(reset.hidden[:, 1], state.hidden[:, 1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_independent_control_and_backward(device):
    torch.manual_seed(7)
    block = AttentionBlock(32, 4, 64, 3).to(device).double().eval()
    q = torch.randn(2, 4, 7, 8, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 0], device=device, dtype=torch.bool)[None, None, None]
    expected = ((q @ k.transpose(-2, -1)) / np.sqrt(8)).masked_fill(~mask, -torch.inf).softmax(
        -1
    ) @ v
    actual = block.attention(q, k, v, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    first = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    block.force_fallback = True
    block.train()
    fallback = block.attention(q, k, v, mask)
    second = torch.autograd.grad(fallback.square().sum(), (q, k, v))
    torch.testing.assert_close(fallback, expected, rtol=1e-9, atol=1e-9)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-8)


def test_cpu_cuda_model_outputs_gradients_and_adam():
    cfg, batch = inputs()
    cpu = TransformerLSTMPolicy(cfg).double()
    gpu = TransformerLSTMPolicy(cfg).double().cuda()
    gpu.load_state_dict(cpu.state_dict())
    for model, obs in ((cpu, batch), (gpu, batch.to("cuda"))):
        out = model.forward_step(obs)
        tiles = model.tile_values(
            out.tile_features,
            out.context,
            torch.ones(len(obs), device=obs.device, dtype=torch.long),
        )
        loss = out.branch_q.square().sum() + tiles.square().sum()
        loss.backward()
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=1e-7, atol=1e-9)
    for model in (cpu, gpu):
        torch.optim.Adam(model.parameters(), lr=1e-4).step()
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(a, b.cpu(), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("dig", [False, True])
def test_greedy_branch_and_only_selected_tiles_explore(dig):
    cfg, batch = inputs(100)
    p = TransformerLSTMPolicy(cfg).eval()
    with torch.no_grad():
        p.branch_head[-1].weight.zero_()
        p.branch_head[-1].bias.zero_()
        p.branch_head[-1].bias[9 if dig else 2] = 5
    p.tile_exploration_epsilon = 1
    actions, _, details = p.decide(batch, deterministic=False)
    assert ((actions >= 361) if dig else ((actions >= 46) & (actions <= 90))).all()
    assert not details["coins"][:, 0].any() and details["coins"][:, 1].all()
    assert len(torch.unique(actions)) > 10


def test_occupancy_masks_and_full_board_proposal():
    cfg = load_config()
    encoder = ObservationEncoder(cfg, Rules())
    obs = Game().reset(
        LevelSpec(
            "full", plants=tuple(InitialPlant("wall_nut", r, c) for r in range(5) for c in range(9))
        )
    )
    batch = collate_observations([encoder.encode(obs), encoder.encode(replace(obs, plants=()))])
    masks = observation_tile_masks(batch)
    assert not masks[0, 1:361].any() and masks[:, 361:].all() and masks[1].all()
    p = TransformerLSTMPolicy(cfg)
    with torch.no_grad():
        p.branch_head[-1].weight.zero_()
        p.branch_head[-1].bias.zero_()
        p.branch_head[-1].bias[1] = 5
    assert p.decide(batch)[0][0] == 1


def test_compact_history_ignores_elapsed_only_waits_and_keeps_cooldowns():
    cfg, batch = inputs(1)
    a = batch.observations()[0]
    b = copy.deepcopy(a)
    b["globals"][1] = 0.1
    c = copy.deepcopy(b)
    c["globals"][6] = 0.5
    records = [dict(decision_index=i, action=0, observation=o) for i, o in enumerate((a, b, c))]
    assert CompactHistoryBuilder().build(records) == [
        dict(decision_index=2, reasons=["cooldown_decrement"])
    ]
