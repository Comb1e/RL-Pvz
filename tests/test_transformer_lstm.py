"""Entity attention, recurrent causality and action geometry controls."""

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules
from pvz_game.types import ProjectileView, ZombieView

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import EntityBatch, ObservationEncoder, collate_observations
from pvz_rl.monitoring.entity_benchmark import observation
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


def test_event_normalization_preserves_large_public_values_and_time():
    from pvz_rl.envs.history import HistoryEvent, normalize_events

    cfg, _ = inputs(1)
    policy = TransformerLSTMPolicy(cfg)
    event = HistoryEvent(
        sun_gained=40000,
        sun_spent=40000,
        zombies_spawned=150,
        plants_removed=150,
        elapsed_ticks=200,
    )
    normalized = normalize_events(torch.tensor(event.inputs(), dtype=torch.float32), policy.layout)
    assert normalized[0] == 40000 / policy.layout.cost_scale and normalized[0] > 1
    assert normalized[2] == 2 and normalized[6] == 2
    torch.testing.assert_close(normalized[7], torch.tensor(np.log1p(2), dtype=torch.float32))
    assert event.writes and not HistoryEvent(elapsed_ticks=200).writes


def test_event_memory_quiet_identity_and_pending_recovery():
    from pvz_rl.policy.runner import PolicyRunner

    cfg, observations = inputs(2)
    policy = TransformerLSTMPolicy(cfg).eval()
    runner = PolicyRunner(policy, cfg, Rules(), 2, "cpu")
    masks = observation_tile_masks(observations)
    with torch.no_grad():
        runner.state.hidden.normal_()
        runner.state.cell.normal_()
    before = runner.state.clone()
    for decision in range(40):
        runner.observe_result([[0] * 7] * 2, [1, 1])
        runner.decide(observations, masks, None)
        torch.testing.assert_close(before.hidden, runner.state.hidden, atol=0, rtol=0)
        torch.testing.assert_close(before.cell, runner.state.cell, atol=0, rtol=0)
    assert runner.state.elapsed_ticks.tolist() == [40, 40]
    runner.observe_result([[25, 25, 1, 1, 1, 1, 1], [0] * 7], [0, 1])
    saved = runner.snapshot()
    recovered = PolicyRunner(policy, cfg, Rules(), 2, "cpu")
    recovered.restore(saved)
    expected = runner.decide(observations, masks, None)[3]
    actual = recovered.decide(observations, masks, None)[3]
    assert actual["memory_write"].tolist() == [True, False]
    torch.testing.assert_close(actual["branch_q"], expected["branch_q"], atol=0, rtol=0)
    assert recovered.state.elapsed_ticks.tolist() == [0, 41]
    assert not recovered.state.pending.any()
    frozen = recovered.state.clone()
    assert not recovered.decide(observations, masks, None)[3]["memory_write"].any()
    torch.testing.assert_close(frozen.hidden, recovered.state.hidden, atol=0, rtol=0)
    recovered.observe_result([[0, 50, 0, 0, 0, 1, 0], [0] * 7], [0, 0])
    assert recovered.decide(observations, masks, None)[3]["memory_write"][0]
    recovered.observe_result([[0, 0, 0, 0, 0, 0, 1], [0] * 7], [0, 0])
    assert recovered.decide(observations, masks, None)[3]["memory_write"][0]
    recovered.reset([0])
    assert not recovered.state.hidden[:, 0].any()
    assert not recovered.state.pending[0].any() and recovered.state.elapsed_ticks[0] == 0


@pytest.mark.parametrize("field", ["position", "health", "cooldown"])
def test_controlled_current_state_changes_q_without_memory_write(field):
    cfg, observations = inputs(1)
    policy = TransformerLSTMPolicy(cfg).eval()
    with torch.no_grad():
        policy.fusion[0].weight.zero_()
        policy.fusion[0].bias.fill_(2)
        policy.fusion[0].weight[0, 32] = 1
        policy.branch_head[0].weight.zero_()
        policy.branch_head[0].bias.fill_(2)
        policy.branch_head[0].weight[0, 0] = 1
        policy.branch_head[2].weight.zero_()
        policy.branch_head[2].weight[:, 0] = 1
        policy.tile_head[0].weight.zero_()
        policy.tile_head[0].bias.fill_(2)
        policy.tile_head[0].weight[0, 32] = 1
        policy.tile_head[2].weight.zero_()
        policy.tile_head[2].weight[0, 0] = 1
        altered = observations.clone()
        if field == "cooldown":
            policy.scalar[0].weight.zero_()
            policy.scalar[0].bias.fill_(2)
            policy.scalar[0].weight[0, 6] = 1
            altered.globals[:, 6] += 1
        else:
            zombie = altered.entities[0, :, 0] == 12
            assert zombie.any()
            altered.entities[0, zombie, 3 if field == "position" else 4] += 100
            original_summary = policy.entity(observations)[0]
            changed_summary = policy.entity(altered)[0]
            difference = (original_summary - changed_summary).abs()[0]
            feature = difference.argmax()
            assert difference[feature] > 1e-6
            policy.fusion[0].weight.zero_()
            policy.fusion[0].weight[0, feature] = 1
        initial = policy.initial_state(1)
        original = policy.forward_step(observations, initial)
        changed = policy.forward_step(altered, initial)
        assert not torch.equal(original.branch_q, changed.branch_q)
        branch = torch.ones(1, dtype=torch.long)
        assert not torch.equal(
            policy.tile_values(original.tile_features, original.context, branch),
            policy.tile_values(changed.tile_features, changed.context, branch),
        )
        torch.testing.assert_close(changed.state.hidden, initial.hidden, atol=0, rtol=0)
        torch.testing.assert_close(changed.state.cell, initial.cell, atol=0, rtol=0)


def test_sparse_events_match_serial_gradients_at_decision_detach_boundaries():
    cfg, batch = inputs(16)
    observations = batch.reshape(2, 8)
    sparse = TransformerLSTMPolicy(cfg)
    serial = TransformerLSTMPolicy(cfg)
    serial.load_state_dict(sparse.state_dict())
    events = torch.zeros(2, 8, 8)
    events[0, [1, 6], 0] = 25
    events[1, [0, 2, 3, 7], 6] = 1
    events[..., 7] = torch.arange(8)
    writes = events[..., :7].any(-1)
    sparse_state = serial_state = None
    for start in (0, 3, 6):
        stop = min(start + 3, 8)
        q, _, sparse_state = sparse.forward_sequence(
            observations[:, start:stop],
            sparse_state,
            events=events[:, start:stop],
            memory_write=writes[:, start:stop],
        )
        controls = []
        for decision in range(start, stop):
            output = serial.forward_step(
                observations[:, decision],
                serial_state,
                events=events[:, decision],
                memory_write=writes[:, decision],
            )
            serial_state = output.state
            controls.append(output.branch_q)
        reference = torch.stack(controls, 1)
        torch.testing.assert_close(q, reference, atol=1e-6, rtol=1e-5)
        q.square().sum().backward()
        reference.square().sum().backward()
        sparse_state, serial_state = sparse_state.detach(), serial_state.detach()
    for actual, reference in zip(sparse.parameters(), serial.parameters()):
        if actual.grad is not None or reference.grad is not None:
            torch.testing.assert_close(actual.grad, reference.grad, atol=2e-5, rtol=1e-4)


def test_step_sequence_causality_and_reset():
    cfg, batch = inputs(10)
    observations = batch.reshape(2, 5)
    policy = TransformerLSTMPolicy(cfg).eval()
    events = torch.randint(0, 2, (2, 5, 8)).float()
    events[:, 2:4, :7] = 0
    q, tiles, state = policy.forward_sequence(observations, events=events)
    carried = None
    outputs = []
    for t in range(5):
        out = policy.forward_step(
            observations[:, t],
            carried,
            events=events[:, t],
        )
        outputs.append(out.branch_q)
        carried = out.state
    torch.testing.assert_close(q, torch.stack(outputs, 1))
    torch.testing.assert_close(state.hidden, carried.hidden)
    changed = observations.clone()
    changed.globals[:, 3:] = 100
    q2 = policy.forward_sequence(changed, events=events)[0]
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


@pytest.mark.parametrize("count", [0, 40, 256])
def test_bf16_features_fp32_core_and_gradient_control(count):
    cfg = load_config()
    torch.manual_seed(101)
    reference = TransformerLSTMPolicy(cfg).cuda().train()
    mixed = TransformerLSTMPolicy(cfg).cuda().train()
    mixed.load_state_dict(reference.state_dict())
    raw, _ = observation(cfg, count)
    if count == 0:
        raw["entities"] = raw["entities"][:0]
        raw["globals"][-4:] = 0
    obs = collate_observations([raw] * 16, "cuda").reshape(2, 8)
    events = torch.ones(2, 8, 8, device="cuda")
    results, gradients = [], []
    seen = {}
    hooks = []
    for name in ("lstm", "branch_head", "tile_head"):
        hooks.append(
            getattr(mixed, name).register_forward_pre_hook(
                lambda module, args, name=name: seen.update({name: args[0].data.dtype})
            )
        )
    hooks.append(
        mixed.entity.layers[0].qkv.register_forward_hook(
            lambda module, args, output: seen.update(qkv=output.dtype)
        )
    )
    for model, precision in ((reference, "fp32"), (mixed, "features_bf16")):
        with model.fitting_precision(precision):
            q, tiles, contexts, state = model.forward_sequence(
                obs, events=events, return_context=True
            )
            tile_q = model.tile_values(
                tiles.flatten(0, 1),
                contexts.flatten(0, 1),
                torch.ones(16, device="cuda", dtype=torch.long),
            )
            loss = (q - 0.5).square().mean() + (tile_q + 0.5).square().mean()
            loss.backward()
        results.append((q.detach(), tile_q.detach()))
        assert state.hidden.dtype == state.cell.dtype == torch.float32
        # BF16 inputs can alter accumulated cell values; independently verify that
        # the recurrent core computes exactly FP32 LSTM on those actual inputs.
        with model.fitting_precision(precision):
            _, core_inputs, _ = model._inputs(obs.reshape(-1), events.reshape(-1, 8))
        initial = model.initial_state(2, device="cuda")
        _, (hidden, cell) = model.lstm(
            torch.nn.utils.rnn.pack_padded_sequence(
                core_inputs.reshape(2, 8, -1), [8, 8], batch_first=True, enforce_sorted=False
            ),
            (initial.hidden, initial.cell),
        )
        torch.testing.assert_close(state.hidden, hidden, atol=0, rtol=0)
        torch.testing.assert_close(state.cell, cell, atol=0, rtol=0)
        gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        assert all(p.dtype == p.grad.dtype == torch.float32 for p in model.parameters())
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=0.002, rtol=0.01)
    relative = torch.linalg.vector_norm(gradients[0] - gradients[1]) / torch.linalg.vector_norm(
        gradients[0]
    )
    assert relative < 0.02
    assert seen == dict(
        lstm=torch.float32, branch_head=torch.float32, tile_head=torch.float32, qkv=torch.bfloat16
    )
    assert torch.isfinite(gradients[1]).all()
    for hook in hooks:
        hook.remove()
    # Small updates survive in the master weight even if BF16 would round them away.
    parameter = mixed.scalar[0].weight
    with torch.no_grad():
        parameter.fill_(1)
    parameter.grad = torch.ones_like(parameter)
    torch.optim.SGD([parameter], lr=1e-5).step()
    assert torch.all(parameter < 1) and torch.all(parameter.bfloat16() == 1)


@pytest.mark.parametrize("precision", ["fp32", "features_bf16"])
def test_compiled_microbatches_delayed_backward_matches_eager(precision):
    """Warmup, replay, partial batches and saved activations survive until backward."""
    torch._dynamo.reset()
    cfg = load_config()
    raw = [observation(cfg, count)[0] for count in (0, 40, 256)]
    raw[0]["entities"] = raw[0]["entities"][:0]
    raw[0]["globals"][-4:] = 0
    batch = collate_observations((raw * 12)[:34], "cuda")
    torch.manual_seed(101)
    eager = TransformerLSTMPolicy(cfg).cuda().train()
    compiled = TransformerLSTMPolicy(cfg).cuda().train()
    compiled.load_state_dict(eager.state_dict())
    for model in (eager, compiled):
        model.entity.microbatch = 4
    compiled.entity.enable_compilation()
    states = [None, None]
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in (eager, compiled)]
    observations = batch.reshape(2, 17)
    previous = torch.arange(34, device="cuda").reshape(2, 17)
    events = torch.ones(2, 17, 8, device="cuda")
    events[:, ::2, :7] = 0
    with warnings.catch_warnings(record=True) as seen:
        for iteration in range(4):
            observations.globals[..., 0] = iteration / 10
            results = []
            for index, model in enumerate((eager, compiled)):
                with model.fitting_precision(precision):
                    q, tiles, context, state = model.forward_sequence(
                        observations,
                        states[index],
                        events=events,
                        return_context=True,
                    )
                    values = model.tile_values(
                        tiles.flatten(0, 1), context.flatten(0, 1), previous.flatten() % 9 + 1
                    )
                    loss = (q - 0.3).square().mean() + (values + 0.2).square().mean()
                    loss.backward()
                states[index] = state.detach()
                results.append(
                    (q.detach(), values.detach(), state.hidden.detach(), state.cell.detach())
                )
            for left, right in zip(*results):
                # AOT BF16 backward rounds a few reductions differently. After
                # weight updates a tiny master-weight difference can cross a
                # BF16 rounding boundary; use the existing feature tolerance.
                atol, rtol = (0.002, 0.01) if precision == "features_bf16" else (2e-5, 2e-4)
                torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
            left = torch.cat([p.grad.flatten() for p in eager.parameters()])
            right = torch.cat([p.grad.flatten() for p in compiled.parameters()])
            assert torch.isfinite(right).all()
            assert torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(left) < 0.002
            if iteration % 2:
                # Keep gradients across fitting chunks and commit only at the
                # pass boundary, as the actual complete-return learner does.
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for left, right in zip(eager.parameters(), compiled.parameters()):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert compiled.entity.compilation_status == "compiled", compiled.entity.compilation_error
    assert not any("CUDAGraph" in str(w.message) for w in seen)


def test_fp32_batching_attention_gradients_and_update_match_reference():
    cfg = load_config()
    torch.manual_seed(19)
    reference = TransformerLSTMPolicy(cfg).cuda().train()
    optimized = TransformerLSTMPolicy(cfg).cuda().train()
    optimized.load_state_dict(reference.state_dict())
    reference.entity.microbatch = 4
    reference.entity.activation_checkpointing = True
    for layer in reference.entity.layers:
        layer.force_fallback = True
    raw = [observation(cfg, n)[0] for n in (5, 40, 128, 256)] * 4
    batch = collate_observations(raw, "cuda").reshape(2, 8)
    outputs = []
    for model in (reference, optimized):
        q, tiles, context, state = model.forward_sequence(
            batch, events=torch.ones(2, 8, 8, device="cuda"), return_context=True
        )
        values = model.tile_values(
            tiles.flatten(0, 1), context.flatten(0, 1), torch.arange(16, device="cuda") % 9 + 1
        )
        (q.square().mean() + values.square().mean()).backward()
        outputs.append((q, values, state.hidden, state.cell))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(reference.parameters(), optimized.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    updates = []
    for model in (reference, optimized):
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
        originals = [p.detach().clone() for p in model.parameters()]
        # Independent first-step Adam formula. Near-zero key-bias gradients are
        # rounding noise (softmax is shift-invariant); Adam's epsilon can amplify
        # that tiny gradient difference, so verify the propagated difference.
        deltas = [
            cfg["training"]["learning_rate"] * p.grad / (p.grad.abs() + 1e-8)
            for p in model.parameters()
        ]
        torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"]).step()
        for p, original, delta in zip(model.parameters(), originals, deltas):
            torch.testing.assert_close(p, original - delta, atol=1e-7, rtol=1e-6)
        updates.append(deltas)
    for a, b, da, db in zip(reference.parameters(), optimized.parameters(), *updates):
        assert torch.all((a - b).abs() <= (da - db).abs() + 2e-7)
