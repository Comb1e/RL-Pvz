"""Independent stateless attention, geometry, conditioning and execution controls."""

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules
from pvz_game.types import ProjectileView, ZombieView

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import EntityBatch, ObservationEncoder, collate_observations
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.active_batch import ActiveBatchPlan
from pvz_rl.policy.entity_attention import AttentionBlock
from pvz_rl.policy.sequential_q import action_parts, observation_tile_masks
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.presentation.demo_recording import CompactHistoryBuilder


@pytest.fixture(autouse=True)
def full_precision_controls():
    with torch.backends.cudnn.flags(allow_tf32=False):
        yield


def small_config():
    cfg = load_config()
    cfg["policy"].update(
        species_width=32,
        species_layers=2,
        species_heads=2,
        species_feedforward=64,
        tile_width=48,
        tile_layers=2,
        tile_heads=3,
        tile_feedforward=96,
    )
    return cfg


def inputs(count=2, device="cpu"):
    cfg = small_config()
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
    return cfg, collate_observations([encoder.encode(public)] * count, device)


def test_confirmed_dimensions_independent_parameters_and_no_history():
    cfg = load_config()
    policy = TwoStageTransformerPolicy(cfg)
    for encoder, width, depth, heads, feedforward, overhead in (
        (policy.species_encoder, 128, 4, 8, 512, 1),
        (policy.tile_encoder, 192, 6, 12, 768, 47),
    ):
        assert encoder.width == width and len(encoder.layers) == depth
        assert encoder.token_overhead == overhead
        for layer in encoder.layers:
            assert layer.heads == heads and layer.ff[0].out_features == feedforward
            assert width // heads == 16
        assert not any(isinstance(module, torch.nn.Dropout) for module in encoder.modules())
    assert not {id(parameter) for parameter in policy.species_encoder.parameters()} & {
        id(parameter) for parameter in policy.tile_encoder.parameters()
    }
    assert not any(isinstance(module, torch.nn.LSTM) for module in policy.modules())
    assert not any(hasattr(policy, name) for name in ("scalar", "event", "fusion", "initial_state"))
    assert policy.species_head[0].in_features == 128
    assert policy.species_head[-1].out_features == 10
    assert policy.tile_head[0].in_features == 192 and policy.tile_head[-1].out_features == 1


@pytest.mark.parametrize("live", [128, 64, 32, 16, 4, 1, 0])
def test_active_plan_buckets_preserve_original_ids_and_stage_overheads(live):
    active = np.zeros(128, dtype=bool)
    order = np.random.default_rng(17).permutation(128)
    active[order[:live]] = True
    counts = np.resize([0, 32, 33, 64, 65, 128, 129, 256], 128)
    for overhead in (1, 47):
        plan = ActiveBatchPlan.from_counts(active, counts, 256, 128, 4096, token_overhead=overhead)
        np.testing.assert_array_equal(plan.original_slot_ids, np.flatnonzero(active))
        assert plan.logical_rows == live
        assert sorted(slot for bucket in plan.buckets for slot in bucket.slot_ids) == list(
            plan.original_slot_ids
        )
        for bucket in plan.buckets:
            assert bucket.padded_size & (bucket.padded_size - 1) == 0
            assert bucket.padded_size * (bucket.entity_width + overhead) <= 4096
            assert np.all(counts[bucket.slot_ids] <= bucket.entity_width)
            np.testing.assert_array_equal(
                bucket.compact_indices, plan.compact_indices[bucket.slot_ids]
            )
            assert bucket.padding_mask.sum() == bucket.padded_size - len(bucket.slot_ids)
        assert np.all(plan.compact_indices[~active] == -1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_active_only_outputs_serial_control_and_original_slot_randomness(device):
    from pvz_rl.learning.exploration import CommittedPlantExploration

    cfg = small_config()
    counts = [5, 40, 6, 70, 70, 9, 40]
    observations = collate_observations([observation(cfg, count)[0] for count in counts], device)
    policy = TwoStageTransformerPolicy(cfg).to(device).eval()
    policy.tile_exploration_epsilon = 0.5
    policy.species_head[-1].weight.zero_()
    policy.species_head[-1].bias.zero_()
    policy.species_head[-1].bias[1] = 2
    policy.tile_head[-1].weight.zero_()
    policy.tile_head[-1].bias.zero_()
    active_host = np.array([True, False, True, True, True, True, False])
    active = torch.tensor(active_host, device=device)
    plan = ActiveBatchPlan.from_counts(active_host, counts, cfg["encoding"]["max_entities"])
    species = policy.species_values(observations, plan=plan)
    branches = torch.where(active, torch.ones(len(counts), device=device, dtype=torch.long), 0)
    tiles = policy.conditional_tiles(observations, branches, plan=plan)
    for slot in plan.original_slot_ids:
        torch.testing.assert_close(
            species[slot], policy.species_values(observations[slot : slot + 1])[0]
        )
        reference = policy.conditional_tiles(
            observations[slot : slot + 1], branches[slot : slot + 1]
        )
        torch.testing.assert_close(tiles[slot], reference[0], atol=2e-6, rtol=1e-5)
    assert not species[~active].any() and not tiles[~active].any()
    masks = observation_tile_masks(observations)
    masks[~active] = False
    results = []
    for execution in (None, plan):
        controller = CommittedPlantExploration(len(counts), device, 0.1, 0.5)
        controller.pending[3] = 2
        torch.manual_seed(27)
        actions, details = policy.decide(
            observations,
            action_masks=masks,
            active=active,
            deterministic=False,
            active_plan=execution,
        )
        chosen, _, _, details = controller.apply(
            policy,
            actions,
            details,
            masks,
            torch.ones(len(counts), device=device) * 1000,
            torch.zeros(len(counts), 8, device=device),
            torch.ones(8, device=device) * 50,
            active,
            plan=execution,
        )
        controller.observe(
            chosen, active, details["exploration_mode"], active, torch.zeros_like(active)
        )
        results.append((chosen, details, controller.pending.clone(), torch.rand(8, device=device)))
    for index in (0, 2, 3):
        assert torch.equal(results[0][index], results[1][index])
    for name in ("coins", "greedy_actions", "exploration_mode", "pending_species", "policy_action"):
        assert torch.equal(results[0][1][name], results[1][1][name])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_permutation_padding_batching_and_empty_entities(device):
    cfg, batch = inputs(3, device)
    policy = TwoStageTransformerPolicy(cfg).to(device).eval()
    branches = torch.tensor([1, 8, 9], device=device)
    with torch.no_grad():
        species = policy.species_values(batch)
        tiles = policy.conditional_tiles(batch, branches)
        order = torch.randperm(batch.entities.shape[1], device=device)
        shuffled = EntityBatch(batch.entities[:, order], batch.entity_mask[:, order], batch.globals)
        padded = EntityBatch(
            torch.cat((shuffled.entities, torch.full((3, 7, 11), 999999, device=device)), 1),
            torch.cat(
                (shuffled.entity_mask, torch.zeros(3, 7, dtype=torch.bool, device=device)), 1
            ),
            batch.globals,
        )
        for other in (shuffled, padded):
            torch.testing.assert_close(species, policy.species_values(other), atol=0.005, rtol=0.02)
            torch.testing.assert_close(
                tiles, policy.conditional_tiles(other, branches), atol=0.005, rtol=0.02
            )
        serial_species = torch.cat(
            [policy.species_values(batch[index : index + 1]) for index in range(3)]
        )
        serial_tiles = torch.cat(
            [
                policy.conditional_tiles(batch[index : index + 1], branches[index : index + 1])
                for index in range(3)
            ]
        )
        torch.testing.assert_close(species, serial_species, atol=0.005, rtol=0.02)
        torch.testing.assert_close(tiles, serial_tiles, atol=0.005, rtol=0.02)
        empty = collate_observations({"entities": [], "globals": [0] * 18}, device)
        assert torch.isfinite(policy.species_values(empty)).all()
        assert torch.isfinite(policy.conditional_tiles(empty, branches[:1])).all()


def test_stateless_conditioning_and_duplicate_pair_gradient_accumulation():
    cfg, observations = inputs(3)
    policy = TwoStageTransformerPolicy(cfg)
    original_species = policy.species_values(observations)
    original_tiles = policy.conditional_tiles(observations, torch.ones(3, dtype=torch.long))
    different = observations.clone()
    different.globals[:, 0] += 20
    policy.decide(different)
    torch.testing.assert_close(
        original_species, policy.species_values(observations), atol=0, rtol=0
    )
    torch.testing.assert_close(
        original_tiles,
        policy.conditional_tiles(observations, torch.ones(3, dtype=torch.long)),
        atol=0,
        rtol=0,
    )
    assert not torch.equal(
        original_tiles, policy.conditional_tiles(observations, torch.full((3,), 9))
    )
    seen = []
    hook = policy.tile_encoder.register_forward_pre_hook(
        lambda module, args: seen.append(len(args[0]))
    )
    requests = torch.tensor([1, 9, 1, 0])
    rows = torch.tensor([0, 2, 0, 1])
    requested = policy.conditional_tiles(observations, requests, rows=rows)
    hook.remove()
    assert seen == [2] and not requested[-1].any()
    torch.testing.assert_close(requested[0], requested[2], atol=0, rtol=0)
    requested.sum().backward()
    duplicate_gradient = [parameter.grad.clone() for parameter in policy.tile_encoder.parameters()]
    policy.zero_grad(set_to_none=True)
    control = policy.conditional_tiles(observations, requests[:2], rows=rows[:2])
    (2 * control[0].sum() + control[1].sum()).backward()
    for expected, parameter in zip(
        duplicate_gradient, policy.tile_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, parameter.grad, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("field", ["position", "health", "sun", "cooldown"])
def test_current_public_board_responsiveness(field):
    cfg, observations = inputs(1)
    torch.manual_seed(31)
    policy = TwoStageTransformerPolicy(cfg).eval()
    altered = observations.clone()
    if field in ("sun", "cooldown"):
        altered.globals[:, 0 if field == "sun" else 6] += 1
    else:
        zombie = altered.entities[0, :, 0] == 12
        altered.entities[0, zombie, 3 if field == "position" else 4] += 100
    with torch.no_grad():
        assert not torch.equal(policy.species_values(observations), policy.species_values(altered))
        branch = torch.ones(1, dtype=torch.long)
        assert not torch.equal(
            policy.conditional_tiles(observations, branch),
            policy.conditional_tiles(altered, branch),
        )


def test_wait_skips_tile_encoder_and_geometry_fallback_does_not_hide_species():
    cfg, observations = inputs(2)
    policy = TwoStageTransformerPolicy(cfg)
    seen = []
    hook = policy.tile_encoder.register_forward_pre_hook(
        lambda module, args: seen.append(len(args[0]))
    )
    with torch.no_grad():
        policy.species_head[-1].weight.zero_()
        policy.species_head[-1].bias.zero_()
        policy.species_head[-1].bias[0] = 5
        actions, details = policy.decide(observations)
        assert not actions.any() and seen == [] and not details["tile_q"].any()
        assert not details["placement_gap"].any()
        policy.species_head[-1].bias[2] = 6
        masks = observation_tile_masks(observations)
        masks[:, 1 : A.dig_start] = False
        actions, details = policy.decide(observations, action_masks=masks)
        assert actions.tolist() == [46, 46]
        assert seen == [2]
    hook.remove()


@pytest.mark.parametrize("dig", [False, True])
def test_only_selected_branch_tiles_explore_and_gap_is_conditional(dig):
    cfg, observations = inputs(100)
    policy = TwoStageTransformerPolicy(cfg)
    with torch.no_grad():
        policy.species_head[-1].weight.zero_()
        policy.species_head[-1].bias.zero_()
        policy.species_head[-1].bias[9 if dig else 2] = 100
        policy.tile_head[-1].weight.zero_()
        policy.tile_head[-1].bias.fill_(3)
        policy.tile_exploration_epsilon = 1
        actions, details = policy.decide(observations, deterministic=False)
    branches, _ = action_parts(actions)
    assert (branches == (9 if dig else 2)).all()
    assert details["coins"][:, 1].all() and not details["coins"][:, 0].any()
    assert len(torch.unique(actions)) > 10
    assert not details["placement_gap"].any()
    assert (details["selected_value"] == 3).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_independent_values_and_gradients(device):
    torch.manual_seed(7)
    block = AttentionBlock(32, 2, 64, 3).to(device).double()
    reference = copy.deepcopy(block)
    reference.force_fallback = True
    values = torch.randn(2, 9, 32, dtype=torch.float64, device=device, requires_grad=True)
    control = values.detach().clone().requires_grad_()
    valid = torch.tensor([[True] * 9, [True] * 5 + [False] * 4], device=device)
    actual = block(values, valid)
    expected = reference(control, valid)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(values.grad, control.grad, atol=1e-8, rtol=1e-8)
    for parameter, counterpart in zip(block.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, counterpart.grad, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("stage", ["species", "tile"])
def test_serial_batched_transformer_values_and_gradients(stage):
    cfg, observations = inputs(3)
    batched = TwoStageTransformerPolicy(cfg)
    serial = TwoStageTransformerPolicy(cfg)
    serial.load_state_dict(batched.state_dict())
    branches = torch.tensor([1, 8, 9])
    if stage == "species":
        actual = batched.species_values(observations)
        expected = torch.cat(
            [serial.species_values(observations[index : index + 1]) for index in range(3)]
        )
    else:
        actual = batched.conditional_tiles(observations, branches)
        expected = torch.cat(
            [
                serial.conditional_tiles(
                    observations[index : index + 1], branches[index : index + 1]
                )
                for index in range(3)
            ]
        )
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for parameter, counterpart in zip(batched.parameters(), serial.parameters(), strict=True):
        if parameter.grad is None:
            assert counterpart.grad is None
        else:
            torch.testing.assert_close(parameter.grad, counterpart.grad, atol=0.01, rtol=0.05)


def test_cpu_cuda_both_stage_gradients_and_adam_match():
    cfg, observations = inputs(2)
    cpu = TwoStageTransformerPolicy(cfg)
    gpu = TwoStageTransformerPolicy(cfg).cuda()
    gpu.load_state_dict(cpu.state_dict())
    for policy, batch in ((cpu, observations), (gpu, observations.to("cuda"))):
        species = policy.species_values(batch)
        tiles = policy.conditional_tiles(batch, torch.tensor([1, 9], device=batch.device))
        (species.square().sum() + tiles.square().sum()).backward()
    for actual, expected in zip(gpu.parameters(), cpu.parameters(), strict=True):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, atol=0.01, rtol=0.05)
    for policy in (cpu, gpu):
        torch.optim.Adam(policy.parameters(), lr=1e-4).step()
    for actual, expected in zip(gpu.parameters(), cpu.parameters(), strict=True):
        torch.testing.assert_close(actual.cpu(), expected, atol=0.01, rtol=0.05)


def test_compact_viewer_history_remains_recording_only():
    _, batch = inputs(1)
    first = batch.observations()[0]
    second = copy.deepcopy(first)
    second["globals"][1] = 0.1
    third = copy.deepcopy(second)
    third["globals"][6] = 0.5
    records = [
        dict(decision_index=index, action=0, observation=board)
        for index, board in enumerate((first, second, third))
    ]
    assert CompactHistoryBuilder().build(records) == [
        dict(decision_index=2, reasons=["cooldown_decrement"])
    ]


@pytest.mark.parametrize("count", [0, 40, 256])
def test_bf16_runtime_and_fp32_masters_cover_parameters_and_gradients(count):
    cfg = small_config()
    raw, _ = observation(cfg, count)
    if count == 0:
        raw["entities"] = raw["entities"][:0]
        raw["globals"][-4:] = 0
    observations = collate_observations([raw] * 3, "cuda")
    torch.manual_seed(101)
    policy = TwoStageTransformerPolicy(cfg).cuda()
    seen, hooks = {}, []
    for stage in ("species", "tile"):
        head = getattr(policy, f"{stage}_head")
        encoder = getattr(policy, f"{stage}_encoder")
        hooks.append(
            head.register_forward_pre_hook(
                lambda module, args, stage=stage: seen.update({f"{stage}_head": args[0].dtype})
            )
        )
        hooks.append(
            encoder.layers[0].qkv.register_forward_hook(
                lambda module, args, output, stage=stage: seen.update(
                    {f"{stage}_qkv": output.dtype}
                )
            )
        )

    def values_and_regions():
        regions = {}
        handles = [
            getattr(policy, f"{stage}_head")[0].register_forward_hook(
                lambda module, args, output, stage=stage: regions.update(
                    {stage: output.detach().clone()}
                )
            )
            for stage in ("species", "tile")
        ]
        try:
            with policy.fitting_precision("bf16_master"):
                species = policy.species_values(observations)
                tiles = policy.conditional_tiles(
                    observations, torch.tensor([1, 8, 9], device="cuda")
                )
            return species, tiles, regions
        finally:
            for handle in handles:
                handle.remove()

    species, tiles, regions = values_and_regions()
    ((species - 0.5).square().mean() + (tiles + 0.5).square().mean()).backward()
    assert all(parameter.dtype == torch.bfloat16 for parameter in policy.runtime_parameters())
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.bfloat16
        for parameter in policy.runtime_parameters()
    )
    policy.accumulate_master_gradients()
    assert all(parameter.dtype == torch.float32 for parameter in policy.master_parameters())
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in policy.master_parameters()
    )
    assert seen == dict(
        species_head=torch.bfloat16,
        tile_head=torch.bfloat16,
        species_qkv=torch.bfloat16,
        tile_qkv=torch.bfloat16,
    )
    for hook in hooks:
        hook.remove()
    parameter = policy.species_encoder.numeric.weight
    with torch.no_grad():
        parameter.fill_(1)
    parameter.grad = torch.ones_like(parameter)
    name = "species_encoder.numeric.weight"
    master = dict(zip(policy._master_names, policy.master_parameters(), strict=True))[name]
    before = master.grad.detach().clone()
    policy.accumulate_master_gradients()
    torch.testing.assert_close(master.grad, before + 1)


@pytest.mark.parametrize(
    "readout,rounded,threshold,gates",
    [
        (1.0004, 1.0, 0.99, (True, True)),
        (1.0004, 1.0, 1.0002, (True, False)),
        (1.0074, 1.0078125, 1.0076, (False, True)),
        (1.0004, 1.0, 1.01, (False, False)),
        (1.25, 1.25, 1.25, (False, False)),
    ],
    ids=["positive", "rounded-negative", "rounded-positive", "negative", "exact-boundary"],
)
def test_quantized_readout_relu_boundary_has_independent_piecewise_gradients(
    readout, rounded, threshold, gates
):
    """Isolate the FP32 Q head's piecewise map from encoder approximation errors."""
    exact = torch.tensor([[readout]], dtype=torch.float32, requires_grad=True)
    quantized = exact.detach().bfloat16().float().requires_grad_()
    assert quantized.item() == rounded
    values, derivatives = [], []
    for source, active in zip((exact, quantized), gates, strict=True):
        head = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1))
        with torch.no_grad():
            head[0].weight.fill_(1)
            head[0].bias.fill_(-threshold)
            head[2].weight.fill_(1)
            head[2].bias.zero_()
        preactivation = np.float32(source.item()) - np.float32(threshold)
        expected = max(float(preactivation), 0.0)
        assert bool(preactivation > 0) == active
        value = head(source)
        torch.testing.assert_close(value, value.new_tensor([[expected]]), atol=0, rtol=0)
        value.sum().backward()
        assert source.grad.item() == float(active)
        assert head[0].bias.grad.item() == float(active)
        assert head[0].weight.grad.item() == (source.item() if active else 0)
        assert head[2].weight.grad.item() == expected
        assert head[2].bias.grad.item() == 1
        assert all(
            parameter.dtype == parameter.grad.dtype == torch.float32
            for parameter in head.parameters()
        )
        values.append(value.detach())
        derivatives.append(source.grad)
    torch.testing.assert_close(values[0], values[1], atol=0.002, rtol=0)
    assert torch.equal(derivatives[0], derivatives[1]) == (gates[0] == gates[1])


@pytest.mark.parametrize("stage", ["species", "tile"])
@pytest.mark.parametrize("microbatch", [2, 4])
def test_outer_activation_checkpointing_matches_direct_loss_and_gradients(stage, microbatch):
    cfg = small_config()
    boards = [observation(cfg, count)[0] for count in (0, 5, 8, 13, 20, 25, 31, 40)]
    boards[0]["entities"] = boards[0]["entities"][:0]
    boards[0]["globals"][-4:] = 0
    observations = collate_observations(boards, "cuda")
    padded = EntityBatch(
        torch.cat((observations.entities, observations.entities.new_full((8, 7, 11), 999999)), 1),
        torch.cat((observations.entity_mask, observations.entity_mask.new_zeros(8, 7)), 1),
        observations.globals,
    )
    torch.manual_seed(73)
    direct = TwoStageTransformerPolicy(cfg).cuda().train()
    recomputed = TwoStageTransformerPolicy(cfg).cuda().train()
    recomputed.load_state_dict(direct.state_dict())
    branches = torch.tensor([1, 2, 9, 3, 4, 9, 7, 8], device="cuda")
    results = []
    for policy, checkpointed in ((direct, False), (recomputed, True)):
        encoder = getattr(policy, f"{stage}_encoder")
        encoder.microbatch = microbatch
        encoder.activation_checkpointing = checkpointed
        batch = padded.clone()
        batch.globals.requires_grad_()
        calls = []
        handle = encoder.layers[0].register_forward_pre_hook(
            lambda module, args: calls.append(True)
        )
        try:
            with policy.fitting_precision("bf16_master"):
                values = (
                    policy.species_values(batch)
                    if stage == "species"
                    else policy.conditional_tiles(batch, branches)
                )
            target = torch.linspace(-0.4, 0.6, values.numel(), device="cuda").reshape_as(values)
            loss = (values - target).square().mean()
            assert len(calls) == 8 // microbatch
            loss.backward()
            assert len(calls) == (2 if checkpointed else 1) * (8 // microbatch)
            gradients = {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in policy.named_parameters()
            }
            results.append(
                (
                    values.detach(),
                    loss.detach(),
                    gradients,
                    batch.globals.grad.clone(),
                )
            )
        finally:
            handle.remove()
    torch.testing.assert_close(results[0][0], results[1][0], atol=0.005, rtol=0.02)
    torch.testing.assert_close(results[0][1], results[1][1], atol=0.005, rtol=0.02)
    for name, gradient in results[0][2].items():
        counterpart = results[1][2][name]
        if gradient is None:
            assert counterpart is None
        else:
            torch.testing.assert_close(gradient, counterpart, atol=0.01, rtol=0.05)
    torch.testing.assert_close(results[0][3], results[1][3], atol=0.01, rtol=0.05)


def test_compiled_microbatches_delayed_backward_matches_eager():
    torch._dynamo.reset()
    cfg, observations = inputs(7, "cuda")
    eager = TwoStageTransformerPolicy(cfg).cuda().train()
    compiled = TwoStageTransformerPolicy(cfg).cuda().train()
    compiled.load_state_dict(eager.state_dict())
    for model in (eager, compiled):
        for encoder in model.encoders:
            encoder.microbatch = 4
    compiled.enable_compilation()
    branches = torch.arange(7, device="cuda") % 9 + 1
    with warnings.catch_warnings(record=True) as recorded:
        for iteration in range(2):
            observations.globals[:, 0] += iteration / 10
            outputs = []
            for model in (eager, compiled):
                with model.fitting_precision("bf16_master"):
                    species = model.species_values(observations)
                    tiles = model.conditional_tiles(observations, branches)
                    (species.square().mean() + tiles.square().mean()).backward()
                outputs.append((species, tiles))
            tolerance = dict(atol=0.005, rtol=0.02)
            for actual, expected in zip(*outputs, strict=True):
                torch.testing.assert_close(actual, expected, **tolerance)
        for actual, expected in zip(compiled.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=0.002, rtol=0.02)
    assert all(encoder.compilation_status == "compiled" for encoder in compiled.encoders)
    assert not any("CUDAGraph" in str(record.message) for record in recorded)
    for encoder in compiled.encoders:
        encoder.release_fitting()


def test_compilation_cache_isolated_by_stage_and_fitting_phase():
    torch._dynamo.reset()
    cfg, observations = inputs(5, "cuda")
    eager = TwoStageTransformerPolicy(cfg).cuda().train()
    compiled = TwoStageTransformerPolicy(cfg).cuda().train()
    compiled.load_state_dict(eager.state_dict())
    compiled.enable_compilation()
    branches = torch.arange(len(observations), device="cuda") % 9 + 1
    compiler_limit = torch._dynamo.config.recompile_limit
    codes = [
        encoder._compiled_encode._torchdynamo_orig_callable.__func__.__code__
        for encoder in compiled.encoders
    ]
    assert codes[0] is not codes[1]
    with warnings.catch_warnings(record=True) as recorded:
        for microbatch in (4, 3, 2):
            for model in (eager, compiled):
                model.zero_grad(set_to_none=True)
                for encoder in model.encoders:
                    encoder.microbatch = microbatch
            for width in (32, 64, 128, 256, 16, 32):
                padding = width - observations.entities.shape[1]
                batch = EntityBatch(
                    torch.nn.functional.pad(observations.entities, (0, 0, 0, padding)),
                    torch.nn.functional.pad(observations.entity_mask, (0, padding)),
                    observations.globals,
                    True,
                )
                outputs = []
                for model in (eager, compiled):
                    species = model.species_values(batch)
                    tiles = model.conditional_tiles(batch, branches)
                    (species.float().square().mean() + tiles.float().square().mean()).backward()
                    outputs.append((species, tiles))
                for actual, expected in zip(*outputs, strict=True):
                    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
            for actual, expected in zip(compiled.parameters(), eager.parameters(), strict=True):
                torch.testing.assert_close(actual.grad, expected.grad, atol=0.02, rtol=0.05)
            for encoder in compiled.encoders:
                assert encoder.compilation_status == "compiled"
                assert set(encoder._compiled_shapes) == {
                    (microbatch, width) for width in (32, 64, 128, 256)
                }
                encoder.release_fitting()
                assert encoder._fitting_backend.captures == []
                code = encoder._compiled_encode._torchdynamo_orig_callable.__func__.__code__
                assert all(code is not previous for previous in codes)
                codes.append(code)
    assert torch._dynamo.config.recompile_limit == compiler_limit
    assert not any("Encoder compilation failed" in str(record.message) for record in recorded)


def test_fitting_compiler_failure_remains_eager_after_phase_release(monkeypatch):
    cfg, batch = inputs(3, "cuda")
    encoder = TwoStageTransformerPolicy(cfg).cuda().train().tile_encoder
    encoder.enable_compilation()
    branches = torch.ones(len(batch), device="cuda", dtype=torch.long)

    def failed(*args):
        raise RuntimeError("forced fitting compilation failure")

    monkeypatch.setattr(encoder, "_compiled_encode", failed)
    with pytest.warns(RuntimeWarning, match="forced fitting compilation failure"):
        actual = encoder(batch, branches)
    expected = encoder._encode(*batch.tensors(), branches)
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()
    assert encoder.numeric.weight.grad is not None
    encoder.release_fitting()
    assert encoder._compiled_encode is None
    assert encoder.compilation_status == "fallback"
    torch.testing.assert_close(encoder(batch, branches), expected)


@torch.no_grad()
def test_collection_graph_cache_is_per_encoder_bounded_and_owns_replays(monkeypatch):
    cfg = small_config()
    first = TwoStageTransformerPolicy(cfg).cuda().train().tile_encoder
    second = TwoStageTransformerPolicy(cfg).cuda().eval().tile_encoder
    for encoder in (first, second):
        encoder.enable_compilation()

    def unexpected_fitting(*args):
        raise AssertionError("Collection must not invoke the fitting compiler")

    first._compiled_encode = unexpected_fitting
    batches = [collate_observations([observation(cfg, count)[0]], "cuda") for count in range(5, 14)]
    retained = None
    for index, batch in enumerate(batches):
        expected = first._encode(
            *batch.tensors(), torch.ones(len(batch), device=batch.device, dtype=torch.long)
        )
        actual = first(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        if index == 0:
            retained, saved = actual, actual.clone()
    assert first.collection_metrics["entries"] == first.collection_metrics["cache_entries"] == 8
    assert first.collection_metrics["evictions"] == 1
    assert first.collection_metrics["resident_entries"] <= 16
    assert first._compiled_shapes == {}
    assert second.collection_metrics["entries"] == 0
    second(batches[0], torch.ones(len(batches[0]), device=batches[0].device, dtype=torch.long))
    assert second.collection_metrics["entries"] == 1
    before_hits = first.collection_metrics["hits"]

    def unexpected_poll(*args):
        raise AssertionError("A replay hit must not synchronize or poll memory")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "mem_get_info", unexpected_poll)
        patch.setattr(torch.cuda, "synchronize", unexpected_poll)
        batches[1].globals[:, 0] += 0.75
        first.tile_marker.add_(0.1)
        expected = first._encode(
            *batches[1].tensors(),
            torch.ones(len(batches[1]), device=batches[1].device, dtype=torch.long),
        )
        actual = first(
            batches[1], torch.ones(len(batches[1]), device=batches[1].device, dtype=torch.long)
        )
        torch.testing.assert_close(actual, expected)
    assert first.collection_metrics["hits"] == before_hits + 1
    torch.testing.assert_close(retained, saved, rtol=0, atol=0)
    torch.cuda.synchronize()
    assert first.release_retired() == 1
    assert first.collection_metrics["retired_entries"] == 0
    first.release_collection()
    first.release_collection()
    assert first.collection_metrics["entries"] == 0
    assert second.collection_metrics["entries"] == 1
    second.release_collection()


@torch.no_grad()
def test_collection_graph_lru_deferred_retirement_and_transient_bound(monkeypatch):
    from pvz_rl.policy import cudagraph_backend as backend

    created = []
    free_mib = [512]

    class Fence:
        ready = False

        def query(self):
            return self.ready

    class Capture:
        def __init__(self, encode):
            self.encode = encode
            self.device = torch.device("cuda", 0)
            self.fence = Fence()
            self.released = False
            created.append(self)

        def __call__(self, inputs):
            return self.encode(*inputs).clone()

        def retire(self):
            return self.fence

        def release(self):
            assert self.fence.ready
            self.released = True

    def unexpected_wait(*args):
        raise AssertionError("Ordinary LRU eviction and fence queries must not synchronize")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "GraphCapture", Capture)
        patch.setattr(torch.cuda, "synchronize", unexpected_wait)
        patch.setattr(torch.cuda, "mem_get_info", lambda device: (free_mib[0] * 1024**2, 2**30))
        cache = backend.CollectionGraphCache(2, 512)
        batches = [
            (torch.arange(count, device="cuda", dtype=torch.float32),) for count in range(1, 7)
        ]

        def encode(value):
            return value.square() + 3

        for batch in batches[:2]:
            torch.testing.assert_close(cache(encode, batch), encode(*batch))
        cache(encode, batches[0])
        cache(encode, batches[2])
        assert cache.retired[0][0] is created[1]
        assert not created[1].released
        free_mib[0] = 511
        torch.testing.assert_close(cache(encode, batches[3]), encode(*batches[3]))
        assert cache.retired[1][0] is created[0]
        assert cache.metrics["headroom_fallbacks"] == 1
        free_mib[0] = 512
        cache(encode, batches[4])
        torch.testing.assert_close(cache(encode, batches[5]), encode(*batches[5]))
        assert cache.metrics["capacity_fallbacks"] == 1
        assert cache.metrics["evictions"] == 2
        assert cache.metrics["entries"] == cache.metrics["retired_entries"] == 2
        assert cache.metrics["resident_entries"] == 4
        assert cache.release_retired() == 0
        created[1].fence.ready = True
        assert cache.release_retired() == 1
        assert created[1].released and not created[0].released
        for capture in created:
            capture.fence.ready = True
        assert cache.release_retired() == 1
        assert cache.metrics["retired_entries"] == 0
        assert cache.metrics["retired_releases"] == 2
    cache.release()


@pytest.mark.parametrize("free_mib", [(511, 511), (512, 511), (512, 512)])
@torch.no_grad()
def test_collection_graph_headroom_before_and_after_admission(monkeypatch, free_mib):
    cfg, batch = inputs(1, "cuda")
    encoder = TwoStageTransformerPolicy(cfg).cuda().tile_encoder
    encoder.enable_compilation()
    expected = encoder._encode(
        *batch.tensors(), torch.ones(len(batch), device=batch.device, dtype=torch.long)
    )
    readings = iter(free_mib)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (next(readings) * 1024**2, 2**30)
    )
    actual = encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
    torch.testing.assert_close(actual, expected)
    admitted = int(free_mib == (512, 512))
    assert encoder.collection_metrics["entries"] == admitted
    assert encoder.collection_metrics["captures"] == admitted
    assert encoder.collection_metrics["headroom_fallbacks"] == 1 - admitted
    assert encoder.compilation_status == ("compiled" if admitted else "requested")
    assert encoder._compiled_encode is not None
    encoder.release_collection()


@torch.no_grad()
def test_collection_graph_oom_releases_cache_and_retries_eager(monkeypatch):
    from pvz_rl.policy.cudagraph_backend import GraphCapture

    cfg, batch = inputs(1, "cuda")
    encoder = TwoStageTransformerPolicy(cfg).cuda().tile_encoder
    encoder.enable_compilation()
    actual = encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
    owned = actual
    saved = owned.clone()
    captures = list(encoder._collection_cache.captures.values())
    calls = []

    def failed(self, inputs):
        calls.append(True)
        raise torch.cuda.OutOfMemoryError("forced graph allocation failure")

    with monkeypatch.context() as patch:
        patch.setattr(GraphCapture, "__call__", failed)
        batch.globals[:, 0] += 1
        expected = encoder._encode(
            *batch.tensors(), torch.ones(len(batch), device=batch.device, dtype=torch.long)
        )
        actual = encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
        torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    assert encoder.collection_metrics["oom_fallbacks"] == 1
    assert encoder.collection_metrics["entries"] == 0
    assert encoder.collection_metrics["capture_failures"] == 0
    assert all(capture.graph is None and capture.outputs is None for capture in captures)
    assert encoder._compiled_encode is not None
    torch.testing.assert_close(owned, saved, atol=0, rtol=0)
    encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
    assert encoder.collection_metrics["entries"] == 1
    encoder.release_collection()


def test_collection_settings_refresh_and_disabled_phase_helpers(monkeypatch):
    from pvz_rl.learning.performance import phase_memory

    cfg, batch = inputs(1)
    encoder = TwoStageTransformerPolicy(cfg).tile_encoder

    def unexpected_cuda(*args):
        raise AssertionError("Disabled or CPU phase cleanup must not touch CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda)
    encoder.release_collection()
    encoder.release_fitting()
    assert encoder.compilation_status == "disabled"
    assert encoder.collection_metrics["entries"] == 0
    assert phase_memory("cpu", empty_cache=True) == {}
    cfg["training"]["performance"].update(
        collection_graph_cache_entries=3, collection_vram_headroom_mib=64
    )
    encoder.configure_collection(cfg["training"]["performance"])
    assert encoder.collection_metrics["cache_entries"] == 3
    assert encoder.collection_metrics["vram_headroom_mib"] == 64
    encoder.configure_collection(
        dict(collection_graph_cache_entries=0, collection_vram_headroom_mib=0)
    )
    assert encoder.collection_metrics["cache_entries"] == 0
    encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))[
        0
    ].sum().backward()
    assert encoder.numeric.weight.grad is not None


@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_graph_phase_release_preserves_request_and_recaptures_fitting(precision):
    from pvz_rl.learning.performance import phase_memory

    torch._dynamo.reset()
    cfg, batch = inputs(2, "cuda")
    encoder = TwoStageTransformerPolicy(cfg).cuda().train().tile_encoder
    encoder.enable_compilation()
    with torch.no_grad():
        encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
    assert encoder.collection_metrics["entries"] == 1
    encoder.release_collection()
    for iteration in range(2):
        with torch.autocast("cuda", dtype=precision, enabled=precision == torch.bfloat16):
            actual = encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
            actual.square().mean().backward()
        assert encoder.collection_metrics["entries"] == 0
        captures = list(encoder._fitting_backend.captures)
        assert len(captures) == 2
        assert all(capture.graph is not None for capture in captures)
        encoder.zero_grad(set_to_none=True)
        encoder.release_fitting()
        assert all(
            capture.graph is None and capture.static_inputs is None and capture.outputs is None
            for capture in captures
        )
        assert encoder._fitting_backend.captures == []
        assert encoder._compiled_shapes == {}
        assert encoder.compilation_status == "requested"
        assert encoder._compiled_encode is not None
        batch.globals[:, 0] += iteration / 10
    with torch.no_grad():
        encoder(batch, torch.ones(len(batch), device=batch.device, dtype=torch.long))
    assert encoder._fitting_backend.captures == []
    assert encoder.collection_metrics["entries"] == 1
    encoder.release_collection()
    memory = phase_memory("cuda", empty_cache=True)
    assert 0 < memory["allocated_mib"] <= memory["reserved_mib"] <= memory["total_mib"]
    assert 0 <= memory["free_mib"] <= memory["total_mib"]
