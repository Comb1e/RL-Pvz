"""Independent controls for structured exploration and spatial placement."""

from types import SimpleNamespace

import pytest
import torch
from pvz_game import Dig, Place

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.learning.curriculum import CurriculumState
from pvz_rl.learning.deadline import RunBudget


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def test_stage_residency_counts_only_matching_episode_starts():
    cfg = load_config()
    state = CurriculumState(stage=1)
    for _ in range(100):
        state.completed_episode(0)
    assert not state.observe({"easy": 100}, 500, cfg)
    assert not state.observe({"easy": 100}, 1000, cfg)
    for _ in range(99):
        state.completed_episode(1)
    restored = CurriculumState(**state.to_dict())
    assert not restored.observe({"easy": 100}, 1500, cfg)
    restored.completed_episode(1)
    assert restored.observe({"easy": 100}, 2000, cfg)
    assert restored.name == "standard" and restored.completed_stage_games == 0


def test_wall_budget_reserves_cleanup_and_restores_consumption():
    cfg = load_config()
    now = [100.0]
    budget = RunBudget(cfg, elapsed=6000, clock=lambda: now[0])
    assert budget.deadline == 1300
    assert budget.learning_deadline == 400
    assert budget.can_collect(200)
    assert not budget.can_collect(251)
    now[0] = 250
    restored = RunBudget(cfg, elapsed=budget.state()["elapsed_seconds"], clock=lambda: 500)
    assert restored.state()["remaining_seconds"] == 1050
    assert restored.state()["reserve_seconds"] == 900


def test_early_dig_metric_tracks_actual_accepted_actions(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(seed=1)
    env.step(env.codec.encode(Place("sunflower", 0, 0)))
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("categories,width", [(9, 8), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_categorical_embedding_matches_lookup_gradients_and_adam(device, categories, width, dtype):
    from pvz_rl.policy.spatial_policy import CategoricalEmbedding

    torch.manual_seed(102)
    actual = CategoricalEmbedding(categories, width, padding_idx=0).to(device=device, dtype=dtype)
    reference = torch.nn.Embedding(categories, width, padding_idx=0).to(device=device, dtype=dtype)
    reference.load_state_dict(actual.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=3e-4, eps=1e-5) for m in (actual, reference)]
    indices = torch.randint(categories, (7, 48, 45), device=device)
    indices[:, :, :30] = 0  # Repeated empty tiles dominate real placement histories.
    targets = torch.randn(*indices.shape, width, device=device, dtype=dtype)
    atol, rtol = (2e-7, 2e-6) if dtype == torch.float32 else (1e-12, 1e-10)
    for _ in range(3):
        outputs = []
        for module, optimizer in zip((actual, reference), optimizers):
            optimizer.zero_grad(set_to_none=True)
            output = module(indices)
            outputs.append(output)
            (output - targets).square().mean().backward()
        torch.testing.assert_close(*outputs, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual.weight.grad, reference.weight.grad, atol=atol, rtol=rtol)
        assert torch.count_nonzero(actual.weight.grad[0]) == 0
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(actual.weight, reference.weight, atol=atol, rtol=rtol)
        with torch.no_grad():
            torch.testing.assert_close(actual(indices), reference(indices), atol=atol, rtol=rtol)
        assert torch.count_nonzero(actual.weight[0]) == 0
    if device == "cpu":
        with pytest.raises((RuntimeError, IndexError)):
            actual(torch.tensor([categories]))


def test_missing_cuda_compiler_backend_never_enters_tracing(monkeypatch):
    from pvz_rl.policy.spatial_policy import SequentialQPolicy

    monkeypatch.setattr("pvz_rl.policy.spatial_policy.importlib.util.find_spec", lambda _: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unavailable compiler must not trace beside the collector")

    monkeypatch.setattr(torch, "compile", unexpected)
    policy = SimpleNamespace(device=torch.device("cuda"))
    SequentialQPolicy.enable_compilation(policy)
    assert policy.compilation_status == "unavailable" and "Triton" in policy.compilation_error


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sequential_heads_masks_ties_conditioning_and_roundtrips(device):
    from gymnasium import spaces

    from pvz_rl.config import load_config
    from pvz_rl.envs.actions import ActionCodec
    from pvz_rl.envs.actions import ActionSchema as A
    from pvz_rl.policy.sequential_q import action_parts, assemble, branch_masks, greedy_choice
    from pvz_rl.policy.spatial_policy import SequentialQPolicy

    cfg = load_config()
    p = SequentialQPolicy(
        spaces.Box(-1, 1, (286,)),
        spaces.Discrete(A.size),
        lambda _: 3e-4,
        features_extractor_kwargs={"layout_cfg": cfg},
    ).to(device)
    obs = torch.zeros(5, 286, device=device)
    mask = torch.zeros(5, A.size, device=device, dtype=torch.bool)
    mask[0, 0] = True  # wait only
    mask[1, A.dig_start + 9] = True  # dig only
    mask[2, [46, 50]] = True  # one species, row-major tie
    mask[3, [0, 1, 46, A.dig_start]] = True
    mask[4, [91, 182, A.dig_start]] = True
    with torch.no_grad():
        q = p.predict_values(obs)
        torch.testing.assert_close(q[:, :-1], torch.zeros_like(q[:, :-1]))
        assert (q[:, -1] == -2).all()
        result = p.decide(obs, mask, deterministic=True)
        assert result[0].tolist() == [0, A.dig_start + 9, 46, 0, 91]
        assert (mask.gather(1, result[0][:, None])).all()
        board, pooled = p.encode(obs)
        plants = p.tile_values(board, pooled, torch.ones(5, device=device, dtype=torch.long))
        digs = p.tile_values(board, pooled, torch.full((5,), 9, device=device, dtype=torch.long))
        assert plants.shape == digs.shape == (5, 45)
        assert (plants == 0).all() and (digs == -2).all()
        # Changing the branch offset changes only that conditional value map.
        p.tile_offsets[1] = 3
        assert (p.tile_values(board, pooled, torch.full((5,), 2, device=device)) == 3).all()
        assert (
            p.tile_values(board, pooled, torch.ones(5, device=device, dtype=torch.long)) == 0
        ).all()
    with pytest.raises(ValueError, match="legal"):
        p.decide(obs, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="finite"):
        greedy_choice(torch.full((5, 10), torch.nan, device=device), branch_masks(mask))
    commands = torch.arange(A.size, device=device)
    torch.testing.assert_close(assemble(*action_parts(commands)), commands)
    codec = ActionCodec(cfg)
    assert all(codec.encode(codec.decode(i)) == i for i in range(A.size))
    # Unselected outputs receive no direct regression gradient; digging is trainable.
    actions = torch.tensor([0, A.dig_start + 9, 46, 0, 91], device=device)
    first, second = p.selected_values(obs, actions)
    loss = (first - 1).square().mean() + (second[1] - 1).square()
    loss.backward()
    assert p.tile_offsets.grad[-1] < 0
    assert (p.branch_head[-1].bias.grad[[1, 4, 5, 6, 7, 8]] == 0).all()


@pytest.mark.parametrize("budget", [0.0, 0.1, 1.0])
def test_exploration_unequal_tiles_matches_independent_enumeration(budget):
    from pvz_rl.policy.sequential_q import explore, per_head_epsilon

    torch.manual_seed(103)
    count = 50000
    alpha = per_head_epsilon(budget)
    legal = torch.tensor([[True, False, True]]).expand(count, -1)
    species, first_coin = explore(torch.zeros(count, dtype=torch.long), legal, alpha)
    tile_mask = torch.arange(5)[None] < torch.where(species == 0, 2, 5)[:, None]
    tile, second_coin = explore(torch.zeros(count, dtype=torch.long), tile_mask, alpha)
    for k, n in ((0, 2), (2, 5)):
        for t in range(n):
            expected = ((1 - alpha) * (k == 0) + alpha / 2) * ((1 - alpha) * (t == 0) + alpha / n)
            measured = ((species == k) & (tile == t)).float().mean().item()
            assert measured == pytest.approx(expected, abs=0.007)
    assert (first_coin | second_coin).float().mean().item() == pytest.approx(budget, abs=0.007)
    assert not (species == 1).any()


def test_shared_q_cpu_cuda_outputs_gradients_and_adam_parity():

    from gymnasium import spaces

    from pvz_rl.config import load_config
    from pvz_rl.policy.spatial_policy import SequentialQPolicy

    torch.manual_seed(102)
    policy = SequentialQPolicy(
        spaces.Box(-1, 1, (286,)),
        spaces.Discrete(406),
        lambda _: 3e-4,
        features_extractor_kwargs={"layout_cfg": load_config()},
    ).double()
    with torch.no_grad():
        policy.branch_head[-1].weight.normal_(std=0.01)
        policy.tile_head[-1].weight.normal_(std=0.01)
    cuda = (
        SequentialQPolicy(
            spaces.Box(-1, 1, (286,)),
            spaces.Discrete(406),
            lambda _: 3e-4,
            features_extractor_kwargs={"layout_cfg": load_config()},
        )
        .double()
        .cuda()
    )
    cuda.load_state_dict(policy.state_dict())
    obs = torch.zeros(3, 286, dtype=torch.float64)
    obs[:, 270] = torch.tensor([0.5, 1, 3])
    actions = torch.tensor([0, 47, 365])
    values = []
    for network, device in ((policy, "cpu"), (cuda, "cuda")):
        first, second = network.selected_values(obs.to(device), actions.to(device))
        values.append((first.detach().cpu(), second.detach().cpu()))
        loss = (first - torch.tensor([-2.0, 1.0, -0.3], device=device)).square().mean() + second[
            1:
        ].square().mean()
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()
    for a, b in zip(*values):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    for a, b in zip(policy.parameters(), cuda.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(a, b.cpu(), rtol=1e-8, atol=1e-10)
        for key, value in policy.optimizer.state.get(a, {}).items():
            torch.testing.assert_close(
                value, cuda.optimizer.state[b][key].cpu(), rtol=1e-8, atol=1e-10
            )
