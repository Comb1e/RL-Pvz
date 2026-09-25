"""Independent controls for structured exploration and spatial placement."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pvz_game import Dig, Place

from pvz_rl.config import load_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.deadline import RunBudget
from pvz_rl.env import PvZEnv
from pvz_rl.exploration import exploration_loss
from pvz_rl.grouped_policy import GroupedDistribution
from pvz_rl.training import build_model, vector_env


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def test_balanced_bonus_does_not_push_wait_toward_one_over_136():
    logits = torch.zeros((1, 416), requires_grad=True)
    masks = torch.zeros((1, 406), dtype=torch.bool)
    masks[:, 0] = True
    for group in (0, 2, 4):
        masks[:, 1 + 45 * group : 1 + 45 * (group + 1)] = True
    dist = GroupedDistribution().proba_distribution(logits)
    dist.apply_masking(masks)
    assert dist.probs[0, 0].item() == pytest.approx(0.25)
    np.testing.assert_allclose(dist.probs.detach().numpy().sum(), 1, atol=1e-7)
    assert torch.equal(dist.probs[~masks], torch.zeros_like(dist.probs[~masks]))
    true_gradient = torch.autograd.grad(dist.entropy().sum(), logits, retain_graph=True)[0]
    assert true_gradient[0, 0].item() == pytest.approx((3 / 16) * np.log(3 / 135))
    policy = SimpleNamespace(
        action_dist=dist,
        exploration_settings={
            "objective": "balanced_heads_v2",
            "type_coef": 0.01,
            "plant_coef": 0.001,
            "tile_coef": 0.001,
        },
    )
    loss, metrics = exploration_loss(policy, dist.entropy(), dist.log_prob(torch.tensor([0])))
    assert metrics["exploration_bonus"].item() == pytest.approx(
        0.01 * (-0.25 * np.log(0.25) - 0.75 * np.log(0.75)) + 0.002
    )
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(
        gradient[:, :3],
        torch.tensor(
            [[-0.01 * 3 / 16 * np.log(3), 0, 0.01 * 3 / 16 * np.log(3)]], dtype=gradient.dtype
        ),
        atol=1e-8,
        rtol=0,
    )
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("counts", [[0] * 9, [1, 0, 45, 3, 0, 0, 0, 0, 0]])
def test_bonus_matches_independent_entropy_and_single_choice_boundaries(counts):
    rng = np.random.default_rng(102)
    logits = torch.tensor(rng.normal(size=(2, 416)), dtype=torch.float64, requires_grad=True)
    mask = torch.zeros((2, 406), dtype=torch.bool)
    mask[:, 0] = True
    for group, count in enumerate(counts):
        mask[:, 1 + group * 45 : 1 + group * 45 + count] = True
    dist = GroupedDistribution().proba_distribution(logits)
    dist.apply_masking(mask)
    policy = SimpleNamespace(
        action_dist=dist,
        exploration_settings={
            "objective": "balanced_heads_v2",
            "type_coef": 0.01,
            "plant_coef": 0.001,
            "tile_coef": 0.001,
        },
    )
    loss, _ = exploration_loss(policy, dist.entropy(), dist.log_prob(torch.zeros(2)))
    p = dist.types.probs.detach().numpy()
    type_h = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=1)
    tile_h = []
    for g, count in enumerate(counts):
        if count:
            probs = dist.locations.probs[:, g, :count].detach().numpy()
            tile_h.append(-(probs * np.log(probs)).sum(1) / np.log(max(2, count)))
    species_count = sum(n > 0 for n in counts[:8])
    q = dist.plants.probs.detach().numpy()
    plant_h = -(q * np.log(np.maximum(q, 1e-300))).sum(-1) / np.log(max(2, species_count))
    expected = (
        0.01 * type_h
        + 0.001 * plant_h * (species_count > 0)
        + 0.001 * (np.mean(tile_h, axis=0) if tile_h else 0)
    )
    assert -loss.item() == pytest.approx(expected.mean())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for action in torch.nonzero(mask[0]).flatten():
        actual = dist.log_prob(action.expand(2)).detach().numpy()
        np.testing.assert_allclose(
            actual, np.log(dist.probs[:, action].detach().numpy()), atol=1e-6
        )


def test_spatial_policy_shapes_gradients_and_board_dependence():
    cfg = load_config()
    cfg["training"].update(n_envs=1, device="cuda", rollout_size=128, batch_size=128)
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        raw = PvZEnv(cfg)
        obs, _ = raw.reset(seed=3)
        data = torch.tensor(np.stack([obs, obs]), device="cuda")
        data[1, raw.encoder.slices["plants"].start] = 1
        masks = torch.tensor(np.stack([raw.action_masks()] * 2), device="cuda")
        actor_features, critic_features = model.policy.extract_features(data)
        for features in (actor_features, critic_features):
            assert features.shape == (2, 64 + 32 * 45)
            assert not torch.equal(features[0], features[1])
        actions, values, logs = model.policy(data, action_masks=masks)
        assert actions.shape == logs.shape == (2,)
        assert values.shape == (2, 1)
        assert all(masks[i, a] for i, a in enumerate(actions))
        loss = values.sum() + logs.sum()
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None
        )
        optimizer_ids = {id(p) for g in model.policy.optimizer.param_groups for p in g["params"]}
        critic_ids = {
            id(p) for g in model.policy.critic_optimizer.param_groups for p in g["params"]
        }
        assert optimizer_ids.isdisjoint(critic_ids)
        assert optimizer_ids | critic_ids == {id(p) for p in model.policy.parameters()}
        assert sum(p.numel() for p in model.policy.parameters()) == 1_128_060
    finally:
        env.close()


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


def test_dig_initialization_is_trainable_and_checkpointed(tmp_path):
    from pvz_rl.config import load_config
    from pvz_rl.cuda_ppo import CudaMaskablePPO
    from pvz_rl.env import PvZEnv
    from pvz_rl.training import build_model, vector_env

    torch.set_num_threads(1)
    cfg = load_config("configs/train.toml")
    cfg["training"].update(device="cuda", n_envs=1, rollout_size=128, batch_size=128)
    gpu = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", gpu, 101)
    gpu.close()
    model.policy.to("cpu")
    env = PvZEnv(cfg)
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(obs).unsqueeze(0)
    mask = torch.zeros(1, 406, dtype=torch.bool)
    mask[:, 0] = mask[:, 361] = True
    distribution = model.policy.get_distribution(x, action_masks=mask)
    assert cfg["policy"]["initial_dig_logit"] == -12
    assert 0 < distribution.probs[0, 361] < 0.00001
    assert distribution.probs[0, 1:361].sum() == 0
    (-distribution.log_prob(torch.tensor([361]))).backward()
    bias = model.policy.action_net.kind_head.bias
    assert bias.grad[1].isfinite() and bias.grad[1].abs() > 0.9
    # Learning may make digging preferable; it is not a permanent suppression rule.
    with torch.no_grad():
        bias[1] = 6
    assert model.policy.get_distribution(x, action_masks=mask).mode().item() == 361
    path = tmp_path / "policy.zip"
    model.save(path)
    loaded = CudaMaskablePPO.load(path, device="cpu")
    actual = loaded.policy.get_distribution(x, action_masks=mask).probs
    expected = model.policy.get_distribution(x, action_masks=mask).probs
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("categories,width", [(9, 8), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_categorical_embedding_matches_lookup_gradients_and_adam(device, categories, width, dtype):
    from pvz_rl.spatial_policy import CategoricalEmbedding

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


def test_masked_distribution_single_pass_preserves_probabilities_and_dig_hazard():
    from pvz_rl.grouped_policy import GroupedDistribution

    logits = torch.zeros(1, 416, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[0, 1] = -12
    mask = torch.zeros(1, 406, dtype=torch.bool)
    mask[0, 0] = mask[0, 361] = True
    direct = GroupedDistribution(0.1).proba_distribution(logits, masks=mask)
    previous = GroupedDistribution(0.1).proba_distribution(logits)
    previous.apply_masking(mask)
    torch.testing.assert_close(direct.probs, previous.probs, atol=0, rtol=0)
    probability = float(direct.probs[0, 361].detach())
    assert probability == pytest.approx(0.9 / (1 + math.exp(12)), rel=1e-12)
    assert 1 - (1 - probability) ** 500 == pytest.approx(0.002761067438, abs=1e-10)
    grads = [
        torch.autograd.grad(d.entropy().sum(), logits, retain_graph=True)[0]
        for d in (direct, previous)
    ]
    torch.testing.assert_close(*grads, atol=0, rtol=0)
    with pytest.raises(ValueError, match="at least one legal action"):
        GroupedDistribution().proba_distribution(logits, masks=torch.zeros_like(mask))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
@pytest.mark.parametrize("action", [0, 361, 46])
def test_conditional_likelihood_updates_only_used_arguments(device, epsilon, action):
    logits = torch.zeros(1, 416, device=device, dtype=torch.float64, requires_grad=True)
    dist = GroupedDistribution(epsilon).proba_distribution(logits)
    (-dist.log_prob(torch.tensor([action], device=device))).backward()
    grad = logits.grad[0]
    if action in (0, 361):
        torch.testing.assert_close(grad[3:11], torch.zeros_like(grad[3:11]), rtol=0, atol=1e-14)
    else:
        assert grad[4] < 0
    tile_grad = grad[11:].reshape(9, 45)
    used = None if action == 0 else 8 if action == 361 else 1
    for group in range(9):
        if group == used:
            assert tile_grad[group, 0] < 0
        else:
            assert torch.count_nonzero(tile_grad[group]) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("only", [0, 1, 360, 361, 405])
def test_single_legal_action_has_zero_entropy_and_likelihood_gradient(device, only):
    x = torch.linspace(-1000, 1000, 416, device=device, dtype=torch.float64)[None].requires_grad_()
    mask = torch.zeros(1, 406, device=device, dtype=torch.bool)
    mask[0, only] = True
    d = GroupedDistribution(0.1).proba_distribution(x, mask)
    assert d.sample().item() == d.mode().item() == only
    assert d.probs[0, only].item() == 1
    assert abs(d.entropy().item()) < 1e-12
    (-d.log_prob(torch.tensor([only], device=device)) + d.entropy()).sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x.grad), atol=1e-12, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_extreme_kind_logits_keep_species_exploration_and_finite_gradients(device):
    x = torch.zeros(1, 416, device=device, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[0, 0] = 1000
        x[0, 3] = -1000
    d = GroupedDistribution(0.1).proba_distribution(x)
    assert d.probs[0, 1:46].sum().item() == pytest.approx(0.1 / 9.2)
    assert d.mode().item() == 0
    (-d.log_prob(torch.tensor([1], device=device))).backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("epsilon", [0, 0.1, 1])
def test_balanced_initialization_and_persistent_uniform_tile_exploration(device, epsilon):
    from pvz_rl.spatial_policy import SpatialFeatures, SpatialGroupedPolicy

    cfg = load_config()
    env = PvZEnv(cfg, family="saving")
    obs, _ = env.reset(seed=4)
    policy = SpatialGroupedPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 1e-4,
        features_extractor_class=SpatialFeatures,
        features_extractor_kwargs={"layout_cfg": cfg},
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        exploration_epsilon=epsilon,
    ).to(device)
    dist = policy.get_distribution(torch.tensor(obs[None], device=device), env.action_masks())
    assert dist.probs[0, 0].item() == pytest.approx(1.2 / 5.2)
    for group in (0, 1, 2, 4):
        torch.testing.assert_close(
            dist.probs[0, 1 + 45 * group : 1 + 45 * (group + 1)],
            torch.full((45,), 1 / 5.2 / 45, device=device),
        )
    # Change affordable species, partial tiles and legal digging. Every species
    # retains one branch weight independently of its number of available tiles.
    logits = policy.action_net(policy.pi_features_extractor(torch.tensor(obs[None], device=device)))
    masks = torch.zeros(1, 406, dtype=torch.bool, device=device)
    masks[:, [0, 1, 3, 46, 361]] = True
    dist.proba_distribution(logits, masks)
    total = 3.2 + math.exp(-12)
    expected = torch.zeros(406, device=device)
    expected[0] = (1 - epsilon) * 1.2 / total + epsilon * 1.2 / 3.2
    expected[[1, 3]] = ((1 - epsilon) / total + epsilon / 3.2) / 2
    expected[46] = (1 - epsilon) / total + epsilon / 3.2
    expected[361] = (1 - epsilon) * math.exp(-12) / total
    torch.testing.assert_close(dist.probs[0], expected)
    with torch.no_grad():
        policy.action_net.tiles.weight.normal_()  # A learned location preference.
    dist = policy.get_distribution(torch.tensor(obs[None], device=device), masks)
    if epsilon:
        assert dist.probs[0, 1].item() >= epsilon / 3.2 / 2 - 1e-7
        assert dist.probs[0, 3].item() >= epsilon / 3.2 / 2 - 1e-7
    if epsilon == 1:
        assert dist.probs[0, 1].item() == pytest.approx(dist.probs[0, 3].item())


def test_positive_planting_signal_can_increase_planting_and_digging_remains_trainable():
    logits = torch.zeros(1, 416, requires_grad=True)
    with torch.no_grad():
        logits[0, 0], logits[0, 1] = math.log(1.2), -12
    masks = torch.zeros(1, 406, dtype=torch.bool)
    masks[:, [0, 1, 361]] = True
    optimizer = torch.optim.Adam([logits], lr=0.01)
    for action in (1, 361):
        before = GroupedDistribution(0.1).proba_distribution(logits, masks).probs[0, action].item()
        optimizer.zero_grad()
        loss = (
            -GroupedDistribution(0.1)
            .proba_distribution(logits, masks)
            .log_prob(torch.tensor([action]))
        )
        loss.backward()
        optimizer.step()
        assert (
            GroupedDistribution(0.1).proba_distribution(logits, masks).probs[0, action].item()
            > before
        )


def test_missing_cuda_compiler_backend_never_enters_tracing(monkeypatch):
    from pvz_rl.spatial_policy import SpatialGroupedPolicy

    monkeypatch.setattr("pvz_rl.spatial_policy.importlib.util.find_spec", lambda _: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unavailable compiler must not trace beside the collector")

    monkeypatch.setattr(torch, "compile", unexpected)
    policy = SimpleNamespace(device=torch.device("cuda"))
    SpatialGroupedPolicy.enable_compilation(policy)
    assert policy.compilation_status == "unavailable" and "Triton" in policy.compilation_error
