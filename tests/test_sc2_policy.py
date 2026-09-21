"""Independent controls for structured exploration and spatial placement."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pvz_game import Dig, Place

from pvz_rl.config import load_config, validate_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.deadline import RunBudget
from pvz_rl.env import PvZEnv
from pvz_rl.exploration import exploration_loss
from pvz_rl.grouped_policy import GroupedDistribution
from pvz_rl.sc2_experiments import comparison_protocol, sc2_profile
from pvz_rl.training import build_model, vector_env


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def test_balanced_bonus_does_not_push_wait_toward_one_over_136():
    logits = torch.zeros((1, 415), requires_grad=True)
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
    assert true_gradient[0, 0].item() == pytest.approx(-3 * np.log(45) / 16)
    policy = SimpleNamespace(
        action_dist=dist,
        exploration_settings={
            "objective": "balanced_heads_v1",
            "type_coef": 0.01,
            "tile_coef": 0.001,
        },
    )
    loss, metrics = exploration_loss(policy, dist.entropy(), dist.log_prob(torch.tensor([0])), 0.01)
    assert metrics["exploration_bonus"].item() == pytest.approx(0.01 * np.log(4) + 0.001)
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(
        gradient[:, :10], torch.zeros_like(gradient[:, :10]), atol=1e-8, rtol=0
    )
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("counts", [[0] * 9, [1, 0, 45, 3, 0, 0, 0, 0, 0]])
def test_bonus_matches_independent_entropy_and_single_choice_boundaries(counts):
    rng = np.random.default_rng(102)
    logits = torch.tensor(rng.normal(size=(2, 415)), dtype=torch.float64, requires_grad=True)
    mask = torch.zeros((2, 406), dtype=torch.bool)
    mask[:, 0] = True
    for group, count in enumerate(counts):
        mask[:, 1 + group * 45 : 1 + group * 45 + count] = True
    dist = GroupedDistribution().proba_distribution(logits)
    dist.apply_masking(mask)
    policy = SimpleNamespace(
        action_dist=dist,
        exploration_settings={
            "objective": "balanced_heads_v1",
            "type_coef": 0.01,
            "tile_coef": 0.001,
        },
    )
    loss, _ = exploration_loss(policy, dist.entropy(), dist.log_prob(torch.zeros(2)), 0.01)
    p = dist.types.probs.detach().numpy()
    type_h = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=1)
    tile_h = []
    for g, count in enumerate(counts):
        if count:
            probs = dist.locations.probs[:, g, :count].detach().numpy()
            tile_h.append(-(probs * np.log(probs)).sum(1) / np.log(max(2, count)))
    expected = 0.01 * type_h + 0.001 * (np.mean(tile_h, axis=0) if tile_h else 0)
    assert -loss.item() == pytest.approx(expected.mean())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for action in torch.nonzero(mask[0]).flatten():
        actual = dist.log_prob(action.expand(2)).detach().numpy()
        np.testing.assert_allclose(
            actual, np.log(dist.probs[:, action].detach().numpy()), atol=1e-6
        )


def test_spatial_policy_shapes_gradients_and_board_dependence():
    cfg = sc2_profile("E")
    cfg["simulation"]["backend"] = "cpu"
    cfg["training"].update(n_envs=1, device="cpu", rollout_size=128, batch_size=128)
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        raw = PvZEnv(cfg)
        obs, _ = raw.reset(seed=3)
        data = torch.tensor(np.stack([obs, obs]))
        data[1, raw.encoder.slices["plants"].start] = 1
        masks = torch.tensor(np.stack([raw.action_masks()] * 2))
        features = model.policy.extract_features(data)
        assert features.shape == (2, 64 * 46)
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
        assert optimizer_ids == {id(p) for p in model.policy.parameters()}
        assert sum(p.numel() for p in model.policy.parameters()) < 1_000_000
    finally:
        env.close()


def test_stage_residency_counts_only_matching_episode_starts():
    cfg = sc2_profile("E")
    state = CurriculumState(stage=1)
    for _ in range(100):
        state.completed_episode(0)
    assert not state.observe({"saving": 20}, 100, cfg)
    assert not state.observe({"saving": 20}, 200, cfg)
    for _ in range(99):
        state.completed_episode(1)
    restored = CurriculumState(**state.to_dict())
    assert not restored.observe({"saving": 20}, 300, cfg)
    restored.completed_episode(1)
    assert restored.observe({"saving": 20}, 400, cfg)
    assert restored.name == "easy" and restored.completed_stage_games == 0


def test_wall_budget_reserves_cleanup_and_restores_consumption():
    cfg = sc2_profile("E")
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


def test_profiles_share_controls_but_keep_gamma_ablation_explicit():
    configs = [sc2_profile(letter) for letter in "ABCDEF"]
    assert len({comparison_protocol(c) for c in configs}) == 1
    assert configs[-1]["reward"]["gamma"] == 0.9999
    assert all(c["reward"]["gamma"] == 0.999 for c in configs[:-1])
    assert all(c["training"]["eval_interval_games"] == 1000 for c in configs)
    for letter, path in [
        ("E", "configs/sc2-inspired.toml"),
        ("F", "configs/sc2-long-horizon.toml"),
    ]:
        assert sc2_profile(letter) == load_config(path)
    bad = sc2_profile("C")
    bad["policy"]["kind"] = "flat"
    with pytest.raises(ValueError, match="grouped"):
        validate_config(bad)


def test_early_dig_metric_tracks_actual_accepted_actions(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(seed=1)
    env.step(env.codec.encode(Place("sunflower", 0, 0)))
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1
