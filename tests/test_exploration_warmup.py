"""Independent mixture probabilities and stage-specific critic adaptation controls."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config, validate_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.exploration import configure_exploration, exploration_rate, set_exploration_rate
from pvz_rl.grouped_policy import GroupedDistribution
from pvz_rl.training import ResearchCallback, build_model, load_policy, train, vector_env
from pvz_rl.visualization import read_series


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.9])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixture_joint_probabilities_entropy_and_gradient(epsilon, device):
    rng = np.random.default_rng(34)
    raw = rng.normal(size=(1, 416))
    logits = torch.tensor(raw, device=device, dtype=torch.float64, requires_grad=True)
    masks = torch.zeros((1, 406), dtype=torch.bool, device=device)
    masks[:, [0, 1, 3, 46, 361, 365]] = True
    dist = GroupedDistribution(epsilon).proba_distribution(logits)
    dist.apply_masking(masks)

    # Enumerate this six-action case independently of the implementation.
    def softmax(values):
        w = np.exp(values - values.max())
        return w / w.sum()

    kinds = softmax(raw[0, :3])
    species = softmax(raw[0, 3:5])
    base = np.array([kinds[0], kinds[2] * species[0], kinds[2] * species[1], kinds[1]])
    p = (1 - epsilon) * base + epsilon * np.array([1 / 3, 1 / 3, 1 / 3, 0])
    expected = np.zeros(406)
    expected[0] = p[0]
    for index, group, tiles in [(1, 0, [0, 2]), (2, 1, [0]), (3, 8, [0, 4])]:
        tile_logits = raw[0, 11 + 45 * group + np.array(tiles)]
        expected[1 + 45 * group + np.array(tiles)] = p[index] * softmax(tile_logits)
    np.testing.assert_allclose(dist.probs.detach().cpu()[0], expected, atol=1e-12)
    assert dist.probs.sum().item() == pytest.approx(1)
    assert not dist.probs[~masks].any()
    for action in [0, 1, 3, 46, 361, 365]:
        assert dist.log_prob(torch.tensor([action], device=device)).item() == pytest.approx(
            np.log(expected[action]), abs=1e-12
        )
    nonzero = expected[expected > 0]
    assert dist.entropy().item() == pytest.approx(-(nonzero * np.log(nonzero)).sum(), abs=1e-12)
    loss = -dist.log_prob(torch.tensor([1], device=device)).sum()
    loss.backward()
    # Independent derivative of the mixture's selected planting probability.
    scale = (1 - epsilon) * base[1] / p[1]
    root_gradient = scale * (kinds - np.array([0, 0, 1]))
    species_gradient = scale * (species - np.array([1, 0]))
    np.testing.assert_allclose(logits.grad.detach().cpu()[0, :3], root_gradient, atol=1e-12)
    np.testing.assert_allclose(logits.grad.detach().cpu()[0, 3:5], species_gradient, atol=1e-12)
    assert torch.isfinite(logits.grad).all()


def test_collapse_floor_greedy_policy_and_forced_action_boundaries():
    logits = torch.zeros((1, 416))
    logits[:, 0] = 100
    masks = torch.zeros((1, 406), dtype=torch.bool)
    masks[:, [0, 1, 46, 361]] = True
    dist = GroupedDistribution(0.05).proba_distribution(logits)
    dist.apply_masking(masks)
    assert dist.probs[0, 1].item() == pytest.approx(0.05 / 3)
    assert dist.probs[0, 46].item() == pytest.approx(0.05 / 3)
    assert dist.probs[0, 361].item() < 1e-30  # No destructive random dig floor.
    assert dist.mode().item() == 0
    torch.manual_seed(7)
    samples = dist.types.sample((20000,)).flatten()
    assert ((samples == 2).float().mean().item()) == pytest.approx(2 * 0.05 / 3, abs=0.003)
    # Dig can still be the learned greedy choice, regardless of injected exploration.
    logits[:, 1], logits[:, 0] = 100, 99.999
    dist.proba_distribution(logits).apply_masking(masks)
    assert dist.mode().item() == 361
    for only in [0, 1, 361]:
        masks[:] = False
        masks[:, only] = True
        dist.apply_masking(masks)
        assert dist.probs[0, only] == 1 and dist.sample().item() == only


@pytest.mark.parametrize(
    "key,value",
    [
        ("warmup_epsilon", 0),
        ("warmup_epsilon", 1.1),
        ("warmup_epsilon", float("nan")),
        ("formal_epsilon_start", 0),
        ("formal_epsilon_floor", 0),
        ("formal_epsilon_floor", 0.2),
        ("formal_entropy_start_fraction", 0),
        ("formal_entropy_floor_fraction", 0),
        ("formal_entropy_floor_fraction", 1.1),
        ("formal_decay_games", -1),
        ("formal_decay_games", 1.5),
        ("critic_warmup_games", -1),
        ("critic_warmup_games", 1.5),
    ],
)
def test_invalid_exploration_and_warmup(key, value):
    cfg = load_config()
    target = cfg["training"]["exploration"] if key != "critic_warmup_games" else cfg["training"]
    target[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_exploration_schedule_has_separate_phases_and_floors():
    from pvz_rl.exploration import exploration_state

    cfg = load_config()
    warm = exploration_state(cfg, 1023)
    start = exploration_state(cfg, 1024)
    middle = exploration_state(cfg, 2524)
    floor = exploration_state(cfg, 4024)
    assert (warm.phase, warm.epsilon, warm.entropy_factor) == ("warmup", 0.1, 1.0)
    assert (start.phase, start.epsilon, start.entropy_factor) == ("formal", 0.05, 1.0)
    assert 0.001 < middle.epsilon < 0.05
    assert 0.1 < middle.entropy_factor < 1.0
    assert floor.at_floor and floor.epsilon == pytest.approx(0.001)
    assert floor.entropy_factor == pytest.approx(0.1)
    assert exploration_state(cfg, 100000).epsilon == pytest.approx(floor.epsilon)
    assert exploration_state(cfg, 200, staged=False).phase == "formal"


def test_entropy_schedule_is_independent_of_injected_noise():
    from pvz_rl.exploration import entropy_factor, set_entropy_factor

    cfg = load_config()
    assert entropy_factor(cfg, 1024) == pytest.approx(1.0)
    assert entropy_factor(cfg, 4024) == pytest.approx(0.1)
    model = SimpleNamespace(policy=SimpleNamespace())
    set_entropy_factor(model, cfg, 0.1)
    assert model.policy.exploration_settings["type_coef"] == pytest.approx(0.001)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_exact_hierarchical_kl_matches_enumerated_actions_and_gradients(device, epsilon):
    torch.manual_seed(101)
    old_logits = torch.randn(5, 416, dtype=torch.float64, device=device)
    new_logits = torch.randn(5, 416, dtype=torch.float64, device=device, requires_grad=True)
    mask = torch.zeros(5, 406, dtype=torch.bool, device=device)
    mask[0, 0] = True
    mask[1, 361] = True
    mask[2, [0, 1, 3, 46, 365]] = True
    mask[3, [0, 361]] = True
    mask[4, [1, 4, 49]] = True
    old = GroupedDistribution(epsilon).proba_distribution(old_logits, mask)
    new = GroupedDistribution(epsilon).proba_distribution(new_logits, mask)
    p, q = old.probs, new.probs
    expected = (p * (p.clamp_min(1e-300).log() - q.clamp_min(1e-300).log())).sum(-1)
    actual = old.kl_divergence(new)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual[:2].tolist() == [0, 0]
    actual_grad = torch.autograd.grad(actual.sum(), new_logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), new_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
    old_logits.zero_()
    old_logits[3, 1] = np.log(1e-6 / (1 - 1e-6))
    with torch.no_grad():
        new_logits.zero_()
        new_logits[3, 1] = np.log(0.1 / 0.9)
    old = GroupedDistribution().proba_distribution(old_logits, mask)
    new = GroupedDistribution().proba_distribution(new_logits, mask)
    assert old.kl_divergence(new)[3].item() == pytest.approx(0.1053478973723457)
    changed = mask.clone()
    changed[0, 1] = True
    new.apply_masking(changed)
    with pytest.raises(ValueError, match="identical legal masks"):
        old.kl_divergence(new)


def test_warmup_uses_persisted_stage_residency():
    state = CurriculumState(stage=1, completed_stage_games=255)
    state.completed_episode(0)
    assert state.critic_warming_up(256)
    state = CurriculumState(**state.to_dict())
    state.completed_episode(1)
    assert not state.critic_warming_up(256)
    cfg = load_config()
    assert state.observe({"saving": 100}, 2000, cfg)
    assert state.stage == 2 and state.critic_warming_up(256)
    assert not state.critic_warming_up(0)


def test_exploration_decay_uses_stage_games_and_has_no_discontinuity():
    from pvz_rl.exploration import exploration_state

    cfg = load_config()
    assert exploration_rate(cfg, 0) == pytest.approx(0.1)
    assert exploration_rate(cfg, 1023) == pytest.approx(0.1)
    assert exploration_rate(cfg, 1024) == pytest.approx(0.05)
    assert exploration_rate(cfg, 4024) == pytest.approx(0.001)
    assert exploration_rate(cfg, 100000) == pytest.approx(0.001)
    assert 0.001 < exploration_rate(cfg, 1, staged=False) < 0.05
    assert exploration_state(cfg, 1024).phase == "formal"


def test_stage_clock_advances_without_warmup_but_never_changes_rate_mid_rollout(tmp_path):
    cfg = load_config()
    cfg["training"]["critic_warmup_games"] = 0
    cfg["curriculum"].pop("residency")  # Archived residency mode is independent.
    callback = ResearchCallback(cfg, "masked", 101, tmp_path)
    callback.model = SimpleNamespace(
        num_timesteps=128,
        training_games=4000,
        exploration_rate=0.003,
        get_env=lambda: SimpleNamespace(env_method=lambda *args: None),
    )
    callback.curriculum = CurriculumState(stage=1, completed_stage_games=2999)
    callback.stream = io.StringIO()
    callback.log_progress = lambda: None
    callback.locals = {
        "infos": [
            {
                "ticks_advanced": 1,
                "episode_metrics": {
                    "family": "saving",
                    "level": "easy",
                    "episode_start_stage": stage,
                },
            }
            for stage in (0, 1, 1)
        ]
    }
    try:
        callback._on_step()
        assert callback.curriculum.completed_stage_games == 3001
        assert callback.model.training_games == 4003
        assert callback.model.exploration_rate == 0.003
    finally:
        callback.close()


def test_critic_only_update_preserves_actor_and_adam_state(tmp_path):
    cfg = load_config()
    cfg["training"].update(
        n_envs=1,
        rollout_steps_per_env=16,
        rollout_size=16,
        batch_size=16,
        n_epochs=2,
        # A local descent control; Adam need not improve every minibatch at the
        # production rate, especially after changing encoder capacity.
        critic_learning_rate=1e-5,
    )
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        model.set_logger(configure(format_strings=[]))
        obs = env.reset().repeat(16, 1)
        masks = env.action_masks().repeat(16, 1)
        with torch.no_grad():
            actions, logs = model.policy.sample_actions(obs, masks)
            values = model.policy.predict_values(obs).flatten()
        b = model.rollout_buffer
        for i in range(16):
            b.add(
                obs[i : i + 1],
                actions[i : i + 1],
                torch.zeros(1, device="cuda"),
                torch.zeros(1, device="cuda"),
                values[i : i + 1],
                logs[i : i + 1],
                masks[i : i + 1],
            )
        b.returns.fill_(-0.3)
        b.advantages.copy_(b.returns - b.values)
        model.train()  # Establish both Adam states before freezing the actor.
        before = [p.detach().clone() for p in model.policy.actor_parameters()]
        steps = [v["step"].clone() for v in model.policy.optimizer.state.values()]
        with torch.no_grad():
            old_error = (model.policy.predict_values(obs) + 0.3).square().mean().item()
        model.critic_warmup_active = True
        model.train()
        for a, p in zip(before, model.policy.actor_parameters()):
            torch.testing.assert_close(a, p, rtol=0, atol=0)
        for old, current in zip(steps, model.policy.optimizer.state.values()):
            assert old == current["step"]
        assert model.logger.name_to_value["train/actor_optimizer_steps"] == 0
        assert model.logger.name_to_value["train/critic_optimizer_steps"] == 2
        assert model.logger.name_to_value["train/kl_stopped"] == 0
        assert model.logger.name_to_value["train/joint_entropy"] > 0
        with torch.no_grad():
            assert (model.policy.predict_values(obs) + 0.3).square().mean().item() < old_error
        # Save partway through annealing; loading must not restore the initial rate.
        from pvz_rl.cuda_ppo import CudaMaskablePPO

        set_exploration_rate(model, 0.003425)
        model.save(tmp_path / "mid-decay.zip")
        loaded = CudaMaskablePPO.load(tmp_path / "mid-decay.zip", device="cuda")
        configure_exploration(loaded, cfg)
        assert loaded.exploration_rate == loaded.policy.action_dist.epsilon == 0.003425
        with torch.no_grad():
            expected = model.policy.get_distribution(obs, masks).probs.clone()
            torch.testing.assert_close(
                loaded.policy.get_distribution(obs, masks).probs, expected, rtol=0, atol=0
            )
    finally:
        env.close()


@pytest.mark.learning
def test_stage_warmup_interrupt_resume_and_actor_release(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["curriculum"]["run_stage"] = "saving"
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["visualization"].update(enabled=False, demos=False)
    cfg["training"].update(
        n_envs=2,
        rollout_steps_per_env=128,
        rollout_size=256,
        batch_size=32,
        n_epochs=1,
        total_games=12,
        critic_warmup_games=6,
    )
    cfg["training"]["exploration"]["formal_decay_games"] = 12
    original = ResearchCallback._on_rollout_start

    def interrupt(self):
        if self.model.num_timesteps:
            raise KeyboardInterrupt()
        return original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, tmp_path / "first", validation_limit=1)
    first, _ = load_policy(tmp_path / "first/interrupted.zip", "cuda")
    assert first.curriculum_state["completed_stage_games"] == 4
    assert not first.policy.optimizer.state and first.policy.critic_optimizer.state
    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", original)
    result = train(
        cfg,
        "masked",
        101,
        tmp_path / "resumed",
        validation_limit=1,
        resume=tmp_path / "first/interrupted.zip",
    )
    records = read_series(result / "training-metrics.jsonl")
    rows = [r["optimization"] for r in records if r.get("optimization")]
    assert rows[0]["critic_warmup"] == 1 and rows[0]["actor_optimizer_steps"] == 0
    assert any(r["critic_warmup"] == 0 and r["actor_optimizer_steps"] > 0 for r in rows)
    final, _ = load_policy(result / "final.zip", "cuda")
    last_rate = next(r["exploration_rate"] for r in reversed(records) if r.get("optimization"))
    assert 0 < last_rate < cfg["training"]["exploration"]["warmup_epsilon"]
    assert final.policy.action_dist.epsilon == last_rate
