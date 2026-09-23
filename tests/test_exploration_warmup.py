"""Independent mixture probabilities and stage-specific critic adaptation controls."""

import copy
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
    raw = rng.normal(size=(1, 415))
    logits = torch.tensor(raw, device=device, dtype=torch.float64, requires_grad=True)
    masks = torch.zeros((1, 406), dtype=torch.bool, device=device)
    masks[:, [0, 1, 3, 46, 361, 365]] = True
    dist = GroupedDistribution(epsilon).proba_distribution(logits)
    dist.apply_masking(masks)
    # Enumerate this six-action case independently of the implementation.
    active = [0, 1, 2, 9]
    p = np.exp(raw[0, active] - raw[0, active].max())
    p /= p.sum()
    p = (1 - epsilon) * p + epsilon * np.array([1 / 3, 1 / 3, 1 / 3, 0])
    expected = np.zeros(406)
    expected[0] = p[0]
    for index, group, tiles in [(1, 0, [0, 2]), (2, 1, [0]), (3, 8, [0, 4])]:
        tile_logits = raw[0, 10 + 45 * group + np.array(tiles)]
        q = np.exp(tile_logits - tile_logits.max())
        expected[1 + 45 * group + np.array(tiles)] = p[index] * q / q.sum()
    np.testing.assert_allclose(dist.probs.detach().cpu()[0], expected, atol=2e-8)
    assert dist.probs.sum().item() == pytest.approx(1)
    assert not dist.probs[~masks].any()
    for action in [0, 1, 3, 46, 361, 365]:
        assert dist.log_prob(torch.tensor([action], device=device)).item() == pytest.approx(
            np.log(expected[action]), abs=2e-7
        )
    nonzero = expected[expected > 0]
    assert dist.entropy().item() == pytest.approx(-(nonzero * np.log(nonzero)).sum(), abs=2e-7)
    loss = -dist.log_prob(torch.tensor([1], device=device)).sum()
    loss.backward()
    # Analytical derivative of -log((1-eps)*softmax(type)[flower] + eps/3).
    base = np.exp(raw[0, active] - raw[0, active].max())
    base /= base.sum()
    gradient = (1 - epsilon) * base[1] / p[1] * (base - np.array([0, 1, 0, 0]))
    np.testing.assert_allclose(logits.grad.detach().cpu()[0, active], gradient, atol=2e-7)
    assert torch.isfinite(logits.grad).all()


def test_collapse_floor_greedy_policy_and_forced_action_boundaries():
    logits = torch.zeros((1, 415))
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
    assert ((samples == 1).float().mean().item()) == pytest.approx(0.05 / 3, abs=0.003)
    # Dig can still be the learned greedy choice, regardless of injected exploration.
    logits[:, 9], logits[:, 0] = 100, 99.999
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
        ("epsilon", -0.1),
        ("epsilon", 1),
        ("epsilon", float("nan")),
        ("epsilon_target_games", -1),
        ("epsilon_target_games", 1.5),
        ("epsilon_target_games", float("inf")),
        ("epsilon_target_games", 1024),
        ("epsilon_target", 0),
        ("epsilon_target", float("nan")),
        ("epsilon_target", 0.2),
        ("critic_warmup_games", -1),
        ("critic_warmup_games", 1.5),
    ],
)
def test_invalid_exploration_and_warmup(key, value):
    cfg = load_config()
    target = cfg["training"]["exploration"] if key.startswith("epsilon") else cfg["training"]
    target[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


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
    cfg = load_config()
    for games, expected in [
        (0, 0.1),
        (1023, 0.1),
        (1024, 0.1),
        (2012, 0.01),
        (3000, 0.001),
        (4976, 0.00001),
    ]:
        assert exploration_rate(cfg, games) == pytest.approx(expected)
    assert exploration_rate(cfg, 1025) < exploration_rate(cfg, 1024)
    # Diagnostic/fixed-task runs have no critic adaptation period to hold through.
    assert exploration_rate(cfg, 1, staged=False) < 0.1
    assert exploration_rate(cfg, 1500, staged=False) == pytest.approx(0.01)
    assert exploration_rate(cfg, 3000, staged=False) == pytest.approx(0.001)
    state = CurriculumState(stage=1, completed_stage_games=2012)
    restored = CurriculumState(**state.to_dict())
    assert exploration_rate(cfg, restored.completed_stage_games) == pytest.approx(0.01)
    assert exploration_rate(cfg, 1_000_000) == 0  # Harmless floating-point underflow.
    # The same additional game count multiplies the rate by the same factor.
    for games in (1024, 2048, 4096):
        assert exploration_rate(cfg, games + 988) == pytest.approx(
            exploration_rate(cfg, games) / 10, rel=1e-12
        )
    state.observe({"saving": 100}, 3000, cfg)
    assert exploration_rate(cfg, state.completed_stage_games) == 0.1
    cfg["training"]["exploration"]["epsilon"] = 0
    validate_config(cfg)
    assert exploration_rate(cfg, 9000) == 0
    cfg["training"]["exploration"] = dict(
        objective="balanced_heads_v1", type_coef=0.01, tile_coef=0.001, epsilon=0.002
    )
    assert exploration_rate(cfg, 9000) == 0.002  # Archived constant rate.


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
        n_envs=1, rollout_steps_per_env=16, rollout_size=16, batch_size=16, n_epochs=2
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
        rollout_steps_per_env=32,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        total_games=12,
        critic_warmup_games=4,
    )
    cfg["training"]["exploration"]["epsilon_target_games"] = 8
    original = ResearchCallback._on_rollout_start

    def interrupt(self):
        if self.model.num_timesteps:
            raise KeyboardInterrupt()
        return original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, tmp_path / "first", validation_limit=1)
    first, _ = load_policy(tmp_path / "first/interrupted.zip", "cuda")
    assert first.curriculum_state["completed_stage_games"] == 2
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
    assert 0 < last_rate < cfg["training"]["exploration"]["epsilon"]
    assert final.policy.action_dist.epsilon == last_rate
    # Old archived configurations retain their original sampling and update schedule.
    archived = copy.deepcopy(cfg)
    for key in ("epsilon", "epsilon_target", "epsilon_target_games"):
        archived["training"]["exploration"].pop(key)
    archived["training"].pop("critic_warmup_games")
    validate_config(archived)
