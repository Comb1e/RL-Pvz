"""Fixed-data PPO and discount controls independent of the regularizer helper."""

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.env import PvZEnv
from pvz_rl.training import build_model, vector_env


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_balanced_ppo_update_matches_joint_probability_reference(forced, device):
    torch.set_num_threads(1)
    cfg = load_config()
    cfg["runtime"]["cache_rollout_on_device"] = False
    cfg["training"].update(
        n_envs=1,
        rollout_steps_per_env=32,
        rollout_size=32,
        batch_size=32,
        device="cuda",
        n_epochs=1,
        hidden_sizes=[16, 16],
        target_kl=0,
    )
    env = vector_env(cfg, "masked", 102)
    try:
        model = build_model(cfg, "masked", env, 102)
        model.set_logger(configure(format_strings=[]))
        model.policy.to(device)
        model.device = torch.device(device)
        for key, value in vars(model.rollout_buffer).items():
            if isinstance(value, torch.Tensor):
                setattr(model.rollout_buffer, key, value.to(device))
        model.rollout_buffer.device = torch.device(device)
        reference = type(model.policy)(
            model.observation_space,
            model.action_space,
            lambda _: cfg["training"]["learning_rate"],
            **model.policy_kwargs,
        ).to(device)
        reference.load_state_dict(model.policy.state_dict())
        reference.action_dist.epsilon = cfg["training"]["exploration"]["epsilon"]
        rng = np.random.default_rng(44)
        raw = PvZEnv(cfg)
        observation, _ = raw.reset(seed=7)
        obs = np.repeat(observation[None], 32, axis=0)
        mask = np.repeat(raw.action_masks()[None], 32, axis=0)
        actions = rng.choice(np.flatnonzero(mask[0]), size=32)
        if forced:
            mask[:16, 1:] = False
            actions[:16] = 0
        # Deliberately exercise both clipped and unclipped policy ratios.
        with torch.no_grad():
            _, logs, _ = reference.evaluate_actions(
                torch.tensor(obs, device=device),
                torch.tensor(actions, device=device),
                action_masks=mask,
            )
        old_logs = logs + torch.linspace(-0.7, 0.7, 32, device=device)
        advantages = torch.tensor(rng.normal(size=32), dtype=torch.float32, device=device)
        returns = torch.tensor(rng.normal(size=32), dtype=torch.float32, device=device)
        buffer = model.rollout_buffer
        for key, value in {
            "observations": obs,
            "actions": actions[:, None],
            "values": np.zeros(32),
            "log_probs": old_logs.cpu().numpy(),
            "advantages": advantages.cpu().numpy(),
            "returns": returns.cpu().numpy(),
            "action_masks": mask,
        }.items():
            getattr(buffer, key).copy_(
                torch.as_tensor(
                    np.asarray(value).reshape(getattr(buffer, key).shape), device=device
                )
            )
        buffer.full = True
        values, _, _ = reference.evaluate_actions(
            torch.tensor(obs, device=device),
            torch.tensor(actions, device=device),
            action_masks=mask,
        )
        joint = reference.action_dist.probs
        group_mass = joint[:, 1:].reshape(32, 9, 45).sum(-1)
        types = torch.cat((joint[:, :1], group_mass), 1)
        conditional = joint[:, 1:].reshape(32, 9, 45) / group_mass.clamp_min(1e-30).unsqueeze(-1)

        def h(p):
            return -(p * p.clamp_min(1e-30).log()).sum(-1)

        counts = torch.tensor(mask[:, 1:].reshape(32, 9, 45).sum(-1), device=device)
        available = counts > 0
        bonus = 0.01 * h(types) + 0.001 * (
            h(conditional) / counts.clamp_min(2).float().log() * available
        ).sum(-1) / available.sum(-1).clamp_min(1)
        logs = joint[torch.arange(32, device=device), torch.tensor(actions, device=device)].log()
        ratio = (logs - old_logs).exp()
        selected = slice(None)
        chosen = advantages[selected]
        normalized = (chosen - chosen.mean()) / (chosen.std() + 1e-8)
        pg = -torch.minimum(
            ratio[selected] * normalized, ratio[selected].clamp(0.8, 1.2) * normalized
        ).mean()
        loss = pg - bonus[selected].mean() + 0.5 * (values.flatten() - returns).square().mean()
        reference.optimizer.zero_grad()
        reference.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.actor_parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(reference.critic_parameters(), 0.5)
        reference.optimizer.step()
        reference.critic_optimizer.step()
        np.random.seed(44)
        model.train()
        assert model.logger.name_to_value["train/loss"] == pytest.approx(loss.item(), abs=2e-6)
        assert model.logger.name_to_value["train/exploration_bonus"] == pytest.approx(
            bonus[selected].mean().item(), abs=1e-7
        )
        for actual, expected in zip(model.policy.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-6)
    finally:
        env.close()


@pytest.mark.parametrize("gamma", [0.999, 0.9999])
def test_physical_time_discount_has_no_zero_time_action_advantage(gamma):
    rewards = np.array([0.1, -0.3, 1.0])
    times = np.array([0, 50, 99])
    original = np.dot(rewards, gamma**times)
    inserted_rewards = np.array([0, 0.1, 0, -0.3, 0, 0, 1.0])
    inserted_times = np.array([0, 0, 50, 50, 99, 99, 99])
    assert np.dot(inserted_rewards, gamma**inserted_times) == pytest.approx(original)
    assert np.dot(rewards, gamma ** (times + 10)) == pytest.approx(original * gamma**10)
