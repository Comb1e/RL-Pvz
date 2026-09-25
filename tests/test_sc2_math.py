"""Fixed-data PPO and discount controls independent of the regularizer helper."""

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.env import PvZEnv
from pvz_rl.training import build_model, vector_env


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_balanced_ppo_update_matches_joint_probability_reference(forced, device, batch_size):
    torch.set_num_threads(1)
    cfg = load_config()
    cfg["runtime"]["cache_rollout_on_device"] = False
    cfg["training"].update(
        n_envs=1,
        rollout_steps_per_env=32,
        rollout_size=32,
        batch_size=batch_size,
        device="cuda",
        n_epochs=1,
        hidden_sizes=[16, 16],
        target_kl=0,
    )
    env = vector_env(cfg, "masked", 102)
    try:
        model = build_model(cfg, "masked", env, 102)
        model.set_logger(configure(format_strings=[]))
        model.policy.to(device=device, dtype=torch.float64)
        model.device = torch.device(device)
        for key, value in vars(model.rollout_buffer).items():
            if isinstance(value, torch.Tensor):
                setattr(
                    model.rollout_buffer,
                    key,
                    value.to(
                        device=device,
                        dtype=torch.float64 if value.is_floating_point() else value.dtype,
                    ),
                )
        model.rollout_buffer.device = torch.device(device)
        reference = type(model.policy)(
            model.observation_space,
            model.action_space,
            lambda _: cfg["training"]["learning_rate"],
            **model.policy_kwargs,
        ).to(device=device, dtype=torch.float64)
        reference.load_state_dict(model.policy.state_dict())
        reference.action_dist.epsilon = cfg["training"]["exploration"]["warmup_epsilon"]
        rng = np.random.default_rng(44)
        raw = PvZEnv(cfg)
        observation, _ = raw.reset(seed=7)
        obs = np.repeat(observation[None], 32, axis=0).astype(np.float64)
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
        advantages = torch.tensor(rng.normal(size=32), dtype=torch.float64, device=device)
        returns = torch.tensor(rng.normal(size=32), dtype=torch.float64, device=device)
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
        np.random.seed(44)
        order = np.random.permutation(32)
        losses, bonuses = [], []
        for start in range(0, 32, batch_size):
            selected = torch.tensor(order[start : start + batch_size], device=device)
            values, _, _ = reference.evaluate_actions(
                torch.tensor(obs, device=device),
                torch.tensor(actions, device=device),
                action_masks=mask,
            )
            joint = reference.action_dist.probs
            group_mass = joint[:, 1:].reshape(32, 9, 45).sum(-1)
            plant_mass = group_mass[:, :8].sum(-1, keepdim=True)
            types = torch.cat((joint[:, :1], group_mass[:, -1:], plant_mass), 1)
            species = group_mass[:, :8] / plant_mass.clamp_min(1e-30)
            conditional = joint[:, 1:].reshape(32, 9, 45) / group_mass.clamp_min(1e-30).unsqueeze(
                -1
            )

            def h(p):
                return -(p * p.clamp_min(1e-30).log()).sum(-1)

            counts = torch.tensor(mask[:, 1:].reshape(32, 9, 45).sum(-1), device=device)
            available = counts > 0
            species_count = available[:, :8].sum(-1)
            bonus = (
                0.01 * h(types)
                + 0.001 * h(species) / species_count.clamp_min(2).double().log()
                + 0.001
                * (h(conditional) / counts.clamp_min(2).float().log() * available).sum(-1)
                / available.sum(-1).clamp_min(1)
            )
            logs = joint[
                torch.arange(32, device=device), torch.tensor(actions, device=device)
            ].log()
            ratio = (logs - old_logs).exp()
            chosen = advantages
            normalized = (chosen - chosen.mean()) / (chosen.std() + 1e-8)
            pg = -torch.minimum(
                ratio[selected] * normalized[selected],
                ratio[selected].clamp(0.8, 1.2) * normalized[selected],
            ).mean()
            loss = (
                pg
                - bonus[selected].mean()
                + 0.5 * (values.flatten()[selected] - returns[selected]).square().mean()
            )
            reference.optimizer.zero_grad()
            reference.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.actor_parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(reference.critic_parameters(), 0.5)
            reference.optimizer.step()
            reference.critic_optimizer.step()
            losses.append(loss.item())
            bonuses.append(bonus[selected].mean().item())
        np.random.seed(44)
        model.train()
        assert model.logger.name_to_value["train/loss"] == pytest.approx(np.mean(losses), abs=2e-6)
        assert model.logger.name_to_value["train/exploration_bonus"] == pytest.approx(
            np.mean(bonuses), abs=1e-7
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
