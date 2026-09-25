"""Independent controls for encoder isolation, deferred values and weight transfer."""

import copy
import json

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.save_util import save_to_zip_file

from pvz_rl.config import load_config, validate_config
from pvz_rl.cuda_buffer import TensorRolloutBuffer
from pvz_rl.env import PvZEnv
from pvz_rl.metrics import task_statistics
from pvz_rl.spatial_policy import SpatialFeatures, SpatialGroupedPolicy
from pvz_rl.training import initial_weights


def policy_and_state():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    cfg = load_config()
    env = PvZEnv(cfg, level="easy")
    obs, _ = env.reset(seed=4)
    policy = SpatialGroupedPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        features_extractor_class=SpatialFeatures,
        features_extractor_kwargs={"layout_cfg": cfg},
    )
    return policy, torch.tensor(obs).unsqueeze(0), env.action_masks()[None], cfg


def test_complete_disjoint_parameter_and_optimizer_ownership():
    policy, _, _, _ = policy_and_state()
    actor = {id(p) for p in policy.actor_parameters()}
    critic = {id(p) for p in policy.critic_parameters()}
    assert actor.isdisjoint(critic)
    assert actor | critic == {id(p) for p in policy.parameters()}
    assert sum(p.numel() for p in policy.parameters()) == 1_128_060
    assert {id(p) for g in policy.optimizer.param_groups for p in g["params"]} == actor
    assert {id(p) for g in policy.critic_optimizer.param_groups for p in g["params"]} == critic


@pytest.mark.parametrize("update", ["actor", "critic"])
def test_update_cannot_change_other_network_or_optimizer(update):
    policy, obs, mask, _ = policy_and_state()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    with torch.no_grad():
        probs = policy.get_distribution(obs, mask).probs.clone()
        value = policy.predict_values(obs).clone()
    for _ in range(3):
        if update == "critic":
            loss = (policy.predict_values(obs) - 3).square().mean()
            optimizer, params = policy.critic_optimizer, policy.critic_parameters()
        else:
            dist = policy.get_distribution(obs, mask)
            loss = -dist.log_prob(torch.tensor([1])).mean() - 0.01 * dist.entropy().mean()
            optimizer, params = policy.optimizer, policy.actor_parameters()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        optimizer.step()
    with torch.no_grad():
        if update == "critic":
            torch.testing.assert_close(
                policy.get_distribution(obs, mask).probs, probs, rtol=0, atol=0
            )
            assert not policy.optimizer.state
        else:
            torch.testing.assert_close(policy.predict_values(obs), value, rtol=0, atol=0)
            assert not policy.critic_optimizer.state
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [1, 4, 1024])
def test_deferred_critic_timeout_values_and_gae_match_independent_control(device, batch_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    observation = spaces.Box(-100, 100, (2,), dtype=np.float32)
    action = spaces.Discrete(2)
    buffer = TensorRolloutBuffer(
        5, observation, action, device=device, n_envs=3, gamma=0.999, gae_lambda=0.999
    )
    reference = RolloutBuffer(
        5, observation, action, device="cpu", n_envs=3, gamma=0.999, gae_lambda=0.999
    )
    sizes = []

    class Critic:
        def predict_values(self, obs):
            sizes.append(len(obs))
            return obs[:, :1] * 0.3 + obs[:, 1:] * 0.7

    for step in range(5):
        obs = np.array([[step, j + 1] for j in range(3)], np.float32)
        rewards = np.array([0.1, -0.1, 0.2], np.float32)
        starts = np.array([step in (0, 2), step in (0, 3), step == 0], np.float32)
        if step in (1, 4):
            # Terminal differs sharply from reset/current states; bootstrap exactly once.
            terminal = torch.tensor([[7.0, 9.0]], device=device)
            buffer.add_timeouts(torch.tensor([0], device=device), terminal)
            ref_reward = rewards.copy()
            ref_reward[0] += 0.999 * (7 * 0.3 + 9 * 0.7)
        else:
            ref_reward = rewards
        values = torch.tensor(obs[:, 0] * 0.3 + obs[:, 1] * 0.7)
        reference.add(obs, np.zeros((3, 1)), ref_reward, starts, values, torch.zeros(3))
        buffer.add(
            torch.tensor(obs, device=device),
            torch.zeros(3, device=device),
            torch.tensor(rewards, device=device),
            torch.tensor(starts, device=device),
            None,
            torch.zeros(3, device=device),
            torch.ones(3, 2, dtype=torch.bool, device=device),
        )
    last_obs = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], device=device)
    last_values = buffer.evaluate_values(Critic(), last_obs, batch_size)
    assert max(sizes) <= batch_size
    torch.testing.assert_close(
        buffer.values.cpu(), torch.tensor(reference.values), rtol=1e-6, atol=1e-6
    )
    rewards = buffer.rewards.clone()
    buffer.evaluate_values(Critic(), last_obs, batch_size)
    assert torch.equal(buffer.rewards, rewards)  # No double timeout correction.
    done = torch.tensor([True, True, False], device=device)
    buffer.compute_returns_and_advantage(last_values, done)
    reference.compute_returns_and_advantage(last_values.cpu(), done.cpu().numpy())
    np.testing.assert_allclose(buffer.advantages.cpu(), reference.advantages, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(buffer.returns.cpu(), reference.returns, rtol=2e-6, atol=2e-6)


def test_current_weights_only_transfer_preserves_outputs_and_rejects_old_reward(tmp_path):
    original, obs, mask, cfg = policy_and_state()
    old = copy.deepcopy(cfg)
    metadata = {"config": old, "condition": "masked", "learner_seed": 101}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    checkpoint = tmp_path / "source.zip"
    save_to_zip_file(
        checkpoint,
        data={"num_timesteps": 123, "training_games": 6, "_n_updates": 4},
        params={"policy": original.state_dict()},
    )
    converted, meta = initial_weights(checkpoint, cfg)
    target, _, _, _ = policy_and_state()
    target.load_state_dict(converted)
    with torch.no_grad():
        torch.testing.assert_close(
            target.get_distribution(obs, mask).probs,
            original.get_distribution(obs, mask).probs,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            target.predict_values(obs), original.predict_values(obs), rtol=0, atol=0
        )
    assert not target.optimizer.state and not target.critic_optimizer.state
    assert meta["conversion"] == "none" and meta["steps"] == 123
    old["reward"]["version"] = "potential_mower_v1"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fresh"):
        initial_weights(checkpoint, cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("critic_learning_rate", 0),
        ("critic_learning_rate", float("nan")),
        ("value_batch_size", 0),
        ("value_batch_size", True),
    ],
)
def test_invalid_value_controls(field, value):
    cfg = load_config()
    cfg["training"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(cfg)


def test_early_dig_ratios_do_not_confuse_more_planting_with_regression():
    rows = [
        dict(
            win=1,
            simulated_seconds=100,
            early_voluntary_digs=2,
            attacker_purchases=4,
            plant_usage={"sunflower": 6, "peashooter": 4},
        )
    ]
    result = task_statistics(rows)
    assert result["early_digs_per_game"] == 2
    assert result["early_digs_per_planting"] == 0.2
    assert result["plant_usage"]["peashooter"] == 4
    assert task_statistics([])["win_rate"] is None
    rows[0]["plant_usage"] = {}
    assert task_statistics(rows)["early_digs_per_planting"] is None


def test_task_counts_track_partial_games_and_reset_at_episode_boundaries():
    from pvz_rl.training import vector_env

    cfg = load_config()
    cfg["training"].update(n_envs=3, rollout_steps_per_env=128, rollout_size=384, batch_size=128)
    cfg["environment"]["cutoff_seconds"] = 1
    env = vector_env(cfg, "masked", 101)
    try:
        env.reset()
        for _ in range(101):
            env.step_tensors(torch.zeros(3, device="cuda", dtype=torch.long))
        counts = env.task_counts()
        assert sum(v["started_games"] for v in counts.values()) == 6
        assert sum(v["active_games"] for v in counts.values()) == 3
        assert sum(v["completed_games"] for v in counts.values()) == 3
        assert sum(v["transitions"] for v in counts.values()) == 303
        env.env_method("set_curriculum_stage", 2)
        before = env.task_counts()
        assert env.task_counts() == before
        env.reset_indices([0], cases=[("easy", "preset", 1000)])
        assert env.task_counts()["easy"]["active_games"] == 1
        assert sum(v["active_games"] for v in env.task_counts().values()) == 3
    finally:
        env.close()


def test_deferred_collection_never_calls_critic_per_action(tmp_path):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.training import build_model, vector_env

    cfg = load_config()
    cfg["training"].update(
        n_envs=2, rollout_steps_per_env=32, rollout_size=64, batch_size=32, value_batch_size=16
    )
    cfg["environment"]["cutoff_seconds"] = 1
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        model.set_logger(configure(format_strings=[]))
        calls = []
        original = model.policy.predict_values

        def predict(obs, **kwargs):
            calls.append(len(obs))
            return original(obs, **kwargs)

        model.policy.predict_values = predict

        class Callback(BaseCallback):
            def _on_step(self):
                assert not calls
                return True

        _, cb = model._setup_learn(64, Callback())
        original_sample = model.policy.sample_actions

        def sample(*args, **kwargs):
            assert not calls  # Critic inference is deferred until collection ends.
            return original_sample(*args, **kwargs)

        model.policy.sample_actions = sample
        model.collect_slot(
            env,
            model.policy,
            model.rollout_buffer,
            model._last_obs,
            model._last_episode_starts,
            None,
            0,
            "control",
        )
        assert calls and max(calls) <= 16
        assert not model.policy.optimizer.state and not model.policy.critic_optimizer.state
    finally:
        env.close()


def test_action_inference_never_evaluates_critic():
    policy, obs, mask, _ = policy_and_state()

    def forbidden(*args, **kwargs):
        raise AssertionError("Action inference invoked the critic")

    policy.vf_features_extractor.forward = forbidden
    policy.value_net.forward = forbidden
    with torch.no_grad():
        policy.sample_actions(obs, mask)
        policy.predict(obs.numpy(), deterministic=True, action_masks=mask)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neural_critic_batching_preserves_predictions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy, obs, _, _ = policy_and_state()
    policy.to(device).set_training_mode(False)
    batch = obs.repeat(17, 1).to(device)
    # Distinct public sun inputs; category fields stay valid.
    sun_index = policy.vf_features_extractor.layout.slices["globals"].start
    batch[:, sun_index] = torch.linspace(0, 9, len(batch), device=device)
    # Select true FP32 convolutions for this mathematical batching control.
    # cuDNN's default TF32 kernels vary by batch shape; they are measured separately.
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        per_step = torch.cat([policy.predict_values(row[None]) for row in batch])
        batched = policy.predict_values(batch)
        policy.double()
        reference = policy.predict_values(batch.double())
        single_reference = torch.cat([policy.predict_values(row[None].double()) for row in batch])
    torch.testing.assert_close(reference, single_reference, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(batched.double(), reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(per_step.double(), reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(batched, per_step, rtol=2e-6, atol=2e-6)


def test_normalization_and_reward_defaults():
    cfg = load_config()
    assert cfg["training"]["learning_rate"] == 1e-4
    assert cfg["training"]["critic_learning_rate"] == 3e-4
    assert cfg["training"]["gamma"] == 1
    assert cfg["training"]["gae_lambda"] == pytest.approx(0.999**0.4)
    assert cfg["reward"] == dict(
        version="net_value_v1",
        win_reward=1.0,
        loss_penalty=2.0,
        mower_value=200.0,
        basic_zombie_value=50.0,
        progress_weight=0.01,
        value_scale=300.0,
    )
    assert cfg["training"]["exploration"] == dict(
        objective="balanced_heads_v2",
        type_coef=0.01,
        plant_coef=0.001,
        tile_coef=0.001,
        epsilon=0.1,
        epsilon_target=0.001,
        epsilon_target_games=3000,
        entropy_target_fraction=0.01,
    )
