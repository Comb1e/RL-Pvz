"""Independent controls for encoder isolation, deferred values and weight transfer."""

import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.monitoring.metrics import task_statistics
from pvz_rl.policy.spatial_policy import SpatialFeatures, SpatialGroupedPolicy


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
    from pvz_rl.learning.training import vector_env

    cfg = load_config()
    cfg["training"].update(n_envs=3, batch_size=128, role_phase_games=24)
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
