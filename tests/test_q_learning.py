"""Independent controls for shared encoder, deferred values and weight transfer."""

import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.monitoring.metrics import task_statistics
from pvz_rl.policy.spatial_policy import SequentialQPolicy, SpatialFeatures


def policy_and_state():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    cfg = load_config()
    env = PvZEnv(cfg, level="easy")
    obs, _ = env.reset(seed=4)
    policy = SequentialQPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        hidden_sizes=[128, 128],
        features_extractor_class=SpatialFeatures,
        features_extractor_kwargs={"layout_cfg": cfg},
    )
    return policy, torch.tensor(obs).unsqueeze(0), env.action_masks()[None], cfg


def test_one_optimizer_owns_entire_shared_network():
    policy, obs, _, _ = policy_and_state()
    assert {id(p) for g in policy.optimizer.param_groups for p in g["params"]} == {
        id(p) for p in policy.parameters()
    }
    assert not hasattr(policy, "critic_optimizer")
    actions = torch.tensor([1])
    before = tuple(q.detach().clone() for q in policy.selected_values(obs, actions))
    for _ in range(3):
        first, second = policy.selected_values(obs, actions)
        loss = ((first - 1).square() + (second - 1).square()).mean() / 2
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
    after = policy.selected_values(obs, actions)
    assert all(bool(a > b) for a, b in zip(after, before))
    assert policy.optimizer.defaults["eps"] == 1e-5


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
    cfg["training"].update(n_envs=3, batch_size=128)
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
def test_neural_q_batching_preserves_predictions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy, obs, _, _ = policy_and_state()
    policy.to(device).set_training_mode(False)
    batch = obs.repeat(17, 1).to(device)
    with torch.no_grad():
        policy.branch_head[-1].weight.normal_(std=0.01)
    # Distinct public sun inputs; category fields stay valid.
    sun_index = policy.features_extractor.layout.slices["globals"].start
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
