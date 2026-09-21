"""Independent forced-action, singleton and CPU/CUDA reduction controls."""

import pytest
import torch

from pvz_rl.optimization import actor_weights, policy_advantages, weighted_mean


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("choices", [0, 1, 3])
def test_forced_states_cannot_dilute_or_change_actor_gradient(device, choices):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mask = torch.zeros(5, 406, dtype=torch.bool, device=device)
    mask[:, 0] = True
    mask[:choices, 46] = True
    advantages = torch.tensor([2.0, -1.0, 8.0, 1000.0, -500.0], device=device)
    w = actor_weights(mask, advantages, "choice_points_v1")
    norm = policy_advantages(advantages, w, True)
    logits = torch.zeros(5, device=device, requires_grad=True)
    loss = -weighted_mean(logits.exp() * norm, w)
    loss.backward()
    expected = torch.zeros_like(logits)
    if choices:
        selected = advantages[:choices]
        if choices > 1:
            selected = (selected - selected.mean()) / (selected.std() + 1e-8)
        expected[:choices] = -selected / choices
    torch.testing.assert_close(logits.grad, expected)
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_legacy_default_retains_all_steps_and_invalid_objective_fails():
    values = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(actor_weights(None, values), torch.ones_like(values))
    with pytest.raises(ValueError, match="masked"):
        actor_weights(None, values, "choice_points_v1")


@pytest.mark.learning
def test_flat_choice_objective_reloads_custom_optimizer(smoke_cfg, tmp_path):
    from pvz_rl.cuda_ppo import ResearchMaskablePPO
    from pvz_rl.training import load_policy, train

    smoke_cfg["training"]["actor_objective"] = "choice_points_v1"
    run = train(
        smoke_cfg, "masked", 101, tmp_path / "flat", family="diagnostic", validation_limit=1
    )
    model, _ = load_policy(run / "final.zip", "cpu")
    assert isinstance(model, ResearchMaskablePPO)
    assert model.actor_objective == "choice_points_v1"


@pytest.mark.parametrize("spatial", [False, True])
def test_dig_initialization_is_trainable_and_checkpointed(spatial, tmp_path):
    from pvz_rl.config import load_config
    from pvz_rl.cuda_ppo import ResearchMaskablePPO
    from pvz_rl.env import PvZEnv
    from pvz_rl.training import build_model

    torch.set_num_threads(1)
    cfg = load_config("configs/sc2-plant-rewards.toml")
    cfg["simulation"]["backend"] = "cpu"
    cfg["training"].update(device="cpu", n_envs=1, rollout_size=128, batch_size=128)
    if not spatial:
        cfg["policy"]["kind"] = "grouped_v1"
    env = PvZEnv(cfg)
    model = build_model(cfg, "masked", env, 101)
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(obs).unsqueeze(0)
    mask = torch.zeros(1, 406, dtype=torch.bool)
    mask[:, 0] = mask[:, 361] = True
    distribution = model.policy.get_distribution(x, action_masks=mask)
    assert 0 < distribution.probs[0, 361] < 0.004
    assert distribution.probs[0, 1:361].sum() == 0
    (-distribution.log_prob(torch.tensor([361]))).backward()
    bias = model.policy.action_net.type_head[-1].bias if spatial else model.policy.action_net.bias
    assert bias.grad[9].isfinite() and bias.grad[9].abs() > 0.9
    # Learning may make digging preferable; it is not a permanent suppression rule.
    with torch.no_grad():
        bias[9] = 6
    assert model.policy.get_distribution(x, action_masks=mask).mode().item() == 361
    path = tmp_path / "policy.zip"
    model.save(path)
    loaded = ResearchMaskablePPO.load(path, device="cpu")
    actual = loaded.policy.get_distribution(x, action_masks=mask).probs
    expected = model.policy.get_distribution(x, action_masks=mask).probs
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
