"""Digging remains legal and its initial bias is trainable."""

import torch


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
    assert 0 < distribution.probs[0, 361] < 0.004
    assert distribution.probs[0, 1:361].sum() == 0
    (-distribution.log_prob(torch.tensor([361]))).backward()
    bias = model.policy.action_net.type_head[-1].bias
    assert bias.grad[9].isfinite() and bias.grad[9].abs() > 0.9
    # Learning may make digging preferable; it is not a permanent suppression rule.
    with torch.no_grad():
        bias[9] = 6
    assert model.policy.get_distribution(x, action_masks=mask).mode().item() == 361
    path = tmp_path / "policy.zip"
    model.save(path)
    loaded = CudaMaskablePPO.load(path, device="cpu")
    actual = loaded.policy.get_distribution(x, action_masks=mask).probs
    expected = model.policy.get_distribution(x, action_masks=mask).probs
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
