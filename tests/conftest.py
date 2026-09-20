import pytest

from pvz_rl.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def smoke_cfg(cfg):
    # Most learning tests exercise the algorithm; dedicated visualization tests
    # explicitly enable reports/video to avoid encoding dozens of duplicate demos.
    cfg["visualization"].update(enabled=False, videos=False)
    cfg["environment"]["cutoff_seconds"] = 2
    cfg["training"].update(
        device="cpu",  # Keep general regressions portable; a dedicated test exercises CUDA.
        total_steps=128,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        n_envs=1,
        hidden_sizes=[32, 32],
        eval_interval=64,
    )
    return cfg
