import pytest

from pvz_rl.config import load_config


@pytest.fixture
def cfg():
    cfg = load_config()
    # Preserve established 10-tick controls as legacy regression cases.
    # per_tick_cfg and smoke_cfg exercise the current default protocol.
    cfg["environment"].update(action_timing="fixed", decision_ticks=10)
    cfg["training"]["budget_unit"] = "decisions"
    cfg["training"].pop("rollout_steps_per_env", None)
    cfg["simulation"] = {"backend": "cpu"}
    for key in (
        "mower_sun_weight",
        "mower_sun_scale",
        "wall_nut_damage_weight",
        "empty_explosion_penalty",
    ):
        cfg["reward"].pop(key)
    return cfg


@pytest.fixture
def per_tick_cfg():
    return load_config()


@pytest.fixture
def smoke_cfg(per_tick_cfg):
    cfg = per_tick_cfg
    cfg["training"].pop("rollout_steps_per_env", None)
    # Most learning tests exercise the algorithm; dedicated visualization tests
    # explicitly enable reports/video to avoid encoding dozens of duplicate demos.
    cfg["visualization"].update(enabled=False, videos=False)
    cfg["environment"]["cutoff_seconds"] = 2
    cfg["training"].update(
        budget_unit="decisions",  # Archived decision-budget regression controls.
        device="cuda",  # Every production learning path now requires CUDA.
        total_steps=128,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        n_envs=1,
        hidden_sizes=[32, 32],
        eval_interval=64,
    )
    return cfg
