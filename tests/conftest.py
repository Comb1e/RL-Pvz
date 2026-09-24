import pytest

from pvz_rl.config import load_config


@pytest.fixture
def legacy_teaching():
    """Explicit archived protocol, independent of current recipe defaults."""

    def restore(cfg):
        c = cfg["curriculum"]
        c.update(probe_interval_games=100, probe_cases=20, consecutive_passes=2)
        c.pop("residency", None)
        cfg["splits"].pop("curriculum", None)
        for stage, requirements in {
            "placement": {"placement": 18},
            "saving": {"saving": 18},
            "easy": {"easy": 16},
            "standard": {"easy": 16, "standard": 12},
            "shared": {},
        }.items():
            c["stages"][stage]["requirements"] = requirements
        return cfg

    return restore


@pytest.fixture
def cfg():
    cfg = load_config()
    # Preserve established 10-tick controls as legacy regression cases.
    # per_tick_cfg and smoke_cfg exercise the current default protocol.
    cfg["environment"].update(action_timing="fixed", decision_ticks=10)
    cfg["training"]["budget_unit"] = "decisions"
    cfg["training"].pop("rollout_steps_per_env", None)
    cfg["simulation"] = {"backend": "cpu"}
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
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["training"].update(
        validation_schedule="periodic",  # Retain independent archived scheduling controls.
        budget_unit="decisions",  # Archived decision-budget regression controls.
        device="cuda",  # Every production learning path now requires CUDA.
        total_steps=128,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        critic_warmup_games=0,  # Warm-up has dedicated stage/resume controls.
        target_kl=0,
        n_envs=1,
        hidden_sizes=[32, 32],
        eval_interval=64,
    )
    return cfg


@pytest.fixture
def tiny_cli_config(tmp_path):
    from pathlib import Path

    path = tmp_path / "tiny.toml"
    path.write_text(
        Path("configs/train.toml")
        .read_text()
        .replace("cutoff_seconds = 1200", "cutoff_seconds = 1")
    )
    return path
