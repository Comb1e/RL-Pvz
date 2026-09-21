import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from pvz_rl.env import PvZEnv
from pvz_rl.provenance import file_hash, verify_engine
from pvz_rl.recordings import verify_replay
from pvz_rl.training import load_policy, train


def test_installed_engine_matches_recorded_commit(cfg):
    result = verify_engine(cfg)
    assert result["commit"] == "8861824df6893a34c2cd4df7f9b68613376d7964"
    assert result["package_version"] == "1.3.0" and result["version"] == "1.0.0"


@pytest.mark.learning
@pytest.mark.parametrize("condition", ["masked", "unmasked", "sparse", "mixed", "hybrid"])
def test_real_training_serialization_and_replay(smoke_cfg, tmp_path, condition):
    output = tmp_path / condition
    train(smoke_cfg, condition, 101, output, validation_limit=1)
    model, data = load_policy(output / "final.zip")
    assert data["engine"]["commit"] == smoke_cfg["engine_commit"]
    assert model.num_timesteps == 128 and model._n_updates == 2
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    assert (output / "best.zip").exists()
    episodes = [
        json.loads(line) for line in (output / "training-episodes.jsonl").read_text().splitlines()
    ]
    assert episodes and all(
        all(
            key in row
            for key in (
                "damage_reward",
                "mower_activation_penalty",
                "mower_sun_penalty",
                "wall_nut_reward",
                "empty_explosion_penalty",
            )
        )
        for row in episodes
    )
    updates = [
        json.loads(line) for line in (output / "training-metrics.jsonl").read_text().splitlines()
    ]
    assert all(
        "rolling_damage_reward" in row and "rolling_empty_mower_activations" in row
        for row in updates
    )
    assert all(row["simulation_ticks"] > 0 and "rolling_wall_nut_reward" in row for row in updates)
    model.save(output / "roundtrip.zip")
    reloaded, _ = load_policy(output / "roundtrip.zip")
    env = PvZEnv(smoke_cfg, condition=condition, record=True)
    obs, _ = env.reset(seed=1)
    while env.state == "running":
        kwargs = {"action_masks": env.action_masks()} if condition != "unmasked" else {}
        action, _ = model.predict(obs, deterministic=True, **kwargs)
        restored, _ = reloaded.predict(obs, deterministic=True, **kwargs)
        np.testing.assert_array_equal(action, restored)
        obs, _, _, _, info = env.step(int(action))
        if condition != "unmasked":
            assert info["accepted"]
    replay = output / "probe.json"
    env.recorder.save(replay)
    assert verify_replay(replay).state_hash() == env.game.state_hash()


@pytest.mark.learning
def test_interrupt_resume_preserves_best_and_finishes_budget(smoke_cfg, tmp_path, monkeypatch):
    from pvz_rl.training import ResearchCallback

    original = ResearchCallback._on_rollout_start

    def interrupt(self):
        original(self)
        if self.model.num_timesteps == 64:
            raise KeyboardInterrupt("synthetic interruption after first PPO update")

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", interrupt)
    first = tmp_path / "first"
    # An old configuration and stock buffer remain resumable on the same engine pin.
    smoke_cfg["runtime"] = {key: False for key in smoke_cfg["runtime"]}
    with pytest.raises(KeyboardInterrupt):
        train(smoke_cfg, "masked", 101, first, validation_limit=1)
    previous_hash = file_hash(first / "best.zip")
    assert json.loads((first / "status.json").read_text())["steps"] == 64
    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", original)
    resumed = tmp_path / "resumed"
    smoke_cfg.pop("runtime")  # Missing runtime section gets current defaults on resume.
    train(smoke_cfg, "masked", 101, resumed, validation_limit=1, resume=first / "interrupted.zip")
    model, _ = load_policy(resumed / "final.zip")
    assert model.num_timesteps == 128 and model._n_updates == 2
    assert file_hash(resumed / "best.zip") == previous_hash  # equal scores keep earliest checkpoint
    with pytest.raises(FileExistsError):
        train(smoke_cfg, "masked", 101, resumed, validation_limit=1)


@pytest.mark.learning
def test_legacy_rewards_reload_but_cannot_resume_into_new_objective(smoke_cfg, tmp_path):
    import copy

    legacy = copy.deepcopy(smoke_cfg)
    legacy["reward"] = dict(
        gamma=0.999,
        defeated_weight=0.5,
        sun_weight=0.3,
        flower_weight=0.2,
        sun_target=300,
        flower_target=8,
        plant_kill_weight=1.0,
        mower_kill_weight=1.0,
        normalize_kills=True,
    )
    run = train(legacy, "masked", 101, tmp_path / "legacy", validation_limit=1)
    _, data = load_policy(run / "final.zip")
    assert data["config"]["reward"] == legacy["reward"]
    with pytest.raises(ValueError, match="start a fresh run"):
        train(
            smoke_cfg,
            "masked",
            101,
            tmp_path / "changed",
            resume=run / "final.zip",
            validation_limit=1,
        )


@pytest.mark.learning
def test_legacy_timing_reload_cannot_resume_into_per_tick(smoke_cfg, tmp_path):
    import copy

    legacy = copy.deepcopy(smoke_cfg)
    legacy["environment"].pop("action_timing")
    legacy["environment"]["decision_ticks"] = 10
    run = train(legacy, "masked", 101, tmp_path / "fixed", validation_limit=1)
    _, data = load_policy(run / "final.zip")
    assert "action_timing" not in data["config"]["environment"]
    with pytest.raises(ValueError, match="start a fresh run"):
        train(
            smoke_cfg,
            "masked",
            101,
            tmp_path / "changed",
            resume=run / "final.zip",
            validation_limit=1,
        )


@pytest.mark.learning
def test_windows_spawn_and_cuda_when_available(tmp_path):
    output = tmp_path / "spawn"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pvz_rl",
            "train",
            "--condition",
            "masked",
            "--steps",
            "64",
            "--n-envs",
            "2",
            "--rollout-size",
            "64",
            "--batch-size",
            "32",
            "--eval-interval",
            "64",
            "--device",
            device,
            "--validation-count",
            "1",
            "--family",
            "diagnostic",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / "metadata.json").read_text())
    assert data["config"]["training"]["device"] == device
    assert json.loads((output / "status.json").read_text())["steps"] == 64
    demos = json.loads((output / "visualizations/demos.json").read_text())["demos"]
    assert len(demos) == 1 and demos[0]["family"] == "diagnostic"


@pytest.mark.learning
def test_whole_suite_on_tiny_protocol_and_idempotent_resume(smoke_cfg, tmp_path):
    from pvz_rl.suite import run_suite

    smoke_cfg["training"].update(total_steps=64, learner_seeds=[101])
    for split, (lo, _) in smoke_cfg["splits"].items():
        smoke_cfg["splits"][split] = [lo, lo + 1]
    smoke_cfg["evaluation"].update(replays_per_outcome=1, bootstrap_replicates=50)
    root = tmp_path / "suite"
    result = run_suite(smoke_cfg, root)
    assert result["state"] == "complete" and len(result["jobs"]) == 5
    assert len(result["evaluations"]) == 9 * 4
    assert (root / "report-1" / "report.md").exists()
    assert (root / "report-1" / "win-rates.png").stat().st_size > 1000
    assert run_suite(smoke_cfg, root, resume=True) == result


def test_rendering_does_not_change_simulation(cfg):
    env = PvZEnv(cfg, render_mode="rgb_array")
    env.reset(seed=3)
    before = env.game.state_hash()
    frame = env.render()
    assert frame.shape == (600, 1000, 3) and frame.dtype == np.uint8
    assert env.game.state_hash() == before
