import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from pvz_rl.envs.env import PvZEnv
from pvz_rl.learning.training import load_policy, train
from pvz_rl.presentation.recordings import verify_replay
from pvz_rl.provenance import verify_engine


def test_installed_engine_matches_recorded_commit(cfg):
    result = verify_engine(cfg)
    assert result["commit"] == cfg["engine_commit"]
    assert result["package_version"] == "1.6.0" and result["version"] == "1.3.0"


@pytest.mark.learning
@pytest.mark.parametrize("condition", ["masked"])
def test_real_training_serialization_and_replay(smoke_cfg, tmp_path, condition):
    output = tmp_path / condition
    train(smoke_cfg, condition, 101, output, validation_limit=1)
    model, data = load_policy(output / "final.zip")
    assert data["engine"]["commit"] == smoke_cfg["engine_commit"]
    assert model.num_timesteps >= 200 and model._n_updates == 2
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
                "nonlethal_health_damage",
                "mower_activation_penalty",
                "mower_activations",
                "wall_nut_damage",
                "empty_explosions",
            )
        )
        for row in episodes
    )
    updates = [
        json.loads(line) for line in (output / "training-metrics.jsonl").read_text().splitlines()
    ]
    assert all(
        "rolling_nonlethal_health_damage" in row and "rolling_empty_mower_activations" in row
        for row in updates
    )
    assert all(row["simulation_ticks"] > 0 and "rolling_wall_nut_damage" in row for row in updates)
    model.save(output / "roundtrip.zip")
    reloaded, _ = load_policy(output / "roundtrip.zip")
    env = PvZEnv(smoke_cfg, condition=condition, record=True)
    obs, _ = env.reset(seed=1)
    state, reloaded_state = None, None
    while env.state == "running":
        kwargs = {"action_masks": env.action_masks()} if condition != "unmasked" else {}
        action, state = model.predict(obs, state=state, deterministic=True, **kwargs)
        restored, reloaded_state = reloaded.predict(
            obs, state=reloaded_state, deterministic=True, **kwargs
        )
        np.testing.assert_array_equal(action, restored)
        obs, _, _, _, info = env.step(int(action))
        if not info["accepted"]:
            assert info["ticks_advanced"] == 1
            assert (
                info["reward_parts"]["invalid_plant_penalty"] < 0
                or info["reward_parts"]["empty_dig_penalty"] < 0
            )
            state.previous_actions.zero_()
            reloaded_state.previous_actions.zero_()
    replay = output / "probe.json"
    env.recorder.save(replay)
    assert verify_replay(replay).state_hash() == env.game.state_hash()


@pytest.mark.learning
def test_reward_changes_load_for_inference_but_require_weights_only_initialization(
    smoke_cfg, tmp_path
):
    import copy

    legacy = copy.deepcopy(smoke_cfg)
    legacy["reward"]["mower_value"] = 0.7
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
def test_windows_cuda_cli(tmp_path, tiny_cli_config):
    output = tmp_path / "spawn"
    device = "cuda"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pvz_rl",
            "train",
            "--config",
            str(tiny_cli_config),
            "--condition",
            "masked",
            "--steps",
            "64",
            "--n-envs",
            "2",
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
    assert json.loads((output / "status.json").read_text())["steps"] >= 200
    demos = json.loads((output / "visualizations/demos.json").read_text())["demos"]
    assert len(demos) == 1 and demos[0]["family"] == "diagnostic"


@pytest.mark.learning
def test_whole_suite_on_tiny_protocol_and_idempotent_resume(smoke_cfg, tmp_path):
    from pvz_rl.learning.suite import run_suite

    smoke_cfg["training"].update(total_steps=64, learner_seeds=[101])
    smoke_cfg["conditions"]["hybrid"] = dict(masked=True, shaped=True, curriculum=True, hybrid=True)
    for split, (lo, _) in smoke_cfg["splits"].items():
        smoke_cfg["splits"][split] = [lo, lo + 1]
    smoke_cfg["curriculum"]["probe_cases"] = 2
    for stage in smoke_cfg["curriculum"]["stages"].values():
        stage["requirements"] = {task: 2 for task in stage["requirements"]}
    smoke_cfg["evaluation"].update(replays_per_outcome=1, bootstrap_replicates=50)
    root = tmp_path / "suite"
    result = run_suite(smoke_cfg, root)
    assert result["state"] == "complete" and len(result["jobs"]) == 1
    assert len(result["evaluations"]) == 5 * 4
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
