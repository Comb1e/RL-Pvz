"""Core startup, public-board inference and explicitly selected CUDA learning."""

import math
import subprocess
import sys
from contextlib import closing
from textwrap import dedent

import pytest


def test_cli_startup(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "pvz_rl", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train" in result.stdout


def test_cpu_agent_inference():
    import torch

    from pvz_rl.config import load_config
    from pvz_rl.envs.encoding import collate_observations
    from pvz_rl.envs.env import PvZEnv
    from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy

    cfg = load_config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    policy = TwoStageTransformerPolicy(cfg).eval()
    with closing(PvZEnv(cfg, level="easy")) as env:
        observation, _ = env.reset(seed=1)
        board = collate_observations(observation)
        with torch.no_grad():
            species = policy.species_values(board)
            tiles = policy.conditional_tiles(board, [1])
            actions, _ = policy.decide(board)
        assert species.shape == (1, len(cfg["environment"]["plants"]) + 2)
        assert tiles.shape == (1, env.codec.rows * env.codec.cols)
        assert torch.isfinite(species).all() and torch.isfinite(tiles).all()
        action = int(actions.item())
        assert env.action_space.contains(action)
        _, reward, _, _, _ = env.step(action)
        assert math.isfinite(reward)


@pytest.mark.learning
def test_cuda_learning_and_checkpoint(tmp_path):
    program = dedent(
        """
        import json
        import sys
        from contextlib import closing
        from pathlib import Path

        from pvz_rl.config import load_config
        from pvz_rl.envs.env import PvZEnv
        from pvz_rl.learning.training import load_policy, train

        cfg = load_config()
        cfg["environment"]["cutoff_seconds"] = 1
        cfg["training"].update(
            n_envs=2, total_games=2, n_epochs=1, batch_size=32, max_minutes=2
        )
        cfg["training"]["performance"].update(
            compile_collection=False, compile_fitting=False
        )
        cfg["visualization"].update(
            enabled=False, demos=False, videos=False, live_enabled=False
        )
        output = Path(sys.argv[1]) / "run"
        train(cfg, "masked", 101, output)
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "complete" and status["budget_complete"], status
        assert status["training_games"] == cfg["training"]["total_games"], status
        model, _ = load_policy(output / "final.zip", device="cuda")
        assert model._n_updates > 0 and model.policy.optimizer.state
        with closing(PvZEnv(cfg, level="easy")) as env:
            observation, _ = env.reset(seed=1)
            action, _ = model.predict(observation, deterministic=True)
            assert env.action_space.contains(action), action
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
