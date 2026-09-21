"""Completed games, not action count, drive new training schedules."""

import argparse
import copy
import json

import pytest
import torch

from pvz_rl.budget import budget_target, evaluation_interval, uses_games
from pvz_rl.cli import configured
from pvz_rl.config import learning_profile, validate_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.env import PvZEnv
from pvz_rl.training import ResearchCallback, load_policy, train
from pvz_rl.visualization import read_series


def game_config(smoke_cfg):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["training"].update(
        budget_unit="games",
        total_games=3,
        eval_interval_games=1,
        total_steps=1,
        eval_interval=1,
        rollout_size=32,
        batch_size=16,
    )
    cfg["environment"]["cutoff_seconds"] = 1
    return cfg


def test_default_config_and_cli_use_games(per_tick_cfg):
    assert uses_games(per_tick_cfg)
    assert budget_target(per_tick_cfg) == 10000 and evaluation_interval(per_tick_cfg) == 250
    args = argparse.Namespace(config=None, command="train", games=20, eval_games=5)
    cfg = configured(args)
    assert budget_target(cfg) == 20 and evaluation_interval(cfg) == 5
    args = argparse.Namespace(config=None, command="train", steps=64, eval_interval=32)
    cfg = configured(args)
    assert not uses_games(cfg) and budget_target(cfg) == 64
    with pytest.raises(ValueError, match="--eval-games"):
        configured(argparse.Namespace(config=None, command="train", games=3, eval_interval=1))


def test_fixed_curriculum_changes_only_on_reset_and_ignores_decisions(per_tick_cfg):
    per_tick_cfg["training"].update(total_games=100, total_steps=1)
    env = PvZEnv(per_tick_cfg, training=True)
    env.reset(seed=5)
    env.set_progress(10)
    family = env.episode_level
    for _ in range(50):
        env.step(0)
    assert env.progress == 10 and env.episode_level == family
    env.set_progress(0)
    assert all(
        env.reset(seed=i)[1]["status"] == "running" and env.episode_level == "easy"
        for i in range(10)
    )
    env.set_progress(40)
    levels = []
    for i in range(20):
        env.reset(seed=i)
        levels.append(env.episode_level)
    assert set(levels) == {"easy", "standard", "hard"}


def test_teaching_gates_use_games_and_resume_counters(per_tick_cfg):
    cfg = learning_profile("pure-rl", per_tick_cfg)
    state = CurriculumState(entered_steps=999999, last_probe=999999)
    assert not state.due(19, cfg) and state.due(20, cfg)
    assert not state.observe({"placement": 18}, 20, cfg)
    assert not state.observe({"placement": 17}, 40, cfg)
    assert state.consecutive_passes == 0
    assert not state.observe({"placement": 20}, 60, cfg)
    assert state.observe({"placement": 18}, 80, cfg)
    assert state.entered_games == state.last_probe_games == 80
    restored = CurriculumState(**state.to_dict())
    assert not restored.due(99, cfg) and restored.due(100, cfg)
    assert restored.name == "saving"
    with pytest.raises(ValueError, match="backwards"):
        restored.observe({"saving": 20}, 79, cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("budget_unit", "ticks"),
        ("total_games", 0),
        ("total_games", True),
        ("eval_interval_games", -1),
    ],
)
def test_invalid_game_settings(per_tick_cfg, key, value):
    per_tick_cfg["training"][key] = value
    with pytest.raises(ValueError):
        validate_config(per_tick_cfg)


@pytest.mark.learning
@pytest.mark.parametrize("condition", ["masked", "unmasked", "sparse", "mixed", "hybrid"])
def test_completed_games_stop_after_optimization_and_validation_does_not_count(
    smoke_cfg, tmp_path, condition
):
    cfg = game_config(smoke_cfg)
    run = train(cfg, condition, 101, tmp_path / condition, validation_limit=1)
    model, _ = load_policy(run / "final.zip")
    episodes = read_series(run / "training-episodes.jsonl")
    assert model.training_games == len(episodes) >= 3
    assert [r["training_games"] for r in episodes] == list(range(1, len(episodes) + 1))
    assert all(r["status"] == "truncated" for r in episodes)
    assert model.num_timesteps == model._n_updates * 32
    status = json.loads((run / "status.json").read_text())
    assert status["budget_complete"] and status["budget_unit"] == "games"
    assert status["target_games"] == 3 and status["target_steps"] is None
    assert status["extra_games_in_final_rollout"] == len(episodes) - 3
    curves = read_series(run / "learning-curve.jsonl")
    assert all(r["training_games"] >= 1 and r["budget_unit"] == "games" for r in curves)
    assert curves[-1]["training_games"] == len(episodes)
    assert "3 games" in (run / "train.log").read_text()
    for file in run.glob("validation/*/episodes.jsonl"):
        assert all(r["training_games"] <= model.training_games for r in read_series(file))


@pytest.mark.learning
def test_global_games_across_windows_workers_cuda_and_shared_demos(smoke_cfg, tmp_path):
    cfg = game_config(smoke_cfg)
    cfg["training"].update(
        n_envs=2,
        rollout_size=128,
        total_games=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    cfg["visualization"].update(enabled=True, videos=False)
    run = train(cfg, "masked", 101, tmp_path / "game-smoke", validation_limit=1)
    model, _ = load_policy(run / "final.zip")
    episodes = read_series(run / "training-episodes.jsonl")
    assert model.training_games == len(episodes) > 2
    assert model.num_timesteps == 128 and model._n_updates == 1
    demos = json.loads((run / "visualizations/demos.json").read_text())["demos"]
    assert {d["level"] for d in demos} == {"easy", "standard", "hard"}
    assert len({d["checkpoint_hash"] for d in demos}) == 1
    assert {d["training_games"] for d in demos} == {model.training_games}
    assert not list((run / "visualizations").rglob("*.mp4"))


@pytest.mark.learning
def test_interrupt_resume_preserves_games_and_schedule(smoke_cfg, tmp_path, monkeypatch):
    cfg = game_config(smoke_cfg)
    cfg["training"]["total_games"] = 6
    original = ResearchCallback._on_rollout_start

    def interrupt(self):
        if getattr(self.model, "training_games", 0) >= 2:
            raise KeyboardInterrupt()
        original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", interrupt)
    first = tmp_path / "first"
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, first, validation_limit=1)
    model, _ = load_policy(first / "interrupted.zip")
    initial = model.training_games
    assert 2 <= initial < 6
    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", original)
    run = train(
        cfg,
        "masked",
        101,
        tmp_path / "resume",
        validation_limit=1,
        resume=first / "interrupted.zip",
    )
    model, _ = load_policy(run / "final.zip")
    episodes = read_series(run / "training-episodes.jsonl")
    assert model.training_games == initial + len(episodes) >= 6
    assert episodes[0]["training_games"] == initial + 1
    changed = copy.deepcopy(cfg)
    changed["training"]["budget_unit"] = "decisions"
    with pytest.raises(ValueError, match="start a fresh run"):
        train(
            changed,
            "masked",
            101,
            tmp_path / "mismatch",
            validation_limit=1,
            resume=first / "interrupted.zip",
        )


@pytest.mark.learning
def test_game_mastery_probes_keep_policy_and_optimizer(smoke_cfg, tmp_path, monkeypatch):
    cfg = learning_profile("pure-rl", game_config(smoke_cfg))
    cfg["training"].update(total_games=7, eval_interval_games=3)
    cfg["curriculum"].update(probe_interval_games=2, minimum_stage_games=2)
    identities = []
    original = ResearchCallback.probe_curriculum

    def evaluate_control(cfg, **kwargs):
        from pvz_rl.evaluation import evaluate

        if kwargs.get("split") == "curriculum_validation":
            return [{"win": 1} for _ in range(20)]
        return evaluate(cfg, **kwargs)

    def probe(self):
        identities.append((id(self.model.policy), id(self.model.policy.optimizer)))
        original(self)

    monkeypatch.setattr("pvz_rl.training.evaluate", evaluate_control)
    monkeypatch.setattr(ResearchCallback, "probe_curriculum", probe)
    run = train(cfg, "masked", 101, tmp_path / "mastery", validation_limit=1)
    model, _ = load_policy(run / "final.zip")
    assert len(set(identities)) == 1 and model.curriculum_state["stage"] >= 1
    assert model.curriculum_state["entered_steps"] == 0
    assert model.curriculum_state["entered_games"] >= 4
    probes = read_series(run / "curriculum-probes.jsonl")
    counts = [p["training_games"] for p in probes]
    assert counts[0] >= 2 and all(b - a >= 2 for a, b in zip(counts, counts[1:]))
    assert any(r["family"] == "saving" for r in read_series(run / "training-episodes.jsonl"))
