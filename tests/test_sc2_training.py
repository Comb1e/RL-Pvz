"""Scheduling, recovery, and real short PPO integration for the new profiles."""

from time import perf_counter

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.deadline import BudgetExpired
from pvz_rl.env import PvZEnv
from pvz_rl.evaluation import evaluate
from pvz_rl.training import ResearchCallback, load_policy, train
from pvz_rl.visualization import read_json, read_series


def small_cfg(device="cuda", backend="cuda", profile="E"):
    cfg = load_config()
    cfg["simulation"]["backend"] = backend
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["visualization"].update(enabled=False, demos=False)
    cfg["training"].update(
        validation_schedule="periodic",  # Original periodic-validation regression protocol.
        device=device,
        n_envs=2,
        rollout_steps_per_env=32,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        total_games=12,
        max_minutes=2,
    )
    return cfg


def test_validation_cache_and_incomplete_results_cannot_select_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cfg = load_config()
    callback = ResearchCallback(cfg, "masked", 101, tmp_path)
    callback.model = SimpleNamespace(num_timesteps=100, training_games=1000)
    calls = []

    def fake_evaluate(cfg, **kwargs):
        calls.append(kwargs)
        return [
            {"family": kwargs["family"], "level": level, "scenario_seed": seed}
            for level in kwargs["levels"]
            for seed in kwargs["seeds"]
        ]

    monkeypatch.setattr("pvz_rl.training.evaluate", fake_evaluate)
    try:
        callback.cached_evaluation(
            list(range(100000, 100050)),
            ["easy", "standard", "hard"],
            "preset",
            tmp_path / "full",
            "validation",
        )
        result = callback.cached_evaluation(
            list(range(100000, 100020)),
            ["easy"],
            "preset",
            tmp_path / "probe",
            "curriculum_validation",
        )
        assert len(result) == 20 and len(calls) == 1
        callback.model.num_timesteps += 64
        callback.cached_evaluation([100000], ["easy"], "preset", tmp_path / "new", "validation")
        assert len(calls) == 2
    finally:
        callback.close()


def test_expired_evaluation_records_pending_not_a_win(smoke_cfg, tmp_path):
    with pytest.raises(BudgetExpired):
        evaluate(
            smoke_cfg,
            seeds=[100000],
            levels=["easy"],
            output=tmp_path / "eval",
            baseline="wait",
            deadline=perf_counter() - 1,
        )
    assert read_json(tmp_path / "eval/status.json")["state"] == "pending"
    assert not (tmp_path / "eval/summary.json").exists()


@pytest.mark.learning
@pytest.mark.parametrize("backend,device", [("cuda", "cuda")])
def test_spatial_training_reload_and_preserved_model_across_stages(
    tmp_path, backend, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = small_cfg(device, backend)
    original = ResearchCallback._on_rollout_start
    identities = []

    def transition(self):
        identities.append((id(self.model.policy), id(self.model.policy.optimizer)))
        if self.model.num_timesteps > 0 and self.curriculum.stage == 0:
            self.curriculum.stage = 1
            self.training_env.env_method("set_curriculum_stage", 1)
        return original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", transition)
    run = tmp_path / "run"
    train(cfg, "masked", 101, run, validation_limit=1)
    assert len(set(identities)) == 1
    metrics = read_series(run / "training-metrics.jsonl")
    assert metrics[-1]["training_steps"] == read_json(run / "status.json")["steps"]
    assert metrics[-1]["optimization"]["exploration_bonus"] is not None
    assert metrics[-1]["optimization"]["joint_entropy"] >= 0
    loaded, _ = load_policy(run / "final.zip", device)
    again, _ = load_policy(run / "final.zip", device)
    env = PvZEnv(cfg)
    observation, _ = env.reset(seed=100000)
    action = loaded.predict(observation, deterministic=True, action_masks=env.action_masks())[0]
    np.testing.assert_array_equal(
        action, again.predict(observation, deterministic=True, action_masks=env.action_masks())[0]
    )
    assert loaded.curriculum_state["stage"] == 1


@pytest.mark.learning
def test_resume_keeps_cumulative_time_schedule_and_optimizer(tmp_path, monkeypatch):
    cfg = small_cfg()
    cfg["training"]["total_games"] = 24
    original = ResearchCallback._on_rollout_start

    def interrupt(self):
        if self.model.num_timesteps >= 64:
            raise KeyboardInterrupt()
        return original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", interrupt)
    first = tmp_path / "first"
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, first, validation_limit=1)
    interrupted, _ = load_policy(first / "interrupted.zip")
    assert interrupted.num_timesteps == 64
    assert interrupted.policy.optimizer.state
    elapsed = read_json(first / "status.json")["time_budget"]["elapsed_seconds"]
    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", original)
    resumed = tmp_path / "resumed"
    train(cfg, "masked", 101, resumed, validation_limit=1, resume=first / "interrupted.zip")
    status = read_json(resumed / "status.json")
    assert status["time_budget"]["elapsed_seconds"] > elapsed
    assert status["training_games"] >= 24
    model, _ = load_policy(resumed / "final.zip")
    assert model._n_updates > interrupted._n_updates
    assert model.research_schedule["next_eval"] == 2000


@pytest.mark.learning
def test_validation_crossed_thresholds_and_final_tie_keep_earlier_weights(tmp_path):
    cfg = small_cfg(profile="C")
    cfg["training"].update(total_games=12, eval_interval_games=2)
    run = tmp_path / "run"
    train(cfg, "masked", 101, run, validation_limit=1)
    curves = read_series(run / "learning-curve.jsonl")
    assert len({r["training_steps"] for r in curves}) == len(curves)
    assert len(curves) < 6  # Many threshold crossings coalesce after full rollouts.
    assert read_json(run / "best.json")["steps"] == curves[0]["training_steps"]
    assert curves[-1]["final"]


def test_finished_old_config_does_not_acquire_new_defaults():
    cfg = load_config()
    cfg["training"].pop("max_minutes")
    cfg["training"]["eval_interval_games"] = 250
    from pvz_rl.config import validate_config

    validate_config(cfg)
    assert cfg["training"]["eval_interval_games"] == 250
    assert "max_minutes" not in cfg["training"]
    assert CurriculumState(**{"stage": 1, "entered_games": 20}).completed_stage_games == 0


@pytest.mark.learning
def test_interrupted_export_counts_time_spent_after_final_checkpoint(tmp_path, monkeypatch):
    from pvz_rl.deadline import RunBudget

    cfg = small_cfg()
    cfg["training"]["total_games"] = 2
    cfg["visualization"].update(enabled=True, demos=True)
    offset = [0.0]

    def clock():
        return perf_counter() + offset[0]

    monkeypatch.setattr("pvz_rl.training.perf_counter", clock)
    monkeypatch.setattr(
        "pvz_rl.training.RunBudget", lambda *a, **kw: RunBudget(*a, **kw, clock=clock)
    )
    monkeypatch.setattr("pvz_rl.visualization.build_run_report", lambda *a, **kw: None)

    def interrupted_export(*args, **kwargs):
        offset[0] += 125  # Simulate costly presentation, without sleeping.
        raise KeyboardInterrupt("controlled export interruption")

    monkeypatch.setattr("pvz_rl.visualization.visualize_run", interrupted_export)
    run = tmp_path / "export"
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, run, validation_limit=1)
    status = read_json(run / "status.json")
    assert status["state"] == "complete"
    assert status["visualization_state"] == "interrupted"
    assert status["wall_seconds"] >= 125
    assert status["time_budget"]["elapsed_seconds"] >= 125
    assert status["time_budget"]["remaining_seconds"] == 0
    assert (run / "final.zip").is_file()


@pytest.mark.learning
def test_spatial_cuda_report_and_three_verified_shared_demos(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_rl.provenance import file_hash
    from pvz_rl.recordings import open_playback

    cfg = small_cfg("cuda", "cuda")
    cfg["training"]["total_games"] = 2
    cfg["visualization"].update(enabled=True, demos=True, videos=False)
    run = train(cfg, "masked", 101, tmp_path / "spatial-report", validation_limit=1)
    assert (run / "visualizations/index.html").is_file()
    demos = read_json(run / "visualizations/demos.json")["demos"]
    assert {d["level"] for d in demos} == {"easy", "standard", "hard"}
    assert {d["checkpoint_hash"] for d in demos} == {file_hash(run / "best.zip")}
    for demo in demos:
        playback = open_playback(run / "visualizations" / demo["replay"])
        playback.verify()
        assert playback.game.state_hash() == demo["state_hash"]
        assert playback.display_outcome == "truncated"
    assert read_json(run / "status.json")["visualization_state"] == "complete"


@pytest.mark.parametrize("wait_ticks", [0, 100, 101])
def test_cuda_long_horizon_reward_and_early_dig_boundaries(wait_ticks):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_game import Dig, LevelSpec, Place, Spawn

    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch as CudaBatch

    cfg = load_config()
    level = LevelSpec("ledger-boundary", (Spawn(20000, "basic", 0),), initial_sun=200)
    cpu = PvZEnv(cfg)
    cpu.reset(seed=9, options={"scenario": level})
    batch = CudaBatch(1, zombie_capacity=1, max_step_ticks=1)
    batch.reset([level], [9])
    cp = batch.cp
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        actions = [
            cpu.codec.encode(Place("sunflower", 0, 0)),
            *([0] * wait_ticks),
            cpu.codec.encode(Dig(0, 0)),
        ]
        for action in actions:
            _, reward, _, _, _ = cpu.step(action)
            features.step(cp.asarray([action], dtype=cp.int64))
            assert features.rewards.get()[0] == pytest.approx(reward, abs=2e-6)
        assert (
            features.totals.get()[0, 35]
            == cpu.episode_metrics()["early_voluntary_digs"]
            == int(wait_ticks <= 100)
        )
        assert batch.state_hash(0) == cpu.game.state_hash()
