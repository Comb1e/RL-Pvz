"""Validation scheduling, finalization, and recurrent checkpoint compatibility."""

from time import perf_counter

import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.evaluation.runner import evaluate
from pvz_rl.learning.curriculum import CurriculumState
from pvz_rl.learning.deadline import BudgetExpired
from pvz_rl.learning.training import ResearchCallback, load_policy, train
from pvz_rl.presentation.visualization import read_json, read_series


def small_cfg():
    cfg = load_config()
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["visualization"].update(enabled=False, demos=False)
    cfg["training"].update(
        validation_schedule="periodic",  # Original periodic-validation regression protocol.
        n_envs=2,
        batch_size=32,
        n_epochs=1,
        total_games=12,
        max_minutes=2,
    )
    cfg["policy"]["chunk_length"] = 16
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

    monkeypatch.setattr("pvz_rl.learning.training.evaluate", fake_evaluate)
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
def test_validation_crossed_thresholds_and_final_tie_keep_earlier_weights(tmp_path):
    cfg = small_cfg()
    cfg["training"].update(total_games=12, eval_interval_games=2)
    run = tmp_path / "run"
    train(cfg, "masked", 101, run, validation_limit=1)
    curves = read_series(run / "learning-curve.jsonl")
    assert len({r["training_steps"] for r in curves}) == len(curves)
    assert [r["training_games"] for r in curves] == [2, 4, 6, 8, 10, 12]
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
    from pvz_rl.learning.deadline import RunBudget

    cfg = small_cfg()
    cfg["training"]["total_games"] = 2
    cfg["visualization"].update(enabled=True, demos=True)
    offset = [0.0]

    def clock():
        return perf_counter() + offset[0]

    monkeypatch.setattr("pvz_rl.learning.training.perf_counter", clock)
    monkeypatch.setattr(
        "pvz_rl.learning.training.RunBudget", lambda *a, **kw: RunBudget(*a, **kw, clock=clock)
    )
    monkeypatch.setattr("pvz_rl.presentation.visualization.build_run_report", lambda *a, **kw: None)

    def interrupted_export(*args, **kwargs):
        offset[0] += 125  # Simulate costly presentation, without sleeping.
        raise KeyboardInterrupt("controlled export interruption")

    monkeypatch.setattr("pvz_rl.presentation.visualization.visualize_run", interrupted_export)
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


@pytest.mark.parametrize("wait_ticks", [0, 500, 501])
def test_cuda_long_horizon_reward_and_early_dig_boundaries(wait_ticks):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_game import Dig, LevelSpec, Place, Spawn

    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch as CudaBatch
    from pvz_rl.envs.cuda_features import CudaFeatures

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
            == int(wait_ticks <= 500)
        )
        assert batch.state_hash(0) == cpu.game.state_hash()


def test_retired_checkpoint_rejected_before_deserialization(smoke_cfg, tmp_path, monkeypatch):
    import copy
    import json
    from zipfile import ZipFile

    from stable_baselines3.common.base_class import BaseAlgorithm

    from pvz_rl.config import validate_config
    from pvz_rl.envs.env import PvZEnv
    from pvz_rl.learning.recurrent_q import CudaRecurrentQ
    from pvz_rl.learning.training import initial_weights

    old = copy.deepcopy(smoke_cfg)
    old["policy"].update(
        kind="event_sequential_q_v1", action_distribution="sequential_plant_epsilon_v1"
    )
    old["training"]["method"] = "sequential_q_mc_v1"
    old["reward"].pop("invalid_plant_penalty")
    old["reward"].pop("empty_dig_penalty")
    archive_path = tmp_path / "old.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "protocol.json",
            json.dumps(
                dict(
                    policy="event_sequential_q_v1",
                    optimizer="sequential_q_mc_v1",
                    exploration="sequential_plant_epsilon_v1",
                )
            ),
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps(dict(config=old, condition="masked")), encoding="utf-8"
    )
    before = archive_path.read_bytes()
    monkeypatch.setattr(
        BaseAlgorithm, "load", lambda *a, **k: pytest.fail("deserialized a retired checkpoint")
    )
    for operation in (
        lambda: load_policy(archive_path),
        lambda: initial_weights(archive_path, smoke_cfg),
        lambda: CudaRecurrentQ.load(archive_path),
        lambda: validate_config(old),
        lambda: PvZEnv(old),
    ):
        with pytest.raises(ValueError, match="Retired|Incompatible"):
            operation()
    assert archive_path.read_bytes() == before
