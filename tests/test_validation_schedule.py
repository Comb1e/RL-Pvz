"""Stage success is an evaluation event, never an inferred training win rate."""

import copy
from types import SimpleNamespace

import pytest

from pvz_rl.config import load_config, validate_config
from pvz_rl.learning.curriculum import STAGES, CurriculumState, validation_after_stage
from pvz_rl.learning.deadline import BudgetExpired
from pvz_rl.learning.training import ResearchCallback, TrainingDeadline, load_policy, train
from pvz_rl.presentation.visualization import read_json, read_series
from pvz_rl.provenance import file_hash


@pytest.fixture
def stage_callback(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["visualization"]["enabled"] = False
    cb = ResearchCallback(cfg, "masked", 101, tmp_path, validation_limit=1)
    env = SimpleNamespace(env_method=lambda *a: None)
    cb.model = SimpleNamespace(
        num_timesteps=64,
        training_games=2000,
        env=env,
        get_env=lambda: env,
        save=lambda p: p.write_bytes(str(cb.model.num_timesteps).encode()),
    )
    cb.curriculum = CurriculumState(completed_stage_games=2000)
    monkeypatch.setattr(cb, "snapshot", lambda: {})
    monkeypatch.setattr(cb, "capture_update", lambda: None)
    yield cb
    cb.close()


def evaluation_rows(seeds, levels, family, *, win=True):
    return [
        dict(
            family=family,
            level=level,
            scenario_seed=seed,
            win=win,
            status="won" if win else "lost",
            decisions=1,
            mowers_used=0,
            invalid_actions=0,
            wait_actions=1,
            simulated_seconds=1,
            inference_seconds=0,
        )
        for level in levels
        for seed in seeds
    ]


@pytest.mark.parametrize("failure", [None, "deadline", "interrupt"])
def test_report_refreshes_once_per_probe_or_validation_attempt(
    stage_callback, monkeypatch, failure
):
    cb = stage_callback
    refreshes = []
    monkeypatch.setattr(cb, "refresh_report", lambda: refreshes.append(1))

    def evaluate(seeds, levels, family, *args, **kwargs):
        if failure == "deadline":
            raise BudgetExpired("controlled")
        if failure == "interrupt":
            raise KeyboardInterrupt
        return evaluation_rows(seeds, levels, family)

    monkeypatch.setattr(cb, "cached_evaluation", evaluate)
    for operation in (cb.probe_curriculum, cb.validate):
        if operation == cb.validate:
            cb.pending_stage_validation = {"stage": "saving"}
        before = len(refreshes)
        if failure == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                operation()
        else:
            operation()
        assert len(refreshes) == before + 1


@pytest.mark.parametrize("failure", ["loss", "truncation", "incomplete", "residency"])
def test_failure_or_insufficient_residency_never_triggers_full_validation(
    stage_callback, monkeypatch, failure
):
    cb = stage_callback
    calls = []

    def evaluate(seeds, levels, family, destination, split, final=False):
        calls.append(split)
        rows = evaluation_rows(seeds, levels, family)
        if failure == "incomplete":
            return rows[:-1]
        if failure == "residency":
            cb.curriculum.completed_stage_games = 99
        else:
            rows[-1].update(win=False, status="truncated" if failure == "truncation" else "lost")
        return rows

    monkeypatch.setattr(cb, "cached_evaluation", evaluate)
    if failure == "incomplete":
        with pytest.raises(ValueError, match="Incomplete curriculum"):
            cb.probe_curriculum()
    else:
        cb._on_training_end()
    assert cb.validate(final=True) is False
    assert calls == ["curriculum_validation"]
    assert not cb.pending_stage_validation and not cb.curriculum.mastered
    assert not (cb.output / "best.zip").exists()
    assert not (cb.output / "learning-curve.jsonl").exists()


def test_every_stage_triggers_once_and_equal_scores_keep_earliest(stage_callback, monkeypatch):
    cb = stage_callback
    calls = []

    def evaluate(seeds, levels, family, destination, split, final=False):
        calls.append((split, family, list(levels), len(seeds)))
        return evaluation_rows(seeds, levels, family)

    monkeypatch.setattr(cb, "cached_evaluation", evaluate)
    for i, stage in enumerate(STAGES, 1):
        cb.model.training_games = 2000 * i
        cb.model.num_timesteps = 64 * i
        cb.curriculum.completed_stage_games = 2000
        cb.probe_curriculum()
        assert cb.pending_stage_validation["stage"] == stage
        assert cb.model.research_schedule["pending_stage_validation"]["stage"] == stage
        cb.finish_stage_validation()
        assert cb.pending_stage_validation is None
        assert cb.model.research_schedule["pending_stage_validation"] is None
        assert cb.validate(final=True) is False
    cb._on_training_end()  # Finalization must not add a sixth normal evaluation.
    rows = read_series(cb.output / "learning-curve.jsonl")
    assert [r["stage_success"]["stage"] for r in rows] == list(STAGES)
    assert [r["training_games"] for r in rows] == [2000, 4000, 6000, 8000]
    assert [c for c in calls if c[0] == "validation"] == [
        ("validation", "preset", ["easy", "standard", "hard"], 1)
    ] * len(STAGES)
    assert read_json(cb.output / "best.json")["stage_success"]["stage"] == "saving"
    assert (cb.output / "best.zip").read_bytes() == b"64"


def test_probe_threshold_coalesces_without_probing_early(stage_callback, monkeypatch):
    cb = stage_callback
    calls = []

    def evaluate(seeds, levels, family, *a, **kw):
        calls.append(cb.model.training_games)
        return evaluation_rows(seeds, levels, family, win=False)

    monkeypatch.setattr(cb, "cached_evaluation", evaluate)
    for games in (1999, 2000, 3999, 6250, 6250, 8249, 8250):
        cb.model.training_games = games
        cb.probe_curriculum()
    assert calls == [2000, 6250, 8250]
    assert cb.pending_stage_validation is None


def test_pending_mastery_evaluation_stops_updates_and_restores_exact_trigger(
    stage_callback, monkeypatch
):
    cb = stage_callback

    def evaluate(seeds, levels, family, destination, split, final=False):
        if split == "validation":
            raise BudgetExpired("controlled deadline")
        return evaluation_rows(seeds, levels, family)

    monkeypatch.setattr(cb, "cached_evaluation", evaluate)
    cb.probe_curriculum()
    pending = copy.deepcopy(cb.pending_stage_validation)
    with pytest.raises(TrainingDeadline):
        cb.finish_stage_validation()
    cb.save_checkpoint("final.zip")
    assert cb.pending_stage_validation == pending and cb.validation_pending
    cb.pending_stage_validation = None
    cb.restore_schedule()
    assert cb.pending_stage_validation == pending and cb.validation_pending
    monkeypatch.setattr(
        cb,
        "cached_evaluation",
        lambda seeds, levels, family, *a, **kw: evaluation_rows(seeds, levels, family),
    )
    cb._on_training_end()
    assert len(read_series(cb.output / "learning-curve.jsonl")) == 1
    assert cb.pending_stage_validation is None


def test_old_configuration_and_non_curriculum_runs_keep_periodic_schedule():
    cfg = load_config()
    assert validation_after_stage(cfg)
    assert not validation_after_stage(cfg, "diagnostic")
    assert not validation_after_stage(cfg, "saving")
    cfg["curriculum"]["mode"] = "fixed"
    assert not validation_after_stage(cfg)
    cfg["curriculum"]["mode"] = "teaching"
    cfg["training"].pop("validation_schedule")
    validate_config(cfg)
    assert not validation_after_stage(cfg) and "validation_schedule" not in cfg["training"]
    cfg["training"]["validation_schedule"] = "misspelled"
    with pytest.raises(ValueError, match="validation_schedule"):
        validate_config(cfg)


def test_suite_marks_unvalidated_jobs_and_excludes_missing_curves(tmp_path, monkeypatch):
    from pvz_rl.learning import suite
    from pvz_rl.provenance import write_json

    cfg = load_config()
    cfg["training"]["learner_seeds"] = [101]
    cfg["evaluation"]["ood_families"] = []
    monkeypatch.setattr(suite, "require_cuda_training", lambda *a, **kw: None)
    monkeypatch.setattr(suite, "BASELINES", ("wait",))

    def incomplete_training(cfg, condition, seed, destination, **kwargs):
        write_json(
            destination / "status.json", {"state": "complete", "curriculum_incomplete": True}
        )

    def baseline_evaluation(cfg, **kwargs):
        assert kwargs["baseline"] == "wait" and kwargs["policy"] is None
        write_json(kwargs["output"] / "status.json", {"state": "complete"})

    def report(paths, output, cfg, curves):
        assert len(paths) == 1 and curves == []

    monkeypatch.setattr(suite, "train", incomplete_training)
    monkeypatch.setattr(suite, "evaluate", baseline_evaluation)
    monkeypatch.setattr(suite, "make_report", report)
    result = suite.run_suite(cfg, tmp_path / "suite")
    assert result["state"] == "complete"
    assert "masked-101" in result["skipped_evaluations"]
    assert set(result["evaluations"]) == {"wait/preset"}


@pytest.mark.learning
def test_incomplete_training_saves_checkpoint_report_and_never_evaluates(
    smoke_cfg, tmp_path, monkeypatch
):
    cfg = smoke_cfg
    cfg["training"].update(validation_schedule="stage_success", budget_unit="games", total_games=2)
    cfg["visualization"].update(enabled=True, demos=True)

    def unexpected(*a, **kw):
        pytest.fail("No mastery: normal validation/demos must not run")

    monkeypatch.setattr("pvz_rl.learning.training.evaluate", unexpected)
    monkeypatch.setattr("pvz_rl.presentation.visualization.create_demonstrations", unexpected)
    run = train(cfg, "masked", 101, tmp_path / "incomplete", validation_limit=1)
    status = read_json(run / "status.json")
    assert status["curriculum_incomplete"] and status["validation_deferred"]
    assert not (run / "best.zip").exists()
    assert not (run / "learning-curve.jsonl").exists()
    assert not (run / "visualizations/demos.json").exists()
    assert "only after each curriculum stage passes" in (
        run / "visualizations/index.html"
    ).read_text("utf-8")
    before, _ = load_policy(run / "final.zip")
    again = train(
        cfg, "masked", 101, tmp_path / "resumed", validation_limit=1, resume=run / "final.zip"
    )
    after, _ = load_policy(again / "final.zip")
    assert before.num_timesteps == after.num_timesteps
    assert not (again / "best.zip").exists()


@pytest.mark.learning
@pytest.mark.parametrize("budget_exhausted", [False, True])
def test_interrupted_stage_evaluation_resumes_without_relearning(
    smoke_cfg, tmp_path, monkeypatch, budget_exhausted
):
    cfg = smoke_cfg
    cfg["training"].update(
        validation_schedule="stage_success",
        budget_unit="games",
        total_games=2 if budget_exhausted else 100,
    )
    cfg["curriculum"].update(run_stage="saving", probe_interval_games=1, minimum_stage_games=1)
    original = ResearchCallback.cached_evaluation

    def interrupt(self, seeds, levels, family, destination, split, final=False):
        if split == "curriculum_validation":
            return evaluation_rows(seeds, levels, family)  # Scheduling control only.
        raise KeyboardInterrupt("controlled interruption at mastery evaluation")

    monkeypatch.setattr(ResearchCallback, "cached_evaluation", interrupt)
    first = tmp_path / "first"
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, first, validation_limit=1)
    before, _ = load_policy(first / "interrupted.zip")
    assert before.research_schedule["pending_stage_validation"]["stage"] == "saving"
    monkeypatch.setattr(ResearchCallback, "cached_evaluation", original)
    resumed = train(
        cfg,
        "masked",
        101,
        tmp_path / "resumed",
        validation_limit=1,
        resume=first / "interrupted.zip",
    )
    after, _ = load_policy(resumed / "final.zip")
    assert before.num_timesteps == after.num_timesteps and before._n_updates == after._n_updates
    rows = read_series(resumed / "learning-curve.jsonl")
    assert len(rows) == 1 and rows[0]["stage_success"]["stage"] == "saving"
    assert read_json(resumed / "best.json")["checkpoint_hash"] == file_hash(resumed / "best.zip")
    assert after.research_schedule["pending_stage_validation"] is None
