"""Stage isolation, mastery boundaries and learning-state handoff controls."""

import argparse
import copy

import pytest
import torch

from pvz_rl.cli import configured, main
from pvz_rl.config import load_config, validate_config
from pvz_rl.curriculum import STAGES, CurriculumState, stage_distribution
from pvz_rl.provenance import file_hash, write_json
from pvz_rl.training import ResearchCallback, load_policy, train
from pvz_rl.training_requirements import require_cuda_training, transfer_protocol
from pvz_rl.visualization import build_run_report, read_json, read_series


@pytest.fixture
def stage_cfg():
    cfg = load_config("configs/sc2-plant-rewards.toml")
    cfg["curriculum"]["run_stage"] = "placement"
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["training"].update(
        n_envs=2,
        rollout_steps_per_env=32,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        total_games=6,
        max_minutes=2,
    )
    cfg["visualization"].update(enabled=False, demos=False)
    return cfg


def assert_tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensor_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tensor_tree_equal(a, b)
    else:
        assert left == right


def test_mastery_holds_stage_requires_residency_and_consecutive_passes(stage_cfg):
    state = CurriculumState()
    state.completed_stage_games = 99
    assert not state.observe({"placement": 18}, 100, stage_cfg, advance=False)
    assert not state.observe({"placement": 18}, 200, stage_cfg, advance=False)
    assert not state.mastered  # Passing probes cannot replace stage residency.
    state.completed_episode(1)
    assert state.completed_stage_games == 99
    state.completed_episode(0)
    state.observe({"placement": 17}, 300, stage_cfg, advance=False)
    assert state.consecutive_passes == 0
    state.observe({"placement": 18}, 400, stage_cfg, advance=False)
    restored = CurriculumState(**state.to_dict())
    restored.observe({"placement": 20}, 500, stage_cfg, advance=False)
    assert restored.mastered and restored.name == "placement"
    assert not restored.due(600, stage_cfg)
    assert stage_distribution(stage_cfg, 1) == (["saving", "placement"], [0.8, 0.2])
    # The ordinary automatic curriculum still promotes on the same gates.
    state.observe({"placement": 18}, 500, stage_cfg)
    assert state.name == "saving" and not state.mastered
    shared = CurriculumState(stage=4, completed_stage_games=1000)
    shared.observe({}, 1000, stage_cfg, advance=False)
    assert not shared.mastered and not shared.due(2000, stage_cfg)


@pytest.mark.parametrize("change", ["stage", "reward", "policy", "tasks", "optimizer", "engine"])
def test_transfer_compatibility_preserves_learning_contract(stage_cfg, change):
    altered = copy.deepcopy(stage_cfg)
    altered["training"].update(total_games=1200, max_minutes=60, eval_interval_games=500)
    if change == "stage":
        altered["curriculum"]["run_stage"] = "saving"
    elif change == "reward":
        altered["reward"]["gamma"] = 0.9999
    elif change == "policy":
        altered["policy"]["channels"] = 32
    elif change == "tasks":
        altered["curriculum"]["lessons"]["saving"]["initial_sun"] = 100
    elif change == "optimizer":
        altered["training"]["learning_rate"] = 1e-4
    else:
        altered["engine_commit"] = "different"
    assert (transfer_protocol(stage_cfg, "masked") == transfer_protocol(altered, "masked")) == (
        change == "stage"
    )
    assert stage_cfg["curriculum"]["run_stage"] == "placement"


def test_cli_inherits_source_config_and_seed_but_requires_explicit_stage(stage_cfg, tmp_path):
    write_json(
        tmp_path / "metadata.json",
        {
            "config": stage_cfg,
            "learner_seed": 102,
            "condition": "sparse",
            "validation_limit": 1,
        },
    )
    args = argparse.Namespace(
        command="train",
        config=None,
        init_from=tmp_path / "final.zip",
        stage="saving",
        games=50,
        max_minutes=10,
    )
    cfg = configured(args)
    assert cfg["curriculum"]["run_stage"] == "saving"
    assert cfg["training"]["total_games"] == 50 and cfg["training"]["max_minutes"] == 10
    assert (args.seed, args.condition, args.validation_count) == (102, "sparse", 1)
    args.stage = None
    with pytest.raises(ValueError, match="requires --stage"):
        configured(args)
    with pytest.raises(SystemExit):
        main(["train", "--resume", "one.zip", "--init-from", "two.zip", "--output", "unused"])


@pytest.mark.parametrize(
    "case", ["fixed", "unknown", "mixed", "unmasked", "diagnostic", "missing-stage", "both"]
)
def test_invalid_stage_requests_leave_no_run(stage_cfg, tmp_path, case):
    kwargs = {}
    condition = "masked"
    if case == "fixed":
        stage_cfg["curriculum"]["mode"] = "fixed"
    elif case == "unknown":
        stage_cfg["curriculum"]["run_stage"] = "hard"
    elif case in ("mixed", "unmasked"):
        condition = case
    elif case == "diagnostic":
        kwargs["family"] = "diagnostic"
    elif case == "missing-stage":
        stage_cfg["curriculum"].pop("run_stage")
        kwargs["init_from"] = tmp_path / "absent.zip"
    else:
        kwargs.update(resume=tmp_path / "one.zip", init_from=tmp_path / "two.zip")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        train(stage_cfg, condition, 101, output, **kwargs)
    assert not output.exists()


def test_incompatible_checkpoint_rejected_before_run(stage_cfg, tmp_path):
    saved = copy.deepcopy(stage_cfg)
    saved["reward"]["gamma"] = 0.9999
    write_json(
        tmp_path / "metadata.json", {"config": saved, "condition": "masked", "family": "preset"}
    )
    with pytest.raises(ValueError, match="matching engine, policy, rewards"):
        train(stage_cfg, "masked", 101, tmp_path / "out", init_from=tmp_path / "absent.zip")
    assert not (tmp_path / "out").exists()


@pytest.mark.learning
def test_all_stage_handoffs_preserve_weights_optimizer_and_reset_schedules(
    stage_cfg, tmp_path, monkeypatch
):
    original = ResearchCallback._on_training_start
    expected, starts = None, []

    def inspect_start(self):
        if expected is not None:
            assert_tensor_tree_equal(expected.policy.state_dict(), self.model.policy.state_dict())
            assert_tensor_tree_equal(
                expected.policy.optimizer.state_dict(), self.model.policy.optimizer.state_dict()
            )
            assert self.model._n_updates == expected._n_updates
        original(self)
        starts.append(
            (
                self.model.num_timesteps,
                self.model.training_games if hasattr(self.model, "training_games") else 0,
            )
        )
        assert self.initial_games == self.initial_steps == 0
        assert self.next_eval == stage_cfg["training"]["eval_interval_games"]
        assert self.wall_budget.prior_elapsed == 0
        assert self.curriculum.completed_stage_games == self.curriculum.consecutive_passes == 0
        assert self.training_env.queue.stage == STAGES.index(stage_cfg["curriculum"]["run_stage"])

    monkeypatch.setattr(ResearchCallback, "_on_training_start", inspect_start)
    previous = None
    for index, name in enumerate(STAGES):
        stage_cfg["curriculum"]["run_stage"] = name
        run = tmp_path / name
        train(stage_cfg, "masked", 102, run, validation_limit=1, init_from=previous)
        status = read_json(run / "status.json")
        assert status["selected_stage"] == status["curriculum_stage"] == name
        assert status["stop_reason"] == "game_budget" and not status["stage_mastered"]
        assert status["training_games"] >= 6
        episodes = read_series(run / "training-episodes.jsonl")
        assert {row["episode_start_stage"] for row in episodes} == {index}
        allowed = set(stage_cfg["curriculum"]["stages"][name]["tasks"])
        assert all(
            (row["level"] if row["family"] == "preset" else row["family"]) in allowed
            for row in episodes
        )
        curves = read_series(run / "learning-curve.jsonl")
        assert len(curves) == 1 and len(curves[0]["levels"]) == 3
        model, meta = load_policy(run / "final.zip", "cuda")
        if previous:
            assert meta["initialization"]["checkpoint_sha256"] == file_hash(previous)
            assert meta["initialization"]["games"] == expected.training_games
            assert model._n_updates > expected._n_updates
        else:
            assert meta["initialization"] is None
        expected, previous = model, run / "final.zip"
        assert model.policy.optimizer.state
    assert starts == [(0, 0)] * 5
    build_run_report(tmp_path / "shared", stage_cfg)
    page = (tmp_path / "shared/visualizations/index.html").read_text("utf-8")
    assert "Single stage: shared" in page and "initialization" in page


@pytest.mark.learning
def test_mastered_stage_saves_without_collecting_next_stage(stage_cfg, tmp_path, monkeypatch):
    stage_cfg["training"]["total_games"] = 1000
    stage_cfg["curriculum"].update(probe_interval_games=1, minimum_stage_games=1)
    original = ResearchCallback.cached_evaluation

    def passing_probe(self, seeds, levels, family, destination, split, final=False):
        if split == "curriculum_validation":
            # Scheduling control only; this is not evidence of learned mastery.
            return [{"win": True} for _ in seeds]
        return original(self, seeds, levels, family, destination, split, final)

    monkeypatch.setattr(ResearchCallback, "cached_evaluation", passing_probe)
    run = train(stage_cfg, "masked", 101, tmp_path / "mastered", validation_limit=1)
    status = read_json(run / "status.json")
    assert status["stage_mastered"] and status["stop_reason"] == "stage_mastered"
    assert not status["budget_complete"]
    assert {r["episode_start_stage"] for r in read_series(run / "training-episodes.jsonl")} == {0}
    assert len(read_series(run / "curriculum-probes.jsonl")) == 2
    model, _ = load_policy(run / "latest.zip")
    assert model.curriculum_state["mastered"] and model.curriculum_state["stage"] == 0
    assert read_series(run / "training-metrics.jsonl")[-1]["training_steps"] == status["steps"]
    again = train(
        stage_cfg,
        "masked",
        101,
        tmp_path / "done-resume",
        validation_limit=1,
        resume=run / "final.zip",
    )
    restored, _ = load_policy(again / "final.zip")
    assert restored.num_timesteps == model.num_timesteps
    assert_tensor_tree_equal(
        model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()
    )


@pytest.mark.learning
def test_stage_interrupt_resume_retains_stage_and_budget(stage_cfg, tmp_path, monkeypatch):
    stage_cfg["curriculum"]["run_stage"] = "saving"
    stage_cfg["training"]["total_games"] = 24
    original = ResearchCallback._on_rollout_start

    def interrupt(self):
        if self.model.num_timesteps:
            raise KeyboardInterrupt()
        return original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", interrupt)
    first = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        train(stage_cfg, "masked", 102, first, validation_limit=1)
    checkpoint, _ = load_policy(first / "interrupted.zip")
    assert checkpoint.curriculum_state["stage"] == 1
    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", original)
    second = train(
        stage_cfg,
        "masked",
        102,
        tmp_path / "resumed",
        validation_limit=1,
        resume=first / "interrupted.zip",
    )
    model, meta = load_policy(second / "final.zip")
    assert model.training_games >= 24
    assert meta["resume_games"] == checkpoint.training_games
    assert model.curriculum_state["stage"] == 1
    assert (
        model.wall_budget_state["elapsed_seconds"] > checkpoint.wall_budget_state["elapsed_seconds"]
    )
    assert model._n_updates > checkpoint._n_updates
    assert {r["episode_start_stage"] for r in read_series(second / "training-episodes.jsonl")} == {
        1
    }
    changed = copy.deepcopy(stage_cfg)
    changed["curriculum"]["run_stage"] = "easy"
    with pytest.raises(ValueError, match="Resume requires identical"):
        train(
            changed,
            "masked",
            102,
            tmp_path / "bad-resume",
            validation_limit=1,
            resume=first / "interrupted.zip",
        )
    assert not (tmp_path / "bad-resume").exists()


def test_existing_profiles_remain_valid_without_stage():
    cfg = load_config()
    validate_config(cfg)
    require_cuda_training(cfg, runtime=False)
    assert "run_stage" not in cfg["curriculum"]
