"""Stage isolation, mastery boundaries and learning-state handoff controls."""

import argparse
import copy
from pathlib import Path

import pytest
import torch

from pvz_rl.cli import configured, main
from pvz_rl.config import curriculum_probe_seeds, load_config, seed_values, validate_config
from pvz_rl.learning.curriculum import STAGES, CurriculumState, stage_distribution
from pvz_rl.learning.training import ResearchCallback, load_policy, train
from pvz_rl.learning.training_requirements import require_cuda_training, transfer_protocol
from pvz_rl.presentation.visualization import build_run_report, read_json, read_series
from pvz_rl.provenance import file_hash, write_json


@pytest.fixture
def stage_cfg():
    cfg = load_config("configs/train.toml")
    cfg["curriculum"]["run_stage"] = "saving"
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["training"].update(
        n_envs=2,
        rollout_steps_per_env=32,
        rollout_size=64,
        batch_size=32,
        n_epochs=1,
        critic_warmup_games=2,
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


def test_mastery_holds_stage_requires_residency_and_consecutive_passes(stage_cfg, legacy_teaching):
    legacy_teaching(stage_cfg)
    stage_cfg["curriculum"]["residency"] = "episode_start_stage"
    state = CurriculumState()
    state.completed_stage_games = 99
    assert not state.observe({"saving": 18}, 100, stage_cfg, advance=False)
    assert not state.observe({"saving": 18}, 200, stage_cfg, advance=False)
    assert not state.mastered  # Passing probes cannot replace stage residency.
    state.completed_episode(1)
    assert state.completed_stage_games == 99
    state.completed_episode(0)
    state.observe({"saving": 17}, 300, stage_cfg, advance=False)
    assert state.consecutive_passes == 0
    state.observe({"saving": 18}, 400, stage_cfg, advance=False)
    restored = CurriculumState(**state.to_dict())
    restored.observe({"saving": 20}, 500, stage_cfg, advance=False)
    assert restored.mastered and restored.name == "saving"
    assert not restored.due(600, stage_cfg)
    assert stage_distribution(stage_cfg, 1) == (["easy", "saving"], [0.8, 0.2])
    # The ordinary automatic curriculum still promotes on the same gates.
    state.observe({"saving": 18}, 500, stage_cfg)
    assert state.name == "easy" and not state.mastered
    shared = CurriculumState(stage=3, completed_stage_games=1000)
    shared.observe({}, 1000, stage_cfg, advance=False)
    assert not shared.mastered and not shared.due(2000, stage_cfg)


@pytest.mark.parametrize(
    "change", ["stage", "reward", "discount", "policy", "tasks", "optimizer", "engine"]
)
def test_transfer_compatibility_preserves_learning_contract(stage_cfg, change):
    altered = copy.deepcopy(stage_cfg)
    altered["training"].update(total_games=1200, max_minutes=60, eval_interval_games=500)
    if change == "stage":
        altered["curriculum"]["run_stage"] = "saving"
    elif change == "reward":
        altered["reward"]["basic_zombie_value"] = 25
    elif change == "discount":
        altered["training"]["gamma"] = 0.9999
    elif change == "policy":
        altered["policy"]["channels"] = [48, 48]
    elif change == "tasks":
        altered["curriculum"]["lessons"]["saving"]["initial_sun"] = 100
    elif change == "optimizer":
        altered["training"]["learning_rate"] = 1e-4
    else:
        altered["engine_commit"] = "different"
    assert (transfer_protocol(stage_cfg, "masked") == transfer_protocol(altered, "masked")) == (
        change not in ("policy", "engine")
    )
    assert stage_cfg["curriculum"]["run_stage"] == "saving"


def test_cli_inherits_source_config_and_seed_but_requires_explicit_stage(stage_cfg, tmp_path):
    write_json(
        tmp_path / "metadata.json",
        {
            "config": stage_cfg,
            "learner_seed": 102,
            "condition": "masked",
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
    assert (args.seed, args.condition, args.validation_count) == (102, "masked", 1)
    args.stage = None
    assert configured(args)["curriculum"]["run_stage"] == "saving"
    with pytest.raises(SystemExit):
        main(["train", "--resume", "one.zip", "--init-from", "two.zip", "--output", "unused"])


@pytest.mark.parametrize("case", ["fixed", "unknown", "mixed", "unmasked", "diagnostic", "both"])
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
    saved["policy"]["plant_embedding"] = 16
    write_json(
        tmp_path / "metadata.json", {"config": saved, "condition": "masked", "family": "preset"}
    )
    with pytest.raises(ValueError, match="matching engine, observation and network structure"):
        train(stage_cfg, "masked", 101, tmp_path / "out", init_from=tmp_path / "absent.zip")
    assert not (tmp_path / "out").exists()


@pytest.mark.learning
def test_all_stage_handoffs_preserve_only_weights_and_reset_optimizer_schedules(
    stage_cfg, tmp_path, monkeypatch
):
    original = ResearchCallback._on_training_start
    expected, starts = None, []

    def inspect_start(self):
        if expected is not None:
            assert_tensor_tree_equal(expected.policy.state_dict(), self.model.policy.state_dict())
            assert not self.model.policy.optimizer.state
            assert not self.model.policy.critic_optimizer.state
            assert self.model._n_updates == 0
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
        assert not curves and not (run / "best.zip").exists()
        assert status["validation_deferred"]
        model, meta = load_policy(run / "final.zip", "cuda")
        if previous:
            assert meta["initialization"]["checkpoint_sha256"] == file_hash(previous)
            assert meta["initialization"]["games"] == expected.training_games
            assert model._n_updates > 0
        else:
            assert meta["initialization"] is None
        expected, previous = model, run / "final.zip"
        assert model.policy.optimizer.state
        assert model.policy.critic_optimizer.state
    assert starts == [(0, 0)] * len(STAGES)
    build_run_report(tmp_path / "shared", stage_cfg)
    page = (tmp_path / "shared/visualizations/index.html").read_text("utf-8")
    assert "Single stage: shared" in page and "initialization" in page


@pytest.mark.learning
@pytest.mark.parametrize("unlimited", [False, True])
def test_mastered_stage_saves_without_collecting_next_stage(
    stage_cfg, tmp_path, monkeypatch, unlimited
):
    from pvz_rl.presentation.recordings import open_playback

    stage_cfg["training"]["total_games"] = 1000
    if unlimited:
        stage_cfg["training"].update(until_stage_complete=True, total_games=1, max_minutes=1e-9)
    stage_cfg["visualization"].update(enabled=True, demos=True)
    stage_cfg["curriculum"].update(probe_interval_games=1, minimum_stage_games=1)
    original = ResearchCallback.cached_evaluation

    def passing_probe(self, seeds, levels, family, destination, split, final=False):
        if split == "curriculum_validation":
            # Scheduling control only; this is not evidence of learned mastery.
            # The first unlimited probe fails; neither former ceiling may stop learning.
            return [
                {"win": not (unlimited and self.curriculum.last_probe_games == 0 and i == 0)}
                for i, _ in enumerate(seeds)
            ]
        return original(self, seeds, levels, family, destination, split, final)

    monkeypatch.setattr(ResearchCallback, "cached_evaluation", passing_probe)
    run = train(stage_cfg, "masked", 101, tmp_path / "mastered", validation_limit=1)
    status = read_json(run / "status.json")
    assert status["stage_mastered"] and status["stop_reason"] == "stage_mastered"
    assert not status["budget_complete"]
    assert {r["episode_start_stage"] for r in read_series(run / "training-episodes.jsonl")} == {0}
    assert len(read_series(run / "curriculum-probes.jsonl")) == (2 if unlimited else 1)
    if unlimited:
        assert (
            status["target_games"] is status["budget_target"] is status["budget_complete"] is None
        )
        assert status["estimated_remaining_training_seconds"] is None
        assert status["time_budget"]["limit_seconds"] is None
        assert status["training_games"] > 1
        log = (run / "train.log").read_text()
        assert "until stage mastery" in log and "training ETA" not in log
    curves = read_series(run / "learning-curve.jsonl")
    assert len(curves) == 1 and curves[0]["stage_success"]["stage"] == "saving"
    demos = read_json(run / "visualizations/demos.json")["demos"]
    assert {d["level"] for d in demos} == {"easy", "standard", "hard"}
    assert {d["checkpoint_hash"] for d in demos} == {file_hash(run / "best.zip")}
    for demo in demos:
        playback = open_playback(run / "visualizations" / demo["replay"])
        playback.verify()
        assert playback.game.state_hash() == demo["state_hash"]
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
    assert not read_series(again / "learning-curve.jsonl")
    assert restored.num_timesteps == model.num_timesteps
    assert_tensor_tree_equal(
        model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()
    )
    assert_tensor_tree_equal(
        model.policy.critic_optimizer.state_dict(), restored.policy.critic_optimizer.state_dict()
    )


@pytest.mark.learning
@pytest.mark.parametrize("unlimited", [False, True])
def test_stage_interrupt_resume_retains_stage_and_budget(
    stage_cfg, tmp_path, monkeypatch, unlimited
):
    stage_cfg["curriculum"]["run_stage"] = "saving"
    stage_cfg["training"]["total_games"] = 24
    if unlimited:
        stage_cfg["training"].update(until_stage_complete=True, total_games=1, max_minutes=1e-9)
        stage_cfg["curriculum"].update(probe_interval_games=2, minimum_stage_games=1)
        original_eval = ResearchCallback.cached_evaluation

        def mastery_control(self, seeds, levels, family, destination, split, final=False):
            if split == "curriculum_validation":
                return [{"win": self.model.training_games >= 24} for _ in seeds]
            return original_eval(self, seeds, levels, family, destination, split, final)

        monkeypatch.setattr(ResearchCallback, "cached_evaluation", mastery_control)
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
    assert checkpoint.curriculum_state["stage"] == 0
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
    if unlimited:
        status = read_json(second / "status.json")
        assert (
            status["stop_reason"] == "stage_mastered"
            and status["time_budget"]["limit_seconds"] is None
        )
    assert meta["resume_games"] == checkpoint.training_games
    assert model.curriculum_state["stage"] == 0
    assert (
        model.wall_budget_state["elapsed_seconds"] > checkpoint.wall_budget_state["elapsed_seconds"]
    )
    assert model._n_updates > checkpoint._n_updates
    assert {r["episode_start_stage"] for r in read_series(second / "training-episodes.jsonl")} == {
        0
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


@pytest.mark.parametrize("retired", ["architecture", "observation"])
def test_cli_retired_architecture_rejected_for_init_and_resume(stage_cfg, tmp_path, retired):
    section, key, value = (
        ("policy", "kind", "spatial_grouped_v3")
        if retired == "architecture"
        else ("encoding", "version", "event_v4")
    )
    stage_cfg[section][key] = value
    write_json(
        tmp_path / "metadata.json",
        {"config": stage_cfg, "learner_seed": 101, "condition": "masked"},
    )
    for mode in ("init_from", "resume"):
        with pytest.raises(ValueError, match="Retired"):
            configured(
                argparse.Namespace(command="train", config=None, **{mode: tmp_path / "final.zip"})
            )
    assert read_json(tmp_path / "metadata.json")["config"][section][key] == value


@pytest.mark.parametrize("stage", STAGES)
def test_new_mastery_needs_every_case_and_correct_stage_residency(stage_cfg, stage):
    state = CurriculumState(stage=STAGES.index(stage), completed_stage_games=99)
    wins = {task: 100 for task in state.requirements(stage_cfg)}
    assert not state.due(1999, stage_cfg) and state.due(2000, stage_cfg)
    state.observe(wins, 500, stage_cfg, advance=False)
    assert not state.mastered  # Even perfect probes cannot replace training residency.
    state.completed_episode((state.stage + 1) % len(STAGES))
    assert state.completed_stage_games == 99
    state.completed_episode(state.stage)
    for index, task in enumerate(wins, 2):
        # One failure on any required task blocks the whole stage, including shared.
        state.observe({**wins, task: 99}, index * 500, stage_cfg, advance=False)
        assert not state.mastered and state.consecutive_passes == 0
    assert not state.complete(stage_cfg)
    restored = CurriculumState(**state.to_dict())
    restored.observe(wins, 2500, stage_cfg, advance=False)
    assert restored.mastered and restored.name == stage
    assert restored.complete(stage_cfg) == (stage == "shared")
    assert not restored.due(3000, stage_cfg)


def test_automatic_curriculum_finishes_only_after_shared_mastery(stage_cfg):
    state = CurriculumState()
    for index, stage in enumerate(STAGES):
        assert state.name == stage
        state.completed_stage_games = 100
        wins = {task: 100 for task in state.requirements(stage_cfg)}
        assert state.observe(wins, (index + 1) * 500, stage_cfg) == (stage != "shared")
        assert state.complete(stage_cfg) == (stage == "shared")
    assert state.mastered and state.stage == 3
    assert not state.observe({"easy": 100, "standard": 100, "hard": 100}, 3000, stage_cfg)
    assert state.stage == 3  # Never advance beyond the last valid stage.


def test_mastery_seed_pool_and_retained_recipe_defaults():
    from pathlib import Path

    for path in Path("configs").glob("*.toml"):
        cfg = load_config(path)
        assert cfg["training"]["eval_interval_games"] == 2000
        assert seed_values(cfg, "validation") == list(range(100000, 100050))
        if cfg["curriculum"].get("mode") != "teaching":
            continue
        assert cfg["curriculum"]["probe_interval_games"] == 2000
        assert cfg["training"]["validation_schedule"] == "stage_success"
        assert cfg["curriculum"]["consecutive_passes"] == 1
        cases = curriculum_probe_seeds(cfg)
        assert cases == list(range(100050, 100150))
        assert not set(cases) & set(seed_values(cfg, "validation"))
        assert not set(cases) & set(seed_values(cfg, "train"))
        assert not set(cases) & set(seed_values(cfg, "test"))
        for stage in STAGES:
            assert set(cfg["curriculum"]["stages"][stage]["requirements"].values()) == {100}


def test_old_gates_are_preserved_on_resume_but_can_change_on_handoff(
    stage_cfg, legacy_teaching, tmp_path
):
    from pvz_rl.learning.training_requirements import resume_protocol

    old = legacy_teaching(copy.deepcopy(stage_cfg))
    old["training"]["eval_interval_games"] = 1000
    write_json(tmp_path / "metadata.json", {"config": old})
    args = argparse.Namespace(command="train", config=None, resume=tmp_path / "final.zip")
    restored = configured(args)
    assert restored == old
    assert curriculum_probe_seeds(restored) == list(range(100000, 100020))
    assert transfer_protocol(old, "masked") == transfer_protocol(stage_cfg, "masked")
    assert resume_protocol(old, "masked") != resume_protocol(stage_cfg, "masked")
    old_state = CurriculumState(stage=3)
    assert old_state.complete(old) and not old_state.due(10000, old)


@pytest.mark.parametrize("count", [-1, 101, 99.5, True])
def test_impossible_probe_counts_are_rejected(stage_cfg, count):
    state = CurriculumState(completed_stage_games=100)
    with pytest.raises(ValueError, match="Probe wins"):
        state.observe({"saving": count}, 500, stage_cfg)
    assert state.last_probe_games == 0 and not state.mastered


@pytest.mark.parametrize("failure", ["loss", "truncation", "partial"])
def test_probe_failure_does_not_certify_mastery(stage_cfg, tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    cb = ResearchCallback(stage_cfg, "masked", 101, tmp_path, validation_limit=1)
    cb.model = SimpleNamespace(num_timesteps=64, training_games=2000, save=lambda p: None)
    cb.curriculum = CurriculumState(completed_stage_games=2000)
    seen = []

    def results(seeds, *args, **kwargs):
        seen.extend(seeds)
        rows = [{"win": True} for _ in seeds]
        if failure == "partial":
            return rows[:-1]
        rows[-1] = {"win": False, "truncated": failure == "truncation"}
        return rows

    monkeypatch.setattr(cb, "cached_evaluation", results)
    try:
        if failure == "partial":
            with pytest.raises(ValueError, match="Incomplete curriculum"):
                cb.probe_curriculum()
        else:
            cb.probe_curriculum()
        assert len(seen) == 100  # --validation-count never shrinks mastery probes.
        assert not cb.curriculum.mastered
    finally:
        cb.close()


@pytest.mark.learning
@pytest.mark.parametrize("standalone", [False, True])
def test_shared_mastery_stops_after_update_and_is_resumable(
    stage_cfg, tmp_path, monkeypatch, standalone
):
    from pvz_rl.learning.curriculum import initial_state

    stage_cfg["training"]["total_games"] = 1000
    stage_cfg["curriculum"].update(
        run_stage="shared", probe_interval_games=1, minimum_stage_games=1
    )
    if not standalone:
        stage_cfg["curriculum"].pop("run_stage")
    original = ResearchCallback.cached_evaluation

    def passing_probe(self, seeds, levels, family, destination, split, final=False):
        if split == "curriculum_validation":
            return [{"win": True} for _ in seeds]  # Scheduling control, not learned wins.
        return original(self, seeds, levels, family, destination, split, final)

    if not standalone:
        original_build = __import__("pvz_rl.learning.training", fromlist=["build_model"]).build_model

        def build(*args, **kwargs):
            model = original_build(*args, **kwargs)
            state = initial_state(stage_cfg)
            state.stage = len(STAGES) - 1
            model.curriculum_state = state.to_dict()
            return model

        monkeypatch.setattr("pvz_rl.learning.training.build_model", build)
    monkeypatch.setattr(ResearchCallback, "cached_evaluation", passing_probe)
    run = train(stage_cfg, "masked", 101, tmp_path / "shared", validation_limit=1)
    status = read_json(run / "status.json")
    assert status["stage_mastered"] and not status["curriculum_incomplete"]
    assert status["stop_reason"] == ("stage_mastered" if standalone else "curriculum_mastered")
    model, _ = load_policy(run / "final.zip")
    assert model._n_updates >= 1 and model.curriculum_state["stage"] == len(STAGES) - 1
    resumed = train(
        stage_cfg, "masked", 101, tmp_path / "resume", validation_limit=1, resume=run / "final.zip"
    )
    restored, _ = load_policy(resumed / "final.zip")
    assert model.num_timesteps == restored.num_timesteps
    assert_tensor_tree_equal(
        model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()
    )
    assert_tensor_tree_equal(
        model.policy.critic_optimizer.state_dict(), restored.policy.critic_optimizer.state_dict()
    )


@pytest.mark.parametrize("option", ["games", "steps", "max_minutes"])
def test_until_mastery_cli_rejects_explicit_ceilings_before_output(tmp_path, option):
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="cannot be combined"):
        configured(
            argparse.Namespace(
                command="train",
                config=None,
                stage="saving",
                until_stage_complete=True,
                output=output,
                **{option: 1},
            )
        )
    assert not output.exists()


@pytest.mark.parametrize("change", ["stage", "schedule", "mode", "requirements", "boolean"])
def test_until_mastery_invalid_configuration_leaves_no_output(stage_cfg, tmp_path, change):
    stage_cfg["training"]["until_stage_complete"] = True
    if change == "stage":
        stage_cfg["curriculum"].pop("run_stage")
    elif change == "schedule":
        stage_cfg["training"]["budget_unit"] = "decisions"
    elif change == "mode":
        stage_cfg["curriculum"]["mode"] = "fixed"
    elif change == "boolean":
        stage_cfg["training"]["until_stage_complete"] = 1
    else:
        stage_cfg["curriculum"]["stages"]["saving"]["requirements"] = {}
    with pytest.raises(ValueError, match="until_stage_complete"):
        train(stage_cfg, "masked", 101, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_until_mastery_cli_saved_resume_and_unbounded_clock(stage_cfg, tmp_path):
    from pvz_rl.learning.budget import budget_target, effective_limits
    from pvz_rl.learning.deadline import RunBudget

    cfg = configured(
        argparse.Namespace(command="train", config=None, stage="saving", until_stage_complete=True)
    )
    assert budget_target(cfg) is None
    assert effective_limits(cfg) == {"games": None, "decisions": None, "minutes": None}
    assert cfg["training"]["total_games"] == 10000  # inactive defaults remain explicit
    clock = [100.0]
    budget = RunBudget(cfg, elapsed=10**10, clock=lambda: clock[0])
    clock[0] += 10**10
    assert budget.can_collect(10**12) and budget.deadline is None
    assert budget.state()["limit_seconds"] is budget.state()["remaining_seconds"] is None
    assert budget.state()["reserve_seconds"] == 0
    assert budget.state()["elapsed_seconds"] == 2 * 10**10
    write_json(
        tmp_path / "metadata.json",
        {"config": cfg, "learner_seed": 101, "condition": "masked", "validation_limit": None},
    )
    restored = configured(
        argparse.Namespace(command="train", config=None, resume=tmp_path / "latest.zip")
    )
    assert restored == cfg
    assert transfer_protocol(cfg) == transfer_protocol(stage_cfg)
    with pytest.raises(ValueError, match="explicit training deadline"):
        train(cfg, "masked", 101, tmp_path / "absent", deadline=1)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("mode", ["resume", "init_from"])
def test_previous_action_head_rejected_before_loading(stage_cfg, tmp_path, mode):
    stage_cfg["policy"]["kind"] = "event_transformer_v1"
    write_json(
        tmp_path / "metadata.json",
        {"config": stage_cfg, "learner_seed": 101, "condition": "masked"},
    )
    with pytest.raises(ValueError, match="Retired"):
        configured(
            argparse.Namespace(command="train", config=None, **{mode: tmp_path / "absent.zip"})
        )


def test_previous_action_distribution_rejected_by_metadata_and_direct_loader(stage_cfg, tmp_path, monkeypatch):
    import json
    import zipfile

    from pvz_rl.learning.cuda_ppo import CudaMaskablePPO
    from pvz_rl.learning.training_requirements import current_model_config

    stage_cfg["policy"].pop("action_distribution")
    assert not current_model_config(stage_cfg)
    write_json(tmp_path / "metadata.json", {"config": stage_cfg, "condition": "masked"})
    with pytest.raises(ValueError, match="Retired"):
        load_policy(tmp_path / "absent.zip")
    checkpoint = tmp_path / "old.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"optimizer_protocol": "periodic_exact_kl_v1"}))
    with pytest.raises(ValueError, match="Action distribution changed"):
        CudaMaskablePPO.load(checkpoint)
    # A mismatched sidecar must not allow old archive weights into a new experiment.
    stage_cfg["policy"]["action_distribution"] = "balanced_species_tiles_v1"
    write_json(tmp_path / "metadata.json", {"config": stage_cfg, "condition": "masked",
               "exploration_protocol": "phase_floor_v1"})
    monkeypatch.setattr("stable_baselines3.common.save_util.load_from_zip_file",
                        lambda *args, **kwargs: ({}, {"policy": {}}, {}))
    from pvz_rl.learning.training import initial_weights
    with pytest.raises(ValueError, match="Action distribution changed"):
        initial_weights(checkpoint, stage_cfg)


def test_until_stage_config_does_not_disable_benchmark_window(tmp_path):
    source = (
        Path("configs/train.toml")
        .read_text()
        .replace("until_stage_complete = false", "until_stage_complete = true")
    )
    source = source.replace("[curriculum]", '[curriculum]\nrun_stage = "saving"')
    path = tmp_path / "stage.toml"
    path.write_text(source)
    cfg = configured(argparse.Namespace(command="benchmark-gpu", config=path, steps=16384))
    assert cfg["training"]["until_stage_complete"]
    assert cfg["training"]["total_steps"] == 16384
