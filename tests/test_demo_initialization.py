"""Complete tiny replay fitting, independent group loss and corrupt-input controls."""

import json
from collections import defaultdict
from copy import deepcopy

import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Wait

from pvz_rl.config import load_config, load_demo_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.learning import demo_initialization as demo
from pvz_rl.learning import flat_q
from pvz_rl.learning.checkpoints import inspect_checkpoint
from pvz_rl.learning.objective import components, training_rewards
from pvz_rl.presentation.demo_recording import TransitionArchive
from pvz_rl.presentation.recordings import ActionPhaseRecorder


@pytest.fixture
def completed_demo(tmp_path, request):
    cfg = load_demo_config()
    cfg["environment"]["cutoff_seconds"] = 1
    # A recording made before clipping was shared must remain reusable.
    cfg["training"]["demo"]["gradient_clip"] = 0.5
    game = ActionPhaseGame()
    game.reset(LevelSpec("tiny-fit", initial_sun=200), seed=1000)
    recorder = ActionPhaseRecorder(game)
    replay, archive = tmp_path / "tiny.pvzdemo", tmp_path / "tiny.jsonl"
    writer = TransitionArchive(archive, cfg=cfg)
    ending = {
        "wait": Wait(),
        "plant": Place("sunflower", 0, 1),
        "dig": Dig(0, 1),
    }[getattr(request, "param", "wait")]
    try:
        for action in (
            Place("sunflower", 0, 0),
            Place("peashooter", 0, 1),
            Dig(0, 0),
            Dig(0, 1),
            ending,
        ):
            before = game.observe()
            ticks = int(isinstance(action, Wait) or not game.validate_action(action).accepted)
            result = recorder.step(action, ticks=ticks)
            writer.append(
                before,
                action=writer.codec.encode(action),
                accepted=result.action_result.accepted,
                rejection_reason=result.action_result.reason,
                reward=reward_parts(
                    before,
                    result.observation,
                    cfg,
                    events=result.events,
                    rules=game.rules,
                    action=action,
                    action_result=result.action_result,
                ),
                terminal=result.status.value,
                tick=before.tick,
                ticks_advanced=result.ticks_advanced,
            )
        assert game.observe().status.value == "won"
        recorder.save(replay)
        writer.finalize(game.observe(), outcome="won", replay_path=replay)
    finally:
        writer.close()
    return cfg, archive, replay


@pytest.mark.parametrize(
    "configured_passes, override, expected",
    [(20, None, 20), (3, None, 3), (20, 1, 1), (20, 0, None)],
)
def test_demo_passes_are_independent_of_autonomous_epochs(
    completed_demo, tmp_path, configured_passes, override, expected
):
    cfg, archive, replay = completed_demo
    assert cfg["training"]["demo"]["passes"] == 20
    assert cfg["training"]["n_epochs"] == load_config()["training"]["n_epochs"] == 4
    cfg["training"]["n_epochs"] = 1
    cfg["training"]["demo"]["passes"] = configured_passes
    output = tmp_path / "fit"
    if expected is None:
        with pytest.raises(ValueError, match="settings must be positive"):
            demo.initialize_demo(archive, replay, output, cfg=cfg, passes=override)
        assert not output.exists()
        return
    result = demo.initialize_demo(archive, replay, output, cfg=cfg, passes=override)
    assert result["state"] == "complete"
    assert result["passes_completed"] == expected
    curves = json.loads((output / "learning-curves.json").read_text())["curves"]
    assert len(curves) == expected
    assert cfg["training"]["n_epochs"] == 1


def test_demo_teacher_is_frozen_per_pass_and_updates_once_after_adam(
    completed_demo, tmp_path, monkeypatch
):
    cfg, archive, replay = completed_demo
    cfg["training"].update(batch_size=2)
    cfg["training"]["performance"]["fit_batch_groups"] = 1
    snapshots, committed = [], []
    original_loss, original_step = demo.batch_loss, torch.optim.Adam.step
    teacher_ref = None

    def observe_loss(policy, *args, teacher, **kwargs):
        nonlocal teacher_ref
        teacher_ref = teacher
        snapshots.append([parameter.detach().clone() for parameter in teacher.parameters()])
        return original_loss(policy, *args, teacher=teacher, **kwargs)

    def observe_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        committed.append(
            [
                parameter.detach().clone()
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        )
        return result

    monkeypatch.setattr(demo, "batch_loss", observe_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", observe_step)
    result = demo.initialize_demo(archive, replay, tmp_path / "ema-fit", cfg=cfg, passes=2)
    assert result["passes_completed"] == len(committed) == 2
    assert len(snapshots) == 6
    for before, within, within_last in zip(*snapshots[:3], strict=True):
        torch.testing.assert_close(before, within, atol=0, rtol=0)
        torch.testing.assert_close(before, within_last, atol=0, rtol=0)
    for old, online, expected in zip(snapshots[0], committed[0], snapshots[3], strict=True):
        torch.testing.assert_close(expected, old * 0.95 + online * 0.05)
    for before, within, within_last in zip(*snapshots[3:], strict=True):
        torch.testing.assert_close(before, within, atol=0, rtol=0)
        torch.testing.assert_close(before, within_last, atol=0, rtol=0)
    for old, online, final in zip(
        snapshots[3], committed[1], teacher_ref.parameters(), strict=True
    ):
        torch.testing.assert_close(final, old * 0.95 + online * 0.05)
        assert final.grad is None and not final.requires_grad


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_complete_demo_fit_uses_whole_game_group_loss_and_reloads(
    completed_demo, tmp_path, monkeypatch, device
):
    cfg, archive, replay = completed_demo
    cfg["training"]["demo"].pop("gradient_clip")
    cfg["training"]["max_grad_norm"] = 1.25
    cfg["training"]["objective"].update(tile_probes=1)
    samples = defaultdict(list)
    shared_loss = flat_q.balanced_q_loss
    clip_calls = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def observe_clip(parameters, max_norm):
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm)

    def observe(first, targets, actions, counts, **kwargs):
        for q1, target, action in zip(first.detach().tolist(), targets.tolist(), actions.tolist()):
            group = "wait" if action == 0 else "plant" if action < 361 else "dig"
            error = (q1 - target) ** 2
            samples[group].append(error)
        return shared_loss(first, targets, actions, counts, **kwargs)

    monkeypatch.setattr(flat_q, "balanced_q_loss", observe)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe_clip)
    result = demo.initialize_demo(
        archive, replay, tmp_path / "fit", cfg=cfg, passes=1, device=device
    )
    assert result["state"] == "complete" and result["passes_completed"] == 1
    assert clip_calls == [1.25]
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert "performance" not in payload["config"]["training"]
    assert "gradient_clip" not in payload["config"]["training"]["demo"]
    assert payload["config"]["training"]["max_grad_norm"] == 1.25
    assert not {"encoder_microbatch", "encoder_token_budget", "attention_query_chunk"} & set(
        payload["config"]["policy"]
    )
    inspected = inspect_checkpoint(result["checkpoint"])
    current = load_config()
    assert inspected["config"]["training"]["performance"] == current["training"]["performance"]
    assert inspected["config"]["training"]["objective"] == cfg["training"]["objective"]
    assert inspected["config"]["reward"] == cfg["reward"]
    assert (
        inspected["config"]["policy"]["encoder_microbatch"]
        == current["policy"]["encoder_microbatch"]
    )
    expected = sum(sum(errors) / len(errors) for errors in samples.values()) / len(samples)
    curves = json.loads((tmp_path / "fit/learning-curves.json").read_text())
    assert curves["curves"][0]["loss"] == pytest.approx(expected, rel=1e-6)
    coverage = json.loads((tmp_path / "fit/action-coverage.json").read_text())["branches"]
    assert coverage == [1, 1, 1, 0, 0, 0, 0, 0, 0, 2]
    model, saved = demo.load_demo_checkpoint(result["checkpoint"], cfg)
    assert saved["source_replay_sha256"] == demo.file_hash(replay)
    observations, *_ = demo._training_tensors(demo._load_verified_demo(archive, replay, cfg))
    assert torch.isfinite(model.species_values(observations[:1])).all()
    assert not list((tmp_path / "fit").glob("*.tmp"))
    saved["model_storage_protocol"] = "unknown"
    torch.save(saved, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="model storage"):
        demo.load_demo_checkpoint(tmp_path / "invalid.pt", cfg)


@pytest.mark.parametrize("completed_demo", ["wait", "plant", "dig"], indirect=True)
def test_reused_demo_fits_current_rewards_without_rewriting_recording(
    completed_demo, tmp_path, monkeypatch
):
    cfg, archive, replay = completed_demo
    before = {
        path: path.read_bytes()
        for path in (archive, replay, archive.with_suffix(".jsonl.manifest.json"))
    }
    current = deepcopy(cfg)
    current["reward"].update(
        win_reward=3.0,
        loss_penalty=4.0,
        progress_weight=0.02,
        value_scale=200.0,
        basic_zombie_value=100.0,
        mower_value=500.0,
        invalid_plant_penalty=0.09,
        empty_dig_penalty=0.07,
    )
    current["training"]["max_grad_norm"] = 2.0
    verified = demo._load_verified_demo(archive, replay, current)
    original = demo._load_verified_demo(archive, replay, cfg)
    tensors = demo._training_tensors(verified)
    original_tensors = demo._training_tensors(original)
    assert tensors[0] == original_tensors[0]
    torch.testing.assert_close(tensors[1], original_tensors[1], rtol=0, atol=0)
    # Independent control: dig loses 50 and 100 sun of plant assets; rejected
    # proposals cost their new price, including when the same tick ends the game.
    last = verified.transitions[-1]
    assert last["accepted"] == (last["action"] == 0)
    assert last["rejection_reason"] == (
        None if last["action"] == 0 else "card_recharging" if last["action"] < 361 else "empty_tile"
    )
    expected_returns = torch.zeros(len(verified.transitions))
    running = 0.0
    for index in range(len(verified.transitions) - 1, -1, -1):
        running += float(
            training_rewards(components(verified.transitions[index]["reward_parts"]), current)
        )
        expected_returns[index] = running
    torch.testing.assert_close(tensors[-1], expected_returns)
    assert original.report["reconstruction"]["reward_changes"] == 0
    assert verified.report["reconstruction"]["reward_changes"] == 3
    assert last["executed_action"] == 0  # Wait or rejected proposal.
    assert last["ticks_advanced"] == 1
    fitted_targets = []
    shared_loss = flat_q.balanced_q_loss

    def observe(first, targets, *args, **kwargs):
        fitted_targets.extend(targets.detach().cpu().tolist())
        return shared_loss(first, targets, *args, **kwargs)

    monkeypatch.setattr(flat_q, "balanced_q_loss", observe)
    result = demo.initialize_demo(archive, replay, tmp_path / "fit", cfg=current, passes=1)
    torch.testing.assert_close(torch.tensor(fitted_targets), expected_returns)
    assert result["state"] == "complete"
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize(
    "field",
    ["engine", "observation_schema", "replay", "complete", "outcome", "decisions", "episode_id"],
)
def test_manifest_corruption_blocks_initialization(completed_demo, tmp_path, field):
    cfg, archive, replay = completed_demo
    manifest = archive.with_suffix(".jsonl.manifest.json")
    data = json.loads(manifest.read_text())
    data[field] = {
        "engine": {**data["engine"], "commit": "wrong"},
        "observation_schema": {},
        "replay": "wrong",
        "complete": False,
        "outcome": "lost",
        "decisions": 4,
        "episode_id": "different",
    }[field]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        demo.initialize_demo(archive, replay, tmp_path / "fit", cfg=cfg, passes=1)
    assert not (tmp_path / "fit").exists()


@pytest.mark.parametrize(
    "field",
    [
        "observation",
        "action",
        "executed_action",
        "reward_parts",
        "terminal",
        "tick",
        "effective_damage",
        "net_value",
        "invalid_plant_penalty",
        "nonfinite_reward",
    ],
)
def test_transition_corruption_rejected_by_reconstruction(completed_demo, field):
    cfg, archive, replay = completed_demo
    rows = [json.loads(line) for line in archive.read_text().splitlines()]
    if field == "reward_parts":
        rows[0][field]["total"] += 1
    elif field in ("effective_damage", "net_value"):
        rows[0]["reward_parts"][field] += 1
    elif field == "invalid_plant_penalty":
        # Balanced totals alone cannot make a penalty on an accepted action valid.
        rows[0]["reward_parts"][field] = -0.001
        rows[0]["reward_parts"]["total"] = -0.001
    elif field == "nonfinite_reward":
        rows[0]["reward_parts"]["combat_value"] = float("nan")
    elif field == "observation":
        rows[0][field]["globals"][0] += 1
    elif field == "terminal":
        rows[0][field] = "lost"
    else:
        rows[0][field] += 1
    archive.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cfg["reward"]["win_reward"] += 1  # Changed settings must not bypass corruption checks.
    with pytest.raises(ValueError, match="mismatch"):
        demo.verify_demo(archive, replay, cfg)


def test_replay_reconstructs_only_current_boards_actions_and_returns(completed_demo):
    cfg, archive, replay = completed_demo
    verified = demo._load_verified_demo(archive, replay, cfg)
    observations, actions, returns = demo._training_tensors(verified)
    assert len(observations) == len(actions) == len(returns) == 5
    assert all("events" not in row and "memory_write" not in row for row in verified.transitions)
    assert [len(row["entities"]) for row in observations] == [5, 6, 7, 6, 5]


def test_early_sun_reconstructed_from_previous_ledger_without_rewriting(tmp_path):
    from pvz_game import InitialPlant, Spawn

    cfg = load_demo_config()
    cfg["reward"]["early_sun_extra_multiplier"] = 0
    game = ActionPhaseGame()
    game.reset(
        LevelSpec(
            "early-demo",
            tuple(Spawn(2000, "basic", row, x=1) for row in range(3)),
            plants=(InitialPlant("sunflower", 4, 0),),
            mowers=False,
        ),
        seed=7,
    )
    recorder = ActionPhaseRecorder(game)
    replay, archive = tmp_path / "early.pvzdemo", tmp_path / "early.jsonl"
    writer = TransitionArchive(archive, cfg=cfg)
    try:
        for _ in range(3000):
            before = game.observe()
            result = recorder.step(Wait(), ticks=1)
            parts = reward_parts(
                before,
                result.observation,
                cfg,
                events=result.events,
                rules=game.rules,
                action=Wait(),
                action_result=result.action_result,
            )
            parts.pop("early_sun")
            parts.pop("early_sun_bonus")
            writer.append(
                before,
                action=0,
                accepted=True,
                rejection_reason=None,
                reward=parts,
                terminal=result.status.value,
                tick=before.tick,
                ticks_advanced=1,
            )
            if result.status.value != "running":
                break
        assert result.status.value == "lost"
        recorder.save(replay)
        writer.finalize(game.observe(), outcome="lost", replay_path=replay)
    finally:
        writer.close()
    source = {
        path: path.read_bytes()
        for path in (archive, replay, archive.with_suffix(".jsonl.manifest.json"))
    }
    current = deepcopy(cfg)
    current["reward"]["early_sun_extra_multiplier"] = 2
    baseline = demo._load_verified_demo(archive, replay, cfg)
    verified = demo._load_verified_demo(archive, replay, current)
    early = sum(row["reward_parts"]["early_sun"] for row in verified.transitions)
    assert early > 0
    old_targets = demo._training_tensors(baseline)[-1]
    new_targets = demo._training_tensors(verified)[-1]
    assert float(new_targets[0] - old_targets[0]) == pytest.approx(2 * early / 3000, abs=2e-7)
    assert all(path.read_bytes() == contents for path, contents in source.items())


def test_checkpoint_write_failure_preserves_previous_pass(tmp_path, monkeypatch):
    checkpoint = tmp_path / "initialization.pt"
    checkpoint.write_bytes(b"previous complete pass")

    def fail(payload, path):
        path.write_bytes(b"partial write")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        demo._save_checkpoint({}, checkpoint)
    assert checkpoint.read_bytes() == b"previous complete pass"
    assert list(tmp_path.iterdir()) == [checkpoint]
