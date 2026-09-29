"""Complete tiny replay fitting, independent group loss and corrupt-input controls."""

import json
from collections import defaultdict

import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Wait

from pvz_rl.config import load_config, load_demo_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.learning import demo_initialization as demo
from pvz_rl.learning.checkpoints import inspect_checkpoint
from pvz_rl.presentation.demo_recording import TransitionArchive
from pvz_rl.presentation.recordings import ActionPhaseRecorder


@pytest.fixture
def completed_demo(tmp_path):
    cfg = load_demo_config()
    cfg["environment"]["cutoff_seconds"] = 1
    # A recording made before clipping was shared must remain reusable.
    cfg["training"]["demo"]["gradient_clip"] = 0.5
    cfg["policy"].update(
        entity_width=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_feedforward=16,
        scalar_width=8,
        lstm_hidden=8,
        action_embedding=4,
        outcome_width=4,
        chunk_length=2,
    )
    game = ActionPhaseGame()
    game.reset(LevelSpec("tiny-fit", initial_sun=200), seed=1000)
    recorder = ActionPhaseRecorder(game)
    replay, archive = tmp_path / "tiny.pvzdemo", tmp_path / "tiny.jsonl"
    writer = TransitionArchive(archive, cfg=cfg)
    try:
        for action in (
            Place("sunflower", 0, 0),
            Place("peashooter", 0, 1),
            Dig(0, 0),
            Dig(0, 1),
            Wait(),
        ):
            before = game.observe()
            result = recorder.step(action, ticks=int(isinstance(action, Wait)))
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


def test_complete_demo_fit_uses_whole_game_group_loss_and_reloads(
    completed_demo, tmp_path, monkeypatch
):
    cfg, archive, replay = completed_demo
    cfg["training"]["demo"].pop("gradient_clip")
    cfg["training"]["max_grad_norm"] = 1.25
    samples = defaultdict(list)
    shared_loss = demo.balanced_q_loss
    clip_calls = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def observe_clip(parameters, max_norm):
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm)

    def observe(first, second, targets, actions, counts, batch_size):
        for q1, q2, target, action in zip(
            first.detach().tolist(), second.detach().tolist(), targets.tolist(), actions.tolist()
        ):
            group = "wait" if action == 0 else "plant" if action < 361 else "dig"
            error = (q1 - target) ** 2
            if action:
                error = (error + (q2 - target) ** 2) / 2
            samples[group].append(error)
        return shared_loss(first, second, targets, actions, counts, batch_size)

    monkeypatch.setattr(demo, "balanced_q_loss", observe)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe_clip)
    result = demo.initialize_demo(archive, replay, tmp_path / "fit", cfg=cfg, passes=1)
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
    observations, *_ = demo._training_tensors(archive, cfg)
    assert torch.isfinite(model.forward_step(observations[:1]).branch_q).all()
    assert not list((tmp_path / "fit").glob("*.tmp"))
    saved["recurrent_storage_protocol"] = "unknown"
    torch.save(saved, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="recurrent storage"):
        demo.load_demo_checkpoint(tmp_path / "invalid.pt", cfg)


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
    "field", ["observation", "action", "executed_action", "reward_parts", "terminal", "tick"]
)
def test_transition_corruption_rejected_by_reconstruction(completed_demo, field):
    cfg, archive, replay = completed_demo
    rows = [json.loads(line) for line in archive.read_text().splitlines()]
    if field == "reward_parts":
        rows[0][field]["total"] += 1
    elif field == "observation":
        rows[0][field]["globals"][0] += 1
    elif field == "terminal":
        rows[0][field] = "lost"
    else:
        rows[0][field] += 1
    archive.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        demo.verify_demo(archive, replay, cfg)


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
