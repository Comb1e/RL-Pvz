"""Verify and fit a Transformer--LSTM policy from one complete demonstration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from pvz_game import Rules
from pvz_game.replay import decode_action, read_recording

from pvz_rl.config import digest, load_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy
from pvz_rl.presentation.recordings import open_playback, verify_replay
from pvz_rl.provenance import file_hash, write_json

CHECKPOINT_PROTOCOL = "pvz-rl/demo-initialization-checkpoint-v1"
RECURRENT_STORAGE_PROTOCOL = "pvz-rl/lstm-state-v1"


def _branch(action: torch.Tensor) -> torch.Tensor:
    return torch.where(
        action == 0,
        torch.zeros_like(action),
        torch.where(action < A.dig_start, (action - 1) // A.tiles + 1, torch.full_like(action, 9)),
    )


def _verify_action_order(archive_path: Path, replay_path: Path, cfg: dict) -> dict:
    archive_rows = [
        json.loads(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    transitions = [row for row in archive_rows if row.get("record_type") == "transition"]
    open_playback(replay_path)
    data = read_recording(replay_path)
    game = ActionPhaseGame(Rules(data["initial"]["rules"]))
    game.restore(data["initial"])
    encoder = ObservationEncoder(cfg, game.rules)
    codec = ActionCodec(cfg)
    observations = []
    native_actions = []
    native_outcomes = []
    for entry in data["entries"]:
        before_tick = game.observe().tick
        observations.append(encoder.encode(game.observe()).tolist())
        native_actions.append(codec.encode(decode_action(entry["action"])))
        action = decode_action(entry["action"])
        if entry["ticks"] == 0:
            result = game.step(action, ticks=0)
        else:
            result = game.step(action, ticks=entry["ticks"])
        native_outcomes.append((before_tick, result.action_result.accepted, result.action_result.reason, result.ticks_advanced))
        if "state_hash" in entry and entry["state_hash"] != game.state_hash():
            raise ValueError("native replay state hash mismatch")
    if len(native_actions) != len(transitions):
        raise ValueError("archive and replay decision counts differ")
    for row, action, obs, outcome in zip(transitions, native_actions, observations, native_outcomes, strict=True):
        if row["action"] != action:
            raise ValueError(f"action order mismatch at decision {row['decision_index']}")
        if row["observation"] != obs:
            # JSON float round-tripping is exact for the float32 values emitted
            # by the archive; use a tight tolerance for older writers.
            import numpy as np

            if not np.allclose(row["observation"], obs, atol=1e-6, rtol=0):
                raise ValueError(f"observation reconstruction mismatch at decision {row['decision_index']}")
        if (
            row["tick"], row["accepted"], row.get("rejection_reason"), row["ticks_advanced"]
        ) != outcome:
            raise ValueError(f"action outcome mismatch at decision {row['decision_index']}")
    final_rows = [row for row in archive_rows if row.get("record_type") == "final_observation"]
    if final_rows:
        final = encoder.encode(game.observe()).tolist()
        import numpy as np

        if not np.allclose(final_rows[-1]["observation"], final, atol=1e-6, rtol=0):
            raise ValueError("final observation reconstruction mismatch")
    return {
        "replay_sha256": file_hash(replay_path),
        "decisions": len(transitions),
        "state_hash": game.state_hash(),
        "outcome": game.observe().status.value,
        "observations_reconstructed": True,
    }


def verify_demo(archive: str | Path, replay: str | Path, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    archive, replay = Path(archive), Path(replay)
    manifest_path = archive.with_suffix(archive.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("protocol") != "pvz-rl/transition-archive-v1" or manifest.get("config_digest") != digest(cfg):
        raise ValueError("archive protocol or configuration digest does not match")
    rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines() if line]
    transitions = [row for row in rows if row.get("record_type") == "transition"]
    finals = [row for row in rows if row.get("record_type") == "final_observation"]
    if any(row.get("decision_index") != i for i, row in enumerate(transitions)):
        raise ValueError("decision indices are not monotonic")
    if any(
        len(row.get("observation", [])) != ObservationEncoder(cfg, Rules()).size
        for row in transitions + finals
    ):
        raise ValueError("archive observation size mismatch")
    archive_check = {"decisions": len(transitions), "complete": bool(manifest.get("complete") and finals), "outcome": manifest.get("outcome")}
    replay_check = verify_replay(replay)
    order = _verify_action_order(archive, replay, cfg)
    if archive_check["outcome"] not in ("won", "lost") or order["outcome"] not in ("won", "lost"):
        raise ValueError("initialization requires a completed won or lost demonstration")
    return {
        "protocol": CHECKPOINT_PROTOCOL,
        "archive": archive_check,
        "replay": replay_check.state_hash(),
        "reconstruction": order,
        "verified": archive_check["complete"] and order["observations_reconstructed"],
    }


def _training_tensors(archive: Path, cfg: dict):
    rows = [
        json.loads(line)
        for line in archive.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [row for row in rows if row.get("record_type") == "transition"]
    if not rows:
        raise ValueError("demonstration archive contains no transitions")
    observations = torch.tensor([row["observation"] for row in rows], dtype=torch.float32)
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.long)
    returns = torch.zeros(len(rows), dtype=torch.float32)
    running = 0.0
    for i in range(len(rows) - 1, -1, -1):
        running += float(rows[i]["reward_parts"].get("total", 0.0))
        returns[i] = running
    previous = torch.zeros_like(actions)
    outcomes = torch.zeros(len(rows), 2, dtype=torch.float32)
    if len(rows) > 1:
        previous[1:] = actions[:-1]
        outcomes[1:, 0] = torch.tensor([float(row["accepted"]) for row in rows[:-1]])
        outcomes[1:, 1] = torch.tensor([float(row["ticks_advanced"]) for row in rows[:-1]])
    return observations, actions, previous, outcomes, returns


def initialize_demo(
    archive: str | Path,
    replay: str | Path,
    output: str | Path,
    *,
    cfg: dict | None = None,
    passes: int | None = None,
    learning_rate: float | None = None,
    gradient_clip: float | None = None,
    seed: int | None = None,
    time_budget_minutes: float | None = None,
    device: str = "cpu",
) -> dict:
    """Fit one complete game and publish a checkpoint only after whole passes."""
    cfg = cfg or load_config()
    verification = verify_demo(archive, replay, cfg)
    if not verification["verified"]:
        raise ValueError("demonstration verification did not complete")
    settings = cfg["training"].get("demo", {})
    passes = int(settings.get("passes", 20) if passes is None else passes)
    learning_rate = float(settings.get("learning_rate", 3e-4) if learning_rate is None else learning_rate)
    gradient_clip = float(settings.get("gradient_clip", 0.5) if gradient_clip is None else gradient_clip)
    seed = int(settings.get("learner_seed", 101) if seed is None else seed)
    time_budget_minutes = float(settings.get("time_budget_minutes", 30) if time_budget_minutes is None else time_budget_minutes)
    if passes < 1 or learning_rate <= 0 or gradient_clip <= 0 or time_budget_minutes <= 0:
        raise ValueError("demo initialization settings must be positive")
    torch.manual_seed(seed)
    model = TransformerLSTMPolicy(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    observations, actions, previous, outcomes, returns = _training_tensors(Path(archive), cfg)
    observations, actions = observations.to(device), actions.to(device)
    previous, outcomes, returns = previous.to(device), outcomes.to(device), returns.to(device)
    chunk = int(cfg["policy"].get("chunk_length", 256))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    curves, coverage = [], torch.zeros(10, dtype=torch.long)
    all_branch = _branch(actions)
    group_masks = (all_branch == 0, (all_branch > 0) & (all_branch < 9), all_branch == 9)
    populated = sum(bool(mask.any()) for mask in group_masks)
    group_weights = [
        len(actions) / (populated * int(mask.sum())) if mask.any() else 0.0
        for mask in group_masks
    ]
    started = time.monotonic()
    completed = 0
    for pass_index in range(passes):
        if time.monotonic() - started >= time_budget_minutes * 60:
            break
        optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(1, device=device)
        pass_loss = 0.0
        for start in range(0, len(observations), chunk):
            stop = min(start + chunk, len(observations))
            branch_q, tile_features, contexts, state = model.forward_sequence(
                observations[start:stop][None],
                state,
                previous_actions=previous[start:stop][None],
                execution_outcomes=outcomes[start:stop][None],
                return_context=True,
            )
            branch_q, tile_features, contexts = branch_q[0], tile_features[0], contexts[0]
            branch = _branch(actions[start:stop])
            first = branch_q.gather(1, branch[:, None]).squeeze(1)
            target = returns[start:stop]
            first_error = (first - target).square()
            # Equal aggregate weight for wait, planting and digging groups.
            groups = (branch == 0, (branch > 0) & (branch < 9), branch == 9)
            loss_terms = [
                first_error[group].mean() * weight
                for group, weight in zip(groups, group_weights, strict=True)
                if group.any()
            ]
            nonwait = branch != 0
            if nonwait.any():
                tile_q = model.tile_values(tile_features[nonwait], contexts[nonwait], branch[nonwait])
                selected_actions = actions[start:stop][nonwait]
                tile = torch.where(
                    selected_actions < A.dig_start,
                    (selected_actions - 1) % A.tiles,
                    selected_actions - A.dig_start,
                )
                tile_error = (tile_q.gather(1, tile[:, None]).squeeze(1) - target[nonwait]).square()
                tile_weight = (group_weights[1] + group_weights[2]) / 2
                loss_terms.append(tile_error.mean() * tile_weight)
            loss = torch.stack(loss_terms).sum() / len(actions)
            loss.backward()
            pass_loss += float(loss.detach())
            state = state.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        completed += 1
        coverage += torch.bincount(_branch(actions).detach().cpu(), minlength=10)
        curves.append({"pass": completed, "loss": pass_loss, "decisions": len(actions)})
        torch.save(
            {
                "protocol": CHECKPOINT_PROTOCOL,
                "recurrent_storage_protocol": RECURRENT_STORAGE_PROTOCOL,
                "config_digest": digest(cfg),
                "config": cfg,
                "model": model.state_dict(),
                "passes_completed": completed,
                "learner_seed": seed,
                "source_replay_sha256": verification["reconstruction"]["replay_sha256"],
            },
            output / "initialization.pt",
        )
    write_json(output / "learning-curves.json", {"fit_scope": "one completed demonstration", "curves": curves})
    write_json(output / "action-coverage.json", {"branches": coverage.tolist()})
    write_json(output / "replay-verification.json", verification)
    write_json(output / "status.json", {"state": "complete" if completed == passes else "incomplete", "passes_completed": completed})
    return {
        "state": "complete" if completed == passes else "incomplete",
        "passes_completed": completed,
        "checkpoint": str((output / "initialization.pt").resolve()) if completed else None,
        "verification": verification,
    }


def load_demo_checkpoint(path: str | Path, cfg: dict | None = None, *, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg or saved["config"]
    if saved.get("protocol") != CHECKPOINT_PROTOCOL:
        raise ValueError("unsupported demonstration checkpoint protocol")
    if saved.get("config_digest") != digest(cfg):
        raise ValueError("checkpoint configuration does not match current observation/policy protocol")
    model = TransformerLSTMPolicy(cfg).to(device)
    model.load_state_dict(saved["model"])
    return model, saved
