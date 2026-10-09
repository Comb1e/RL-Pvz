"""Verify and fit a Transformer--LSTM policy from one complete demonstration."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pvz_game import Rules
from pvz_game.replay import decode_action, read_recording

from pvz_rl.config import digest, load_demo_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec, ActionSchema
from pvz_rl.envs.encoding import (
    ObservationEncoder,
    collate_observations,
    observation_json,
    observations_equal,
)
from pvz_rl.envs.history import public_history_event
from pvz_rl.envs.rewards import (
    HOME_FACT_FIELDS,
    REWARD_FACT_FIELDS,
    HomeProximityLedger,
    reward_parts,
)
from pvz_rl.learning.checkpoints import DEMO_PROTOCOL as CHECKPOINT_PROTOCOL
from pvz_rl.learning.checkpoints import STATE_PROTOCOL as RECURRENT_STORAGE_PROTOCOL
from pvz_rl.learning.objective import (
    components,
    demonstration_rank_loss,
    ranking_counts,
    training_rewards,
)
from pvz_rl.learning.performance import without_performance
from pvz_rl.policy.sequential_q import action_groups, action_parts, balanced_q_loss
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy
from pvz_rl.presentation.recordings import open_playback, verify_replay
from pvz_rl.provenance import file_hash, write_json


@dataclass
class VerifiedDemo:
    report: dict
    transitions: list[dict]


def _verify_reward_facts(row: dict, reconstructed: dict) -> None:
    """Validate historical facts/accounting, never historical reward coefficients."""
    saved = row.get("reward_parts")
    error = f"reward ledger mismatch at decision {row['decision_index']}"
    if (
        not isinstance(saved, dict)
        or set(saved)
        not in tuple(
            set(reconstructed) - omitted
            for omitted in (
                set(),
                {"early_sun", "early_sun_bonus"},
                {*HOME_FACT_FIELDS, "home_proximity", "victory_time"},
                {
                    *HOME_FACT_FIELDS,
                    "home_proximity",
                    "victory_time",
                    "early_sun",
                    "early_sun_bonus",
                },
            )
        )
        or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in saved.values()
        )
        or any(saved[key] != reconstructed[key] for key in REWARD_FACT_FIELDS)
        or any(saved[key] != reconstructed[key] for key in HOME_FACT_FIELDS if key in saved)
    ):
        raise ValueError(error)
    # Even old v2 archives without their reward configuration can verify these
    # identities. Historical prices are diagnostics; only replay rewards train.
    total = (
        saved["terminal"]
        + saved["development"]
        + saved["invalid_plant_penalty"]
        + saved["empty_dig_penalty"]
        + saved.get("home_proximity", 0)
        + saved.get("victory_time", 0)
    )
    net = (
        saved["produced_sun"]
        - saved["plant_value_loss"]
        + saved["combat_value"]
        - saved["mower_expenditure"]
    )
    if not math.isclose(saved["total"], total, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(
        saved["net_value"], net, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(error)
    invalid_plant = not row["accepted"] and 0 < row["action"] < ActionSchema.dig_start
    empty_dig = (
        not row["accepted"]
        and row["action"] >= ActionSchema.dig_start
        and row["rejection_reason"] == "empty_tile"
    )
    for key, applicable in (
        ("invalid_plant_penalty", invalid_plant),
        ("empty_dig_penalty", empty_dig),
    ):
        if saved[key] > 0 or (not applicable and saved[key] != 0):
            raise ValueError(error)
    terminal = saved["terminal"]
    if (
        (row["terminal"] == "running" and terminal != 0)
        or (row["terminal"] == "won" and terminal < 0)
        or (row["terminal"] == "lost" and terminal > 0)
    ):
        raise ValueError(error)


def _verify_action_order(archive_rows: list[dict], replay_path: Path, cfg: dict) -> dict:
    transitions = [row for row in archive_rows if row.get("record_type") == "transition"]
    open_playback(replay_path)
    data = read_recording(replay_path)
    game = ActionPhaseGame(Rules(data["initial"]["rules"]))
    game.restore(data["initial"])
    encoder = ObservationEncoder(cfg, game.rules)
    codec = ActionCodec(cfg)
    proximity = HomeProximityLedger(game.observe(), game.rules)
    native_masks = []
    observations = []
    native_truncation = []
    native_actions = []
    native_outcomes = []
    native_rewards = []
    native_terminals = []
    history_inputs = []
    pending = np.zeros(7, dtype=np.int64)
    elapsed = 0
    for entry in data["entries"]:
        before = game.observe()
        history_inputs.append(np.append(pending, elapsed))
        if pending.any():
            elapsed = 0
        before_tick = before.tick
        native_masks.append(codec.mask(game))
        observations.append(observation_json(encoder.encode(before)))
        native_truncation.append(dict(encoder.last_truncation))
        native_actions.append(codec.encode(decode_action(entry["action"])))
        action = decode_action(entry["action"])
        result = game.step(action, ticks=entry["ticks"])
        pending = public_history_event(result.events, game.rules).array()
        elapsed += result.ticks_advanced
        native_rewards.append(
            reward_parts(
                before,
                result.observation,
                cfg,
                events=result.events,
                rules=game.rules,
                action=action,
                action_result=result.action_result,
                proximity=proximity,
            )
        )
        native_terminals.append(result.status.value)
        native_outcomes.append(
            (
                before_tick,
                result.action_result.accepted,
                result.action_result.reason,
                result.ticks_advanced,
            )
        )
        if "state_hash" in entry and entry["state_hash"] != game.state_hash():
            raise ValueError("native replay state hash mismatch")
    if len(native_actions) != len(transitions):
        raise ValueError("archive and replay decision counts differ")
    reward_changes = 0
    recorded_total = recomputed_total = 0.0
    for row, action, obs, outcome, reward, terminal, omitted, legal, history in zip(
        transitions,
        native_actions,
        observations,
        native_outcomes,
        native_rewards,
        native_terminals,
        native_truncation,
        native_masks,
        history_inputs,
        strict=True,
    ):
        if row["action"] != action:
            raise ValueError(f"action order mismatch at decision {row['decision_index']}")
        if not observations_equal(row["observation"], obs):
            raise ValueError(
                f"observation reconstruction mismatch at decision {row['decision_index']}"
            )
        if row.get("executed_action") != (action if outcome[1] else 0):
            raise ValueError(f"execution action mismatch at decision {row['decision_index']}")
        if row.get("entity_truncation") != omitted:
            raise ValueError(f"entity truncation mismatch at decision {row['decision_index']}")
        if (
            row["tick"],
            row["accepted"],
            row.get("rejection_reason"),
            row["ticks_advanced"],
        ) != outcome:
            raise ValueError(f"action outcome mismatch at decision {row['decision_index']}")
        if row["terminal"] != terminal:
            raise ValueError(f"terminal mismatch at decision {row['decision_index']}")
        _verify_reward_facts(row, reward)
        reward_changes += row["reward_parts"] != reward
        recorded_total += row["reward_parts"]["total"]
        recomputed_total += reward["total"]
        # Replace only the in-memory fitting data, never the user's recording.
        row["reward_parts"] = reward
        row["training_reward"] = float(training_rewards(components(reward), cfg))
        row["legal_mask"] = legal
        row["events"] = history
        row["memory_write"] = bool(history[:7].any())
    final_rows = [row for row in archive_rows if row.get("record_type") == "final_observation"]
    if final_rows:
        final = observation_json(encoder.encode(game.observe()))
        if not observations_equal(final_rows[-1]["observation"], final):
            raise ValueError("final observation reconstruction mismatch")
    return {
        "replay_sha256": file_hash(replay_path),
        "decisions": len(transitions),
        "state_hash": game.state_hash(),
        "outcome": game.observe().status.value,
        "observations_reconstructed": True,
        "rewards_recomputed": True,
        "reward_changes": reward_changes,
        "recorded_reward_total": recorded_total,
        "recomputed_reward_total": recomputed_total,
        "reward_settings": dict(cfg["reward"]),
    }


def verify_demo(archive: str | Path, replay: str | Path, cfg: dict | None = None) -> dict:
    return _load_verified_demo(archive, replay, cfg).report


def _load_verified_demo(
    archive: str | Path, replay: str | Path, cfg: dict | None = None
) -> VerifiedDemo:
    cfg = cfg or load_demo_config()
    archive, replay = Path(archive), Path(replay)
    manifest_path = archive.with_suffix(archive.suffix + ".manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    if not manifest_path.exists():
        raise ValueError("Demonstration archive manifest is missing")
    if manifest.get("protocol") != "pvz-rl/transition-archive-v2":
        raise ValueError(
            "Unsupported demonstration archive protocol; record a fresh entity_v1 demonstration"
        )
    # Learning settings are not recording inputs. Validate the pinned engine,
    # schema and native replay facts instead of the complete configuration hash.
    if manifest.get("engine") != {
        "commit": cfg["engine_commit"],
        "version": cfg["engine_version"],
        "package_version": cfg["engine_package_version"],
    }:
        raise ValueError("Archive engine does not match the configured pinned engine")
    if manifest.get("observation_schema") != ObservationEncoder(cfg, Rules()).schema():
        raise ValueError("Archive entity schema does not match; record a fresh demonstration")
    if manifest.get("replay") != file_hash(replay):
        raise ValueError("archive replay hash does not match")
    rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines() if line]
    transitions = [row for row in rows if row.get("record_type") == "transition"]
    finals = [row for row in rows if row.get("record_type") == "final_observation"]
    if (
        not manifest.get("complete")
        or len(finals) != 1
        or not transitions
        or rows[-1] is not finals[0]
        or len(transitions) != manifest.get("decisions")
        or finals[0].get("decision_index") != len(transitions)
        or any(row.get("episode_id") != manifest.get("episode_id") for row in rows)
    ):
        raise ValueError("initialization requires one completed demonstration archive")
    if any(row.get("decision_index") != i for i, row in enumerate(transitions)):
        raise ValueError("decision indices are not monotonic")
    for row in transitions + finals:
        collate_observations(row["observation"])
    archive_check = {
        "decisions": len(transitions),
        "complete": bool(manifest.get("complete") and finals),
        "outcome": manifest.get("outcome"),
    }
    replay_check = verify_replay(replay)
    order = _verify_action_order(rows, replay, cfg)
    if archive_check["outcome"] not in ("won", "lost") or order["outcome"] not in ("won", "lost"):
        raise ValueError("initialization requires a completed won or lost demonstration")
    if (
        archive_check["outcome"] != order["outcome"]
        or finals[0].get("terminal") != order["outcome"]
        or finals[0].get("tick") != replay_check.observe().tick
    ):
        raise ValueError("archive and replay completion do not match")
    report = {
        "protocol": CHECKPOINT_PROTOCOL,
        "archive": archive_check,
        "replay": replay_check.state_hash(),
        "reconstruction": order,
        "verified": archive_check["complete"] and order["observations_reconstructed"],
    }
    return VerifiedDemo(report, transitions)


def _training_tensors(demonstration: VerifiedDemo):
    rows = demonstration.transitions
    if not rows:
        raise ValueError("demonstration archive contains no transitions")
    observations = [row["observation"] for row in rows]
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.long)
    returns = torch.zeros(len(rows), dtype=torch.float32)
    running = 0.0
    for i in range(len(rows) - 1, -1, -1):
        running += float(rows[i]["training_reward"])
        returns[i] = running
    events = torch.as_tensor(np.stack([row["events"] for row in rows]), dtype=torch.float32)
    writes = torch.tensor([row["memory_write"] for row in rows], dtype=torch.bool)
    return observations, actions, events, writes, returns


def initialize_demo(
    archive: str | Path,
    replay: str | Path,
    output: str | Path,
    *,
    cfg: dict | None = None,
    passes: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    time_budget_minutes: float | None = None,
    device: str = "cpu",
) -> dict:
    """Fit one complete game and publish a checkpoint only after whole passes."""
    cfg = cfg or load_demo_config()
    demonstration = _load_verified_demo(archive, replay, cfg)
    verification = demonstration.report
    if not verification["verified"]:
        raise ValueError("demonstration verification did not complete")
    settings = cfg["training"].get("demo", {})
    passes = int(settings.get("passes", 20) if passes is None else passes)
    learning_rate = float(
        settings.get("learning_rate", 3e-4) if learning_rate is None else learning_rate
    )
    max_grad_norm = float(cfg["training"]["max_grad_norm"])
    seed = int(settings.get("learner_seed", 101) if seed is None else seed)
    time_budget_minutes = float(
        settings.get("time_budget_minutes", 30)
        if time_budget_minutes is None
        else time_budget_minutes
    )
    if passes < 1 or any(
        not math.isfinite(value) or value <= 0
        for value in (learning_rate, max_grad_norm, time_budget_minutes)
    ):
        raise ValueError("demo initialization settings must be positive")
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.manual_seed(seed)
    model = TransformerLSTMPolicy(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    observations, actions, events, writes, returns = _training_tensors(demonstration)
    actions = actions.to(device)
    events, writes, returns = events.to(device), writes.to(device), returns.to(device)
    write_plan = np.asarray([row["memory_write"] for row in demonstration.transitions], dtype=bool)
    chunk = int(cfg["policy"].get("chunk_length", 256))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    curves, coverage = [], torch.zeros(10, dtype=torch.long)
    # Initialization transfers weights only.  Keep structural and learning
    # metadata for validation, but leave execution settings to the current
    # training profile when this artifact is consumed.
    checkpoint_cfg = without_performance(cfg)
    checkpoint_cfg["training"].get("demo", {}).pop("gradient_clip", None)
    accepted = torch.tensor([r["accepted"] for r in demonstration.transitions], device=device)
    legal = torch.as_tensor(
        np.asarray([r["legal_mask"] for r in demonstration.transitions], dtype=np.bool_),
        device=device,
    )
    group_counts = torch.bincount(
        action_groups(actions) * 2 + accepted.long(), minlength=6
    ).reshape(3, 2)
    rank_counts = ranking_counts(actions, accepted, legal)[0]
    objective = cfg["training"]["objective"]
    started = time.monotonic()
    completed = 0
    for pass_index in range(passes):
        if time.monotonic() - started >= time_budget_minutes * 60:
            break
        optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(1, device=device)
        pass_loss = 0.0
        rank_accuracy = torch.zeros(4, device=device)
        ranking_total = 0.0
        for start in range(0, len(observations), chunk):
            stop = min(start + chunk, len(observations))
            branch_q, tile_features, contexts, state = model.forward_sequence(
                collate_observations(observations[start:stop], device).reshape(1, stop - start),
                state,
                events=events[start:stop][None],
                memory_write=writes[start:stop][None],
                write_plan=write_plan[None, start:stop],
                return_context=True,
            )
            branch_q, tile_features, contexts = branch_q[0], tile_features[0], contexts[0]
            branch = action_parts(actions[start:stop])[0]
            first = branch_q.gather(1, branch[:, None]).squeeze(1)
            target = returns[start:stop]
            second = torch.zeros_like(first)
            nonwait = branch != 0
            if nonwait.any():
                tile_q = model.tile_values(
                    tile_features[nonwait], contexts[nonwait], branch[nonwait]
                )
                selected_actions = actions[start:stop][nonwait]
                tile = action_parts(selected_actions)[1]
                second[nonwait] = tile_q.gather(1, tile[:, None]).squeeze(1)
            loss, _, _ = balanced_q_loss(
                first,
                second,
                target,
                actions[start:stop],
                group_counts,
                accepted=accepted[start:stop],
                accepted_share=cfg["training"]["objective"]["accepted_outcome_share"],
            )
            complete_loss = loss
            rank, accuracy = demonstration_rank_loss(
                model,
                branch_q,
                tile_features,
                contexts,
                actions[start:stop],
                accepted[start:stop],
                legal[start:stop],
                rank_counts,
                objective["demo_rank_margin"],
            )
            loss = loss + objective["demo_rank_loss_weight"] * rank
            rank_accuracy += accuracy.detach()
            ranking_total += float(rank.detach())
            loss.backward()
            pass_loss += float(complete_loss.detach())
            state = state.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        completed += 1
        coverage += torch.bincount(action_parts(actions)[0].detach().cpu(), minlength=10)
        scores = rank_accuracy.cpu().tolist()
        curves.append(
            {
                "pass": completed,
                "loss": pass_loss,
                "ranking_loss": ranking_total,
                "branch_ranking_accuracy": scores[0] / scores[1] if scores[1] else None,
                "tile_ranking_accuracy": scores[2] / scores[3] if scores[3] else None,
                "decisions": len(actions),
            }
        )
        _save_checkpoint(
            {
                "protocol": CHECKPOINT_PROTOCOL,
                "recurrent_storage_protocol": RECURRENT_STORAGE_PROTOCOL,
                "observation_schema": model.layout.schema(),
                "config_digest": digest(checkpoint_cfg),
                "config": checkpoint_cfg,
                "model": model.state_dict(),
                "objective": dict(objective),
                "passes_completed": completed,
                "learner_seed": seed,
                "source_replay_sha256": verification["reconstruction"]["replay_sha256"],
            },
            output / "initialization.pt",
        )
    write_json(
        output / "learning-curves.json",
        {"fit_scope": "one completed demonstration", "curves": curves},
    )
    write_json(output / "action-coverage.json", {"branches": coverage.tolist()})
    write_json(output / "replay-verification.json", verification)
    write_json(
        output / "status.json",
        {
            "state": "complete" if completed == passes else "incomplete",
            "passes_completed": completed,
        },
    )
    return {
        "state": "complete" if completed == passes else "incomplete",
        "passes_completed": completed,
        "checkpoint": str((output / "initialization.pt").resolve()) if completed else None,
        "verification": verification,
    }


def _save_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_demo_checkpoint(path: str | Path, cfg: dict | None = None, *, device="cpu"):
    from pvz_rl.learning.checkpoints import compatible_config, inspect_checkpoint

    info = inspect_checkpoint(path)
    if info["initialization_type"] != "demonstration":
        raise ValueError("Expected a demonstration initialization checkpoint")
    saved = info["payload"]
    if cfg is not None:
        compatible_config(saved["config"], cfg, weights_only=True)
    model = TransformerLSTMPolicy(cfg or info["config"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved
