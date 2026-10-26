"""Verify and fit a stateless two-stage policy from one complete demonstration."""

from __future__ import annotations

import json
import math
import time
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pvz_game import Rules
from pvz_game.cuda.schema import REASONS
from pvz_game.replay import decode_action, read_recording

from pvz_rl.config import digest, load_demo_config
from pvz_rl.envs.action_credit import CausalCreditLedger
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec, ActionSchema
from pvz_rl.envs.encoding import (
    ObservationEncoder,
    collate_observations,
    observation_json,
    observations_equal,
)
from pvz_rl.envs.rewards import (
    HOME_FACT_FIELDS,
    REWARD_FACT_FIELDS,
    HomeProximityLedger,
    reward_parts,
)
from pvz_rl.learning.action_routing import (
    branch_availability,
    classify_actions,
    compact_invalid_rows,
)
from pvz_rl.learning.batch_transport import PreparedDataSource
from pvz_rl.learning.checkpoints import DEMO_PROTOCOL as CHECKPOINT_PROTOCOL
from pvz_rl.learning.checkpoints import MODEL_STATE_PROTOCOL
from pvz_rl.learning.cuda_buffer import trajectory_dtype
from pvz_rl.learning.flat_q import CohortCoefficients, batch_loss
from pvz_rl.learning.objective import (
    COMPONENTS,
    components,
    consistency_diagnostics,
    demonstration_rank_loss,
    ranking_counts,
    raw_rewards,
    training_rewards,
)
from pvz_rl.learning.performance import without_performance
from pvz_rl.policy.sequential_q import action_parts
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.presentation.recordings import open_playback, verify_replay
from pvz_rl.provenance import file_hash, write_json


@dataclass
class VerifiedDemo:
    report: dict
    transitions: list[dict]


def _verify_reward_facts(row: dict, reconstructed: dict, cfg=None) -> None:
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
                {
                    "global_terminal",
                    "global_asset_value",
                    "action_damage",
                    "action_blocking",
                    "action_sun",
                    "action_mower",
                },
                {
                    *HOME_FACT_FIELDS,
                    "home_proximity",
                    "victory_time",
                    "early_sun",
                    "early_sun_bonus",
                    "global_terminal",
                    "global_asset_value",
                    "action_damage",
                    "action_blocking",
                    "action_sun",
                    "action_mower",
                },
            )
        )
        or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in saved.values()
        )
        or any(saved[key] != reconstructed[key] for key in REWARD_FACT_FIELDS)
        or any(saved[key] != reconstructed[key] for key in HOME_FACT_FIELDS if key in saved)
        or (
            "global_asset_value" in saved
            and not math.isclose(
                float(saved["global_asset_value"]),
                float(reconstructed.get("global_asset_value", 0.0)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    ):
        raise ValueError(error)
    # Even old v2 archives without their reward configuration can verify these
    # identities. Historical prices are diagnostics; only replay rewards train.
    has_current_components = all(key in saved for key in COMPONENTS)
    if not has_current_components:
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
    # New archives may have been recorded with a different reward coefficient
    # profile. Their fact fields remain verifiable, but their historical total
    # is deliberately not compared with the current profile; replay below
    # recomputes the current six-component reward before fitting.
    total_matches = has_current_components or math.isclose(
        saved["total"], total, rel_tol=1e-12, abs_tol=1e-12
    )
    if has_current_components:
        # Current archives retain enough scalar facts to recover the scale
        # used when their immediate reward was written.  This catches a
        # tampered ``total`` even when the caller deliberately changes the
        # current reward coefficients to reprice a verified demonstration.
        net_value = float(saved.get("net_value", 0.0))
        development = float(saved.get("development", 0.0))
        early_bonus = float(saved.get("early_sun_bonus", 0.0))
        direct_mower = float(saved["action_mower"]) if saved["mower_expenditure"] == 0 else 0.0
        physical = sum(float(saved.get(key, 0.0)) for key in COMPONENTS[1:]) - direct_mower
        if abs(net_value) > 1e-15:
            scale = (development - early_bonus) / net_value
            # Historical mowers used physical expenditure; current mowers are
            # final-unit penalties with zero physical expenditure. Infer this
            # from archived accounting, never from today's reward coefficients.
            expected = float(saved["terminal"]) + direct_mower + scale * physical
            total_matches = total_matches and math.isclose(
                float(saved["total"]), expected, rel_tol=1e-9, abs_tol=1e-9
            )
        elif abs(development) <= 1e-15 and abs(early_bonus) <= 1e-15:
            # A completely idle transition has no scale-dependent term; its
            # total contains only the terminal and any direct mower penalty.
            if abs(physical) <= 1e-15:
                total_matches = total_matches and math.isclose(
                    float(saved["total"]),
                    float(saved["terminal"]) + direct_mower,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
    if not total_matches or not math.isclose(saved["net_value"], net, rel_tol=1e-12, abs_tol=1e-12):
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
    # Reconstruct the same causal source ledger used by the live CPU adapter.
    # Replay facts are authoritative, but provenance is deliberately kept out
    # of the observation and archive schema.  Delayed effects are applied to
    # their source transition only after the complete replay has been checked.
    causal = CausalCreditLedger(cfg, game.rules)
    causal.seed_observation(game.observe())
    source_credits = {}
    native_masks = []
    native_branch_availability = []
    observations = []
    native_truncation = []
    native_actions = []
    native_outcomes = []
    native_immediate_rewards = []
    native_rewards = []
    native_terminals = []
    for entry in data["entries"]:
        before = game.observe()
        before_tick = before.tick
        native_masks.append(codec.mask(game))
        cards = {card.plant_type: card for card in before.cards}
        native_branch_availability.append(
            branch_availability(
                [before.sun],
                [[cards[kind].cooldown_ticks if kind in cards else -1 for kind in codec.plants]],
                [game.rules.plants[kind]["cost"] for kind in codec.plants],
                card_present=[[kind in cards for kind in codec.plants]],
            )[0]
        )
        observations.append(observation_json(encoder.encode(before)))
        native_truncation.append(dict(encoder.last_truncation))
        native_actions.append(codec.encode(decode_action(entry["action"])))
        action_id = len(native_actions) - 1
        action = decode_action(entry["action"])
        result = game.step(action, ticks=entry["ticks"])
        credits = causal.attribute(result.events, action_id=action_id)
        causal.register_plant_kinds(result.observation)
        for source, values in credits.items():
            # The current transition is already supplied to reward_parts via
            # ``causal_credits`` below.  Only retain credits that must be
            # moved back to an earlier source row.
            if source is None or int(source) == action_id:
                continue
            bucket = source_credits.setdefault(
                int(source), {"damage": 0.0, "blocking": 0.0, "sun": 0.0, "mower": 0.0}
            )
            for field in bucket:
                bucket[field] += float(values.get(field, 0.0))
        native_immediate_rewards.append(
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
                causal_credits=credits.get(action_id),
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
    for row, action, obs, outcome, immediate, reward, terminal, omitted, legal, available in zip(
        transitions,
        native_actions,
        observations,
        native_outcomes,
        native_immediate_rewards,
        native_rewards,
        native_terminals,
        native_truncation,
        native_masks,
        native_branch_availability,
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
        # Validate the archived, immediate event facts before applying
        # retroactive source credits.  Older archives may have used a
        # different coefficient profile, but their public event accounting
        # must still agree with the replay.
        _verify_reward_facts(row, immediate, cfg)
        recorded_total += row["reward_parts"]["total"]
        # Replace only the in-memory fitting data, never the user's recording.
        # A source row can receive several delayed effects over later replay
        # entries; update all six learning components atomically and derive the
        # scalar reward from that final component vector.
        source = source_credits.get(int(row["decision_index"]))
        if source:
            for field, component in (
                ("damage", "action_damage"),
                ("blocking", "action_blocking"),
                ("sun", "action_sun"),
                ("mower", "action_mower"),
            ):
                reward[component] = float(reward.get(component, 0.0)) + float(source[field])
            reward["total"] = float(raw_rewards(components(reward), cfg))
        reward_changes += row["reward_parts"] != reward
        recomputed_total += reward["total"]
        # Replace only the in-memory fitting data, never the user's recording.
        row["reward_parts"] = reward
        row["training_reward"] = float(training_rewards(components(reward), cfg))
        row["legal_mask"] = legal
        routes, usable = classify_actions(
            [action],
            [outcome[1]],
            legal[None],
            [outcome[2]],
            branch_available=np.asarray(available, dtype=bool)[None],
        )
        row["route"], row["branch_usable"] = int(routes[0]), bool(usable[0])
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
        "causal_source_credited_rows": len(source_credits),
        "causal_source_credits": source_credits,
        "causal_unattributable_events": int(
            causal.last_diagnostics.get("unattributable_events", 0)
        ),
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
    return observations, actions, returns


def _initialize_demo(
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
    init_from: str | Path | None = None,
    _resources=None,
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
    model = TwoStageTransformerPolicy(cfg).to(device)
    initialization = None
    if init_from is not None:
        from pvz_rl.learning.training import initial_weights

        weights, initialization = initial_weights(init_from, cfg)
        model.load_state_dict(weights, strict=True)
    with torch.random.fork_rng(
        devices=[torch.device(device)] if str(device).startswith("cuda") else []
    ):
        teacher = TwoStageTransformerPolicy(cfg).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(model.runtime_state_dict())
    optimizer = torch.optim.Adam(model.master_parameters(), lr=learning_rate)
    observations, actions, returns = _training_tensors(demonstration)
    actions = actions.to(device)
    returns = returns.to(device)
    batch_size = int(cfg["training"]["batch_size"]) * int(
        cfg["training"]["performance"]["fit_batch_groups"]
    )
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
    objective = cfg["training"]["objective"]
    raw_decisions = len(actions)
    rows = np.zeros(
        len(actions),
        dtype=trajectory_dtype(cfg),
    )
    rows["active"] = True
    rows["action"] = actions.cpu().numpy()
    rows["accepted"] = accepted.cpu().numpy()
    rows["target"] = returns.cpu().numpy()
    rows["route"] = [r["route"] for r in demonstration.transitions]
    rows["branch_usable"] = [r["branch_usable"] for r in demonstration.transitions]
    rows["env"] = 0
    rows["episode"] = [r.get("episode_index", 0) for r in demonstration.transitions]
    rows["decision"] = [r.get("decision_index", i) for i, r in enumerate(demonstration.transitions)]
    if len(rows):
        rows["reset"][0] = True
    rows["tick"] = [int(r.get("tick", index)) for index, r in enumerate(demonstration.transitions)]
    rows["done"] = [r.get("terminal") not in (None, "running") for r in demonstration.transitions]
    for index, transition in enumerate(demonstration.transitions):
        reason = transition.get("rejection_reason")
        if reason is None:
            reason = 0
        elif isinstance(reason, str):
            try:
                reason = REASONS.index(reason)
            except ValueError as exc:
                raise ValueError(f"unknown demonstration rejection reason: {reason}") from exc
        rows["rejection_reason"][index] = int(reason)
    compact_positions, compact_counts, compaction = compact_invalid_rows(
        rows,
        group_size=objective.get("invalid_fit_group_size", 100),
    )
    rows = rows[compact_positions]
    observations = [observations[index] for index in compact_positions]
    selection = torch.as_tensor(compact_positions, device=device)
    actions, accepted, legal = actions[selection], accepted[selection], legal[selection]
    group_counts = torch.as_tensor(compact_counts, device=device)
    rank_counts = ranking_counts(actions, accepted, legal)[0]
    # Returns were computed in episode order. Reorder only fitting work, with
    # all real rows and complete-cohort returns retained unchanged.
    widths = np.asarray([len(obs["entities"]) for obs in observations])
    order = np.lexsort(
        (
            rows["route"],
            np.searchsorted(cfg["training"]["performance"]["fit_entity_buckets"], widths),
        )
    )
    rows = rows[order]
    observations = [observations[i] for i in order]
    selection = torch.as_tensor(order, device=device)
    actions, accepted, legal = actions[selection], accepted[selection], legal[selection]
    keys = (
        np.searchsorted(cfg["training"]["performance"]["fit_entity_buckets"], widths[order]) * 4
        + rows["route"]
    )
    boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(rows)]
    prepared = PreparedDataSource(observations, cfg, rows=rows)
    prepared.boundaries = boundaries
    _resources.callback(prepared.close)
    consistency_counts = group_counts[1:] * group_counts.new_tensor([0, 1])
    coefficients = CohortCoefficients.prepare(
        compact_counts, consistency_counts.cpu(), device, objective["accepted_outcome_share"]
    )
    started = time.monotonic()
    completed = 0
    for pass_index in range(passes):
        if time.monotonic() - started >= time_budget_minutes * 60:
            break
        model.zero_grad(set_to_none=True)
        rank_accuracy = torch.zeros(4, device=device)
        pass_metrics = torch.zeros(10, device=device, dtype=torch.float64)
        batches = _resources.enter_context(closing(prepared.batches(batch_size, device)))
        start = 0
        for batch_rows, current, fields in batches:
            stop = start + len(batch_rows)
            loss, _, gap, forward = batch_loss(
                model,
                rows[start:stop],
                group_counts,
                current,
                fields,
                teacher=teacher,
                consistency_counts=consistency_counts,
                coefficients=coefficients,
                return_forward=True,
            )
            selected = torch.as_tensor(np.flatnonzero(rows["accepted"][start:stop]), device=device)
            if len(selected):
                rank, accuracy = demonstration_rank_loss(
                    model,
                    current[selected],
                    actions[start:stop][selected],
                    accepted[start:stop][selected],
                    legal[start:stop][selected],
                    rank_counts,
                    objective["demo_rank_margin"],
                    species=forward.species[selected],
                    tiles=forward.tiles[selected],
                )
            else:
                rank, accuracy = loss.detach() * 0, loss.new_zeros(4)
            loss = loss + objective["demo_rank_loss_weight"] * rank
            rank_accuracy += accuracy.detach()
            loss.backward()
            model.accumulate_master_gradients()
            model.clear_runtime_grad(set_to_none=True)
            del forward
            pass_metrics += torch.cat(
                (
                    torch.stack(
                        (
                            model._last_selected_loss.double(),
                            model._last_species_consistency_loss.double(),
                            rank.detach().double(),
                            loss.detach().double(),
                            gap.double().sum(),
                            gap.new_tensor(gap.numel(), dtype=torch.float64),
                        )
                    ),
                    model._last_species_consistency_metrics,
                )
            )
            start = stop
        torch.nn.utils.clip_grad_norm_(model.master_parameters(), max_grad_norm)
        optimizer.step()
        model.sync_runtime_from_masters()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(teacher.parameters()), list(model.parameters()), 1 - objective["ema_decay"]
            )
        completed += 1
        coverage += torch.bincount(action_parts(actions)[0].detach().cpu(), minlength=10)
        scores = rank_accuracy.cpu().tolist()
        pass_loss, consistency_total, ranking_total, total_loss, gap_sum, gap_n, *totals = (
            pass_metrics.cpu().tolist()
        )
        curves.append(
            {
                "pass": completed,
                "loss": pass_loss,
                "ranking_loss": ranking_total,
                **consistency_diagnostics(consistency_total, totals),
                "total_loss": total_loss,
                "placement_gap": gap_sum / gap_n if gap_n else None,
                "branch_ranking_accuracy": scores[0] / scores[1] if scores[1] else None,
                "tile_ranking_accuracy": scores[2] / scores[3] if scores[3] else None,
                "decisions": len(actions),
                "raw_decisions": raw_decisions,
            }
        )
        _save_checkpoint(
            {
                "protocol": CHECKPOINT_PROTOCOL,
                "model_storage_protocol": MODEL_STATE_PROTOCOL,
                "precision_contract": "bf16_master",
                "observation_schema": model.layout.schema(),
                "config_digest": digest(checkpoint_cfg),
                "config": checkpoint_cfg,
                "model": model.state_dict(),
                "objective": dict(objective),
                "passes_completed": completed,
                "learner_seed": seed,
                "initialization": initialization,
                "source_replay_sha256": verification["reconstruction"]["replay_sha256"],
            },
            output / "initialization.pt",
        )
    write_json(
        output / "learning-curves.json",
        {"fit_scope": "one completed demonstration", "curves": curves},
    )
    write_json(output / "fitting-compaction.json", compaction)
    write_json(output / "action-coverage.json", {"branches": coverage.tolist()})
    write_json(output / "replay-verification.json", verification)
    write_json(
        output / "status.json",
        {
            "state": "complete" if completed == passes else "incomplete",
            "passes_completed": completed,
            "initialization": initialization,
        },
    )
    return {
        "state": "complete" if completed == passes else "incomplete",
        "passes_completed": completed,
        "initialization": initialization,
        "checkpoint": str((output / "initialization.pt").resolve()) if completed else None,
        "verification": verification,
        "fitting_compaction": compaction,
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
    model = TwoStageTransformerPolicy(cfg or info["config"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved


def initialize_demo(archive, replay, output, **kwargs):
    with ExitStack() as resources:
        return _initialize_demo(archive, replay, output, _resources=resources, **kwargs)
