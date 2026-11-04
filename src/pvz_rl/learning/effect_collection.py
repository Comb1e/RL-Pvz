"""Act-time causal labels. Private fork choices never change main-game behavior."""

import numpy as np
import torch

from pvz_rl.config import absolute_outcomes, effect_extra_actions
from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.probe_intervals import ProbeState, interval_probes_enabled
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_labels import effect_labels
from pvz_rl.policy.sequential_q import action_parts


def sampling_rng(model, slot, stream):
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(model.seed or 0),
                int(slot),
                int(model.env._episode_serial[slot]),
                int(model._scheduler.decisions[slot]),
                stream,
            ]
        )
    )


def interval_alternatives(action, legal, branch, scores, rng):
    """Two fixed slots: branch runner-up, then a uniform remaining legal action."""
    candidates = np.flatnonzero(legal)
    candidates = candidates[candidates != 0]
    ranked = sorted(
        (int(a) for a in candidates if (a - 1) // 45 + 1 == branch),
        key=lambda a: (-float(scores[(a - 1) % 45]), a),
    )
    runner_up = ranked[1] if len(ranked) >= 2 and ranked[1] != action else None
    remaining = (
        candidates[(candidates != action) & (candidates != runner_up)]
        if runner_up is not None
        else candidates[candidates != action]
    )
    random = int(rng.choice(remaining)) if len(remaining) else None
    return runner_up, random


def begin_effects(model, actions, details, issued):
    env, policy = model.env, model.policy
    enabled = np.flatnonzero(issued)
    proposed = actions.detach().cpu().numpy()
    scratch_masks = env.cp.empty_like(env.batch.masks)
    env.batch._mask(
        ((env.batch.n * ActionSchema.size + 255) // 256,),
        (256,),
        (env.batch.header, env.batch.plants, env.batch.cooldowns, scratch_masks, env.batch.n),
    )
    legal = scratch_masks.get()
    absolute = absolute_outcomes(model.cfg)
    interval = interval_probes_enabled(model.cfg) or absolute
    maximum = effect_extra_actions(model.cfg, model.tile_skills.phase)
    effective_wait = (proposed == 0) | ~legal[np.arange(env.num_envs), proposed]
    sampled = issued.copy() if maximum else np.zeros_like(issued)
    if interval and maximum:
        probability = model.cfg["training"]["forecasts"]["wait_effect_sample_probability"]
        for slot in enabled[effective_wait[enabled]]:
            sampled[slot] = sampling_rng(model, slot, 0xEFF0).random() < probability
    if sampled.any():
        tables = details["tile_q"].float().clone()
        branches, _ = action_parts(actions)
        branch_legal = torch.as_tensor(
            legal[:, 1:].reshape(-1, 9, 45).any(-1), device=policy.device
        )
        # Wait/unavailable decisions need a proposal branch for near-Q alternatives.
        # Batch that decoder work over the already encoded memory; no second encoder.
        best_branch = (
            details["branch_q"][:, 1:].masked_fill(~branch_legal, -torch.inf).argmax(-1) + 1
        )
        replacement = (branches == 0) | ~torch.as_tensor(
            legal[np.arange(env.num_envs), proposed], device=policy.device
        )
        at = torch.nonzero(
            replacement & branch_legal.any(-1) & torch.as_tensor(sampled, device=policy.device),
            as_tuple=False,
        ).flatten()
        if len(at):
            tables[at] = policy.conditional_tiles(
                details["observations"], best_branch[at], rows=at, memory=details["board_memory"]
            ).float()
        near_branch = torch.where(replacement, best_branch, branches).cpu().numpy()
        scores = tables.cpu().numpy()
    commands = np.zeros((env.num_envs, 5), np.uint16)
    valid = np.zeros((env.num_envs, 5), bool)
    labels = np.broadcast_to(np.array([10, 5, 1, 10], np.int8), (env.num_envs, 5, 5, 4)).copy()
    jobs, shortages = [], 0
    for slot in enabled:
        action = int(proposed[slot])
        if interval:
            selected = [action, None, None]
            if sampled[slot]:
                selected[1:] = interval_alternatives(
                    action,
                    legal[slot],
                    int(near_branch[slot]),
                    scores[slot],
                    sampling_rng(model, slot, 0xEFFE),
                )
            shortages += sum(a is None for a in selected[1:]) if sampled[slot] else 0
            for token, candidate in enumerate(selected):
                if candidate is not None:
                    commands[slot, token], valid[slot, token] = candidate, True
                    if candidate and legal[slot, candidate]:
                        jobs.append((int(slot), token))
            continue
        selected = [action]
        if maximum:
            candidates = np.flatnonzero(legal[slot])
            candidates = candidates[(candidates != action) & (candidates != 0)]
            branch = int(near_branch[slot])
            branch_candidates = [int(a) for a in candidates if (a - 1) // 45 + 1 == branch]
            branch_candidates.sort(key=lambda a: (-float(scores[slot, (a - 1) % 45]), a))
            selected.extend(branch_candidates[: maximum // 2])
            remaining = np.array([a for a in candidates if a not in selected], np.int64)
            # Separate keyed RNG: neither policy exploration nor real engine RNG is consumed.
            rng = sampling_rng(model, slot, 0xEFFE)
            take = min(maximum + 1 - len(selected), len(remaining))
            if take:
                selected.extend(int(a) for a in rng.choice(remaining, take, replace=False))
        shortages += maximum + 1 - len(selected)
        for token, action in enumerate(selected):
            commands[slot, token], valid[slot, token] = action, True
            if action and legal[slot, action]:
                jobs.append((int(slot), token))
    request = dict(
        actions=commands,
        valid=valid,
        labels=labels,
        jobs=jobs,
        ready=not bool(jobs),
        shortages=shortages,
        policy_identity=(model.cohort_number, model.optimizer_commits),
    )
    if absolute:
        from pvz_rl.envs.cuda_snapshots import download_snapshots

        need = np.flatnonzero(
            issued & (((proposed != 0) & ~effective_wait) | valid[:, 1:].any(-1))
        ).tolist()
        request.update(
            version=3,
            ready=True,
            sampled=sampled,
            effective_wait=effective_wait & issued,
            legal_proposal=~effective_wait,
            pre_states=dict(zip(need, download_snapshots(env.batch, indices=need), strict=True)),
        )
        request["labels"].fill(0)
        request["jobs"] = [(slot, token) for slot, token in jobs if token != 0]
        model._pending_effect = request
        model._scheduler.pending |= issued
        return request
    if interval:
        request.update(
            version=2,
            sampled=sampled,
            effective_wait=effective_wait & issued,
            actual_probe=np.zeros((env.num_envs, 5), np.int64),
            actual_anchor=np.full(env.num_envs, -1, np.int64),
            actual_complete=np.zeros(env.num_envs, bool),
        )
        metrics = env.fork_service().metrics
        metrics["sampled_waits"] += int((sampled & effective_wait).sum())
        metrics["skipped_waits"] += int((issued & effective_wait & ~sampled).sum())
    # Persist choices before launching scratch work; resume never samples again.
    model._pending_effect = request
    model._scheduler.pending |= issued
    if jobs:
        env.fork_service().begin(
            [slot for slot, _ in jobs], [commands[slot, token] for slot, token in jobs]
        )
    return request


def finish_effects(model):
    req, env = model._pending_effect, model.env
    if req["ready"]:
        return True
    service = env.fork_service()
    if service.request is None:
        # Queue preparation may have failed before any job launched. Retain the
        # original probability decision and candidates across that recovery too.
        service.begin(
            [slot for slot, _ in req["jobs"]],
            [req["actions"][slot, token] for slot, token in req["jobs"]],
        )
    if not service.advance():
        return False
    anchors = service.execution.anchors[: len(req["jobs"])].get()
    outcomes = service.take()
    for (slot, token), after, anchor in zip(req["jobs"], outcomes, anchors, strict=True):
        req["labels"][slot, token] = effect_labels(
            env._baseline_outcomes[slot], after, reference_tau(env.batch.rules)
        )
        if req.get("version") == 2 and token == 0:
            req["actual_probe"][slot] = after[:, 3]
            req["actual_anchor"][slot] = anchor
            req["actual_complete"][slot] = True
    req["ready"] = True
    return True


def promote_executed_probes(env, request, active, done):
    if request.get("version") != 2:
        return
    actions = request["actions"][:, 0]
    accepted = env.last_action_result_host[:, 0].astype(bool)
    indices = np.flatnonzero(
        active & accepted & (actions != 0) & ~done & ~env._last_spawned & request["actual_complete"]
    )
    # step_host has already ended the old interval using actual execution facts.
    if np.any(env.probe_intervals.state[indices] != ProbeState.UNINITIALIZED):
        raise RuntimeError("Accepted action did not end its probe interval")
    env.probe_intervals.begin(indices)
    env.probe_intervals.complete(
        indices, request["actual_probe"][indices], request["actual_anchor"][indices]
    )


def validate_effect_request(request, model):
    if request is None:
        if model._scheduler.pending.any():
            raise ValueError("Pending effect recovery is missing; use an intact checkpoint")
        return
    count = model.n_envs
    if request.get("version") == 3:
        if not absolute_outcomes(model.cfg) or request.get("ready") is not True:
            raise ValueError("Absolute action tickets disagree with the model protocol")
        for name, shape, dtype in (
            ("actions", (count, 5), np.uint16),
            ("valid", (count, 5), bool),
            ("labels", (count, 5, 5, 4), np.int8),
            ("sampled", (count,), bool),
            ("effective_wait", (count,), bool),
            ("legal_proposal", (count,), bool),
        ):
            value = np.asarray(request[name])
            if value.shape != shape or value.dtype != dtype:
                raise ValueError("Invalid saved absolute action ticket")
        if (
            tuple(request["policy_identity"]) != (model.cohort_number, model.optimizer_commits)
            or np.any(request["actions"] >= ActionSchema.size)
            or np.any(request["labels"] < 0)
            or np.any(request["labels"] >= np.array([11, 6, 2, 11]))
        ):
            raise ValueError("Absolute ticket identity, action or label changed")
        for slot in np.flatnonzero(model._scheduler.pending):
            if not request["valid"][slot, 0]:
                raise ValueError("Pending command has no absolute action ticket")
            candidates = request["actions"][slot, request["valid"][slot]]
            if len(set(candidates)) != len(candidates):
                raise ValueError("Duplicated saved alternative")
            needs_state = (
                request["legal_proposal"][slot] and request["actions"][slot, 0] != 0
            ) or request["valid"][slot, 1:].any()
            if needs_state and int(slot) not in request["pre_states"]:
                raise ValueError("Pending action lost its owned physical root")
        pending = model._scheduler.pending
        if pending.any() and not np.array_equal(
            request["actions"][pending, 0], model._scheduler.columns[pending, 1].cpu().numpy()
        ):
            raise ValueError("Absolute tickets disagree with retained commands")
        return
    if request.get("version", 1) not in (1, 2):
        raise ValueError("Unsupported pending effect version")
    if interval_probes_enabled(model.cfg) != (request.get("version") == 2):
        raise ValueError("Pending effects disagree with saved collection protocol")
    if request.get("version") == 2:
        for name, shape, dtype in (
            ("sampled", (count,), bool),
            ("effective_wait", (count,), bool),
            ("actual_probe", (count, 5), np.int64),
            ("actual_anchor", (count,), np.int64),
            ("actual_complete", (count,), bool),
        ):
            value = np.asarray(request[name])
            if value.shape != shape or value.dtype != dtype:
                raise ValueError("Invalid saved interval effect metadata")
    for name, shape, dtype in (
        ("actions", (count, 5), np.uint16),
        ("valid", (count, 5), bool),
        ("labels", (count, 5, 5, 4), np.int8),
    ):
        value = np.asarray(request[name])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError("Invalid pending effect storage")
    if (
        type(request["ready"]) is not bool
        or np.any(request["actions"] >= ActionSchema.size)
        or np.any(request["labels"] < 0)
        or np.any(request["labels"] >= np.array([21, 11, 3, 21]))
        or tuple(request["policy_identity"]) != (model.cohort_number, model.optimizer_commits)
    ):
        raise ValueError("Pending effect labels or frozen policy identity disagree")
    jobs = [tuple(job) for job in request["jobs"]]
    if len(set(jobs)) != len(jobs) or any(
        not (
            0 <= root < count
            and 0 <= token < 5
            and request["valid"][root, token]
            and request["actions"][root, token] > 0
        )
        for root, token in jobs
    ):
        raise ValueError("Invalid pending effect jobs")
    pending = model._scheduler.pending
    if pending.any() and not np.array_equal(
        request["actions"][pending, 0], model._scheduler.columns[pending, 1].cpu().numpy()
    ):
        raise ValueError("Effect roots disagree with retained main commands")
