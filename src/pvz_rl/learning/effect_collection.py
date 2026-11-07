"""Act-time causal labels. Private fork choices never change main-game behavior."""

import numpy as np
import torch

from pvz_rl.config import effect_extra_actions
from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.tile_scenarios import is_placement_skill, is_single_frame_skill
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


def begin_effects(model, actions, details, issued, legal):
    env, policy = model.env, model.policy
    enabled = np.flatnonzero(issued)
    proposed = actions.detach().cpu().numpy()
    legal = legal.cpu().numpy()
    placement = is_single_frame_skill(model.tile_skills.phase)
    maximum = effect_extra_actions(model.cfg, model.tile_skills.phase)
    effective_wait = (proposed == 0) | ~legal[np.arange(env.num_envs), proposed]
    sampled = issued.copy() if maximum else np.zeros_like(issued)
    if maximum:
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
    jobs, shortages = [], 0
    for slot in enabled:
        action = int(proposed[slot])
        if is_placement_skill(model.tile_skills.phase) and action == 0:
            raise RuntimeError("A placement exercise cannot propose Wait")
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
                if token and candidate and legal[slot, candidate]:
                    jobs.append((int(slot), token))
    from pvz_rl.envs.cuda_snapshots import download_snapshots

    need = (
        enabled.tolist()
        if placement
        else np.flatnonzero(
            issued & (((proposed != 0) & ~effective_wait) | valid[:, 1:].any(-1))
        ).tolist()
    )
    request = dict(
        version=4,
        placement=placement,
        ready=True,
        actions=commands,
        valid=valid,
        labels=np.zeros((env.num_envs, 5, 5, 4), np.int8),
        jobs=jobs,
        shortages=shortages,
        policy_identity=(model.cohort_number, model.optimizer_commits),
        sampled=sampled,
        effective_wait=effective_wait & issued,
        legal_proposal=~effective_wait,
        pre_states=dict(
            zip(
                need,
                env.auxiliary.capture(env.batch, need, env._physical_header[need])
                if getattr(env, "auxiliary", None) is not None
                else download_snapshots(env.batch, indices=need),
                strict=True,
            )
        ),
    )
    # Persist commands, probability draws and owned roots before the real mutation.
    model._pending_effect = request
    model._scheduler.pending |= issued
    return request


def validate_effect_request(request, model):
    if request is None:
        if model._scheduler.pending.any():
            raise ValueError("Pending effect recovery is missing; use an intact checkpoint")
        return
    count = model.n_envs
    if request.get("version") == 4:
        if request.get("ready") is not True or request.get("placement") != is_single_frame_skill(
            model.tile_skills.phase
        ):
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
                request["placement"]
                or (request["legal_proposal"][slot] and request["actions"][slot, 0] != 0)
                or request["valid"][slot, 1:].any()
            )
            if needs_state and int(slot) not in request["pre_states"]:
                raise ValueError("Pending action lost its owned physical root")
        pending = model._scheduler.pending
        if pending.any() and not np.array_equal(
            request["actions"][pending, 0], model._scheduler.columns[pending, 1].cpu().numpy()
        ):
            raise ValueError("Absolute tickets disagree with retained commands")
        return
    raise ValueError("Unsupported pending action outcome version; start fresh")
