"""Act-time causal labels. Private fork choices never change main-game behavior."""

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_labels import effect_labels
from pvz_rl.policy.sequential_q import action_parts


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
    tables = details["tile_q"].float().clone()
    branches, _ = action_parts(actions)
    branch_legal = torch.as_tensor(legal[:, 1:].reshape(-1, 9, 45).any(-1), device=policy.device)
    # Wait/unavailable decisions need a proposal branch for near-Q alternatives.
    # Batch that decoder work over the already encoded memory; no second encoder.
    best_branch = details["branch_q"][:, 1:].masked_fill(~branch_legal, -torch.inf).argmax(-1) + 1
    replacement = (branches == 0) | ~torch.as_tensor(
        legal[np.arange(env.num_envs), proposed], device=policy.device
    )
    at = torch.nonzero(
        replacement & branch_legal.any(-1) & torch.as_tensor(issued, device=policy.device),
        as_tuple=False,
    ).flatten()
    if len(at):
        tables[at] = policy.conditional_tiles(
            details["observations"], best_branch[at], rows=at, memory=details["board_memory"]
        ).float()
    near_branch = torch.where(replacement, best_branch, branches).cpu().numpy()
    scores = tables.cpu().numpy()
    maximum = model.cfg["training"]["forecasts"]["extra_actions"]
    commands = np.zeros((env.num_envs, 5), np.uint16)
    valid = np.zeros((env.num_envs, 5), bool)
    labels = np.broadcast_to(np.array([10, 5, 1, 10], np.int8), (env.num_envs, 5, 5, 4)).copy()
    jobs, shortages = [], 0
    for slot in enabled:
        action = int(proposed[slot])
        selected = [action]
        candidates = np.flatnonzero(legal[slot])
        candidates = candidates[(candidates != action) & (candidates != 0)]
        branch = int(near_branch[slot])
        branch_candidates = [int(a) for a in candidates if (a - 1) // 45 + 1 == branch]
        branch_candidates.sort(key=lambda a: (-float(scores[slot, (a - 1) % 45]), a))
        selected.extend(branch_candidates[: maximum // 2])
        remaining = np.array([a for a in candidates if a not in selected], np.int64)
        # Separate keyed RNG: neither policy exploration nor real engine RNG is consumed.
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(model.seed or 0),
                    int(slot),
                    int(env._episode_serial[slot]),
                    int(model._scheduler.decisions[slot]),
                    0xEFFE,
                ]
            )
        )
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
        ready=False,
        shortages=shortages,
        policy_identity=(model.cohort_number, model.optimizer_commits),
    )
    env.fork_service().begin(
        [slot for slot, _ in jobs], [commands[slot, token] for slot, token in jobs]
    )
    return request


def finish_effects(model):
    req, env = model._pending_effect, model.env
    if req["ready"]:
        return True
    service = env.fork_service()
    if not service.advance():
        return False
    outcomes = service.take()
    for (slot, token), after in zip(req["jobs"], outcomes, strict=True):
        req["labels"][slot, token] = effect_labels(
            env._baseline_outcomes[slot], after, reference_tau(env.batch.rules)
        )
    req["ready"] = True
    return True


def validate_effect_request(request, model):
    if request is None:
        if model._scheduler.pending.any():
            raise ValueError("Pending effect recovery is missing; use an intact checkpoint")
        return
    count = model.n_envs
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
