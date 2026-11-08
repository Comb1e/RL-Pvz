"""Act-time causal labels. Private fork choices never change main-game behavior."""

import numpy as np

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.outcome_math import ABSOLUTE_SIZES, OUTCOME_SHAPE
from pvz_rl.envs.tile_scenarios import is_placement_skill, is_single_frame_skill


def begin_effects(model, actions, details, issued, legal):
    del details, legal
    env = model.env
    enabled = np.flatnonzero(issued)
    proposed = actions.detach().cpu().numpy()
    placement = is_single_frame_skill(model.tile_skills.phase)
    # The effect head predicts the untouched board, so no action fork is sampled.
    effective_wait = proposed == 0
    sampled = np.zeros_like(issued)
    commands = np.zeros((env.num_envs, 5), np.uint16)
    valid = np.zeros((env.num_envs, 5), bool)
    jobs, shortages = [], 0
    for slot in enabled:
        action = int(proposed[slot])
        if is_placement_skill(model.tile_skills.phase) and action == 0:
            raise RuntimeError("A placement exercise cannot propose Wait")
        commands[slot, 0], valid[slot, 0] = action, True
    from pvz_rl.envs.cuda_snapshots import download_snapshots

    need = (
        enabled.tolist()
        if placement
        else np.flatnonzero(issued & ((proposed != 0) | valid[:, 1:].any(-1))).tolist()
    )
    request = dict(
        version=5,
        placement=placement,
        ready=True,
        actions=commands,
        valid=valid,
        labels=np.zeros((env.num_envs, 5, *OUTCOME_SHAPE), np.int8),
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
    if request.get("version") == 5:
        if request.get("ready") is not True or request.get("placement") != is_single_frame_skill(
            model.tile_skills.phase
        ):
            raise ValueError("Absolute action tickets disagree with the model protocol")
        for name, shape, dtype in (
            ("actions", (count, 5), np.uint16),
            ("valid", (count, 5), bool),
            ("labels", (count, 5, *OUTCOME_SHAPE), np.int8),
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
            or np.any(request["labels"] >= ABSOLUTE_SIZES)
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
