"""Flat complete-action fitting with separate detached species consistency."""

import numpy as np
import torch

from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.objective import (
    species_consistency_loss,
    species_consistency_targets,
)
from pvz_rl.policy.sequential_q import (
    action_parts,
    balanced_q_loss,
    candidate_maxima,
    observation_tile_masks,
)


def batch_loss(
    policy,
    rows,
    counts,
    observations,
    fields=None,
    *,
    teacher,
    consistency_counts=None,
):
    """One flat batch's additive contribution to whole-cohort denominators."""
    device = next(policy.parameters()).device

    def tensor(name, dtype):
        value = (
            fields[name]
            if fields is not None
            else torch.as_tensor(np.array(rows[name]), device=device, dtype=dtype)
        )
        return value.to(device=device, dtype=dtype)

    active = tensor("active", torch.bool).reshape(-1)
    observations = collate_observations(observations, device).reshape(-1)[active]
    actions = tensor("action", torch.long).reshape(-1)[active]
    accepted = tensor("accepted", torch.bool).reshape(-1)[active]
    targets = tensor("target", torch.bfloat16).reshape(-1)[active]
    if not len(actions):
        zero = next(policy.parameters()).sum() * 0
        policy._last_selected_loss = zero.detach()
        policy._last_species_consistency_loss = zero.detach()
        policy._last_species_consistency_metrics = zero.new_zeros(4, dtype=torch.float64)
        return zero, zero.new_empty(0), zero.new_empty(0)
    branches, positions = action_parts(actions)
    nonwait = branches != 0
    source = nonwait.nonzero(as_tuple=True)[0]
    species = policy.species_values(observations)
    masks = observation_tile_masks(observations)
    objective = policy.cfg["training"]["objective"]
    request_source, request_branches = source, branches[nonwait]
    tables = (
        policy.conditional_tiles(observations, request_branches, rows=request_source)
        if len(request_source)
        else species.new_empty(0, 45)
    )
    actual_tiles = species.new_zeros(len(actions), 45)
    actual_tiles = actual_tiles.index_copy(0, source, tables[: len(source)])
    selected = policy.selected_values(observations, actions, species=species, tiles=actual_tiles)
    loss, selected_errors = balanced_q_loss(
        selected,
        targets,
        actions,
        counts,
        accepted=accepted,
        accepted_share=objective["accepted_outcome_share"],
    )
    policy._last_selected_loss = loss.detach()
    consistency = species.sum() * 0
    policy._last_species_consistency_metrics = species.new_zeros(4, dtype=torch.float64)
    if len(source) and objective["species_consistency_weight"]:
        with torch.no_grad():
            ema_tiles = teacher.conditional_tiles(observations, branches[nonwait], rows=source)
            consistency_targets, _, evaluation_gap = species_consistency_targets(
                tables[: len(source)], ema_tiles, masks[nonwait], branches[nonwait]
            )
            residual = (
                species[nonwait].gather(1, branches[nonwait, None]).flatten() - consistency_targets
            ).abs()
            policy._last_species_consistency_metrics = torch.stack(
                (
                    residual.double().sum(),
                    (residual > objective["species_consistency_huber_delta"]).double().sum(),
                    evaluation_gap.double().sum(),
                    residual.new_tensor(len(source), dtype=torch.float64),
                )
            )
        consistency = species_consistency_loss(
            species[nonwait],
            consistency_targets,
            actions[nonwait],
            accepted[nonwait],
            torch.as_tensor(counts)[1:] if consistency_counts is None else consistency_counts,
            delta=objective["species_consistency_huber_delta"],
            accepted_share=objective["accepted_outcome_share"],
        )
        loss = loss + objective["species_consistency_weight"] * consistency
    policy._last_species_consistency_loss = consistency.detach()
    gap = (
        candidate_maxima(tables[: len(source)], masks[nonwait], branches[nonwait])
        - selected[nonwait]
    )
    return loss, selected_errors, gap.detach()
