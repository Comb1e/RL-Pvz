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

    trusted = fields is not None and getattr(fields, "trusted", False)
    observations = collate_observations(observations, device).reshape(-1)
    actions = tensor("action", torch.long).reshape(-1)
    accepted = tensor("accepted", torch.bool).reshape(-1)
    targets = tensor("target", torch.bfloat16).reshape(-1)
    if not trusted:
        active = tensor("active", torch.bool).reshape(-1)
        observations, actions = observations[active], actions[active]
        accepted, targets = accepted[active], targets[active]
    if not len(actions):
        zero = next(policy.parameters()).sum() * 0
        policy._last_selected_loss = zero.detach()
        policy._last_species_consistency_loss = zero.detach()
        policy._last_species_consistency_metrics = zero.new_zeros(4, dtype=torch.float64)
        return zero, zero.new_empty(0), zero.new_empty(0)
    branches, positions = action_parts(actions)
    nonwait = branches != 0
    source = (
        fields.nonwait_indices.to(device=device)
        if trusted
        else nonwait.nonzero(as_tuple=True)[0]
    )
    species = policy.species_values(observations)
    masks = observation_tile_masks(observations)
    objective = policy.cfg["training"]["objective"]
    request_source, request_branches = source, branches.index_select(0, source)
    tile_forward = policy.fitting_tiles if trusted and hasattr(policy, "fitting_tiles") else policy.conditional_tiles
    tables = (
        tile_forward(observations, request_branches, rows=request_source)
        if len(request_source)
        else species.new_empty(0, 45)
    )
    if trusted:
        chosen = tables.gather(1, positions.index_select(0, source)[:, None]).flatten()
        selected = species[:, 0].index_copy(0, source, chosen)
    else:
        actual_tiles = species.new_zeros(len(actions), 45).index_copy(0, source, tables)
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
            ema_forward = teacher.fitting_tiles if trusted and hasattr(teacher, "fitting_tiles") else teacher.conditional_tiles
            ema_tiles = ema_forward(observations, request_branches, rows=source)
            consistency_targets, _, evaluation_gap = species_consistency_targets(
                tables[: len(source)], ema_tiles, masks.index_select(0, source), request_branches
            )
            residual = (
                species.index_select(0, source).gather(1, request_branches[:, None]).flatten() - consistency_targets
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
            species.index_select(0, source),
            consistency_targets,
            actions.index_select(0, source),
            accepted.index_select(0, source),
            torch.as_tensor(counts)[1:] if consistency_counts is None else consistency_counts,
            delta=objective["species_consistency_huber_delta"],
            accepted_share=objective["accepted_outcome_share"],
        )
        loss = loss + objective["species_consistency_weight"] * consistency
    policy._last_species_consistency_loss = consistency.detach()
    gap = (
        candidate_maxima(tables[: len(source)], masks.index_select(0, source), request_branches)
        - selected.index_select(0, source)
    )
    return loss, selected_errors, gap.detach()
