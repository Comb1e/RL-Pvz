"""Flat complete-action fitting with separate detached species consistency."""

import numpy as np
import torch

from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.objective import probe_loss, species_consistency_loss
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
    probe_counts=None,
    consistency_counts=None,
):
    """One flat batch's additive contribution to whole-cohort denominators."""
    device = next(policy.parameters()).device

    def tensor(name, dtype):
        return (
            fields[name]
            if fields is not None
            else torch.as_tensor(np.array(rows[name]), device=device, dtype=dtype)
        )

    active = tensor("active", torch.bool).reshape(-1)
    observations = collate_observations(observations, device).reshape(-1)[active]
    actions = tensor("action", torch.long).reshape(-1)[active]
    accepted = tensor("accepted", torch.bool).reshape(-1)[active]
    targets = tensor("target", torch.float32).reshape(-1)[active]
    if not len(actions):
        zero = next(policy.parameters()).sum() * 0
        policy._last_selected_loss = zero.detach()
        policy._last_probe_loss = zero.detach()
        policy._last_species_consistency_loss = zero.detach()
        return zero, zero.new_empty(0), zero.new_empty(0)
    branches, positions = action_parts(actions)
    nonwait = branches != 0
    source = nonwait.nonzero(as_tuple=True)[0]
    species = policy.species_values(observations)
    masks = observation_tile_masks(observations)
    objective = policy.cfg["training"]["objective"]
    probes = None
    probe_source = probe_slot = None
    request_source, request_branches = source, branches[nonwait]
    if probe_counts is not None and objective.get("probe_loss_weight", 0):
        if fields is not None:
            probes = fields["probes"][active]
        else:
            probe_rows = rows["probes"].reshape(len(active), -1)[rows["active"].reshape(-1)]
            probes = torch.as_tensor(
                np.stack(
                    [probe_rows[name] for name in ("valid", "action", "target", "accepted")], -1
                ),
                device=device,
                dtype=torch.float32,
            )
        probe_source, probe_slot = probes[..., 0].bool().nonzero(as_tuple=True)
        probe_branches, _ = action_parts(probes[probe_source, probe_slot, 1].long())
        request_source = torch.cat((source, probe_source))
        request_branches = torch.cat((branches[nonwait], probe_branches))
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
    if len(source) and objective["species_consistency_weight"]:
        with torch.no_grad():
            ema_tiles = teacher.conditional_tiles(observations, branches[nonwait], rows=source)
            maxima = candidate_maxima(ema_tiles, masks[nonwait], branches[nonwait])
        consistency = species_consistency_loss(
            species[nonwait],
            maxima,
            actions[nonwait],
            accepted[nonwait],
            torch.as_tensor(counts)[1:] if consistency_counts is None else consistency_counts,
            accepted_share=objective["accepted_outcome_share"],
        )
        loss = loss + objective["species_consistency_weight"] * consistency
    auxiliary = species.sum() * 0
    if probes is not None:
        _, probe_positions = action_parts(probes[probe_source, probe_slot, 1].long())
        predictions = tables[len(source) :].gather(1, probe_positions[:, None]).flatten()
        auxiliary = probe_loss(
            policy,
            observations,
            probes,
            probe_counts,
            objective["probe_huber_delta"],
            predictions=predictions,
        )
        loss = loss + objective["probe_loss_weight"] * auxiliary
    policy._last_probe_loss = auxiliary.detach()
    policy._last_species_consistency_loss = consistency.detach()
    gap = (
        candidate_maxima(tables[: len(source)], masks[nonwait], branches[nonwait])
        - selected[nonwait]
    )
    return loss, selected_errors, gap.detach()
