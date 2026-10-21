"""Route-aware flat fitting with whole-cohort coefficients and detached EMA."""

from dataclasses import dataclass

import numpy as np
import torch

from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.action_routing import FitRoute, route_indices
from pvz_rl.learning.objective import species_consistency_loss, species_consistency_targets
from pvz_rl.policy.sequential_q import (
    action_parts,
    balanced_q_loss,
    candidate_maxima,
    observation_tile_masks,
    outcome_weights,
)


@dataclass(frozen=True)
class CohortCoefficients:
    selected: torch.Tensor
    consistency: torch.Tensor

    @classmethod
    def prepare(cls, counts, consistency_counts, device, accepted_share):
        def weights(value):
            return outcome_weights(
                value, device="cpu", dtype=torch.float32, accepted_share=accepted_share
            ).to(device)

        return cls(weights(counts), weights(consistency_counts))


@dataclass(frozen=True)
class FittingForwardResult:
    species: torch.Tensor
    tiles: torch.Tensor


def batch_loss(
    policy,
    rows,
    counts,
    observations,
    fields=None,
    *,
    teacher,
    consistency_counts=None,
    coefficients=None,
    return_forward=False,
):
    """CPU-planned routes skip all irrelevant encoders, including rejected EMA."""
    device = next(policy.parameters()).device
    observations = collate_observations(observations, device).reshape(-1)
    trusted = fields is not None and getattr(fields, "trusted", False)
    if not trusted:
        active = np.flatnonzero(rows["active"])
        rows = rows[active]
        observations = observations[torch.as_tensor(active, device=device)]
        cpu_plan = route_indices(rows)
        cpu_plan["ema_species"] = np.searchsorted(cpu_plan["species"], cpu_plan["ema"])
        cpu_plan["ema_tiles"] = np.searchsorted(cpu_plan["tiles"], cpu_plan["ema"])
        plan = {k: torch.as_tensor(v, device=device, dtype=torch.long) for k, v in cpu_plan.items()}
    else:
        plan = {k: v.to(device=device) for k, v in fields.plan.items()}

    def tensor(name, dtype):
        value = fields[name] if trusted else torch.as_tensor(np.array(rows[name]), device=device)
        return value.to(device=device, dtype=dtype).reshape(-1)

    actions = tensor("action", torch.long)
    accepted = tensor("accepted", torch.bool)
    targets = tensor("target", torch.float32)
    routes = tensor("route", torch.long)
    if not len(actions):
        zero = next(policy.parameters()).sum() * 0
        policy._last_selected_loss = policy._last_species_consistency_loss = zero.detach()
        policy._last_species_consistency_metrics = zero.new_zeros(4, dtype=torch.float64)
        result = zero, zero.new_empty(0), zero.new_empty(0)
        return (
            (*result, FittingForwardResult(zero.new_empty((0, 10)), zero.new_empty((0, 45))))
            if return_forward
            else result
        )
    branch, positions = action_parts(actions)
    objective = policy.cfg["training"]["objective"]
    species_rows, tile_rows, ema_rows = plan["species"], plan["tiles"], plan["ema"]
    selected = torch.zeros(len(actions), device=device, dtype=torch.float32)
    species = None
    if len(species_rows):
        species = policy.species_values(observations[species_rows])
        predictions = species.float().gather(1, branch[species_rows, None]).flatten()
        selected = selected.index_copy(0, species_rows, predictions)
    tile_forward = (
        policy.fitting_tiles
        if trusted and hasattr(policy, "fitting_tiles")
        else policy.conditional_tiles
    )
    tables = None
    if len(tile_rows):
        tables = tile_forward(observations, branch[tile_rows], rows=tile_rows)
        predictions = tables.float().gather(1, positions[tile_rows, None]).flatten()
        selected = selected.index_copy(0, tile_rows, predictions)
    invalid = (routes == FitRoute.SPECIES_INVALID) | (routes == FitRoute.TILE_INVALID)
    fitting_targets = torch.where(
        invalid, targets.new_tensor(objective["invalid_action_target"]), targets
    )
    loss, selected_errors = balanced_q_loss(
        selected,
        fitting_targets,
        actions,
        counts,
        accepted=accepted,
        accepted_share=objective["accepted_outcome_share"],
        weights=coefficients.selected if coefficients is not None else None,
    )
    policy._last_selected_loss = loss.detach()
    consistency = loss.detach() * 0
    policy._last_species_consistency_metrics = selected.new_zeros(4, dtype=torch.float64)
    if len(ema_rows) and objective["species_consistency_weight"]:
        online = tables.index_select(0, plan["ema_tiles"])
        current_species = species.index_select(0, plan["ema_species"]).float()
        masks = None if trusted else observation_tile_masks(observations[ema_rows])
        with torch.no_grad():
            ema_forward = (
                teacher.fitting_tiles
                if trusted and hasattr(teacher, "fitting_tiles")
                else teacher.conditional_tiles
            )
            ema_tiles = ema_forward(observations, branch[ema_rows], rows=ema_rows)
            expected, _, evaluation_gap = species_consistency_targets(
                online, ema_tiles, masks, branch[ema_rows]
            )
            expected = expected.float()
            residual = (
                current_species.gather(1, branch[ema_rows, None]).flatten() - expected
            ).abs()
            policy._last_species_consistency_metrics = torch.stack(
                (
                    residual.double().sum(),
                    (residual > objective["species_consistency_huber_delta"]).double().sum(),
                    evaluation_gap.double().sum(),
                    residual.new_tensor(len(ema_rows), dtype=torch.float64),
                )
            )
        denominators = None
        if coefficients is None:
            denominators = (
                torch.as_tensor(counts)[1:].clone()
                if consistency_counts is None
                else torch.as_tensor(consistency_counts).clone()
            )
            denominators[:, 0] = 0
        consistency = species_consistency_loss(
            current_species,
            expected,
            actions[ema_rows],
            accepted[ema_rows],
            denominators,
            delta=objective["species_consistency_huber_delta"],
            accepted_share=objective["accepted_outcome_share"],
            weights=coefficients.consistency if coefficients is not None else None,
        )
        loss = loss + objective["species_consistency_weight"] * consistency
    policy._last_species_consistency_loss = consistency.detach()
    gap = selected.new_empty(0)
    if len(tile_rows):
        maxima = (
            tables.float().max(-1).values
            if trusted
            else candidate_maxima(
                tables.float(), observation_tile_masks(observations[tile_rows]), branch[tile_rows]
            )
        )
        gap = maxima - selected[tile_rows]
    result = loss, selected_errors, gap.detach()
    if return_forward:
        dense_species = (selected if species is None else species).new_zeros(len(actions), 10)
        dense_tiles = (selected if tables is None else tables).new_zeros(len(actions), 45)
        if species is not None:
            dense_species = dense_species.index_copy(0, species_rows, species)
        if tables is not None:
            dense_tiles = dense_tiles.index_copy(0, tile_rows, tables)
        return (*result, FittingForwardResult(dense_species, dense_tiles))
    return result
