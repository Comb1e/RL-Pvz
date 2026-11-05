"""Joint shared-encoder Q fitting; verified demos retain their separate objectives."""

from dataclasses import dataclass

import numpy as np
import torch

from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.action_routing import route_indices
from pvz_rl.learning.forecast_metrics import OUTCOME_METRIC_WIDTH
from pvz_rl.learning.objective import (
    selected_action_targets,
    species_consistency_loss,
    species_consistency_targets,
)
from pvz_rl.policy.sequential_q import (
    action_parts,
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
    teacher=None,
    consistency_counts=None,
    coefficients=None,
    fit_stage="joint",
    return_forward=False,
    outcome_decision_count=None,
    outcome_enabled=False,
    demonstration=False,
):
    """One encoder pass; attempted-branch Q calibration and optional exact outcome fitting."""
    from pvz_rl.learning.forecast_objective import forecast_auxiliary
    from pvz_rl.policy.sequential_q import action_groups

    device = policy.device
    observations = collate_observations(observations, device).reshape(-1)
    active = np.flatnonzero(rows["active"])
    if len(active) != len(rows):
        rows, observations = rows[active], observations[torch.as_tensor(active, device=device)]
    plan = route_indices(rows)
    actions = torch.as_tensor(np.array(rows["action"]), device=device, dtype=torch.long)
    accepted = torch.as_tensor(np.array(rows["accepted"]), device=device, dtype=torch.bool)
    targets = torch.as_tensor(np.array(rows["target"]), device=device, dtype=torch.float32)
    routes = torch.as_tensor(np.array(rows["route"]), device=device, dtype=torch.long)
    branches, positions = action_parts(actions)
    objective = policy.cfg["training"]["objective"]
    n = len(rows)
    species_rows = torch.as_tensor(
        plan["species"] if fit_stage == "joint" else [], device=device, dtype=torch.long
    )
    tile_rows = torch.as_tensor(plan["tiles"], device=device, dtype=torch.long)
    errors = torch.zeros(n, device=device)
    heads = torch.zeros(n, device=device)
    dense_species = torch.zeros((n, 10), device=device, dtype=torch.bfloat16)
    dense_tiles = torch.zeros((n, 45), device=device, dtype=torch.bfloat16)
    features = None
    with policy.board_context(observations) as memory:
        if len(species_rows):
            # The complete board batch was encoded once, including tile-only routes.
            species = policy.action_values(memory, rows=species_rows)
            dense_species = dense_species.index_copy(0, species_rows, species)
            prediction = species.float().gather(1, branches[species_rows, None]).flatten()
            expected = selected_action_targets(
                targets[species_rows],
                routes[species_rows],
                species.detach().float().amin(-1),
                objective["invalid_action_q_offset"],
            )
            include = torch.ones_like(prediction)
            if demonstration:
                include = (routes[species_rows] != 1).to(prediction.dtype)
            errors = errors.index_add(0, species_rows, (prediction - expected).square() * include)
            heads = heads.index_add(0, species_rows, include)
        if len(tile_rows):
            features = policy.tile_features(
                observations, branches[tile_rows], rows=tile_rows, memory=memory
            )
            tables = policy.tile_values(
                features, null=policy.null_features((memory[0][tile_rows], memory[1][tile_rows]))
            )
            dense_tiles = dense_tiles.index_copy(0, tile_rows, tables)
            prediction = tables.float().gather(1, positions[tile_rows, None]).flatten()
            expected = selected_action_targets(
                targets[tile_rows],
                routes[tile_rows],
                tables.detach().float().amin(-1),
                objective["invalid_action_q_offset"],
            )
            errors = errors.index_add(0, tile_rows, (prediction - expected).square())
            heads = heads.index_add(0, tile_rows, torch.ones_like(prediction))
    errors = errors / heads.clamp_min(1)
    weights = (
        coefficients.selected
        if coefficients is not None
        else outcome_weights(
            counts,
            device=device,
            dtype=torch.float32,
            accepted_share=objective["accepted_outcome_share"],
        )
    )
    q_loss = (errors * weights[action_groups(actions), accepted.long()]).sum()
    loss = q_loss
    policy._primary_graph = q_loss
    policy._last_outcome_metrics = torch.zeros(
        OUTCOME_METRIC_WIDTH, device=device, dtype=torch.float64
    )
    if outcome_enabled:
        auxiliary, metrics = forecast_auxiliary(
            policy,
            observations,
            rows,
            memory,
            features=features,
            feature_rows=plan["tiles"],
            decision_count=outcome_decision_count,
        )
        loss = loss + auxiliary
        policy._last_outcome_metrics = metrics
    consistency = q_loss.detach() * 0
    policy._last_species_consistency_metrics = q_loss.new_zeros(4, dtype=torch.float64)
    if demonstration and len(plan["ema"]) and objective["species_consistency_weight"]:
        ema = torch.as_tensor(plan["ema"], device=device, dtype=torch.long)
        online = dense_tiles[ema]
        with torch.no_grad():
            expected, _, evaluation_gap = species_consistency_targets(
                online,
                teacher.conditional_tiles(observations, branches[ema], rows=ema),
                None,
                branches[ema],
            )
        current = dense_species[ema].float()
        consistency = species_consistency_loss(
            current,
            expected.float(),
            actions[ema],
            accepted[ema],
            consistency_counts,
            delta=objective["species_consistency_huber_delta"],
            accepted_share=objective["accepted_outcome_share"],
            weights=coefficients.consistency if coefficients is not None else None,
        )
        loss = loss + objective["species_consistency_weight"] * consistency
        residual = (current.gather(1, branches[ema, None]).flatten() - expected).abs().detach()
        policy._last_species_consistency_metrics = torch.stack(
            (
                residual.double().sum(),
                (residual > objective["species_consistency_huber_delta"]).double().sum(),
                evaluation_gap.double().sum(),
                residual.new_tensor(len(ema), dtype=torch.float64),
            )
        )
    if not loss.requires_grad:
        loss = loss + next(policy.parameters()).sum() * 0
    policy._last_selected_loss = q_loss.detach()
    policy._last_species_consistency_loss = consistency.detach()
    gap = loss.new_empty(0)
    if len(tile_rows):
        gap = (
            tables.float().max(-1).values
            - tables.float().gather(1, positions[tile_rows, None]).flatten()
        )
    result = loss, errors.detach(), gap.detach()
    return (*result, FittingForwardResult(dense_species, dense_tiles)) if return_forward else result
