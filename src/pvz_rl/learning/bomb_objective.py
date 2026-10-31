"""Configurable-margin hit ranking with differentiable spatial guidance."""

import torch

from pvz_rl.envs.tile_scenarios import bomb_tile_distances
from pvz_rl.learning.annotation_fit import _label_masks

BOMB_OBJECTIVE = "margin_distance_illegal_v3"
LEGACY_BOMB_OBJECTIVE = "zero_margin_distance_v1"
DISTANCE_OBJECTIVES = (BOMB_OBJECTIVE, "margin_distance_v2", LEGACY_BOMB_OBJECTIVE)


def bomb_distance_loss(
    values, feasible_tiles, illegal_tiles, *, rows, cols, margin, weight, temperature,
):
    """Return equally weighted board losses and detached hinge/distance diagnostics.

    Every output participates, including occupied cells at maximum distance.
    Legacy objectives pass empty illegal sets to retain geometric distances.
    No engine or label data enters the encoder.
    """
    values = values.float()
    positive = _label_masks(feasible_tiles, rows * cols).to(values.device)
    if values.shape != positive.shape or len(illegal_tiles) != len(feasible_tiles):
        raise ValueError("Bomb outputs must match the configured board")
    nearest = torch.tensor(
        [
            bomb_tile_distances(yes, no, rows=rows, cols=cols)
            for yes, no in zip(feasible_tiles, illegal_tiles)
        ],
        device=values.device,
    )
    best = values.masked_fill(~positive, -torch.inf).max(-1).values
    has_other = (~positive).any(-1)
    competitor = values.masked_fill(positive, -torch.inf).max(-1).values
    hinge = torch.where(has_other, torch.relu(margin + competitor - best), 0)
    distance = weight * (
        torch.softmax((values - values.max(-1, keepdim=True).values) / temperature, dim=-1)
        * (nearest / (rows + cols - 2))
    ).sum(-1)
    terms = torch.stack((hinge.mean(), distance.mean()))
    return terms.sum(), terms.detach()
