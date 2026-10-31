"""Zero-margin hit ranking with bounded, differentiable spatial guidance."""

import torch

from pvz_rl.learning.annotation_fit import _label_masks

BOMB_OBJECTIVE = "zero_margin_distance_v1"


def bomb_distance_loss(values, feasible_tiles, *, rows, cols, weight, temperature):
    """Return equally weighted board losses and detached hinge/distance diagnostics.

    Every output participates, including occupied cells. Only oracle-confirmed
    legal hits have distance zero. No engine or label data enters the encoder.
    """
    values = values.float()
    positive = _label_masks(feasible_tiles, rows * cols).to(values.device)
    if values.shape != positive.shape:
        raise ValueError("Bomb outputs must match the configured board")
    coordinates = torch.arange(rows * cols, device=values.device)
    distances = (
        (coordinates[:, None] // cols - coordinates[None, :] // cols).abs()
        + (coordinates[:, None] % cols - coordinates[None, :] % cols).abs()
    )
    nearest = distances[None].masked_fill(~positive[:, None, :], rows + cols).amin(-1)
    best = values.masked_fill(~positive, -torch.inf).max(-1).values
    has_other = (~positive).any(-1)
    competitor = values.masked_fill(positive, -torch.inf).max(-1).values
    hinge = torch.relu(torch.where(has_other, competitor, best) - best)
    distance = weight * (
        torch.softmax((values - values.max(-1, keepdim=True).values) / temperature, dim=-1)
        * (nearest / (rows + cols - 2))
    ).sum(-1)
    terms = torch.stack((hinge.mean(), distance.mean()))
    return terms.sum(), terms.detach()
