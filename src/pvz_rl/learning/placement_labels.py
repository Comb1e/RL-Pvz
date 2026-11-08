"""Shared positive-set label construction and legacy automatic ranking."""

import math

import torch


def positive_set_mask(feasible, width):
    positive = torch.zeros(len(feasible), width, dtype=torch.bool)
    for index, yes in enumerate(feasible):
        if not yes:
            raise ValueError("empty labeled feasible set")
        if any(type(label) is not int or not 0 <= label < width for label in yes):
            raise ValueError("placement labels must be in range")
        positive[index, list(set(yes))] = True
    return positive


def _ranking_terms(values, positive, margin, legal=None):
    legal = torch.ones_like(positive) if legal is None else legal
    other = legal & ~positive
    has_other = other.any(-1)
    # Indexed reductions preserve the original lowest-index tie subgradient.
    best = values.masked_fill(~positive, -torch.inf).max(-1).values
    competitor = values.masked_fill(~other, -torch.inf).max(-1).values
    top = torch.relu(margin + torch.where(has_other, competitor, best) - best)
    return torch.where(has_other, top, 0)


def set_ranking_loss(values, feasible_tiles, *, margin=1.0, legal_masks=None):
    """Rank one positive above every alternative, including occupied tiles."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    positive = positive_set_mask(feasible_tiles, values.shape[-1])
    if legal_masks is not None:
        legal_cpu = legal_masks.detach().cpu()
        if bool((positive & ~legal_cpu).any()):
            raise ValueError("positive tiles must be legal")
    positive = positive.to(values.device)
    top = _ranking_terms(values, positive, margin, legal_masks)
    return top.mean() if len(values) else values.sum() * 0
