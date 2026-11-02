"""Factual lane-boundary prediction, normalized by decisions rather than targets."""

import torch
from torch.nn import functional as F


def reach_loss(logits, labels, *, decision_count=None):
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    valid = labels >= 0
    if labels.shape != logits.shape[:-1] or logits.shape[-1] != 10:
        raise ValueError("Reach requires [decisions, 5, 10] logits and five lane labels")
    eligible = valid.any(-1)
    errors = F.cross_entropy(
        logits.float().flatten(0, 1), labels.clamp_min(0).flatten(), reduction="none"
    ).reshape_as(labels)
    losses = (errors * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    denominator = eligible.sum().clamp_min(1) if decision_count is None else max(1, decision_count)
    loss = (losses * eligible).sum() / denominator
    predicted = logits.detach().float().argmax(-1)
    # Outside is ordered immediately to the right of column nine for MAE.
    metrics = torch.stack(
        (
            loss.detach(),
            ((predicted == labels) & valid).sum(),
            ((predicted - labels).abs() * valid).sum(),
            valid.sum(),
            eligible.sum(),
        )
    ).double()
    return loss, metrics
