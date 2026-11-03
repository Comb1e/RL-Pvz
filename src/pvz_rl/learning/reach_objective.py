"""Masked future cell/home BCE with equal lane and decision weighting."""

import numpy as np
import torch
from torch.nn import functional as F

REACH_METRICS = 10


def reach_loss(logits, labels, home_labels, *, decision_count=None):
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.float32)
    home_labels = torch.as_tensor(home_labels, device=logits.device, dtype=torch.float32)
    if (
        logits.shape[1:] != (5, 10)
        or labels.shape != logits.shape[:2] + (9,)
        or home_labels.shape != logits.shape[:2]
    ):
        raise ValueError(
            "Reach requires [batch,5,10] logits, [batch,5,9] cells and [batch,5] homes"
        )
    valid, home_valid = labels >= 0, home_labels >= 0
    lanes = valid.any(-1)
    eligible, home_eligible = lanes.any(-1), home_valid.any(-1)
    values = logits.float()
    errors = F.binary_cross_entropy_with_logits(
        values[..., :9], labels.clamp_min(0), reduction="none"
    )
    home_errors = F.binary_cross_entropy_with_logits(
        values[..., 9], home_labels.clamp_min(0), reduction="none"
    )
    per_lane = (errors * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    per_decision = (per_lane * lanes).sum(-1) / lanes.sum(-1).clamp_min(1)
    home_decision = (home_errors * home_valid).sum(-1) / home_valid.sum(-1).clamp_min(1)
    counts = (
        (eligible.sum().clamp_min(1), home_eligible.sum().clamp_min(1))
        if decision_count is None
        else tuple(max(1, int(x)) for x in decision_count)
    )
    cell_loss, home_loss = per_decision.sum() / counts[0], home_decision.sum() / counts[1]
    probability = values.detach().sigmoid()
    metrics = torch.stack(
        (
            cell_loss.detach(),
            ((probability[..., :9] - labels).square() * valid).sum(),
            valid.sum(),
            eligible.sum(),
            home_loss.detach(),
            ((probability[..., 9] - home_labels).square() * home_valid).sum(),
            home_valid.sum(),
            home_eligible.sum(),
            (((probability[..., :9] >= 0.5) == labels.bool()) & valid).sum(),
            (((probability[..., 9] >= 0.5) == home_labels.bool()) & home_valid).sum(),
        )
    ).double()
    return cell_loss, home_loss, metrics


def reach_eligible(rows):
    return rows["reach_valid"] & (
        (rows["reach_labels"] >= 0).any(axis=(-2, -1)) | (rows["home_labels"] >= 0).any(-1)
    )


def reach_auxiliary(policy, features, rows, *, decision_count=None, stage="normal"):
    from pvz_rl.config import home_prediction_weight, reach_prediction_weight

    cell_weight, home_weight = (
        reach_prediction_weight(policy.cfg, stage),
        home_prediction_weight(policy.cfg, stage),
    )
    with torch.set_grad_enabled(torch.is_grad_enabled() and (cell_weight > 0 or home_weight > 0)):
        cell, home, metrics = reach_loss(
            policy.reach_values(features),
            np.array(rows["reach_labels"]),
            np.array(rows["home_labels"]),
            decision_count=decision_count,
        )
    return cell_weight * cell + home_weight * home, metrics
