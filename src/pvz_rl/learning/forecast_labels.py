"""Absolute lane outcomes; fixed 56-slot readout with 28 categorical logits."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.envs.outcome_math import ABSOLUTE_SIZES, time_bin
from pvz_rl.envs.outcome_math import depth_code as depth_code

FORECAST_PROTOCOL = "static-fork-action-outcomes-v3"
DEPTH_CLASSES = 11
TIME_CLASSES = 6
EFFECT_WIDTH = 56
EFFECT_OFFSETS = (0, 21, 35)  # Keep the old lane-loss slots reserved.


def effect_blocks(logits):
    """Reserved logits never enter a distribution, loss, summary or forecast."""
    return tuple(
        logits[..., start : start + size].float()
        for start, size in zip(EFFECT_OFFSETS, ABSOLUTE_SIZES, strict=True)
    )


def outcome_codes(outcomes, tau):
    """Convert relative first home ticks to categorical breach bins."""
    outcomes = np.asarray(outcomes)
    result = outcomes.astype(np.int8).copy()
    result[..., 1] = time_bin(outcomes[..., 1], tau)
    return result


def categorical_loss(logits, labels):
    """Unknown (-1) labels contribute no categorical term."""
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    known = labels >= 0
    exact = (
        -F.log_softmax(logits.float(), -1).gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1)
    )
    return torch.where(known, exact, 0), known


def effect_loss_components(logits, labels):
    """Separate field/probe contributions with the same known-term denominator."""
    losses, known = [], []
    for index, block in enumerate(effect_blocks(logits)):
        loss, valid = categorical_loss(block, labels[..., index])
        losses.append(loss)
        known.append(valid)
    denominator = torch.stack(known, -1).sum(-1).clamp_min(1)
    field_depth, breach, probe = losses
    return torch.stack((field_depth + breach, probe), -1) / denominator[..., None]


def effect_loss(logits, labels):
    return effect_loss_components(logits, labels).sum(-1)


def effect_forecast(logits):
    """Post-action field/probe CDFs, in home and left-to-right column order."""
    field, _, probe = effect_blocks(logits)
    result = []
    support = torch.arange(11, device=logits.device)
    thresholds = torch.arange(9, 0, -1, device=logits.device)
    for block in (field, probe):
        p = block.softmax(-1)
        cells = (p[..., None, :] * (support >= thresholds[:, None])).sum(-1)
        home = p[..., 10]
        result.append(torch.cat((home[..., None], cells), -1))
    return torch.stack(result, -3)


def effect_summary(logits):
    summaries = []
    for classes, block in zip(ABSOLUTE_SIZES, effect_blocks(logits), strict=True):
        support = torch.arange(classes, device=logits.device)
        summaries.append((block.softmax(-1) * support).sum(-1) / (classes - 1))
    return torch.stack(summaries, -1).flatten(-2)
