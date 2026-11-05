"""Absolute lane outcomes; fixed 56-slot readout with 30 categorical logits."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.envs.outcome_math import depth_code as depth_code
from pvz_rl.envs.outcome_math import time_bin

FORECAST_PROTOCOL = "static-fork-action-outcomes-v2"
DEPTH_CLASSES = 11
TIME_CLASSES = 6
EFFECT_SIZES = (21, 11, 3, 21)  # Carrier offsets; unused classes stay reserved.
EFFECT_WIDTH = sum(EFFECT_SIZES)
ABSOLUTE_SIZES = (11, 6, 2, 11)


def effect_blocks(logits):
    """Reserved logits never enter a distribution, loss, summary or forecast."""
    return tuple(
        block[..., :size]
        for block, size in zip(logits.float().split(EFFECT_SIZES, -1), ABSOLUTE_SIZES, strict=True)
    )


def outcome_codes(outcomes, tau):
    """[lane, field depth / relative first home tick / loss / probe depth]."""
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


def effect_loss(logits, labels):
    losses, known = [], []
    for index, block in enumerate(effect_blocks(logits)):
        loss, valid = categorical_loss(block, labels[..., index])
        losses.append(loss)
        known.append(valid)
    return torch.stack(losses, -1).sum(-1) / torch.stack(known, -1).sum(-1).clamp_min(1)


def effect_forecast(logits):
    """Post-action field/probe CDFs, in home and left-to-right column order."""
    blocks, result = effect_blocks(logits), []
    support = torch.arange(11, device=logits.device)
    thresholds = torch.arange(9, 0, -1, device=logits.device)
    for term in (0, 3):
        p = blocks[term].softmax(-1)
        cells = (p[..., None, :] * (support >= thresholds[:, None])).sum(-1)
        home = blocks[2].softmax(-1)[..., 1] if term == 0 else p[..., 10]
        result.append(torch.cat((home[..., None], cells), -1))
    return torch.stack(result, -3)


def effect_summary(logits):
    summaries = []
    for classes, block in zip(ABSOLUTE_SIZES, effect_blocks(logits), strict=True):
        support = torch.arange(classes, device=logits.device)
        summaries.append((block.softmax(-1) * support).sum(-1) / (classes - 1))
    return torch.stack(summaries, -1).flatten(-2)
