"""Ordered lane outcomes, signed causal changes and censored categorical loss.

Depth order is none, column 8, ..., column 0, home. Increasing codes mean
greater penetration. Time order is <tau, <2tau, <4tau, <8tau, >=8tau, none.
Time differences describe *bin changes*, never invented finite breach times.
"""

import numpy as np
import torch
from torch.nn import functional as F

FORECAST_PROTOCOL = "static-fork-threat-effect-v1"
DEPTH_CLASSES = 11
TIME_CLASSES = 6
THREAT_WIDTH = 3 * DEPTH_CLASSES + TIME_CLASSES + 1
EFFECT_SIZES = (21, 11, 3, 21)
EFFECT_WIDTH = sum(EFFECT_SIZES)
ABSOLUTE_SIZES = (11, 6, 2, 11)


def effect_blocks(logits, *, absolute=False):
    """Keep the public 56-slot carrier; absolute readouts use 30 valid slots.

    Legacy block offsets remain stable. Reserved slots never participate in a
    probability distribution, loss, summary or displayed forecast.
    """
    blocks = logits.float().split(EFFECT_SIZES, -1)
    return (
        tuple(block[..., :size] for block, size in zip(blocks, ABSOLUTE_SIZES, strict=True))
        if absolute
        else blocks
    )


def depth_code(x, units=1000, house=-500):
    if x <= house:
        return 10
    return max(0, min(9, 9 - max(0, int(x) // units)))


def time_bin(ticks, tau):
    """The right-open thresholds make an exact tau breach part of bin 1."""
    ticks = np.asarray(ticks)
    return np.where(
        ticks < 0, 5, np.searchsorted(np.array([1, 2, 4, 8]) * tau, ticks, side="right")
    ).astype(np.int8)


def outcome_codes(outcomes, tau):
    """[lane, field depth / relative first home tick / loss / probe depth]."""
    outcomes = np.asarray(outcomes)
    result = outcomes.astype(np.int8).copy()
    result[..., 1] = time_bin(outcomes[..., 1], tau)
    return result


def effect_labels(baseline, after, tau):
    before, after = outcome_codes(baseline, tau), outcome_codes(after, tau)
    delta = after.astype(np.int16) - before.astype(np.int16)
    return (delta + np.array([10, 5, 1, 10], np.int16)).astype(np.int8)


def baseline_features(outcomes, tau):
    values = np.asarray(outcomes, dtype=np.float32).copy()
    values[..., 0] /= 10
    values[..., 1] = np.where(values[..., 1] < 0, 1, np.minimum(values[..., 1] / (8 * tau), 1))
    values[..., 3] /= 10
    return values.reshape(*values.shape[:-2], 20)


def categorical_loss(logits, labels, *, censored=None):
    """-1 is unknown; a censored depth observes a lower bound, not an exact class."""
    logits = logits.float()
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    known = labels >= 0
    safe = labels.clamp_min(0)
    logp = F.log_softmax(logits, -1)
    exact = -logp.gather(-1, safe[..., None]).squeeze(-1)
    if censored is not None:
        known = known & ~(torch.as_tensor(censored, device=logits.device) & (labels == 0))
        support = torch.arange(logits.shape[-1], device=logits.device)
        tail = -torch.logsumexp(logp.masked_fill(support < safe[..., None], -torch.inf), -1)
        exact = torch.where(torch.as_tensor(censored, device=logits.device), tail, exact)
    return torch.where(known, exact, 0), known


def allowed_class_loss(logits, allowed):
    """Zero/full bitsets are uninformative; partial sets describe censored futures."""
    logits = logits.float()
    bits = torch.as_tensor(
        allowed if torch.is_tensor(allowed) else np.array(allowed),
        device=logits.device,
        dtype=torch.long,
    )
    full = (1 << logits.shape[-1]) - 1
    known = (bits != 0) & (bits != full)
    # Never evaluate logsumexp on an empty set, even in a masked-out branch.
    bits = torch.where(known, bits, full)
    support = torch.arange(logits.shape[-1], device=logits.device)
    mask = (bits[..., None] & (1 << support)) != 0
    logp = F.log_softmax(logits, -1)
    loss = -torch.logsumexp(logp.masked_fill(~mask, -torch.inf), -1)
    return torch.where(known, loss, 0), known


def threat_eligible(rows):
    if "threat_depth_allowed" in rows.dtype.names:
        d, t = rows["threat_depth_allowed"], rows["threat_time_allowed"]
        return (
            ((d != 0) & (d != 2047)).any(axis=(-1, -2))
            | ((t != 0) & (t != 63)).any(-1)
            | (rows["threat_loss"] >= 0).any(-1)
        )
    informative = (rows["threat_depth"] >= 0) & ~(
        rows["threat_censored"] & (rows["threat_depth"] == 0)
    )
    return (
        informative.any(axis=(-1, -2))
        | (rows["threat_time"] >= 0).any(-1)
        | (rows["threat_loss"] >= 0).any(-1)
    )


def threat_loss(
    logits,
    depths,
    censored,
    times,
    losses,
    *,
    decision_count=None,
    depth_allowed=None,
    time_allowed=None,
):
    """Normalize terms within lane, lanes within root, and eligible roots equally."""
    logits = logits.float()
    depth_logits = logits[..., :33].reshape(*logits.shape[:-1], 3, 11)
    if depth_allowed is None:
        dl, dm = categorical_loss(depth_logits, depths, censored=censored)
        tl, tm = categorical_loss(logits[..., 33:39], times)
    else:
        dl, dm = allowed_class_loss(depth_logits, depth_allowed)
        tl, tm = allowed_class_loss(logits[..., 33:39], time_allowed)
    target = torch.as_tensor(losses, device=logits.device, dtype=torch.float32)
    lm = target >= 0
    ll = F.binary_cross_entropy_with_logits(logits[..., 39], target.clamp_min(0), reduction="none")
    lane_count = dm.sum(-1) + tm + lm
    lane_loss = (dl.sum(-1) + tl + torch.where(lm, ll, 0)) / lane_count.clamp_min(1)
    lanes = lane_count > 0
    per_root = (lane_loss * lanes).sum(-1) / lanes.sum(-1).clamp_min(1)
    roots = lanes.any(-1)
    divisor = roots.sum().clamp_min(1) if decision_count is None else max(1, decision_count)
    return per_root.sum() / divisor


def effect_loss(logits, labels, *, root_count=None, absolute=False):
    """One root may have many candidate tokens; average candidates before roots."""
    losses = []
    for index, block in enumerate(effect_blocks(logits, absolute=absolute)):
        loss, known = categorical_loss(block, labels[..., index])
        losses.append(loss * known)
    return torch.stack(losses, -1).mean(-1)


def conditional_effect_loss(logits, labels):
    """Occurrence keeps natural frequencies; magnitude averages changed terms separately.

    Return per-token gate/conditional losses and conditional eligibility. Callers
    aggregate tokens within roots, never give roots with more alternatives more weight.
    Occurrence/nonzero separation adapts Deep Hurdle Networks (arXiv:2010.16040).
    """
    gates, conditionals, changed = [], [], []
    for term, block in enumerate(logits.float().split(EFFECT_SIZES, -1)):
        zero = block.shape[-1] // 2
        nonzero = torch.cat((block[..., :zero], block[..., zero + 1 :]), -1)
        target = labels[..., term].long()
        change = target != zero
        gate = torch.logsumexp(nonzero, -1) - block[..., zero]
        gates.append(F.binary_cross_entropy_with_logits(gate, change.float(), reduction="none"))
        index = target - (target > zero).long()
        error = (
            -F.log_softmax(nonzero, -1)
            .gather(-1, index.clamp_max(nonzero.shape[-1] - 1)[..., None])
            .squeeze(-1)
        )
        conditionals.append(torch.where(change, error, 0))
        changed.append(change)
    gate = torch.stack(gates, -1).mean(-1).mean(-1)
    changes = torch.stack(changed, -1)
    count = changes.sum(-1)
    lane = torch.stack(conditionals, -1).sum(-1) / count.clamp_min(1)
    valid = count > 0
    conditional = (lane * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return gate, conditional, valid.any(-1)


def effect_forecast(logits, baseline=None, *, absolute=False):
    """Derive post-action field/probe CDFs from learned outcome distributions."""
    if not absolute:
        baseline = torch.as_tensor(baseline, device=logits.device, dtype=torch.float32)
    result = []
    for term in (0, 3):
        block = effect_blocks(logits, absolute=absolute)[term]
        support = (
            torch.arange(11, device=logits.device)
            if absolute
            else torch.arange(-10, 11, device=logits.device)
        )
        after = support if absolute else baseline[..., term, None] + support
        p = block.softmax(-1)
        thresholds = torch.arange(9, 0, -1, device=logits.device)
        cells = (p[..., None, :] * (after[..., None, :] >= thresholds[:, None])).sum(-1)
        home = (p * (after >= 10)).sum(-1)
        if term == 0:
            loss_p = effect_blocks(logits, absolute=absolute)[2].softmax(-1)
            home = (
                loss_p[..., 1]
                if absolute
                else (
                    loss_p
                    * (baseline[..., 2, None] + torch.arange(-1, 2, device=logits.device) >= 1)
                ).sum(-1)
            )
        result.append(torch.cat((home[..., None], cells), -1))
    return torch.stack(result, -3)


def effect_summary(logits, *, absolute=False):
    summaries = []
    sizes = ABSOLUTE_SIZES if absolute else EFFECT_SIZES
    for classes, block in zip(sizes, effect_blocks(logits, absolute=absolute), strict=True):
        radius = classes - 1 if absolute else classes // 2
        support = (
            torch.arange(classes, device=logits.device)
            if absolute
            else torch.arange(-radius, radius + 1, device=logits.device)
        )
        summaries.append((block.softmax(-1) * support).sum(-1) / max(1, radius))
    return torch.stack(summaries, -1).flatten(-2)


def threat_summary(logits):
    depth = logits[..., :33].float().reshape(*logits.shape[:-1], 3, 11).softmax(-1)
    support = torch.arange(11, device=logits.device, dtype=torch.float32) / 10
    times = logits[..., 33:39].float().softmax(-1)
    return torch.cat(
        (
            (depth * support).sum(-1),
            depth[..., -1, 10, None],
            (1 - times[..., 5])[..., None],
            logits[..., 39, None].float().sigmoid(),
        ),
        -1,
    ).flatten(-2)


def penetration_map(depth_logits):
    """CDF in physical left-to-right order: home + every depth at least this column."""
    p = depth_logits.float().softmax(-1)
    tail = p.flip(-1).cumsum(-1).flip(-1)
    return tail[..., torch.arange(9, 0, -1, device=p.device)]
