"""Shared component targets, alternative-action losses and demonstration ranking."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.policy.sequential_q import action_parts

OBJECTIVE_PROTOCOL = "complete_return_probe_v2"
COMPONENTS = (
    "terminal",
    "development",
    "invalid_plant_penalty",
    "empty_dig_penalty",
    "home_proximity",
    "victory_time",
)
MAX_PROBES = 4


def components(parts):
    return np.asarray([parts.get(k, 0.0) for k in COMPONENTS], dtype=np.float64)


def training_rewards(values, cfg):
    """Retain FP64 arithmetic; scale development exactly once."""
    value = np.asarray(values, dtype=np.float64)
    multiplier = cfg["training"].get("objective", {}).get("dense_multiplier", 1.0)
    return value.sum(axis=-1) + (multiplier - 1) * value[..., 1]


def victory_time(duration, median, won, weight):
    duration = np.asarray(duration, dtype=np.float64)
    denominator = duration + median
    ratio = np.divide(
        median - duration, denominator, out=np.zeros_like(duration), where=denominator != 0
    )
    return np.where(won, weight * ratio, 0.0)


def finalize_episode_metrics(rows, cfg, *, median=None):
    """Finalize one cohort idempotently, including standalone evaluation batches."""
    if not rows:
        return 0.0
    median = (
        float(np.median([r["simulated_seconds"] for r in rows]))
        if median is None
        else float(median)
    )
    for row in rows:
        old = row.get("victory_time", 0.0)
        value = float(
            victory_time(
                row["simulated_seconds"],
                median,
                bool(row["win"]),
                cfg["reward"].get("win_time_weight", 0),
            )
        )
        for key in ("return", "discounted_return"):
            if key in row:
                row[key] += value - old
        row.update(
            victory_time=value,
            time_reward_pending=False,
            duration_median_seconds=median,
            duration_reference_count=len(rows),
        )
    return median


def probe_loss(policy, q, tiles, context, probes, counts, delta):
    """probes[..., :] = valid, branch-role, proposal, finalized target."""
    valid = probes[..., 0].bool()
    source, slot = valid.nonzero(as_tuple=True)
    if not len(source):
        return q.sum() * 0
    chosen = probes[source, slot]
    branch, location = action_parts(chosen[:, 2].long())
    target = chosen[:, 3]
    branch_role = chosen[:, 1].bool()
    # Cohort denominators are host metadata; inspecting them must not wait on
    # active device work in every recurrent chunk.
    represented = np.count_nonzero(np.asarray(counts), axis=1)
    count = torch.as_tensor(counts, device=q.device, dtype=q.dtype)
    terms = []
    if represented[0]:
        error = F.huber_loss(q[source, branch], target, reduction="none", delta=delta)
        terms.append(
            (error * branch_role / count[0, branch].clamp_min(1)).sum() / int(represented[0])
        )
    nonwait = branch != 0
    if represented[1] and len(source[nonwait]):
        predicted = (
            policy.tile_values(tiles[source[nonwait]], context[source[nonwait]], branch[nonwait])
            .gather(1, location[nonwait, None])
            .flatten()
        )
        error = F.huber_loss(predicted, target[nonwait], reduction="none", delta=delta)
        terms.append((error / count[1, branch[nonwait]].clamp_min(1)).sum() / int(represented[1]))
    return sum(terms, q.sum() * 0) / max(1, sum(bool(value) for value in represented))


def ranking_counts(actions, accepted, masks):
    branch, tile = action_parts(actions)
    branch_legal = torch.cat((masks[:, :1], masks[:, 1:].reshape(-1, 9, 45).any(-1)), -1)
    alternatives = branch_legal.clone()
    alternatives.scatter_(1, branch[:, None], False)
    first = accepted & alternatives.any(-1)
    geometry = (
        masks[:, 1:]
        .reshape(-1, 9, 45)[
            torch.arange(len(actions), device=actions.device), (branch - 1).clamp_min(0)
        ]
        .clone()
    )
    geometry.scatter_(1, tile[:, None], False)
    second = accepted & (branch != 0) & geometry.any(-1)
    return (
        torch.stack([torch.bincount(branch[x], minlength=10) for x in (first, second)]),
        alternatives,
        geometry,
    )


def demonstration_rank_loss(policy, q, tiles, context, actions, accepted, masks, counts, margin):
    branch, tile = action_parts(actions)
    _, alternatives, geometry = ranking_counts(actions, accepted, masks)
    result = q.sum() * 0
    accuracy = q.new_zeros(4)
    for head, candidates in enumerate((alternatives, geometry)):
        active = accepted & candidates.any(-1) & ((branch != 0) if head else True)
        if not bool(active.any()):
            continue
        values = (
            policy.tile_values(tiles[active], context[active], branch[active])
            if head
            else q[active]
        )
        indices = tile[active] if head else branch[active]
        selected = values.gather(1, indices[:, None])
        mask = candidates[active]
        losses = (F.softplus(margin - selected + values) * mask).sum(-1) / mask.sum(-1)
        denominator = counts[head].to(q.device)
        result = result + (losses / denominator[branch[active]]).sum() / (denominator > 0).sum()
        accuracy[head * 2] = ((selected > values) & mask).sum()
        accuracy[head * 2 + 1] = mask.sum()
    return result, accuracy
