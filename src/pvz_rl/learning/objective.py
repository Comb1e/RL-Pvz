"""Shared component targets, alternative-action losses and demonstration ranking."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.policy.sequential_q import action_groups, action_parts, outcome_weights

OBJECTIVE_PROTOCOL = "complete_return_sequential_tile_probe_v1"
COMPONENTS = (
    "terminal",
    "development",
    "invalid_plant_penalty",
    "empty_dig_penalty",
    "home_proximity",
    "victory_time",
)


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


def probe_loss(policy, observations, probes, counts, delta, *, predictions=None):
    """probes[..., :] = valid, proposal, target, initial acceptance."""
    valid = probes[..., 0].bool()
    source, slot = valid.nonzero(as_tuple=True)
    if not len(source):
        return next(policy.parameters()).sum() * 0
    chosen = probes[source, slot]
    branch, _ = action_parts(chosen[:, 1].long())
    target = chosen[:, 2]
    if predictions is None:
        tiles = policy.conditional_tiles(observations, branch, rows=source)
        _, positions = action_parts(chosen[:, 1].long())
        predictions = tiles.gather(1, positions[:, None]).flatten()
    weights = outcome_weights(
        counts,
        device=predictions.device,
        dtype=predictions.dtype,
        accepted_share=policy.cfg["training"]["objective"]["accepted_outcome_share"],
    )
    accepted = chosen[:, 3].long()
    error = F.huber_loss(predictions, target, reduction="none", delta=delta)
    return (error * weights[branch, accepted]).sum()


def species_consistency_loss(species, targets, actions, accepted, counts, *, accepted_share=0.5):
    """Non-wait branch regression toward detached conditional EMA maxima."""
    branches, _ = action_parts(actions)
    predictions = species.gather(1, branches[:, None]).flatten()
    errors = (predictions - targets.detach()).square()
    weights = outcome_weights(
        counts, device=species.device, dtype=species.dtype, accepted_share=accepted_share
    )
    return (errors * weights[action_groups(actions) - 1, accepted.long()]).sum()


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


def demonstration_rank_loss(
    policy, observations, actions, accepted, masks, counts, margin, *, species=None, tiles=None
):
    branch, tile = action_parts(actions)
    _, alternatives, geometry = ranking_counts(actions, accepted, masks)
    species = policy.species_values(observations) if species is None else species
    result = species.sum() * 0
    accuracy = species.new_zeros(4)
    for head, candidates in enumerate((alternatives, geometry)):
        active = accepted & candidates.any(-1) & ((branch != 0) if head else True)
        if not bool(active.any()):
            continue
        if head:
            source = active.nonzero(as_tuple=True)[0]
            values = (
                policy.conditional_tiles(observations, branch[active], rows=source)
                if tiles is None
                else tiles[active]
            )
            selected = values.gather(1, tile[active, None])
        else:
            values = species[active]
            selected = values.gather(1, branch[active, None])
        mask = candidates[active]
        losses = (F.softplus(margin - selected + values) * mask).sum(-1) / mask.sum(-1)
        denominator = counts[head].to(species.device)
        result = result + (losses / denominator[branch[active]]).sum() / (denominator > 0).sum()
        accuracy[head * 2] = ((selected > values) & mask).sum()
        accuracy[head * 2 + 1] = mask.sum()
    return result, accuracy
