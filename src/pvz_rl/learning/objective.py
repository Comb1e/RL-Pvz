"""Shared component targets, alternative-action losses and demonstration ranking."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.policy.sequential_q import (
    action_groups,
    action_parts,
    candidate_maxima,
    greedy_choice,
    outcome_weights,
    tile_candidates,
)

OBJECTIVE_PROTOCOL = "complete_return_action_global_v5"
# These are the only reward components that train the policy.  The simulator
# still publishes the older diagnostic fields (combat_value, home_proximity,
# victory_time, and so on) for reports and replay verification. Cohort-ranked
# completion time is applied to the terminal component during finalization.
COMPONENTS = (
    "global_terminal",
    "global_asset_value",
    "action_damage",
    "action_blocking",
    "action_sun",
    "action_mower",
)


def components(parts):
    return np.asarray([parts.get(k, 0.0) for k in COMPONENTS], dtype=np.float64)


def training_rewards(values, cfg):
    """Convert raw global/action values into the scalar fitting reward.

    The reward ledger stores physical terms in sun-equivalent units.  The
    existing progress/value conversion and dense multiplier are applied once
    here. Terminal outcomes and mower penalties are already in final units.
    """
    value = np.asarray(values, dtype=np.float64)
    multiplier = cfg["training"].get("objective", {}).get("dense_multiplier", 1.0)
    scale = cfg["reward"]["progress_weight"] / cfg["reward"]["value_scale"]
    physical = value[..., 1:-1].sum(axis=-1)
    return value[..., 0] + value[..., -1] + multiplier * scale * physical


def raw_rewards(values, cfg):
    """Compute the immediate un-densified reward stored by the simulator."""
    value = np.asarray(values, dtype=np.float64)
    scale = cfg["reward"]["progress_weight"] / cfg["reward"]["value_scale"]
    return value[..., 0] + value[..., -1] + scale * value[..., 1:-1].sum(axis=-1)


def victory_time(duration, median, won, weight):
    """Legacy report-only smooth time diagnostic.

    Training uses :func:`endgame_terminal_rewards`; this helper remains for
    archived report readers and independent diagnostics.
    """
    duration = np.asarray(duration, dtype=np.float64)
    denominator = duration + median
    ratio = np.divide(
        median - duration, denominator, out=np.zeros_like(duration), where=denominator != 0
    )
    return np.where(won, weight * ratio, 0.0)


def endgame_terminal_rewards(durations, won, *, win_reward=1.0, loss_penalty=1.0, weight=0.3):
    """Return signed-quadratic ranked rewards and applied time adjustments.

    Completion time is used only once, on the terminal transition.  The first
    game in a cohort receives ``+weight`` when it wins and the last receives
    ``-weight``.  A loss keeps the base ``-loss_penalty`` until the median and
    becomes more negative after it, reaching ``-loss_penalty-weight`` for the
    last-ranked game.  Equal-duration games share their average rank.
    """
    durations = np.asarray(durations, dtype=np.float64)
    won = np.asarray(won, dtype=bool)
    if durations.ndim != 1 or won.shape != durations.shape:
        raise ValueError("durations and won must be one-dimensional arrays of equal length")
    if not len(durations):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if not np.isfinite(durations).all():
        raise ValueError("durations must be finite")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("endgame time weight must be finite and nonnegative")
    if len(durations) == 1:
        adjustment = np.zeros(1, dtype=np.float64)
    else:
        order = np.argsort(durations, kind="stable")
        sorted_durations = durations[order]
        positions = np.arange(len(durations), dtype=np.float64)
        ranks = np.empty(len(durations), dtype=np.float64)
        start = 0
        while start < len(durations):
            stop = start + 1
            while stop < len(durations) and sorted_durations[stop] == sorted_durations[start]:
                stop += 1
            ranks[order[start:stop]] = positions[start:stop].mean()
            start = stop
        normalized = 1.0 - 2.0 * ranks / (len(durations) - 1)
        adjustment = float(weight) * normalized * np.abs(normalized)
    # Early losses receive no adjustment; reports describe the applied value.
    adjustment = np.where(won, adjustment, np.minimum(adjustment, 0.0))
    terminal = np.where(won, float(win_reward), -float(loss_penalty)) + adjustment
    return terminal, adjustment


def finalize_episode_metrics(rows, cfg, *, median=None):
    """Finalize one cohort idempotently, including standalone evaluation batches."""
    if not rows:
        return 0.0
    durations = np.asarray([r["simulated_seconds"] for r in rows], dtype=np.float64)
    won = np.asarray([bool(r["win"]) for r in rows], dtype=bool)
    terminal, adjustment = endgame_terminal_rewards(
        durations,
        won,
        win_reward=cfg["reward"]["win_reward"],
        loss_penalty=cfg["reward"]["loss_penalty"],
        weight=cfg["reward"].get("win_time_weight", 0.3),
    )
    for row, value, delta in zip(rows, terminal, adjustment, strict=True):
        baseline = cfg["reward"]["win_reward"] if row["win"] else -cfg["reward"]["loss_penalty"]
        change = float(value - row.get("terminal_reward", baseline))
        for key in ("terminal", "global_terminal"):
            if key in row:
                row[key] += change
        for key in ("return", "discounted_return", "discounted_outcome_return"):
            if key in row:
                row[key] += change
        row.update(
            terminal_reward=float(value),
            endgame_time_adjustment=float(delta),
            # Retain the field for report readers that understand the old
            # provisional protocol; the actual terminal adjustment is now in
            # terminal_reward and is part of the learning target.
            victory_time=0.0,
            time_reward_pending=False,
            duration_median_seconds=float(np.median(durations) if median is None else median),
            duration_reference_count=len(rows),
        )
    return float(np.median(durations) if median is None else median)


@torch.no_grad()
def species_consistency_targets(online_tiles, ema_tiles, masks, branches):
    """Select with online geometry values and evaluate with the frozen EMA."""
    selected = (
        online_tiles.detach().argmax(-1)
        if masks is None
        else greedy_choice(online_tiles.detach(), tile_candidates(masks, branches))
    )
    targets = ema_tiles.detach().gather(1, selected[:, None]).flatten()
    maxima = (
        ema_tiles.detach().max(-1).values
        if masks is None
        else candidate_maxima(ema_tiles.detach(), masks, branches)
    )
    gaps = maxima - targets
    return targets, selected, gaps


def consistency_diagnostics(loss, totals):
    """Publish pass totals only; undefined non-wait statistics remain absent."""
    absolute, saturated, gaps, count = totals
    return {
        "species_consistency_loss": loss if count else None,
        "species_consistency_mae": absolute / count if count else None,
        "species_consistency_saturation_fraction": saturated / count if count else None,
        "species_selection_evaluation_gap": gaps / count if count else None,
    }


def species_consistency_loss(
    species, targets, actions, accepted, counts, *, delta=1.0, accepted_share=0.5, weights=None
):
    """Balanced MSE-matched Huber toward detached double-estimator targets."""
    branches, _ = action_parts(actions)
    predictions = species.gather(1, branches[:, None]).flatten()
    errors = 2 * F.huber_loss(predictions, targets.detach(), reduction="none", delta=delta)
    weights = (
        outcome_weights(
            counts, device=species.device, dtype=species.dtype, accepted_share=accepted_share
        )
        if weights is None
        else weights
    )
    return (errors * weights[action_groups(actions) - 1, accepted.long()]).sum()


def ranking_counts(actions, accepted, masks):
    branch, tile = action_parts(actions)
    active = masks.any(-1)
    branch_legal = torch.ones((len(actions), 10), dtype=torch.bool, device=actions.device)
    branch_legal = branch_legal & active[:, None]
    alternatives = branch_legal.clone()
    alternatives.scatter_(1, branch[:, None], False)
    first = accepted & alternatives.any(-1)
    geometry = torch.ones((len(actions), 45), dtype=torch.bool, device=actions.device)
    geometry &= active[:, None]
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
