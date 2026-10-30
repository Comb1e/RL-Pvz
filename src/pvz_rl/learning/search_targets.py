"""Confirmed same-board supervision, independent of trajectory outcome targets."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.objective import endgame_terminal_rewards


def frozen_time_adjustment(reference, duration, won, weight):
    if not reference:
        raise ValueError("A search duration reference must be complete and nonempty")
    _, delta = endgame_terminal_rewards(
        [*reference, duration], [False] * len(reference) + [won], weight=weight
    )
    return float(delta[-1])


def paired_targets(samples):
    samples = np.asarray(samples, np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2 or not np.isfinite(samples).all():
        raise ValueError("Confirmation needs finite [actions, worlds] samples")
    means = samples.mean(1)
    differences = samples[:, None, :] - samples[None, :, :]
    se = differences.std(-1, ddof=1) / np.sqrt(samples.shape[1])
    delta = differences.mean(-1)
    weights = np.divide(delta**2, delta**2 + se**2 + 1e-8)
    weights[(se == 0)] = 1.0
    return means, delta, se, weights


def _head_loss(values, targets, pair_weights, invalid, floor, settings):
    """One root/head has unit weight irrespective of its candidate/invalid count."""
    zero = values.sum() * 0
    valid = torch.isfinite(targets)
    q, y = values[valid], targets[valid]
    calibration = (2 * F.huber_loss(q, y, delta=settings["huber_delta"])) if q.numel() else zero
    difference = zero
    if q.numel() > 1:
        upper = torch.triu(torch.ones_like(pair_weights, dtype=torch.bool), diagonal=1)
        error = 2 * F.huber_loss(
            q[:, None] - q[None, :],
            y[:, None] - y[None, :],
            delta=settings["huber_delta"],
            reduction="none",
        )
        difference = (error[upper] * pair_weights[upper]).mean()
    infeasible = F.relu(values[invalid] - floor).square().mean() if invalid.any() else zero
    total = (
        settings["value_weight"] * calibration
        + settings["difference_weight"] * difference
        + settings["feasibility_weight"] * infeasible
    )
    return total, torch.stack((calibration.detach(), difference.detach(), infeasible.detach()))


def teacher_loss(policy, records, *, stage, total_roots, settings, invalid_offset):
    """Use the same online masters, whole-cohort accumulation and Adam commit path."""
    obs = collate_observations([r["observation"] for r in records], policy.device)
    device = policy.device
    tile_requests = [(i, b) for i, r in enumerate(records) for b in sorted(r["tiles"])]
    if tile_requests:
        rows, branches = zip(*tile_requests)
        tables = policy.fitting_tiles(
            obs, torch.tensor(branches, device=device), rows=torch.tensor(rows, device=device)
        ).float()
    else:
        tables = None
    species = policy.species_values(obs).float() if stage == "joint" else None
    root_losses = []
    metrics = []
    lookup = {pair: j for j, pair in enumerate(tile_requests)}
    for i, record in enumerate(records):
        heads = []
        diagnostics = []
        for branch, label in record["tiles"].items():
            q = tables[lookup[i, branch]]
            y = torch.tensor(label["targets"], device=device, dtype=torch.float32)
            invalid = torch.tensor(label["invalid"], device=device, dtype=torch.bool)
            weights = torch.tensor(label["pair_weights"], device=device, dtype=torch.float32)
            loss, detail = _head_loss(
                q, y, weights, invalid, q.new_tensor(label["reference"] + invalid_offset), settings
            )
            heads.append(loss)
            diagnostics.append(detail)
        if species is not None and record["species"] is not None:
            label = record["species"]
            loss, detail = _head_loss(
                species[i],
                torch.tensor(label["targets"], device=device, dtype=torch.float32),
                torch.tensor(label["pair_weights"], device=device, dtype=torch.float32),
                torch.tensor(label["invalid"], device=device, dtype=torch.bool),
                species.new_tensor(label["reference"] + invalid_offset),
                settings,
            )
            heads.append(loss)
            diagnostics.append(detail)
        if heads:
            root_losses.append(torch.stack(heads).mean())
            metrics.append(torch.stack(diagnostics).mean(0))
    if not root_losses:
        # Wait-only roots cannot supervise a frozen species head.
        zero = torch.zeros((), device=policy.device, requires_grad=True)
        return zero, zero.new_zeros(3)
    return torch.stack(root_losses).sum() / max(1, total_roots), torch.stack(metrics).sum(0) / max(
        1, total_roots
    )


def make_teacher_record(
    observation, candidates, samples, legal, representatives, *, peashooter, branch_usable=None
):
    means, differences, standard_errors, weights = paired_targets(samples)
    legal = set(map(int, legal))
    groups = {}
    sampled = {(a - 1) // 45 + 1 for a in candidates if a}
    usable = np.ones(10, bool) if branch_usable is None else np.asarray(branch_usable, bool)
    branches = sampled | {
        b
        for b in range(1, 10)
        if usable[b]
        and any(1 + (b - 1) * 45 + t not in legal for t in range(45))
        and (not peashooter or b == 2)
    }
    for b in sorted(branches):
        indexed = [
            (j, (a - 1) % 45) for j, a in enumerate(candidates) if a and (a - 1) // 45 + 1 == b
        ]
        target = np.full(45, np.nan)
        for j, tile in indexed:
            target[tile] = means[j]
        ids = [j for j, tile in sorted(indexed, key=lambda p: p[1])]
        groups[b] = dict(
            targets=target,
            invalid=np.array([1 + (b - 1) * 45 + t not in legal for t in range(45)]),
            pair_weights=weights[np.ix_(ids, ids)],
            reference=float(min((means[j] for j, _ in indexed), default=means[0])),
        )
    species = None
    if not peashooter:
        target = np.full(10, np.nan)
        selected = {}
        for b, j in representatives.items():
            target[b] = means[j]
            selected[b] = j
        ids = [selected[b] for b in sorted(selected)]
        species = dict(
            targets=target,
            pair_weights=weights[np.ix_(ids, ids)],
            invalid=~usable,
            reference=float(means.min()),
        )
    return dict(
        observation=observation,
        tiles=groups,
        species=species,
        confirmed_samples=np.asarray(samples),
        paired_differences=differences,
        paired_standard_errors=standard_errors,
        candidates=list(candidates),
    )
