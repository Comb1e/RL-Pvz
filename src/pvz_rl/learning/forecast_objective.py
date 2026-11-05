"""Shared-forward categorical fitting of exact post-action field/probe outcomes."""

import numpy as np
import torch

from pvz_rl.config import effect_prediction_weight, forecast_forks_enabled
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_labels import effect_blocks, effect_loss, outcome_codes
from pvz_rl.learning.forecast_metrics import categorical_histogram, reset_calibration


def forecast_eligible(rows):
    return rows["null_target_valid"] | rows["effect_valid"].any(-1)


def forecast_auxiliary(
    policy,
    observations,
    rows,
    memory,
    *,
    features=None,
    feature_rows=None,
    decision_count=None,
    stage="normal",
):
    count = len(rows) if decision_count is None else decision_count
    if forecast_forks_enabled(policy.cfg, stage):
        loss, metric_tokens, tokens = _effect_objective(
            policy, observations, rows, memory, features, feature_rows, count
        )
    else:
        loss, metric_tokens = memory[0].new_zeros((), dtype=torch.float32), []
        tokens = loss.new_zeros(len(rows))
    weight = effect_prediction_weight(policy.cfg, stage) * getattr(policy, "auxiliary_scale", 1)
    policy._forecast_components = (weight * loss,)
    if not hasattr(policy, "_forecast_calibration"):
        reset_calibration(policy)
    brier, terms, correct = (loss.new_zeros(()) for _ in range(3))
    for logits, target in metric_tokens:
        for term, block in enumerate(effect_blocks(logits)):
            label = target[..., term]
            known = label >= 0
            p = block.detach().float().softmax(-1)
            categorical_histogram(
                policy._forecast_calibration["effect"], p, label, known, term=term
            )
            expected = torch.nn.functional.one_hot(label.long().clamp_min(0), block.shape[-1])
            brier += ((p - expected).square().sum(-1) * known).sum()
            terms += known.sum()
            correct += ((block.argmax(-1) == label) & known).sum()
    metrics = torch.stack((loss.detach(), brier, terms, (tokens > 0).sum(), correct)).double()
    return weight * loss, metrics


def _token_loss(logits, targets):
    lanes = (targets >= 0).any(-1)
    per_lane = effect_loss(logits, targets)
    return per_lane.sum(-1) / lanes.sum(-1).clamp_min(1)


def _effect_objective(policy, observations, rows, memory, features, feature_rows, root_count):
    """Average known terms, lanes, valid tokens and decision roots equally."""
    device = policy.device
    per_root = memory[0].new_zeros(len(rows), dtype=torch.float32)
    null_rows = np.flatnonzero(rows["null_target_valid"])
    tokens = torch.as_tensor(rows["null_target_valid"].astype(np.float32), device=device)
    metric_tokens = []
    if len(null_rows):
        at = torch.as_tensor(null_rows, device=device)
        null_logits = policy.effect_values(policy.null_features((memory[0][at], memory[1][at])))
        target = torch.as_tensor(
            outcome_codes(
                np.array(rows["baseline_outcomes"][null_rows]), reference_tau(policy.layout.rules)
            ),
            device=device,
            dtype=torch.long,
        )
        per_root = per_root.index_add(0, at, _token_loss(null_logits, target))
        metric_tokens.append((null_logits, target))
    root, token = np.nonzero(rows["effect_valid"] & (rows["effect_actions"] != 0))
    if len(root):
        actions = rows["effect_actions"][root, token].astype(np.int64)
        branches, tiles = (actions - 1) // 45 + 1, (actions - 1) % 45
        # Illegal tiles in an available branch retain their own tile features.
        unavailable = (token == 0) & ~rows["branch_usable"][root] & rows["null_target_valid"][root]
        lookup = np.full((len(rows), 10), -1, np.int64)
        banks = []
        if features is not None:
            at = np.asarray(feature_rows)
            lookup[at, (rows["action"][at].astype(np.int64) - 1) // 45 + 1] = np.arange(len(at))
            banks.append(features)
        legal = np.flatnonzero(~unavailable)
        missing = np.unique(np.stack((root[legal], branches[legal]), -1), axis=0)
        missing = missing[lookup[missing[:, 0], missing[:, 1]] < 0]
        if len(missing):
            offset = 0 if features is None else len(features)
            lookup[missing[:, 0], missing[:, 1]] = offset + np.arange(len(missing))
            banks.append(
                policy.tile_features(
                    observations,
                    torch.as_tensor(missing[:, 1], device=device),
                    rows=torch.as_tensor(missing[:, 0], device=device),
                    memory=memory,
                )
            )
        gathered = memory[0].new_zeros((len(root), policy.cfg["policy"]["shared_width"]))
        if len(legal):
            bank = banks[0] if len(banks) == 1 else torch.cat(banks)
            selected = bank[
                torch.as_tensor(lookup[root[legal], branches[legal]], device=device),
                torch.as_tensor(tiles[legal], device=device),
            ]
            gathered = gathered.index_copy(0, torch.as_tensor(legal, device=device), selected)
        rejected = np.flatnonzero(unavailable)
        if len(rejected):
            at = torch.as_tensor(root[rejected], device=device)
            selected = policy.null_features(
                (memory[0][at], memory[1][at]), torch.as_tensor(branches[rejected], device=device)
            )
            gathered = gathered.index_copy(0, torch.as_tensor(rejected, device=device), selected)
        predictions = policy.effect_values(gathered)
        targets = torch.as_tensor(
            np.array(rows["effect_labels"][root, token]), device=device, dtype=torch.long
        )
        at = torch.as_tensor(root, device=device)
        per_root = per_root.index_add(0, at, _token_loss(predictions, targets))
        tokens = tokens.index_add(0, at, tokens.new_ones(len(root)))
        metric_tokens.append((predictions, targets))
    return (per_root / tokens.clamp_min(1)).sum() / max(1, int(root_count)), metric_tokens, tokens


def gradient_norm_ratios(policy):
    primary = getattr(policy, "_primary_graph", None)
    if primary is None:
        return {}
    trunk = tuple(policy.board_encoder.parameters())
    values = []
    for loss in (primary, *policy._forecast_components):
        if not loss.requires_grad:
            values.append(loss.new_zeros(()))
            continue
        grads = torch.autograd.grad(loss, trunk, retain_graph=True, allow_unused=True)
        squared = [g.detach().float().square().sum() for g in grads if g is not None]
        values.append(torch.stack(squared).sum().sqrt() if squared else loss.new_zeros(()))
    q, effect = torch.stack(values).cpu().tolist()
    return dict(
        primary_trunk_norm=q, effect_trunk_norm=effect, effect_to_primary=effect / q if q else None
    )
