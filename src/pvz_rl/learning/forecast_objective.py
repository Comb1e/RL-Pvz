"""Shared-forward categorical fitting of exact post-action field/probe outcomes."""

import numpy as np
import torch
from torch._functorch import config as functorch_config

from pvz_rl.config import forecast_forks_enabled, outcome_prediction_weights
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_labels import effect_blocks, effect_loss_components, outcome_codes
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
        components, metric_tokens, tokens = _effect_objective(
            policy, observations, rows, memory, features, feature_rows, count
        )
    else:
        components, metric_tokens = memory[0].new_zeros(2, dtype=torch.float32), []
        tokens = components.new_zeros(len(rows))
    loss = components.sum()
    weights = components.new_tensor(outcome_prediction_weights(policy.cfg, stage))
    weighted = components * weights * getattr(policy, "auxiliary_scale", 1)
    auxiliary = weighted.sum()
    policy._forecast_components = (auxiliary,)
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
    metrics = torch.cat(
        (torch.stack((loss.detach(), brier, terms, (tokens > 0).sum(), correct)), weighted.detach())
    ).double()
    return auxiliary, metrics


def _token_loss_components(logits, targets):
    lanes = (targets >= 0).any(-1)
    per_lane = effect_loss_components(logits, targets)
    return per_lane.sum(-2) / lanes.sum(-1).clamp_min(1)[..., None]


def scene_outcome_loss(logits, targets):
    """Unweighted categorical loss for each fully labeled scene."""
    return _token_loss_components(logits, targets).sum(-1)


def _effect_objective(policy, observations, rows, memory, features, feature_rows, root_count):
    """Average known terms, lanes, valid tokens and decision roots equally.

    Each labeled action reads its action-decoder vector and its chosen tile-decoder
    cell, then cross-attends the full board memory. An explicit Wait uses the
    learned empty tile vector.
    """
    del observations, features, feature_rows
    device = policy.device
    per_root = memory[0].new_zeros((len(rows), 2), dtype=torch.float32)
    has_null = rows["null_target_valid"]
    null_rows = np.flatnonzero(has_null)
    root, token = np.nonzero(
        rows["effect_valid"] & ((rows["effect_actions"] != 0) | ~has_null[:, None])
    )
    actions = np.where(
        token == 0, rows["executed_action"][root], rows["effect_actions"][root, token]
    )
    tokens = torch.as_tensor(has_null.astype(np.float32), device=device)
    metric_tokens = []
    if len(null_rows) or len(root):
        needed = np.unique(np.concatenate((null_rows, root)))
        at = torch.as_tensor(needed, device=device, dtype=torch.long)
        subset = (memory[0][at], memory[1][at])
        vectors = policy.action_vectors(subset)
        if len(null_rows):
            pick = torch.as_tensor(np.searchsorted(needed, null_rows), device=device)
            null_logits = policy.forecast_logits(
                vectors[pick],
                torch.zeros(len(null_rows), dtype=torch.long, device=device),
                (subset[0][pick], subset[1][pick]),
            )
            target = torch.as_tensor(
                outcome_codes(
                    np.array(rows["baseline_outcomes"][null_rows]),
                    reference_tau(policy.layout.rules),
                ),
                device=device,
                dtype=torch.long,
            )
            per_root = per_root.index_add(
                0,
                torch.as_tensor(null_rows, device=device),
                _token_loss_components(null_logits, target),
            )
            metric_tokens.append((null_logits, target))
        if len(root):
            labeled = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=device)
            pick = torch.as_tensor(np.searchsorted(needed, root), device=device)
            predictions = policy.forecast_logits(
                vectors[pick], labeled, (subset[0][pick], subset[1][pick])
            )
            targets = torch.as_tensor(
                np.array(rows["effect_labels"][root, token]), device=device, dtype=torch.long
            )
            root_at = torch.as_tensor(root, device=device)
            per_root = per_root.index_add(0, root_at, _token_loss_components(predictions, targets))
            tokens = tokens.index_add(0, root_at, tokens.new_ones(len(root)))
            metric_tokens.append((predictions, targets))
    loss = (per_root / tokens.clamp_min(1)[:, None]).sum(0) / max(1, int(root_count))
    return loss, metric_tokens, tokens


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
        # Persistent compiled shapes reject retained backwards once their graph was
        # first run with donated buffers; owned CUDA-graph captures donate nothing.
        with functorch_config.patch(donated_buffer=False):
            grads = torch.autograd.grad(loss, trunk, retain_graph=True, allow_unused=True)
        squared = [g.detach().float().square().sum() for g in grads if g is not None]
        values.append(torch.stack(squared).sum().sqrt() if squared else loss.new_zeros(()))
    q, effect = torch.stack(values).cpu().tolist()
    return dict(
        primary_trunk_norm=q, effect_trunk_norm=effect, effect_to_primary=effect / q if q else None
    )
