"""Shared-forward categorical fitting of the pre-action field, breach, and probe."""

import numpy as np
import torch
from torch._functorch import config as functorch_config

from pvz_rl.config import forecast_forks_enabled, outcome_prediction_weights
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.forecast_labels import (
    effect_blocks,
    effect_loss_components,
    outcome_codes,
    scene_class_matches,
)
from pvz_rl.learning.forecast_metrics import categorical_histogram, reset_calibration


def forecast_eligible(rows):
    return rows["null_target_valid"]


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
    scene_hits, scene_total = (loss.new_zeros(()) for _ in range(2))
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
        # Same comparison validation uses. The count is a display statistic.
        matched = scene_class_matches(logits, target)
        scene_hits = scene_hits + matched.all(-1).all(-1).sum()
        scene_total = scene_total + matched.new_tensor(float(matched.shape[0]))
    metrics = torch.cat(
        (
            torch.stack((loss.detach(), brier, terms, (tokens > 0).sum(), correct)),
            weighted.detach(),
            torch.stack((scene_hits.detach(), scene_total.detach())),
        )
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
    """One pre-action forecast per labeled decision, from the encode just run."""
    del observations, memory, features, feature_rows
    logits = getattr(policy, "_effect_logits", None)
    if logits is None or logits.shape[0] != len(rows):
        raise RuntimeError("Pre-action logits must come from this decision batch")
    labeled = np.flatnonzero(rows["null_target_valid"])
    per_root = logits.new_zeros((len(rows), 2), dtype=torch.float32)
    tokens = logits.new_zeros(len(rows))
    metric_tokens = []
    if len(labeled):
        at = torch.as_tensor(labeled, device=logits.device, dtype=torch.long)
        target = torch.as_tensor(
            outcome_codes(
                np.array(rows["baseline_outcomes"][labeled]),
                reference_tau(policy.layout.rules),
            ),
            device=logits.device,
            dtype=torch.long,
        )
        chosen = logits.index_select(0, at)
        per_root = per_root.index_add(0, at, _token_loss_components(chosen, target))
        tokens = tokens.index_add(0, at, tokens.new_ones(len(labeled)))
        metric_tokens.append((chosen, target))
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
