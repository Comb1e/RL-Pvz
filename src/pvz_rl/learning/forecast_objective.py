"""Shared-forward categorical threat and signed causal-effect supervision."""

import numpy as np
import torch

from pvz_rl.config import effect_prediction_weight, forecast_forks_enabled, threat_prediction_weight
from pvz_rl.learning.forecast_labels import (
    EFFECT_SIZES,
    conditional_effect_loss,
    effect_loss,
    threat_eligible,
    threat_loss,
)
from pvz_rl.learning.forecast_metrics import categorical_histogram, reset_calibration


def forecast_eligible(rows):
    return rows["reach_valid"] & (threat_eligible(rows) | rows["effect_valid"].any(-1))


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
    loss_weights=None,
):
    counts = (len(rows), len(rows)) if decision_count is None else decision_count
    threats = policy.threat_values(memory=memory)
    tloss = threat_loss(
        threats,
        np.array(rows["threat_depth"]),
        np.array(rows["threat_censored"]),
        np.array(rows["threat_time"]),
        np.array(rows["threat_loss"]),
        decision_count=counts[0],
        **(
            dict(
                depth_allowed=np.array(rows["threat_depth_allowed"]),
                time_allowed=np.array(rows["threat_time_allowed"]),
            )
            if "threat_depth_allowed" in rows.dtype.names
            else {}
        ),
    )
    enabled = forecast_forks_enabled(policy.cfg, stage) or (
        loss_weights is not None and loss_weights[1] > 0
    )
    if enabled:
        eloss, metric_tokens, tokens = _effect_objective(
            policy, observations, rows, memory, features, feature_rows, counts[1]
        )
    else:
        eloss, metric_tokens = threats.new_zeros((), dtype=torch.float32), []
        tokens = threats.new_zeros(len(rows), dtype=torch.float32)
    scale = getattr(policy, "auxiliary_scale", 1.0)
    tw, ew = (
        threat_prediction_weight(policy.cfg, stage) * scale,
        effect_prediction_weight(policy.cfg, stage) * scale,
    )
    if loss_weights is not None:
        tw, ew = loss_weights
    policy._forecast_components = (tw * tloss, ew * eloss)
    return _forecast_metrics(policy, rows, threats, tloss, eloss, metric_tokens, tokens, tw, ew)


def _effect_objective(policy, observations, rows, memory, features, feature_rows, root_count):
    device = policy.device
    settings = policy.cfg["training"]["forecasts"]
    factored = settings.get("effect_objective", "categorical_v1") == "change_conditional_v2"
    zero = torch.tensor([10, 5, 1, 10], device=device)
    null = policy.null_features(memory)
    null_logits = policy.effect_values(null)
    null_targets = zero.expand(len(rows), 5, 4)
    per_root = None if factored else effect_loss(null_logits, null_targets).mean(-1)
    tokens = torch.ones(len(rows), device=device)
    cache = {}
    if features is not None:
        for i, row in enumerate(np.asarray(feature_rows)):
            action = int(rows["action"][row])
            cache[int(row), (action - 1) // 45 + 1] = features[i]
    requests = []
    for root, row in enumerate(rows):
        for token in np.flatnonzero(row["effect_valid"]):
            action = int(row["effect_actions"][token])
            if action == 0:
                continue  # the single null reference was already included
            branch, tile = (action - 1) // 45 + 1, (action - 1) % 45
            unavailable = token == 0 and not row["branch_usable"]
            requests.append((root, token, branch, tile, unavailable))
    missing = sorted({(r, b) for r, _, b, _, u in requests if not u and (r, b) not in cache})
    if missing:
        root_index = torch.tensor([r for r, _ in missing], device=device)
        branches = torch.tensor([b for _, b in missing], device=device)
        decoded = policy.tile_features(observations, branches, rows=root_index, memory=memory)
        for key, feature in zip(missing, decoded, strict=True):
            cache[key] = feature
    if requests:
        gathered = []
        for root, _, branch, tile, unavailable in requests:
            if unavailable:
                branch_tensor = torch.tensor([branch], device=device)
                feature = policy.null_features(
                    (memory[0][root : root + 1], memory[1][root : root + 1]), branch_tensor
                )[0]
            else:
                feature = cache[root, branch][tile]
            gathered.append(feature)
        predictions = policy.effect_values(torch.stack(gathered))
        targets = torch.as_tensor(
            np.array([rows[r]["effect_labels"][t] for r, t, *_ in requests]),
            device=device,
            dtype=torch.long,
        )
        roots = torch.tensor([r for r, *_ in requests], device=device)
        if not factored:
            per_root = per_root.index_add(0, roots, effect_loss(predictions, targets).mean(-1))
        tokens = tokens.index_add(0, roots, tokens.new_ones(len(requests)))
    metric_tokens = [(null_logits, null_targets)]
    if requests:
        metric_tokens.append((predictions, targets))
    if factored:
        gate, conditional, valid = conditional_effect_loss(null_logits, null_targets)
        conditional_count = valid.float()
        if requests:
            g, c, v = conditional_effect_loss(predictions, targets)
            gate = gate.index_add(0, roots, g)
            conditional = conditional.index_add(0, roots, c)
            conditional_count = conditional_count.index_add(0, roots, v.float())
        gate = (gate / tokens).sum() / max(1, int(root_count))
        conditional = (conditional / conditional_count.clamp_min(1)).sum() / max(1, int(root_count))
        eloss = gate + settings.get("effect_conditional_weight", 1.0) * conditional
    else:
        eloss = (per_root / tokens).sum() / max(1, int(root_count))
    return eloss, metric_tokens, tokens


def _forecast_metrics(policy, rows, threats, tloss, eloss, metric_tokens, tokens, tw, ew):
    device = policy.device
    # Keep a fixed metric carrier for the existing cohort transport. Reporting
    # names below identify the new tasks, rather than calling them cell/home BCE.
    depth_labels = torch.as_tensor(np.array(rows["threat_depth"]), device=device)
    depth_logits = threats[..., :33].reshape(-1, 5, 3, 11).float()
    censored = torch.as_tensor(np.array(rows["threat_censored"]), device=device)
    valid = (depth_labels >= 0) & ~(censored & (depth_labels == 0))
    exact = valid & ~censored
    if "threat_depth_allowed" in rows.dtype.names:
        allowed = torch.as_tensor(np.array(rows["threat_depth_allowed"]), device=device).long()
        valid = (allowed != 0) & (allowed != 2047)
        exact = (allowed != 0) & ((allowed & (allowed - 1)) == 0)
    correct = (depth_logits.argmax(-1) == depth_labels) & exact
    if not hasattr(policy, "_forecast_calibration"):
        reset_calibration(policy)
    calibration = policy._forecast_calibration
    categorical_histogram(calibration["depth"], depth_logits.softmax(-1), depth_labels, exact)
    time_labels = torch.as_tensor(np.array(rows["threat_time"]), device=device)
    categorical_histogram(
        calibration["breach_time"],
        threats[..., 33:39].float().softmax(-1),
        time_labels,
        time_labels >= 0,
    )
    loss_labels = torch.as_tensor(np.array(rows["threat_loss"]), device=device)
    prob_loss = threats[..., 39].float().sigmoid()
    categorical_histogram(
        calibration["loss"],
        torch.stack((1 - prob_loss, prob_loss), -1),
        loss_labels,
        loss_labels >= 0,
    )
    depth_p = depth_logits.detach().softmax(-1)
    depth_target = torch.nn.functional.one_hot(depth_labels.long().clamp_min(0), 11)
    depth_brier = ((depth_p - depth_target).square().sum(-1) * exact).sum()
    effect_brier = threats.new_zeros((), dtype=torch.float32)
    effect_count = threats.new_zeros((), dtype=torch.float32)
    effect_correct = threats.new_zeros((), dtype=torch.float32)
    for logits, target in metric_tokens:
        for term, block in enumerate(logits.float().split(EFFECT_SIZES, -1)):
            label = target[..., term]
            categorical_histogram(
                calibration["effect"], block.softmax(-1), label, label >= 0, term=term
            )
            p = block.detach().softmax(-1)
            expected = torch.nn.functional.one_hot(label.long(), block.shape[-1])
            effect_brier += (p - expected).square().sum()
            effect_count += label.numel()
            effect_correct += (block.argmax(-1) == label).sum()
            changed = label != block.shape[-1] // 2
            policy._forecast_score_totals[6] += ((block.argmax(-1) == label) & changed).sum()
            policy._forecast_score_totals[7] += changed.sum()
            change_probability = 1 - p[..., block.shape[-1] // 2]
            change_prediction = change_probability >= 0.5
            zero = block.shape[-1] // 2
            nonzero = torch.cat((block[..., :zero], block[..., zero + 1 :]), -1)
            conditional_target = (label - (label > zero).long()).clamp_max(nonzero.shape[-1] - 1)
            categorical_histogram(
                calibration["effect_change"],
                torch.stack((1 - change_probability, change_probability), -1),
                changed.long(),
                label >= 0,
                term=term,
            )
            categorical_histogram(
                calibration["effect_conditional"],
                nonzero.softmax(-1),
                conditional_target,
                changed,
                term=term,
            )
            policy._forecast_score_totals[8:14] += torch.stack(
                (
                    (change_prediction & changed).sum(),
                    (change_prediction & ~changed).sum(),
                    (~change_prediction & changed).sum(),
                    (change_probability - changed.float()).square().sum(),
                    (~changed & (block.argmax(-1) == label)).sum(),
                    (~changed).sum(),
                )
            ).double()
            policy._forecast_score_totals[14:] += torch.stack(
                (
                    ((nonzero.argmax(-1) == conditional_target) & changed).sum(),
                    changed.sum(),
                )
            ).double()
    times_p = threats[..., 33:39].detach().float().softmax(-1)
    times_valid = time_labels >= 0
    times_target = torch.nn.functional.one_hot(time_labels.long().clamp_min(0), 6)
    policy._forecast_score_totals[:6] += torch.stack(
        (
            ((times_p - times_target).square().sum(-1) * times_valid).sum(),
            times_valid.sum(),
            ((times_p.argmax(-1) == time_labels) & times_valid).sum(),
            ((prob_loss.detach() - loss_labels).square() * (loss_labels >= 0)).sum(),
            (loss_labels >= 0).sum(),
            (((prob_loss >= 0.5) == loss_labels.bool()) & (loss_labels >= 0)).sum(),
        )
    ).double()
    metrics = torch.stack(
        (
            tloss.detach(),
            depth_brier,
            exact.sum(),
            valid.any(-1).any(-1).sum(),
            eloss.detach(),
            effect_brier,
            effect_count,
            tokens.new_tensor(len(rows) if metric_tokens else 0),
            correct.sum(),
            effect_correct,
        )
    ).double()
    return tw * tloss + ew * eloss, metrics


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
    q, threat, effect = torch.stack(values).cpu().tolist()
    return dict(
        primary_trunk_norm=q,
        threat_trunk_norm=threat,
        effect_trunk_norm=effect,
        threat_to_primary=threat / q if q else None,
        effect_to_primary=effect / q if q else None,
    )
