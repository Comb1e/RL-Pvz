"""Device-side class reliability histograms, transferred at fitted boundaries."""

import torch

OUTCOME_METRIC_WIDTH = 7  # Raw loss, calibration (4), weighted field and weighted probe.


def reset_calibration(policy):
    policy._forecast_calibration = {
        "effect": torch.zeros((5, 4, 11, 10, 3), device=policy.device, dtype=torch.float64)
    }


def categorical_histogram(histogram, probabilities, labels, valid, *, term=0):
    """Counts / summed probability / summed observed class, for each lane/class/bin."""
    with torch.no_grad():
        p = probabilities.detach().double()
        _, lanes, classes = p.shape
        rows = torch.arange(lanes, device=p.device)[None, :, None]
        cs = torch.arange(classes, device=p.device)[None, None, :]
        bins = (p * 10).long().clamp(0, 9)
        maximum = histogram.shape[2]
        index = ((rows * histogram.shape[1] + term) * maximum + cs) * 10 + bins
        mask = valid[..., None].expand_as(p)
        # Boolean compaction allocates a dynamic CUDA result and synchronizes
        # the host to discover its size. Keep the known shape and zero unknown
        # terms instead; their contribution to every histogram remains zero.
        weight = mask.double()
        predicted = torch.where(mask, p, 0)
        outcome = (cs == labels[..., None]).expand_as(p).double() * weight
        at = index.flatten()
        values = torch.stack((weight, predicted, outcome), -1).reshape(-1, 3)
        histogram.reshape(-1, 3).index_add_(0, at, values)


def calibration_report(policy):
    report = {}
    for name, histogram in getattr(policy, "_forecast_calibration", {}).items():
        h = histogram.cpu()
        count, prediction, actual = h.unbind(-1)
        total = float(count.sum())
        class_count = count.sum(-1)
        by_class = torch.stack(
            (
                class_count,
                prediction.sum(-1) / class_count.clamp_min(1),
                actual.sum(-1) / class_count.clamp_min(1),
                (prediction - actual).abs().sum(-1) / class_count.clamp_min(1),
            ),
            -1,
        )
        report[name] = dict(
            by_lane_term_class=by_class.tolist(),
            fields=["count", "mean_probability", "observed_frequency", "ece"],
            classwise_ece=float((prediction - actual).abs().sum()) / total if total else None,
            class_predictions=int(total),
        )
    if policy.absolute_outcomes and "effect" in report:
        report["effect"]["objective"] = "absolute_outcome_v3"
        report["effect"]["valid_classes"] = list(policy.effect_sizes)
    return report
