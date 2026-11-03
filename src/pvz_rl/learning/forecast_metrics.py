"""Device-side class reliability histograms, transferred at fitted boundaries."""

import torch


def reset_calibration(policy):
    policy._forecast_score_totals = torch.zeros(8, device=policy.device, dtype=torch.float64)
    policy._forecast_calibration = {
        "depth": torch.zeros((5, 3, 11, 10, 3), device=policy.device, dtype=torch.float64),
        "breach_time": torch.zeros((5, 1, 6, 10, 3), device=policy.device, dtype=torch.float64),
        "loss": torch.zeros((5, 1, 2, 10, 3), device=policy.device, dtype=torch.float64),
        "effect": torch.zeros((5, 4, 21, 10, 3), device=policy.device, dtype=torch.float64),
    }


def categorical_histogram(histogram, probabilities, labels, valid, *, term=0):
    """Counts / summed probability / summed observed class, for each lane/class/bin."""
    with torch.no_grad():
        p = probabilities.detach().double()
        if p.ndim == 3:
            p, labels, valid = p[:, :, None], labels[:, :, None], valid[:, :, None]
        b, lanes, horizons, classes = p.shape
        rows = torch.arange(lanes, device=p.device)[None, :, None, None]
        hs = torch.arange(horizons, device=p.device)[None, None, :, None] + term
        cs = torch.arange(classes, device=p.device)[None, None, None, :]
        bins = (p * 10).long().clamp(0, 9)
        maximum = histogram.shape[2]
        index = (((rows * histogram.shape[1] + hs) * maximum + cs) * 10 + bins).expand_as(p)
        mask = valid[..., None].expand_as(p)
        predicted = p[mask]
        outcome = (cs == labels[..., None]).expand_as(p)[mask].double()
        at = index[mask].flatten()
        values = torch.stack((torch.ones_like(predicted), predicted, outcome), -1)
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
            by_lane_horizon_class=by_class.tolist(),
            fields=["count", "mean_probability", "observed_frequency", "ece"],
            classwise_ece=float((prediction - actual).abs().sum()) / total if total else None,
            class_predictions=int(total),
        )
    return report
