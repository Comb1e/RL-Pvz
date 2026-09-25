"""Crossed bootstrap: resample independent learner runs and paired scenarios."""

import numpy as np


def result_matrix(rows):
    runs = sorted({r["learner_seed"] for r in rows}, key=str)
    cases = sorted({r["scenario_seed"] for r in rows})
    if not runs or not cases:
        raise ValueError("No evaluation rows")
    indexed = {}
    for row in rows:
        key = (row["learner_seed"], row["scenario_seed"])
        if key in indexed:
            raise ValueError("Duplicate learner/scenario pair; select one checkpoint per run")
        indexed[key] = row["win"]
    if len(indexed) != len(runs) * len(cases):
        raise ValueError("Incomplete evaluation rectangle; missing episodes cannot be dropped")
    return (
        np.array([[indexed[(run, case)] for case in cases] for run in runs], dtype=float),
        runs,
        cases,
    )


def bootstrap_interval(matrix, *, replicates=2000, seed=1739):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Expected a nonempty finite run-by-scenario matrix")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for i in range(replicates):
        runs = rng.integers(matrix.shape[0], size=matrix.shape[0])
        cases = rng.integers(matrix.shape[1], size=matrix.shape[1])
        means[i] = matrix[np.ix_(runs, cases)].mean()
    return {
        "mean": float(matrix.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "runs": matrix.shape[0],
        "scenarios": matrix.shape[1],
        "per_run": matrix.mean(axis=1).tolist(),
    }


def paired_difference(rows, reference, **kwargs):
    a, aruns, acases = result_matrix(rows)
    b, bruns, bcases = result_matrix(reference)
    if acases != bcases:
        raise ValueError("Paired comparison requires identical scenario seeds")
    if bruns == [None]:
        b = np.repeat(b, len(aruns), axis=0)
    elif aruns != bruns:
        raise ValueError("Learned comparisons require paired learner seeds")
    return bootstrap_interval(a - b, **kwargs)
