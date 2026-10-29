"""Geometry-support matching for interfacial-state comparisons.

Callers define features, physical units, continuity, strata and time thinning.
Matching establishes measured overlap, not exchangeability or causal support.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_caliper_states(left, right, calipers):
    """Maximize pair count, then minimize squared normalized feature distance.

    Inputs are finite N-by-D and M-by-D state tables; every caliper is a
    strictly positive feature-wise absolute tolerance. Only pairs satisfying
    all tolerances are eligible. Each row is used at most once. Outcomes must
    be kept out of the supplied features by the caller. Empty support remains
    explicit. Exact ties may depend on input row order.
    """
    a, b, c = np.asarray(left, float), np.asarray(right, float), np.asarray(calipers, float)
    if a.ndim != 2 or b.ndim != 2 or not a.shape[1] or a.shape[1] != b.shape[1]:
        raise ValueError("Matching N-by-D and M-by-D state tables required")
    if c.shape != (a.shape[1],) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("Finite positive feature-wise calipers required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("State features must be finite")
    n, m, d = len(a), len(b), a.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.abs(a[:, None, :] - b[None, :, :]) / c
    allowed = np.all(delta <= 1, axis=2)
    pair_cost = np.sum(np.minimum(delta, 1) ** 2, axis=2)
    pairs = np.empty((0, 2), dtype=np.int64)
    if n and m and allowed.any():
        # One missed real pair costs more than every possible eligible cost.
        penalty = float((min(n, m) + 1) * (d + 1))
        cost = np.full((n, m + n), penalty)
        cost[:, :m] = np.where(allowed, pair_cost, penalty * (n + 1))
        li, ri = linear_sum_assignment(cost)
        real = ri < m
        pairs = np.column_stack((li[real], ri[real])).astype(np.int64)
        assert np.all(allowed[pairs[:, 0], pairs[:, 1]])
    li, ri = pairs.T
    return dict(pairs=pairs, unmatched_left=np.setdiff1d(np.arange(n), li),
                unmatched_right=np.setdiff1d(np.arange(m), ri),
                left_candidate_counts=allowed.sum(axis=1),
                right_candidate_counts=allowed.sum(axis=0),
                eligible_pair_count=int(allowed.sum()),
                normalized_gaps=delta[li, ri],
                total_squared_cost=float(pair_cost[li, ri].sum()))
