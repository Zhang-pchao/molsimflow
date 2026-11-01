"""Exact centroid budgets for a changing molecular membership.

Positions must use the same physical coordinate chart at both endpoints.
The caller owns identity, PBC, continuity, and cluster definitions.
"""
from __future__ import annotations
import numpy as np

def moving_membership_centroid_budget(
    positions_before: np.ndarray, positions_after: np.ndarray,
    membership_before: np.ndarray, membership_after: np.ndarray,
) -> dict:
    """Decompose an equal-mass open-population centroid displacement.

    With R retained, J joined, L departed, and N1 final population:
      delta C = sum_R(x1-x0)/N1
              + sum_J(x1-C0)/N1 - sum_L(x0-C0)/N1.

    The finite-step convention assigns displacement only to retained IDs.
    It is exact, but its terms depend on sampling cadence and membership
    definition. Entry/exit is not necessarily dissolution/condensation.
    Empty populations return an explicitly undefined centroid budget.
    """
    x0, x1 = np.asarray(positions_before, float), np.asarray(positions_after, float)
    b, a = np.asarray(membership_before), np.asarray(membership_after)
    if x0.ndim != 2 or x0.shape != x1.shape or x0.shape[1] < 1:
        raise ValueError("Positions must have matching N by D shapes")
    if b.dtype != np.bool_ or a.dtype != np.bool_:
        raise ValueError("Membership arrays must be boolean")
    if b.shape != (len(x0),) or a.shape != b.shape:
        raise ValueError("Membership must match the ordered positions")
    if not np.isfinite(x0).all() or not np.isfinite(x1).all():
        raise ValueError("Positions must be finite in one common chart")
    n0, n1 = int(b.sum()), int(a.sum())
    retained, joined, departed = b & a, ~b & a, b & ~a
    undef = np.full(x0.shape[1], np.nan)
    c0 = x0[b].mean(0) if n0 else undef.copy()
    c1 = x1[a].mean(0) if n1 else undef.copy()
    valid = bool(n0 and n1)
    if valid:
        move = (x1[retained] - x0[retained]).sum(0) / n1
        gain = (x1[joined] - c0).sum(0) / n1
        loss = -(x0[departed] - c0).sum(0) / n1
        change = c1 - c0
        residual = change - move - gain - loss
    else:
        move, gain, loss, change, residual = [undef.copy() for _ in range(5)]
    return dict(
        count_before=n0, count_after=n1, retained=int(retained.sum()),
        joined=int(joined.sum()), departed=int(departed.sum()),
        count_residual=n1-n0-int(joined.sum())+int(departed.sum()),
        centroid_before=c0, centroid_after=c1, centroid_change=change,
        retained_displacement=move, joined_contribution=gain,
        departed_contribution=loss, centroid_residual=residual,
        centroid_valid=valid,
    )
