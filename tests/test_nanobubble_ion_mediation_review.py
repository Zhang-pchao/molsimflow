"""Numerical guards for the integrated exploratory comparisons."""

import math

import numpy as np

from molsimflow.postprocess.nanobubble_ion_mediation_review import (
    bh_qvalues,
    shift_placebo,
    weighted_mean,
)


def test_weighted_mean_ignores_unobserved_cells():
    mean, cells = weighted_mean([
        {"metric": "0.5", "count": "2"},
        {"metric": "0.8", "count": "1"},
        {"metric": "nan", "count": "0"},
    ], "metric", "count")
    assert cells == 3
    assert math.isclose(mean, 0.6)


def test_circular_shift_and_bh_values():
    x = np.arange(12.0)
    rho, p, n = shift_placebo(x, np.roll(x, 1))
    assert -1 <= rho <= 1
    assert 0 < p <= 1
    assert n == 12
    q = bh_qvalues([0.01, 0.04, math.nan, 0.5])
    assert math.isclose(q[0], 0.03)
    assert math.isnan(q[2])
