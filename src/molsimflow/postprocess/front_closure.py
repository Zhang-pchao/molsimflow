"""Column-level closure tests for a migrating gas contact front.

The modules next to this one measure geometry: how the contacted
footprint is connected, how isolated the leading contact is, how far the
front advanced. This module adds the missing step, the one that
decides whether the chemistry is the controlling variable: does the
contact preference of a column predict whether the front actually gets
through that column.

The central difficulty is that a per-column preference is computed from
the observations that the dynamics happened to produce, and the
dynamics are themselves conditioned on the chemistry. A naive
correlation between preference and advance therefore mixes cause with
consequence. Two guards are used here.

* The comparison is made within a column, so two motifs that share a
  column also share a position along the transport axis.
* The null hypothesis is a permutation that shuffles the preference
  between columns of the same case, which destroys any link to the
  advance while preserving the marginal distributions of both.

Everything here is descriptive. The frames of one trajectory are
correlated samples, not independent repeats, so a permutation p value
describes the trajectory and not a population of bubbles.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "spearman_rank_correlation",
    "permutation_p_value",
    "area_normalised_rate",
    "shrink_ratio_toward_unity",
    "stabilised_site_contact_rate",
    "column_selectivity",
    "closure_test",
]


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Spearman rank correlation of two arrays.

    Rows where either value is not finite are dropped. The result is
    NaN when fewer than three pairs survive, or when either array is
    constant, because the correlation is then not defined.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("a and b must have the same shape")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = _rank(x[ok])
    ry = _rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Return the average ranks of a 1-D array."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # average the ranks of tied values
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def permutation_p_value(rho: float, a: np.ndarray, b: np.ndarray,
                         n_permutations: int = 5000,
                         seed: int = 20261031) -> float:
    """Return a two-sided permutation p value for a rank correlation.

    The second array is shuffled while the first is held fixed. This
    destroys the association between the two while preserving the
    marginal distribution of the shuffled quantity, which is the null
    hypothesis of interest here.
    """
    if not np.isfinite(rho):
        return float("nan")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = spearman_rank_correlation(x, rng.permutation(y))
    return float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_permutations + 1))


def area_normalised_rate(hits: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Return the contact rate per site of each motif in a column.

    A ratio of raw counts is dominated by the site counts of the two
    motifs. Dividing by the number of sites of the motif removes that
    area factor, so that a motif with twice as many sites is not
    credited with twice the contact simply for existing twice as often.
    """
    h = np.asarray(hits, dtype=float)
    a = np.asarray(area, dtype=float)
    if h.shape != a.shape:
        raise ValueError("hits and area must have the same shape")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a > 0, h / np.where(a > 0, a, 1.0), np.nan)


def shrink_ratio_toward_unity(ratio: float, support: float,
                               pseudo_count: float = 10.0) -> float:
    """Shrink a selectivity ratio towards one according to its support.

    A ratio measured from a handful of contacts is not a property of the
    surface. This pulls such a ratio back towards one, which is the
    value of no preference, with a weight that grows with the number of
    observations that back the ratio.

    The shrinkage is applied in log space, so that it is symmetric: a
    ratio of ten and a ratio of one tenth are damped by the same
    factor, and the result is always a ratio rather than an
    asymmetrically clipped one.

    The weight is ``support / (support + pseudo_count)``, so a ratio
    backed by many observations is kept almost unchanged, while a ratio
    backed by few is pulled close to one. The pseudo count is the number
    of observations at which the measured ratio is trusted halfway.
    """
    if not np.isfinite(ratio):
        return float("nan")
    if not np.isfinite(support) or support < 0:
        support = 0.0
    if ratio <= 0:
        return float("nan")
    w = support / (support + pseudo_count)
    return float(np.exp(w * np.log(ratio)))


def stabilised_site_contact_rate(hits: np.ndarray, area: np.ndarray,
                                 pseudo_count: float = 1.0) -> np.ndarray:
    """Return the contact rate per site of each motif in a column.

    The count-based rate is inflated by a pseudo count so that a cell
    that was never contacted still carries a small positive rate and
    can be compared with one that was. For the selectivity ratio
    itself use :func:`shrink_ratio_toward_unity`, which damps a sparse
    tail without distorting the well-supported value.
    """
    h = np.asarray(hits, dtype=float)
    a = np.asarray(area, dtype=float)
    if h.shape != a.shape:
        raise ValueError("hits and area must have the same shape")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (h + pseudo_count) / (a + pseudo_count)


def column_selectivity(hits: np.ndarray, area: np.ndarray,
                       numerator: int = 0, denominator: int = 2,
                       pseudo_count: float = 10.0) -> float:
    """Return the stabilised selectivity of two motifs inside one column.

    The raw ratio is taken between the two *area-normalised* contact
    rates, so that it measures a preference rather than a difference in
    how many sites each motif has. The ratio is then shrunk towards
    one according to the number of observations that back it, so that a
    ratio produced by a handful of contacts does not masquerade as a
    surface property.

    The result is NaN when the denominator motif is absent from the
    column, because the comparison is then undefined rather than zero.
    """
    rates = area_normalised_rate(hits, area)
    if area[denominator] <= 0:
        return float("nan")
    den = rates[denominator]
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    ratio = float(rates[numerator] / den)
    support = float(min(hits[numerator], hits[denominator]))
    return shrink_ratio_toward_unity(ratio, support, pseudo_count)


def closure_test(preference: Sequence[float],
                 advance: Sequence[float],
                 n_permutations: int = 5000,
                 seed: int = 20261031) -> dict:
    """Test whether the contact preference of a column tracks its advance.

    ``preference`` and ``advance`` are one value per column. The result
    reports the rank correlation, the permutation p value, and the
    number of columns that carried the test.
    """
    p = np.asarray(preference, dtype=float)
    a = np.asarray(advance, dtype=float)
    rho = spearman_rank_correlation(p, a)
    return {
        "n_columns": int(np.isfinite(p).sum() and np.count_nonzero(
            np.isfinite(p) & np.isfinite(a))),
        "spearman_rho": rho,
        "permutation_p": permutation_p_value(rho, p, a, n_permutations, seed),
    }
