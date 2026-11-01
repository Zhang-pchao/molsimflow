"""Column-level closure tests for a migrating gas contact front.

The modules next to this one measure geometry: how the contacted
footprint is connected, how isolated the leading contact is, how far the
front advanced. This module describes associations between the contact
preference of a column and subsequent front advance. These associations
do not decide whether chemistry is a causal controlling variable.

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
    "travel_ordered_pattern_match",
    "seam_depth",
    "net_displacement",
    "exposure_normalised_preference",
    "envelope_leading_edge",
    "leading_edge_advance",
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


def seam_depth(x: Sequence[float], direction: int, seam_x: float) -> float:
    """Depth of the farthest visited column measured from the seam.

    ``direction`` is ``+1`` for a face the front moves into (the x grows
    away from the seam) and ``-1`` for a face it moves out of. The depth
    is a distance along x, so it is always positive and it is measured
    from the seam rather than from the first column.

    The column index is *not* a distance. The first column stands
    ``x_first - seam_x`` away from the seam, and the columns are
    ``column_pitch`` apart, so an index-based reach silently mixes the
    two offsets. Working with x directly removes that ambiguity.
    """
    xs = np.asarray(x, dtype=float)
    if xs.size == 0:
        return float("nan")
    if direction > 0:
        return float(xs.max() - seam_x)
    return float(seam_x - xs.min())


def travel_ordered_pattern_match(pattern_a: Sequence[str],
                                 pattern_b: Sequence[str],
                                 distance_a: Sequence[float],
                                 distance_b: Sequence[float],
                                 direction_a: int = 1,
                                 direction_b: int = 1) -> dict:
    """Match two faces column by column *in the direction of travel.

    Two faces carry the same chemical pattern when their motif labels
    agree column by column. That comparison is only physically
    meaningful when the columns are lined up along the direction the
    front actually moves: a column the front reaches first must be
    matched against a column the front reaches first, not against the
    column with the same index in an unrelated coordinate frame.

    ``pattern_a`` and ``pattern_b`` are the motif labels ordered by
    increasing x. ``distance_a`` and ``distance_b`` are the distances
    of those columns from the seam, positive and increasing away from
    the seam in the direction of travel.

    The result reports how many columns could be matched, whether the
    matched sequences are identical, and the distance travelled on each
    face.
    """
    pa = list(pattern_a)
    pb = list(pattern_b)
    if len(pa) != len(pb):
        return {"n_matched": min(len(pa), len(pb)), "identical": False,
                "mismatch_at": None, "n_mismatch": None,
                "depth_a": float("nan"), "depth_b": float("nan"),
                "depth_ratio": float("nan")}

    n_mismatch = 0
    first_mismatch = None
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x != y:
            n_mismatch += 1
            if first_mismatch is None:
                first_mismatch = i

    depth_a = float(np.nanmax(distance_a)) if len(distance_a) else float("nan")
    depth_b = float(np.nanmax(distance_b)) if len(distance_b) else float("nan")
    ratio = depth_b / depth_a if (np.isfinite(depth_a) and depth_a > 0
                                  and np.isfinite(depth_b)) else float("nan")
    return {
        "n_matched": len(pa),
        "identical": n_mismatch == 0,
        "n_mismatch": n_mismatch,
        "mismatch_at": first_mismatch,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "depth_ratio": ratio,
    }


def net_displacement(x: Sequence[float]) -> float:
    """Net displacement of a front, from the first to the last sample.

    The span of a front is not a displacement. A front that explores
    far to one side and comes back has a large span and a small net
    displacement, and it is the net displacement that measures whether
    the bubble actually made progress.
    """
    xs = np.asarray(x, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size < 2:
        return float("nan")
    return float(xs[-1] - xs[0])


def exposure_normalised_preference(hits_numerator: float,
                                   area_numerator: float,
                                   hits_denominator: float,
                                   area_denominator: float,
                                   n_frames: float) -> float:
    """Contact preference per unit exposure, not per unit observation.

    A contact rate computed from raw hit counts grows with the time the
    bubble spends on the face. Two faces that carry the same pattern
    but are visited for very different lengths of time therefore have
    different raw rates even if their chemistry is identical. Dividing
    by the number of frames in which the face was sampled removes the
    exposure and leaves the preference.
    """
    if not np.isfinite(n_frames) or n_frames <= 0:
        return float("nan")
    if area_numerator <= 0 or area_denominator <= 0:
        return float("nan")
    num = hits_numerator / area_numerator / n_frames
    den = hits_denominator / area_denominator / n_frames
    if den <= 0:
        return float("nan")
    return float(num / den)


def envelope_leading_edge(signed_distance: np.ndarray,
                          x: np.ndarray,
                          face_mask: np.ndarray,
                          min_run: int = 3,
                          gap: float = 3.0) -> np.ndarray:
    """Leading edge of a gas envelope over a face, per frame.

    This is an observable that is built from the *bulk geometry* of the
    gas body rather than from a count of atomic contacts, and it is
    intended as a cross-check on any conclusion that was reached through
    contact bookkeeping.

    ``signed_distance`` is the signed distance from each surface site to
    the boundary of the projected gas envelope, positive outside the
    envelope. A site with a non-positive value lies under the envelope:
    the gas body reaches over it. The leading edge is then the largest
    ``x`` at which the envelope still covers the surface.

    Two guards keep a single isolated site from manufacturing a front.

    * ``min_run``: the contiguous covered run must contain at least this
      many sites, otherwise the frame contributes no edge.
    * ``gap``: the run is grown leftwards from the rightmost covered
      site while the next covered site lies within this distance, so a
      lone outlier far away cannot extend the edge.

    Parameters
    ----------
    signed_distance
        Array of shape ``(n_frames, n_sites)``.
    x
        The x coordinate of every site, shape ``(n_sites,)``.
    face_mask
        Boolean mask of the sites belonging to the face of interest.
    min_run
        Minimum number of sites in the covered run.
    gap
        Maximum spacing between neighbouring sites of the run.

    Returns
    -------
    An array of shape ``(n_frames,)`` holding the x coordinate of the
    leading edge, and NaN for the frames that carry no qualifying
    envelope.
    """
    sd = np.asarray(signed_distance, dtype=float)
    xs = np.asarray(x, dtype=float)
    mask = np.asarray(face_mask, dtype=bool)
    if sd.ndim != 2:
        raise ValueError("signed_distance must be two dimensional")
    if xs.shape != (sd.shape[1],):
        raise ValueError("x must have one entry per site")
    if mask.shape != (sd.shape[1],):
        raise ValueError("face_mask must have one entry per site")
    if min_run < 1:
        raise ValueError("min_run must be at least one")

    sites = np.flatnonzero(mask)
    edge = np.full(sd.shape[0], np.nan)
    if sites.size == 0:
        return edge

    for j in range(sd.shape[0]):
        covered = sites[sd[j][sites] <= 0.0]
        if covered.size < min_run:
            continue
        ordered = np.sort(xs[covered])
        # Split the covered sites into contiguous runs, then keep the
        # rightmost run that is long enough to count as a front. Taking
        # the rightmost *qualifying* run, rather than growing a run
        # leftwards from the rightmost covered site, stops a lone
        # outlier on the far side from setting the edge.
        breaks = np.flatnonzero(np.diff(ordered) > gap)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [ordered.size - 1]))
        best = None
        for lo, hi in zip(starts, ends):
            if hi - lo + 1 >= min_run:
                best = max(best if best is not None else -np.inf, ordered[hi])
        if best is not None:
            edge[j] = float(best)
    return edge


def leading_edge_advance(edge: np.ndarray,
                         seam_x: float,
                         direction: int = 1) -> float:
    """Advance of a leading edge, measured from the seam.

    The distance is always reported as a positive depth from the seam,
    while the *advance* is signed by the direction of travel: a front
    that moves towards increasing x has ``direction`` +1.

    The advance is measured between the first and the last frame that
    carries an edge, so that a frame without a qualifying envelope does
    not reset the reference point.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    values = np.asarray(edge, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    depths = (values - seam_x) if direction > 0 else (seam_x - values)
    return float(depths[-1] - depths[0])
