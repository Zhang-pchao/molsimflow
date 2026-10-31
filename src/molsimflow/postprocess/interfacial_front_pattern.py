"""Chemical-pattern gateway for a migrating gas contact front on a joined surface.

The module turns a static surface motif map plus a time-resolved gas/surface
contact list into three auditable quantities:

* the *column map*: how parent Si atoms group into columns along the
  transport axis, and which columns are chemically uniform or striped;
* the *front-tip track*: the leading contacted parent on the destination
  half-face over time, resolved onto its column;
* the *column-conditional advance hazard*: the probability that the tip
  passes the next column within a fixed look-ahead, computed only on
  columns that the trajectory actually left, so that a stalled run does
  not contribute unearned advance events.

Design constraints that this module deliberately encodes:

1. Columns are defined by the prepared parent-site geometry, not by the
   motif content, so that a chemical pattern cannot be defined by the
   outcome it is supposed to explain.
2. Right-censored columns are reported as censored and are excluded from
   hazard denominators. A column the tip never left has an undefined
   advance hazard, not a zero.
3. "Bare" means an OH-free terminal site produced by the preparation
   chemistry. It is a CH3-terminated Si anchor, not an unsaturated Si.
4. Motif selectivity is reported against the area fraction of the same
   half-face, so a face that is chemically uniform has selectivity 1.0 by
   construction and heterogeneous faces are compared on equal footing.
5. No quantity here is a rate, a free energy, a wetting rank, or a
   confidence interval. Frames are correlated samples from one
   trajectory; they are not independent events.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

MOTIF_NAMES = {0: "bare", 1: "single_oh", 2: "dual_oh"}


@dataclass(frozen=True)
class ColumnMap:
    """Columns of parent sites along the transport axis.

    ``column_x`` is the mean prepared X coordinate of the parents in each
    column, ``members`` maps each column index to the parent indices it
    contains, and ``motif_counts`` records, per column, how many parents
    carry zero, one, or two OH terminals.
    """

    column_x: np.ndarray
    members: tuple[np.ndarray, ...]
    motif_counts: np.ndarray
    parent_motif: np.ndarray
    parent_column: np.ndarray

    @property
    def n_columns(self) -> int:
        return len(self.column_x)

    def column_motif_label(self, column: int) -> str:
        """Return the dominant motif label of a column.

        A column is ``mixed`` when its members do not all share one motif.
        """
        counts = self.motif_counts[column]
        present = [k for k in range(3) if counts[k] > 0]
        if len(present) == 1:
            return MOTIF_NAMES[present[0]]
        if not present:
            return "empty"
        return "mixed"

    def column_is_uniform(self, column: int) -> bool:
        """Return whether every member of a column shares one motif."""
        counts = self.motif_counts[column]
        return int(np.count_nonzero(counts)) == 1

    def _is_dual_oh_column(self, column: int) -> bool:
        counts = self.motif_counts[column]
        return bool(counts[2] > 0 and counts[0] == 0 and counts[1] == 0)

    def _is_bare_column(self, column: int) -> bool:
        counts = self.motif_counts[column]
        return bool(counts[0] > 0 and counts[1] == 0 and counts[2] == 0)

    def pattern_string(self) -> str:
        """Return a one-character-per-column motif summary.

        ``W`` is a fully dual-OH column, ``H`` a fully OH-free column, and
        ``M`` anything else.
        """
        chars = []
        for j in range(self.n_columns):
            if self._is_dual_oh_column(j):
                chars.append("W")
            elif self._is_bare_column(j):
                chars.append("H")
            else:
                chars.append("M")
        return "".join(chars)

    def alternating_period_A(self) -> float:
        """Return the column spacing of the fully dual-OH stripes.

        The value is ``nan`` when the face holds fewer than two such
        columns, so that a single stripe is never reported as a period.
        """
        stripes = [j for j in range(self.n_columns) if self._is_dual_oh_column(j)]
        if len(stripes) < 2:
            return float("nan")
        return float(np.mean(np.diff(self.column_x[stripes])))


def build_column_map(parent_xy: np.ndarray, parent_motif: np.ndarray, *,
                     half_mask: np.ndarray, tolerance: float = 0.05) -> ColumnMap:
    """Group parent sites into columns along X for one half-face.

    ``parent_xy`` holds prepared parent coordinates, ``parent_motif`` the
    number of OH terminals per parent, and ``half_mask`` selects the
    half-face of interest. Parents whose X coordinates differ by no more
    than ``tolerance`` share a column.
    """
    xy = np.asarray(parent_xy, dtype=float)
    motif = np.asarray(parent_motif)
    mask = np.asarray(half_mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("parent_xy must have shape (n, 2)")
    if motif.shape != (len(xy),):
        raise ValueError("parent_motif must match parent_xy rows")
    if mask.shape != (len(xy),):
        raise ValueError("half_mask must match parent_xy rows")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise ValueError("half_mask selects no parent sites")

    order = selected[np.argsort(xy[selected, 0], kind="stable")]
    groups: list[list[int]] = []
    current: list[int] = []
    anchor = None
    for idx in order:
        x = xy[idx, 0]
        if anchor is None or abs(x - anchor) <= tolerance:
            current.append(int(idx))
            if anchor is None:
                anchor = x
        else:
            groups.append(current)
            current = [int(idx)]
            anchor = x
    if current:
        groups.append(current)

    column_x = np.array([xy[g, 0].mean() for g in groups], dtype=float)
    motif_counts = np.zeros((len(groups), 3), dtype=int)
    parent_column = np.full(len(xy), -1, dtype=int)
    for j, g in enumerate(groups):
        members = np.asarray(g, dtype=int)
        parent_column[members] = j
        for k in range(3):
            motif_counts[j, k] = int(np.count_nonzero(motif[members] == k))
    return ColumnMap(
        column_x=column_x,
        members=tuple(np.asarray(g, dtype=int) for g in groups),
        motif_counts=motif_counts,
        parent_motif=motif.astype(int),
        parent_column=parent_column,
    )


def front_tip_track(contacted_parents: Sequence[np.ndarray], parent_xy: np.ndarray,
                    half_mask: np.ndarray) -> np.ndarray:
    """Return the leading contacted parent X per frame on one half-face.

    ``contacted_parents`` holds the parent indices touched by the main gas
    cluster in each frame. The result is ``nan`` for frames in which the
    gas does not touch the half-face at all, so that "not yet there" is
    never recorded as a position.
    """
    xy = np.asarray(parent_xy, dtype=float)
    mask = np.asarray(half_mask, dtype=bool)
    track = np.full(len(contacted_parents), np.nan, dtype=float)
    for f, parents in enumerate(contacted_parents):
        arr = np.asarray(parents, dtype=int)
        arr = arr[mask[arr]] if arr.size else arr
        if arr.size:
            track[f] = xy[arr, 0].max()
    return track


def resolve_to_columns(track: np.ndarray, column_x: np.ndarray, *,
                       tolerance: float = 0.25) -> np.ndarray:
    """Map a front-tip track onto column indices.

    Returns ``-1`` for frames without a tip. A tip is assigned to the
    nearest column when it lies within ``tolerance`` of that column;
    otherwise the frame is left unresolved rather than snapped to a
    neighbouring column.
    """
    tip = np.asarray(track, dtype=float)
    cols = np.asarray(column_x, dtype=float)
    if len(cols) == 0:
        raise ValueError("column_x must not be empty")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    out = np.full(len(tip), -1, dtype=int)
    valid = np.isfinite(tip)
    if not valid.any():
        return out
    nearest = np.argmin(np.abs(cols[None, :] - tip[valid, None]), axis=1)
    resid = np.abs(cols[nearest] - tip[valid])
    resolved = resid <= tolerance
    idx = np.flatnonzero(valid)[resolved]
    out[idx] = nearest[resolved]
    return out


def column_advance_hazard(column_index: np.ndarray, n_columns: int, *,
                           lookahead: int) -> dict[int, dict[str, float]]:
    """Estimate the probability of advancing past a column within a look-ahead.

    Only columns the trajectory actually left contribute. A column that
    was occupied but never left is right-censored: it is reported with
    ``censored=True`` and no hazard value, because the advance event
    never had the chance to occur inside the observation window.

    ``lookahead`` counts saved frames; a negative or zero value is
    rejected.
    """
    idx = np.asarray(column_index, dtype=int)
    if idx.ndim != 1:
        raise ValueError("column_index must be one dimensional")
    if n_columns < 1:
        raise ValueError("n_columns must be positive")
    if lookahead < 1:
        raise ValueError("lookahead must be positive")

    passed = np.zeros(n_columns, dtype=bool)
    if idx.size:
        reached = idx[idx >= 0]
        if reached.size:
            passed[: int(reached.max())] = True

    out: dict[int, dict[str, float]] = {}
    for j in range(n_columns):
        frames = np.flatnonzero(idx == j)
        if frames.size == 0:
            out[j] = {"n_occupied": 0, "n_evaluable": 0, "n_advanced": 0,
                      "hazard": float("nan"), "censored": True}
            continue
        if not passed[j]:
            out[j] = {"n_occupied": int(frames.size), "n_evaluable": 0,
                      "n_advanced": 0, "hazard": float("nan"), "censored": True}
            continue
        evaluable = frames[frames + lookahead < len(idx)]
        advanced = 0
        for f in evaluable:
            window = idx[f + 1: f + 1 + lookahead]
            if np.any(window > j):
                advanced += 1
        n_eval = int(evaluable.size)
        out[j] = {"n_occupied": int(frames.size), "n_evaluable": n_eval,
                  "n_advanced": int(advanced),
                  "hazard": (advanced / n_eval) if n_eval else float("nan"),
                  "censored": False}
    return out


def motif_selectivity(contacted_counts: np.ndarray, motif: np.ndarray, *,
                      half_mask: np.ndarray) -> dict[int, float]:
    """Return contacted-versus-area selectivity for each motif on a half-face.

    ``contacted_counts`` holds the number of contacted parent sites per
    motif, as returned by :func:`contacted_parent_motif_counts`.
    Selectivity is the contacted fraction of a motif divided by that
    motif's area fraction on the same half-face. A motif with zero area
    yields ``nan`` because the ratio is undefined, not zero. A uniform
    half-face yields 1.0 for its single motif by construction.
    """
    counts = np.asarray(contacted_counts, dtype=float)
    motif = np.asarray(motif, dtype=int)
    mask = np.asarray(half_mask, dtype=bool)
    if counts.shape != (3,):
        raise ValueError("contacted_counts must hold one count per motif")
    if motif.shape != mask.shape:
        raise ValueError("motif and half_mask must match")
    total = float(counts.sum())
    area_counts = np.array([np.count_nonzero((motif == k) & mask) for k in range(3)], dtype=float)
    area_total = float(area_counts.sum())
    out: dict[int, float] = {}
    for k in range(3):
        if area_counts[k] <= 0.0 or total <= 0.0:
            out[k] = float("nan")
            continue
        out[k] = float((counts[k] / total) / (area_counts[k] / area_total))
    return out


def contacted_parent_motif_counts(contacted_parents: Sequence[np.ndarray],
                                  parent_motif: np.ndarray, *,
                                  half_mask: np.ndarray | None = None) -> np.ndarray:
    """Return the per-motif count of contacted parents across all frames.

    When ``half_mask`` is given, only parents on that half-face are
    counted, so the result can be compared against the area fraction of
    the same half-face.
    """
    motif = np.asarray(parent_motif, dtype=int)
    if half_mask is None:
        mask = np.ones(len(motif), dtype=bool)
    else:
        mask = np.asarray(half_mask, dtype=bool)
        if mask.shape != motif.shape:
            raise ValueError("half_mask must match parent_motif shape")
    counts = np.zeros(3, dtype=float)
    for parents in contacted_parents:
        arr = np.asarray(parents, dtype=int)
        if arr.size == 0:
            continue
        arr = arr[mask[arr]]
        for k in range(3):
            counts[k] += np.count_nonzero(motif[arr] == k)
    return counts


def frame_contacted_parents(gas_edges: np.ndarray, gas_offsets: np.ndarray,
                             parent_index: np.ndarray) -> list[np.ndarray]:
    """Split a segmented gas/terminal edge list into per-frame parent sets.

    ``gas_edges`` holds ``[gas_molecule_index, terminal_site_index]`` rows
    grouped by frame through ``gas_offsets``, which has one entry per
    frame plus a final terminator. The first column of an edge is a gas
    molecule identifier, not a frame index.
    """
    edges = np.asarray(gas_edges, dtype=int)
    offsets = np.asarray(gas_offsets, dtype=int)
    pindex = np.asarray(parent_index, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("gas_edges must have shape (n, 2)")
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("gas_offsets must have at least two entries")
    if offsets[0] != 0 or offsets[-1] != len(edges):
        raise ValueError("gas_offsets must span the edge list exactly")
    n_frames = len(offsets) - 1
    out: list[np.ndarray] = []
    for f in range(n_frames):
        lo, hi = int(offsets[f]), int(offsets[f + 1])
        if hi <= lo:
            out.append(np.empty(0, dtype=int))
            continue
        out.append(np.unique(pindex[edges[lo:hi, 1]]))
    return out


def per_column_motif_selectivity(contacted_parents: Sequence[np.ndarray],
                                 parent_motif: np.ndarray, parent_xy: np.ndarray,
                                 *, half_mask: np.ndarray,
                                 column_tolerance: float = 0.05) -> list[dict]:
    """Return within-column contacted-versus-area selectivity for each motif.

    This is the sharpest available contrast, because the comparison is
    made between parent sites that share a column, and therefore share a
    position along the transport axis and a neighbourhood. A selectivity
    that differs between two motifs inside one column cannot be explained
    by how far along the surface the gas happened to reach.

    The statistic is frame weighted: in every frame in which the gas
    touches a column, each contacted parent contributes one observation.
    This answers the conditional question "given that the gas is present
    at this column, which chemistry does it preferentially touch", and is
    not diluted by how often the gas visits the column.

    Columns are grouped by X with ``column_tolerance``. Motifs with zero
    area in a column yield ``nan`` because the ratio is undefined.
    """
    motif = np.asarray(parent_motif, dtype=int)
    xy = np.asarray(parent_xy, dtype=float)
    mask = np.asarray(half_mask, dtype=bool)
    if motif.shape != xy.shape[:1] or mask.shape != motif.shape:
        raise ValueError("parent_motif, parent_xy and half_mask must agree")

    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return []
    order = selected[np.argsort(xy[selected, 0], kind="stable")]
    groups: list[list[int]] = []
    current: list[int] = []
    anchor = None
    for idx in order:
        x = xy[idx, 0]
        if anchor is None or abs(x - anchor) <= column_tolerance:
            current.append(int(idx))
            if anchor is None:
                anchor = x
        else:
            groups.append(current)
            current = [int(idx)]
            anchor = x
    if current:
        groups.append(current)

    group_of: dict[int, int] = {}
    for j, g in enumerate(groups):
        for p in g:
            group_of[p] = j
    hits = np.zeros((len(groups), 3), dtype=float)
    for parents in contacted_parents:
        arr = np.asarray(parents, dtype=int)
        arr = arr[mask[arr]] if arr.size else arr
        for p in arr:
            j = group_of.get(int(p))
            if j is not None:
                hits[j, motif[p]] += 1.0

    rows: list[dict] = []
    for j, g in enumerate(groups):
        members = np.asarray(g, dtype=int)
        area = [int(np.count_nonzero(motif[members] == k)) for k in range(3)]
        total = float(hits[j].sum())
        for k in range(3):
            value = float("nan")
            if area[k] > 0 and total > 0:
                value = float((hits[j, k] / total) / (area[k] / len(members)))
            rows.append({
                "column": j,
                "x_A": float(xy[members, 0].mean()),
                "motif": MOTIF_NAMES[k],
                "area_parents": area[k],
                "contacted_observations": float(hits[j, k]),
                "selectivity": value,
            })
    return rows


def column_motif_ratio(rows: Sequence[dict], column: int, *,
                       numerator: str = "bare",
                       denominator: str = "dual_oh") -> float:
    """Return the within-column selectivity ratio of two motifs.

    The ratio is ``nan`` when either motif is absent from the column or
    was never contacted, because the comparison is then undefined rather
    than zero.
    """
    def pick(name: str) -> float:
        for row in rows:
            if row["column"] == column and row["motif"] == name:
                value = row["selectivity"]
                return float(value) if value == value else float("nan")
        return float("nan")
    num = pick(numerator)
    den = pick(denominator)
    if num != num or den != den or den == 0.0:
        return float("nan")
    return float(num / den)


def intermittency_ratio(track: np.ndarray, column_index: np.ndarray) -> dict[str, float]:
    """Summarise how intermittent the tip track is.

    ``contact_fraction`` is the fraction of frames in which the tip exists
    at all, ``advance_fraction`` the fraction of occupied frames in which
    the tip sits on a strictly later column than the previous occupied
    frame, and ``backtrack_fraction`` the fraction in which it sits on a
    strictly earlier column.
    """
    tip = np.asarray(track, dtype=float)
    idx = np.asarray(column_index, dtype=int)
    occupied = np.flatnonzero(np.isfinite(tip))
    if occupied.size == 0:
        return {"contact_fraction": 0.0, "advance_fraction": float("nan"),
                "backtrack_fraction": float("nan"), "n_occupied": 0}
    seq = idx[occupied]
    keep = seq[seq >= 0]
    if keep.size < 2:
        return {"contact_fraction": float(occupied.size) / len(tip),
                "advance_fraction": float("nan"), "backtrack_fraction": float("nan"),
                "n_occupied": int(occupied.size)}
    diffs = np.diff(keep)
    return {"contact_fraction": float(occupied.size) / len(tip),
            "advance_fraction": float(np.count_nonzero(diffs > 0)) / len(diffs),
            "backtrack_fraction": float(np.count_nonzero(diffs < 0)) / len(diffs),
            "n_occupied": int(occupied.size)}
