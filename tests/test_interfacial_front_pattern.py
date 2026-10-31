"""Tests for the chemical-pattern front gateway module."""

import numpy as np
import pytest

from molsimflow.postprocess.interfacial_front_pattern import (
    build_column_map,
    column_advance_hazard,
    column_motif_ratio,
    contacted_parent_motif_counts,
    footprint_coherence,
    frame_contacted_parents,
    hazard_by_label,
    front_tip_track,
    intermittency_ratio,
    motif_selectivity,
    per_column_motif_selectivity,
    resolve_to_columns,
)


def _staggered_grid(n_columns: int = 6, n_rows: int = 3, spacing: float = 2.47):
    """Return a simple staggered parent grid with deterministic X columns."""
    xy = []
    for j in range(n_columns):
        for i in range(n_rows):
            offset = 0.5 * spacing if j % 2 else 0.0
            xy.append((10.0 + j * spacing, 5.0 + i * 4.28 + offset))
    return np.array(xy, dtype=float)


def test_column_map_groups_by_x():
    xy = _staggered_grid(n_columns=6)
    motif = np.zeros(len(xy), dtype=int)
    mask = np.ones(len(xy), dtype=bool)
    cmap = build_column_map(xy, motif, half_mask=mask)
    assert cmap.n_columns == 6
    assert np.allclose(cmap.column_x, 10.0 + 2.47 * np.arange(6), atol=1e-9)
    for j in range(6):
        assert cmap.column_is_uniform(j)
        assert cmap.column_motif_label(j) == "bare"
        assert cmap.parent_motif[cmap.members[j]].size == 3


def test_column_map_motif_labels_and_pattern():
    xy = _staggered_grid(n_columns=4)
    motif = np.zeros(len(xy), dtype=int)
    # column 0 -> dual OH, column 1 -> bare, column 2 -> mixed
    members = {}
    for j, x in enumerate(np.unique(np.round(xy[:, 0], 6))):
        members[j] = np.flatnonzero(np.isclose(xy[:, 0], x))
    motif[members[0]] = 2
    motif[members[1]] = 0
    motif[members[2][:1]] = 2
    motif[members[2][1:]] = 1
    cmap = build_column_map(xy, motif, half_mask=np.ones(len(xy), dtype=bool))
    assert cmap.column_motif_label(0) == "dual_oh"
    assert cmap.column_motif_label(1) == "bare"
    assert cmap.column_motif_label(2) == "mixed"
    assert cmap.column_is_uniform(2) is False
    assert cmap.pattern_string()[0] == "W"
    assert cmap.pattern_string()[1] == "H"
    assert cmap.pattern_string()[2] == "M"


def test_alternating_period_requires_two_stripes():
    xy = _staggered_grid(n_columns=4)
    motif = np.zeros(len(xy), dtype=int)
    for j, x in enumerate(np.unique(np.round(xy[:, 0], 6))):
        members = np.flatnonzero(np.isclose(xy[:, 0], x))
        if j % 2 == 0:
            motif[members] = 2
    cmap = build_column_map(xy, motif, half_mask=np.ones(len(xy), dtype=bool))
    assert cmap.alternating_period_A() > 0.0
    # a single stripe has no defined period
    motif2 = np.zeros(len(xy), dtype=int)
    members = np.flatnonzero(np.isclose(xy[:, 0], np.unique(np.round(xy[:, 0], 6))[0]))
    motif2[members] = 2
    cmap2 = build_column_map(xy, motif2, half_mask=np.ones(len(xy), dtype=bool))
    assert np.isnan(cmap2.alternating_period_A())


def test_half_mask_selects_subset():
    xy = _staggered_grid(n_columns=4)
    motif = np.zeros(len(xy), dtype=int)
    mask = xy[:, 0] < 13.0
    cmap = build_column_map(xy, motif, half_mask=mask)
    assert cmap.n_columns == 2
    assert np.all(cmap.parent_column[~mask] == -1)


def test_front_tip_track_uses_max_x_and_nan():
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0], [20.0, 0.0]])
    contacted = [np.array([0, 1]), np.array([], dtype=int), np.array([3, 2])]
    track = front_tip_track(contacted, xy, np.ones(4, dtype=bool))
    assert track[0] == 5.0
    assert np.isnan(track[1])
    # the leading contacted parent is the one with the largest X
    assert track[2] == 20.0


def test_front_tip_track_respects_half_mask():
    xy = np.array([[0.0, 0.0], [5.0, 0.0]])
    contacted = [np.array([0, 1])]
    track = front_tip_track(contacted, xy, np.array([True, False]))
    assert track[0] == 0.0


def test_resolve_to_columns_rejects_far_tips():
    cols = np.array([10.0, 12.47, 14.94])
    track = np.array([10.0, 12.47, 100.0, np.nan])
    idx = resolve_to_columns(track, cols, tolerance=0.25)
    assert idx[0] == 0
    assert idx[1] == 1
    assert idx[2] == -1
    assert idx[3] == -1


def test_advance_hazard_excludes_censored_columns():
    # tip visits columns 0,1,2 and never leaves column 2
    idx = np.array([0, 0, 1, 1, 1, 2, 2])
    res = column_advance_hazard(idx, n_columns=4, lookahead=2)
    assert res[0]["censored"] is False
    assert res[0]["hazard"] == 1.0
    assert res[2]["censored"] is True
    assert np.isnan(res[2]["hazard"])
    assert res[3]["n_occupied"] == 0


def test_advance_hazard_counts_forward_only():
    idx = np.array([0, 1, 0, 1, 0, 1])
    res = column_advance_hazard(idx, n_columns=2, lookahead=1)
    # column 0 frames: after f=0 tip goes to 1 -> advanced; f=2 -> 1 advanced; f=4 -> 1 advanced
    assert res[0]["n_evaluable"] == 3
    assert res[0]["n_advanced"] == 3
    assert res[0]["hazard"] == 1.0
    # column 1 never left -> censored
    assert res[1]["censored"] is True


def test_advance_hazard_rejects_bad_lookahead():
    with pytest.raises(ValueError):
        column_advance_hazard(np.array([0]), 1, lookahead=0)


def test_motif_selectivity_uniform_face_is_one():
    motif = np.array([0, 0, 2, 2])
    mask = np.array([True, True, True, True])
    counts = np.array([10.0, 0.0, 0.0])
    sel = motif_selectivity(counts, motif, half_mask=mask)
    # contacted bare fraction is 1.0 while the bare area fraction is 0.5
    assert sel[0] == pytest.approx(2.0)
    # no single-OH site exists on this face, so the ratio is undefined
    assert np.isnan(sel[1])
    # dual-OH exists but was never contacted
    assert sel[2] == pytest.approx(0.0)


def test_motif_selectivity_zero_area_is_nan_not_zero():
    motif = np.array([0, 0])
    mask = np.array([True, True])
    counts = np.array([1.0, 0.0, 0.0])
    sel = motif_selectivity(counts, motif, half_mask=mask)
    assert sel[0] == pytest.approx(1.0)
    assert np.isnan(sel[1])
    assert np.isnan(sel[2])


def test_contacted_parent_motif_counts():
    motif = np.array([0, 1, 2, 2])
    contacted = [np.array([0, 1]), np.array([2, 3])]
    counts = contacted_parent_motif_counts(contacted, motif)
    assert counts.tolist() == [1.0, 1.0, 2.0]


def test_contacted_parent_motif_counts_respects_half_mask():
    motif = np.array([0, 1, 2, 2])
    contacted = [np.array([0, 1, 2, 3])]
    counts = contacted_parent_motif_counts(contacted, motif,
                                           half_mask=np.array([True, True, False, False]))
    assert counts.tolist() == [1.0, 1.0, 0.0]


def test_frame_contacted_parents_splits_by_offsets():
    # two frames; column 0 is a molecule index, not a frame index
    edges = np.array([[0, 5], [3, 6], [7, 7], [9, 8]])
    offsets = np.array([0, 2, 4])
    parent_index = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    frames = frame_contacted_parents(edges, offsets, parent_index)
    assert len(frames) == 2
    # frame 0 uses terminal sites 5 and 6 -> parents 1 and 2
    assert frames[0].tolist() == [1, 2]
    # frame 1 uses terminal sites 7 and 8 -> parents 2 (site 7) and 2 (site 8)
    assert frames[1].tolist() == [2]


def test_frame_contacted_parents_validates_offsets():
    with pytest.raises(ValueError):
        frame_contacted_parents(np.array([[0, 1]]), np.array([0, 5]), np.array([0]))


def test_intermittency_ratio():
    track = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 2.0])
    idx = np.array([0, -1, 1, 2, -1, 1])
    res = intermittency_ratio(track, idx)
    assert res["n_occupied"] == 4
    assert res["contact_fraction"] == pytest.approx(4 / 6)
    assert res["advance_fraction"] == pytest.approx(2 / 3)
    assert res["backtrack_fraction"] == pytest.approx(1 / 3)


def test_per_column_selectivity_separates_motifs_in_one_column():
    # one column at x=10 holding one bare and one dual-OH parent
    xy = np.array([[10.0, 0.0], [10.0, 4.28]])
    motif = np.array([0, 2])
    mask = np.array([True, True])
    contacted = [np.array([0]), np.array([0, 1])]
    rows = per_column_motif_selectivity(contacted, motif, xy, half_mask=mask)
    assert len(rows) == 3
    by_motif = {r["motif"]: r for r in rows}
    # frame weighted: bare was touched twice, dual-OH once, over equal areas
    assert by_motif["bare"]["selectivity"] == pytest.approx(4.0 / 3.0)
    assert by_motif["dual_oh"]["selectivity"] == pytest.approx(2.0 / 3.0)
    assert np.isnan(by_motif["single_oh"]["selectivity"])
    assert column_motif_ratio(rows, 0) == pytest.approx(2.0)


def test_per_column_selectivity_absent_motif_is_nan_ratio():
    xy = np.array([[10.0, 0.0], [10.0, 4.28]])
    motif = np.array([0, 0])
    mask = np.array([True, True])
    rows = per_column_motif_selectivity([np.array([0, 1])], motif, xy, half_mask=mask)
    assert np.isnan(column_motif_ratio(rows, 0))


def test_per_column_selectivity_untouched_motif_gives_nan_ratio():
    xy = np.array([[10.0, 0.0], [10.0, 4.28]])
    motif = np.array([0, 2])
    mask = np.array([True, True])
    rows = per_column_motif_selectivity([np.array([0]), np.array([0])],
                                        motif, xy, half_mask=mask)
    by_motif = {r["motif"]: r for r in rows}
    assert by_motif["bare"]["selectivity"] == pytest.approx(2.0)
    assert by_motif["dual_oh"]["selectivity"] == pytest.approx(0.0)
    # a zero denominator makes the ratio undefined rather than infinite
    assert np.isnan(column_motif_ratio(rows, 0))


def test_footprint_coherence_separates_size_from_patchiness():
    # Three contacted parents in a row: one connected patch of size 3.
    xy = np.array([[0.0, 0.0], [2.47, 0.0], [4.94, 0.0]])
    mask = np.array([True, True, True])
    res = footprint_coherence([np.array([0, 1, 2])], xy, mask)
    assert res["median_parents"] == pytest.approx(3.0)
    assert res["median_largest_component"] == pytest.approx(3.0)
    assert res["median_n_components"] == pytest.approx(1.0)
    assert res["mean_largest_fraction"] == pytest.approx(1.0)


def test_footprint_coherence_detects_disconnected_patches():
    # Two parents 10 A apart touch the gas but do not form one basal patch.
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    mask = np.array([True, True])
    res = footprint_coherence([np.array([0, 1])], xy, mask)
    assert res["median_parents"] == pytest.approx(2.0)
    assert res["median_largest_component"] == pytest.approx(1.0)
    assert res["median_n_components"] == pytest.approx(2.0)
    assert res["mean_largest_fraction"] == pytest.approx(0.5)


def test_footprint_coherence_ignores_other_half():
    # A parent on the other half-face is excluded from the footprint.
    xy = np.array([[0.0, 0.0], [0.1, 0.0]])
    mask = np.array([True, False])
    res = footprint_coherence([np.array([0, 1])], xy, mask)
    assert res["median_parents"] == pytest.approx(1.0)
    assert res["median_largest_component"] == pytest.approx(1.0)


def test_footprint_coherence_empty_is_nan():
    xy = np.array([[0.0, 0.0]])
    mask = np.array([True])
    res = footprint_coherence([np.array([], dtype=int)], xy, mask)
    assert res["n_frames"] == 0
    assert np.isnan(res["median_parents"])


def test_hazard_by_label_pools_over_columns_and_excludes_censored():
    rows = [
        {"label": "dual_oh", "n_evaluable_frames": 10, "n_advanced": 2,
         "right_censored": False},
        {"label": "dual_oh", "n_evaluable_frames": 10, "n_advanced": 3,
         "right_censored": False},
        {"label": "dual_oh", "n_evaluable_frames": 99, "n_advanced": 99,
         "right_censored": True},
        {"label": "mixed", "n_evaluable_frames": 20, "n_advanced": 18,
         "right_censored": False},
    ]
    out = {r["label"]: r for r in hazard_by_label(rows)}
    assert out["dual_oh"]["n_evaluable_frames"] == 20
    assert out["dual_oh"]["n_advanced"] == 5
    assert out["dual_oh"]["pooled_advance_hazard"] == pytest.approx(0.25)
    assert out["dual_oh"]["n_columns"] == 2
    assert out["mixed"]["pooled_advance_hazard"] == pytest.approx(0.9)


def test_hazard_by_label_drops_fully_censored_labels():
    rows = [{"label": "dual_oh", "n_evaluable_frames": 0, "n_advanced": 0,
             "right_censored": True}]
    assert hazard_by_label(rows) == []
