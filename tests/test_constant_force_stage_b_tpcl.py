import math

from molsimflow.postprocess.constant_force_stage_b_tpcl import (
    anchor_interval_rows,
    morphology_applicability,
    summarize_anchor_branch,
    summarize_region_frames,
)


def test_morphology_applicability_preserves_morphology_boundaries():
    rows = [
        {
            "case_id": "finite",
            "branch_id": "fx",
            "direction": "x",
            "morphology_class": "finite_droplet",
            "morphology_gate": "PASS",
        },
        {
            "case_id": "islands",
            "branch_id": "fx",
            "direction": "x",
            "morphology_class": "water_islands",
            "morphology_gate": "PASS",
        },
        {
            "case_id": "film",
            "branch_id": "fy",
            "direction": "y",
            "morphology_class": "spread_film",
            "morphology_gate": "PASS",
        },
    ]
    result = morphology_applicability(rows)
    assert [row["applicability"] for row in result] == [
        "DIRECTED_TPCL_APPLICABLE",
        "NOT_APPLICABLE_MULTIPLE_ISLANDS",
        "NOT_APPLICABLE_PERIODIC_FILM",
    ]


def test_region_summary_uses_count_weighted_denominators():
    rows = []
    for hbond, water, length, sites in ((1.0, 1.0, 2.0, 1.0), (9.0, 3.0, 3.0, 2.0)):
        rows.append(
            {
                "case_id": "case",
                "branch_id": "fx",
                "branch_direction": "x",
                "analysis_axis": "x",
                "region": "leading_edge",
                "surface_hbond_count": hbond,
                "contact_water_count": water,
                "contact_line_length_A": length,
                "accessible_site_count": sites,
                "ch3_fraction": 0.5,
                "water_water_hbond_degree": 2.0,
                "count_definition": "test",
            }
        )
    summary = summarize_region_frames(rows)[0]
    assert summary["surface_hbond_per_contact_water"] == 2.5
    assert summary["surface_hbond_per_contact_line_length_A-1"] == 2.0
    assert math.isclose(summary["surface_hbond_per_accessible_site"], 10.0 / 3.0)


def test_anchor_intervals_keep_empty_network_undefined():
    rows = anchor_interval_rows(
        "case",
        "fx",
        "x",
        {0: set(), 20: set(), 40: {(10, 1)}, 60: {(10, 1), (20, 2)}},
        timestep_fs=0.5,
    )
    assert math.isnan(rows[0]["anchor_pair_jaccard"])
    assert rows[1]["formed_anchor_pair_count"] == 1
    assert rows[2]["shared_anchor_pair_count"] == 1
    assert rows[2]["anchor_pair_retained_fraction"] == 1.0
    assert rows[2]["sampling_interval_ps"] == 0.01


def test_anchor_summary_reports_cross_snapshot_support_not_lifetime():
    frame_sets = {
        0: {(10, 1)},
        20: {(10, 1), (20, 2)},
        40: {(20, 2)},
    }
    intervals = anchor_interval_rows("case", "fx", "x", frame_sets, timestep_fs=0.5)
    frames = [
        {"surface_anchor_water_count": 1, "surface_anchor_site_count": 1},
        {"surface_anchor_water_count": 2, "surface_anchor_site_count": 2},
        {"surface_anchor_water_count": 1, "surface_anchor_site_count": 1},
    ]
    summary = summarize_anchor_branch("case", "fx", "x", frame_sets, frames, intervals)
    assert summary["maximum_consecutive_snapshot_support"] == 2
    assert summary["anchor_pairs_with_at_least_two_consecutive_snapshots"] == 2
    assert summary["evidence_limit"] == "10_ps_snapshot_persistence_not_hbond_lifetime"
