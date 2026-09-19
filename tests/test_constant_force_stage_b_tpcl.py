import math

from molsimflow.postprocess.constant_force_stage_b_tpcl import (
    anchor_interval_rows,
    matched_control_definitions,
    matched_response_rows,
    morphology_applicability,
    summarize_anchor_branch,
    summarize_region_frames,
    surface_site_residence_rows,
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


def test_surface_site_residence_counts_intermittent_snapshot_support():
    rows = surface_site_residence_rows(
        "case",
        "fx",
        "x",
        {
            0: {(10, 1)},
            20: {(10, 1), (20, 2)},
            40: {(30, 1)},
            60: set(),
            80: {(10, 1)},
        },
        [
            {"atom_id": "1", "site_type": "SiOH", "x_A": "1", "y_A": "2"},
            {"atom_id": "2", "site_type": "SiOH", "x_A": "3", "y_A": "4"},
        ],
        timestep_fs=0.5,
    )
    first, second = rows
    assert first["occupied_snapshot_count"] == 4
    assert first["occupancy_spell_count"] == 2
    assert first["unique_anchor_water_count"] == 2
    assert first["maximum_consecutive_snapshot_support"] == 3
    assert first["maximum_supported_residence_ps"] == 0.02
    assert second["occupied_snapshot_fraction"] == 0.2


def test_matched_controls_are_equal_width_and_do_not_overlap_events():
    events = [
        {
            "event_id": "slow",
            "event_class": "slow",
            "mechanism_label": "DWELL_CANDIDATE",
            "event_center_ps": 300.0,
            "event_score": -1.0,
        },
        {
            "event_id": "fast",
            "event_class": "fast",
            "mechanism_label": "ADVANCE_CANDIDATE",
            "event_center_ps": 700.0,
            "event_score": 1.0,
        },
    ]
    controls = matched_control_definitions(
        events,
        list(range(0, 2001, 50)),
        half_window_ps=100.0,
    )
    centers = [row["control_center_ps"] for row in controls]
    assert len(set(centers)) == 2
    assert all(100.0 <= center <= 1900.0 for center in centers)
    assert all(
        abs(center - event_center) > 200.0 for center in centers for event_center in (300, 700)
    )


def test_matched_response_reports_difference_in_differences():
    common = {
        "case_id": "case",
        "branch_id": "fx",
        "direction": "x",
        "event_id": "event",
        "event_class": "fast",
        "mechanism_label": "ADVANCE_CANDIDATE",
        "half_window_ps": 100.0,
    }
    result = matched_response_rows(
        [{**common, "event_center_ps": 300.0, "delta_anchor": 2.5}],
        [{**common, "event_center_ps": 800.0, "delta_anchor": 0.5}],
        [
            {
                "event_id": "event",
                "control_center_ps": 800.0,
                "event_to_control_time_distance_ps": 500.0,
                "control_definition": "nearest_time_nonoverlapping_equal_window",
            }
        ],
    )[0]
    assert result["event_minus_control_delta_anchor"] == 2.0
