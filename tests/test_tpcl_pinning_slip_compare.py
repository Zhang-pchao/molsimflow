import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.tpcl_pinning_slip_compare import (
    LOCAL_FLOAT_FIELDS,
    CaseData,
    CaseSpec,
    LocalData,
    _kind_styles,
    _write_claim_ledger,
    _write_no_event_figures,
    build_case_summary,
    build_important_data_rows,
    circular_shift_null,
    frame_local_means,
    summarize_null,
    threshold_sensitivity,
)


def _synthetic_case(tmp_path):
    steps = np.repeat([0, 10, 20, 30], 4)
    arcs = np.tile(np.arange(4), 4)
    values = {name: np.ones(16) for name in LOCAL_FLOAT_FIELDS}
    values["chemical_boundary_proxy_A"] = np.where(arcs == 0, 0.5, 5.0)
    values["local_ch3_fraction"] = np.where(arcs == 0, 0.25, 0.75)
    local = LocalData(
        step=steps,
        time_ns=steps.astype(float),
        arc=arcs,
        segment=np.zeros(16, dtype=int),
        nearest_ch3=np.where(arcs == 0, 0.0, 1.0),
        values=values,
    )
    events = []
    for cluster_id, start in ((1, 0), (2, 20)):
        events.append(
            {
                "quality_status": "candidate_stick_slip",
                "event_cluster_id": str(cluster_id),
                "arc_index": "0",
                "start_step": str(start),
                "end_step": str(start + 10),
                "transition_step": str(start + 10),
                "post_end_step": str(start + 10),
                "jump_distance_A": str(2.0 + cluster_id),
                "cluster_mechanism_class": "boundary",
            }
        )
    return CaseData(
        spec=CaseSpec("mixed", "nanodroplet", tmp_path, 2, 2, None),
        job_id="1",
        results_root=tmp_path,
        summary={"expected_step_interval": 10, "arc_bins": 4, "jump_threshold_A": 2.0},
        frames=[],
        local=local,
        events=events,
    )


def test_compare_cli_and_frame_aggregation(tmp_path):
    args = build_parser().parse_args(
        [
            "postprocess",
            "tpcl-pinning-slip-compare",
            "--manifest",
            str(tmp_path / "cases.tsv"),
            "--output-dir",
            str(tmp_path / "out"),
            "--font-path",
            str(tmp_path / "Arial.ttf"),
        ]
    )
    assert args.postprocess_command == "tpcl-pinning-slip-compare"
    case = _synthetic_case(tmp_path)
    steps, means = frame_local_means(case.local, ("local_ch3_fraction",))
    np.testing.assert_array_equal(steps, [0, 10, 20, 30])
    np.testing.assert_allclose(means["local_ch3_fraction"], 0.625)


def test_circular_shift_null_and_stricter_sensitivity(tmp_path):
    case = _synthetic_case(tmp_path)
    rows = circular_shift_null(case)
    assert len(rows) == 4
    observed = next(row for row in rows if row["is_observed"])
    assert observed["boundary_event_fraction"] == 1.0
    assert all(row["boundary_event_fraction"] == 0.0 for row in rows if not row["is_observed"])
    null_summary = summarize_null(rows)
    assert len(null_summary) == 1
    assert 0.0 <= null_summary[0]["bh_q_boundary_event_fraction"] <= 1.0

    sensitivity = threshold_sensitivity(case, (1.0, 1.5, 2.0))
    assert [row["retained_repeated_event_clusters"] for row in sensitivity] == [2, 2, 0]


def test_homogeneous_common_window_and_important_data(tmp_path):
    case = _synthetic_case(tmp_path)
    case.summary.update(
        {
            "frame_interval_ps": 10.0,
            "valid_contour_frames": 4,
            "contour_valid_fraction": 1.0,
            "contour_valid_fraction_after_first_valid": 1.0,
            "localization_noise_A": 0.2,
            "candidate_arc_record_count": 2,
        }
    )
    case.frames = [
        {
            "step": str(step),
            "time_ns": str(float(step)),
            "contour_valid": "True",
            "contour_invalid_reason": "",
        }
        for step in (0, 10, 20, 30)
    ]
    case.spec = CaseSpec(
        "mixed",
        "nanodroplet",
        tmp_path,
        2,
        2,
        None,
        comparison_group="droplet-series",
        fair_window_mode="common_valid_overlap",
    )
    clusters = [
        {
            "case_id": "mixed",
            "mechanism_class": "boundary",
            "transition_time_ns": 10.0,
        },
        {
            "case_id": "mixed",
            "mechanism_class": "CH3",
            "transition_time_ns": 20.0,
        },
    ]
    rows = build_case_summary([case], clusters)
    assert rows[0]["fair_window"] == "common_valid_overlap:droplet-series"
    assert rows[0]["fair_window_comparison_status"] == "ADMITTED"
    important = build_important_data_rows(rows)
    assert important[0]["candidate_dwell_jump_clusters"] == 2
    assert important[0]["comparison_status"] == "ADMITTED"


def test_claim_ledger_marks_zero_event_analysis_not_applicable(tmp_path):
    output = tmp_path / "ledger.md"
    _write_claim_ledger(
        output,
        [{"fair_window_comparison_status": "ADMITTED", "fair_event_clusters": 0}],
        [],
    )
    text = output.read_text(encoding="utf-8")
    assert "no repeated coarse dwell--jump candidate" in text
    assert "Not applicable: no registered candidate event window" in text


def test_zero_event_figures_show_condition_level_observables(tmp_path):
    from matplotlib import pyplot as plt

    case = _synthetic_case(tmp_path)
    case.events = []
    case.frames = [
        {
            "time_ns": str(time_ns),
            "decomposed_mean_radius_A": str(radius),
        }
        for time_ns, radius in ((0.0, 10.0), (1.0, 11.0), (2.0, 10.5))
    ]
    case_rows = [
        {
            "case_id": "mixed",
            "kind": "nanodroplet",
            "ch3_fraction": 0.5,
            "time_mean_tpcl_radius_A": 10.5,
            "time_mean_footprint_area_A2": 300.0,
            "time_mean_contact_line_circularity": 0.9,
            "fair_window_comparison_status": "ADMITTED",
        }
    ]
    metrics = (
        "local_contact_angle_deg",
        "local_hydration_areal_density_A-2",
        "local_water_water_hbond_degree",
        "local_surface_water_hbond_per_h2o",
    )
    block_rows = [
        {
            "case_id": "mixed",
            "kind": "nanodroplet",
            "ch3_fraction": 0.5,
            "metric": metric,
            "mean": 1.0,
            "bootstrap_ci025": 0.8,
            "bootstrap_ci975": 1.2,
        }
        for metric in metrics
    ]
    sensitivity_rows = [
        {
            "case_id": "mixed",
            "kind": "nanodroplet",
            "jump_threshold_multiplier": 1.0,
            "retained_repeated_event_clusters": 0,
        }
    ]
    figures = tmp_path / "figures"
    figures.mkdir()
    _write_no_event_figures(
        [case],
        case_rows,
        block_rows,
        sensitivity_rows,
        figures,
        _kind_styles([case]),
        plt,
    )
    assert (figures / "01_geometry_wetting_state.png").is_file()
    assert (figures / "02_tpcl_radius_timeseries.png").is_file()
    assert (figures / "03_block_environment_by_chemistry.png").is_file()
    assert (figures / "04_detector_robustness.png").is_file()
    assert not (figures / "02_event_aligned_environment.png").exists()
