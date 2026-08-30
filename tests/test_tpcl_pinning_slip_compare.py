import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.tpcl_pinning_slip_compare import (
    LOCAL_FLOAT_FIELDS,
    CaseData,
    CaseSpec,
    LocalData,
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
