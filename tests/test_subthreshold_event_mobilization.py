import csv

from molsimflow.postprocess.circular_event_association import DistanceBin
from molsimflow.postprocess.subthreshold_event_mobilization import RatioBin, run_analysis
from molsimflow.postprocess.tpcl_dynamic_state import LagWindow


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_mobilization_excludes_primary_members_and_detects_secondary_events(tmp_path):
    cluster_path = tmp_path / "clusters.csv"
    _write(
        cluster_path,
        [
            {
                "primary_event_id": 1,
                "transition_time_ns": 0.010,
                "primary_arc_index": 0,
                "member_arcs": "0;5",
                "affected_threshold_A": 1.0,
            },
            {
                "primary_event_id": 2,
                "transition_time_ns": 0.012,
                "primary_arc_index": 1,
                "member_arcs": "1",
                "affected_threshold_A": 1.0,
            },
            {
                "primary_event_id": 3,
                "transition_time_ns": 0.110,
                "primary_arc_index": 0,
                "member_arcs": "0",
                "affected_threshold_A": 1.0,
            },
            {
                "primary_event_id": 4,
                "transition_time_ns": 0.130,
                "primary_arc_index": 1,
                "member_arcs": "1",
                "affected_threshold_A": 1.0,
            },
        ],
    )
    block_path = tmp_path / "blocks.csv"
    _write(
        block_path,
        [
            {
                "case_id": "x",
                "block_index": block,
                "block_start_ps": block * 100.0,
                "block_end_ps": (block + 1) * 100.0,
                "event_count": 3,
            }
            for block in range(2)
        ],
    )
    response_rows = []
    for event_id, block in ((1, 0), (3, 1)):
        for offset in (-3, -2, -1, 0, 1, 2):
            change = 1.2 if event_id == 1 and offset == 1 else 0.8
            response_rows.append(
                {
                    "case_id": "x",
                    "event_id": event_id,
                    "time_block_200ps": block,
                    "primary_arc_index": 0,
                    "field": "residual",
                    "lag_ps": 0.5,
                    "arc_offset_signed": offset,
                    "arc_distance": abs(offset),
                    "aligned_change": change,
                }
            )
    response_path = tmp_path / "responses.csv"
    _write(response_path, response_rows)

    output = tmp_path / "out"
    summary = run_analysis(
        response_path,
        {"x": cluster_path},
        block_path,
        output,
        response_field="residual",
        response_lag_ps=0.5,
        distance_bins=(DistanceBin("near", 1, 2), DistanceBin("far", 2, 4)),
        ratio_bins=(
            RatioBin("low", 0.0, 0.5),
            RatioBin("near", 0.5, 1.0),
            RatioBin("mobilized", 1.0, float("inf")),
        ),
        windows=(LagWindow("fast", 0.0, 5.0), LagWindow("slow", 5.0, 50.0)),
        mobilization_threshold_ratio=1.0,
        null_samples=20,
        bootstrap_samples=20,
        random_seed=7,
    )

    assert summary["admitted_primary_event_count"] == 2
    assert summary["excluded_primary_cluster_member_arc_count"] == 1
    assert summary["cross_arc_opportunity_count"] == 9
    with (output / "event_arc_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    event_one_target_one_fast = next(
        row
        for row in rows
        if row["primary_event_id"] == "1"
        and row["target_arc_index"] == "1"
        and row["window"] == "fast"
    )
    assert event_one_target_one_fast["mobilized"] == "True"
    assert event_one_target_one_fast["secondary_event_detected"] == "True"
    assert not any(
        row["primary_event_id"] == "1" and row["target_arc_index"] == "5" for row in rows
    )
    assert (output / "mobilization_conversion_summary.csv").is_file()
    assert (output / "manifest.json").is_file()
