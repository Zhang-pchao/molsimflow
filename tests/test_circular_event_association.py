import csv

from molsimflow.postprocess.circular_event_association import (
    BlockColumns,
    DistanceBin,
    EventColumns,
    run_analysis,
)
from molsimflow.postprocess.tpcl_dynamic_state import LagWindow


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_signed_near_fast_association_and_block_accounting(tmp_path):
    events, blocks = [], []
    for block in range(20):
        left = block * 200.0
        for arc, time_ps in ((0, left + 20.0), (1, left + 21.0)):
            events.append(
                {
                    "system": "case",
                    "time_ps": time_ps,
                    "site": arc,
                    "site_count": 12,
                    "block": block,
                }
            )
        for duplicate in range(2):
            blocks.append(
                {
                    "system": "case",
                    "block": block,
                    "left_ps": left,
                    "right_ps": left + 200.0,
                    "events": 3,
                    "duplicate": duplicate,
                }
            )
    event_path, block_path = tmp_path / "events.csv", tmp_path / "blocks.csv"
    _write_csv(event_path, events)
    _write_csv(block_path, blocks)

    summary = run_analysis(
        event_path,
        block_path,
        tmp_path / "out",
        event_columns=EventColumns("system", "time_ps", "site", "site_count", "block"),
        block_columns=BlockColumns("system", "block", "left_ps", "right_ps", "events"),
        event_time_scale_to_ps=1.0,
        block_time_scale_to_ps=1.0,
        distance_bins=(
            DistanceBin("near", 1, 2),
            DistanceBin("mid", 2, 4),
            DistanceBin("far", 4, 7),
        ),
        windows=(LagWindow("fast", 0.0, 5.0), LagWindow("slow", 5.0, 50.0)),
        null_samples=500,
        bootstrap_samples=100,
        random_seed=7,
    )

    assert summary["event_count"] == 40
    assert summary["block_count"] == 20
    with (tmp_path / "out" / "signed_primary_associations.csv").open() as handle:
        rows = {(row["distance_bin"], row["window"]): row for row in csv.DictReader(handle)}
    near_fast = rows[("near", "fast")]
    assert int(near_fast["observed_ordered_pairs"]) == 20
    assert float(near_fast["chi_signed"]) > 0.0
    assert near_fast["association_class"] == "cooperative_event_association"
    assert float(near_fast["bh_q_primary_family"]) <= 0.05
    with (tmp_path / "out" / "block_metrics.csv").open() as handle:
        block_row = next(csv.DictReader(handle))
    assert int(block_row["source_event_row_count"]) == 3
    assert int(block_row["cluster_event_count"]) == 2
    assert (tmp_path / "out" / "same_arc_dead_time_diagnostics.csv").is_file()
    assert (tmp_path / "out" / "manifest.json").is_file()
