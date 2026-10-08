import csv

from molsimflow.postprocess.circular_event_association import (
    BlockColumns,
    DistanceBin,
    EventColumns,
)
from molsimflow.postprocess.circular_event_injection_power import run_power
from molsimflow.postprocess.tpcl_dynamic_state import LagWindow


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_injection_power_writes_complete_kernel_grid(tmp_path):
    events, blocks = [], []
    for block in range(3):
        left = block * 20.0
        for arc, time_ps in ((0, left + 2.0), (1, left + 3.0)):
            events.append(
                {"case": "x", "time": time_ps, "arc": arc, "narc": 8, "block": block}
            )
        blocks.append(
            {"case": "x", "block": block, "left": left, "right": left + 20.0, "count": 4}
        )
    event_path, block_path = tmp_path / "events.csv", tmp_path / "blocks.csv"
    _write_csv(event_path, events)
    _write_csv(block_path, blocks)
    bins = (
        DistanceBin("near", 1, 2),
        DistanceBin("mid", 2, 3),
        DistanceBin("far", 3, 5),
    )
    windows = (LagWindow("fast", 0.0, 2.0), LagWindow("slow", 2.0, 5.0))
    baseline = tmp_path / "baseline.csv"
    _write_csv(
        baseline,
        [
            {
                "case_id": "x",
                "distance_bin": distance.name,
                "window": window.name,
                "empirical_two_sided_p": 1.0,
                "chi_signed": 0.0,
                "bh_q_primary_family": 1.0,
                "association_class": "not_qualified",
            }
            for distance in bins
            for window in windows
        ],
    )

    summary = run_power(
        event_path,
        block_path,
        baseline,
        tmp_path / "out",
        event_columns=EventColumns("case", "time", "arc", "narc", "block"),
        block_columns=BlockColumns("case", "block", "left", "right", "count"),
        event_time_scale_to_ps=1.0,
        block_time_scale_to_ps=1.0,
        distance_bins=bins,
        windows=windows,
        branching_probabilities=(1.0,),
        frame_ps=1.0,
        null_samples=20,
        injection_replicates=2,
        random_seed=11,
        workers=2,
    )

    assert summary["combination_count"] == 6
    assert summary["replicate_row_count"] == 12
    with (tmp_path / "out" / "injection_replicates.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(int(row["attempted_child_count"]) == 6 for row in rows)
    assert all(0.0 <= float(row["target_bh_q_primary_family"]) <= 1.0 for row in rows)
    assert (tmp_path / "out" / "power_summary.csv").is_file()
    assert (tmp_path / "out" / "recovery_thresholds.csv").is_file()
