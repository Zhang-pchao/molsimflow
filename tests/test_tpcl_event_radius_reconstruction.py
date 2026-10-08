from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from molsimflow.postprocess.tpcl_event_radius_reconstruction import main


def _write_csv(path: Path, rows: list[dict[str, object]], delimiter: str = ",") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _case_table(path: Path, case_id: str, phase: float) -> tuple[Path, list[dict[str, object]]]:
    radius = 20.0
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    for frame in range(241):
        time_ns = frame * 0.0005
        mark = 0.0
        if frame > 5 and frame % 12 == 0:
            mark = 1.0 if (frame // 12) % 2 else -1.0
            events.append(
                {
                    "case_id": case_id,
                    "transition_time_ns": time_ns,
                    "primary_arc_index": 0,
                }
            )
        previous_mark = 0.0
        if frame > 0 and (frame - 1) > 5 and (frame - 1) % 12 == 0:
            previous_mark = 1.0 if ((frame - 1) // 12) % 2 else -1.0
        radius += 0.18 * mark + 0.08 * previous_mark + 0.002 * np.sin(frame + phase)
        for arc in (0, 1):
            rows.append(
                {
                    "time_ns": time_ns,
                    "arc_index": arc,
                    "mean_radius_component_A": radius,
                    "local_residual_displacement_A": mark if arc == 0 else 0.0,
                }
            )
    _write_csv(path, rows)
    return path, events


def test_blocked_reconstruction_writes_complete_deterministic_outputs(tmp_path: Path, monkeypatch) -> None:
    source_a, events_a = _case_table(tmp_path / "a.csv", "case_a", 0.0)
    source_b, events_b = _case_table(tmp_path / "b.csv", "case_b", 0.4)
    sources = tmp_path / "sources.tsv"
    _write_csv(
        sources,
        [
            {"case_id": "case_a", "arc_kinematics": str(source_a)},
            {"case_id": "case_b", "arc_kinematics": str(source_b)},
        ],
        delimiter="\t",
    )
    event_table = tmp_path / "events.csv"
    _write_csv(event_table, events_a + events_b)
    output = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tpcl-event-radius-reconstruction",
            "--sources-table",
            str(sources),
            "--events-table",
            str(event_table),
            "--output-dir",
            str(output),
            "--block-ps",
            "10",
            "--kernel-max-lag-ps",
            "1",
            "--ridge-penalty",
            "0.01",
            "--bootstrap-samples",
            "20",
            "--null-samples",
            "5",
            "--random-seed",
            "11",
        ],
    )
    assert main() == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["case_count"] == 2
    assert summary["total_event_count"] == summary["total_mapped_event_count"]
    assert summary["null_control_row_count"] == 2 * 2 * 5
    for name in (
        "case_coverage.csv",
        "fold_reconstruction.csv",
        "heldout_traces.csv",
        "reconstruction_summary.csv",
        "kernel_coefficients.csv",
        "null_controls.csv",
        "manifest.json",
        "REPORT.md",
    ):
        assert (output / name).is_file()
    with (output / "reconstruction_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 3
    marked = [row for row in rows if row["model"] == "event_timing_and_signed_local_residual"]
    assert all(float(row["radius_sse_improvement_vs_baseline"]) > 0.0 for row in marked)


def test_plot_mode_requires_a_new_output_directory(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "figures"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tpcl-event-radius-reconstruction",
            "--plot-results-dir",
            str(results),
            "--plot-output-dir",
            str(output),
        ],
    )
    try:
        main()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("plot mode should read required result tables")
