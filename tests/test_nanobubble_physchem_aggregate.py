from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from molsimflow.postprocess import nanobubble_physchem_aggregate as aggregate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case(root: Path, case_id: str) -> Path:
    run = root / case_id
    physchem = run / "physchem"
    physchem.mkdir(parents=True)
    (run / "VALIDATION.json").write_text('{"status": "PASS"}\n', encoding="utf-8")
    (run / "ANALYSIS-RESULT.txt").write_text("status=PASS\n", encoding="utf-8")
    geometry = []
    thermo = []
    for time in (0.01, 8.5, 10.0):
        geometry.append({"time_ns": time, "footprint_equivalent_radius_A": 20.0 + time, "bubble_height_q05_q95_A": 10.0, "gas_side_angle_candidate_deg": 40.0, "relative_shape_anisotropy": 0.1, "largest_cluster_n2_count": 300})
        thermo.append({"time_ns": time, "Temp": 330.0, "Press": 2.0, "normal_minus_tangential_bar": 1.0})
    _write_csv(physchem / "geometry_timeseries.csv", geometry)
    _write_csv(physchem / "thermo_timeseries.csv", thermo)
    metrics = [metric for metric, _, _ in aggregate.PLOT_METRICS]
    _write_csv(physchem / "late_window_summary.csv", [{"metric": metric, "mean": 1.0, "std": 0.2, "sample_count": 2} for metric in metrics])
    records = []
    for path in sorted(path for path in run.rglob("*") if path.is_file()):
        if path.name != "OUTPUT-SHA256SUMS":
            records.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
    (run / "OUTPUT-SHA256SUMS").write_text("".join(records), encoding="utf-8")
    return run


def test_aggregate_keeps_deferred_case_in_coverage(tmp_path: Path) -> None:
    run = _case(tmp_path, "oh")
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tsurface\tcondition\tstatus\trun_dir\n"
        f"oh__pure\toh\tpure_water\tACCEPTED\t{run}\n"
        "mixed__naoh\tmixed\tnaoh_ph13p4\tDEFERRED\t\n",
        encoding="utf-8",
    )
    output = tmp_path / "aggregate"
    result = aggregate.run(aggregate.build_parser().parse_args(["--case-manifest", str(manifest), "--output-dir", str(output), "--allow-incomplete", "--no-plots"]))
    assert result["status"] == "PASS"
    assert result["coverage_complete"] is False
    assert result["accepted_case_count"] == 1
    ledger = list(csv.DictReader((output / "coverage_ledger.csv").open(encoding="utf-8")))
    assert [row["status"] for row in ledger] == ["ACCEPTED", "DEFERRED"]


def test_aggregate_rejects_invalid_case_checksum(tmp_path: Path) -> None:
    run = _case(tmp_path, "oh")
    (run / "ANALYSIS-RESULT.txt").write_text("status=FAIL\n", encoding="utf-8")
    manifest = tmp_path / "cases.tsv"
    manifest.write_text("case_id\tsurface\tcondition\tstatus\trun_dir\n" f"oh__pure\toh\tpure_water\tACCEPTED\t{run}\n", encoding="utf-8")
    try:
        aggregate.run(aggregate.build_parser().parse_args(["--case-manifest", str(manifest), "--output-dir", str(tmp_path / "aggregate"), "--no-plots"]))
    except ValueError as error:
        assert "not PASS" in str(error)
    else:
        raise AssertionError("invalid case was accepted")
