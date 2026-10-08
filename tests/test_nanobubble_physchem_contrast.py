from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from molsimflow.postprocess import nanobubble_physchem_contrast as contrast


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case(root: Path, case_id: str, offset: float = 0.0) -> Path:
    run = root / case_id
    physchem = run / "physchem"
    physchem.mkdir(parents=True)
    (run / "VALIDATION.json").write_text('{"status": "PASS"}\n', encoding="utf-8")
    (run / "ANALYSIS-RESULT.txt").write_text("status=PASS\n", encoding="utf-8")
    geometry = []
    thermo = []
    for index in range(1, 4):
        time = index * 0.01
        geometry.append(
            {
                "time_ns": time,
                "largest_cluster_n2_count": 300.0 - offset,
                "dissolved_or_disconnected_n2_count": offset,
                "bubble_height_q05_q95_A": 20.0 + offset,
                "bubble_lateral_displacement_A": 1.0,
                "relative_shape_anisotropy": 0.1,
                "footprint_equivalent_radius_A": 10.0 + offset,
                "gas_side_angle_candidate_deg": 100.0,
            }
        )
        thermo.append({"time_ns": time, "Temp": 330.0, "Press": 1.0, "normal_minus_tangential_bar": 0.0})
    _write_csv(physchem / "geometry_timeseries.csv", geometry)
    _write_csv(physchem / "thermo_timeseries.csv", thermo)
    _write_csv(
        physchem / "late_window_summary.csv",
        [
            {"metric": metric, "mean": 1.0, "std": 0.1, "sample_count": 3}
            for metric in (
                *contrast.METRICS,
                "gas_side_angle_candidate_deg",
                "Temp",
                "Press",
                "normal_minus_tangential_bar",
            )
        ],
    )
    records = []
    for path in sorted(path for path in run.rglob("*") if path.is_file()):
        records.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
    (run / "OUTPUT-SHA256SUMS").write_text("".join(records), encoding="utf-8")
    return run


def test_common_grid_contrast_keeps_deferred_case(tmp_path: Path) -> None:
    pure = _case(tmp_path, "pure")
    hcl = _case(tmp_path, "hcl", offset=2.0)
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tsurface\tcondition\tstatus\trun_dir\n"
        f"oh__pure\toh\tpure_water\tACCEPTED\t{pure}\n"
        f"oh__hcl\toh\thcl_63pairs\tACCEPTED\t{hcl}\n"
        "mixed__naoh\tmixed\tnaoh_ph13p4\tDEFERRED\t\n",
        encoding="utf-8",
    )
    output = tmp_path / "contrast"
    result = contrast.run(
        contrast.build_parser().parse_args(
            [
                "--case-manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--allow-incomplete",
                "--required-end-ns",
                "0.03",
                "--late-start-ns",
                "0.01",
                "--late-end-ns",
                "0.03",
                "--block-ns",
                "0.01",
                "--no-plots",
            ]
        )
    )
    assert result["status"] == "PASS"
    assert result["deferred_case_count"] == 1
    ledger = list(csv.DictReader((output / "coverage_ledger.csv").open(encoding="utf-8")))
    assert any(row["status"] == "DEFERRED" for row in ledger)
    summary = list(csv.DictReader((output / "late_contrast_summary.csv").open(encoding="utf-8")))
    height = next(row for row in summary if row["metric"] == "bubble_height_q05_q95_A")
    assert float(height["mean_effect_condition_minus_reference"]) == 2.0


def test_common_grid_rejects_invalid_case_checksum(tmp_path: Path) -> None:
    pure = _case(tmp_path, "pure")
    (pure / "ANALYSIS-RESULT.txt").write_text("status=FAIL\n", encoding="utf-8")
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tsurface\tcondition\tstatus\trun_dir\n"
        f"oh__pure\toh\tpure_water\tACCEPTED\t{pure}\n",
        encoding="utf-8",
    )
    try:
        contrast.run(
            contrast.build_parser().parse_args(
                [
                    "--case-manifest",
                    str(manifest),
                    "--output-dir",
                    str(tmp_path / "contrast"),
                    "--required-end-ns",
                    "0.03",
                    "--late-start-ns",
                    "0.01",
                    "--late-end-ns",
                    "0.03",
                    "--block-ns",
                    "0.01",
                    "--no-plots",
                ]
            )
        )
    except ValueError as error:
        assert "not PASS" in str(error)
    else:
        raise AssertionError("invalid case was accepted")
