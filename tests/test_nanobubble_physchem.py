from __future__ import annotations

import csv
import json
from pathlib import Path

from molsimflow.postprocess.nanobubble_physchem import main


def _write_core(path: Path) -> None:
    fields = [
        "step", "largest_cluster_n2_count", "dissolved_or_disconnected_n2_count",
        "bubble_height_q05_q95_A", "footprint_convex_hull_area_A2",
        "relative_shape_anisotropy", "bubble_lateral_displacement_A",
    ]
    rows = [
        [0, 300, 0, 20, 100, 0.01, 0],
        [2_000_000, 300, 0, 20, 100, 0.02, 1],
        [4_000_000, 100, 200, 5, 0, 0.3, 2],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


def _write_log(path: Path, values: list[tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Step Time Temp Density PotEng KinEng TotEng Volume Press Pxx Pyy Pzz Pxy Pxz Pyz\n")
        for step, temp in values:
            handle.write(f"{step} {step * 0.0005} {temp} 1 -10 1 -9 1000 2 1 2 3 0 0 0\n")


def test_restart_merge_and_fragment_mask(tmp_path: Path) -> None:
    core = tmp_path / "core.csv"; _write_core(core)
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    _write_log(first, [(0, 300), (2_000_000, 301)])
    _write_log(second, [(2_000_000, 310), (4_000_000, 311)])
    manifest = tmp_path / "thermo.tsv"
    manifest.write_text(
        "path\tmin_step\tmax_step\tpriority\n"
        f"{first}\t0\t2000000\t1\n{second}\t2000000\t4000000\t2\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    assert main(["--case-id", "case", "--core-metrics", str(core), "--thermo-manifest", str(manifest), "--output-dir", str(output), "--no-plots"]) == 0
    thermo = list(csv.DictReader((output / "thermo_timeseries.csv").open()))
    geometry = list(csv.DictReader((output / "geometry_timeseries.csv").open()))
    assert [int(row["step"]) for row in thermo] == [0, 2_000_000, 4_000_000]
    assert float(thermo[1]["Temp"]) == 310.0
    assert geometry[-1]["fragmented"] == "True"
    assert geometry[-1]["geometry_valid"] == "False"
    assert json.loads((output / "VALIDATION.json").read_text())["status"] == "PASS"
