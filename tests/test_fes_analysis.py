import csv
import math

import numpy as np

from molsimflow.postprocess.fes_analysis import (
    FesCurveSpec,
    analyze_fes_barriers,
    load_curve_manifest,
    load_fes_curve,
    moving_average_smooth,
    parse_window,
    process_curve,
)


def _write_curve(path, values):
    with path.open("w") as handle:
        handle.write("#! FIELDS cv free_energy uncertainty\n")
        for cv, free_energy, uncertainty in values:
            handle.write(f"{cv} {free_energy} {uncertainty}\n")


def test_load_fes_curve_and_zeroed_processing(tmp_path):
    curve_path = tmp_path / "fes.dat"
    _write_curve(
        curve_path,
        [
            (0.0, 5.0, 0.1),
            (1.0, 1.0, 0.1),
            (2.0, 7.0, 0.2),
            (3.0, 3.0, 0.2),
        ],
    )

    curve = load_fes_curve(FesCurveSpec(curve_path, "caseA", "groupA", "a"))
    rows, shifted, smoothed, zeroed = process_curve(curve, smooth_window=1)

    assert len(rows) == 4
    assert np.allclose(shifted, [4.0, 0.0, 6.0, 2.0])
    assert np.allclose(smoothed, shifted)
    assert np.allclose(zeroed, shifted)
    assert rows[0]["label"] == "caseA"


def test_moving_average_smooth_uses_odd_window():
    values = moving_average_smooth([0.0, 3.0, 6.0, 9.0, 12.0], window_length=4, passes=1)

    assert len(values) == 5
    assert math.isclose(values[2], 6.0)


def test_analyze_fes_barriers_writes_outputs(tmp_path):
    curve_path = tmp_path / "fes.dat"
    _write_curve(
        curve_path,
        [
            (0.0, 5.0, 0.1),
            (1.0, 1.0, 0.1),
            (2.0, 7.0, 0.2),
            (3.0, 3.0, 0.2),
        ],
    )

    outputs = analyze_fes_barriers(
        [FesCurveSpec(curve_path, "caseA", "groupA", "a")],
        output_dir=tmp_path / "out",
        windows=[parse_window("low:0:2"), parse_window("all:-inf:inf")],
        smooth_window=1,
    )

    assert outputs["processed_curves"].exists()
    assert outputs["barrier_summary"].exists()
    assert outputs["manifest"].exists()

    with outputs["barrier_summary"].open() as handle:
        rows = list(csv.DictReader(handle))

    low = next(row for row in rows if row["barrier_region"] == "low")
    all_region = next(row for row in rows if row["barrier_region"] == "all")
    assert math.isclose(float(low["barrier_original_kj_mol"]), 4.0)
    assert math.isclose(float(all_region["barrier_original_kj_mol"]), 6.0)


def test_curve_manifest_loader(tmp_path):
    manifest = tmp_path / "manifest.csv"
    curve_path = tmp_path / "fes.dat"
    _write_curve(curve_path, [(0.0, 0.0, 0.1), (1.0, 2.0, 0.1)])
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "dataset_key"])
        writer.writeheader()
        writer.writerow({"path": str(curve_path), "label": "caseA", "group": "set1", "dataset_key": "a"})

    specs = load_curve_manifest(manifest)

    assert len(specs) == 1
    assert specs[0].dataset_key == "a"
