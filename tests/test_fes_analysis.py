import csv
import json
import math

import numpy as np

from molsimflow.postprocess.fes_analysis import (
    FesCurveSpec,
    analyze_fes_barriers,
    load_curve_manifest,
    load_fes2d_grid,
    load_fes_curve,
    moving_average_smooth,
    normalized_gaussian_smooth_2d,
    parse_window,
    process_fes2d_grid,
    process_curve,
)


def _write_curve(path, values):
    with path.open("w") as handle:
        handle.write("#! FIELDS cv free_energy uncertainty\n")
        for cv, free_energy, uncertainty in values:
            handle.write(f"{cv} {free_energy} {uncertainty}\n")


def _write_fes2d(path):
    rows = [
        (0.0, 0.0, 4.0, 0.1),
        (1.0, 0.0, 2.0, 0.1),
        (2.0, 0.0, 8.0, 0.1),
        (0.0, 1.0, 1.0, 0.2),
        (1.0, 1.0, 3.0, 0.2),
        (2.0, 1.0, 9.0, 0.2),
        (0.0, 2.0, 5.0, 0.3),
        (1.0, 2.0, 6.0, 0.3),
        (2.0, 2.0, 7.0, 0.3),
    ]
    with path.open("w") as handle:
        handle.write("#! FIELDS d bridge free_energy uncertainty\n")
        handle.write("#! SET min_d 0\n")
        for row in rows:
            handle.write("{} {} {} {}\n".format(*row))


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


def test_load_fes2d_grid_reads_regular_grid(tmp_path):
    grid_path = tmp_path / "fes2d.dat"
    _write_fes2d(grid_path)

    grid = load_fes2d_grid(grid_path)

    assert grid.x_name == "d"
    assert grid.y_name == "bridge"
    assert grid.metadata["min_d"] == "0"
    assert grid.free_energy.shape == (3, 3)
    assert math.isclose(grid.free_energy[1, 0], 1.0)


def test_normalized_gaussian_smooth_2d_handles_missing_values():
    values = np.array([[0.0, np.nan, 4.0], [2.0, 3.0, np.nan], [5.0, 6.0, 7.0]])

    smooth, support = normalized_gaussian_smooth_2d(values, sigma=0.6, valid_threshold=0.05)

    assert smooth.shape == values.shape
    assert support.shape == values.shape
    assert np.isfinite(smooth[1, 1])
    assert support[0, 1] > 0.0


def test_process_fes2d_grid_writes_csv_and_metadata(tmp_path):
    grid_path = tmp_path / "fes2d.dat"
    _write_fes2d(grid_path)

    outputs = process_fes2d_grid(
        grid_path,
        tmp_path / "out",
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        smooth_sigma=0.0,
        prefix="caseA",
    )

    assert outputs["plot_grid_csv"].exists()
    assert outputs["metadata_json"].exists()

    with outputs["metadata_json"].open() as handle:
        metadata = json.load(handle)

    assert metadata["x_bins_selected"] == 2
    assert metadata["y_bins_selected"] == 2
    assert math.isclose(metadata["raw_zero_value_kj_mol"], 1.0)

    with outputs["plot_grid_csv"].open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["d"] == "0"
    assert rows[0]["bridge"] == "0"
