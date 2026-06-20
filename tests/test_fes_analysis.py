import csv
import json
import math

import numpy as np

from molsimflow.postprocess.fes_analysis import (
    Fes2DBatchConfig,
    FesCumulativeReweightConfig,
    FesCurveSpec,
    analyze_fes_barriers,
    analyze_fes_convergence,
    load_fes_cumulative_reweight_manifest,
    load_fes2d_batch_case_manifest,
    load_curve_manifest,
    load_fes_convergence_manifest,
    load_fes2d_grid,
    load_fes_curve,
    moving_average_smooth,
    normalized_gaussian_smooth_2d,
    parse_window,
    prepare_fes_cumulative_reweight_manifest,
    prepare_fes2d_batch_manifest,
    process_fes2d_grid,
    process_curve,
    reweight_plumed_colvar,
)


def _write_curve(path, values):
    with path.open("w") as handle:
        handle.write("#! FIELDS cv free_energy uncertainty\n")
        for cv, free_energy, uncertainty in values:
            handle.write(f"{cv} {free_energy} {uncertainty}\n")


def _write_curve_with_metadata(path, values, sample_size=None):
    with path.open("w") as handle:
        handle.write("#! FIELDS cv free_energy uncertainty\n")
        if sample_size is not None:
            handle.write(f"#! SET sample_size {sample_size}\n")
            handle.write(f"#! SET effective_sample_size {0.5 * float(sample_size)}\n")
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


def test_fes_convergence_manifest_and_outputs(tmp_path):
    final_path = tmp_path / "caseA.dat"
    block1_path = tmp_path / "caseA_1.dat"
    block2_path = tmp_path / "caseA_2.dat"
    cumulative_dir = tmp_path / "cumulative"
    cumulative_dir.mkdir()
    cumulative1_path = cumulative_dir / "fes-cum_050.dat"
    cumulative2_path = cumulative_dir / "fes-cum_100.dat"
    values_final = [(0.0, 5.0, 0.1), (1.0, 1.0, 0.1), (2.0, 7.0, 0.2), (3.0, 3.0, 0.2)]
    values_block = [(0.0, 4.0, 0.1), (1.0, 2.0, 0.1), (2.0, 6.0, 0.2), (3.0, 3.0, 0.2)]
    values_cum1 = [(0.0, 5.0, 0.1), (1.0, 1.0, 0.1), (2.0, 6.0, 0.2), (3.0, 2.0, 0.2)]
    _write_curve_with_metadata(final_path, values_final, sample_size=100)
    _write_curve_with_metadata(block1_path, values_block, sample_size=30)
    _write_curve_with_metadata(block2_path, values_block, sample_size=40)
    _write_curve_with_metadata(cumulative1_path, values_cum1, sample_size=50)
    _write_curve_with_metadata(cumulative2_path, values_final, sample_size=100)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "group", "dataset_key", "series", "chemistry", "cumulative_dir"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": final_path.name,
                "label": "case A",
                "group": "set1",
                "dataset_key": "caseA",
                "series": "S",
                "chemistry": "water",
                "cumulative_dir": cumulative_dir.name,
            }
        )

    specs = load_fes_convergence_manifest(manifest, infer_blocks=True)
    outputs = analyze_fes_convergence(
        specs,
        output_dir=tmp_path / "out",
        window_low=0.0,
        window_high=3.0,
        smooth_window=1,
    )

    assert len(specs) == 1
    assert len(specs[0].block_paths) == 2
    assert len(specs[0].cumulative_paths) == 2
    assert outputs["summary"].exists()
    assert outputs["blocks"].exists()
    assert outputs["cumulative"].exists()
    assert outputs["curves"].exists()

    with outputs["summary"].open() as handle:
        summary_rows = list(csv.DictReader(handle))
    summary = summary_rows[0]
    assert math.isclose(float(summary["delta_f_win_final_smooth_kj_mol"]), 6.0)
    assert math.isclose(float(summary["delta_f_win_block_mean_smooth_kj_mol"]), 4.0)
    assert math.isclose(float(summary["delta_f_win_cumulative_last_smooth_kj_mol"]), 6.0)
    assert math.isclose(float(summary["delta_f_win_final_minus_cumulative_last_smooth_kj_mol"]), 0.0)
    assert summary["rank_final_smooth_within_series"] == "1"

    with outputs["cumulative"].open() as handle:
        cumulative_rows = list(csv.DictReader(handle))
    assert math.isclose(float(cumulative_rows[0]["trajectory_fraction"]), 0.5)
    assert math.isclose(float(cumulative_rows[1]["trajectory_fraction"]), 1.0)


def test_prepare_fes2d_batch_manifest_from_case_manifest(tmp_path):
    bias_dir = tmp_path / "caseA" / "bias_100_50"
    bias_dir.mkdir(parents=True)
    (bias_dir / "COLVAR_tmp").write_text("#! FIELDS time d bridge\n")
    manifest = tmp_path / "cases.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bias_dir", "case_label", "family"])
        writer.writeheader()
        writer.writerow({"bias_dir": "caseA/bias_100_50", "case_label": "Case A", "family": "S"})

    cases = load_fes2d_batch_case_manifest(manifest)
    rows = prepare_fes2d_batch_manifest(
        cases,
        tmp_path / "batch.csv",
        config=Fes2DBatchConfig(
            output_subdir="plots",
            prefix_template="{safe_label}_d3d_bridge",
            x_low=1.0,
            x_high=2.0,
            y_low=3.0,
            y_high=4.0,
            write_plots=True,
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["safe_label"] == "case_a"
    assert row["colvar_tmp_exists"] == 1
    assert row["fes_file"].endswith("caseA/bias_100_50/fes2D/bins50/fes-rew.dat")
    assert row["plot_dir"].endswith("caseA/bias_100_50/fes2D/bins50/plots")
    assert "--x-range 1.0 2.0" in row["fes2d_grid_command"]
    assert "--write-plots" in row["fes2d_grid_command"]

    with (tmp_path / "batch.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["case_label"] == "Case A"


def test_prepare_fes_cumulative_reweight_manifest(tmp_path):
    workdir = tmp_path / "caseA" / "fes_reweight"
    workdir.mkdir(parents=True)
    colvar = tmp_path / "caseA" / "COLVAR_tmp"
    colvar.write_text("#! FIELDS time d3d_all\n")
    driver = tmp_path / "FES_from_Reweighting.py"
    driver.write_text("# driver placeholder\n")
    manifest = tmp_path / "cumulative_cases.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "workdir", "colvar", "sample_size"])
        writer.writeheader()
        writer.writerow({"system": "Case A", "workdir": "caseA/fes_reweight", "colvar": "caseA/COLVAR_tmp", "sample_size": 100})

    specs = load_fes_cumulative_reweight_manifest(manifest)
    rows = prepare_fes_cumulative_reweight_manifest(
        specs,
        tmp_path / "cumulative_manifest.csv",
        config=FesCumulativeReweightConfig(
            driver=driver,
            output_root=tmp_path / "out",
            fractions=(0.5, 1.0),
            skiprows=10,
        ),
    )

    assert len(rows) == 2
    assert rows[0]["trajectory_fraction"] == 0.5
    assert rows[0]["keep_after_skip"] == 50
    assert rows[0]["skipfoot"] == 50
    assert rows[1]["skipfoot"] == 0
    assert rows[0]["colvar_exists"] == 1
    assert "--skipfoot 50" in rows[0]["reweight_command"]
    assert "--skiprows 10" in rows[0]["reweight_command"]

    with (tmp_path / "cumulative_manifest.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert written[1]["trajectory_fraction"] == "1.0"


def test_reweight_plumed_colvar_writes_1d_and_2d_outputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "COLVAR").write_text(
        "\n".join(
            [
                "#! FIELDS time sum_cn.sum cgs foot_total opes.bias opes_e.bias",
                "0.0 10.0 2.0 5.0 -1.0 -0.5",
                "1.0 11.0 2.5 5.5 -0.8 -0.4",
                "2.0 12.0 3.0 6.0 -0.5 -0.3",
                "3.0 13.0 3.5 7.0 -0.2 -0.1",
                "4.0 14.0 4.0 8.0 -0.1 -0.1",
                "5.0 15.0 4.5 9.0 -0.1 -0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "HILLS").write_text(
        "#! FIELDS time sum_cn.sum cgs sigma_sum_cn.sum sigma_cgs height logweight\n"
        "0.0 10.0 2.0 0.5 0.2 1.0 0.0\n"
        "5.0 15.0 4.5 0.7 0.3 1.0 0.0\n",
        encoding="utf-8",
    )

    outputs = reweight_plumed_colvar(
        run_dir,
        output_root=run_dir,
        cv_names=("sum_cn.sum", "foot_total"),
        pairs=(("sum_cn.sum", "cgs"),),
        bias_names=("opes.bias", "opes_e.bias"),
        ranges={
            "sum_cn.sum": (9.0, 16.0),
            "cgs": (1.5, 5.0),
            "foot_total": (4.0, 10.0),
        },
        temperature=330.0,
        blocks=2,
        bins=12,
        pair_bins=(8, 6),
    )

    assert outputs["summary"].exists()
    assert outputs["metadata"].exists()
    assert outputs["report"].exists()

    sumcn_fes = run_dir / "fes_reweight" / "sum_cn_sum" / "fes-rew.dat"
    foot_fes = run_dir / "fes_reweight" / "foot_total" / "fes-rew.dat"
    pair_fes = run_dir / "fes2D" / "sum_cn_sum__cgs" / "fes-rew.dat"
    assert sumcn_fes.exists()
    assert foot_fes.exists()
    assert pair_fes.exists()
    assert (run_dir / "fes_reweight" / "sum_cn_sum" / "fes-rew_1.dat").exists()

    summary_rows = list(csv.DictReader(outputs["summary"].open()))
    by_cv = {row["cvs"]: row for row in summary_rows}
    assert by_cv["sum_cn.sum"]["bandwidth_sources"] == "sum_cn.sum:hills"
    assert by_cv["foot_total"]["bandwidth_sources"] == "foot_total:default_bins"
    assert by_cv["sum_cn.sum,cgs"]["kind"] == "2d"
    assert int(by_cv["sum_cn.sum"]["finite_points"]) > 0


def test_reweight_plumed_colvar_concatenates_restart_segments(tmp_path):
    run_dir = tmp_path / "restart"
    run_dir.mkdir()
    (run_dir / "seg1.COLVAR").write_text(
        "\n".join(
            [
                "#! FIELDS time sum_cn.sum foot_total opes.bias opes_e.bias",
                "0.0 10.0 5.0 -1.0 -0.5",
                "1.0 11.0 5.5 -0.8 -0.4",
                "2.0 12.0 6.0 -0.5 -0.3",
                "3.0 13.0 6.5 -0.2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "seg2.COLVAR").write_text(
        "\n".join(
            [
                "#! FIELDS time sum_cn.sum foot_total opes.bias opes_e.bias",
                "2.0 12.5 6.1 -0.4 -0.2",
                "3.0 13.0 7.0 -0.2 -0.1",
                "4.0 14.0 8.0 -0.1 -0.1",
                "5.0 15.0 9.0 -0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "seg1.HILLS").write_text(
        "#! FIELDS time sum_cn.sum sigma_sum_cn.sum height logweight\n"
        "0.0 10.0 0.5 1.0 0.0\n",
        encoding="utf-8",
    )
    (run_dir / "seg2.HILLS").write_text(
        "#! FIELDS time sum_cn.sum sigma_sum_cn.sum height logweight\n"
        "4.0 14.0 0.8 1.0 0.0\n",
        encoding="utf-8",
    )

    outputs = reweight_plumed_colvar(
        run_dir,
        output_root=run_dir,
        colvar_names=("seg1.COLVAR", "seg2.COLVAR"),
        hills_names=("seg1.HILLS", "seg2.HILLS"),
        cv_names=("sum_cn.sum", "foot_total"),
        pairs=(("sum_cn.sum", "foot_total"),),
        bias_names=("opes.bias", "opes_e.bias"),
        ranges={"sum_cn.sum": (9.0, 16.0), "foot_total": (4.0, 10.0)},
        skip_last_data_line_per_file=True,
        blocks=2,
        bins=12,
        pair_bins=(8, 6),
    )

    rows = list(csv.DictReader(outputs["summary"].open()))
    by_cv = {row["cvs"]: row for row in rows}
    assert by_cv["sum_cn.sum"]["sample_size"] == "5"
    assert by_cv["sum_cn.sum"]["bandwidth_sources"] == "sum_cn.sum:hills"
    assert by_cv["foot_total"]["bandwidth_sources"] == "foot_total:default_bins"
