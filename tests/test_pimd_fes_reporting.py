"""Primary quantum support and diagnostic-only reporting contracts."""

import csv
import json

import numpy as np
import pytest

from molsimflow.postprocess import pimd_reweight as reweight


def _write_contract(root, dimensions=1):
    names = ["x", "y"][:dimensions]
    time = np.arange(12, dtype=float)
    sampling = np.column_stack([np.zeros(len(time)), np.tile([-0.1, 0.0, 0.1], 4)])
    filenames = ["sampling.colvar", "bead-0.colvar", "bead-1.colvar"]
    np.savetxt(
        root / filenames[0],
        np.column_stack([time, sampling[:, :dimensions], np.zeros(len(time))]),
        header="FIELDS time " + " ".join(names) + " logw", comments="#! ",
    )
    for bead, offset in enumerate((-1.0, 1.0)):
        values = sampling.copy()
        values[:, 0] = offset
        np.savetxt(
            root / filenames[bead + 1],
            np.column_stack([time, values[:, :dimensions]]),
            header="FIELDS time " + " ".join(names), comments="#! ",
        )
    manifest = root / "RAW-SHA256SUMS"
    manifest.write_text("".join(
        f"{reweight.sha256(root / name)}  {name}\n" for name in filenames
    ), encoding="utf-8")
    contract = {
        "analysis_profile": "core",
        "source": {
            "run_root": str(root), "raw_manifest": manifest.name,
            "raw_manifest_sha256": reweight.sha256(manifest),
            "sampling_colvar": filenames[0], "bead_colvars": filenames[1:],
            "expected_beads": 2, "sampling_label": "Mean coordinates",
            "sampling_slug": "bead_mean",
        },
        "selection": {
            "first_time_ps": 0.0, "last_time_ps": 0.011,
            "timestep_fs": 1.0, "expected_frames": len(time),
        },
        "reweight": {
            "cv_names": names, "bias_mode": "bead_mean",
            "weight_kind": "precomputed", "log_weight_column": "logw",
            "temperature_K": 300.0, "kbt_eV": reweight.KB_EV_PER_K * 300.0,
            "grid": {name: [-1.5, 1.5, 61] for name in names},
            "bandwidth_variants": {
                "primary": [0.05] * dimensions, "wide": [0.07] * dimensions,
            },
            "primary_bandwidth": "primary", "relative_density_support": 1e-6,
            "blocks": 2, "plot_max_kcal_mol": 12.0,
            "difference_max_kcal_mol": 2.0,
        },
        "plots": {"cv_labels": {name: name for name in names}},
    }
    path = root / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path, contract


def _csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("dimensions", [1, 2])
def test_disjoint_bead_support_keeps_primary_report(tmp_path, monkeypatch, dimensions):
    contract_path, _ = _write_contract(tmp_path, dimensions)
    # Rendering is still exercised; omit expensive image encoding in this table test.
    monkeypatch.setattr(reweight, "save_figure", lambda *args: None)
    output = tmp_path / "analysis"
    summary = reweight.analyze(contract_path, output)
    assert summary["status"] == "PASS"
    assert summary["schema_version"] == 2
    assert summary["fes"]["primary_estimator"] == "probability_mean"
    assert summary["fes"]["target_observable"] == "bead_marginal"
    assert summary["fes"]["common_support_points"] == 0
    for name in (
        "probability_free_energy_mean_rmse_common_support_kcal_mol",
        "probability_free_energy_mean_max_abs_common_support_kcal_mol",
    ):
        assert summary["fes"][name] is None
    # The finite quantum probability survives even when no diagnostic point does.
    table_path = output / ("fes1d/x.csv" if dimensions == 1 else "fes2d/primary.csv")
    rows = _csv_rows(table_path)
    supported = [row for row in rows if int(row["probability_mean_support"])]
    assert len(supported) >= 2
    assert all(np.isfinite(float(row["F_quantum_probability_mean_kcal_mol"]))
               for row in supported)
    assert not any(int(row["free_energy_mean_support"]) for row in rows)
    assert not any(int(row["common_support"]) for row in rows)
    assert "F_sampling_kcal_mol" in rows[0]
    assert "F_bead_free_energy_mean_diagnostic_kcal_mol" in rows[0]
    assert not any("eq8" in name or "eq10" in name for name in rows[0])

    sensitivity = _csv_rows(output / "qc/bandwidth-sensitivity.csv")
    assert all(int(row["comparison_support_points"]) >= 2 for row in sensitivity)
    assert all(np.isfinite(float(row["probability_mean_rmse_vs_primary_kcal_mol"]))
               for row in sensitivity)
    blocks = _csv_rows(output / "blocks/block-diagnostics.csv")
    assert all(int(row["comparison_support_points"]) >= 2 for row in blocks)
    assert all(float(row["probability_mean_rmse_vs_full_kcal_mol"]) < 1e-10
               for row in blocks)
    persisted = json.loads((output / "qc/summary.json").read_text(encoding="utf-8"))
    assert persisted == summary


@pytest.mark.parametrize("temperature", [-300.0, 0.0])
def test_precomputed_weights_still_require_positive_temperature(tmp_path, temperature):
    contract_path, contract = _write_contract(tmp_path)
    contract["reweight"]["temperature_K"] = temperature
    contract["reweight"]["kbt_eV"] = reweight.KB_EV_PER_K * temperature
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        reweight.analyze(contract_path, tmp_path / "analysis")



@pytest.mark.parametrize("profile", ["core", "water_ionization_opes"])
def test_precomputed_weights_reject_bias_only_reference(tmp_path, profile):
    contract_path, contract = _write_contract(tmp_path)
    contract["analysis_profile"] = profile
    contract["reference"] = {"output_dir": "unused-reference"}
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="reference cross-check requires bias-energy weights"):
        reweight.analyze(contract_path, tmp_path / "analysis")


def test_diagnostic_cannot_be_selected_as_primary(tmp_path):
    contract_path, contract = _write_contract(tmp_path)
    contract["reweight"]["primary_estimator"] = "free_energy_mean"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="primary_estimator.*probability_mean"):
        reweight.analyze(contract_path, tmp_path / "analysis")


def test_one_dimensional_plot_masks_each_curve_by_its_own_support(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(reweight, "save_figure", lambda figure, path: figures.append(figure))
    grid = np.arange(5, dtype=float)
    keys = ("centroid", "probability_mean", "free_energy_mean")
    curves = {key: grid.copy() for key in keys}
    supports = {
        "centroid": np.array([True, True, False, False, False]),
        "probability_mean": np.array([False, True, True, True, False]),
        "free_energy_mean": np.zeros(5, dtype=bool),
    }
    reweight.plot_fes1d(tmp_path, "x", grid, curves, supports, 10.0, "Coordinate")
    lines = figures[0].axes[0].lines
    assert len(lines) == 3
    for line, key in zip(lines, keys):
        plotted = line.get_ydata()
        np.testing.assert_array_equal(np.isfinite(plotted), supports[key])
        np.testing.assert_allclose(
            plotted[supports[key]], curves[key][supports[key]] * reweight.KCAL_TO_KJ_MOL,
        )


def test_two_dimensional_plot_masks_support_and_labels_absent_diagnostic(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    rendered = []
    figures = []
    original = Axes.contourf

    def record_contour(axis, x, y, values, *args, **kwargs):
        rendered.append(np.ma.asarray(values).copy())
        return original(axis, x, y, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "contourf", record_contour)
    monkeypatch.setattr(reweight, "save_figure", lambda figure, path: figures.append(figure))
    grid = np.arange(4, dtype=float)
    surface = np.add.outer(grid, grid)
    keys = ("centroid", "probability_mean", "free_energy_mean")
    surfaces = {key: surface.copy() for key in keys}
    sampling_support = np.ones((4, 4), dtype=bool)
    sampling_support[0] = False
    probability_support = np.ones((4, 4), dtype=bool)
    probability_support[:, -1] = False
    supports = {
        "centroid": sampling_support,
        "probability_mean": probability_support,
        "free_energy_mean": np.zeros((4, 4), dtype=bool),
    }
    reweight.plot_fes2d(tmp_path, grid, grid, surfaces, supports, 10.0, ["x", "y"])
    assert len(rendered) == 2
    for actual, key in zip(rendered, keys[:2]):
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), ~supports[key])
    assert "Insufficient support" in [text.get_text() for text in figures[0].axes[2].texts]


@pytest.mark.parametrize("surface_axis", [None, 0, 1])
@pytest.mark.parametrize("offset,scale,reference,shift", [
    (0.03, 1.0, 1.0, 0.9704412),
    (0.10, 1.10, 1.10, 1.0),
])
def test_coordinate_transform_preserves_shared_zero_diagnostic_gap(
    surface_axis, offset, scale, reference, shift,
):
    options = dict(offset=offset, log_scale=scale, log_reference=reference,
                   linear_shift=shift)
    distance = np.array([0.0, 0.2, 0.7, 1.0, 2.0])
    source = reweight.piecewise_logdistance(distance, **options)
    primary = np.array([2.0, 0.5, 0.0, 0.2, 1.0])
    gap = np.array([0.3, 0.4, 0.5, 0.2, 0.1])
    axis = -1 if surface_axis is None else surface_axis
    if surface_axis is not None:
        primary = np.stack([primary, primary + 0.7], axis=1 - surface_axis)
        gap = np.stack([gap, gap + 0.2], axis=1 - surface_axis)
    diagnostic = primary + gap
    target, jacobian, transformed = reweight.transform_piecewise_logdistance_fes(
        source, primary, 0.6, axis=axis, zero_minimum=False, **options,
    )
    _, _, transformed_diagnostic = reweight.transform_piecewise_logdistance_fes(
        source, diagnostic, 0.6, axis=axis, zero_minimum=False, **options,
    )
    shape = [1] * primary.ndim
    shape[axis] = len(distance)
    np.testing.assert_allclose(target, distance, atol=1e-12)
    np.testing.assert_allclose(transformed, primary - 0.6 * np.log(jacobian).reshape(shape))
    np.testing.assert_allclose(transformed_diagnostic - transformed, gap, atol=1e-12)
    # Applying one primary reference preserves the Jensen gap on the new axis.
    zero = np.min(transformed)
    np.testing.assert_allclose(
        (transformed_diagnostic - zero) - (transformed - zero), gap, atol=1e-12,
    )
