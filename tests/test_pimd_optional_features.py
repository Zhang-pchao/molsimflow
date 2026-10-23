"""Opt-in extensions keep complete-path statistics and explicit precedence."""

import numpy as np
import pytest
from molsimflow.postprocess.pimd.kde import automatic_kde_bandwidths, resolve_bandwidths


def test_scott_uses_path_count_and_is_invariant_to_bead_duplication():
    beads = np.arange(8.0, dtype=float).reshape(4, 2, 1)
    grid = [np.linspace(0, 7, 701)]
    expected = np.std(beads) * 4 ** (-1 / 5)
    width = automatic_kde_bandwidths(beads, np.ones(4), grid)
    np.testing.assert_allclose(width, [expected])
    assert width == automatic_kde_bandwidths(np.tile(beads, (1, 3, 1)), np.full(4, 1e300), grid)


def test_scott_explicit_bandwidth_wins_and_grid_floor_is_declared():
    config = {"bandwidth_rule": "weighted_scott", "bandwidth_variants": {"chosen": [0.25]}}
    assert resolve_bandwidths(config, None, None) == config
    assert automatic_kde_bandwidths(np.ones((3, 2, 1)), np.ones(3), [np.arange(4.0)]) == (1.5,)
    with pytest.raises(ValueError, match="grid"):
        automatic_kde_bandwidths(np.ones((3, 2, 1)), np.ones(3), [np.array([0, np.inf])])


def test_scott_contract_is_opt_in(tmp_path, monkeypatch):
    import json
    from test_pimd_fes_reporting import _write_contract
    from molsimflow.postprocess import pimd_reweight as legacy

    path, contract = _write_contract(tmp_path)
    contract["reweight"].pop("bandwidth_variants")
    contract["reweight"]["bandwidth_rule"] = "weighted_scott"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(legacy, "save_figure", lambda *args: None)
    output = tmp_path / "analysis"
    assert legacy.analyze(path, output)["status"] == "PASS"
    receipt = json.loads((output / "qc/bandwidth-selection.json").read_text())
    assert receipt["selection"] == "computed"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_unsmoothed_histogram_path_mass_empty_bins_and_pairing(dimensions):
    from molsimflow.postprocess.pimd.estimators import weighted_unsmoothed_path_histogram

    beads = np.array([[[0.2, 0.2], [0.2, 0.2]], [[2.2, 2.2], [4.0, 4.0]]])[:, :, :dimensions]
    result = weighted_unsmoothed_path_histogram(
        beads, np.log([0.25, 0.75]), [np.arange(4.0)] * dimensions, 0.025
    )
    assert result["weight_mass"].sum() == pytest.approx(0.625)
    assert result["out_of_grid_weight_mass"] == pytest.approx(0.375)
    assert result["observed_complete_frames"].sum() == 2
    assert result["observed_beads"].sum() == 3
    assert np.isnan(result["fes_kcal_mol"][~result["support"]]).all()
    if dimensions == 2:
        assert result["weight_mass"][0, 2] == 0
        assert result["weight_mass"][2, 0] == 0


def test_histogram_contract_has_separate_opt_in_artifacts(tmp_path, monkeypatch):
    import json
    from test_pimd_fes_reporting import _write_contract
    from molsimflow.postprocess import pimd_reweight as legacy

    path, contract = _write_contract(tmp_path)
    contract["reweight"]["histogram"] = {"edges": {"x": [-2.0, 0.0, 2.0]}}
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(legacy, "save_figure", lambda *args: None)
    output = tmp_path / "analysis"
    assert legacy.analyze(path, output)["status"] == "PASS"
    metadata = json.loads((output / "qc/unsmoothed-path-histogram.json").read_text())
    assert metadata["pseudocount"] == 0
    assert metadata["out_of_grid_weight_mass"] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("bead_count", [1, 2, 4])
def test_ionization_plot_handles_small_path_counts(tmp_path, monkeypatch, bead_count):
    from molsimflow.postprocess import pimd_reweight as legacy

    figures = []
    monkeypatch.setattr(legacy, "save_figure", lambda fig, path: figures.append(fig))
    legacy.plot_ionization_events(
        tmp_path, np.arange(5.0), np.zeros(5), np.zeros((5, bead_count)), np.zeros(5), [0.5], 2.0
    )
    assert len(figures) == 1


@pytest.mark.parametrize("unit", ["kJ/mol", "kcal/mol"])
def test_discrete_style_unit_and_support_are_explicit(tmp_path, unit):
    from molsimflow.postprocess.pimd.plotting import plot_fes2d_discrete

    grid = np.arange(4.0)
    surface = np.add.outer(grid, grid)
    surfaces = {key: surface.copy() for key in ("centroid", "probability_mean", "free_energy_mean")}
    supports = {key: np.ones((4, 4), bool) for key in surfaces}
    supports["free_energy_mean"][:] = False
    figures = []
    plot_fes2d_discrete(
        tmp_path,
        grid,
        grid,
        surfaces,
        supports,
        6.0,
        ["x", "y"],
        energy_unit=unit,
        _save_figure=lambda fig, path: figures.append(fig),
    )
    assert any(unit in (axis.get_xlabel() + axis.get_ylabel()) for axis in figures[0].axes)
    assert any(
        "Insufficient support" in text.get_text() for axis in figures[0].axes for text in axis.texts
    )
    np.testing.assert_array_equal(surfaces["centroid"], surface)


def test_optional_features_work_together_without_changing_primary_schema(tmp_path, monkeypatch):
    import json
    from test_pimd_fes_reporting import _write_contract
    from molsimflow.postprocess import pimd_reweight as legacy

    path, contract = _write_contract(tmp_path, 2, bead_offsets=(-0.1, 0.1))
    config = contract["reweight"]
    config.pop("bandwidth_variants")
    config["bandwidth_rule"] = "weighted_scott"
    config["histogram"] = {"edges": {"x": [-2.0, 0.0, 2.0], "y": [-2.0, 0.0, 2.0]}}
    contract["plots"].update(fes2d_style="discrete", fes2d_energy_unit="kcal/mol")
    path.write_text(json.dumps(contract))
    figures = []
    monkeypatch.setattr(legacy, "save_figure", lambda fig, path: figures.append(fig))
    output = tmp_path / "analysis"
    summary = legacy.analyze(path, output)
    assert summary["schema_version"] == 3
    assert summary["fes"]["primary_estimator"] == "probability_mean"
    assert (output / "tables/unsmoothed-path-histogram.csv").is_file()
    assert any("kcal/mol" in axis.get_ylabel() for fig in figures for axis in fig.axes)
