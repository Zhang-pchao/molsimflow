"""Opt-in extensions keep complete-path statistics and explicit precedence."""
import numpy as np
import pytest
from molsimflow.postprocess.pimd.kde import automatic_kde_bandwidths, resolve_bandwidths


def test_scott_uses_path_count_and_is_invariant_to_bead_duplication():
    beads = np.arange(8., dtype=float).reshape(4, 2, 1)
    grid = [np.linspace(0, 7, 701)]
    expected = np.std(beads) * 4 ** (-1 / 5)
    width = automatic_kde_bandwidths(beads, np.ones(4), grid)
    np.testing.assert_allclose(width, [expected])
    assert width == automatic_kde_bandwidths(np.tile(beads, (1, 3, 1)), np.full(4, 1e300), grid)


def test_scott_explicit_bandwidth_wins_and_grid_floor_is_declared():
    config = {"bandwidth_rule": "weighted_scott", "bandwidth_variants": {"chosen": [.25]}}
    assert resolve_bandwidths(config, None, None) == config
    assert automatic_kde_bandwidths(np.ones((3, 2, 1)), np.ones(3), [np.arange(4.)]) == (1.5,)
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
    beads = np.array([[[.2, .2], [.2, .2]], [[2.2, 2.2], [4., 4.]]])[:, :, :dimensions]
    result = weighted_unsmoothed_path_histogram(beads, np.log([.25, .75]),
                                               [np.arange(4.)] * dimensions, .025)
    assert result["weight_mass"].sum() == pytest.approx(.625)
    assert result["out_of_grid_weight_mass"] == pytest.approx(.375)
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
    contract["reweight"]["histogram"] = {"edges": {"x": [-2., 0., 2.]}}
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(legacy, "save_figure", lambda *args: None)
    output = tmp_path / "analysis"
    assert legacy.analyze(path, output)["status"] == "PASS"
    metadata = json.loads((output / "qc/unsmoothed-path-histogram.json").read_text())
    assert metadata["pseudocount"] == 0
    assert metadata["out_of_grid_weight_mass"] == pytest.approx(0., abs=1e-14)
