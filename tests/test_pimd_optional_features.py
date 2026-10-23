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
