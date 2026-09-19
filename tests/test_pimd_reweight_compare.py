"""Semantic compatibility tests for quantum FES comparisons."""

import csv
import json

import numpy as np
import pytest

from molsimflow.postprocess import pimd_reweight_compare as comparison


COLUMNS = (
    "sampling_support", "F_sampling_kcal_mol",
    "probability_mean_support", "F_quantum_probability_mean_kcal_mol",
    "free_energy_mean_support", "F_bead_free_energy_mean_diagnostic_kcal_mol",
)
LEGACY = dict(zip(COLUMNS, (
    "centroid_support", "F_centroid_kcal_mol", "eq8_support", "F_eq8_kcal_mol",
    "eq10_support", "F_eq10_kcal_mol",
)))
CORE_LEGACY = {
    "free_energy_mean_support": "logmean_support",
    "F_bead_free_energy_mean_diagnostic_kcal_mol": "F_bead_logmean_diagnostic_kcal_mol",
}


def _table(schema, size=3, run=0):
    values = np.arange(size, dtype=float)
    table = {
        "sampling_support": np.ones(size),
        "F_sampling_kcal_mol": values,
        "probability_mean_support": np.ones(size),
        "F_quantum_probability_mean_kcal_mol": values**2 + 2.0 * run,
        "free_energy_mean_support": np.ones(size),
        "F_bead_free_energy_mean_diagnostic_kcal_mol": values**2 * (1.0 + run),
    }
    mapping = LEGACY if schema == "legacy" else CORE_LEGACY if schema == "core_legacy" else {}
    return {mapping.get(key, key): value for key, value in table.items()}


def _write_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(table)
        writer.writerows(zip(*table.values()))


def _run(tmp_path, index, schema, dimensions):
    root = tmp_path / f"run-{index}"
    names = ["coordination"] if dimensions == 1 else ["coordination", "torsion"]
    for name in names:
        _write_table(root / "fes1d" / f"{name}.csv", {
            name: np.arange(3), **_table(schema, run=index),
        })
    if dimensions == 2:
        x, y = np.meshgrid(np.arange(3), np.arange(2))
        _write_table(root / "fes2d" / "primary.csv", {
            names[0]: x.ravel(), names[1]: y.ravel(), **_table(schema, size=6, run=index),
        })
    summary = {
        "status": "PASS", "source_job": None,
        "sampling_representation": {
            "bias_mode": "centroid_coord" if index == 0 else "bead_mean",
            "label": "Centroid" if index == 0 else "Bead mean",
            "logical_cv_names": names,
        },
        "selection": {"frames": 3},
        "reweighting": {"ess": 3, "ess_fraction": 1, "maximum_normalized_weight": 1/3},
        "fes": {}, "gates": {},
    }
    (root / "qc").mkdir()
    (root / "qc" / "summary.json").write_text(json.dumps(summary))
    return {
        "analysis_root": str(root), "label": f"Run {index}", "short_label": f"R{index}",
        "configuration": "synthetic", "walltime_seconds": 1, "engine_loop_seconds": 1,
    }


@pytest.mark.parametrize("schema", ["legacy", "core_legacy"])
def test_legacy_adapter_preserves_estimator_identity(schema):
    canonical = comparison.canonical_fes_table(_table("canonical"))
    migrated = comparison.canonical_fes_table(_table(schema))
    for name in COLUMNS:
        np.testing.assert_array_equal(canonical[name], migrated[name])


def test_conflicting_alias_columns_are_rejected():
    table = _table("canonical")
    table["F_eq8_kcal_mol"] = np.array([0.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="conflicting FES columns"):
        comparison.canonical_fes_table(table)


@pytest.mark.parametrize("schema", ["canonical", "legacy", "core_legacy"])
@pytest.mark.parametrize("dimensions", [1, 2])
def test_comparison_defaults_to_probability_mean_and_generic_axes(tmp_path, monkeypatch, schema, dimensions):
    captured = []
    monkeypatch.setattr(comparison, "save_figure", lambda figure, path: captured.append({
        "labels": [(axis.get_xlabel(), axis.get_ylabel()) for axis in figure.axes],
        "titles": [axis.get_title() for axis in figure.axes],
    }))
    runs = [_run(tmp_path, index, schema, dimensions) for index in range(2)]
    contract = {
        "runs": runs, "window_label": "synthetic", "plot_max_kcal_mol": 50,
        "difference_max_kcal_mol": 5, "comparison_boundary": "Synthetic only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    result = comparison.compare(contract_path, tmp_path / "comparison")
    assert result["primary_estimator"] == "probability_mean"
    assert result["coordinate_definition_gate"] == "NOT_VERIFIED"
    assert not result["sampling_observables_match"]
    assert len(result["one_dimensional"]) == dimensions
    for metrics in result["one_dimensional"]:
        assert "sampling" not in metrics
        assert metrics["probability_mean"]["shape_rmse_kcal_mol"] == pytest.approx(0)
        assert metrics["probability_mean"]["alignment_offset_kcal_mol"] == pytest.approx(2)
    assert all("Eq." not in title for item in captured for title in item["titles"])
    if dimensions == 2:
        assert set(result["two_dimensional"]) == {"probability_mean"}
        assert result["two_dimensional"]["probability_mean"]["shape_rmse_kcal_mol"] == pytest.approx(0)
        assert any(("coordination", "torsion") in item["labels"] for item in captured)
    else:
        assert result["two_dimensional"] == {}
    assert "eq8" not in json.dumps(result)
    assert "eq10" not in json.dumps(result)


def test_sampling_comparison_requires_known_matching_observables():
    def run(mode=None, explicit=None):
        return {
            "config": {"sampling_observable_id": explicit},
            "summary": {"sampling_representation": {
                "bias_mode": mode, "logical_cv_names": ["coordination"],
            }},
        }
    assert not comparison.sampling_observables_match([run("centroid_coord"), run("centroid_coord")])
    assert not comparison.sampling_observables_match([run("centroid_coord"), run("bead_mean")])
    assert not comparison.sampling_observables_match([run(), run()])
    assert comparison.sampling_observables_match([run(explicit="q-centroid"), run(explicit="q-centroid")])


def test_cv_order_mismatch_is_rejected():
    runs = [{"summary": {"sampling_representation": {"logical_cv_names": ["a", "b"]}}}]
    with pytest.raises(ValueError, match="CV names/order mismatch"):
        comparison.comparison_cv_names({"cvs": ["b", "a"]}, runs)


def test_missing_cv_metadata_requires_explicit_contract():
    runs = [{"summary": {}}]
    with pytest.raises(ValueError, match="requires cvs"):
        comparison.comparison_cv_names({}, runs)
    assert comparison.comparison_cv_names({"cvs": ["a"]}, runs) == ("a",)


def test_custom_primary_bandwidth_name_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "save_figure", lambda *args: None)
    configs = [_run(tmp_path, index, "canonical", 2) for index in range(2)]
    runs = [comparison.load_run(config) for config in configs]
    for index, run in enumerate(runs):
        name = f"bandwidth-{index}"
        old_path = run["root"] / "fes2d" / "primary.csv"
        old_path.rename(old_path.with_name(f"{name}.csv"))
        run["summary"]["fes"]["primary_bandwidth_name"] = name
    metrics = comparison._two_dimensional_figures(
        tmp_path / "output", runs, 50.0, 5.0, "synthetic",
        ["Coordination", "Torsion"], ["coordination", "torsion"],
    )
    assert metrics["probability_mean"]["shape_rmse_kcal_mol"] == pytest.approx(0)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "", ".", ".."])
def test_primary_bandwidth_name_rejects_unsafe_paths(tmp_path, name):
    run = {"root": tmp_path, "summary": {"fes": {"primary_bandwidth_name": name}}}
    with pytest.raises(ValueError, match="basename"):
        comparison.primary_surface_path(run)


def test_recorded_observable_ids_are_supported():
    run = {
        "config": {},
        "summary": {
            "sampling_representation": {"sampling_observable_id": "coordinate-definition-v1"},
            "fes": {"target_observable_id": "bead-coordinate-definition-v1"},
        },
    }
    assert comparison.sampling_observables_match([run, run])
    assert comparison.coordinate_definition_gate([run, run]) == "DECLARED_COMPATIBLE"


def test_target_identity_requires_explicit_matching_definitions():
    def run(identity=None):
        return {"config": {"target_observable_id": identity}, "summary": {}}
    assert comparison.coordinate_definition_gate([run(), run()]) == "NOT_VERIFIED"
    assert comparison.coordinate_definition_gate([run("a"), run()]) == "NOT_VERIFIED"
    assert comparison.coordinate_definition_gate([run("a"), run("a")]) == "DECLARED_COMPATIBLE"
    with pytest.raises(ValueError, match="target_observable_id mismatch"):
        comparison.coordinate_definition_gate([run("a"), run("b")])


def test_conflicting_observable_identity_sources_are_rejected():
    run = {
        "config": {"sampling_observable_id": "coordinate-v2"},
        "summary": {"sampling_representation": {"sampling_observable_id": "coordinate-v1"}},
    }
    with pytest.raises(ValueError, match="conflicting sampling_observable_id"):
        comparison.sampling_observables_match([run, run])
