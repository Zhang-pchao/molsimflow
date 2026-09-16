"""End-to-end checks of the portable quantum-path analysis command."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from molsimflow.cli import main
from molsimflow.postprocess import quantum_path_analysis as workflow


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_contract(tmp_path, weighted=False):
    beads = []
    for bead in range(2):
        path = tmp_path / f"bead-{bead}.dump"
        frames = []
        for index in range(8):
            h = [2.2, 3.8] if index % 2 == 0 else [2.3, 2.5]
            frames.append("ITEM: TIMESTEP\n" + str(index * 10)
                          + "\nITEM: NUMBER OF ATOMS\n4\nITEM: BOX BOUNDS pp pp pp\n"
                          + "0 12\n0 12\n0 12\nITEM: ATOMS id type x y z\n"
                          + f"1 2 2 5 5\n2 1 {h[0] + .03 * bead} 5 5\n"
                          + f"3 2 4 5 5\n4 1 {h[1] + .03 * bead} 5 5\n")
        path.write_text("".join(frames))
        beads.append({"bead_id": bead, "path": path.name, "sha256": digest(path)})
    contract = {
        "schema_version": 1,
        "run": {"run_id": "synthetic", "seed_id": "deterministic", "initial_path_id": "mini",
                "parent_restart_id": None, "bias_mode": "bead_mean", "data_role": "engineering"},
        "atom_identity": [{"id": tag, "type": 2 if tag % 2 else 1} for tag in range(1, 5)],
        "beads": beads, "steps": {"first": 0, "last": 70, "stride": 10, "timestep_fs": .25},
        "geometry": {"center_type": 2, "assigned_type": 1, "kappa": 5,
                     "distance_kappa": 8, "reference": 1, "environment_r0": 3.2,
                     "length_unit": "angstrom"},
        "weights": {"kind": "uniform_sampler"},
        "analysis": {"block_frames": 2, "conditioning_fields": ["q_centroid"],
                     "bin_edges": [[-.1, 3]],
                     "regions": [{"name": "small_q", "bounds": {"q": [0, 1]}}]},
    }
    if weighted:
        weights = tmp_path / "weights.csv"
        weights.write_text("step,log_weight\n" + "".join(
            f"{i * 10},{np.log(9) if i % 2 == 0 else 0}\n" for i in range(8)))
        contract["weights"] = {"kind": "precomputed", "path": weights.name,
                               "sha256": digest(weights), "target_id": "synthetic_oracle",
                               "admission_reference": "Analytic discrete weight oracle"}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path, contract


def test_cli_extracts_complete_paths_and_conditional_output(tmp_path):
    contract, _ = make_contract(tmp_path)
    output = tmp_path / "analysis"
    assert main(["postprocess", "quantum-path", "--contract", str(contract),
                 "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["physical_frames"] == 8 and result["beads"] == 2
    assert result["distribution"] == "SAMPLED_DISTRIBUTION_ONLY"
    assert result["scientific_acceptance"] == "NOT_ASSESSED"
    assert result["source_unchanged"] and result["input_hashes_unchanged"]
    with (output / "frames.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["time_fs"]) for row in rows] == list(np.arange(8) * 2.5)
    assert max(abs(float(row["variance_identity_residual"])) for row in rows) < 1e-12
    with (output / "beads.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 16
    conditional = json.loads((output / "conditional.json").read_text())
    assert conditional["distribution"] == "SAMPLED_DISTRIBUTION_ONLY"
    assert conditional["regions"] == [{"name": "small_q", "bounds": {"q": [0, 1]}}]


def test_precomputed_weighted_workflow_uses_one_weight_per_frame(tmp_path):
    path, _ = make_contract(tmp_path, weighted=True)
    output = tmp_path / "analysis"
    result = workflow.analyze(path, output)
    assert result["status"] == "PASS"
    assert result["distribution"] == "DECLARED_REWEIGHTED_TARGET_NOT_INDEPENDENTLY_ADMITTED"
    data = json.loads((output / "conditional.json").read_text())
    # Independent two-state oracle: four states of weight 9 and four of weight 1.
    assert data["cells"][0]["moments"]["region_probabilities"][0] == pytest.approx(.9)


def test_region_components_are_paired_within_each_bead():
    values = {"q": np.array([0., 2.]), "distance": np.array([2., 0.]),
              "logdistance": np.zeros(2)}
    regions = [{"name": "joint", "bounds": {"q": [0, 1], "distance": [0, 1]}}]
    assert workflow._region_fractions(values, regions).tolist() == [0.0]


def test_logdistance_preserves_native_boundary_and_domain():
    values = np.array([.1, 1., 2.])
    np.testing.assert_allclose(workflow.native_logdistance(values),
                               [np.log(.13), np.log(1.03) + .0295588, 1.0295588])
    with pytest.raises(ValueError, match="domain"):
        workflow.native_logdistance(-.03)


def test_changed_hash_returns_failure_receipt(tmp_path):
    path, contract = make_contract(tmp_path)
    (tmp_path / contract["beads"][0]["path"]).write_text("changed")
    result = workflow.analyze(path, tmp_path / "analysis")
    assert result["status"] == "FAIL"
    assert (tmp_path / "analysis/result.json").is_file()


def test_selected_prefix_does_not_hide_malformed_tail(tmp_path):
    path, contract = make_contract(tmp_path)
    for bead in contract["beads"]:
        dump = tmp_path / bead["path"]
        with dump.open("a") as handle:
            handle.write("ITEM: TIMESTEP\n999\n")
        bead["sha256"] = digest(dump)
    contract["steps"]["last"] = 30
    path.write_text(json.dumps(contract))
    result = workflow.analyze(path, tmp_path / "analysis")
    assert result["status"] == "FAIL"
    assert "error" in result


def test_missing_weight_frame_is_not_interpolated(tmp_path):
    path, contract = make_contract(tmp_path, weighted=True)
    weights = tmp_path / contract["weights"]["path"]
    weights.write_text("\n".join(weights.read_text().splitlines()[:-1]) + "\n")
    contract["weights"]["sha256"] = digest(weights)
    path.write_text(json.dumps(contract))
    result = workflow.analyze(path, tmp_path / "analysis")
    assert result["status"] == "FAIL" and "selected physical steps" in result["error"]


def test_existing_output_is_never_overwritten(tmp_path):
    path, _ = make_contract(tmp_path)
    output = tmp_path / "analysis"
    output.mkdir()
    marker = output / "result.json"
    marker.write_text("authoritative")
    with pytest.raises(FileExistsError):
        workflow.analyze(path, output)
    assert marker.read_text() == "authoritative"


def test_input_changed_during_processing_cannot_pass(tmp_path, monkeypatch):
    path, _ = make_contract(tmp_path)
    original = workflow.quantum_path.describe_quantum_path

    def changing_input(*args, **kwargs):
        result = original(*args, **kwargs)
        with path.open("a") as handle:
            handle.write(" ")
        return result

    monkeypatch.setattr(workflow.quantum_path, "describe_quantum_path", changing_input)
    result = workflow.analyze(path, tmp_path / "analysis")
    assert result["status"] == "FAIL" and "preservation_error" in result


def test_native_workflow_rejects_implicit_unit_change(tmp_path):
    path, contract = make_contract(tmp_path)
    contract["geometry"]["length_unit"] = "nm"
    path.write_text(json.dumps(contract))
    result = workflow.analyze(path, tmp_path / "analysis")
    assert result["status"] == "FAIL" and "length_unit=angstrom" in result["error"]


def test_changed_geometry_dependency_cannot_pass(tmp_path, monkeypatch):
    path, _ = make_contract(tmp_path)
    fake_source = tmp_path / "geometry.py"
    fake_source.write_text("original")
    monkeypatch.setattr(workflow.lammps_dump, "__file__", str(fake_source))
    original = workflow.quantum_path.describe_quantum_path

    def changing_source(*args, **kwargs):
        result = original(*args, **kwargs)
        fake_source.write_text("changed")
        return result

    monkeypatch.setattr(workflow.quantum_path, "describe_quantum_path", changing_source)
    result = workflow.analyze(path, tmp_path / "analysis")
    assert result["status"] == "FAIL" and result["source_unchanged"] is False
