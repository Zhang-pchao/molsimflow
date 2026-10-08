import hashlib
import json
import os
from pathlib import Path

import pytest

from molsimflow.postprocess.quantum_path_contract import (
    load_contract,
    verify_inputs_unchanged,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _contract(tmp_path):
    beads = []
    for bead_id in (7, 2):
        source = tmp_path / f"bead-{bead_id}.dump"
        source.write_bytes(f"synthetic source for bead {bead_id}".encode())
        beads.append({"bead_id": bead_id, "path": source.name, "sha256": _hash(source)})
    data = {
        "schema_version": 1,
        "run": {
            "run_id": "run-a", "seed_id": "seed-a", "initial_path_id": "initial-a",
            "parent_restart_id": None, "bias_mode": "bead_mean", "data_role": "engineering",
        },
        "atom_identity": [{"id": 9, "type": 2}, {"id": 3, "type": 1}],
        "beads": beads,
        "steps": {"first": 0, "last": 6, "stride": 2, "timestep_fs": 0.25},
        "geometry": {
            "center_type": 2, "assigned_type": 1, "kappa": 5, "distance_kappa": 8,
            "reference": 2, "environment_r0": 3.5, "length_unit": "A",
        },
        "weights": {"kind": "uniform_sampler"},
        "analysis": {
            "block_frames": 2,
            "conditioning_fields": ["q_centroid", "oo_coordination_centroid_mean"],
            "bin_edges": [[0, 1, 3], [0, 4, 20]],
            "regions": [{"name": "neutral", "bounds": {"q": [0, 0.5]}}],
        },
    }
    return tmp_path / "contract.json", data


def _precomputed(tmp_path, data):
    source = tmp_path / "weights.csv"
    source.write_text("step,log_weight\n0,0\n2,1\n4,0\n6,-1\n")
    data["weights"] = {
        "kind": "precomputed", "path": source.name, "sha256": _hash(source),
        "target_id": "declared-target", "admission_reference": "external-contract-v1",
    }
    return source


def _at(data, location):
    for key in location:
        data = data[key]
    return data


def test_load_preserves_declared_order_and_hashes_every_input(tmp_path):
    path, data = _contract(tmp_path)
    data["beads"][0]["sha256"] = data["beads"][0]["sha256"].upper()
    data["beads"][1]["path"] = str(tmp_path / data["beads"][1]["path"])
    loaded = load_contract(_write(path, data))
    assert [b["bead_id"] for b in loaded["beads"]] == [7, 2]
    assert [a["id"] for a in loaded["atom_identity"]] == [9, 3]
    assert loaded["_contract_path"] == path
    assert all(isinstance(b["path"], Path) for b in loaded["beads"])
    assert loaded["_input_hashes"] == {
        str(p): _hash(p) for p in [path, tmp_path / "bead-7.dump", tmp_path / "bead-2.dump"]
    }
    verify_inputs_unchanged(loaded)


def test_classical_one_bead_and_no_analysis(tmp_path):
    path, data = _contract(tmp_path)
    data["run"]["bias_mode"] = "classical"
    data["beads"] = data["beads"][:1]
    data["steps"]["last"] = 0
    del data["analysis"]
    assert len(load_contract(_write(path, data))["beads"]) == 1


def test_precomputed_labels_are_declarations_and_csv_validation_is_deferred(tmp_path):
    path, data = _contract(tmp_path)
    source = _precomputed(tmp_path, data)
    source.write_text("not valid CSV; schema-only fixture")
    data["weights"]["sha256"] = _hash(source)
    loaded = load_contract(_write(path, data))
    assert loaded["weights"]["path"] == source
    assert str(source) in loaded["_input_hashes"]


@pytest.mark.parametrize("location", [
    (), ("run",), ("atom_identity", 0), ("beads", 0), ("steps",), ("geometry",),
    ("weights",), ("analysis",), ("analysis", "regions", 0),
    ("analysis", "regions", 0, "bounds"),
])
def test_unknown_keys_are_rejected_at_every_schema_level(tmp_path, location):
    path, data = _contract(tmp_path)
    _at(data, location)["typo"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        load_contract(_write(path, data))


@pytest.mark.parametrize("location,key,value", [
    ((), "schema_version", True), ((), "schema_version", 2),
    (("run",), "seed_id", 4), (("run",), "initial_path_id", " "),
    (("run",), "parent_restart_id", ""), (("run",), "bias_mode", "centroid"),
    (("run",), "data_role", "accepted"), ((), "beads", []), ((), "atom_identity", []),
    (("atom_identity", 0), "id", 0), (("atom_identity", 0), "type", True),
    (("atom_identity", 1), "id", 9), (("beads", 0), "bead_id", -1),
    (("beads", 1), "bead_id", 7), (("beads", 0), "path", ""),
    (("steps",), "first", -1), (("steps",), "stride", 0),
    (("steps",), "stride", 4), (("steps",), "last", -1),
    (("steps",), "last", 6.0), (("steps",), "timestep_fs", 0),
    (("geometry",), "center_type", 1), (("geometry",), "assigned_type", 0),
    (("geometry",), "kappa", -1), (("geometry",), "distance_kappa", 0),
    (("geometry",), "reference", "2"), (("geometry",), "environment_r0", False),
    (("geometry",), "length_unit", " "), (("weights",), "kind", "opes"),
])
def test_invalid_schema_values_fail_closed(tmp_path, location, key, value):
    path, data = _contract(tmp_path)
    _at(data, location)[key] = value
    with pytest.raises(ValueError):
        load_contract(_write(path, data))


@pytest.mark.parametrize("key,value", [
    ("block_frames", 3), ("block_frames", 4), ("block_frames", 0),
    ("conditioning_fields", []), ("conditioning_fields", ["q_bead_mean"]),
    ("conditioning_fields", ["q_centroid", "q_centroid"]),
    ("bin_edges", [[0, 1]]), ("bin_edges", [[0, 0], [0, 1]]),
    ("bin_edges", [[0], [0, 1]]), ("regions", []),
    ("regions", [{"name": "r", "bounds": {}}]),
    ("regions", [{"name": "r", "bounds": {"q": [1, 0]}}]),
    ("regions", [{"name": "r", "bounds": {"q": [0, 1, 2]}}]),
    ("regions", [{"name": "r", "bounds": {"q": [0, 1]}}] * 2),
])
def test_analysis_requires_declared_finite_bins_regions_and_complete_blocks(tmp_path, key, value):
    path, data = _contract(tmp_path)
    data["analysis"][key] = value
    with pytest.raises(ValueError):
        load_contract(_write(path, data))


def test_missing_required_keys_and_wrong_container(tmp_path):
    path, data = _contract(tmp_path)
    del data["run"]["parent_restart_id"]
    with pytest.raises(ValueError, match="missing keys"):
        load_contract(_write(path, data))
    with pytest.raises(ValueError, match="must be an object"):
        load_contract(_write(path, []))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_are_rejected(tmp_path, literal):
    path, data = _contract(tmp_path)
    text = json.dumps(data).replace('"reference": 2', f'"reference": {literal}')
    path.write_text(text)
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_contract(path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path, data = _contract(tmp_path)
    path.write_text(json.dumps(data).replace(
        '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
    ))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_contract(path)


@pytest.mark.parametrize("digest", ["abc", "x" * 64, "0" * 64])
def test_bad_or_mismatched_hash_is_rejected(tmp_path, digest):
    path, data = _contract(tmp_path)
    data["beads"][0]["sha256"] = digest
    with pytest.raises(ValueError, match="SHA256"):
        load_contract(_write(path, data))


@pytest.mark.parametrize("alias_kind", ["same_path", "hardlink", "symlink", "directory_symlink"])
def test_aliases_are_rejected(tmp_path, alias_kind):
    path, data = _contract(tmp_path)
    original = tmp_path / data["beads"][0]["path"]
    alias = tmp_path / "alias.dump"
    if alias_kind == "same_path":
        alias = original
    elif alias_kind == "hardlink":
        os.link(original, alias)
    elif alias_kind == "symlink":
        alias.symlink_to(original)
    else:
        directory = tmp_path / "linked-directory"
        directory.symlink_to(tmp_path, target_is_directory=True)
        alias = directory / original.name
    data["beads"][1].update(path=str(alias), sha256=_hash(original))
    with pytest.raises(ValueError, match="duplicate input|symlink"):
        load_contract(_write(path, data))


def test_contract_symlink_and_missing_input_are_rejected(tmp_path):
    path, data = _contract(tmp_path)
    _write(path, data)
    link = tmp_path / "linked-contract.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        load_contract(link)
    (tmp_path / data["beads"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="unavailable"):
        load_contract(path)


def test_weights_cannot_alias_bead_and_uniform_cannot_smuggle_weight_fields(tmp_path):
    path, data = _contract(tmp_path)
    data["weights"] = {"kind": "precomputed", **data["beads"][0],
                       "target_id": "target", "admission_reference": "reference"}
    del data["weights"]["bead_id"]
    with pytest.raises(ValueError, match="duplicate input"):
        load_contract(_write(path, data))
    data["weights"] = {"kind": "uniform_sampler", "target_id": "target"}
    with pytest.raises(ValueError, match="unknown keys"):
        load_contract(_write(path, data))


@pytest.mark.parametrize("which", ["contract", "bead", "weights"])
def test_verify_detects_postload_mutation(tmp_path, which):
    path, data = _contract(tmp_path)
    weights = _precomputed(tmp_path, data)
    loaded = load_contract(_write(path, data))
    target = {"contract": path, "bead": tmp_path / data["beads"][0]["path"],
              "weights": weights}[which]
    with target.open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="SHA256 changed"):
        verify_inputs_unchanged(loaded)


@pytest.mark.parametrize("replacement", ["missing", "same_content", "symlink"])
def test_verify_detects_replacement_or_loss(tmp_path, replacement):
    path, data = _contract(tmp_path)
    loaded = load_contract(_write(path, data))
    source = loaded["beads"][0]["path"]
    backup = tmp_path / "replacement.dump"
    backup.write_bytes(source.read_bytes())
    source.unlink()
    if replacement == "same_content":
        backup.rename(source)
    elif replacement == "symlink":
        source.symlink_to(backup)
    with pytest.raises(ValueError):
        verify_inputs_unchanged(loaded)
