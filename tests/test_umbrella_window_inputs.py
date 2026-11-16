"""Regression coverage for portable umbrella inputs and legacy table callers."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from molsimflow.io.lammps_table import read_fix_ave_time_table
from molsimflow.postprocess.constant_force_events import read_fix_ave_table
from molsimflow.postprocess.umbrella_wham import load_window, main


def write_lammps(path, rows="0 0 0\n100 1 0.5\n200 2 2\n300 3 4.5\n"):
    path.write_text("# Time-averaged data for fix cv\n# TimeStep v_q f_bias\n" + rows)
    return path


def test_named_reader_and_constant_force_facade_match(tmp_path):
    path = write_lammps(tmp_path / "COLVAR.dat", "100 1 0.5\n")
    names, data = read_fix_ave_time_table(path)
    old_names, old_data = read_fix_ave_table(path)
    assert names == old_names == ["TimeStep", "v_q", "f_bias"]
    assert data.shape == (1, 3)
    np.testing.assert_array_equal(data, old_data)


def test_lammps_clock_energy_conversion_and_exact_selection(tmp_path):
    path = write_lammps(tmp_path / "COLVAR.dat")
    t, q = load_window(
        path, 0.0, 2.0, input_format="lammps_fix_ave_time",
        cv_field="v_q", bias_field="f_bias", time_scale=0.01,
        bias_energy_scale=2.0, discard_time=1.0, end_time=3.0,
        expected_samples=2, expected_time_spacing=1.0,
    )
    np.testing.assert_array_equal(t, [2.0, 3.0])
    np.testing.assert_array_equal(q, [2.0, 3.0])
    with pytest.raises(ValueError, match="Recorded bias"):
        load_window(path, 0, 2, input_format="lammps_fix_ave_time",
                    cv_field="v_q", bias_field="f_bias", discard_time=0)


@pytest.mark.parametrize("option,value,match", [
    ("time_scale", 0, "scales"),
    ("bias_energy_scale", float("nan"), "scales"),
    ("expected_samples", 5, "sample count"),
    ("expected_samples", True, "positive integer"),
    ("expected_time_spacing", 50, "timestamps"),
    ("expected_time_spacing", -1, "spacing"),
    ("bias_tolerance", -1, "tolerance"),
    ("end_time", -1, "time selection"),
    ("cv_field", "absent", "Missing umbrella"),
    ("input_format", "guess", "Unsupported umbrella"),
])
def test_lammps_rejects_inconsistent_contract(tmp_path, option, value, match):
    path = write_lammps(tmp_path / "COLVAR.dat")
    options = dict(input_format="lammps_fix_ave_time", cv_field="v_q",
                   bias_field="f_bias", discard_time=0)
    options[option] = value
    with pytest.raises(ValueError, match=match):
        load_window(path, 0, 1, **options)


@pytest.mark.parametrize("rows", [
    "100 1 0.5\n100 1 0.5\n", "200 2 2\n100 1 0.5\n", "100 nan 0.5\n",
])
def test_lammps_bad_production_values_are_rejected(tmp_path, rows):
    path = write_lammps(tmp_path / "COLVAR.dat", rows)
    with pytest.raises(ValueError):
        load_window(path, 0, 1, input_format="lammps_fix_ave_time",
                    cv_field="v_q", bias_field="f_bias", discard_time=0)


@pytest.mark.parametrize("content", [
    "# only one header\n1 2 3\n",
    "# scalar\n# TimeStep q bias\n1 2\n",
    "# scalar\n# TimeStep q q\n1 0 0\n",
])
def test_invalid_named_tables_are_rejected(tmp_path, content):
    path = tmp_path / "data"; path.write_text(content)
    with pytest.raises(ValueError):
        load_window(path, 0, 1, input_format="lammps_fix_ave_time",
                    cv_field="q", bias_field="bias", discard_time=0)


def manifest_inputs(tmp_path):
    write_lammps(tmp_path / "lammps.dat", "10 0 0\n20 1 0.5\n30 2 2\n")
    (tmp_path / "plumed.dat").write_text(
        "#! FIELDS time s bias\n0 0 0\n1 1 0.5\n2 2 2\n")
    return {
        "kbt": 1, "discard_time": 0, "input_format": "lammps_fix_ave_time",
        "cv_field": "v_q", "bias_field": "f_bias", "expected_samples": 3,
        "windows": [
            {"id": "scalar", "colvar": "lammps.dat", "center": 0, "kappa": 1,
             "time_scale": 0.1, "expected_time_spacing": 1},
            {"id": "named", "colvar": "plumed.dat", "center": 0, "kappa": 1,
             "input_format": "plumed", "cv_field": "s", "bias_field": "bias",
             "discard_time": -1, "expected_time_spacing": 1},
        ],
    }


def test_manifest_normal_entrypoint_per_window_overrides_and_counts(tmp_path):
    config = manifest_inputs(tmp_path)
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps(config))
    output = tmp_path / "result"
    assert main([str(manifest), "--output", str(output), "--bin-width", "1"]) == 0
    report = json.loads((output / "solver.json").read_text())
    assert report["converged"] and report["counts"] == [3, 3]
    assert [w["id"] for w in report["windows"]] == ["scalar", "named"]
    assert report["windows"][0]["first_time"] == 1
    assert not report["equilibrium_qualified"]
    data = np.genfromtxt(output / "pmf.csv", names=True, delimiter=",")
    expected = np.array([2.0, 1.5, 0.0])
    np.testing.assert_allclose(data["pmf"], expected, atol=1e-9)
    original = (output / "solver.json").read_bytes()
    with pytest.raises(FileExistsError):
        main([str(manifest), "--output", str(output)])
    assert (output / "solver.json").read_bytes() == original


@pytest.mark.parametrize("duplicate", ["id", "same_path", "hard_link"])
def test_manifest_rejects_duplicate_window_identity_or_file(tmp_path, duplicate):
    config = manifest_inputs(tmp_path)
    if duplicate == "id":
        config["windows"][1]["id"] = "scalar"
    else:
        if duplicate == "hard_link":
            os.link(tmp_path / "lammps.dat", tmp_path / "alias.dat")
            config["windows"][1]["colvar"] = "alias.dat"
        else:
            config["windows"][1]["colvar"] = "./lammps.dat"
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Duplicate umbrella"):
        main([str(manifest), "--output", str(tmp_path / "result")])
    assert not (tmp_path / "result").exists()
