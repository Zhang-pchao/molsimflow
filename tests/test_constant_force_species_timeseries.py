import csv
import json
from pathlib import Path

import numpy as np
import pytest

from molsimflow.postprocess.constant_force_species_timeseries import (
    assign_hydrogen_parents,
    identify_fixed_carbon_hydrogen_ids,
    read_type_symbols,
    run_contract,
)


def _write_model(path: Path) -> None:
    path.write_text(
        """9 atoms
4 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 15.999 # O
2 1.008 # H
3 12.011 # C
4 28.085 # Si

Atoms # atomic

1 1 2.0 2.0 2.0
2 1 7.0 7.0 2.0
3 2 2.8 2.0 2.0
4 2 1.2 2.0 2.0
5 2 7.8 7.0 2.0
6 3 5.0 5.0 2.0
7 2 5.9 5.0 2.0
8 2 4.55 5.78 2.0
9 2 4.55 4.22 2.0
""",
        encoding="utf-8",
    )


def _write_dump(path: Path, *, boundary: str = "pp pp ff") -> None:
    atoms = [
        "1 1 2.0 2.0 2.0 0 0 0",
        "2 1 7.0 7.0 2.0 0 0 0",
        "3 2 2.8 2.0 2.0 0 0 0",
        "4 2 1.2 2.0 2.0 0 0 0",
        "5 2 7.8 7.0 2.0 0 0 0",
        "6 3 5.0 5.0 2.0 0 0 0",
        "7 2 5.9 5.0 2.0 0 0 0",
        "8 2 4.55 5.78 2.0 0 0 0",
        "9 2 4.55 4.22 2.0 0 0 0",
    ]
    lines = []
    for step in (0, 10, 20):
        lines.extend(
            [
                "ITEM: TIMESTEP",
                str(step),
                "ITEM: NUMBER OF ATOMS",
                "9",
                f"ITEM: BOX BOUNDS {boundary}",
                "0 10",
                "0 10",
                "0 10",
                "ITEM: ATOMS id type x y z ix iy iz",
                *atoms,
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_species_timeseries_uses_inclusive_safe_endpoint(tmp_path):
    model = tmp_path / "model.data"
    trajectory = tmp_path / "state.lammpstrj"
    solution_ids = tmp_path / "solution.ids"
    _write_model(model)
    _write_dump(trajectory)
    solution_ids.write_text("1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_origin_step": 0,
                "timestep_fs": 1000.0,
                "sampling_stride_steps": 10,
                "oh_cutoff_A": 1.35,
                "oh_cutoff_sensitivity_A": [1.25, 1.35, 1.45],
                "write_plots": False,
                "cases": [
                    {
                        "case_id": "surface",
                        "branch_id": "f0",
                        "direction": "none",
                        "model_data": str(model),
                        "solution_oxygen_ids": str(solution_ids),
                        "trajectories": [str(trajectory)],
                        "maximum_timestep": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = run_contract(contract, tmp_path / "results")
    assert summary["status"] == "PASS"
    assert summary["periodic_z"] is False
    rows = _rows(tmp_path / "results" / "species_timeseries_10ps.tsv")
    assert [int(row["step"]) for row in rows] == [0, 10]
    assert all(int(row["solution_H2O"]) == 1 for row in rows)
    assert all(int(row["framework_OH"]) == 1 for row in rows)
    branch = _rows(tmp_path / "results" / "branch_species_summary.tsv")
    assert branch[0]["inventory_integrity_gate"] == "PASS"
    assert branch[0]["fixed_carbon_H"] == "3"
    assert branch[0]["proton_partition_stationarity"] == "NOT_ASSESSED_SHORT_WINDOW"


def test_periodic_z_preserves_a_water_across_the_box_boundary():
    ids = np.array([1, 2, 3])
    types = np.array([1, 2, 2])
    coords = np.array([[5.0, 5.0, 9.8], [5.0, 5.0, 0.2], [5.0, 5.0, 9.0]])
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    args = (ids, types, coords, bounds, {1: "O", 2: "H"}, 1.35, set())

    _, _, open_parents, open_assigned = assign_hydrogen_parents(*args)
    assert open_parents.tolist() == [-1, 1]
    assert open_assigned.tolist() == [False, True]

    _, _, periodic_parents, periodic_assigned = assign_hydrogen_parents(
        *args, periodic_z=True
    )
    assert periodic_parents.tolist() == [1, 1]
    assert periodic_assigned.tolist() == [True, True]


def test_contract_requires_matching_periodic_z_boundary(tmp_path):
    model = tmp_path / "model.data"
    trajectory = tmp_path / "state.lammpstrj"
    solution_ids = tmp_path / "solution.ids"
    _write_model(model)
    _write_dump(trajectory, boundary="pp pp pp")
    dump_text = trajectory.read_text(encoding="utf-8")
    dump_text = dump_text.replace("1 1 2.0 2.0 2.0 0 0 0", "1 1 2.0 2.0 9.8 0 0 0")
    dump_text = dump_text.replace("3 2 2.8 2.0 2.0 0 0 0", "3 2 2.0 2.0 0.2 0 0 0")
    dump_text = dump_text.replace("4 2 1.2 2.0 2.0 0 0 0", "4 2 2.0 2.0 9.0 0 0 0")
    trajectory.write_text(dump_text, encoding="utf-8")
    solution_ids.write_text("1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    payload = {
        "schema_version": 1,
        "time_origin_step": 0,
        "timestep_fs": 1000.0,
        "sampling_stride_steps": 10,
        "write_plots": False,
        "cases": [{
            "case_id": "surface",
            "branch_id": "f0",
            "direction": "none",
            "model_data": str(model),
            "solution_oxygen_ids": str(solution_ids),
            "trajectories": [str(trajectory)],
            "maximum_timestep": 10,
        }],
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with"):
        run_contract(contract, tmp_path / "rejected")

    payload["periodic_z"] = True
    contract.write_text(json.dumps(payload), encoding="utf-8")
    summary = run_contract(contract, tmp_path / "accepted")
    assert summary["status"] == "PASS"
    assert summary["periodic_z"] is True
    rows = _rows(tmp_path / "accepted" / "species_timeseries_10ps.tsv")
    assert all(int(row["solution_H2O"]) == 1 for row in rows)
    assert all(int(row["hydrogen_unassigned"]) == 0 for row in rows)


def test_unassigned_hydrogen_recovery_is_not_persistent():
    from molsimflow.postprocess.constant_force_species_timeseries import _assignment_events

    events = _assignment_events(
        case_id="surface", branch_id="f0", direction="none",
        steps=[0, 10, 20, 30], times=[0.0, 10.0, 20.0, 30.0],
        hydrogen_ids=np.array([2]),
        assignments=[np.array([-1]), np.array([1]), np.array([1]), np.array([1])],
        solution_oxygen_ids={1}, minimum_persistence_ps=20.0,
    )
    assert len(events) == 1
    assert events[0]["event_class"] == "UNASSIGNED_TO_SOLUTION"
    assert events[0]["persistent"] == "false"


def test_model_defined_methyl_hydrogens_are_fixed_when_an_oxygen_is_closer(tmp_path):
    model = tmp_path / "ambiguous.data"
    model.write_text(
        """5 atoms
3 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 15.999 # O
2 1.008 # H
3 12.011 # C

Atoms # atomic/kk

1 3 5.0 5.0 5.0 0 0 0
2 2 5.9 5.0 5.0 0 0 0
3 2 4.55 5.78 5.0 0 0 0
4 2 4.55 4.22 5.0 0 0 0
5 1 6.75 5.0 5.0 0 0 0
""",
        encoding="utf-8",
    )
    symbols = read_type_symbols(model)
    assert identify_fixed_carbon_hydrogen_ids(model, symbols, 1.25) == {2, 3, 4}
