import csv
import json
from pathlib import Path

from molsimflow.postprocess.constant_force_species_timeseries import run_contract


def _write_model(path: Path) -> None:
    path.write_text(
        """7 atoms
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
""",
        encoding="utf-8",
    )


def _write_dump(path: Path) -> None:
    atoms = [
        "1 1 2.0 2.0 2.0 0 0 0",
        "2 1 7.0 7.0 2.0 0 0 0",
        "3 2 2.8 2.0 2.0 0 0 0",
        "4 2 1.2 2.0 2.0 0 0 0",
        "5 2 7.8 7.0 2.0 0 0 0",
        "6 3 5.0 5.0 2.0 0 0 0",
        "7 2 5.9 5.0 2.0 0 0 0",
    ]
    lines = []
    for step in (0, 10, 20):
        lines.extend(
            [
                "ITEM: TIMESTEP",
                str(step),
                "ITEM: NUMBER OF ATOMS",
                "7",
                "ITEM: BOX BOUNDS pp pp ff",
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
    rows = _rows(tmp_path / "results" / "species_timeseries_10ps.tsv")
    assert [int(row["step"]) for row in rows] == [0, 10]
    assert all(int(row["solution_H2O"]) == 1 for row in rows)
    assert all(int(row["framework_OH"]) == 1 for row in rows)
    branch = _rows(tmp_path / "results" / "branch_species_summary.tsv")
    assert branch[0]["inventory_integrity_gate"] == "PASS"
