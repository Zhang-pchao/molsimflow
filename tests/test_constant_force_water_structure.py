import csv
import json
from pathlib import Path

import pytest

from molsimflow.cli import build_parser
from molsimflow.postprocess.constant_force_water_structure import (
    edge_similarity,
    hbond_component_metrics,
    run_contract,
)


def _write_frame(
    lines: list[str], step: int, atoms: list[tuple[int, int, float, float, float]]
) -> None:
    lines.extend(
        [
            "ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(atoms)),
            "ITEM: BOX BOUNDS pp pp ff", "0 12", "0 12", "0 12",
            "ITEM: ATOMS id type x y z",
        ]
    )
    lines.extend(f"{atom_id} {atom_type} {x} {y} {z}" for atom_id, atom_type, x, y, z in atoms)


def _molecule(
    atom_id: int, x: float, y: float, z: float
) -> list[tuple[int, int, float, float, float]]:
    return [
        (atom_id, 2, x, y, z),
        (atom_id + 1, 1, x + 0.9, y, z),
        (atom_id + 2, 1, x, y + 0.9, z),
    ]


def _trajectory(path: Path) -> None:
    surface = [(1, 2, 5.0, 5.0, 1.0), (2, 1, 5.0, 5.0, 1.9),
               (3, 2, 1.0, 1.0, 1.0), (4, 1, 1.0, 1.0, 1.9)]
    first = surface + _molecule(5, 5.0, 5.0, 3.0) + _molecule(8, 6.0, 5.0, 3.0)
    first += _molecule(11, 5.0, 5.0, 6.0) + _molecule(14, 6.0, 5.0, 6.0)
    second = surface + _molecule(5, 5.0, 5.0, 5.0) + _molecule(8, 6.0, 5.0, 3.0)
    second += _molecule(11, 5.0, 5.0, 6.0) + _molecule(14, 6.0, 5.0, 6.0)
    lines: list[str] = []
    _write_frame(lines, 0, first)
    _write_frame(lines, 10, second)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_edge_similarity_and_network_metrics():
    previous = {(1, 2), (2, 3)}
    current = {(2, 3), (3, 4)}
    jaccard, turnover = edge_similarity(previous, current)
    assert jaccard == pytest.approx(1.0 / 3.0)
    assert turnover == pytest.approx(0.5)
    count, largest = hbond_component_metrics({1, 2, 3, 4}, previous | current)
    assert count == 1
    assert largest == 1.0


def test_layer_contract_reports_exchange_and_one_ps_persistence(tmp_path):
    trajectory = tmp_path / "state.lammpstrj"
    _trajectory(trajectory)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_origin_step": 0,
                "timestep_fs": 100.0,
                "surface_atom_range": [1, 4],
                "water_atom_range": [5, 16],
                "oxygen_type": 2,
                "hydrogen_type": 1,
                "oh_cutoff_A": 1.25,
                "oo_cutoff_A": 3.5,
                "hbond_angle_deg": 30.0,
                "lsi_cutoff_A": 3.7,
                "lsi_neighbor_cap": 8,
                "cluster_cutoff_A": 3.5,
                "write_plots": False,
                "cases": [
                    {
                        "case_id": "film",
                        "branch_id": "fx",
                        "direction": "x",
                        "region_mode": "layers",
                        "surface_z_A": 1.0,
                        "z_edges_A": [0.0, 4.0, 8.0],
                        "trajectories": [str(trajectory)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = run_contract(contract, tmp_path / "output")
    assert summary["status"] == "PASS"
    with (tmp_path / "output" / "region_exchange.tsv").open(newline="") as handle:
        exchanges = list(csv.DictReader(handle, delimiter="\t"))
    assert any(
        row["from_region"] == "layer_0" and row["to_region"] == "layer_1"
        for row in exchanges
    )
    with (tmp_path / "output" / "hbond_persistence_summary.tsv").open(newline="") as handle:
        persistence = list(csv.DictReader(handle, delimiter="\t"))
    assert {float(row["sampling_interval_ps"]) for row in persistence} == {1.0}
    assert (tmp_path / "output" / "OUTPUT-SHA256SUMS").is_file()


def test_cli_registers_water_structure_contract(tmp_path):
    args = build_parser().parse_args(
        [
            "postprocess", "constant-force-water-structure",
            "--contract", str(tmp_path / "contract.json"),
            "--output", str(tmp_path / "output"),
        ]
    )
    assert args.func.__name__ == "_cmd_postprocess_constant_force_water_structure"
