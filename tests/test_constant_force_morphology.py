import csv
import json
from pathlib import Path

import numpy as np
import pytest

from molsimflow.cli import build_parser
from molsimflow.postprocess.constant_force_islands import (
    _exchange_class,
    match_components,
)
from molsimflow.postprocess.constant_force_islands import (
    run_contract as run_islands,
)
from molsimflow.postprocess.constant_force_layers import nearest_sites
from molsimflow.postprocess.constant_force_layers import run_contract as run_layers
from molsimflow.postprocess.constant_force_oxygen import connected_components


def _write_dump(path: Path, frames: list[tuple[int, list[tuple[int, float, float, float, float, float, float]]]]) -> None:
    lines = []
    for step, atoms in frames:
        lines.extend(
            [
                "ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(atoms)),
                "ITEM: BOX BOUNDS pp pp ff", "0 10", "0 10", "0 10",
                "ITEM: ATOMS id type x y z vx vy vz",
            ]
        )
        lines.extend(
            f"{atom_id} 2 {x} {y} {z} {vx} {vy} {vz}"
            for atom_id, x, y, z, vx, vy, vz in atoms
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_connected_components_is_periodic_in_xy_and_not_z():
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    points = np.asarray([[0.2, 1.0, 1.0], [9.8, 1.0, 1.0], [0.2, 1.0, 9.8]])
    components = connected_components(points, bounds, 1.0)
    assert [len(component) for component in components] == [2, 1]


def test_island_tracking_resolves_merge_and_split(tmp_path):
    trajectory = tmp_path / "oxygen.lammpstrj"
    _write_dump(
        trajectory,
        [
            (0, [(1, 1.0, 1.0, 1.0, 0, 0, 0), (2, 1.5, 1.0, 1.0, 0, 0, 0),
                 (3, 7.0, 7.0, 1.0, 0, 0, 0), (4, 7.5, 7.0, 1.0, 0, 0, 0)]),
            (10, [(1, 4.6, 5.0, 1.0, 0, 0, 0), (2, 4.9, 5.0, 1.0, 0, 0, 0),
                  (3, 5.2, 5.0, 1.0, 0, 0, 0), (4, 5.5, 5.0, 1.0, 0, 0, 0)]),
            (20, [(1, 1.0, 1.0, 1.0, 0, 0, 0), (2, 1.5, 1.0, 1.0, 0, 0, 0),
                  (3, 7.0, 7.0, 1.0, 0, 0, 0), (4, 7.5, 7.0, 1.0, 0, 0, 0)]),
        ],
    )
    contract = tmp_path / "islands.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_origin_step": 0,
                "timestep_fs": 1000.0,
                "cluster_cutoff_A": 1.0,
                "lineage_overlap_fraction": 0.25,
                "event_overlap_fraction": 0.25,
                "event_minimum_overlap_count": 1,
                "write_plots": False,
                "cases": [{
                    "case_id": "surface", "branch_id": "fx", "direction": "x",
                    "trajectories": [str(trajectory)],
                }],
            }
        ),
        encoding="utf-8",
    )
    summary = run_islands(contract, tmp_path / "islands-output")
    assert summary["merge_events"] == 1
    assert summary["split_events"] == 1
    events = _read_tsv(tmp_path / "islands-output" / "lineage_events.tsv")
    assert {row["event_type"] for row in events} == {"MERGE", "SPLIT"}
    exchanges = _read_tsv(tmp_path / "islands-output" / "molecule_exchange.tsv")
    assert len(exchanges) == 4
    assert {row["exchange_class"] for row in exchanges} == {
        "MERGE_LINEAGE_REASSIGNMENT", "SPLIT_LINEAGE_REASSIGNMENT",
    }
    exchanged_ids = {int(row["oxygen_id"]) for row in exchanges}
    assert len(exchanged_ids) == 2
    assert all(sum(int(row["oxygen_id"]) == oxygen_id for row in exchanges) == 2
               for oxygen_id in exchanged_ids)
    branch = _read_tsv(tmp_path / "islands-output" / "branch_transport_summary.tsv")
    assert len(branch) == 1
    assert int(branch[0]["track_to_track_transfer_count"]) == 4
    assert int(branch[0]["lineage_reassignment_count"]) == 4
    assert (tmp_path / "islands-output" / "OUTPUT-SHA256SUMS").is_file()


def test_component_matching_keeps_large_lineage_during_satellite_merge():
    previous = {1: set(range(100)), 2: {100}}
    current = [set(range(101))]
    assignment, overlaps = match_components(
        previous,
        current,
        overlap_fraction=0.25,
    )
    assert overlaps == {(1, 0): 100, (2, 0): 1}
    assert assignment == {0: 1}


def test_island_exchange_tracks_persistent_identity_transfer(tmp_path):
    trajectory = tmp_path / "oxygen.lammpstrj"
    _write_dump(
        trajectory,
        [
            (0, [(1, 1.0, 1.0, 1.0, 0, 0, 0), (2, 1.5, 1.0, 1.0, 0, 0, 0),
                 (3, 7.0, 7.0, 1.0, 0, 0, 0), (4, 7.5, 7.0, 1.0, 0, 0, 0)]),
            (10, [(1, 1.0, 1.0, 1.0, 0, 0, 0),
                  (2, 6.5, 7.0, 1.0, 0, 0, 0), (3, 7.0, 7.0, 1.0, 0, 0, 0),
                  (4, 7.5, 7.0, 1.0, 0, 0, 0)]),
        ],
    )
    contract = tmp_path / "islands.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1, "time_origin_step": 0, "timestep_fs": 1000.0,
                "cluster_cutoff_A": 1.0, "lineage_overlap_fraction": 0.25,
                "event_overlap_fraction": 0.25, "event_minimum_overlap_count": 1,
                "write_plots": False,
                "cases": [{
                    "case_id": "surface", "branch_id": "fx", "direction": "x",
                    "trajectories": [str(trajectory)],
                }],
            }
        ),
        encoding="utf-8",
    )
    summary = run_islands(contract, tmp_path / "islands-output")
    assert summary["molecule_exchange_rows"] == 1
    exchanges = _read_tsv(tmp_path / "islands-output" / "molecule_exchange.tsv")
    assert exchanges[0]["exchange_class"] == "PERSISTENT_ISLAND_TRANSFER"
    assert int(exchanges[0]["oxygen_id"]) == 2
    tracks = _read_tsv(tmp_path / "islands-output" / "track_summary.tsv")
    main = max(tracks, key=lambda row: float(row["mean_size"]))
    assert int(main["net_oxygen_transfer"]) == 1


def test_exchange_class_distinguishes_untracked_entry_and_exit():
    assert _exchange_class(None, 2, {1}, {1, 2}) == "ENTRY_FROM_UNTRACKED"
    assert _exchange_class(2, None, {1, 2}, {1}) == "EXIT_TO_UNTRACKED"


def test_island_center_unwraps_across_periodic_boundary_without_identity_exchange(tmp_path):
    trajectory = tmp_path / "oxygen.lammpstrj"
    _write_dump(
        trajectory,
        [
            (0, [(1, 9.8, 1.0, 1.0, 0, 0, 0), (2, 0.2, 1.0, 1.0, 0, 0, 0)]),
            (10, [(1, 0.2, 1.0, 1.0, 0, 0, 0), (2, 0.6, 1.0, 1.0, 0, 0, 0)]),
        ],
    )
    contract = tmp_path / "islands.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1, "time_origin_step": 0, "timestep_fs": 1000.0,
                "cluster_cutoff_A": 1.0, "write_plots": False,
                "cases": [{
                    "case_id": "surface", "branch_id": "fx", "direction": "x",
                    "trajectories": [str(trajectory)],
                }],
            }
        ),
        encoding="utf-8",
    )
    summary = run_islands(contract, tmp_path / "islands-output")
    assert summary["molecule_exchange_rows"] == 0
    rows = _read_tsv(tmp_path / "islands-output" / "island_timeseries.tsv")
    assert [int(row["track_id"]) for row in rows] == [1, 1]
    assert float(rows[1]["center_x_unwrapped_A"]) - float(rows[0]["center_x_unwrapped_A"]) == pytest.approx(0.4)
    assert float(rows[1]["vx_mps"]) == pytest.approx(4.0)


def _layer_contract(tmp_path: Path) -> Path:
    positions = [(1, 1.0, 1.0, 1.0), (2, 2.0, 1.0, 1.5), (3, 3.0, 1.0, 2.5), (4, 4.0, 1.0, 3.0)]
    cases = []
    for branch, direction, vx, vy in (("f0", "none", 0.0, 0.0), ("fx", "x", 0.01, 0.0), ("fy", "y", 0.0, 0.02)):
        path = tmp_path / f"{branch}.lammpstrj"
        first = [(atom_id, x, y, z, vx, vy, 0.0) for atom_id, x, y, z in positions]
        second = [(atom_id, x, y, (2.2 if atom_id == 2 else z), vx, vy, 0.0) for atom_id, x, y, z in positions]
        _write_dump(path, [(0, first), (10, second)])
        cases.append({
            "case_id": "film", "branch_id": branch, "direction": direction,
            "surface_z_A": 0.0, "trajectories": [str(path)],
        })
    contract = tmp_path / "layers.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1, "time_origin_step": 0, "timestep_fs": 1000.0,
                "z_edges_A": [0.0, 2.0, 4.0], "density_modes": [[1, 0], [0, 1]],
                "block_ps": 10.0, "write_plots": False, "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return contract


def test_layer_transport_subtracts_baseline_and_tracks_exchange(tmp_path):
    summary = run_layers(_layer_contract(tmp_path), tmp_path / "layers-output")
    assert summary["layers"] == 2
    exchanges = _read_tsv(tmp_path / "layers-output" / "layer_exchange.tsv")
    assert any(row["from_layer"] == "0" and row["to_layer"] == "1" for row in exchanges)
    responses = _read_tsv(tmp_path / "layers-output" / "layer_response_summary.tsv")
    fx = [row for row in responses if row["branch_id"] == "fx"]
    fy = [row for row in responses if row["branch_id"] == "fy"]
    assert [float(row["mean_excess_axis_velocity_mps"]) for row in fx] == pytest.approx([1.0, 1.0])
    assert [float(row["mean_excess_axis_velocity_mps"]) for row in fy] == pytest.approx([2.0, 2.0])
    assert [float(row["mean_count"]) for row in fx] == pytest.approx([1.5, 2.5])
    assert [float(row["occupied_fraction"]) for row in fx] == pytest.approx([1.0, 1.0])
    assert [float(row["mean_excess_surface_flux_molecules_per_A_ps"]) for row in fx] == pytest.approx(
        [0.00015, 0.00025]
    )
    assert (tmp_path / "layers-output" / "density_modes.tsv").is_file()


def test_nearest_sites_respects_xy_periodicity():
    indices, distances = nearest_sites(
        np.asarray([[9.9, 1.0]]), np.asarray([[0.1, 1.0]]), np.asarray([10.0, 10.0]), 0.5
    )
    assert indices.tolist() == [0]
    assert distances.tolist() == pytest.approx([0.2])


def test_cli_registers_morphology_contract_commands(tmp_path):
    parser = build_parser()
    islands = parser.parse_args([
        "postprocess", "constant-force-islands", "--contract", str(tmp_path / "c.json"),
        "--output", str(tmp_path / "out"),
    ])
    layers = parser.parse_args([
        "postprocess", "constant-force-layers", "--contract", str(tmp_path / "c.json"),
        "--output", str(tmp_path / "out"),
    ])
    assert islands.func.__name__ == "_cmd_postprocess_constant_force_islands"
    assert layers.func.__name__ == "_cmd_postprocess_constant_force_layers"
