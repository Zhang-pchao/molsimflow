from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from molsimflow.cli import build_parser
from molsimflow.io.lammps_dump import iter_lammps_dump_records
from molsimflow.postprocess.nanobubble_local_flow import (
    flow_corrected_temperature,
    run_contract,
)


def _write_data(path: Path) -> None:
    atoms = [
        "1 8 1 2 0", "2 8 3 2 0", "3 8 5 2 0",
        "4 8 1 4 0", "5 8 3 4 0", "6 8 5 4 0",
        "7 3 2.4 3 1", "8 3 2.6 3 1",
        "9 3 3.4 3 1", "10 3 3.6 3 1",
        "11 3 4.4 3 1", "12 3 4.6 3 1",
        "13 2 1.5 3 1", "14 1 1.4 3 1", "15 1 1.6 3 1",
        "16 2 5.5 3 1", "17 1 5.4 3 1", "18 1 5.6 3 1",
        "19 2 8.0 8 6", "20 1 7.9 8 6", "21 1 8.1 8 6",
    ]
    path.write_text(
        "LAMMPS data\n\n21 atoms\n8 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 1.008\n2 15.999\n3 14.007\n4 22.99\n5 35.45\n"
        "6 47.867\n7 12.011\n8 28.085\n\nAtoms # atomic\n\n"
        + "\n".join(atoms) + "\n",
        encoding="utf-8",
    )


def _write_dumps(state: Path, velocity: Path) -> None:
    state_frames = []
    velocity_frames = []
    base = {
        1: (8, 1, 2, 0), 2: (8, 3, 2, 0), 3: (8, 5, 2, 0),
        4: (8, 1, 4, 0), 5: (8, 3, 4, 0), 6: (8, 5, 4, 0),
        7: (3, 2.4, 3, 1), 8: (3, 2.6, 3, 1),
        9: (3, 3.4, 3, 1), 10: (3, 3.6, 3, 1),
        11: (3, 4.4, 3, 1), 12: (3, 4.6, 3, 1),
        13: (2, 1.5, 3, 1), 14: (1, 1.4, 3, 1), 15: (1, 1.6, 3, 1),
        16: (2, 5.5, 3, 1), 17: (1, 5.4, 3, 1), 18: (1, 5.6, 3, 1),
        19: (2, 8, 8, 6), 20: (1, 7.9, 8, 6), 21: (1, 8.1, 8, 6),
    }
    for step, shift in ((0, 0.0), (10, 1.0), (20, 2.0)):
        state_rows = []
        for atom_id, (atom_type, x, y, z) in base.items():
            moved = shift if 7 <= atom_id <= 18 else 0.0
            state_rows.append(
                f"{atom_id} {atom_type} {x + moved} {y} {z} 0 0 0 0.05 0 0"
            )
        state_frames.append(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n21\n"
            "ITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type x y z ix iy iz vx vy vz\n"
            + "\n".join(state_rows) + "\n"
        )
        velocity_rows = []
        for atom_id in range(7, 22):
            atom_type = base[atom_id][0]
            velocity_rows.append(f"{atom_id} {atom_type} 0.05 0 0")
        velocity_frames.append(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n15\n"
            "ITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type vx vy vz\n" + "\n".join(velocity_rows) + "\n"
        )
    state.write_text("".join(state_frames), encoding="utf-8")
    velocity.write_text("".join(velocity_frames), encoding="utf-8")


def _write_synchronized_subsets(state: Path, velocity: Path, paths: list[Path]) -> None:
    outputs = [[] for _ in paths]
    ranges = ((1, 6), (7, 12), (13, 18))
    for coordinate, speeds in zip(
        iter_lammps_dump_records(state), iter_lammps_dump_records(velocity)
    ):
        coordinate_column = {name: coordinate.atom_fields.index(name) for name in coordinate.atom_fields}
        speed_column = {name: speeds.atom_fields.index(name) for name in speeds.atom_fields}
        speed_by_id = {
            int(row[speed_column["id"]]): row for row in speeds.atom_rows
        }
        merged = []
        for row in coordinate.atom_rows:
            atom_id = int(row[coordinate_column["id"]])
            atom_type = int(row[coordinate_column["type"]])
            xyz = [row[coordinate_column[axis]] for axis in "xyz"]
            speed = speed_by_id.get(atom_id)
            velocity_values = (
                [speed[speed_column[f"v{axis}"]] for axis in "xyz"]
                if speed is not None else ["0", "0", "0"]
            )
            merged.append((atom_id, [str(atom_id), str(atom_type), *xyz, *velocity_values]))
        for output, (lower, upper) in zip(outputs, ranges):
            selected = [row for atom_id, row in merged if lower <= atom_id <= upper]
            output.append(
                f"ITEM: TIMESTEP\n{coordinate.timestep}\n"
                f"ITEM: NUMBER OF ATOMS\n{len(selected)}\n"
                "ITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n0 10\n"
                "ITEM: ATOMS id type x y z vx vy vz\n"
                + "\n".join(" ".join(row) for row in selected) + "\n"
            )
    for path, frames in zip(paths, outputs):
        path.write_text("".join(frames), encoding="utf-8")


def test_flow_corrected_temperature_removes_uniform_streaming() -> None:
    import numpy as np

    atom_types = np.asarray([1, 1, 2])
    velocities = np.asarray([[2.0, 0.0, 0.0]] * 3)
    temperature = flow_corrected_temperature(
        atom_types, velocities, {1: 1.0, 2: 16.0},
        direction_index=0, transverse_index=1,
    )
    assert temperature == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_nanobubble_local_flow_joins_frames_and_removes_advection(tmp_path: Path) -> None:
    model = tmp_path / "model.data"
    state = tmp_path / "state.dump"
    velocity = tmp_path / "velocity.dump"
    _write_data(model); _write_dumps(state, velocity)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "timestep_fs": 1000.0, "sample_stride_steps": 10,
        "cluster_cutoff_A": 1.5, "contact_cutoff_A": 2.0,
        "edge_window_halfwidth_A": 2.0, "transverse_padding_A": 2.0,
        "near_surface_height_A": 4.0, "film_height_A": 3.0,
        "bulk_z_min_A": 4.0, "bulk_z_max_A": 9.0,
        "minimum_region_water_oxygen": 1, "block_ps": 20.0,
        "write_plots": False, "hash_inputs": False,
        "case": {
            "case_id": "surface", "branch_id": "fx", "direction": "x",
            "time_origin_step": 0, "maximum_step": 20,
            "surface_model_data": str(model), "surface_atom_range": [1, 6],
            "nitrogen_atom_range": [7, 12], "water_atom_range": [13, 21],
            "relocated_nitrogen_ids": [], "state_trajectories": [str(state)],
            "velocity_trajectories": [str(velocity)],
        },
    }), encoding="utf-8")
    output = tmp_path / "out"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert summary["frames"] == 3
    assert summary["sample_interval_ps"] == pytest.approx(10.0)
    with (output / "edge_flow_intervals.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    assert float(rows[0]["bubble_velocity_mps"]) == pytest.approx(10.0)
    assert float(rows[0]["front_water_velocity_mps"]) == pytest.approx(5.0)
    assert float(rows[0]["bubble_minus_mean_edge_water_mps"]) == pytest.approx(5.0)


def test_nanobubble_local_flow_joins_spatially_filtered_streams(tmp_path: Path) -> None:
    model = tmp_path / "model.data"
    state = tmp_path / "state.dump"
    velocity = tmp_path / "velocity.dump"
    subsets = [tmp_path / name for name in ("surface.dump", "nitrogen.dump", "water.dump")]
    _write_data(model); _write_dumps(state, velocity)
    _write_synchronized_subsets(state, velocity, subsets)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "timestep_fs": 1000.0, "sample_stride_steps": 10,
        "cluster_cutoff_A": 1.5, "contact_cutoff_A": 2.0,
        "edge_window_halfwidth_A": 2.0, "transverse_padding_A": 2.0,
        "near_surface_height_A": 4.0, "film_height_A": 3.0,
        "bulk_z_min_A": 4.0, "bulk_z_max_A": 9.0,
        "minimum_region_water_oxygen": 1, "block_ps": 20.0,
        "write_plots": False, "hash_inputs": False,
        "case": {
            "case_id": "surface", "branch_id": "fx", "direction": "x",
            "time_origin_step": 0, "maximum_step": 20,
            "surface_model_data": str(model), "surface_atom_range": [1, 6],
            "nitrogen_atom_range": [7, 12], "water_atom_range": [13, 21],
            "relocated_nitrogen_ids": [],
            "synchronized_state_trajectories": [{"paths": [str(path) for path in subsets]}],
        },
    }), encoding="utf-8")
    output = tmp_path / "out"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert summary["spatially_filtered_synchronized_streams"] is True
    assert summary["frames"] == 3
    with (output / "edge_flow_intervals.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert float(rows[0]["bubble_minus_mean_edge_water_mps"]) == pytest.approx(5.0)


def test_nanobubble_local_flow_accepts_one_complete_synchronized_stream(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.data"
    state = tmp_path / "state.dump"
    velocity = tmp_path / "velocity.dump"
    _write_data(model)
    _write_dumps(state, velocity)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "timestep_fs": 1000.0, "sample_stride_steps": 10,
        "cluster_cutoff_A": 1.5, "contact_cutoff_A": 2.0,
        "edge_window_halfwidth_A": 2.0, "transverse_padding_A": 2.0,
        "near_surface_height_A": 4.0, "film_height_A": 3.0,
        "bulk_z_min_A": 4.0, "bulk_z_max_A": 9.0,
        "minimum_region_water_oxygen": 1, "block_ps": 20.0,
        "write_plots": False, "hash_inputs": False,
        "case": {
            "case_id": "surface", "branch_id": "fx", "direction": "x",
            "time_origin_step": 0, "maximum_step": 20,
            "surface_model_data": str(model), "surface_atom_range": [1, 6],
            "nitrogen_atom_range": [7, 12], "water_atom_range": [13, 21],
            "relocated_nitrogen_ids": [],
            "synchronized_state_trajectories": [{"paths": [str(state)]}],
        },
    }), encoding="utf-8")
    output = tmp_path / "out"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert summary["spatially_filtered_synchronized_streams"] is True
    assert summary["frames"] == 3


def test_cli_registers_nanobubble_local_flow(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "postprocess", "nanobubble-local-flow", "--contract", str(tmp_path / "contract.json"),
        "--output", str(tmp_path / "out"),
    ])
    assert args.contract == tmp_path / "contract.json"


def test_nanobubble_local_flow_supports_explicit_no_contact_control(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.data"
    state = tmp_path / "state.dump"
    velocity = tmp_path / "velocity.dump"
    _write_data(model)
    _write_dumps(state, velocity)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "timestep_fs": 1000.0, "sample_stride_steps": 10,
        "cluster_cutoff_A": 1.5, "contact_cutoff_A": 0.1,
        "no_contact_policy": "gas_projection_control",
        "edge_window_halfwidth_A": 2.0, "transverse_padding_A": 2.0,
        "near_surface_height_A": 4.0, "film_height_A": 3.0,
        "bulk_z_min_A": 4.0, "bulk_z_max_A": 9.0,
        "minimum_region_water_oxygen": 1, "block_ps": 20.0,
        "write_plots": False, "hash_inputs": False,
        "case": {
            "case_id": "surface", "branch_id": "no_contact", "direction": "x",
            "time_origin_step": 0, "maximum_step": 20,
            "surface_model_data": str(model), "surface_atom_range": [1, 6],
            "nitrogen_atom_range": [7, 12], "water_atom_range": [13, 21],
            "relocated_nitrogen_ids": [], "state_trajectories": [str(state)],
            "velocity_trajectories": [str(velocity)],
        },
    }), encoding="utf-8")
    output = tmp_path / "out"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert summary["all_frames_have_contact_line"] is False
    with (output / "local_flow_geometry.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["contact_line_valid"] for row in rows} == {"0"}
    assert {row["geometry_reference"] for row in rows} == {
        "gas_projection_control"
    }
