from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from molsimflow.cli import build_parser
from molsimflow.postprocess.nanobubble_flow_response import (
    _tracked_periodic_center,
    run_contract,
)


def _write_data(path: Path) -> None:
    path.write_text(
        "LAMMPS data\n\n10 atoms\n3 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 1\n2 16\n3 14\n\nAtoms # atomic\n\n"
        "1 2 1 1 0\n2 2 5 1 0\n"
        "3 3 2.0 2 1.0\n4 3 2.1 2 1.0\n"
        "5 3 3.0 2 1.0\n6 3 3.1 2 1.0\n"
        "7 3 4.0 2 1.0\n8 3 4.1 2 1.0\n"
        "9 3 8.0 8 8.0\n10 3 8.1 8 8.0\n",
        encoding="utf-8",
    )


def _write_dump(path: Path) -> None:
    frames = []
    for step, shift in ((0, 0.0), (10, 1.0), (20, 2.0)):
        atoms = [
            f"{atom_id} 3 {x + shift} 2 1 1 0 0"
            for atom_id, x in ((3, 2.0), (4, 2.1), (5, 3.0), (6, 3.1), (7, 4.0), (8, 4.1))
        ]
        atoms.extend(("9 3 8.0 8 8 0.1 0 0", "10 3 8.1 8 8 0.1 0 0"))
        frames.append(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n8\n"
            "ITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type x y z f_RAW[1] f_RAW[2] f_RAW[3]\n"
            + "\n".join(atoms) + "\n"
        )
    path.write_text("".join(frames), encoding="utf-8")


def _write_motion(path: Path) -> None:
    columns = (
        "TimeStep v_dxwaterO v_dywaterO v_dxrelN v_dyrelN "
        "c_COMsub[1] c_COMsub[2] c_COMsub[3]"
    )
    path.write_text(
        f"# averaged\n# {columns}\n"
        "0 0 0 0 0 0 0 0\n10 0.5 0 1 0 0 0 0\n20 1 0 2 0 0 0 0\n",
        encoding="utf-8",
    )


def test_nanobubble_flow_response_tracks_main_cluster_and_water(tmp_path: Path) -> None:
    data = tmp_path / "model.data"
    dump = tmp_path / "nitrogen.dump"
    motion = tmp_path / "motion.dat"
    _write_data(data); _write_dump(dump); _write_motion(motion)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "case_id": "surface", "branch_id": "fx", "direction": "x",
        "time_origin_step": 0, "timestep_fs": 1000.0,
        "surface_model_data": str(data), "surface_atom_range": [1, 2],
        "nitrogen_atom_range": [3, 10], "relocated_nitrogen_ids": [9, 10],
        "nitrogen_trajectories": [{"path": str(dump), "min_step": 0, "max_step": 20}],
        "motion_tables": [str(motion)], "analysis_max_step": 20,
        "cluster_cutoff_A": 1.5, "contact_cutoff_A": 2.0, "block_ps": 20.0,
        "minimum_motion_mps": 0.1, "write_plots": False,
    }), encoding="utf-8")
    output = tmp_path / "out"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert summary["frames"] == 3
    assert summary["main_bubble_n2_min"] == 3
    assert summary["relocated_n2"] == 1
    assert summary["bubble_drive_final_A"] == pytest.approx(2.0)
    assert summary["water_drive_final_A"] == pytest.approx(1.0)
    with (output / "frame_metrics.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert float(rows[-1]["bubble_minus_water_drive_A"]) == pytest.approx(1.0)
    assert int(rows[-1]["max_all_raw_force_atom_id"]) in {3, 4, 5, 6, 7, 8}
    assert (output / "OUTPUT-SHA256SUMS").is_file()


def test_cli_registers_nanobubble_flow_response(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "postprocess", "nanobubble-flow-response",
        "--contract", str(tmp_path / "contract.json"),
        "--output", str(tmp_path / "out"),
    ])
    assert args.contract == tmp_path / "contract.json"


def test_tracked_periodic_center_stays_on_previous_branch() -> None:
    import numpy as np

    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    points = np.asarray([[9.8, 2.0, 1.0], [0.2, 2.0, 1.0]])
    center = _tracked_periodic_center(points, bounds, np.asarray([9.9, 2.0, 1.0]))
    assert min(abs(center[0]), abs(center[0] - 10.0)) < 1.0e-12
    assert center[1:] == pytest.approx([2.0, 1.0])
