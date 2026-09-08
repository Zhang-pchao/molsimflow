import csv
import sys
import types

import numpy as np

from molsimflow.postprocess.tpcl_virtual_response import run_virtual_response


def test_symmetric_virtual_response_recovers_quadratic_stiffness(tmp_path, monkeypatch):
    dump = tmp_path / "state.lammpstrj"
    atom_rows = []
    atom_id = 1
    for x in (4.25, 4.75, 5.25, 5.75):
        for y in (4.5, 5.0, 5.5):
            atom_rows.append(f"{atom_id} {1 + atom_id % 2} {x} {y} 2.0")
            atom_id += 1
    dump.write_text(
        "ITEM: TIMESTEP\n10\n"
        "ITEM: NUMBER OF ATOMS\n12\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z\n"
        + "\n".join(atom_rows)
        + "\n",
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots.tsv"
    row = {
        "pair_id": "p1",
        "case_id": "case",
        "sample_kind": "event",
        "phase": "pre",
        "step": 10,
        "dump_path": dump,
        "contact_x_A": 5,
        "contact_y_A": 5,
        "radial_x": 1,
        "radial_y": 0,
        "patch_radius_A": 6,
        "source_time_block_200ps": 0,
        "response_stratum": 0,
        "response_affected_arc_fraction": 0.5,
    }
    with snapshots.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    mobile = tmp_path / "mobile.tsv"
    mobile.write_text(
        "case_id\tfirst_atom_id\tlast_atom_id\tlabel\ncase\t1\t12\tfluid\n",
        encoding="utf-8",
    )

    class QuadraticDeepPot:
        def __init__(self, _path):
            pass

        @staticmethod
        def get_type_map():
            return ["H", "C", "N", "O"]

        @staticmethod
        def eval(coordinates, cell, atom_types, atomic):
            del cell
            assert not atomic
            assert set(atom_types.tolist()) == {0, 3}
            values = coordinates.reshape(1, -1, 3)
            energy = 0.5 * np.sum(values**2, axis=(1, 2), keepdims=True)
            forces = -values
            return energy, forces, np.zeros((1, 9))

    deepmd = types.ModuleType("deepmd")
    infer = types.ModuleType("deepmd.infer")
    infer.DeepPot = QuadraticDeepPot
    monkeypatch.setitem(sys.modules, "deepmd", deepmd)
    monkeypatch.setitem(sys.modules, "deepmd.infer", infer)
    model = tmp_path / "model.pt2"
    model.write_bytes(b"fake")
    output = tmp_path / "output"
    summary = run_virtual_response(
        snapshots,
        mobile,
        model,
        ("H", "O"),
        ("H",),
        output,
        modes=("radial",),
        displacement_magnitudes_A=(0.005, 0.010, 0.020),
        minimum_pair_distance_floor_A=0.1,
    )
    assert summary["response_row_count"] == 1
    assert summary["displacement_row_count"] == 3
    assert summary["linear_response_pass_count"] == 1
    with (output / "snapshot_virtual_response.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["static_stiffness_eV_A2"]) > 0.0
    assert int(rows[0]["linear_response_pass"]) == 1
