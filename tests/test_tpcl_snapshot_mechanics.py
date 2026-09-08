import csv
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from molsimflow.postprocess.tpcl_snapshot_mechanics import (
    DumpFrame,
    SnapshotSpec,
    _evaluate_global,
    _minimum_pair_distance,
    _project_tensor,
    _read_specs,
    read_selected_frames,
    run_predictions,
)


def _snapshot_spec(dump_path, step):
    return SnapshotSpec(
        snapshot_id=f"p1__event__{step}",
        pair_id="p1",
        case_id="case",
        sample_kind="event",
        phase="pre",
        step=step,
        dump_path=dump_path,
        contact_x_A=1.0,
        contact_y_A=2.0,
        radial_x=1.0,
        radial_y=0.0,
        patch_radius_A=6.0,
        source_time_block_200ps=0,
        response_stratum=0,
        response_affected_arc_fraction=0.5,
    )


def _dump_frame(step, atom_rows=("2 2 2 3 4", "1 1 1 2 3")):
    return (
        f"ITEM: TIMESTEP\n{step}\n"
        f"ITEM: NUMBER OF ATOMS\n{len(atom_rows)}\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 5\n0 6\n0 7\n"
        "ITEM: ATOMS id type x y z\n"
        + "\n".join(atom_rows)
        + "\n"
    )


def test_reads_selected_orthogonal_dump_without_changing_atom_order(tmp_path):
    dump = tmp_path / "state.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n10\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 5\n0 6\n0 7\n"
        "ITEM: ATOMS id type x y z\n2 2 2 3 4\n1 1 1 2 3\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "snapshots.tsv"
    row = {
        "pair_id": "p1",
        "case_id": "case",
        "sample_kind": "event",
        "phase": "pre",
        "step": 10,
        "dump_path": dump,
        "contact_x_A": 1,
        "contact_y_A": 2,
        "radial_x": 1,
        "radial_y": 0,
        "patch_radius_A": 6,
        "source_time_block_200ps": 0,
        "response_stratum": 0,
        "response_affected_arc_fraction": 0.5,
    }
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    specs = _read_specs(manifest)
    frames = read_selected_frames(specs)
    frame = frames[specs[0].snapshot_id]
    assert frame.atom_ids.tolist() == [1, 2]
    assert frame.atom_types.tolist() == [1, 2]
    assert np.allclose(frame.coordinates, [[1, 2, 3], [2, 3, 4]])
    assert np.allclose(frame.cell, np.diag([5, 6, 7]))


def test_selected_zstd_frame_ignores_only_later_corrupt_tail(tmp_path):
    if shutil.which("zstd") is None:
        pytest.skip("zstd executable is unavailable")
    first = tmp_path / "first.dump"
    second = tmp_path / "second.dump"
    first.write_text(_dump_frame(10), encoding="utf-8")
    second_rows = tuple(
        f"{index} 1 {index % 97} {index % 89} {index % 83}"
        for index in range(1, 5001)
    )
    second.write_text(_dump_frame(20, second_rows), encoding="utf-8")
    first_zst = tmp_path / "first.zst"
    second_zst = tmp_path / "second.zst"
    for source, target in ((first, first_zst), (second, second_zst)):
        subprocess.run(
            ["zstd", "-q", "-f", str(source), "-o", str(target)],
            check=True,
        )
    combined = tmp_path / "combined.lammpstrj.zst"
    second_bytes = second_zst.read_bytes()
    combined.write_bytes(first_zst.read_bytes() + second_bytes[: len(second_bytes) // 2])

    spec = _snapshot_spec(combined, 10)
    frames = read_selected_frames([spec])

    assert frames[spec.snapshot_id].step == 10


def test_selected_frame_at_corrupt_zstd_tail_fails_closed(tmp_path):
    if shutil.which("zstd") is None:
        pytest.skip("zstd executable is unavailable")
    source = tmp_path / "source.dump"
    rows = tuple(
        f"{index} 1 {index % 97} {index % 89} {index % 83}"
        for index in range(1, 5001)
    )
    source.write_text(_dump_frame(20, rows), encoding="utf-8")
    compressed = tmp_path / "complete.zst"
    subprocess.run(
        ["zstd", "-q", "-f", str(source), "-o", str(compressed)],
        check=True,
    )
    damaged = tmp_path / "damaged.lammpstrj.zst"
    compressed_bytes = compressed.read_bytes()
    damaged.write_bytes(compressed_bytes[: len(compressed_bytes) // 2])

    with pytest.raises(ValueError):
        read_selected_frames([_snapshot_spec(damaged, 20)])


def test_missing_selected_step_fails_after_complete_dump(tmp_path):
    dump = tmp_path / "state.lammpstrj"
    dump.write_text(_dump_frame(10) + _dump_frame(20), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing selected steps \[30\]"):
        read_selected_frames([_snapshot_spec(dump, 30)])


def test_periodic_minimum_distance_and_tensor_projection():
    coordinates = np.asarray([[0.1, 0.0, 0.0], [4.9, 0.0, 0.0], [2.5, 2.5, 2.5]])
    assert np.isclose(_minimum_pair_distance(coordinates, np.asarray([5.0, 5.0, 5.0])), 0.2)
    tensor = np.diag([2.0, 3.0, 4.0])
    assert _project_tensor(tensor, np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])) == 2.0


def test_global_evaluation_omits_atomic_outputs():
    frame = DumpFrame(
        step=10,
        atom_ids=np.asarray([1, 2]),
        atom_types=np.asarray([1, 2]),
        coordinates=np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        cell=np.diag([5.0, 6.0, 7.0]),
    )

    class FakeDeepPot:
        @staticmethod
        def eval(coordinates, cell, atom_types, atomic):
            assert coordinates.shape == (1, 6)
            assert cell.shape == (1, 9)
            assert atom_types.tolist() == [0, 3]
            assert atomic is False
            return (
                np.asarray([1.5]),
                np.ones((1, 2, 3)),
                np.arange(9).reshape(1, 9),
            )

    energy, forces, virial = _evaluate_global(
        FakeDeepPot(), frame, np.asarray([0, 3])
    )
    assert energy == 1.5
    assert forces.shape == (2, 3)
    assert virial.tolist() == list(range(9))


def test_zero_time_prediction_writes_radius_and_element_decompositions(tmp_path, monkeypatch):
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
    manifest = tmp_path / "snapshots.tsv"
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
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    class FakeDeepPot:
        def __init__(self, _path):
            pass

        @staticmethod
        def get_type_map():
            return ["H", "C", "N", "O"]

        @staticmethod
        def eval(coordinates, cell, atom_types, atomic):
            del cell
            assert set(atom_types.tolist()) == {0, 3}
            count = coordinates.size // 3
            if not atomic:
                return (
                    np.asarray([[count * 0.5]]),
                    np.ones((1, count, 3)),
                    np.full((1, 9), count),
                )
            atomic_energy = np.full((1, count, 1), 0.5)
            atomic_virial = np.ones((1, count, 9))
            return (
                np.asarray([[count * 0.5]]),
                np.ones((1, count, 3)),
                np.full((1, 9), count),
                atomic_energy,
                atomic_virial,
            )

    deepmd = types.ModuleType("deepmd")
    infer = types.ModuleType("deepmd.infer")
    infer.DeepPot = FakeDeepPot
    monkeypatch.setitem(sys.modules, "deepmd", deepmd)
    monkeypatch.setitem(sys.modules, "deepmd.infer", infer)
    model = tmp_path / "model.pt2"
    model.write_bytes(b"fake")
    output = tmp_path / "output"
    summary = run_predictions(
        manifest,
        model,
        ("H", "O"),
        ("H",),
        output,
        patch_radii_A=(4.0, 6.0, 8.0),
        minimum_pair_distance_floor_A=0.1,
    )
    assert summary["snapshot_count"] == 1
    with (output / "snapshot_mechanics.csv").open(newline="", encoding="utf-8") as handle:
        mechanics = list(csv.DictReader(handle))
    with (output / "snapshot_patch_by_type.csv").open(newline="", encoding="utf-8") as handle:
        by_type = list(csv.DictReader(handle))
    assert [float(item["patch_radius_A"]) for item in mechanics] == [4.0, 6.0, 8.0]
    assert sum(int(item["is_primary_patch_radius"]) for item in mechanics) == 1
    assert len(by_type) == 6

    force_output = tmp_path / "force_output"
    force_summary = run_predictions(
        manifest,
        model,
        ("H", "O"),
        ("H",),
        force_output,
        atomic_outputs=False,
        minimum_pair_distance_floor_A=0.1,
    )
    assert force_summary["atomic_outputs_requested"] is False
    assert force_summary["maximum_atomic_global_virial_difference_eV"] is None
    with (force_output / "snapshot_mechanics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        force_rows = list(csv.DictReader(handle))
    assert force_rows[0]["atomic_outputs_requested"] == "0"
    assert force_rows[0]["atomic_energy_sum_eV"] == ""
    assert force_rows[0]["patch_atomic_virial_nn_eV"] == ""
    assert float(force_rows[0]["patch_generalized_radial_force_eV_A"]) == 12.0
