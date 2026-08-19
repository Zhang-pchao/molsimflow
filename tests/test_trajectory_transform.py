from molsimflow.io.lammps_dump import iter_lammps_dump_records
from molsimflow.postprocess.trajectory_transform import (
    align_reference_layer,
    prepare_trajectory,
)


def _write_dump(path, frames):
    blocks = []
    for timestep, rows in frames:
        blocks.extend(
            [
                "ITEM: TIMESTEP",
                str(timestep),
                "ITEM: NUMBER OF ATOMS",
                str(len(rows)),
                "ITEM: BOX BOUNDS pp pp pp",
                "0 10",
                "0 10",
                "0 10",
                "ITEM: ATOMS id type x y z charge",
                *(" ".join(str(value) for value in row) for row in rows),
            ]
        )
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def test_prepare_trajectory_unwraps_and_shifts_without_losing_columns(tmp_path):
    source = tmp_path / "input.lammpstrj"
    _write_dump(source, [(0, [(1, 2, 0, 0, 9.8, -0.4)]), (1, [(1, 2, 0, 0, 0.2, -0.4)])])

    output = tmp_path / "prepared.lammpstrj"
    prepare_trajectory([source], output, unwrap_z=True, shift_min_z_A=1.0)
    frames = list(iter_lammps_dump_records(output))

    assert [float(frame.atom_rows[0][4]) for frame in frames] == [1.0, 1.4]
    assert all(frame.atom_rows[0][5] == "-0.4" for frame in frames)


def test_align_reference_layer_uses_configured_type_instead_of_fixed_material(tmp_path):
    source = tmp_path / "input.lammpstrj"
    _write_dump(
        source,
        [
            (0, [(1, 7, 0, 0, 1.0, 0), (2, 3, 0, 0, 5.0, 0)]),
            (1, [(1, 7, 0, 0, 2.0, 0), (2, 3, 0, 0, 6.5, 0)]),
        ],
    )

    output = tmp_path / "aligned.lammpstrj"
    align_reference_layer(source, output, reference_atom_type=7, layer_tolerance_A=0.1)
    frames = list(iter_lammps_dump_records(output))

    assert [float(frame.atom_rows[0][4]) for frame in frames] == [1.0, 1.0]
    assert float(frames[1].atom_rows[1][4]) == 5.5
