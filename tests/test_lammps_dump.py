import numpy as np

from molsimflow.io.lammps_dump import (
    box_lengths,
    cylinder_membership,
    iter_lammps_dump_frames,
    iter_lammps_dump_records,
    midpoint_minimum_image,
    periodic_center,
    validate_lammps_dump_bundle,
    write_lammps_dump_frame,
)


def test_iter_lammps_dump_frames_reads_selected_scaled_coordinates(tmp_path):
    dump = tmp_path / "scaled.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 20\n0 10\n-5 5\nITEM: ATOMS id type xs ys zs\n1 1 0.5 0.5 0.5\n2 1 0.25 0.2 0.1"
        + "\n",
        encoding="utf-8",
    )

    frames = list(iter_lammps_dump_frames(dump, needed_atom_ids=[2]))

    assert len(frames) == 1
    assert frames[0].timestep == 10
    assert np.allclose(box_lengths(frames[0].bounds), [20.0, 10.0, 10.0])
    assert np.allclose(frames[0].selected_positions[2], [5.0, 2.0, -4.0])


def test_periodic_midpoint_and_cylinder_membership():
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    coords = np.asarray([[9.5, 5.0, 5.0], [0.5, 5.0, 5.0]])

    center = periodic_center(coords, bounds)
    midpoint = midpoint_minimum_image(coords[0], coords[1], bounds)
    mask, axial, radial = cylinder_membership(
        coords,
        center=np.asarray([0.0, 5.0, 5.0]),
        bounds=bounds,
        axis_index=0,
        radius_A=1.0,
        lower_A=-1.0,
        upper_A=1.0,
    )

    assert np.allclose(center, [0.0, 5.0, 5.0], atol=1e-12) or np.allclose(center, [10.0, 5.0, 5.0])
    assert np.allclose(midpoint, [0.0, 5.0, 5.0], atol=1e-12) or np.allclose(midpoint, [10.0, 5.0, 5.0])
    assert mask.tolist() == [True, True]
    assert np.allclose(np.abs(axial), [0.5, 0.5])
    assert np.allclose(radial, [0.0, 0.0])


def test_full_dump_records_preserve_extra_atom_columns(tmp_path):
    source = tmp_path / "input.lammpstrj"
    source.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z charge\n1 2 1 2 3 -0.4\n",
        encoding="utf-8",
    )
    frame = next(iter_lammps_dump_records(source))
    output = tmp_path / "output.lammpstrj"
    with output.open("w", encoding="utf-8") as handle:
        write_lammps_dump_frame(handle, frame)

    reread = next(iter_lammps_dump_records(output))
    assert reread.atom_fields == ("id", "type", "x", "y", "z", "charge")
    assert reread.atom_rows[0][-1] == "-0.4"


def test_validate_lammps_dump_bundle(tmp_path):
    def write_dump(path, fields, steps, second_type=2):
        rows = {
            ("x", "y", "z"): ("1 1 1 2 3", f"2 {second_type} 4 5 6"),
            ("vx", "vy", "vz"): ("1 1 0.1 0.2 0.3", f"2 {second_type} 0.4 0.5 0.6"),
            ("fx", "fy", "fz"): ("1 1 1.1 1.2 1.3", f"2 {second_type} 1.4 1.5 1.6"),
        }[tuple(fields)]
        path.write_text(
            "".join(
                f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
                "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
                f"ITEM: ATOMS id type {' '.join(fields)}\n{rows[0]}\n{rows[1]}\n"
                for step in steps
            ),
            encoding="utf-8",
        )

    coordinates = tmp_path / "coordinates.dump"
    velocity = tmp_path / "velocity.dump"
    force = tmp_path / "force.dump"
    write_dump(coordinates, ("x", "y", "z"), (2000, 4000, 6000, 8000))
    write_dump(velocity, ("vx", "vy", "vz"), (4000, 8000))
    write_dump(force, ("fx", "fy", "fz"), (4000, 8000))

    result = validate_lammps_dump_bundle(coordinates, velocity, force, 0, 8000, 2000, 4000)
    assert result["status"] == "PASS"
    assert result["coordinate"]["frames"] == 4
    assert result["velocity"]["frames"] == 2

    write_dump(force, ("fx", "fy", "fz"), (4000, 8000), second_type=3)
    with np.testing.assert_raises_regex(ValueError, "atom id/type mismatch"):
        validate_lammps_dump_bundle(coordinates, velocity, force, 0, 8000, 2000, 4000)
