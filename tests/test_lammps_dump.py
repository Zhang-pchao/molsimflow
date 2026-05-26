import numpy as np

from molsimflow.io.lammps_dump import (
    box_lengths,
    cylinder_membership,
    iter_lammps_dump_frames,
    midpoint_minimum_image,
    periodic_center,
)


def test_iter_lammps_dump_frames_reads_selected_scaled_coordinates(tmp_path):
    dump = tmp_path / "scaled.lammpstrj"
    dump.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "10",
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp pp pp",
                "0 20",
                "0 10",
                "-5 5",
                "ITEM: ATOMS id type xs ys zs",
                "1 1 0.5 0.5 0.5",
                "2 1 0.25 0.2 0.1",
            ]
        )
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
