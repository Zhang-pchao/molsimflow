import numpy as np

from molsimflow.postprocess.particle_flotation import (
    AtomRange,
    angular_coverage,
    compute_n2_molecule_coms,
    minimum_image_delta,
    occupied_voxel_volume,
    parse_atom_range,
)


def test_particle_flotation_geometry_helpers_handle_pbc():
    lengths = np.array([10.0, 10.0, 20.0])
    delta = minimum_image_delta(np.array([[9.0, -9.0, 3.0]]), lengths)
    np.testing.assert_allclose(delta, [[-1.0, 1.0, 3.0]])

    box = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    n2 = np.array([[9.5, 5.0, 5.0], [0.5, 5.0, 5.0]])
    np.testing.assert_allclose(compute_n2_molecule_coms(n2, box), [[10.0, 5.0, 5.0]])


def test_particle_flotation_ranges_and_density_helpers():
    atom_range = parse_atom_range("3:7")
    assert atom_range == AtomRange(3, 7)
    assert atom_range.count == 5
    assert atom_range.contains(5)
    assert not atom_range.contains(8)

    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert angular_coverage(vectors, z_bins=2, phi_bins=4) > 0.0
    assert occupied_voxel_volume(vectors, voxel_A=1.0, probe_radius_A=0.0) > 0.0
