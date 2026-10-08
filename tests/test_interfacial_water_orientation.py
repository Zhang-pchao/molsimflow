import numpy as np

from molsimflow.postprocess.interfacial_water_orientation import molecular_orientations


def test_molecular_orientations_rebuilds_two_h_geometry():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    oxygen = np.array([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
    candidate_oxygen = np.vstack((oxygen, [1.0, 1.0, 1.8]))
    hydrogen = np.array([[5.8, 5.0, 5.6], [4.2, 5.0, 5.6], [1.0, 1.0, 1.6]])
    coordination, dipole_cos, oh_cos = molecular_orientations(
        oxygen, candidate_oxygen, hydrogen, bounds, 1.25
    )
    np.testing.assert_array_equal(coordination, [2, 0])
    assert np.isclose(dipole_cos[0], 1.0)
    assert np.all(oh_cos[0] > 0.0)
    assert np.isnan(dipole_cos[1])
