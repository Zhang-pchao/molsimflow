import numpy as np

from molsimflow.postprocess.planar_motion import time_origin_msd, unwrap_planar


def test_unwrap_planar_and_msd():
    coords = unwrap_planar(np.array([9.0, 0.5, 2.0]), np.zeros(3), 10.0, 10.0)
    assert np.allclose(coords[:, 0], [9.0, 10.5, 12.0])
    assert np.allclose(time_origin_msd(coords), [0.0, 2.25, 9.0])
