import numpy as np
from molsimflow.postprocess.gas_solid_distribution import gas_solid_distances, time_mean_cdf


def test_ppf_does_not_connect_or_measure_through_z():
    box = np.array([[0., 20.], [0., 20.], [0., 20.]])
    gas = np.array([[1., 1., 1.], [1., 1., 2.], [1., 1., 18.], [1., 1., 19.]])
    solid = np.array([[1., 1., 0.]])
    d, m = gas_solid_distances(gas, solid, box, periodic_z=False)
    np.testing.assert_allclose(d, [1., 18.]); assert m.sum() == 1
    dp, mp = gas_solid_distances(gas, solid, box)
    np.testing.assert_allclose(dp, [1., 1.]); assert mp.sum() == 2
    shifted = gas + [20., -20., 0.]
    ds, ms = gas_solid_distances(shifted, solid, box, periodic_z=False)
    np.testing.assert_allclose(ds, d); np.testing.assert_array_equal(ms, m)


def test_equal_frame_not_pooled_molecule_weight():
    d = np.array([[1., 10.], [1., 10.]])
    m = np.array([[True, False], [True, True]])
    np.testing.assert_allclose(time_mean_cdf(d, m, [0., 1., 10.]), [0., .75, 1.])
