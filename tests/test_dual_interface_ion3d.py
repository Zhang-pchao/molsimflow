import numpy as np

from molsimflow.postprocess.dual_interface_ion3d import _bootstrap


def test_block_bootstrap_uses_time_blocks():
    values = np.arange(8.0)
    times = np.arange(8.0) * 0.01
    mean, low, high, blocks = _bootstrap(values, times, 0.02, 200, np.random.default_rng(7))
    assert mean == 3.5
    assert blocks == 4
    assert low <= mean <= high
