"""True winding distinguishes a periodic loop from a wrapped finite cluster."""

import numpy as np

from molsimflow.postprocess.nanobubble_pbc_topology import winding_components


BOUNDS = np.asarray([[0, 10], [0, 10], [0, 10]], dtype=float)


def test_periodic_ring_has_x_winding():
    centers = np.asarray([[x, 5, 5] for x in (0, 2, 4, 6, 8)], dtype=float)
    main = winding_components(centers, BOUNDS, 2.1)[0]
    assert main["size"] == 5
    assert main["winding_x"] == 1
    assert main["winding_y"] == 0


def test_wrapped_finite_chain_has_no_winding():
    centers = np.asarray([[9, 5, 5], [0.5, 5, 5], [2, 5, 5]], dtype=float)
    main = winding_components(centers, BOUNDS, 1.6)[0]
    assert main["size"] == 3
    assert main["winding_x"] == 0
