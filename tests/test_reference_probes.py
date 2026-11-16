"""Identity, mixed-boundary and probe endpoint regressions."""

import numpy as np
import pytest
from molsimflow.postprocess.reference_registration import anchor_translation
from molsimflow.postprocess.spatial_probes import cuboid_counts


def test_registration_matches_ids_under_pbc_and_leaves_z_open():
    shift, residual = anchor_translation(
        [4, 7],
        [[9, 1, 1], [8, 2, 2]],
        [7, 4],
        [[0, 3, 7], [1, 2, 6]],
        lengths=[10, 10, 10],
        periodic=[True, True, False],
    )
    np.testing.assert_array_equal(shift, [2, 1, 5])
    np.testing.assert_array_equal(residual, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="missing"):
        anchor_translation(
            [1], [[0, 0, 0]], [2], [[0, 0, 0]], lengths=[10] * 3, periodic=[True] * 3
        )
    with pytest.raises(ValueError, match="half"):
        anchor_translation(
            [1], [[0, 0, 0]], [1], [[5, 0, 0]], lengths=[10] * 3, periodic=[True] * 3
        )


def test_probe_images_open_z_and_half_open_endpoints():
    points = [[9.5, 0, 0], [0.5, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 10]]
    assert cuboid_counts(
        points, [[0, 0, 0]], [2, 2, 2], lengths=[10] * 3, periodic=[True, True, False]
    ).tolist() == [3]
    assert cuboid_counts(
        points,
        [[0, 0, 0]],
        [2, 2, 2],
        lengths=[10] * 3,
        periodic=[True, True, False],
        lower_closed=[False] * 3,
    ).tolist() == [2]
    assert cuboid_counts(
        np.empty((0, 3)), [[0, 0, 0]], [1] * 3, lengths=[10] * 3, periodic=[False] * 3
    ).tolist() == [0]
