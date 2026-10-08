import math
import unittest

import numpy as np

from molsimflow.postprocess.coupled_expulsion import (
    classify_mobile_partition,
    confined_film_volume_A3,
    in_confined_film,
)


class CoupledExpulsionGeometryTest(unittest.TestCase):
    def test_confined_film_rejects_axial_outer_regions(self):
        gap = 4.0
        points = in_confined_film(
            np.array([0.0, 50.0]),
            np.array([0.0, 0.0]),
            np.array([gap, gap]),
            19.0,
            19.0,
            6.0,
        )
        self.assertEqual(points.tolist(), [True, False])
        self.assertTrue(math.isclose(float(confined_film_volume_A3(gap, 19.0, 19.0, 6.0)), 561.385, rel_tol=1e-3))

    def test_mobile_partition_is_mutually_exclusive(self):
        self.assertEqual(classify_mobile_partition(0.0, 4.0, 2.0, 2.0, True, 19.0, 19.0, 4.0), "film")
        self.assertEqual(classify_mobile_partition(-18.0, 4.0, 2.0, 30.0, False, 19.0, 19.0, 4.0), "A_facing_shell")
        self.assertEqual(classify_mobile_partition(18.0, 4.0, 30.0, 2.0, False, 19.0, 19.0, 4.0), "B_facing_shell")
        self.assertEqual(classify_mobile_partition(40.0, 4.0, 10.0, 2.0, False, 19.0, 19.0, 4.0), "other_liquid")


if __name__ == "__main__":
    unittest.main()
