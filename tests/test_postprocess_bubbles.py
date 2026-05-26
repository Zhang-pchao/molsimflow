import math

import numpy as np

from molsimflow.postprocess.bubble_surface_distance import BubbleSurfaceDistanceAnalyzer
from molsimflow.postprocess.centroids import BubbleCentroidCalculator


def test_centroid_periodic_distance_wraps_box():
    calculator = BubbleCentroidCalculator()

    distance = calculator.periodic_distance(
        np.array([9.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 10.0]),
    )

    assert math.isclose(distance, 2.0)


def test_centroid_clusters_nitrogen_atoms_with_cutoff():
    calculator = BubbleCentroidCalculator(cutoff_distance=2.0)

    clusters = calculator.cluster_nitrogen_atoms(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        ),
        np.array([20.0, 20.0, 20.0]),
    )

    assert [len(cluster) for cluster in clusters] == [2, 1]


def test_surface_distance_uses_minimum_image():
    analyzer = BubbleSurfaceDistanceAnalyzer()

    distance = analyzer._minimum_pair_distance_pbc(
        np.array([[9.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([10.0, 10.0, 10.0]),
    )

    assert math.isclose(distance, 2.0)


def test_surface_fraction_validation():
    try:
        BubbleSurfaceDistanceAnalyzer(surface_fraction=1.2)
    except ValueError as exc:
        assert "surface_fraction" in str(exc)
    else:
        raise AssertionError("surface_fraction > 1 should fail")
