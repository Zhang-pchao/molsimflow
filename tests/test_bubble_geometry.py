from molsimflow.structure.bubble_geometry import equal_volume_radius, two_sphere_intersection_volume


def test_equal_volume_radius_for_reference_pair():
    assert round(equal_volume_radius([19.0, 14.0]), 2) == 16.87


def test_two_sphere_intersection_volume_is_zero_when_separated():
    assert two_sphere_intersection_volume(2.0, 3.0, 5.0) == 0.0
