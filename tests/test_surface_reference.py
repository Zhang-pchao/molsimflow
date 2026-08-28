import numpy as np

from molsimflow.postprocess.axisymmetric_contact_angle import CoordinateFrame, build_density
from molsimflow.postprocess.surface_reference import SurfaceReference


def test_surface_reference_tracks_periodic_slab_translation():
    reference = SurfaceReference(8.0, 5.0, np.array([10.0, 10.0, 20.0]))
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    translated = np.array([[1.0, 1.0, 11.5], [2.0, 2.0, 12.5]])
    assert np.isclose(reference.plane_z(translated, bounds), 15.0)


def test_translated_slab_preserves_surface_relative_density():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    reference = SurfaceReference(5.0, 2.0, np.array([10.0, 10.0, 20.0]))
    initial = CoordinateFrame(
        0, bounds, np.array([[5.0, 5.0, 8.0]]), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 3.0]])
    )
    translated = CoordinateFrame(
        1, bounds, np.array([[5.0, 5.0, 11.0]]), np.array([[1.0, 1.0, 4.0], [2.0, 2.0, 6.0]])
    )
    kwargs = {
        "surface_z": 5.0,
        "cluster_cutoff": 1.0,
        "r_edges": np.array([0.0, 1.0]),
        "z_edges": np.arange(0.0, 6.0),
        "smoothing_sigma": 0.0,
        "surface_reference": reference,
    }
    density_initial, _ = build_density([initial], **kwargs)
    density_translated, _ = build_density([translated], **kwargs)
    np.testing.assert_array_equal(density_initial, density_translated)
