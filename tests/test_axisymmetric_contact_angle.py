import math

import numpy as np

from molsimflow.postprocess.axisymmetric_contact_angle import fit_axisymmetric_circle


def test_axisymmetric_circle_recovers_contact_angle():
    center_z = -5.0
    radius = 10.0
    z = np.linspace(0.0, 4.5, 12)
    radial = np.sqrt(radius**2 - (z - center_z) ** 2)
    result = fit_axisymmetric_circle(np.column_stack((radial, z)))
    assert math.isclose(result["center_z_A"], center_z, abs_tol=1.0e-10)
    assert math.isclose(result["circle_radius_A"], radius, abs_tol=1.0e-10)
    assert math.isclose(result["dense_phase_contact_angle_deg"], 60.0, abs_tol=1.0e-10)
    assert result["fit_rmse_A"] < 1.0e-10
