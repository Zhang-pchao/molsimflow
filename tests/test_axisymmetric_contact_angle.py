import math
import shutil
import subprocess

import numpy as np
import pytest

from molsimflow.postprocess.axisymmetric_contact_angle import (
    fit_axisymmetric_circle,
    iter_selected_frames,
)


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


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd executable is required")
def test_iter_selected_frames_streams_zstd(tmp_path):
    dump = tmp_path / "state.lammpstrj"
    dump.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 0 0 1
2 1 1 0 1
3 2 4 5 6
4 3 4.5 5 6
""",
        encoding="utf-8",
    )
    compressed = tmp_path / "state.lammpstrj.zst"
    subprocess.run(["zstd", "-q", "-f", "-o", str(compressed), str(dump)], check=True)
    frames = list(
        iter_selected_frames(
            compressed,
            (3, 4),
            mode="atom-type",
            atom_type=2,
            surface_range=(1, 2),
        )
    )
    assert len(frames) == 1
    assert frames[0].step == 0
    np.testing.assert_allclose(frames[0].coordinates, [[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(frames[0].surface, [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
