"""Analytic units, axis and block-integrity checks."""

import numpy as np
import pytest
from molsimflow.io.lammps_table import iter_fix_ave_chunk_blocks
from molsimflow.postprocess.planar_interface import mechanical_surface_tension


def test_tensor_units_axes_and_explicit_interface_factor():
    p = np.array([[10, 20, 30], [20, 20, 20]])
    a = mechanical_surface_tension(
        p, 100, normal_axis=2, interfaces=2, pressure_unit="bar", length_unit="angstrom"
    )
    np.testing.assert_allclose(a, [7.5, 0])
    b = mechanical_surface_tension(
        p * 1e5, 1e-8, normal_axis=2, interfaces=2, pressure_unit="Pa", length_unit="m"
    )
    np.testing.assert_allclose(a, b)
    c = mechanical_surface_tension(
        p[:, [2, 0, 1]], 10, normal_axis=0, interfaces=1, pressure_unit="bar", length_unit="nm"
    )
    np.testing.assert_allclose(c, 2 * a)


@pytest.mark.parametrize(
    "changes",
    [{"interfaces": 0}, {"normal_axis": 3}, {"pressure_unit": "guess"}, {"normal_length": -1}],
)
def test_invalid_contract(changes):
    args = dict(
        diagonal_pressure=[1, 2, 3],
        normal_length=10,
        normal_axis=2,
        interfaces=2,
        pressure_unit="bar",
        length_unit="angstrom",
    )
    args.update(changes)
    with pytest.raises(ValueError):
        mechanical_surface_tension(**args)


def test_chunk_blocks_preserve_metadata_and_detect_truncation(tmp_path):
    p = tmp_path / "density.dat"
    header = "# Chunk data\n# Timestep Number-of-rows Total-count\n# Chunk Coord1 Ncount density\n"
    p.write_text(header + "100 2 4\n1 .5 2 .1\n2 1.5 2 .1\n200 1 3\n1 .5 3 .2\n")
    blocks = list(iter_fix_ave_chunk_blocks(p))
    assert len(blocks) == 2 and blocks[0].metadata["Total-count"] == 4
    assert blocks[0].columns == ("Chunk", "Coord1", "Ncount", "density")
    np.testing.assert_array_equal(blocks[0].data[:, 2], [2, 2])
    p.write_text(header + "100 2 4\n1 .5 2 .1\n")
    with pytest.raises(ValueError, match="Truncated"):
        list(iter_fix_ave_chunk_blocks(p))
