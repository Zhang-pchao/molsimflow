"""Projection, explicit neighbor scope and rendered primitive checks."""

import numpy as np
import pytest
from molsimflow.plotting.structures import (
    project_coordinates,
    find_bond_pairs,
    draw_atoms,
    draw_bonds,
)


def test_projection_and_explicit_candidate_scope():
    xy, depth = project_coordinates([[1, 2, 3]], [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(xy, [[1, 2]])
    np.testing.assert_array_equal(depth, [3])
    xyz = [[0, 0, 0], [2.1, 0, 0]]
    assert find_bond_pairs(xyz, ["Si", "C"], {("Si", "C"): 2.2}) == [(0, 1)]
    assert find_bond_pairs(xyz, ["Si", "C"], {("Si", "C"): 2.2}, candidate_radius=2.05) == []
    assert find_bond_pairs(xyz, ["C", "Si"], {("Si", "C"): 2.2}) == [(0, 1)]


def test_draw_primitives_and_styles(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set(xlim=(-1, 2), ylim=(-1, 1), aspect="equal")
    xyz = [[0, 0], [1, 0]]
    atom = draw_atoms(
        ax,
        xyz,
        ["O", "H"],
        radii={"O": 0.3, "H": 0.15},
        colors={"O": "red", "H": "white"},
        depth=[1, 0],
    )
    bond = draw_bonds(ax, xyz, [[0, 1]], color="gray")
    assert len(atom.get_offsets()) == 2 and len(bond.get_segments()) == 1
    path = tmp_path / "structure.png"
    fig.savefig(path)
    assert path.stat().st_size > 100
    with pytest.raises(ValueError, match="Missing"):
        draw_atoms(ax, xyz, ["O", "H"], radii={"O": 0.3}, colors={"O": "red"})
    plt.close(fig)
