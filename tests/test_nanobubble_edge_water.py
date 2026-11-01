from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molsimflow.postprocess.nanobubble_edge_water import (
    _dipole_directions,
    surface_site_descriptors,
    water_hbond_edges,
)


def test_water_hbond_edges_tracks_identity() -> None:
    oxygen_ids = np.asarray([10, 20])
    oxygen = np.asarray([[1.0, 1.0, 1.0], [3.8, 1.0, 1.0]])
    vectors = [np.asarray([[1.0, 0.0, 0.0]]), np.asarray([[-1.0, 0.0, 0.0]])]
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    assert water_hbond_edges(
        oxygen_ids, oxygen, vectors, bounds, oo_cutoff_A=3.5, angle_cutoff_deg=30.0
    ) == {(10, 20)}


def test_surface_site_descriptors_validate_current_types(tmp_path: Path) -> None:
    model = tmp_path / "model.data"
    model.write_text(
        "LAMMPS\n\n2 atoms\n8 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Atoms # atomic\n\n1 2 1 1 1\n2 7 2 1 1\n"
    )
    registry = tmp_path / "sites.csv"
    registry.write_text("atom_id,site_type\n1,SiOH\n2,CH3\n")
    rows = surface_site_descriptors(
        model, registry, neighbor_cutoff_A=3.5, local_radius_A=6.0
    )
    assert rows[1]["site_type"] == "SiOH"
    assert rows[2]["site_type"] == "CH3"
    assert rows[1]["local_ch3_fraction"] == 0.5


def test_dipole_directions_preserve_oxygen_identity() -> None:
    directions = _dipole_directions(
        np.asarray([11, 22]),
        [
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        ],
    )
    assert directions[11] == pytest.approx([2**-0.5, 2**-0.5, 0.0])
    assert directions[22] == pytest.approx([-2**-0.5, -2**-0.5, 0.0])
