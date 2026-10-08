import numpy as np

from molsimflow.postprocess.upper_water_admission import (
    assign_hydrogens_to_heavy_atoms,
    contact_components,
    maximum_slab_coordinate_mismatch,
    read_lammps_atomic_data,
    shortest_periodic_arc,
    species_metrics,
)


def test_heavy_atom_assignment_disables_periodicity_along_slab_normal():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    owner_ids, _, _ = assign_hydrogens_to_heavy_atoms(
        np.array([1]),
        np.array([[1.0, 1.0, 0.2]]),
        np.array([], dtype=int),
        np.empty((0, 3)),
        np.array([2]),
        np.array([[1.0, 1.0, 9.8]]),
        bounds,
        oh_cutoff_A=1.0,
        ch_cutoff_A=1.3,
    )

    assert owner_ids.tolist() == [-1]


def test_atomic_data_reader_and_slab_coordinate_mismatch(tmp_path):
    data = tmp_path / "final.data"
    data.write_text(
        """synthetic

2 atoms
2 atom types

0 10 xlo xhi
0 10 ylo yhi
0 20 zlo zhi

Atoms # atomic

2 1 9.9 2.0 3.0 0 0 0
1 2 0.2 2.0 3.0 0 0 0
""",
        encoding="utf-8",
    )

    ids, types, coordinates, bounds = read_lammps_atomic_data(data)

    assert ids.tolist() == [1, 2]
    assert types.tolist() == [2, 1]
    observed = coordinates.copy()
    observed[0, 0] += 10.0
    assert maximum_slab_coordinate_mismatch(coordinates, observed, bounds) < 1.0e-12


def test_contact_components_use_xy_pbc_but_not_z_pbc():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    coordinates = np.array(
        [
            [0.2, 5.0, 5.0],
            [9.6, 5.0, 5.0],
            [0.2, 5.0, 9.8],
            [0.2, 5.0, 0.2],
        ]
    )

    components = contact_components(coordinates, bounds, cutoff_A=1.0)

    assert [component.tolist() for component in components] == [[0, 1], [2], [3]]


def test_shortest_periodic_arc_handles_boundary_crossing():
    assert np.isclose(shortest_periodic_arc(np.array([9.8, 0.1, 0.3]), 0.0, 10.0, 1.0), 0.5)


def test_species_metrics_tracks_substrate_id_hydrogen_owned_by_water_oxygen():
    ids = np.array([1, 2, 3, 4, 5, 6])
    types = np.array([2, 1, 7, 1, 2, 1])
    coordinates = np.array(
        [
            [2.0, 2.0, 2.0],
            [8.0, 8.0, 8.8],
            [4.0, 4.0, 2.0],
            [4.0, 4.0, 3.0],
            [8.0, 8.0, 8.0],
            [8.0, 8.0, 7.1],
        ]
    )
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])

    report, owned = species_metrics(
        ids,
        types,
        coordinates,
        bounds,
        nsub=4,
        hydrogen_type=1,
        oxygen_type=2,
        carbon_type=7,
        oh_cutoff_A=1.3,
        ch_cutoff_A=1.3,
    )

    assert report["water_oxygen_hydrogen_count_distribution"]["2H"] == 1
    assert report["substrate_id_hydrogen_owned_by_water_oxygen_ids"] == [2]
    assert report["unassigned_hydrogen_count"] == 0
    assert owned[5] == (2, 6)
