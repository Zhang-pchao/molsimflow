import numpy as np

from molsimflow.postprocess.ion_species import find_nearest_oxygen_hydrogens
from molsimflow.postprocess.species_assignment import (
    assign_hydrogen_to_nearest_oxygen,
    classify_oxygen_species_indices,
    count_oxygen_species,
)


def test_assign_hydrogen_to_nearest_oxygen_periodic():
    oxygen = np.asarray([[0.2, 5.0, 5.0], [5.0, 5.0, 5.0]])
    hydrogen = np.asarray([[9.8, 5.0, 5.0], [5.9, 5.0, 5.0], [5.0, 5.9, 5.0]])
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])

    assignment = assign_hydrogen_to_nearest_oxygen(oxygen, hydrogen, bounds, oh_cutoff=1.1)

    assert assignment.h_count_per_oxygen.tolist() == [1, 2]
    assert assignment.hydrogen_to_oxygen_index.tolist() == [0, 1, 1]
    assert np.allclose(assignment.hydrogen_distance, [0.4, 0.9, 0.9])


def test_classify_oxygen_species_indices_and_counts():
    species = classify_oxygen_species_indices(np.asarray([1, 2, 3, 4, 0]))
    counts = count_oxygen_species(np.asarray([1, 2, 3, 4, 0]))

    assert species["oh"].tolist() == [0]
    assert species["h2o"].tolist() == [1]
    assert species["h3o"].tolist() == [2]
    assert species["other"].tolist() == [3]
    assert counts == {"oh": 1, "h2o": 1, "h3o": 1, "other": 1}


def test_ion_species_nearest_oxygen_adapter_uses_shared_assignment():
    h_positions = np.asarray([[9.8, 5.0, 5.0], [5.9, 5.0, 5.0]])
    o_positions = np.asarray([[0.2, 5.0, 5.0], [5.0, 5.0, 5.0]])

    h_counts, bonds = find_nearest_oxygen_hydrogens(h_positions, o_positions, [10.0, 10.0, 10.0], 1.1)

    assert h_counts == {0: 1, 1: 1}
    assert bonds == {0: [0], 1: [1]}
