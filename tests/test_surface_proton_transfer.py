import numpy as np

from molsimflow.postprocess.surface_proton_transfer import (
    assign_hydrogens_to_oxygen_or_carbon,
    classify_ion_pair_state,
    classify_site_state,
    extract_episodes,
    hydrogen_ids_by_owner,
)


def test_heavy_atom_assignment_uses_pbc_and_element_cutoffs():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    assignment = assign_hydrogens_to_oxygen_or_carbon(
        np.array([10]),
        np.array([[0.2, 5.0, 5.0]]),
        np.array([20]),
        np.array([[5.0, 5.0, 5.0]]),
        np.array([101, 102, 103]),
        np.array([[9.3, 5.0, 5.0], [6.1, 5.0, 5.0], [8.0, 8.0, 8.0]]),
        bounds,
        oh_cutoff_A=1.25,
        ch_cutoff_A=1.30,
    )
    assert assignment.owner_ids.tolist() == [10, 20, -1]
    assert hydrogen_ids_by_owner(assignment) == {10: (101,), 20: (102,)}


def test_site_state_distinguishes_loss_and_identity_exchange():
    assert classify_site_state("SiOH", 1, [11], []) == "deprotonated_candidate"
    assert classify_site_state("SiOH", 1, [11], [12]) == "proton_exchanged_candidate"
    assert classify_site_state("CH3", 3, [1, 2, 3], [1, 2]) == "c_h_loss_candidate"
    assert classify_site_state("CH3", 3, [1, 2, 3], [1, 2, 3]) == "nominal"


def test_ion_pair_state_retains_pair_when_an_extra_defect_is_present():
    assert classify_ion_pair_state(0, 0) == "neutral"
    assert classify_ion_pair_state(1, 2) == "paired_candidate"
    assert classify_ion_pair_state(0, 1) == "unbalanced_candidate"


def test_episode_persistence_requires_consecutive_sampled_frames():
    rows = [
        {
            "frame_index": frame,
            "step": frame * 20,
            "time_ns": frame * 0.01,
            "site_atom_id": 7,
            "state": "deprotonated_candidate",
            "region": "tpcl",
        }
        for frame in (1, 2, 4)
    ]
    episodes, persistent = extract_episodes(
        rows, key_columns=("site_atom_id", "state"), min_persistence_frames=2
    )
    assert [row["sample_count"] for row in episodes] == [2, 1]
    assert len(persistent) == 1
    assert persistent[0]["observed_span_ps"] == 10.0


def test_solution_ion_episodes_keep_oxygen_identity():
    rows = [
        {
            "frame_index": frame,
            "step": frame * 20,
            "time_ns": frame * 0.01,
            "oxygen_atom_id": oxygen_id,
            "species_candidate": "H3O-like",
            "region": "footprint",
        }
        for frame, oxygen_id in ((1, 101), (2, 101), (3, 102))
    ]
    episodes, persistent = extract_episodes(
        rows,
        key_columns=("oxygen_atom_id", "species_candidate"),
        min_persistence_frames=2,
    )
    assert [row["sample_count"] for row in episodes] == [2, 1]
    assert [row["oxygen_atom_id"] for row in persistent] == [101]
