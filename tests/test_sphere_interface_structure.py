import math

import numpy as np
import pytest

from molsimflow.postprocess.sphere_interface_structure import (
    AnalysisConfig,
    AtomTypeMap,
    DumpFrame,
    ReferenceState,
    SelectionMap,
    _build_reference,
    compute_topology_frame_metrics,
    compute_water_hbond_metrics,
    parse_atom_selection,
)


def _frame(positions):
    return DumpFrame(
        global_index=0,
        segment_label="0.005",
        local_index=0,
        timestep=10000,
        bounds=np.asarray([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]]),
        positions={atom_id: np.asarray(coord, dtype=float) for atom_id, coord in positions.items()},
    )


def test_parse_atom_selection_supports_ranges_and_strides():
    assert parse_atom_selection("1-5:2,9,11-12") == (1, 3, 5, 9, 11, 12)


def test_atom_type_map_requires_distinct_positive_ids():
    AtomTypeMap(framework=1, oxygen=2, carbon=3, hydrogen=4).validate()

    with pytest.raises(ValueError):
        AtomTypeMap(framework=1, oxygen=1, carbon=3, hydrogen=4).validate()


def test_topology_counts_si_c_as_valid_surface_ligand():
    positions = {
        1: [5.0, 5.0, 5.0],
        2: [8.0, 5.0, 5.0],
        3: [6.5, 5.0, 5.0],
        4: [5.0, 6.5, 5.0],
        5: [5.0, 3.5, 5.0],
        6: [8.0, 6.5, 5.0],
        7: [8.0, 3.5, 5.0],
        8: [5.0, 5.0, 6.9],
        9: [8.0, 5.0, 6.9],
    }
    frame = _frame(positions)
    selections = SelectionMap(
        surf_si_top=(1, 2),
        surface_ch3=(8, 9),
        surface_oh=(),
        water_o=(),
        n2_atoms=(),
    )
    config = AnalysisConfig(make_plots=False)
    reference = _build_reference(
        frame,
        selections,
        si_ids=(1, 2),
        o_ids=(3, 4, 5, 6, 7),
        c_ids=(8, 9),
        substrate_h_ids=(),
        config=config,
    )

    metrics, _ = compute_topology_frame_metrics(
        frame,
        selections,
        si_ids=(1, 2),
        o_ids=(3, 4, 5, 6, 7),
        c_ids=(8, 9),
        reference=reference,
        config=config,
    )

    assert metrics["surface_si_o_coord_mean"] == 3.0
    assert metrics["surface_si_c_coord_mean"] == 1.0
    assert metrics["surface_si_ligand_coord4_fraction"] == 1.0
    assert metrics["ligand_edge_survival_fraction"] == 1.0
    assert metrics["bridging_o_coord2_fraction"] == 1.0


def test_water_orientation_and_hbond_network_under_bubble():
    positions = {
        1: [5.0, 5.0, 0.0],
        10: [5.0, 5.0, 8.0],
        11: [5.0, 5.0, 8.9],
        20: [5.0, 5.0, 2.0],
        21: [6.0, 5.0, 2.0],
        22: [5.0, 5.0, 3.0],
        23: [8.0, 5.0, 2.0],
        24: [7.0, 5.0, 2.0],
        25: [8.0, 5.0, 3.0],
    }
    frame = _frame(positions)
    selections = SelectionMap(
        surf_si_top=(1,),
        surface_ch3=(),
        surface_oh=(),
        water_o=(20, 23),
        n2_atoms=(10, 11),
    )
    reference = ReferenceState(
        ligand_edges=set(),
        backbone_o_ids=set(),
        surf_si_positions={1: np.asarray(positions[1], dtype=float)},
        terminal_positions={},
        terminal_si_by_id={},
        surface_oh_h_by_id={},
    )
    metrics, samples = compute_water_hbond_metrics(
        frame,
        selections,
        reference,
        oxygen_candidate_ids=(20, 23),
        hydrogen_candidate_ids=(21, 22, 24, 25),
        z_ref=0.0,
        config=AnalysisConfig(make_plots=False),
    )

    assert metrics["interface_water_count"] == 2.0
    assert metrics["water_under_bubble_count"] == 2.0
    assert metrics["water_h2o_fraction"] == 1.0
    assert metrics["hbond_under_bubble_edge_count"] == 1.0
    assert metrics["hbond_under_bubble_avg_degree"] == 1.0
    assert len(samples["under_bubble"]) == 2
    assert all(math.isfinite(value) for value in samples["under_bubble"])
