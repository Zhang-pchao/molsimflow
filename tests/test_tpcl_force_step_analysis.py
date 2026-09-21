import math

import numpy as np

from molsimflow.postprocess.tpcl_force_step_analysis import (
    _donates,
    contact_line_metrics,
    select_kinematic_events,
)


def test_contact_line_metrics_are_substrate_fixed():
    top_ids = np.arange(1, 21)
    water_ids = np.arange(101, 401, 3)
    ids = np.concatenate((top_ids, water_ids))
    types = np.concatenate((np.full(len(top_ids), 4), np.full(len(water_ids), 2)))
    top = np.column_stack((np.linspace(0, 9, len(top_ids)), np.zeros(len(top_ids)), np.zeros(len(top_ids))))
    water = np.column_stack((np.linspace(-5, 5, len(water_ids)), np.linspace(-2, 2, len(water_ids)), np.full(len(water_ids), 2.8)))
    coordinates = np.vstack((top, water))
    shifted = coordinates.copy()
    shifted[:, 0] += 7.0
    first = contact_line_metrics(
        ids=ids,
        types=types,
        coordinates=coordinates,
        unwrapped=coordinates,
        top_surface_ids=frozenset(map(int, top_ids)),
        substrate_atoms=100,
        type_symbols={2: "O", 4: "Si"},
        contact_height_A=5.0,
    )
    second = contact_line_metrics(
        ids=ids,
        types=types,
        coordinates=shifted,
        unwrapped=shifted,
        top_surface_ids=frozenset(map(int, top_ids)),
        substrate_atoms=100,
        type_symbols={2: "O", 4: "Si"},
        contact_height_A=5.0,
    )
    assert math.isclose(first["leading_x_A"], second["leading_x_A"], abs_tol=1.0e-12)
    assert math.isclose(first["trailing_x_A"], second["trailing_x_A"], abs_tol=1.0e-12)
    assert first["contact_water_count"] == 100


def test_donor_angle_geometry():
    oh = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert _donates(oh, np.asarray([2.5, 0.0, 0.0]))
    assert not _donates(oh, np.asarray([-2.5, 0.0, 0.0]))


def test_contact_edges_use_collective_periodic_center_not_atom_images():
    top_ids = np.arange(1, 21)
    water_ids = np.arange(101, 401, 3)
    ids = np.concatenate((top_ids, water_ids))
    types = np.concatenate((np.full(len(top_ids), 4), np.full(len(water_ids), 2)))
    top = np.column_stack((np.linspace(0.1, 9.9, len(top_ids)), np.zeros(len(top_ids)), np.zeros(len(top_ids))))
    local_x = np.linspace(-1.0, 1.0, len(water_ids))
    water = np.column_stack(((local_x + 9.6) % 10.0, np.full(len(water_ids), 5.0), np.full(len(water_ids), 2.8)))
    coordinates = np.vstack((top, water))
    arbitrary_images = coordinates.copy()
    arbitrary_images[len(top_ids) : len(top_ids) + 20, 0] += 10.0
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    common = dict(
        ids=ids,
        types=types,
        coordinates=coordinates,
        top_surface_ids=frozenset(map(int, top_ids)),
        substrate_atoms=100,
        type_symbols={2: "O", 4: "Si"},
        contact_height_A=5.0,
        bounds=bounds,
    )
    first = contact_line_metrics(unwrapped=coordinates, **common)
    second = contact_line_metrics(unwrapped=arbitrary_images, **common)
    assert math.isclose(first["leading_x_A"], second["leading_x_A"], abs_tol=1.0e-12)
    assert math.isclose(first["trailing_x_A"], second["trailing_x_A"], abs_tol=1.0e-12)


def test_event_selection_precedes_matched_controls():
    rows = []
    times = np.linspace(0.0, 100.0, 1001)
    response = 0.7 / (1.0 + np.exp(-(times - 30.0) / 0.15))
    response += 0.5 / (1.0 + np.exp(-(times - 70.0) / 0.15))
    rate = np.gradient(response, times)
    for index, (time, value, derivative) in enumerate(zip(times, response, rate)):
        rows.append(
            {
                "case_id": "mixed291",
                "branch_id": "f8e-5_x",
                "direction": "x",
                "step": 36_200_000 + index * 200,
                "time_ps": time,
                "leading_response_rate_A_per_ps": derivative * 1.1,
                "trailing_response_rate_A_per_ps": derivative * 0.9,
                "edge_center_response_rate_A_per_ps": derivative,
                "edge_center_response_smooth_A": value,
                "edge_asymmetry_response_A": 0.02 * math.sin(time),
            }
        )
    events, controls, diagnostics = select_kinematic_events(rows, maximum_events=4)
    assert len(events) == 2
    assert len(controls) == len(events)
    assert diagnostics["event_threshold_A_per_ps"] > 0
    assert {event["selection_basis"] for event in events} == {"kinematics_only"}
    assert all(
        abs(control["control_time_ps"] - event["peak_time_ps"]) >= 1.5
        for event, control in zip(events, controls)
    )
