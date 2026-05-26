import csv

from molsimflow.postprocess.transitions import (
    analyze_species_transitions,
    build_species_transition_matrix,
    load_species_state_rows,
    parse_species_order,
    transition_matrix_to_long_rows,
)


def _write_species_states(path):
    rows = [
        {"frame": 0, "time_ns": 0.0, "oxygen_index": 1, "species": "solution_bulk_oh"},
        {"frame": 0, "time_ns": 0.0, "oxygen_index": 2, "species": "solution_surface_oh"},
        {"frame": 1, "time_ns": 0.1, "oxygen_index": 1, "species": "solution_surface_oh"},
        {"frame": 1, "time_ns": 0.1, "oxygen_index": 2, "species": "solution_surface_h2o"},
        {"frame": 2, "time_ns": 0.2, "oxygen_index": 1, "species": "solution_surface_oh"},
        {"frame": 2, "time_ns": 0.2, "oxygen_index": 2, "species": "solution_surface_h2o"},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_species_transition_matrix_counts(tmp_path):
    state_path = tmp_path / "species_states.csv"
    _write_species_states(state_path)

    rows = load_species_state_rows(
        state_path,
        entity_column="oxygen_index",
        time_column="time_ns",
    )
    order = parse_species_order(["solution_bulk_oh,solution_surface_oh,solution_surface_h2o"])
    result = build_species_transition_matrix(rows, species_order=order)
    count_rows = {
        (row["from_species"], row["to_species"]): row["count"]
        for row in transition_matrix_to_long_rows(result.counts, result.species_order, "count")
    }

    assert result.species_order == order
    assert result.n_frames == 3
    assert result.n_entities == 2
    assert len(result.details) == 4
    assert count_rows[("solution_bulk_oh", "solution_surface_oh")] == 1
    assert count_rows[("solution_surface_oh", "solution_surface_oh")] == 1
    assert count_rows[("solution_surface_oh", "solution_surface_h2o")] == 1
    assert count_rows[("solution_surface_h2o", "solution_surface_h2o")] == 1

    surface_oh_index = result.species_order.index("solution_surface_oh")
    surface_h2o_index = result.species_order.index("solution_surface_h2o")
    assert result.probabilities[surface_oh_index, surface_h2o_index] == 0.5


def test_analyze_species_transitions_writes_outputs(tmp_path):
    state_path = tmp_path / "species_states.csv"
    _write_species_states(state_path)

    outputs = analyze_species_transitions(
        input_csv=state_path,
        output_dir=tmp_path / "species_transitions",
        entity_column="oxygen_index",
        time_column="time_ns",
        species_order=parse_species_order(["solution_bulk_oh;solution_surface_oh;solution_surface_h2o"]),
    )

    assert outputs["transition_counts"].exists()
    assert outputs["transition_probabilities"].exists()
    assert outputs["transition_details"].exists()
    assert outputs["species_state_summary"].exists()
    assert outputs["state_statistics"].exists()

    with outputs["transition_counts"].open() as handle:
        counts = list(csv.DictReader(handle))
    count_rows = {(row["from_species"], row["to_species"]): int(row["count"]) for row in counts}

    assert count_rows[("solution_bulk_oh", "solution_surface_oh")] == 1
    assert count_rows[("solution_surface_h2o", "solution_surface_h2o")] == 1

    with outputs["state_statistics"].open() as handle:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert stats["n_changed_transitions"] == "2"
