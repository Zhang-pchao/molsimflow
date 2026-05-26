import csv

from molsimflow.postprocess.bridge_film import (
    BridgeFilmConfig,
    analyze_bridge_film,
    build_bridge_film_frame_table,
    build_residence_events,
    classify_bridge_film_state,
    summarize_coordination_samples,
    summarize_residence_events,
)


def _write_frame_table(path):
    rows = [
        {
            "frame": 0,
            "time": 0.000,
            "d3d_all": 8.0,
            "N_oxygen_bridge_total": 1,
            "N_water_bridge": 1,
            "N_OH_bridge": 0,
            "N_H3O_bridge": 0,
            "N_Na_bridge": 0,
            "N_Cl_bridge": 0,
            "N_tio2_hydration_bridge": 0,
            "bridge_charge_density_e_per_A3": 0.0,
        },
        {
            "frame": 1,
            "time": 0.001,
            "d3d_all": 6.0,
            "N_oxygen_bridge_total": 5,
            "N_water_bridge": 4,
            "N_OH_bridge": 0,
            "N_H3O_bridge": 0,
            "N_Na_bridge": 0,
            "N_Cl_bridge": 0,
            "N_tio2_hydration_bridge": 0,
            "bridge_charge_density_e_per_A3": 0.0,
        },
        {
            "frame": 2,
            "time": 0.002,
            "d3d_all": 5.0,
            "N_oxygen_bridge_total": 6,
            "N_water_bridge": 3,
            "N_OH_bridge": 2,
            "N_H3O_bridge": 0,
            "N_Na_bridge": 0,
            "N_Cl_bridge": 0,
            "N_tio2_hydration_bridge": 0,
            "bridge_charge_density_e_per_A3": -0.1,
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_events(path):
    rows = [{"event_id": 1, "event_frame": 1, "event_time": 0.001, "event_type": "connectivity_loss"}]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_membership(path):
    rows = [
        {"species": "Na", "atom_id": 10, "frame": 0, "time": 0.000, "in_bridge": 0},
        {"species": "Na", "atom_id": 10, "frame": 1, "time": 0.001, "in_bridge": 1},
        {"species": "Na", "atom_id": 10, "frame": 2, "time": 0.002, "in_bridge": 1},
        {"species": "Na", "atom_id": 10, "frame": 3, "time": 0.003, "in_bridge": 0},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_coordination(path):
    rows = [
        {"species": "Na", "coordination": 4},
        {"species": "Na", "coordination": 5},
        {"species": "Cl", "coordination": 3},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_classify_bridge_film_state():
    config = BridgeFilmConfig()

    assert classify_bridge_film_state({"N_oxygen_bridge_total": 1}, config=config) == "dry_or_vapor"
    assert classify_bridge_film_state({"N_oxygen_bridge_total": 5, "N_water_bridge": 4}, config=config) == "water_film"
    assert (
        classify_bridge_film_state({"N_oxygen_bridge_total": 5, "N_water_bridge": 3, "N_OH_bridge": 2}, config=config)
        == "basic_film"
    )


def test_bridge_film_frame_table_and_residence(tmp_path):
    frame_table = tmp_path / "frames.csv"
    events = tmp_path / "events.csv"
    membership = tmp_path / "membership.csv"
    _write_frame_table(frame_table)
    _write_events(events)
    _write_membership(membership)

    with frame_table.open() as handle:
        frame_rows = list(csv.DictReader(handle))
    with events.open() as handle:
        event_rows = list(csv.DictReader(handle))
    enriched, state = build_bridge_film_frame_table(frame_rows, events=event_rows)

    assert state["barrier_mode"] == "transition_events_window"
    assert [row["bridge_film_state"] for row in enriched] == ["dry_or_vapor", "water_film", "basic_film"]
    assert [row["barrier_top_flag"] for row in enriched] == [True, True, True]

    with membership.open() as handle:
        residence_events = build_residence_events(list(csv.DictReader(handle)))
    residence_summary = summarize_residence_events(residence_events)

    assert residence_events[0]["start_frame"] == 1
    assert residence_events[0]["end_frame"] == 2
    assert residence_summary[0]["n_events"] == 1


def test_analyze_bridge_film_writes_outputs(tmp_path):
    frame_table = tmp_path / "frames.csv"
    events = tmp_path / "events.csv"
    membership = tmp_path / "membership.csv"
    coordination = tmp_path / "coordination.csv"
    _write_frame_table(frame_table)
    _write_events(events)
    _write_membership(membership)
    _write_coordination(coordination)

    outputs = analyze_bridge_film(
        frame_table=frame_table,
        transition_events=events,
        residence_membership=membership,
        coordination_samples=coordination,
        output_dir=tmp_path / "out",
    )

    assert outputs["frame_table"].exists()
    assert outputs["state_summary"].exists()
    assert outputs["residence_events"].exists()
    assert outputs["coordination_summary"].exists()

    with coordination.open() as handle:
        summary = summarize_coordination_samples(list(csv.DictReader(handle)))
    assert {row["species"] for row in summary} == {"Cl", "Na"}
