import csv

import numpy as np

from molsimflow.postprocess.bridge_water_escape import (
    BridgeWaterEscapeConfig,
    analyze_bridge_water_escape,
    classify_escape_direction,
    load_seed_position_rows,
)


def _write_seed_positions(path):
    rows = [
        {"frame": 0, "time": 0.0, "atom_id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "in_bridge": 1, "surface_gap_A": 2.0},
        {"frame": 1, "time": 0.1, "atom_id": 1, "x": 0.0, "y": 0.0, "z": 3.0, "in_bridge": 0, "surface_gap_A": 4.0},
        {"frame": 2, "time": 0.2, "atom_id": 1, "x": 0.0, "y": 0.0, "z": 5.0, "in_bridge": 0, "surface_gap_A": 6.0},
        {"frame": 0, "time": 0.0, "atom_id": 2, "x": 0.0, "y": 0.0, "z": 0.0, "in_bridge": 1, "surface_gap_A": 2.0},
        {"frame": 1, "time": 0.1, "atom_id": 2, "x": 1.0, "y": 0.0, "z": 0.0, "in_bridge": 1, "surface_gap_A": 4.0},
        {"frame": 2, "time": 0.2, "atom_id": 2, "x": 2.0, "y": 0.0, "z": 0.0, "in_bridge": 1, "surface_gap_A": 6.0},
        {"frame": 0, "time": 0.0, "atom_id": 3, "x": 0.0, "y": 0.0, "z": 0.0, "in_bridge": 1, "surface_gap_A": 2.0},
        {"frame": 1, "time": 0.1, "atom_id": 3, "x": 4.0, "y": 0.0, "z": 0.5, "in_bridge": 0, "surface_gap_A": 4.0},
        {"frame": 2, "time": 0.2, "atom_id": 3, "x": 6.0, "y": 0.0, "z": 0.5, "in_bridge": 0, "surface_gap_A": 6.0},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_classify_escape_direction_thresholds():
    assert classify_escape_direction(np.array([0.0, 0.0, 5.0]), "exited") == "toward_bulk_or_zplus"
    assert classify_escape_direction(np.array([0.0, 0.0, -5.0]), "exited") == "toward_TiO2_or_zminus"
    assert classify_escape_direction(np.array([5.0, 0.0, 0.5]), "exited") == "lateral_xy"
    assert classify_escape_direction(np.array([0.1, 0.0, 0.1]), "exited") == "unresolved"
    assert classify_escape_direction(np.array([0.0, 0.0, 5.0]), "retained") == "retained"


def test_analyze_bridge_water_escape_writes_outputs(tmp_path):
    input_path = tmp_path / "seed_positions.csv"
    _write_seed_positions(input_path)

    rows = load_seed_position_rows(input_path, gap_column="surface_gap_A")
    assert len(rows) == 9

    outputs = analyze_bridge_water_escape(
        input_csv=input_path,
        output_dir=tmp_path / "escape",
        case_label="caseA",
        config=BridgeWaterEscapeConfig(exit_confirm_frames=2, destination_lag_frames=1),
        gap_column="surface_gap_A",
    )

    assert outputs["events"].exists()
    assert outputs["direction_summary"].exists()
    assert outputs["gap_summary"].exists()
    assert outputs["state_statistics"].exists()

    with outputs["events"].open() as handle:
        events = list(csv.DictReader(handle))
    by_atom = {row["atom_id"]: row for row in events}

    assert by_atom["1"]["status"] == "exited"
    assert by_atom["1"]["escape_direction"] == "toward_bulk_or_zplus"
    assert by_atom["2"]["status"] == "retained"
    assert by_atom["2"]["escape_direction"] == "retained"
    assert by_atom["3"]["escape_direction"] == "lateral_xy"

    with outputs["direction_summary"].open() as handle:
        summary = list(csv.DictReader(handle))
    counts = {(row["status"], row["escape_direction"]): int(row["count"]) for row in summary}
    assert counts[("exited", "toward_bulk_or_zplus")] == 1
    assert counts[("exited", "lateral_xy")] == 1
    assert counts[("retained", "retained")] == 1

    with outputs["state_statistics"].open() as handle:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert stats["n_seed_atoms"] == "3"
    assert stats["n_exited"] == "2"
