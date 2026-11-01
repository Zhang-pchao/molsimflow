from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.nanobubble_edge_events import (
    REQUIRED_FIELDS,
    detect_events,
    rolling_median,
    run_contract,
)


def _rows(case: str, branch: str, direction: str, front: np.ndarray) -> list[dict[str, str]]:
    rows = []
    for index, value in enumerate(front):
        row = {field: "0" for field in REQUIRED_FIELDS}
        row.update({
            "case_id": case, "branch_id": branch, "direction": direction,
            "step": str(36_000_000 + 2_000 * index), "time_ps": str(float(index)),
            "front_edge_unwrapped_A": str(float(value)),
            "rear_edge_unwrapped_A": "-10.0", "bubble_drive_unwrapped_A": "0.0",
            "bubble_height_A": "20.0", "bubble_drive_width_A": "10.0",
            "bubble_transverse_width_A": "9.0", "bubble_drive_span_fraction": "0.2",
            "bubble_transverse_span_fraction": "0.2", "pbc_spanning_drive": "0",
            "pbc_spanning_transverse": "0", "film_water_oxygen_count": "20",
            "front_water_oxygen_count": "12", "front_oxygen_density_A3": "0.03",
            "front_water_drive_velocity_mps": "2.0",
            "front_flow_corrected_temperature_K": "330.0",
            "rear_water_oxygen_count": "12", "rear_oxygen_density_A3": "0.03",
            "rear_water_drive_velocity_mps": "2.0",
            "rear_flow_corrected_temperature_K": "330.0",
        })
        rows.append(row)
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_FIELDS), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def test_rolling_median_and_persistent_event_detection() -> None:
    values = np.r_[np.zeros(100), np.full(101, 3.0)]
    rows = _rows("case", "driven", "x", values)
    events = detect_events(
        rows, "front", 1.0, cadence_ps=1.0, smoothing_ps=5.0,
        pre_window_ps=10.0, post_window_ps=10.0, persistence_ps=20.0,
        persistence_fraction=0.8, refractory_ps=40.0, drive_box_length_A=75.0,
    )
    assert len(events) == 1
    assert events[0]["process"] == "gas_advance_liquid_retreat"
    assert events[0]["validity_status"] == "VALID"
    assert float(events[0]["displacement_A"]) == 3.0
    assert rolling_median([0, 100, 0], 3).tolist() == [50.0, 0.0, 50.0]


def test_contract_calibrates_from_f0_and_writes_outputs(tmp_path: Path) -> None:
    baseline = _rows("case", "f0", "x", 0.1 * np.sin(np.arange(201)))
    driven = _rows("case", "driven", "x", np.r_[np.zeros(100), np.full(101, 3.0)])
    baseline_path, driven_path = tmp_path / "baseline.tsv", tmp_path / "driven.tsv"
    _write(baseline_path, baseline); _write(driven_path, driven)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "smoothing_ps": 5.0, "pre_window_ps": 10.0,
        "post_window_ps": 10.0, "persistence_ps": 20.0,
        "persistence_fraction": 0.8, "refractory_ps": 40.0,
        "minimum_displacement_A": 1.0, "f0_quantile": 0.995,
        "sensitivity_quantiles": [0.99, 0.995, 0.999], "block_ps": 50.0,
        "write_plots": False,
        "cases": [{
            "driven_frames": str(driven_path), "baseline_frames": str(baseline_path),
            "drive_box_length_A": 75.0,
        }],
    }))
    output = tmp_path / "output"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert summary["valid_events"] == 1
    assert (output / "edge_events.tsv").is_file()
    assert (output / "event_sensitivity.tsv").is_file()
    assert (output / "OUTPUT-SHA256SUMS").is_file()


def test_contract_plot_accepts_non_surface_case_id(tmp_path: Path) -> None:
    baseline = _rows("hcl_63pairs", "f0", "x", 0.1 * np.sin(np.arange(201)))
    driven = _rows(
        "hcl_63pairs", "driven", "x", np.r_[np.zeros(100), np.full(101, 3.0)]
    )
    baseline_path, driven_path = tmp_path / "baseline.tsv", tmp_path / "driven.tsv"
    _write(baseline_path, baseline); _write(driven_path, driven)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": 1, "smoothing_ps": 5.0, "pre_window_ps": 10.0,
        "post_window_ps": 10.0, "persistence_ps": 20.0,
        "persistence_fraction": 0.8, "refractory_ps": 40.0,
        "minimum_displacement_A": 1.0, "f0_quantile": 0.995,
        "sensitivity_quantiles": [0.99, 0.995, 0.999], "block_ps": 50.0,
        "write_plots": True,
        "cases": [{
            "driven_frames": str(driven_path), "baseline_frames": str(baseline_path),
            "drive_box_length_A": 75.0,
        }],
    }))
    output = tmp_path / "output"
    summary = run_contract(contract, output)
    assert summary["status"] == "PASS"
    assert (output / "edge_event_overview.png").is_file()


def test_cli_registers_edge_analysis_commands(tmp_path: Path) -> None:
    parser = build_parser()
    for command in (
        "nanobubble-edge-events", "nanobubble-edge-water", "nanobubble-edge-mechanism"
    ):
        args = parser.parse_args([
            "postprocess", command, "--contract", str(tmp_path / "contract.json"),
            "--output", str(tmp_path / "output"),
        ])
        assert args.contract == tmp_path / "contract.json"
