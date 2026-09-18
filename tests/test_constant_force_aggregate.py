from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from molsimflow.cli import build_parser as build_cli_parser
from molsimflow.postprocess.constant_force_aggregate import run_contract

CASES = ("droplet", "film")
BRANCHES = (("f0", "none"), ("fx", "x"), ("fy", "y"))


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _seal(results: Path) -> None:
    records = []
    for path in sorted(candidate for candidate in results.iterdir() if candidate.is_file()):
        if path.name == "OUTPUT-SHA256SUMS":
            continue
        records.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
    (results / "OUTPUT-SHA256SUMS").write_text("".join(records), encoding="utf-8")


def _water_results(root: Path, case_id: str) -> Path:
    results = root / f"water_{case_id}"
    results.mkdir()
    frames = []
    for branch_id, direction in BRANCHES:
        for step, time_ps in ((0, 0.0), (1, 1.0)):
            frames.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "step": step,
                    "time_ps": time_ps,
                    "region": "all",
                    "water_count": 10,
                    "water_fraction": 1.0,
                    "physical_largest_component_fraction": 0.99,
                    "mean_q_tet": "nan" if step == 0 else 0.5,
                    "mean_lsi_A2": 0.08,
                    "mean_oo_coordination": 4.5,
                    "h_coordination_defect_fraction": 0.0,
                    "water_water_hbond_edges": 15,
                    "water_surface_hbond_edges": 2 if case_id == "droplet" else 5,
                    "hbond_largest_component_fraction": 0.98,
                    "water_water_edge_turnover": 0.2,
                    "water_surface_edge_turnover": 0.3,
                }
            )
    _write_tsv(results / "water_structure_by_frame.tsv", frames)
    _write_tsv(
        results / "region_residence_summary.tsv",
        [
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "region": "all",
                "episodes": 10,
                "mean_residence_ps": 8.0,
                "median_residence_ps": 7.0,
                "p95_residence_ps": 10.0,
                "right_censored_episodes": 1,
            }
            for branch_id, direction in BRANCHES
        ],
    )
    _write_tsv(
        results / "hbond_persistence_summary.tsv",
        [
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "edge_type": edge_type,
                "episodes": 20,
                "mean_persistence_ps": 4.0 if edge_type == "water_surface" else 3.0,
                "median_persistence_ps": 2.0,
                "p95_persistence_ps": 8.0,
                "right_censored_episodes": 0,
                "sampling_interval_ps": 1.0,
            }
            for branch_id, direction in BRANCHES
            for edge_type in ("water_surface", "water_water")
        ],
    )
    (results / "summary.json").write_text('{"status": "PASS"}\n', encoding="utf-8")
    _seal(results)
    return results


def _event_results(root: Path, case_id: str, carbon_h: int) -> Path:
    results = root / f"events_{case_id}"
    results.mkdir()
    events = []
    species = []
    identity = []
    motion = []
    for branch_id, _ in BRANCHES:
        event_id = f"{case_id}__{branch_id}__e0001"
        events.append(
            {
                "event_id": event_id,
                "case_id": case_id,
                "branch_id": branch_id,
                "event_types": "species_geometry,high_z",
                "anchor_time_ps": 20.0,
                "anchor_step": 20,
                "window_start_ps": 0.0,
                "window_end_ps": 40.0,
                "sample_count": 1,
                "tracked_oxygen_ids": 10,
                "state_frames": 41,
                "terminal_O_solution": 0,
                "terminal_OH_solution": 0,
                "terminal_OH4plus_solution": 0,
                "terminal_unassigned_H": 0,
                "proton_pool_first": 0,
                "proton_pool_last": 0,
                "species_returned": True,
                "tracked_frames": 41,
                "tracked_identity_complete": True,
                "tracked_h_counts": 2,
                "tracked_shared_hydrogen_frames": 0,
                "tracked_min_sharing_delta_A": 0.4,
                "tracked_mean_q_tet": 0.5,
                "tracked_mean_lsi_A2": 0.08,
                "tracked_mean_water_hbond_degree": 3.0,
                "tracked_mean_surface_hbond_count": 0.2,
                "tracked_intact_water": True,
                "tracked_max_z_A": 80.5,
                "tracked_returned_below_high_z": True,
                "nonzero_iz_max": 0,
            }
        )
        species.append(
            {
                "event_id": event_id,
                "case_id": case_id,
                "branch_id": branch_id,
                "carbon_owned_H": carbon_h,
                "unassigned_H": 0,
                "nonzero_iz": 0,
            }
        )
        identity.extend(
            [
                {
                    "event_id": event_id,
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "q_tet": "nan",
                    "oo_coordination": 3,
                },
                {
                    "event_id": event_id,
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "q_tet": 0.5,
                    "oo_coordination": 4,
                },
            ]
        )
        motion.append(
            {
                "event_id": event_id,
                "case_id": case_id,
                "branch_id": branch_id,
                "minimum_top_clearance_A": 0.8,
                "wall_samples": 1,
                "vx_pre_mps": 1.0,
                "vx_post_mps": 2.0,
                "vy_pre_mps": 0.0,
                "vy_post_mps": 0.0,
            }
        )
    _write_tsv(results / "events.tsv", events)
    _write_tsv(results / "frame_species.tsv", species)
    _write_tsv(results / "atom_identity.tsv", identity)
    _write_tsv(results / "motion_event_summary.tsv", motion)
    (results / "summary.json").write_text('{"status": "PASS"}\n', encoding="utf-8")
    _seal(results)
    return results


def _wall_results(root: Path) -> Path:
    results = root / "wall_motion"
    results.mkdir()
    _write_tsv(
        results / "group_summary.tsv",
        [
            {
                "group": "wall_associated",
                "metric": "vector_change_post_pre_mps",
                "events": 1,
                "mean": 1.0,
                "median": 1.0,
                "p05": 1.0,
                "p95": 1.0,
            }
        ],
    )
    _write_tsv(
        results / "matched_summary.tsv",
        [
            {
                "metric": "matched_vector_change_difference_mps",
                "wall_events": 1,
                "median_matched_difference": 0.2,
                "mean_matched_difference": 0.2,
                "bootstrap_median_ci95_low": -0.1,
                "bootstrap_median_ci95_high": 0.5,
            }
        ],
    )
    _write_tsv(
        results / "wall_event_matched_contrasts.tsv",
        [
            {
                "event_id": "droplet__fx__e0001",
                "case_id": "droplet",
                "branch_id": "fx",
                "drive_axis": "x",
                "minimum_top_clearance_A": 0.8,
                "matched_vector_change_difference_mps": 0.2,
                "matched_core_deflection_difference_mps": -0.3,
                "matched_delta_speed_difference_mps": -0.4,
            }
        ],
    )
    (results / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "events": 3,
                "wall_associated_events": 1,
                "non_wall_events": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _seal(results)
    return results


def _layered_results(root: Path) -> Path:
    results = root / "layered_film"
    results.mkdir()
    response = []
    density = []
    exchange = []
    residence = []
    for branch_id, direction in BRANCHES:
        response.append(
            {
                "case_id": "film",
                "branch_id": branch_id,
                "direction": direction,
                "layer_index": 1,
                "samples": 2,
                "occupied_fraction": 1.0,
                "mean_count": 10.0,
                "valid_paired_velocity_samples": 2,
                "mean_axis_velocity_mps": 0.5,
                "baseline_axis_velocity_mps": 0.0,
                "mean_excess_axis_velocity_mps": 0.5,
                "block_excess_sem_mps": 0.1,
                "mean_surface_flux_molecules_per_A_ps": 0.02,
                "mean_excess_surface_flux_molecules_per_A_ps": 0.01,
            }
        )
        for step, time_ps in ((0, 0.0), (1, 1.0)):
            density.append(
                {
                    "case_id": "film",
                    "branch_id": branch_id,
                    "direction": direction,
                    "step": step,
                    "time_ps": time_ps,
                    "layer_index": 1,
                    "mode_x": 1,
                    "mode_y": 0,
                    "amplitude": 0.2,
                    "phase_rad": 0.1 + 0.1 * step,
                }
            )
        exchange.append(
            {
                "case_id": "film",
                "branch_id": branch_id,
                "direction": direction,
                "step": 1,
                "time_ps": 1.0,
                "from_layer": 1,
                "to_layer": 2,
                "molecule_count": 2,
                "rate_per_ps": 2.0,
            }
        )
        residence.append(
            {
                "case_id": "film",
                "branch_id": branch_id,
                "direction": direction,
                "layer_index": 1,
                "episodes": 5,
                "mean_residence_ps": 8.0,
                "median_residence_ps": 7.0,
                "p95_residence_ps": 10.0,
                "right_censored_episodes": 1,
            }
        )
    _write_tsv(results / "layer_response_summary.tsv", response)
    _write_tsv(results / "density_modes.tsv", density)
    _write_tsv(results / "layer_exchange.tsv", exchange)
    _write_tsv(results / "residence_summary.tsv", residence)
    (results / "summary.json").write_text('{"status": "PASS"}\n', encoding="utf-8")
    _seal(results)
    return results


def _contract(tmp_path: Path) -> Path:
    kinematics = []
    morphology = []
    energy = []
    for case_id in CASES:
        for branch_id, direction in BRANCHES:
            velocity = {"none": "nan", "x": 1.0, "y": 2.0}[direction]
            kinematics.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "response_class": "CONTROL" if direction == "none" else "SUSTAINED",
                    "excess_block_mean_mps": velocity,
                    "excess_block_sem_mps": "nan" if direction == "none" else 0.1,
                    "excess_axis_velocity_full_mps": velocity,
                    "excess_axis_acf_positive_tau_ps": (
                        "nan" if direction == "none" else 5.0
                    ),
                    "vx_acf_positive_tau_ps": 5.0,
                    "vy_acf_positive_tau_ps": 6.0,
                    "dx_final_A": 1.0,
                    "dy_final_A": 2.0,
                }
            )
            morphology.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "morphology_class": "finite_droplet" if case_id == "droplet" else "spread_film",
                    "morphology_gate": "PASS",
                    "largest_fraction_min": 0.99 if case_id == "droplet" else "nan",
                    "largest_fraction_final": 1.0 if case_id == "droplet" else "nan",
                    "circularity_last500ps": 0.9 if case_id == "droplet" else "nan",
                }
            )
            energy.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "status": "PASS",
                    "drive_work_eV": 0.0 if direction == "none" else 1.0,
                    "thermostat_removed_eV": 0.5,
                    "closure_residual_eV": 0.01,
                    "mean_drive_power_eV_per_ps": 0.0 if direction == "none" else 0.001,
                }
            )
    _write_tsv(tmp_path / "kinematics.tsv", kinematics)
    _write_tsv(tmp_path / "morphology.tsv", morphology)
    _write_tsv(tmp_path / "energy.tsv", energy)
    _write_tsv(
        tmp_path / "finite.tsv",
        [
            {
                "case": "droplet",
                "branch": branch_id,
                "late_com_velocity_mps": 1.2,
                "late_tpcl_velocity_mps": 1.1,
                "late_front_velocity_mps": 1.3,
                "late_rear_velocity_mps": 1.0,
            }
            for branch_id, _ in BRANCHES
        ],
    )
    _write_tsv(
        tmp_path / "contact.tsv",
        [
            {
                "case_id": "droplet",
                "branch_id": branch_id,
                "unique_frames": 2,
                "angle_0p5_deg": 100.0,
                "fit_rmse_0p5_A": 0.2,
                "threshold_angle_span_deg": 1.0,
                "valid_50ps_blocks": 2,
                "block_angle_mean_deg": 101.0,
                "block_angle_std_deg": 2.0,
                "mean_largest_cluster_size": 10.0,
            }
            for branch_id, _ in BRANCHES
        ],
    )
    _write_tsv(
        tmp_path / "island_exchange.tsv",
        [
            {
                "case_id": "droplet",
                "branch_id": branch_id,
                "direction": direction,
                "observation_duration_ps": 1000.0,
                "main_island_mean_vx_mps": 1.0,
                "main_island_mean_vy_mps": 2.0,
                "satellite_size_weighted_mean_vx_mps": 3.0,
                "satellite_size_weighted_mean_vy_mps": 4.0,
                "track_to_track_transfer_count": 20,
                "persistent_island_transfer_count": 5,
                "lineage_reassignment_count": 15,
                "main_island_net_oxygen_transfer": 1,
                "track_to_track_transfer_rate_per_ns": 20.0,
            }
            for branch_id, direction in BRANCHES
        ],
    )
    water = {case_id: _water_results(tmp_path, case_id) for case_id in CASES}
    events = {
        "droplet": _event_results(tmp_path, "droplet", 6),
        "film": _event_results(tmp_path, "film", 0),
    }
    wall = _wall_results(tmp_path)
    layered = _layered_results(tmp_path)
    raw = {
        "schema_version": 1,
        "case_order": list(CASES),
        "branch_order": [branch_id for branch_id, _ in BRANCHES],
        "expected_water_frames_per_branch": 2,
        "required_result_kinds": [
            "water_structure",
            "events",
            "wall_motion",
            "layered_transport",
        ],
        "write_plots": False,
        "sources": [
            {"kind": "kinematics", "path": "kinematics.tsv"},
            {"kind": "morphology", "path": "morphology.tsv"},
            {"kind": "energy", "path": "energy.tsv"},
            {
                "kind": "finite_droplet",
                "path": "finite.tsv",
                "case_column": "case",
                "branch_column": "branch",
            },
            {"kind": "contact_angle", "path": "contact.tsv"},
            {"kind": "island_exchange_summary", "path": "island_exchange.tsv"},
            *[
                {
                    "kind": "water_structure",
                    "case_id": case_id,
                    "path": str(water[case_id]),
                }
                for case_id in CASES
            ],
            {
                "kind": "events",
                "case_id": "droplet",
                "path": str(events["droplet"]),
                "expected_carbon_owned_H": 6,
                "max_unassigned_H": 0,
            },
            {
                "kind": "events",
                "case_id": "film",
                "path": str(events["film"]),
                "expected_carbon_owned_H": 0,
                "max_unassigned_H": 0,
            },
            {"kind": "wall_motion", "path": str(wall)},
            {
                "kind": "layered_transport",
                "case_id": "film",
                "path": str(layered),
            },
        ],
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_aggregate_joins_branches_and_preserves_local_order_support(tmp_path: Path) -> None:
    output = tmp_path / "aggregate"
    result = run_contract(_contract(tmp_path), output)

    assert result["status"] == "PASS"
    assert result["branches"] == 6
    assert result["wall_associated_events"] == 1
    assert result["layer_response_rows"] == 3
    branch_rows = _read_tsv(output / "branch_mechanism_summary.tsv")
    droplet_x = next(
        row for row in branch_rows if row["case_id"] == "droplet" and row["direction"] == "x"
    )
    assert float(droplet_x["water_mean_q_tet"]) == pytest.approx(0.5)
    assert float(droplet_x["water_q_tet_valid_fraction"]) == pytest.approx(0.5)
    assert droplet_x["event_frame_unassigned_H_max"] == "0"
    assert float(droplet_x["finite_late_tpcl_velocity_mps"]) == pytest.approx(1.1)
    assert float(droplet_x["contact_angle_0p5_deg"]) == pytest.approx(100.0)
    assert float(droplet_x["island_exchange_track_to_track_transfer_rate_per_ns"]) == pytest.approx(
        20.0
    )
    contrasts = _read_tsv(output / "mechanism_contrasts.tsv")
    assert float(contrasts[0]["y_over_x_velocity"]) == pytest.approx(2.0)
    assert float(contrasts[0]["x_velocity_acf_positive_tau_ps"]) == pytest.approx(5.0)
    assert (output / "water_region_summary.tsv").is_file()
    assert (output / "event_summary.tsv").is_file()
    assert (output / "wall_motion_matched_summary.tsv").is_file()
    assert (output / "layer_response_summary.tsv").is_file()
    assert (output / "density_mode_summary.tsv").is_file()
    assert (output / "layer_exchange_summary.tsv").is_file()
    assert (output / "layer_residence_summary.tsv").is_file()
    assert (output / "input_manifest.tsv").is_file()
    assert (output / "REPORT.md").is_file()
    assert "Identity-resolved island exchange" in (output / "REPORT.md").read_text()


def test_aggregate_accepts_explicit_carbon_ownership_states(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    event_source = next(
        source
        for source in raw["sources"]
        if source["kind"] == "events" and source["case_id"] == "droplet"
    )
    event_source.pop("expected_carbon_owned_H")
    event_source["allowed_carbon_owned_H"] = [5, 6]
    event_path = Path(event_source["path"])
    species_path = event_path / "frame_species.tsv"
    rows = _read_tsv(species_path)
    rows[0]["carbon_owned_H"] = "5"
    _write_tsv(species_path, rows)
    _seal(event_path)
    contract.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    output = tmp_path / "aggregate"
    run_contract(contract, output)
    event_rows = _read_tsv(output / "event_summary.tsv")
    droplet_rows = [row for row in event_rows if row["case_id"] == "droplet"]
    assert min(int(row["carbon_owned_H_min"]) for row in droplet_rows) == 5
    assert max(int(row["carbon_owned_H_max"]) for row in droplet_rows) == 6


def test_aggregate_rejects_finite_qtet_below_four_neighbors(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    event_path = Path(
        next(
            source["path"]
            for source in raw["sources"]
            if source["kind"] == "events" and source["case_id"] == "film"
        )
    )
    identity_path = event_path / "atom_identity.tsv"
    rows = _read_tsv(identity_path)
    rows[0]["q_tet"] = "0.25"
    _write_tsv(identity_path, rows)
    _seal(event_path)
    with pytest.raises(ValueError, match="finite q_tet"):
        run_contract(contract, tmp_path / "aggregate")


def test_aggregate_rejects_missing_core_branch(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    rows = _read_tsv(tmp_path / "kinematics.tsv")
    _write_tsv(tmp_path / "kinematics.tsv", rows[:-1])
    with pytest.raises(ValueError, match="branch coverage"):
        run_contract(contract, tmp_path / "aggregate")


def test_aggregate_rejects_short_event_window(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    event_path = Path(
        next(source["path"] for source in raw["sources"] if source["kind"] == "events")
    )
    events_path = event_path / "events.tsv"
    rows = _read_tsv(events_path)
    rows[0]["window_start_ps"] = "1.0"
    _write_tsv(events_path, rows)
    _seal(event_path)
    with pytest.raises(ValueError, match="shorter than required before"):
        run_contract(contract, tmp_path / "aggregate")


def test_aggregate_rejects_nonzero_z_image(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    event_path = Path(
        next(source["path"] for source in raw["sources"] if source["kind"] == "events")
    )
    species_path = event_path / "frame_species.tsv"
    rows = _read_tsv(species_path)
    rows[0]["nonzero_iz"] = "1"
    _write_tsv(species_path, rows)
    _seal(event_path)
    with pytest.raises(ValueError, match="nonzero Z image"):
        run_contract(contract, tmp_path / "aggregate")


def test_aggregate_records_nonreturning_species_and_proton_pool_change(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    event_path = Path(
        next(source["path"] for source in raw["sources"] if source["kind"] == "events")
    )
    events_path = event_path / "events.tsv"
    rows = _read_tsv(events_path)
    rows[0]["terminal_OH_solution"] = "1"
    rows[0]["proton_pool_last"] = "1"
    rows[0]["species_returned"] = "False"
    _write_tsv(events_path, rows)
    _seal(event_path)
    output = tmp_path / "aggregate"
    result = run_contract(contract, output)
    assert result["species_not_returned_events"] == 1
    assert result["proton_pool_change_events"] == 1
    event_rows = _read_tsv(output / "event_summary.tsv")
    changed = next(row for row in event_rows if row["case_id"] == "droplet")
    assert changed["terminal_OH_solution_events"] == "1"
    assert changed["proton_pool_change_events"] == "1"


def test_aggregate_can_require_proton_pool_conservation(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    event_source = next(source for source in raw["sources"] if source["kind"] == "events")
    event_source["require_proton_pool_conservation"] = True
    event_path = Path(event_source["path"])
    events_path = event_path / "events.tsv"
    rows = _read_tsv(events_path)
    rows[0]["proton_pool_last"] = "1"
    _write_tsv(events_path, rows)
    _seal(event_path)
    contract.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="proton pool changed"):
        run_contract(contract, tmp_path / "aggregate")


def test_cli_registers_constant_force_aggregate(tmp_path: Path) -> None:
    args = build_cli_parser().parse_args(
        [
            "postprocess",
            "constant-force-aggregate",
            "--contract",
            str(tmp_path / "contract.json"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.func.__name__ == "_cmd_postprocess_constant_force_aggregate"
