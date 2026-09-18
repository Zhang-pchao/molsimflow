"""Aggregate morphology-aware constant-force transport diagnostics.

The aggregator consumes only paths declared in a JSON contract. It joins
validated branch summaries with event and water-structure result directories,
records every consumed file, and keeps single-trajectory temporal statistics
separate from replicate uncertainty.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from molsimflow.postprocess.constant_force_aggregate_sources import (
    ALLOWED_SOURCE_KINDS,
    ENERGY_REQUIRED,
    KINEMATICS_REQUIRED,
    MORPHOLOGY_REQUIRED,
    OPTIONAL_BRANCH_TABLE_KINDS,
    REQUIRED_TABLE_KINDS,
    _aggregate_event_source,
    _aggregate_layered_transport_source,
    _aggregate_wall_motion_source,
    _aggregate_water_source,
    _float,
    _int,
    _load_keyed_tables,
    _optional_prefixed_fields,
    _source_records,
    _write_tsv,
)

def _assemble_branch_rows(
    expected_keys: Sequence[tuple[str, str]],
    kinematics: Mapping[tuple[str, str], Mapping[str, str]],
    morphology: Mapping[tuple[str, str], Mapping[str, str]],
    energy: Mapping[tuple[str, str], Mapping[str, str]],
    optional: Mapping[str, Mapping[tuple[str, str], Mapping[str, str]]],
    water: Mapping[tuple[str, str], Mapping[str, object]],
    events: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for key in expected_keys:
        kin, morph, power = kinematics[key], morphology[key], energy[key]
        row: dict[str, object] = {
            "case_id": key[0],
            "branch_id": key[1],
            "direction": kin["direction"],
            "response_class": kin["response_class"],
            "excess_block_mean_mps": _float(kin["excess_block_mean_mps"]),
            "excess_block_sem_mps": _float(kin["excess_block_sem_mps"]),
            "excess_axis_velocity_full_mps": _float(kin.get("excess_axis_velocity_full_mps")),
            "excess_axis_acf_positive_tau_ps": _float(
                kin["excess_axis_acf_positive_tau_ps"]
            ),
            "vx_acf_positive_tau_ps": _float(kin.get("vx_acf_positive_tau_ps")),
            "vy_acf_positive_tau_ps": _float(kin.get("vy_acf_positive_tau_ps")),
            "dx_final_A": _float(kin.get("dx_final_A")),
            "dy_final_A": _float(kin.get("dy_final_A")),
            "morphology_class": morph["morphology_class"],
            "morphology_gate": morph["morphology_gate"],
            "largest_fraction_min": _float(morph.get("largest_fraction_min")),
            "largest_fraction_final": _float(morph.get("largest_fraction_final")),
            "circularity_last500ps": _float(morph.get("circularity_last500ps")),
            "drive_work_eV": _float(power["drive_work_eV"]),
            "thermostat_removed_eV": _float(power["thermostat_removed_eV"]),
            "closure_residual_eV": _float(power["closure_residual_eV"]),
            "mean_drive_power_eV_per_ps": _float(power["mean_drive_power_eV_per_ps"]),
        }
        row.update(
            _optional_prefixed_fields(
                optional.get("finite_droplet", {}).get(key),
                "finite",
                (
                    "late_com_velocity_mps",
                    "late_tpcl_velocity_mps",
                    "late_front_velocity_mps",
                    "late_rear_velocity_mps",
                    "late_front_minus_rear_velocity_mps",
                    "late_abs_tpcl_com_lag_A",
                    "minimum_largest_cluster",
                    "late_circularity",
                    "late_z95span_A",
                    "late_front_ch3_fraction",
                    "late_rear_ch3_fraction",
                    "late_front_minus_rear_ch3_fraction",
                    "late_tpcl_surface_hbond_per_h2o",
                    "late_tpcl_water_hbond_degree",
                    "late_tpcl_hbond_largest_component_fraction",
                    "late_tpcl_qtet",
                    "late_tpcl_lsi_A2",
                    "late_interfacial_moving_frame_velocity_Aps",
                    "late_interfacial_stressvol_parallel_z_barA3",
                ),
            )
        )
        row.update(
            _optional_prefixed_fields(
                optional.get("contact_angle", {}).get(key),
                "contact",
                (
                    "unique_frames",
                    "angle_0p5_deg",
                    "fit_rmse_0p5_A",
                    "threshold_angle_span_deg",
                    "valid_50ps_blocks",
                    "block_angle_mean_deg",
                    "block_angle_std_deg",
                    "mean_largest_cluster_size",
                ),
            )
        )
        row.update(
            _optional_prefixed_fields(
                optional.get("island_summary", {}).get(key),
                "island",
                (
                    "component_count_mean",
                    "component_count_max",
                    "split_events",
                    "merge_events",
                ),
            )
        )
        row.update(
            _optional_prefixed_fields(
                optional.get("site_exchange_summary", {}).get(key),
                "site_exchange",
                (
                    "mean_assigned_water",
                    "mean_retained_assignments_per_10ps",
                    "mean_site_exchanges_per_10ps",
                    "mean_new_assignments_per_10ps",
                    "mean_lost_assignments_per_10ps",
                    "mean_site_distance_A",
                ),
            )
        )
        water_row = water[key]
        for name in (
            "mean_q_tet",
            "q_tet_valid_fraction",
            "mean_lsi_A2",
            "mean_oo_coordination",
            "mean_surface_hbond_per_water",
            "mean_water_water_hbond_degree",
            "mean_hbond_largest_component_fraction",
            "mean_water_water_edge_turnover",
            "mean_water_surface_edge_turnover",
            "surface_hbond_mean_persistence_ps",
            "water_hbond_mean_persistence_ps",
        ):
            row[f"water_{name}"] = water_row[name]
        event_row = events[key]
        for name in (
            "event_count",
            "species_geometry_events",
            "high_z_events",
            "wall_approach_events",
            "species_returned_events",
            "species_not_returned_events",
            "terminal_O_solution_events",
            "terminal_OH_solution_events",
            "terminal_OH4plus_solution_events",
            "minimum_window_before_ps",
            "minimum_window_after_ps",
            "maximum_absolute_proton_pool_change",
            "proton_pool_change_events",
            "tracked_intact_water_events",
            "tracked_identity_incomplete_events",
            "tracked_returned_below_high_z_events",
            "minimum_tracked_sharing_delta_A",
            "maximum_nonzero_iz",
            "carbon_owned_H_min",
            "carbon_owned_H_max",
            "frame_unassigned_H_max",
            "low_coordination_q_tet_violations",
            "mean_event_vector_change_mps",
            "mean_event_speed_change_mps",
            "minimum_top_clearance_A",
        ):
            row[f"event_{name}"] = event_row[name]
        rows.append(row)
    return rows


def _mechanism_contrasts(
    branch_rows: Sequence[dict[str, object]], case_order: Sequence[str]
) -> list[dict[str, object]]:
    by_case: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in branch_rows:
        by_case[str(row["case_id"])][str(row["direction"]).lower()] = row
    result = []
    for case_id in case_order:
        directions = by_case[case_id]
        if set(directions) != {"none", "x", "y"}:
            raise ValueError(f"{case_id}: expected exactly none/x/y directions")
        x_row, y_row = directions["x"], directions["y"]

        def value(row: Mapping[str, object], name: str) -> float:
            return _float(row.get(name))

        x_velocity = value(x_row, "excess_block_mean_mps")
        y_velocity = value(y_row, "excess_block_mean_mps")
        ratio = y_velocity / x_velocity if abs(x_velocity) > 1.0e-12 else math.nan
        result.append(
            {
                "case_id": case_id,
                "morphology_class": x_row["morphology_class"],
                "x_branch_id": x_row["branch_id"],
                "y_branch_id": y_row["branch_id"],
                "x_response_class": x_row["response_class"],
                "y_response_class": y_row["response_class"],
                "x_excess_velocity_mps": x_velocity,
                "y_excess_velocity_mps": y_velocity,
                "y_minus_x_velocity_mps": y_velocity - x_velocity,
                "y_over_x_velocity": ratio,
                "x_velocity_sem_mps": value(x_row, "excess_block_sem_mps"),
                "y_velocity_sem_mps": value(y_row, "excess_block_sem_mps"),
                "x_velocity_acf_positive_tau_ps": value(
                    x_row, "excess_axis_acf_positive_tau_ps"
                ),
                "y_velocity_acf_positive_tau_ps": value(
                    y_row, "excess_axis_acf_positive_tau_ps"
                ),
                "x_drive_power_eV_per_ps": value(x_row, "mean_drive_power_eV_per_ps"),
                "y_drive_power_eV_per_ps": value(y_row, "mean_drive_power_eV_per_ps"),
                "x_surface_hbond_per_water": value(x_row, "water_mean_surface_hbond_per_water"),
                "y_surface_hbond_per_water": value(y_row, "water_mean_surface_hbond_per_water"),
                "y_minus_x_surface_hbond_per_water": value(
                    y_row, "water_mean_surface_hbond_per_water"
                )
                - value(x_row, "water_mean_surface_hbond_per_water"),
                "x_mean_q_tet": value(x_row, "water_mean_q_tet"),
                "y_mean_q_tet": value(y_row, "water_mean_q_tet"),
                "y_minus_x_mean_q_tet": value(y_row, "water_mean_q_tet")
                - value(x_row, "water_mean_q_tet"),
                "x_mean_lsi_A2": value(x_row, "water_mean_lsi_A2"),
                "y_mean_lsi_A2": value(y_row, "water_mean_lsi_A2"),
                "x_event_count": _int(x_row.get("event_event_count")),
                "y_event_count": _int(y_row.get("event_event_count")),
                "x_high_z_events": _int(x_row.get("event_high_z_events")),
                "y_high_z_events": _int(y_row.get("event_high_z_events")),
            }
        )
    return result


def _plot_overview(
    branch_rows: Sequence[dict[str, object]], case_order: Sequence[str], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directions = ("x", "y")
    colors = {"x": "#2878B5", "y": "#E07032"}
    by_case: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in branch_rows:
        by_case[str(row["case_id"])][str(row["direction"]).lower()] = row
    x_positions = list(range(len(case_order)))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    panels = (
        (axes[0, 0], "excess_block_mean_mps", "Excess velocity (m/s)"),
        (axes[0, 1], "largest_fraction_min", "Minimum largest-component fraction"),
        (
            axes[1, 0],
            "water_mean_surface_hbond_per_water",
            "Surface H bonds per water",
        ),
        (axes[1, 1], "water_mean_q_tet", "Mean local tetrahedral order"),
    )
    for axis, field, label in panels:
        for index, direction in enumerate(directions):
            values = [_float(by_case[case_id][direction].get(field)) for case_id in case_order]
            positions = [position + (index - 0.5) * width for position in x_positions]
            kwargs: dict[str, object] = {}
            if field == "excess_block_mean_mps":
                kwargs["yerr"] = [
                    _float(by_case[case_id][direction].get("excess_block_sem_mps"))
                    for case_id in case_order
                ]
                kwargs["capsize"] = 2.5
            axis.bar(
                positions,
                values,
                width=width,
                color=colors[direction],
                label=direction.upper(),
                **kwargs,
            )
        axis.set_ylabel(label)
        axis.set_xticks(x_positions, case_order, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Constant-force transport: single-trajectory morphology-aware diagnostics",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output / "mechanism_overview.png", dpi=250)
    plt.close(figure)


def _write_report(
    output: Path,
    case_order: Sequence[str],
    contrasts: Sequence[dict[str, object]],
    input_count: int,
    wall_summary: Mapping[str, object] | None,
    wall_matched: Sequence[Mapping[str, object]],
    layer_response: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# Constant-force mechanism aggregate",
        "",
        "Status: `PASS`.",
        "",
        f"The contract covers `{len(case_order)}` cases and `{len(case_order) * 3}` branches.",
        f"The input manifest records `{input_count}` consumed files.",
        "",
        "## X/Y response",
        "",
        "| Case | Morphology | X response | Y response | X velocity (m/s) | Y velocity (m/s) | X ACF tau (ps) | Y ACF tau (ps) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            "| {case_id} | {morphology_class} | {x_response_class} | "
            "{y_response_class} | {x_excess_velocity_mps:.6g} | "
            "{y_excess_velocity_mps:.6g} | {x_velocity_acf_positive_tau_ps:.6g} | "
            "{y_velocity_acf_positive_tau_ps:.6g} |".format(**row)
        )
    if wall_summary is not None:
        lines.extend(
            [
                "",
                "## Wall and high-Z association",
                "",
                f"The matched audit contains `{_int(wall_summary.get('events'))}` events, "
                f"including `{_int(wall_summary.get('wall_associated_events'))}` wall-associated "
                "events.",
                "",
                "| Metric | Matched median difference (m/s) | Bootstrap 95% interval (m/s) |",
                "|---|---:|---:|",
            ]
        )
        for row in wall_matched:
            lines.append(
                f"| {row['metric']} | {_float(row['median_matched_difference']):.6g} | "
                f"[{_float(row['bootstrap_median_ci95_low']):.6g}, "
                f"{_float(row['bootstrap_median_ci95_high']):.6g}] |"
            )
    if layer_response:
        lines.extend(
            [
                "",
                "## Layer-resolved film response",
                "",
                "The layer tables preserve excess velocity, surface flux, density modes, "
                "inter-layer exchange, and residence statistics for the spread-film case.",
                "",
                "| Branch | Direction | Layer | Occupied fraction | Excess velocity (m/s) | Excess flux (molecule/A/ps) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in layer_response:
            if _float(row["occupied_fraction"]) <= 0.0:
                continue
            lines.append(
                f"| {row['branch_id']} | {row['direction']} | {_int(row['layer_index'])} | "
                f"{_float(row['occupied_fraction']):.6g} | "
                f"{_float(row['mean_excess_axis_velocity_mps']):.6g} | "
                f"{_float(row['mean_excess_surface_flux_molecules_per_A_ps']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "All uncertainty columns are temporal block statistics from one trajectory per branch.",
            "They are not independent-replicate confidence intervals. Event association, hydrogen",
            "sharing, surface registry, and network changes are descriptive diagnostics and do not",
            "by themselves establish reaction kinetics, friction coefficients, free-energy barriers,",
            "or causal molecular mechanisms.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_contract(contract_path: Path, output: Path) -> dict[str, object]:
    """Run a contract-driven aggregate and return its terminal summary."""

    contract_path = contract_path.resolve()
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    case_order = [str(value) for value in raw.get("case_order", [])]
    branch_order = [str(value) for value in raw.get("branch_order", [])]
    if not case_order or len(case_order) != len(set(case_order)):
        raise ValueError("case_order must contain unique case identifiers")
    if not branch_order or len(branch_order) != len(set(branch_order)):
        raise ValueError("branch_order must contain unique branch identifiers")
    if len(branch_order) != 3:
        raise ValueError("constant-force aggregate requires three branches per case")
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    normalized_sources: list[dict[str, object]] = []
    for source in sources:
        if not isinstance(source, dict) or source.get("kind") not in ALLOWED_SOURCE_KINDS:
            raise ValueError(f"unsupported source entry {source!r}")
        if not str(source.get("path", "")):
            raise ValueError("every source requires a path")
        normalized_sources.append(source)
    observed_kinds = {str(source["kind"]) for source in normalized_sources}
    missing_kinds = REQUIRED_TABLE_KINDS.difference(observed_kinds)
    if missing_kinds:
        raise ValueError(f"missing required source kinds {sorted(missing_kinds)}")
    required_result_kinds = {str(value) for value in raw.get("required_result_kinds", [])}
    missing_result_kinds = required_result_kinds.difference(observed_kinds)
    if missing_result_kinds:
        raise ValueError(f"missing required result kinds {sorted(missing_result_kinds)}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    input_records = _source_records("contract", "", [contract_path])
    kinematics = _load_keyed_tables(
        normalized_sources,
        "kinematics",
        KINEMATICS_REQUIRED,
        contract_path,
        input_records,
    )
    morphology = _load_keyed_tables(
        normalized_sources,
        "morphology",
        MORPHOLOGY_REQUIRED,
        contract_path,
        input_records,
    )
    energy = _load_keyed_tables(
        normalized_sources,
        "energy",
        ENERGY_REQUIRED,
        contract_path,
        input_records,
    )
    optional: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for kind in OPTIONAL_BRANCH_TABLE_KINDS:
        optional[kind] = _load_keyed_tables(
            normalized_sources,
            kind,
            {"case_id", "branch_id"},
            contract_path,
            input_records,
        )

    expected_keys = [(case_id, branch_id) for case_id in case_order for branch_id in branch_order]
    expected_key_set = set(expected_keys)
    for name, table in (
        ("kinematics", kinematics),
        ("morphology", morphology),
        ("energy", energy),
    ):
        if set(table) != expected_key_set:
            raise ValueError(f"{name}: branch coverage does not match contract")
    if any(row["morphology_gate"] != "PASS" for row in morphology.values()):
        raise ValueError("morphology gate is not PASS for every branch")
    if any(row["status"] != "PASS" for row in energy.values()):
        raise ValueError("energy gate is not PASS for every branch")

    expected_frames_raw = raw.get("expected_water_frames_per_branch")
    expected_frames = int(expected_frames_raw) if expected_frames_raw is not None else None
    water_rows: list[dict[str, object]] = []
    water_summary: dict[tuple[str, str], dict[str, object]] = {}
    water_cases = set()
    event_rows: list[dict[str, object]] = []
    event_cases = set()
    wall_group_rows: list[dict[str, str]] = []
    wall_matched_rows: list[dict[str, str]] = []
    wall_event_rows: list[dict[str, str]] = []
    wall_summary: dict[str, object] | None = None
    layered_rows: dict[str, list[dict[str, object]]] = {
        "response": [],
        "density": [],
        "exchange": [],
        "residence": [],
    }
    layered_cases: set[str] = set()
    for source in normalized_sources:
        if source["kind"] == "water_structure":
            case_id = str(source.get("case_id", ""))
            if case_id in water_cases:
                raise ValueError(f"duplicate water_structure source for {case_id}")
            water_cases.add(case_id)
            rows, summary = _aggregate_water_source(
                source,
                contract_path,
                set(branch_order),
                expected_frames,
                input_records,
            )
            water_rows.extend(rows)
            water_summary.update(summary)
        elif source["kind"] == "events":
            case_id = str(source.get("case_id", ""))
            if case_id in event_cases:
                raise ValueError(f"duplicate events source for {case_id}")
            event_cases.add(case_id)
            event_rows.extend(
                _aggregate_event_source(source, contract_path, set(branch_order), input_records)
            )
        elif source["kind"] == "wall_motion":
            if wall_summary is not None:
                raise ValueError("duplicate wall_motion source")
            (
                wall_group_rows,
                wall_matched_rows,
                wall_event_rows,
                wall_summary,
            ) = _aggregate_wall_motion_source(
                source,
                contract_path,
                set(case_order),
                set(branch_order),
                input_records,
            )
        elif source["kind"] == "layered_transport":
            case_id = str(source.get("case_id", ""))
            if case_id not in set(case_order):
                raise ValueError(f"layered_transport case {case_id!r} is outside contract")
            if case_id in layered_cases:
                raise ValueError(f"duplicate layered_transport source for {case_id}")
            layered_cases.add(case_id)
            current = _aggregate_layered_transport_source(
                source, contract_path, set(branch_order), input_records
            )
            for name, rows in current.items():
                layered_rows[name].extend(rows)
    if water_cases != set(case_order):
        raise ValueError("water_structure sources do not cover every case")
    if event_cases != set(case_order):
        raise ValueError("events sources do not cover every case")
    event_summary = {(str(row["case_id"]), str(row["branch_id"])): row for row in event_rows}
    if set(water_summary) != expected_key_set or set(event_summary) != expected_key_set:
        raise ValueError("water/event branch coverage does not match contract")

    branch_rows = _assemble_branch_rows(
        expected_keys,
        kinematics,
        morphology,
        energy,
        optional,
        water_summary,
        event_summary,
    )
    contrasts = _mechanism_contrasts(branch_rows, case_order)
    _write_tsv(output / "branch_mechanism_summary.tsv", branch_rows)
    _write_tsv(output / "mechanism_contrasts.tsv", contrasts)
    _write_tsv(output / "water_region_summary.tsv", water_rows)
    _write_tsv(output / "event_summary.tsv", event_rows)
    if wall_summary is not None:
        _write_tsv(output / "wall_motion_group_summary.tsv", wall_group_rows)
        _write_tsv(output / "wall_motion_matched_summary.tsv", wall_matched_rows)
        _write_tsv(output / "wall_event_matched_contrasts.tsv", wall_event_rows)
    if layered_rows["response"]:
        _write_tsv(output / "layer_response_summary.tsv", layered_rows["response"])
        _write_tsv(output / "density_mode_summary.tsv", layered_rows["density"])
        _write_tsv(output / "layer_exchange_summary.tsv", layered_rows["exchange"])
        _write_tsv(output / "layer_residence_summary.tsv", layered_rows["residence"])
    _write_tsv(output / "input_manifest.tsv", input_records)
    if raw.get("write_plots", True):
        _plot_overview(branch_rows, case_order, output)
    _write_report(
        output,
        case_order,
        contrasts,
        len(input_records),
        wall_summary,
        wall_matched_rows,
        layered_rows["response"],
    )
    summary: dict[str, object] = {
        "status": "PASS",
        "scientific_status": (
            "SINGLE_TRAJECTORY_DESCRIPTIVE_MECHANISM_DIAGNOSTIC_NOT_CAUSAL_"
            "MECHANISM_FRICTION_FREE_ENERGY_OR_REPLICATE_UNCERTAINTY"
        ),
        "cases": len(case_order),
        "branches": len(branch_rows),
        "water_region_rows": len(water_rows),
        "event_summary_rows": len(event_rows),
        "species_not_returned_events": sum(
            _int(row["species_not_returned_events"]) for row in event_rows
        ),
        "proton_pool_change_events": sum(
            _int(row["proton_pool_change_events"]) for row in event_rows
        ),
        "wall_associated_events": (
            _int(wall_summary.get("wall_associated_events")) if wall_summary else 0
        ),
        "layer_response_rows": len(layered_rows["response"]),
        "density_mode_summary_rows": len(layered_rows["density"]),
        "layer_exchange_summary_rows": len(layered_rows["exchange"]),
        "layer_residence_rows": len(layered_rows["residence"]),
        "input_files": len(input_records),
        "event_window_gate": "PASS",
        "proton_pool_tracking_gate": "PASS",
        "z_image_gate": "PASS",
        "low_coordination_q_tet_gate": "PASS",
        "carbon_hydrogen_ownership_gate": "PASS",
        "output_files": [
            "branch_mechanism_summary.tsv",
            "mechanism_contrasts.tsv",
            "water_region_summary.tsv",
            "event_summary.tsv",
            "wall_motion_group_summary.tsv" if wall_summary is not None else None,
            "wall_motion_matched_summary.tsv" if wall_summary is not None else None,
            "wall_event_matched_contrasts.tsv" if wall_summary is not None else None,
            "layer_response_summary.tsv" if layered_rows["response"] else None,
            "density_mode_summary.tsv" if layered_rows["density"] else None,
            "layer_exchange_summary.tsv" if layered_rows["exchange"] else None,
            "layer_residence_summary.tsv" if layered_rows["residence"] else None,
            "mechanism_overview.png" if raw.get("write_plots", True) else None,
            "REPORT.md",
            "input_manifest.tsv",
        ],
    }
    summary["output_files"] = [name for name in summary["output_files"] if name is not None]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
