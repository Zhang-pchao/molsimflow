"""Input adapters and validation for constant-force mechanism aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

REQUIRED_TABLE_KINDS = {"kinematics", "morphology", "energy"}
OPTIONAL_BRANCH_TABLE_KINDS = {
    "contact_angle": "contact",
    "finite_droplet": "finite",
    "island_exchange_summary": "island_exchange",
    "island_summary": "island",
    "site_exchange_summary": "site_exchange",
}
RESULT_DIRECTORY_KINDS = {
    "water_structure",
    "events",
    "wall_motion",
    "layered_transport",
}
ALLOWED_SOURCE_KINDS = (
    REQUIRED_TABLE_KINDS | set(OPTIONAL_BRANCH_TABLE_KINDS) | RESULT_DIRECTORY_KINDS
)

KINEMATICS_REQUIRED = {
    "case_id",
    "branch_id",
    "direction",
    "response_class",
    "excess_block_mean_mps",
    "excess_block_sem_mps",
    "excess_axis_acf_positive_tau_ps",
}
MORPHOLOGY_REQUIRED = {
    "case_id",
    "branch_id",
    "morphology_class",
    "morphology_gate",
}
ENERGY_REQUIRED = {
    "case_id",
    "branch_id",
    "status",
    "drive_work_eV",
    "thermostat_removed_eV",
    "closure_residual_eV",
    "mean_drive_power_eV_per_ps",
}
WATER_REQUIRED = {
    "case_id",
    "branch_id",
    "direction",
    "step",
    "time_ps",
    "region",
    "water_count",
    "water_fraction",
    "physical_largest_component_fraction",
    "mean_q_tet",
    "mean_lsi_A2",
    "mean_oo_coordination",
    "h_coordination_defect_fraction",
    "water_water_hbond_edges",
    "water_surface_hbond_edges",
    "hbond_largest_component_fraction",
    "water_water_edge_turnover",
    "water_surface_edge_turnover",
}
EVENT_REQUIRED = {
    "event_id",
    "case_id",
    "branch_id",
    "event_types",
    "anchor_time_ps",
    "window_start_ps",
    "window_end_ps",
    "state_frames",
    "terminal_O_solution",
    "terminal_OH_solution",
    "terminal_OH4plus_solution",
    "terminal_unassigned_H",
    "proton_pool_first",
    "proton_pool_last",
    "species_returned",
    "tracked_oxygen_ids",
    "tracked_identity_complete",
    "tracked_intact_water",
    "tracked_returned_below_high_z",
    "tracked_min_sharing_delta_A",
    "nonzero_iz_max",
}
FRAME_SPECIES_REQUIRED = {
    "case_id",
    "branch_id",
    "carbon_owned_H",
    "unassigned_H",
    "nonzero_iz",
}
ATOM_IDENTITY_REQUIRED = {"case_id", "branch_id", "q_tet", "oo_coordination"}
MOTION_EVENT_REQUIRED = {
    "event_id",
    "case_id",
    "branch_id",
    "minimum_top_clearance_A",
    "wall_samples",
    "vx_pre_mps",
    "vx_post_mps",
    "vy_pre_mps",
    "vy_post_mps",
}
WALL_GROUP_REQUIRED = {"group", "metric", "events", "mean", "median", "p05", "p95"}
WALL_MATCHED_REQUIRED = {
    "metric",
    "wall_events",
    "median_matched_difference",
    "mean_matched_difference",
    "bootstrap_median_ci95_low",
    "bootstrap_median_ci95_high",
}
WALL_EVENT_REQUIRED = {
    "event_id",
    "case_id",
    "branch_id",
    "drive_axis",
    "minimum_top_clearance_A",
    "matched_vector_change_difference_mps",
    "matched_core_deflection_difference_mps",
    "matched_delta_speed_difference_mps",
}
LAYER_RESPONSE_REQUIRED = {
    "case_id",
    "branch_id",
    "direction",
    "layer_index",
    "samples",
    "occupied_fraction",
    "mean_count",
    "mean_excess_axis_velocity_mps",
    "block_excess_sem_mps",
    "mean_excess_surface_flux_molecules_per_A_ps",
}
LAYER_DENSITY_REQUIRED = {
    "case_id",
    "branch_id",
    "direction",
    "step",
    "time_ps",
    "layer_index",
    "mode_x",
    "mode_y",
    "amplitude",
    "phase_rad",
}
LAYER_EXCHANGE_REQUIRED = {
    "case_id",
    "branch_id",
    "direction",
    "from_layer",
    "to_layer",
    "molecule_count",
    "rate_per_ps",
}
LAYER_RESIDENCE_REQUIRED = {
    "case_id",
    "branch_id",
    "direction",
    "layer_index",
    "episodes",
    "mean_residence_ps",
    "median_residence_ps",
    "p95_residence_ps",
    "right_censored_episodes",
}


def _read_tsv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        return list(reader)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _resolve(contract_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = contract_path.parent / path
    return path.resolve()


def _float(value: object, *, allow_nan: bool = True) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isfinite(parsed):
        return parsed
    return parsed if allow_nan else math.nan


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return math.fsum(finite) / len(finite) if finite else math.nan


def _minimum(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _maximum(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _quantile(values: Iterable[float], fraction: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    position = max(0.0, min(1.0, fraction)) * (len(finite) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum_manifest(results_dir: Path) -> None:
    candidates = [results_dir / "OUTPUT-SHA256SUMS", results_dir.parent / "OUTPUT-SHA256SUMS"]
    manifest = next((candidate for candidate in candidates if candidate.is_file()), None)
    if manifest is None:
        raise FileNotFoundError(f"{results_dir}: no OUTPUT-SHA256SUMS")
    root = (
        results_dir.parent.resolve()
        if manifest.parent == results_dir.parent
        else results_dir.resolve()
    )
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{manifest}: malformed checksum line")
        target = Path(parts[1].lstrip(" *").strip())
        if not target.is_absolute():
            target = manifest.parent / target
        target = target.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"{manifest}: checksum target escapes result root")
        if not target.is_file() or _sha256(target) != parts[0]:
            raise ValueError(f"{manifest}: checksum mismatch for {target}")
        records.append(target)
    if not records:
        raise ValueError(f"{manifest}: empty checksum manifest")


def _verify_result_directory(results_dir: Path) -> None:
    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError(f"{summary_path}: result status is not PASS")
    _verify_checksum_manifest(results_dir)


def _source_records(kind: str, case_id: str, paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "kind": kind,
            "case_id": case_id,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _load_keyed_tables(
    sources: Sequence[dict[str, object]],
    kind: str,
    required: set[str],
    contract_path: Path,
    input_records: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for source in sources:
        if source["kind"] != kind:
            continue
        path = _resolve(contract_path, str(source["path"]))
        case_column = str(source.get("case_column", "case_id"))
        branch_column = str(source.get("branch_column", "branch_id"))
        source_required = (required - {"case_id", "branch_id"}) | {
            case_column,
            branch_column,
        }
        rows = _read_tsv(path, source_required)
        input_records.extend(_source_records(kind, str(source.get("case_id", "")), [path]))
        for original in rows:
            row = dict(original)
            row["case_id"] = original[case_column]
            row["branch_id"] = original[branch_column]
            key = (row["case_id"], row["branch_id"])
            if key in result:
                raise ValueError(f"duplicate {kind} row for {key}")
            result[key] = row
    return result


def _optional_prefixed_fields(
    row: Mapping[str, str] | None, prefix: str, names: Sequence[str]
) -> dict[str, object]:
    return {f"{prefix}_{name}": row.get(name, "") if row is not None else "" for name in names}


def _aggregate_water_source(
    source: dict[str, object],
    contract_path: Path,
    expected_branches: set[str],
    expected_frames: int | None,
    input_records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    case_id = str(source.get("case_id", ""))
    if not case_id:
        raise ValueError("water_structure source requires case_id")
    results = _resolve(contract_path, str(source["path"]))
    _verify_result_directory(results)
    frame_path = results / "water_structure_by_frame.tsv"
    residence_path = results / "region_residence_summary.tsv"
    persistence_path = results / "hbond_persistence_summary.tsv"
    frames = _read_tsv(frame_path, WATER_REQUIRED)
    residence = _read_tsv(
        residence_path,
        {
            "case_id",
            "branch_id",
            "region",
            "episodes",
            "mean_residence_ps",
            "median_residence_ps",
            "p95_residence_ps",
        },
    )
    persistence = _read_tsv(
        persistence_path,
        {
            "case_id",
            "branch_id",
            "edge_type",
            "episodes",
            "mean_persistence_ps",
            "median_persistence_ps",
            "p95_persistence_ps",
        },
    )
    input_records.extend(
        _source_records(
            "water_structure",
            case_id,
            [results / "summary.json", frame_path, residence_path, persistence_path],
        )
    )
    if any(row["case_id"] != case_id for row in frames + residence + persistence):
        raise ValueError(f"{results}: case_id does not match source contract")

    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in frames:
        by_group[(row["branch_id"], row["region"])].append(row)
    observed_branches = {branch for branch, _ in by_group}
    if observed_branches != expected_branches:
        raise ValueError(
            f"{results}: branch coverage {sorted(observed_branches)} != "
            f"{sorted(expected_branches)}"
        )
    residence_by_key = {(row["branch_id"], row["region"]): row for row in residence}
    persistence_by_key = {(row["branch_id"], row["edge_type"]): row for row in persistence}
    summary_region = str(source.get("summary_region", "all"))
    result: list[dict[str, object]] = []
    branch_summary: dict[tuple[str, str], dict[str, object]] = {}
    for (branch_id, region), group in sorted(by_group.items()):
        group.sort(key=lambda row: (_int(row["step"]), _float(row["time_ps"])))
        steps = [_int(row["step"]) for row in group]
        if len(steps) != len(set(steps)):
            raise ValueError(f"{results}: duplicate frame for {branch_id}/{region}")
        if expected_frames is not None and len(group) != expected_frames:
            raise ValueError(
                f"{results}: {branch_id}/{region} has {len(group)} frames, "
                f"expected {expected_frames}"
            )
        nonempty = [row for row in group if _float(row["water_count"]) > 0.0]
        qtet_values = [_float(row["mean_q_tet"]) for row in nonempty]
        residence_row = residence_by_key.get((branch_id, region))
        water_surface = persistence_by_key.get((branch_id, "water_surface"))
        water_water = persistence_by_key.get((branch_id, "water_water"))

        def edge_per_water(row: Mapping[str, str], field: str, factor: float) -> float:
            count = _float(row["water_count"])
            return factor * _float(row[field]) / count if count > 0.0 else math.nan

        aggregate: dict[str, object] = {
            "case_id": case_id,
            "branch_id": branch_id,
            "direction": group[0]["direction"],
            "region": region,
            "frames": len(group),
            "time_start_ps": _float(group[0]["time_ps"]),
            "time_end_ps": _float(group[-1]["time_ps"]),
            "mean_water_count": _mean(_float(row["water_count"]) for row in group),
            "mean_water_fraction": _mean(_float(row["water_fraction"]) for row in group),
            "mean_physical_largest_component_fraction": _mean(
                _float(row["physical_largest_component_fraction"]) for row in group
            ),
            "mean_q_tet": _mean(qtet_values),
            "q_tet_valid_fraction": (
                sum(math.isfinite(value) for value in qtet_values) / len(qtet_values)
                if qtet_values
                else math.nan
            ),
            "mean_lsi_A2": _mean(_float(row["mean_lsi_A2"]) for row in nonempty),
            "mean_oo_coordination": _mean(_float(row["mean_oo_coordination"]) for row in nonempty),
            "mean_h_coordination_defect_fraction": _mean(
                _float(row["h_coordination_defect_fraction"]) for row in nonempty
            ),
            "mean_water_water_hbond_degree": _mean(
                edge_per_water(row, "water_water_hbond_edges", 2.0) for row in group
            ),
            "mean_surface_hbond_per_water": _mean(
                edge_per_water(row, "water_surface_hbond_edges", 1.0) for row in group
            ),
            "mean_hbond_largest_component_fraction": _mean(
                _float(row["hbond_largest_component_fraction"]) for row in nonempty
            ),
            "mean_water_water_edge_turnover": _mean(
                _float(row["water_water_edge_turnover"]) for row in group
            ),
            "mean_water_surface_edge_turnover": _mean(
                _float(row["water_surface_edge_turnover"]) for row in group
            ),
            "residence_episodes": (_int(residence_row["episodes"]) if residence_row else 0),
            "mean_residence_ps": (
                _float(residence_row["mean_residence_ps"]) if residence_row else math.nan
            ),
            "median_residence_ps": (
                _float(residence_row["median_residence_ps"]) if residence_row else math.nan
            ),
            "p95_residence_ps": (
                _float(residence_row["p95_residence_ps"]) if residence_row else math.nan
            ),
            "surface_hbond_episodes": (_int(water_surface["episodes"]) if water_surface else 0),
            "surface_hbond_mean_persistence_ps": (
                _float(water_surface["mean_persistence_ps"]) if water_surface else math.nan
            ),
            "surface_hbond_median_persistence_ps": (
                _float(water_surface["median_persistence_ps"]) if water_surface else math.nan
            ),
            "water_hbond_mean_persistence_ps": (
                _float(water_water["mean_persistence_ps"]) if water_water else math.nan
            ),
        }
        result.append(aggregate)
        if region == summary_region:
            branch_summary[(case_id, branch_id)] = aggregate
    if {key[1] for key in branch_summary} != expected_branches:
        raise ValueError(f"{results}: missing summary region {summary_region!r}")
    return result, branch_summary


def _aggregate_event_source(
    source: dict[str, object],
    contract_path: Path,
    expected_branches: set[str],
    input_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    case_id = str(source.get("case_id", ""))
    if not case_id:
        raise ValueError("events source requires case_id")
    results = _resolve(contract_path, str(source["path"]))
    _verify_result_directory(results)
    event_path = results / "events.tsv"
    species_path = results / "frame_species.tsv"
    identity_path = results / "atom_identity.tsv"
    motion_path = results / "motion_event_summary.tsv"
    events = _read_tsv(event_path, EVENT_REQUIRED)
    species = _read_tsv(species_path, FRAME_SPECIES_REQUIRED)
    identity = _read_tsv(identity_path, ATOM_IDENTITY_REQUIRED)
    motion = _read_tsv(motion_path, MOTION_EVENT_REQUIRED)
    input_records.extend(
        _source_records(
            "events",
            case_id,
            [results / "summary.json", event_path, species_path, identity_path, motion_path],
        )
    )
    if any(row["case_id"] != case_id for row in events + species + identity + motion):
        raise ValueError(f"{results}: case_id does not match source contract")
    expected_carbon = source.get("expected_carbon_owned_H")
    allowed_carbon_raw = source.get("allowed_carbon_owned_H")
    if expected_carbon is not None and allowed_carbon_raw is not None:
        raise ValueError(
            "events source cannot define both expected_carbon_owned_H "
            "and allowed_carbon_owned_H"
        )
    if allowed_carbon_raw is not None:
        if not isinstance(allowed_carbon_raw, list) or not allowed_carbon_raw:
            raise ValueError("allowed_carbon_owned_H must be a non-empty list")
        allowed_carbon = {_int(value, -1) for value in allowed_carbon_raw}
        if any(value < 0 for value in allowed_carbon):
            raise ValueError("allowed_carbon_owned_H values must be non-negative")
    elif expected_carbon is not None:
        allowed_carbon = {int(expected_carbon)}
    else:
        allowed_carbon = None
    max_unassigned = _int(source.get("max_unassigned_H", 0))
    max_nonzero_iz = _int(source.get("max_nonzero_iz", 0))
    minimum_window_before_ps = _float(source.get("minimum_window_before_ps", 20.0))
    minimum_window_after_ps = _float(source.get("minimum_window_after_ps", 20.0))
    require_species_returned = _bool(source.get("require_species_returned", False))
    require_proton_pool_conservation = _bool(
        source.get("require_proton_pool_conservation", False)
    )
    terminal_limits = {
        "terminal_O_solution": source.get("max_terminal_O_solution"),
        "terminal_OH_solution": source.get("max_terminal_OH_solution"),
        "terminal_OH4plus_solution": source.get("max_terminal_OH4plus_solution"),
        "terminal_unassigned_H": source.get("max_terminal_unassigned_H", max_unassigned),
    }
    for row in events:
        before = _float(row["anchor_time_ps"]) - _float(row["window_start_ps"])
        after = _float(row["window_end_ps"]) - _float(row["anchor_time_ps"])
        if before + 1.0e-9 < minimum_window_before_ps:
            raise ValueError(f"{event_path}: event window is shorter than required before anchor")
        if after + 1.0e-9 < minimum_window_after_ps:
            raise ValueError(f"{event_path}: event window is shorter than required after anchor")
        for field, limit in terminal_limits.items():
            if limit is not None and _int(row[field], -1) > _int(limit):
                raise ValueError(f"{event_path}: {field} exceeds contract at {row['event_id']}")
        if _int(row["nonzero_iz_max"], -1) > max_nonzero_iz:
            raise ValueError(f"{event_path}: nonzero Z image exceeds contract")
        if require_proton_pool_conservation and _int(row["proton_pool_first"], -1) != _int(
            row["proton_pool_last"], -2
        ):
            raise ValueError(f"{event_path}: proton pool changed at {row['event_id']}")
        if require_species_returned and not _bool(row["species_returned"]):
            raise ValueError(f"{event_path}: species did not return at {row['event_id']}")
    for row in species:
        if expected_carbon is not None and _int(row["carbon_owned_H"], -1) != int(expected_carbon):
            raise ValueError(
                f"{species_path}: carbon_owned_H mismatch at "
                f"{row['case_id']}/{row['branch_id']}"
            )
        if _int(row["unassigned_H"], -1) > max_unassigned:
            raise ValueError(f"{species_path}: unassigned_H exceeds contract")
        if _int(row["nonzero_iz"], -1) > max_nonzero_iz:
            raise ValueError(f"{species_path}: nonzero Z image exceeds contract")
    for row in identity:
        coordination = _int(row["oo_coordination"], -1)
        qtet = _float(row["q_tet"])
        if 0 <= coordination < 4 and math.isfinite(qtet):
            raise ValueError(f"{identity_path}: finite q_tet with oo_coordination={coordination}")

    events_by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    species_by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    identity_by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    motion_by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rows, target in (
        (events, events_by_branch),
        (species, species_by_branch),
        (identity, identity_by_branch),
        (motion, motion_by_branch),
    ):
        for row in rows:
            target[row["branch_id"]].append(row)

    result = []
    for branch_id in sorted(expected_branches):
        branch_events = events_by_branch[branch_id]
        branch_species = species_by_branch[branch_id]
        branch_identity = identity_by_branch[branch_id]
        branch_motion = motion_by_branch[branch_id]
        types = [
            event_type
            for row in branch_events
            for event_type in row["event_types"].split(",")
            if event_type
        ]
        vector_changes = []
        speed_changes = []
        for row in branch_motion:
            vx_pre, vx_post = _float(row["vx_pre_mps"]), _float(row["vx_post_mps"])
            vy_pre, vy_post = _float(row["vy_pre_mps"]), _float(row["vy_post_mps"])
            if all(math.isfinite(value) for value in (vx_pre, vx_post, vy_pre, vy_post)):
                vector_changes.append(math.hypot(vx_post - vx_pre, vy_post - vy_pre))
                speed_changes.append(math.hypot(vx_post, vy_post) - math.hypot(vx_pre, vy_pre))
        result.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "event_count": len(branch_events),
                "species_geometry_events": types.count("species_geometry"),
                "high_z_events": types.count("high_z"),
                "wall_approach_events": types.count("wall_approach"),
                "z_image_events": types.count("z_image"),
                "species_returned_events": sum(
                    _bool(row["species_returned"]) for row in branch_events
                ),
                "species_not_returned_events": sum(
                    not _bool(row["species_returned"]) for row in branch_events
                ),
                "terminal_O_solution_events": sum(
                    _int(row["terminal_O_solution"]) > 0 for row in branch_events
                ),
                "terminal_OH_solution_events": sum(
                    _int(row["terminal_OH_solution"]) > 0 for row in branch_events
                ),
                "terminal_OH4plus_solution_events": sum(
                    _int(row["terminal_OH4plus_solution"]) > 0 for row in branch_events
                ),
                "minimum_window_before_ps": _minimum(
                    _float(row["anchor_time_ps"]) - _float(row["window_start_ps"])
                    for row in branch_events
                ),
                "minimum_window_after_ps": _minimum(
                    _float(row["window_end_ps"]) - _float(row["anchor_time_ps"])
                    for row in branch_events
                ),
                "maximum_absolute_proton_pool_change": max(
                    (
                        abs(
                            _int(row["proton_pool_last"])
                            - _int(row["proton_pool_first"])
                        )
                        for row in branch_events
                    ),
                    default=0,
                ),
                "proton_pool_change_events": sum(
                    _int(row["proton_pool_last"]) != _int(row["proton_pool_first"])
                    for row in branch_events
                ),
                "tracked_intact_water_events": sum(
                    _bool(row["tracked_intact_water"]) for row in branch_events
                ),
                "tracked_identity_incomplete_events": sum(
                    bool(str(row["tracked_oxygen_ids"]).strip())
                    and not _bool(row["tracked_identity_complete"])
                    for row in branch_events
                ),
                "tracked_returned_below_high_z_events": sum(
                    _bool(row["tracked_returned_below_high_z"]) for row in branch_events
                ),
                "minimum_tracked_sharing_delta_A": _minimum(
                    _float(row["tracked_min_sharing_delta_A"]) for row in branch_events
                ),
                "maximum_terminal_unassigned_H": max(
                    (_int(row["terminal_unassigned_H"]) for row in branch_events),
                    default=0,
                ),
                "maximum_nonzero_iz": max(
                    (_int(row["nonzero_iz_max"]) for row in branch_events), default=0
                ),
                "carbon_owned_H_min": min(
                    (_int(row["carbon_owned_H"]) for row in branch_species),
                    default=0,
                ),
                "carbon_owned_H_max": max(
                    (_int(row["carbon_owned_H"]) for row in branch_species),
                    default=0,
                ),
                "frame_unassigned_H_max": max(
                    (_int(row["unassigned_H"]) for row in branch_species), default=0
                ),
                "atom_identity_rows": len(branch_identity),
                "low_coordination_q_tet_violations": sum(
                    0 <= _int(row["oo_coordination"], -1) < 4
                    and math.isfinite(_float(row["q_tet"]))
                    for row in branch_identity
                ),
                "mean_event_vector_change_mps": _mean(vector_changes),
                "mean_event_speed_change_mps": _mean(speed_changes),
                "minimum_top_clearance_A": _minimum(
                    _float(row["minimum_top_clearance_A"]) for row in branch_motion
                ),
                "wall_motion_events": sum(_int(row["wall_samples"]) > 0 for row in branch_motion),
            }
        )
    return result


def _aggregate_wall_motion_source(
    source: dict[str, object],
    contract_path: Path,
    expected_cases: set[str],
    expected_branches: set[str],
    input_records: list[dict[str, object]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
]:
    results = _resolve(contract_path, str(source["path"]))
    _verify_result_directory(results)
    group_path = results / "group_summary.tsv"
    matched_path = results / "matched_summary.tsv"
    event_path = results / "wall_event_matched_contrasts.tsv"
    group_rows = _read_tsv(group_path, WALL_GROUP_REQUIRED)
    matched_rows = _read_tsv(matched_path, WALL_MATCHED_REQUIRED)
    event_rows = _read_tsv(event_path, WALL_EVENT_REQUIRED)
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_records.extend(
        _source_records(
            "wall_motion",
            "",
            [summary_path, group_path, matched_path, event_path],
        )
    )
    for row in event_rows:
        if row["case_id"] not in expected_cases or row["branch_id"] not in expected_branches:
            raise ValueError(f"{event_path}: event falls outside contract case/branch coverage")
    if _int(summary.get("wall_associated_events"), -1) != len(event_rows):
        raise ValueError(f"{summary_path}: wall-associated event count mismatch")
    return group_rows, matched_rows, event_rows, summary


def _aggregate_layered_transport_source(
    source: dict[str, object],
    contract_path: Path,
    expected_branches: set[str],
    input_records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    case_id = str(source.get("case_id", ""))
    if not case_id:
        raise ValueError("layered_transport source requires case_id")
    results = _resolve(contract_path, str(source["path"]))
    _verify_result_directory(results)
    response_path = results / "layer_response_summary.tsv"
    density_path = results / "density_modes.tsv"
    exchange_path = results / "layer_exchange.tsv"
    residence_path = results / "residence_summary.tsv"
    response = _read_tsv(response_path, LAYER_RESPONSE_REQUIRED)
    density = _read_tsv(density_path, LAYER_DENSITY_REQUIRED)
    exchange = _read_tsv(exchange_path, LAYER_EXCHANGE_REQUIRED)
    residence = _read_tsv(residence_path, LAYER_RESIDENCE_REQUIRED)
    input_records.extend(
        _source_records(
            "layered_transport",
            case_id,
            [
                results / "summary.json",
                response_path,
                density_path,
                exchange_path,
                residence_path,
            ],
        )
    )
    for rows in (response, density, exchange, residence):
        if any(row["case_id"] != case_id for row in rows):
            raise ValueError(f"{results}: case_id does not match layered source contract")
    if {row["branch_id"] for row in response} != expected_branches:
        raise ValueError(f"{response_path}: branch coverage does not match contract")

    density_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in density:
        density_groups[
            (row["branch_id"], row["layer_index"], row["mode_x"], row["mode_y"])
        ].append(row)
    density_summary: list[dict[str, object]] = []
    for (branch_id, layer_index, mode_x, mode_y), rows in sorted(density_groups.items()):
        amplitudes = [_float(row["amplitude"]) for row in rows]
        phases = [_float(row["phase_rad"]) for row in rows]
        finite_phases = [phase for phase in phases if math.isfinite(phase)]
        cosine = _mean(math.cos(phase) for phase in finite_phases)
        sine = _mean(math.sin(phase) for phase in finite_phases)
        density_summary.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": rows[0]["direction"],
                "layer_index": _int(layer_index),
                "mode_x": _int(mode_x),
                "mode_y": _int(mode_y),
                "frames": len(rows),
                "valid_amplitude_frames": sum(math.isfinite(value) for value in amplitudes),
                "mean_amplitude": _mean(amplitudes),
                "p95_amplitude": _quantile(amplitudes, 0.95),
                "phase_mean_rad": (
                    math.atan2(sine, cosine)
                    if math.isfinite(cosine) and math.isfinite(sine)
                    else math.nan
                ),
                "phase_resultant_length": (
                    math.hypot(cosine, sine)
                    if math.isfinite(cosine) and math.isfinite(sine)
                    else math.nan
                ),
            }
        )

    exchange_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in exchange:
        exchange_groups[(row["branch_id"], row["from_layer"], row["to_layer"])].append(row)
    exchange_summary: list[dict[str, object]] = []
    for (branch_id, from_layer, to_layer), rows in sorted(exchange_groups.items()):
        exchange_summary.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": rows[0]["direction"],
                "from_layer": _int(from_layer),
                "to_layer": _int(to_layer),
                "samples": len(rows),
                "total_molecule_count": sum(_int(row["molecule_count"]) for row in rows),
                "mean_rate_per_ps": _mean(_float(row["rate_per_ps"]) for row in rows),
                "p95_rate_per_ps": _quantile(
                    (_float(row["rate_per_ps"]) for row in rows), 0.95
                ),
            }
        )
    return {
        "response": [dict(row) for row in response],
        "density": density_summary,
        "exchange": exchange_summary,
        "residence": [dict(row) for row in residence],
    }
