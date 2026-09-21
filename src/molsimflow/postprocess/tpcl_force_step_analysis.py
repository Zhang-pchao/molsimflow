"""Paired high-frequency TPCL force-step mechanism analysis.

The analysis is deliberately ordered: kinematic events are selected without
using hydrogen-bond information, then surface anchoring and water-network
turnover are evaluated at those frozen event times and matched non-events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import (
    box_lengths,
    iter_lammps_dump_records,
    minimum_image_vectors,
)
from molsimflow.postprocess.constant_force_species_timeseries import (
    _frame_arrays,
    assign_hydrogen_parents,
    identify_fixed_carbon_hydrogen_ids,
    read_model_arrays,
    read_type_symbols,
)


@dataclass(frozen=True)
class RunSpec:
    case_id: str
    branch_id: str
    direction: str
    run_dir: Path
    model_data: Path
    top_surface_ids: frozenset[int]
    substrate_atoms: int
    start_step: int
    timestep_fs: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        key, separator, value = raw.partition("=")
        if not separator:
            raise ValueError(f"{path}: invalid environment row {raw!r}")
        values[key.strip()] = value.strip()
    return values


def discover_runs(package_root: Path) -> list[RunSpec]:
    """Resolve exactly the six production runs recorded by the package."""

    root = Path(package_root).resolve()
    submission = root / "04_jobs" / "SUBMISSION.tsv"
    with submission.open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle, delimiter="\t") if row["kind"] == "production"]
    if len(records) != 6:
        raise ValueError(f"expected six production records, found {len(records)}")
    runs: list[RunSpec] = []
    for row in records:
        case_id = row["case_id"]
        branch_id = row["branch_id"]
        case_root = root / "03_cases" / case_id / branch_id
        env = _read_env(case_root / "CASE.env")
        run_dir = case_root / "run_100ps" / row["job_id"]
        validation = json.loads((run_dir / "VALIDATION.json").read_text(encoding="utf-8"))
        if validation.get("status") != "PASS" or int(validation["end_step"]) != 36_400_000:
            raise ValueError(f"{case_id}/{branch_id}: production validation did not pass")
        run_result = _read_env(run_dir / "RUN-RESULT.txt")
        if run_result.get("status") != "PASS" or run_result.get("end_step") != "36400000":
            raise ValueError(f"{case_id}/{branch_id}: run result did not pass")
        top_ids = frozenset(
            int(value)
            for value in (root / "02_parents" / case_id / "top_surface.ids")
            .read_text(encoding="utf-8")
            .split()
        )
        runs.append(
            RunSpec(
                case_id=case_id,
                branch_id=branch_id,
                direction=env["DRIVE_DIRECTION"].lower(),
                run_dir=run_dir,
                model_data=root / env["MODEL_DATA"],
                top_surface_ids=top_ids,
                substrate_atoms=int(env["NSUB"]),
                start_step=int(env["START_STEP"]),
                timestep_fs=float(env["TIMESTEP_FS"]),
            )
        )
    keys = {(run.case_id, run.branch_id) for run in runs}
    expected = {
        (case_id, branch_id)
        for case_id in ("ch3_only", "mixed291")
        for branch_id in ("f0_shared", "f8e-5_x", "f8e-5_y")
    }
    if keys != expected:
        raise ValueError(f"production matrix differs from contract: {sorted(keys)}")
    return sorted(runs, key=lambda item: (item.case_id, item.branch_id))


def _unwrapped_frame_arrays(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids, types, coordinates = _frame_arrays(frame)
    columns = {name: index for index, name in enumerate(frame.atom_fields)}
    missing = {"ix", "iy", "iz"}.difference(columns)
    if missing:
        raise ValueError(f"step {frame.timestep}: missing image columns {sorted(missing)}")
    images = np.asarray(
        [[int(row[columns[name]]) for name in ("ix", "iy", "iz")] for row in frame.atom_rows],
        dtype=np.int64,
    )
    unwrapped = coordinates + images * box_lengths(frame.bounds)
    return ids, types, coordinates, unwrapped


def contact_line_metrics(
    *,
    ids: np.ndarray,
    types: np.ndarray,
    coordinates: np.ndarray,
    unwrapped: np.ndarray,
    top_surface_ids: frozenset[int],
    substrate_atoms: int,
    type_symbols: Mapping[int, str],
    contact_height_A: float,
) -> dict[str, float | int]:
    """Measure substrate-fixed water footprint edges in both lateral axes."""

    symbols = np.asarray([type_symbols[int(atom_type)] for atom_type in types])
    top_mask = np.asarray([int(atom_id) in top_surface_ids for atom_id in ids], dtype=bool)
    water_o_mask = (ids > substrate_atoms) & (symbols == "O")
    if np.count_nonzero(top_mask) < 20 or np.count_nonzero(water_o_mask) < 100:
        raise ValueError("insufficient top-surface or water-oxygen support")
    surface_plane = float(np.quantile(coordinates[top_mask, 2], 0.995))
    water_z = coordinates[water_o_mask, 2]
    contact = water_z <= surface_plane + contact_height_A
    minimum_contact = max(50, int(math.ceil(0.05 * len(water_z))))
    if np.count_nonzero(contact) < minimum_contact:
        order = np.argsort(water_z)
        contact = np.zeros(len(water_z), dtype=bool)
        contact[order[:minimum_contact]] = True
    substrate_xy = np.mean(unwrapped[top_mask, :2], axis=0)
    water_xy = unwrapped[water_o_mask, :2] - substrate_xy
    contact_xy = water_xy[contact]
    center_xy = np.mean(water_xy, axis=0)
    contact_center = np.mean(contact_xy, axis=0)
    radial = np.linalg.norm(contact_xy - contact_center, axis=1)
    return {
        "water_oxygen_count": int(np.count_nonzero(water_o_mask)),
        "contact_water_count": int(np.count_nonzero(contact)),
        "surface_plane_z_A": surface_plane,
        "center_x_A": float(center_xy[0]),
        "center_y_A": float(center_xy[1]),
        "trailing_x_A": float(np.quantile(contact_xy[:, 0], 0.02)),
        "leading_x_A": float(np.quantile(contact_xy[:, 0], 0.98)),
        "trailing_y_A": float(np.quantile(contact_xy[:, 1], 0.02)),
        "leading_y_A": float(np.quantile(contact_xy[:, 1], 0.98)),
        "contact_radius_q50_A": float(np.quantile(radial, 0.50)),
        "contact_radius_q90_A": float(np.quantile(radial, 0.90)),
    }


def extract_kinematics(run: RunSpec, contact_height_A: float) -> list[dict[str, object]]:
    type_symbols = read_type_symbols(run.model_data)
    rows: list[dict[str, object]] = []
    dump = run.run_dir / "tpcl_coordinates.lammpstrj.zst"
    for frame in iter_lammps_dump_records(dump):
        ids, types, coordinates, unwrapped = _unwrapped_frame_arrays(frame)
        metrics = contact_line_metrics(
            ids=ids,
            types=types,
            coordinates=coordinates,
            unwrapped=unwrapped,
            top_surface_ids=run.top_surface_ids,
            substrate_atoms=run.substrate_atoms,
            type_symbols=type_symbols,
            contact_height_A=contact_height_A,
        )
        rows.append(
            {
                "case_id": run.case_id,
                "branch_id": run.branch_id,
                "direction": run.direction,
                "step": frame.timestep,
                "time_ps": (frame.timestep - run.start_step) * run.timestep_fs / 1000.0,
                **metrics,
            }
        )
    return rows


def _window_mean(times: np.ndarray, values: np.ndarray, half_width_ps: float) -> np.ndarray:
    output = np.empty(len(values), dtype=float)
    left = 0
    right = 0
    running = 0.0
    for index, time in enumerate(times):
        while right < len(times) and times[right] <= time + half_width_ps:
            running += values[right]
            right += 1
        while left < len(times) and times[left] < time - half_width_ps:
            running -= values[left]
            left += 1
        output[index] = running / max(right - left, 1)
    return output


def paired_kinematics(
    forced: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    direction: str,
) -> list[dict[str, object]]:
    baseline_by_step = {int(row["step"]): row for row in baseline}
    if {int(row["step"]) for row in forced} != set(baseline_by_step):
        raise ValueError("forced and baseline coordinate schedules differ")
    axis = direction.lower()
    if axis not in {"x", "y"}:
        raise ValueError("paired kinematics requires x or y direction")
    rows: list[dict[str, object]] = []
    for row in forced:
        control = baseline_by_step[int(row["step"])]
        leading = float(row[f"leading_{axis}_A"]) - float(control[f"leading_{axis}_A"])
        trailing = float(row[f"trailing_{axis}_A"]) - float(control[f"trailing_{axis}_A"])
        center = float(row[f"center_{axis}_A"]) - float(control[f"center_{axis}_A"])
        rows.append(
            {
                "case_id": row["case_id"],
                "branch_id": row["branch_id"],
                "direction": axis,
                "step": int(row["step"]),
                "time_ps": float(row["time_ps"]),
                "leading_response_A": leading,
                "trailing_response_A": trailing,
                "edge_center_response_A": 0.5 * (leading + trailing),
                "edge_asymmetry_response_A": leading - trailing,
                "water_center_response_A": center,
                "contact_water_count": int(row["contact_water_count"]),
            }
        )
    times = np.asarray([float(row["time_ps"]) for row in rows])
    for field in ("leading_response_A", "trailing_response_A", "edge_center_response_A"):
        values = np.asarray([float(row[field]) for row in rows])
        smooth = _window_mean(times, values, 0.10)
        rate = np.gradient(smooth, times, edge_order=1)
        for row, smoothed, derivative in zip(rows, smooth, rate):
            row[field.replace("_A", "_smooth_A")] = float(smoothed)
            row[field.replace("_A", "_rate_A_per_ps")] = float(derivative)
    return rows


def select_kinematic_events(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_rate_A_per_ps: float = 0.02,
    maximum_events: int = 12,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    """Freeze advance episodes and matched non-events using kinematics only."""

    if len(rows) < 20:
        raise ValueError("too few paired kinematic frames")
    rates = np.asarray([float(row["edge_center_response_rate_A_per_ps"]) for row in rows])
    median = float(np.median(rates))
    mad = float(np.median(np.abs(rates - median)))
    threshold = max(minimum_rate_A_per_ps, median + 3.0 * 1.4826 * mad, float(np.quantile(rates, 0.95)))
    active = rates >= threshold
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for index, state in enumerate(active):
        if state and start is None:
            start = index
        gap_break = start is not None and (
            index == len(active) - 1
            or (not active[index + 1] and float(rows[index + 1]["time_ps"]) - float(rows[index]["time_ps"]) >= 0.0)
        )
        if gap_break:
            episodes.append((start, index))
            start = None
    candidates: list[tuple[float, int, int, int]] = []
    for left, right in episodes:
        peak = left + int(np.argmax(rates[left : right + 1]))
        candidates.append((float(rates[peak]), left, peak, right))
    candidates.sort(reverse=True)
    accepted: list[tuple[float, int, int, int]] = []
    for candidate in candidates:
        peak_time = float(rows[candidate[2]]["time_ps"])
        if all(abs(peak_time - float(rows[item[2]]["time_ps"])) >= 0.75 for item in accepted):
            accepted.append(candidate)
        if len(accepted) >= maximum_events:
            break
    accepted.sort(key=lambda item: int(rows[item[2]]["step"]))
    events: list[dict[str, object]] = []
    event_steps: list[int] = []
    for event_id, (_, left, peak, right) in enumerate(accepted, start=1):
        lead_rate = float(rows[peak]["leading_response_rate_A_per_ps"])
        trail_rate = float(rows[peak]["trailing_response_rate_A_per_ps"])
        if lead_rate >= 0.5 * threshold and trail_rate >= 0.5 * threshold:
            event_type = "coherent_advance"
        elif lead_rate >= trail_rate:
            event_type = "leading_edge_advance"
        else:
            event_type = "trailing_edge_release"
        events.append(
            {
                "case_id": rows[peak]["case_id"],
                "branch_id": rows[peak]["branch_id"],
                "direction": rows[peak]["direction"],
                "event_id": event_id,
                "selection_basis": "kinematics_only",
                "event_type": event_type,
                "start_step": int(rows[left]["step"]),
                "peak_step": int(rows[peak]["step"]),
                "stall_step": int(rows[min(right + 1, len(rows) - 1)]["step"]),
                "peak_time_ps": float(rows[peak]["time_ps"]),
                "peak_center_rate_A_per_ps": float(rates[peak]),
                "peak_leading_rate_A_per_ps": lead_rate,
                "peak_trailing_rate_A_per_ps": trail_rate,
                "pre_event_center_response_A": float(rows[max(0, left - 1)]["edge_center_response_smooth_A"]),
                "pre_event_asymmetry_response_A": float(rows[max(0, left - 1)]["edge_asymmetry_response_A"]),
            }
        )
        event_steps.append(int(rows[peak]["step"]))
    controls: list[dict[str, object]] = []
    used: set[int] = set()
    for event in events:
        target_time = float(event["peak_time_ps"])
        phase = 0 if target_time <= 20.0 else 1
        candidates_control: list[tuple[float, int]] = []
        for index, row in enumerate(rows):
            step = int(row["step"])
            time = float(row["time_ps"])
            if index < 2 or index >= len(rows) - 2 or step in used:
                continue
            if (0 if time <= 20.0 else 1) != phase:
                continue
            if any(abs(time - float(item["peak_time_ps"])) < 1.5 for item in events):
                continue
            if float(row["edge_center_response_rate_A_per_ps"]) >= 0.5 * threshold:
                continue
            center_gap = float(row["edge_center_response_smooth_A"]) - float(event["pre_event_center_response_A"])
            asym_gap = float(row["edge_asymmetry_response_A"]) - float(event["pre_event_asymmetry_response_A"])
            time_penalty = abs(time - target_time) / 100.0
            candidates_control.append((center_gap * center_gap + asym_gap * asym_gap + time_penalty, index))
        if not candidates_control:
            raise ValueError(f"no matched non-event for event {event['event_id']}")
        _, index = min(candidates_control)
        used.add(int(rows[index]["step"]))
        controls.append(
            {
                "case_id": event["case_id"],
                "branch_id": event["branch_id"],
                "direction": event["direction"],
                "event_id": event["event_id"],
                "control_step": int(rows[index]["step"]),
                "control_time_ps": float(rows[index]["time_ps"]),
                "matching_basis": "same_branch_same_output_phase_pre_event_center_and_asymmetry",
                "control_center_rate_A_per_ps": float(rows[index]["edge_center_response_rate_A_per_ps"]),
            }
        )
    return events, controls, {"rate_median": median, "rate_mad": mad, "event_threshold_A_per_ps": threshold}


def _donates(oh_vectors: np.ndarray, donor_to_acceptor: np.ndarray, angle_deg: float = 30.0) -> bool:
    if len(oh_vectors) == 0:
        return False
    target_norm = float(np.linalg.norm(donor_to_acceptor))
    if target_norm <= 0:
        return False
    norms = np.linalg.norm(oh_vectors, axis=1) * target_norm
    valid = norms > 0
    cosines = np.full(len(oh_vectors), -1.0)
    cosines[valid] = (oh_vectors[valid] @ donor_to_acceptor) / norms[valid]
    return bool(np.any(cosines >= math.cos(math.radians(angle_deg))))


def _pair_sets_for_frame(
    frame,
    run: RunSpec,
    type_symbols: Mapping[int, str],
    fixed_carbon_h: set[int],
    frozen_sioh_ids: frozenset[int] | None,
    contact_height_A: float,
    oo_cutoff_A: float,
) -> tuple[frozenset[int], set[tuple[int, int]], set[tuple[int, int]], int]:
    ids, types, coordinates = _frame_arrays(frame)
    symbols = np.asarray([type_symbols[int(atom_type)] for atom_type in types])
    index = {int(atom_id): position for position, atom_id in enumerate(ids)}
    h_ids, oxygen_ids, parents, _ = assign_hydrogen_parents(
        ids,
        types,
        coordinates,
        frame.bounds,
        type_symbols,
        1.35,
        fixed_carbon_h,
    )
    owner_h: dict[int, list[int]] = {}
    for hydrogen_id, parent_id in zip(h_ids, parents):
        if int(parent_id) >= 0:
            owner_h.setdefault(int(parent_id), []).append(int(hydrogen_id))
    top_oxygen = {
        int(atom_id)
        for atom_id, symbol in zip(ids, symbols)
        if int(atom_id) in run.top_surface_ids and symbol == "O"
    }
    current_sioh = frozenset(atom_id for atom_id in top_oxygen if len(owner_h.get(atom_id, ())) == 1)
    sioh_ids = current_sioh if frozen_sioh_ids is None else frozen_sioh_ids
    water_ids = np.asarray(
        [int(atom_id) for atom_id, symbol in zip(ids, symbols) if int(atom_id) > run.substrate_atoms and symbol == "O"],
        dtype=np.int64,
    )
    top_positions = np.asarray([coordinates[index[atom_id]] for atom_id in run.top_surface_ids if atom_id in index])
    surface_plane = float(np.quantile(top_positions[:, 2], 0.995))
    water_positions = np.asarray([coordinates[index[int(atom_id)]] for atom_id in water_ids])
    contact = water_positions[:, 2] <= surface_plane + contact_height_A
    if np.count_nonzero(contact) < 50:
        order = np.argsort(water_positions[:, 2])
        contact = np.zeros(len(water_positions), dtype=bool)
        contact[order[:50]] = True
    contact_ids = water_ids[contact]
    contact_positions = water_positions[contact]
    lengths = box_lengths(frame.bounds)
    center = contact_positions[0] + np.mean(
        minimum_image_vectors(contact_positions - contact_positions[0], lengths), axis=0
    )
    radial = np.linalg.norm(minimum_image_vectors(contact_positions - center, lengths)[:, :2], axis=1)
    tpcl_mask = radial >= np.quantile(radial, 0.80)
    tpcl_ids = contact_ids[tpcl_mask]
    tpcl_positions = contact_positions[tpcl_mask]

    def oh(atom_id: int) -> np.ndarray:
        origin = coordinates[index[atom_id]]
        hydrogens = owner_h.get(atom_id, ())
        if not hydrogens:
            return np.empty((0, 3), dtype=float)
        return minimum_image_vectors(
            np.asarray([coordinates[index[h_id]] - origin for h_id in hydrogens]), lengths
        )

    anchor_pairs: set[tuple[int, int]] = set()
    active_sites = [atom_id for atom_id in sorted(sioh_ids) if atom_id in index and len(owner_h.get(atom_id, ())) == 1]
    for water_id, water_position in zip(tpcl_ids, tpcl_positions):
        for site_id in active_sites:
            site_position = coordinates[index[site_id]]
            vector = minimum_image_vectors(water_position - site_position, lengths)
            if float(np.linalg.norm(vector)) > oo_cutoff_A:
                continue
            if _donates(oh(site_id), vector) or _donates(oh(int(water_id)), -vector):
                anchor_pairs.add((site_id, int(water_id)))
    network_pairs: set[tuple[int, int]] = set()
    for left in range(len(tpcl_ids)):
        for right in range(left + 1, len(tpcl_ids)):
            vector = minimum_image_vectors(tpcl_positions[right] - tpcl_positions[left], lengths)
            if float(np.linalg.norm(vector)) > oo_cutoff_A:
                continue
            left_id = int(tpcl_ids[left])
            right_id = int(tpcl_ids[right])
            if _donates(oh(left_id), vector) or _donates(oh(right_id), -vector):
                network_pairs.add((left_id, right_id))
    return current_sioh, anchor_pairs, network_pairs, len(tpcl_ids)


def extract_anchor_dynamics(
    run: RunSpec,
    contact_height_A: float,
    oo_cutoff_A: float,
) -> list[dict[str, object]]:
    type_symbols = read_type_symbols(run.model_data)
    fixed_carbon_h = identify_fixed_carbon_hydrogen_ids(run.model_data, type_symbols, 1.25)
    frozen_sioh: frozenset[int] | None = None
    previous_anchor: set[tuple[int, int]] = set()
    previous_network: set[tuple[int, int]] = set()
    rows: list[dict[str, object]] = []
    for frame_index, frame in enumerate(iter_lammps_dump_records(run.run_dir / "full_reference.lammpstrj.zst")):
        current_sioh, anchor, network, tpcl_count = _pair_sets_for_frame(
            frame,
            run,
            type_symbols,
            fixed_carbon_h,
            frozen_sioh,
            contact_height_A,
            oo_cutoff_A,
        )
        if frozen_sioh is None:
            frozen_sioh = current_sioh
        anchor_union = anchor | previous_anchor
        network_union = network | previous_network
        rows.append(
            {
                "case_id": run.case_id,
                "branch_id": run.branch_id,
                "direction": run.direction,
                "step": frame.timestep,
                "time_ps": (frame.timestep - run.start_step) * run.timestep_fs / 1000.0,
                "frozen_surface_sioh_count": len(frozen_sioh),
                "current_surface_sioh_count": len(current_sioh),
                "tpcl_water_count": tpcl_count,
                "surface_anchor_pair_count": len(anchor),
                "surface_anchor_formed_count": len(anchor - previous_anchor) if frame_index else 0,
                "surface_anchor_broken_count": len(previous_anchor - anchor) if frame_index else 0,
                "surface_anchor_jaccard": len(anchor & previous_anchor) / len(anchor_union) if frame_index and anchor_union else math.nan,
                "water_network_pair_count": len(network),
                "water_network_formed_count": len(network - previous_network) if frame_index else 0,
                "water_network_broken_count": len(previous_network - network) if frame_index else 0,
                "water_network_jaccard": len(network & previous_network) / len(network_union) if frame_index and network_union else math.nan,
            }
        )
        previous_anchor = anchor
        previous_network = network
    return rows


def _nearest_row(rows: Sequence[Mapping[str, object]], step: int) -> Mapping[str, object]:
    return min(rows, key=lambda row: abs(int(row["step"]) - step))


def event_anchor_contrasts(
    events: Sequence[Mapping[str, object]],
    controls: Sequence[Mapping[str, object]],
    anchor_by_branch: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    window_steps: int = 2000,
) -> list[dict[str, object]]:
    control_by_key = {(str(row["case_id"]), str(row["branch_id"]), int(row["event_id"])): row for row in controls}
    output: list[dict[str, object]] = []
    metrics = (
        "surface_anchor_pair_count",
        "surface_anchor_formed_count",
        "surface_anchor_broken_count",
        "water_network_pair_count",
        "water_network_formed_count",
        "water_network_broken_count",
    )
    for event in events:
        key = (str(event["case_id"]), str(event["branch_id"]))
        control = control_by_key[(*key, int(event["event_id"]))]
        series = anchor_by_branch[key]
        event_step = int(event["peak_step"])
        control_step = int(control["control_step"])
        event_pre = _nearest_row(series, event_step - window_steps)
        event_post = _nearest_row(series, event_step + window_steps)
        control_pre = _nearest_row(series, control_step - window_steps)
        control_post = _nearest_row(series, control_step + window_steps)
        row: dict[str, object] = {
            "case_id": key[0],
            "branch_id": key[1],
            "direction": event["direction"],
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "event_step": event_step,
            "control_step": control_step,
            "window_ps": window_steps * 0.5 / 1000.0,
            "selection_basis": "kinematics_then_frozen_anchor_test",
        }
        for metric in metrics:
            event_delta = float(event_post[metric]) - float(event_pre[metric])
            control_delta = float(control_post[metric]) - float(control_pre[metric])
            row[f"event_delta_{metric}"] = event_delta
            row[f"control_delta_{metric}"] = control_delta
            row[f"event_minus_control_delta_{metric}"] = event_delta - control_delta
        event_turnover = float(event_post["surface_anchor_formed_count"]) + float(event_post["surface_anchor_broken_count"])
        control_turnover = float(control_post["surface_anchor_formed_count"]) + float(control_post["surface_anchor_broken_count"])
        network_event_turnover = float(event_post["water_network_formed_count"]) + float(event_post["water_network_broken_count"])
        network_control_turnover = float(control_post["water_network_formed_count"]) + float(control_post["water_network_broken_count"])
        row["event_minus_control_surface_anchor_turnover"] = event_turnover - control_turnover
        row["event_minus_control_water_network_turnover"] = network_event_turnover - network_control_turnover
        output.append(row)
    return output


def _read_numeric_table(path: Path) -> tuple[list[str], np.ndarray]:
    header: list[str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# TimeStep"):
            header = raw[2:].split()
            break
    if header is None:
        raise ValueError(f"{path}: missing column header")
    values = np.loadtxt(path, comments="#")
    if values.ndim == 1:
        values = values.reshape((1, -1))
    if values.shape[1] != len(header) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: invalid numeric table")
    return header, values


def extract_global_response(run: RunSpec, baseline: RunSpec) -> tuple[list[dict[str, object]], dict[str, object]]:
    motion_names, motion = _read_numeric_table(run.run_dir / "motion_energy_stress_0p01ps.dat")
    base_names, base = _read_numeric_table(baseline.run_dir / "motion_energy_stress_0p01ps.dat")
    force_names, force = _read_numeric_table(run.run_dir / "force_sums_0p01ps.dat")
    if motion_names != base_names or not np.array_equal(motion[:, 0], base[:, 0]) or not np.array_equal(motion[:, 0], force[:, 0]):
        raise ValueError(f"{run.case_id}/{run.branch_id}: global tables are not step-aligned")
    m = {name: index for index, name in enumerate(motion_names)}
    f = {name: index for index, name in enumerate(force_names)}
    axis = run.direction
    rows: list[dict[str, object]] = []
    for current, control, forces in zip(motion, base, force):
        rows.append(
            {
                "case_id": run.case_id,
                "branch_id": run.branch_id,
                "direction": axis,
                "step": int(current[0]),
                "time_ps": (current[0] - run.start_step) * run.timestep_fs / 1000.0,
                "paired_relative_displacement_A": current[m[f"v_d{axis}rel"]] - control[m[f"v_d{axis}rel"]],
                "paired_relative_velocity_A_per_ps": current[m[f"v_vrel{axis}"]] - control[m[f"v_vrel{axis}"]],
                "drive_work_eV": current[m["v_drivework"]],
                "drive_power_eV_per_ps": current[m["v_drivepower"]],
                "actual_force_eV_per_A": current[m[f"v_factual{axis}"]],
                "expected_force_eV_per_A": current[m[f"v_fexpected{axis}"]],
                "force_closure_eV_per_A": current[m[f"v_factual{axis}"]] - current[m[f"v_fexpected{axis}"]],
                "raw_water_force_eV_per_A": forces[f[f"c_FrawWater[{1 if axis == 'x' else 2}]" ]],
                "raw_substrate_force_eV_per_A": forces[f[f"c_FrawSub[{1 if axis == 'x' else 2}]" ]],
                "temperature_water_K": current[m["c_Tliq"]],
            }
        )
    time = np.asarray([float(row["time_ps"]) for row in rows])
    displacement = np.asarray([float(row["paired_relative_displacement_A"]) for row in rows])
    velocity = np.asarray([float(row["paired_relative_velocity_A_per_ps"]) for row in rows])
    post20 = time >= 20.0
    summary = {
        "case_id": run.case_id,
        "branch_id": run.branch_id,
        "direction": axis,
        "final_paired_relative_displacement_A": float(displacement[-1]),
        "mean_paired_relative_velocity_20_100ps_A_per_ps": float(np.mean(velocity[post20])),
        "linear_displacement_slope_20_100ps_A_per_ps": float(np.polyfit(time[post20], displacement[post20], 1)[0]),
        "maximum_absolute_force_closure_eV_per_A": float(max(abs(float(row["force_closure_eV_per_A"])) for row in rows)),
        "maximum_water_temperature_K": float(max(float(row["temperature_water_K"]) for row in rows)),
        "final_drive_work_eV": float(rows[-1]["drive_work_eV"]),
    }
    return rows, summary


def _plot_results(
    output_dir: Path,
    paired: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    contrasts: Sequence[Mapping[str, object]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "06_figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, ((case_id, branch_id), rows) in zip(axes.flat, sorted(paired.items())):
        time = [float(row["time_ps"]) for row in rows]
        axis.plot(time, [float(row["leading_response_smooth_A"]) for row in rows], label="Leading")
        axis.plot(time, [float(row["trailing_response_smooth_A"]) for row in rows], label="Trailing")
        axis.plot(time, [float(row["edge_center_response_smooth_A"]) for row in rows], label="Edge center", lw=1.8)
        axis.set_title(f"{case_id} / {branch_id}")
        axis.set_ylabel("Paired response (A)")
        axis.legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Time (ps)")
    figure.tight_layout()
    path = figures / "paired_edge_response.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    if contrasts:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        labels = [f"{row['case_id']}:{row['direction']}:{row['event_id']}" for row in contrasts]
        values = [float(row["event_minus_control_surface_anchor_turnover"]) for row in contrasts]
        colors = ["#D55E00" if str(row["case_id"]) == "mixed291" else "#0072B2" for row in contrasts]
        axis.bar(np.arange(len(values)), values, color=colors)
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        axis.set_ylabel("Event - control anchor turnover")
        figure.tight_layout()
        path = figures / "event_anchor_turnover_contrast.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def analyze_package(
    package_root: Path,
    output_dir: Path,
    *,
    contact_height_A: float = 5.0,
    oo_cutoff_A: float = 3.5,
) -> dict[str, object]:
    root = Path(package_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"immutable output already exists: {output}")
    for name in ("00_contract", "01_inputs", "02_kinematics", "03_force_energy", "04_events", "05_anchors", "06_figures", "07_review", "08_validation"):
        (output / name).mkdir(parents=True, exist_ok=False)
    runs = discover_runs(root)
    contract = {
        "analysis_order": "kinematics_first_then_anchor_and_hbond",
        "pairing": "same_surface_forced_minus_f0_shared",
        "contact_height_A": contact_height_A,
        "oo_cutoff_A": oo_cutoff_A,
        "event_controls": "same_branch_same_output_phase_matched_non_event",
        "independence_warning": "time blocks are descriptive and are not independent replicas",
        "causal_scope": "event association only; no causal free-energy or converged rate claim",
    }
    (output / "00_contract" / "ANALYSIS-CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    input_rows: list[dict[str, object]] = []
    for run in runs:
        for filename in ("RUN-RESULT.txt", "VALIDATION.json", "OUTPUT-SHA256SUMS", "tpcl_coordinates.lammpstrj.zst", "full_reference.lammpstrj.zst", "motion_energy_stress_0p01ps.dat", "force_sums_0p01ps.dat"):
            path = run.run_dir / filename
            input_rows.append({"case_id": run.case_id, "branch_id": run.branch_id, "path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path) if path.stat().st_size < 20_000_000 else "sealed_by_OUTPUT-SHA256SUMS"})
    _write_tsv(output / "01_inputs" / "INPUT-MANIFEST.tsv", input_rows)

    kinematics: dict[tuple[str, str], list[dict[str, object]]] = {}
    anchors: dict[tuple[str, str], list[dict[str, object]]] = {}
    run_by_key = {(run.case_id, run.branch_id): run for run in runs}
    for run in runs:
        key = (run.case_id, run.branch_id)
        kinematics[key] = extract_kinematics(run, contact_height_A)
        anchors[key] = extract_anchor_dynamics(run, contact_height_A, oo_cutoff_A)
        _write_tsv(output / "02_kinematics" / f"{run.case_id}__{run.branch_id}.tsv", kinematics[key])
        _write_tsv(output / "05_anchors" / f"{run.case_id}__{run.branch_id}.tsv", anchors[key])

    paired: dict[tuple[str, str], list[dict[str, object]]] = {}
    events: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    global_summary: list[dict[str, object]] = []
    for case_id in ("ch3_only", "mixed291"):
        baseline = run_by_key[(case_id, "f0_shared")]
        for branch_id in ("f8e-5_x", "f8e-5_y"):
            run = run_by_key[(case_id, branch_id)]
            key = (case_id, branch_id)
            paired[key] = paired_kinematics(kinematics[key], kinematics[(case_id, "f0_shared")], run.direction)
            selected, matched, diagnostic = select_kinematic_events(paired[key])
            events.extend(selected)
            controls.extend(matched)
            thresholds.append({"case_id": case_id, "branch_id": branch_id, **diagnostic})
            response, summary = extract_global_response(run, baseline)
            global_rows.extend(response)
            global_summary.append(summary)
            _write_tsv(output / "02_kinematics" / f"{case_id}__{branch_id}__paired.tsv", paired[key])
    _write_tsv(output / "03_force_energy" / "paired_global_response.tsv", global_rows)
    _write_tsv(output / "03_force_energy" / "response_summary.tsv", global_summary)
    _write_tsv(output / "04_events" / "kinematic_events.tsv", events)
    _write_tsv(output / "04_events" / "matched_non_events.tsv", controls)
    _write_tsv(output / "04_events" / "event_thresholds.tsv", thresholds)
    contrasts = event_anchor_contrasts(events, controls, anchors)
    _write_tsv(output / "05_anchors" / "event_anchor_contrasts.tsv", contrasts)
    figures = _plot_results(output, paired, contrasts)

    mixed = [row for row in contrasts if row["case_id"] == "mixed291"]
    ch3 = [row for row in contrasts if row["case_id"] == "ch3_only"]
    mixed_positive_anchor = sum(float(row["event_minus_control_surface_anchor_turnover"]) > 0 for row in mixed)
    mixed_positive_network = sum(float(row["event_minus_control_water_network_turnover"]) > 0 for row in mixed)
    ch3_anchor_nonzero = sum(abs(float(row["event_minus_control_surface_anchor_turnover"])) > 1.0e-12 for row in ch3)
    longitudinal_signs = sum(float(row["final_paired_relative_displacement_A"]) > 0 for row in global_summary)
    mechanism_status = (
        "HIGH_FREQUENCY_EVENT_ASSOCIATION_SUPPORTED_NOT_CAUSAL"
        if len(mixed) >= 4
        and mixed_positive_anchor / len(mixed) >= 0.6
        and mixed_positive_network / len(mixed) >= 0.5
        and ch3_anchor_nonzero == 0
        else "HIGH_FREQUENCY_TPCL_MECHANISM_NOT_ESTABLISHED"
    )
    summary = {
        "status": "PASS",
        "scheduler_and_input_gate": "PASS",
        "run_count": len(runs),
        "paired_branch_count": len(paired),
        "kinematic_event_count": len(events),
        "mixed291_event_count": len(mixed),
        "ch3_only_event_count": len(ch3),
        "mixed291_positive_anchor_turnover_contrasts": mixed_positive_anchor,
        "mixed291_positive_water_network_turnover_contrasts": mixed_positive_network,
        "ch3_only_nonzero_surface_anchor_contrasts": ch3_anchor_nonzero,
        "positive_final_longitudinal_displacements": longitudinal_signs,
        "longitudinal_branch_count": len(global_summary),
        "mechanism_status": mechanism_status,
        "scientific_scope": "single-window paired event association; not an independent-replica causal or rate estimate",
        "figure_count": len(figures),
    }
    (output / "08_validation" / "VALIDATION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# High-frequency TPCL mechanism review",
        "",
        f"- Validation: `{summary['status']}`",
        f"- Scientific decision: `{mechanism_status}`",
        f"- Kinematics-selected events: `{len(events)}` (`mixed291={len(mixed)}`, `ch3_only={len(ch3)}`)",
        f"- mixed291 event-minus-control anchor-turnover positives: `{mixed_positive_anchor}/{len(mixed)}`",
        f"- mixed291 event-minus-control water-network-turnover positives: `{mixed_positive_network}/{len(mixed)}`",
        f"- CH3-only nonzero surface-anchor contrasts: `{ch3_anchor_nonzero}`",
        f"- Positive final drive-aligned paired displacements: `{longitudinal_signs}/{len(global_summary)}`",
        "",
        "Events were selected from substrate-fixed leading/trailing-edge kinematics before any anchor or H-bond metric was inspected. Matched non-events come from the same branch and output phase. Time blocks are descriptive, not independent replicas.",
        "",
        "The result may support a high-frequency event association, but it does not by itself establish causality, a free-energy barrier, a converged event rate, or a transferable mobility tensor.",
    ]
    (output / "07_review" / "HIGH-FREQUENCY-TPCL-REVIEW.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashed: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "OUTPUT-SHA256SUMS":
            hashed.append(f"{_sha256(path)}  {path.relative_to(output)}")
    (output / "08_validation" / "OUTPUT-SHA256SUMS").write_text("\n".join(hashed) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-height-A", type=float, default=5.0)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze_package(
        args.package_root,
        args.output_dir,
        contact_height_A=args.contact_height_A,
        oo_cutoff_A=args.oo_cutoff_A,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
