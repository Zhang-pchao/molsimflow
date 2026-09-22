"""Qualify TPCL observables before designing new force-pulse simulations.

This module intentionally separates four questions that were previously mixed:
geometric edge motion, persistent-molecule slip, region membership turnover,
and hydrogen-bond turnover.  It also measures the operating envelope of the
existing event detector on real F0 noise and audits global momentum and
work/heat accounting.  None of these diagnostics is a causal mechanism test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, iter_lammps_dump_records, minimum_image_vectors, periodic_center
from molsimflow.postprocess.constant_force_species_timeseries import (
    _frame_arrays,
    assign_hydrogen_parents,
    identify_fixed_carbon_hydrogen_ids,
    read_type_symbols,
)
from molsimflow.postprocess.tpcl_force_step_analysis import (
    RunSpec,
    _read_numeric_table,
    _window_mean,
    discover_runs,
    select_kinematic_events,
)
from molsimflow.postprocess.v3_mechanism_io import parse_lammps_data_atoms, parse_lammps_data_masses


METAL_FORCE_TO_ACCELERATION_A_PER_PS2_PER_AMU = 9648.533215665327
MOTION_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "axis",
    "edge",
    "step",
    "time_ps",
    "sample_interval_ps",
    "contact_water_count",
    "edge_member_count",
    "persistent_member_count",
    "member_entered_count",
    "member_exited_count",
    "substrate_delta_A",
    "persistent_mean_delta_A",
    "persistent_median_delta_A",
    "persistent_mean_relative_delta_A",
    "persistent_median_relative_delta_A",
    "cumulative_persistent_mean_relative_slip_A",
)
NETWORK_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "axis",
    "edge",
    "step",
    "time_ps",
    "sample_interval_ps",
    "edge_member_count",
    "current_surface_sioh_count",
    "water_hbond_count",
    "water_hbond_formed_persistent_count",
    "water_hbond_broken_persistent_count",
    "water_hbond_formed_membership_count",
    "water_hbond_broken_membership_count",
    "surface_hbond_count",
    "surface_hbond_formed_persistent_count",
    "surface_hbond_broken_persistent_count",
    "surface_hbond_formed_membership_count",
    "surface_hbond_broken_membership_count",
    "sioh_surface_hbond_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    selected = list(fields) if fields is not None else (list(rows[0]) if rows else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, delimiter="\t", extrasaction="ignore")
        if selected:
            writer.writeheader()
            writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _normalized_coordinates(values: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = box_lengths(bounds)
    pseudo_z = max(1.0e5, 10.0 * lengths[2])
    box = np.asarray([lengths[0], lengths[1], pseudo_z])
    result = np.empty_like(values, dtype=float)
    result[:, 0] = (values[:, 0] - bounds[0, 0]) % lengths[0]
    result[:, 1] = (values[:, 1] - bounds[1, 0]) % lengths[1]
    result[:, 2] = values[:, 2] - bounds[2, 0] + 0.25 * pseudo_z
    return result, box


def periodic_pairs_within(
    sources: np.ndarray,
    targets: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> list[tuple[int, int]]:
    """Return source/target index pairs within a cutoff, periodic in x/y only."""

    from scipy.spatial import cKDTree

    if len(sources) == 0 or len(targets) == 0:
        return []
    normalized_sources, box = _normalized_coordinates(sources, bounds)
    normalized_targets, _ = _normalized_coordinates(targets, bounds)
    neighbors = cKDTree(normalized_targets, boxsize=box).query_ball_point(
        normalized_sources, r=cutoff_A
    )
    return [(left, right) for left, choices in enumerate(neighbors) for right in choices]


def classify_pair_changes(
    previous_pairs: set[tuple[int, int]],
    current_pairs: set[tuple[int, int]],
    previous_members: set[int],
    current_members: set[int],
    *,
    water_pair: bool,
) -> dict[str, int]:
    """Separate true pair changes from changes caused by region membership."""

    persistent = previous_members & current_members

    def supported(pair: tuple[int, int]) -> bool:
        return pair[0] in persistent and pair[1] in persistent if water_pair else pair[1] in persistent

    formed = current_pairs - previous_pairs
    broken = previous_pairs - current_pairs
    formed_persistent = sum(supported(pair) for pair in formed)
    broken_persistent = sum(supported(pair) for pair in broken)
    return {
        "formed_persistent": formed_persistent,
        "broken_persistent": broken_persistent,
        "formed_membership": len(formed) - formed_persistent,
        "broken_membership": len(broken) - broken_persistent,
    }


def _owner_hydrogens(
    ids: np.ndarray,
    types: np.ndarray,
    coordinates: np.ndarray,
    bounds: np.ndarray,
    type_symbols: Mapping[int, str],
    fixed_carbon_h: set[int],
) -> dict[int, tuple[int, ...]]:
    h_ids, _, parents, _ = assign_hydrogen_parents(
        ids,
        types,
        coordinates,
        bounds,
        type_symbols,
        1.35,
        fixed_carbon_h,
    )
    owners: dict[int, list[int]] = defaultdict(list)
    for hydrogen_id, parent_id in zip(h_ids, parents):
        if int(parent_id) >= 0:
            owners[int(parent_id)].append(int(hydrogen_id))
    return {owner: tuple(values) for owner, values in owners.items()}


def _donates(
    donor_id: int,
    acceptor_id: int,
    positions: Mapping[int, np.ndarray],
    owner_h: Mapping[int, tuple[int, ...]],
    lengths: np.ndarray,
    angle_deg: float = 30.0,
) -> bool:
    hydrogens = owner_h.get(donor_id, ())
    if not hydrogens:
        return False
    donor_to_acceptor = minimum_image_vectors(positions[acceptor_id] - positions[donor_id], lengths)
    target_norm = float(np.linalg.norm(donor_to_acceptor))
    if target_norm <= 0:
        return False
    threshold = math.cos(math.radians(angle_deg))
    for hydrogen_id in hydrogens:
        oh = minimum_image_vectors(positions[hydrogen_id] - positions[donor_id], lengths)
        norm = float(np.linalg.norm(oh)) * target_norm
        if norm > 0 and float(np.dot(oh, donor_to_acceptor) / norm) >= threshold:
            return True
    return False


def _edge_members(
    ids: np.ndarray,
    types: np.ndarray,
    coordinates: np.ndarray,
    bounds: np.ndarray,
    run: RunSpec,
    type_symbols: Mapping[int, str],
    axis: str,
    edge: str,
    contact_height_A: float,
    tail_fraction: float,
) -> tuple[set[int], int, dict[int, np.ndarray], set[int]]:
    symbols = np.asarray([type_symbols[int(atom_type)] for atom_type in types])
    positions = {int(atom_id): position for atom_id, position in zip(ids, coordinates)}
    top_mask = np.asarray([int(atom_id) in run.top_surface_ids for atom_id in ids], dtype=bool)
    water_mask = (ids > run.substrate_atoms) & (symbols == "O")
    top_positions = coordinates[top_mask]
    water_ids = ids[water_mask]
    water_positions = coordinates[water_mask]
    if len(top_positions) < 20 or len(water_positions) < 100:
        raise ValueError("insufficient surface or water support")
    surface_plane = float(np.quantile(top_positions[:, 2], 0.995))
    contact_mask = water_positions[:, 2] <= surface_plane + contact_height_A
    minimum = max(50, int(math.ceil(0.05 * len(water_positions))))
    if np.count_nonzero(contact_mask) < minimum:
        order = np.argsort(water_positions[:, 2])
        contact_mask = np.zeros(len(water_positions), dtype=bool)
        contact_mask[order[:minimum]] = True
    contact_ids = water_ids[contact_mask]
    contact_positions = water_positions[contact_mask]
    center = periodic_center(water_positions, bounds)
    local = minimum_image_vectors(contact_positions - center, box_lengths(bounds))
    component = local[:, 0 if axis == "x" else 1]
    count = max(5, int(math.ceil(tail_fraction * len(component))))
    order = np.argsort(component)
    chosen = order[-count:] if edge == "leading" else order[:count]
    top_oxygen = {
        int(atom_id)
        for atom_id, symbol in zip(ids, symbols)
        if int(atom_id) in run.top_surface_ids and symbol == "O"
    }
    return set(map(int, contact_ids[chosen])), int(len(contact_ids)), positions, top_oxygen


def _hbond_sets(
    members: set[int],
    top_oxygen: set[int],
    positions: Mapping[int, np.ndarray],
    owner_h: Mapping[int, tuple[int, ...]],
    bounds: np.ndarray,
    oo_cutoff_A: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[int]]:
    lengths = box_lengths(bounds)
    ordered_members = sorted(members)
    water_pairs: set[tuple[int, int]] = set()
    for left_index, left in enumerate(ordered_members):
        for right in ordered_members[left_index + 1 :]:
            vector = minimum_image_vectors(positions[right] - positions[left], lengths)
            if float(np.linalg.norm(vector)) <= oo_cutoff_A and (
                _donates(left, right, positions, owner_h, lengths)
                or _donates(right, left, positions, owner_h, lengths)
            ):
                water_pairs.add((left, right))

    ordered_sites = sorted(site for site in top_oxygen if site in positions)
    member_positions = np.asarray([positions[atom_id] for atom_id in ordered_members])
    site_positions = np.asarray([positions[atom_id] for atom_id in ordered_sites])
    surface_pairs: set[tuple[int, int]] = set()
    sioh_pairs: set[tuple[int, int]] = set()
    sioh_sites = {site for site in ordered_sites if len(owner_h.get(site, ())) == 1}
    for water_index, site_index in periodic_pairs_within(
        member_positions, site_positions, bounds, oo_cutoff_A
    ):
        water_id = ordered_members[water_index]
        site_id = ordered_sites[site_index]
        if _donates(site_id, water_id, positions, owner_h, lengths) or _donates(
            water_id, site_id, positions, owner_h, lengths
        ):
            pair = (site_id, water_id)
            surface_pairs.add(pair)
            if site_id in sioh_sites:
                sioh_pairs.add(pair)
    return water_pairs, surface_pairs, sioh_pairs, sioh_sites


def _update_lifetimes(
    active: dict[tuple[str, str, str], dict[tuple[int, int], tuple[float, float, int]]],
    completed: list[dict[str, object]],
    key: tuple[str, str, str],
    current: set[tuple[int, int]],
    time_ps: float,
    sample_interval_ps: float,
) -> None:
    tracked = active.setdefault(key, {})
    for pair in set(tracked).difference(current):
        start, last, samples = tracked.pop(pair)
        completed.append(
            {
                "axis": key[0],
                "edge": key[1],
                "pair_kind": key[2],
                "start_time_ps": start,
                "end_time_ps": last,
                "samples": samples,
                "continuous_lifetime_ps": samples * sample_interval_ps,
            }
        )
    for pair in current:
        if pair in tracked:
            start, _, samples = tracked[pair]
            tracked[pair] = (start, time_ps, samples + 1)
        else:
            tracked[pair] = (time_ps, time_ps, 1)


def extract_high_cadence_regions(
    run: RunSpec,
    *,
    window_ps: float,
    contact_height_A: float,
    edge_tail_fraction: float,
    oo_cutoff_A: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract motion and network observables from the 10-fs coordinate phase."""

    type_symbols = read_type_symbols(run.model_data)
    fixed_carbon_h = identify_fixed_carbon_hydrogen_ids(run.model_data, type_symbols, 1.25)
    maximum_timestep = run.start_step + int(round(window_ps * 1000.0 / run.timestep_fs))
    previous_members: dict[tuple[str, str], set[int]] = {}
    previous_positions: dict[int, np.ndarray] = {}
    previous_top_positions: dict[int, np.ndarray] = {}
    previous_water_pairs: dict[tuple[str, str], set[tuple[int, int]]] = {}
    previous_surface_pairs: dict[tuple[str, str], set[tuple[int, int]]] = {}
    cumulative_slip: dict[tuple[str, str], float] = defaultdict(float)
    rows: list[dict[str, object]] = []
    active_lifetimes: dict[tuple[str, str, str], dict[tuple[int, int], tuple[float, float, int]]] = {}
    completed_lifetimes: list[dict[str, object]] = []
    previous_time: float | None = None

    for frame in iter_lammps_dump_records(
        run.run_dir / "tpcl_coordinates.lammpstrj.zst", maximum_timestep
    ):
        ids, types, coordinates = _frame_arrays(frame)
        owner_h = _owner_hydrogens(
            ids, types, coordinates, frame.bounds, type_symbols, fixed_carbon_h
        )
        time_ps = (frame.timestep - run.start_step) * run.timestep_fs / 1000.0
        sample_interval = 0.0 if previous_time is None else time_ps - previous_time
        all_positions = {int(atom_id): position for atom_id, position in zip(ids, coordinates)}
        top_positions = {
            atom_id: all_positions[atom_id]
            for atom_id in run.top_surface_ids
            if atom_id in all_positions
        }
        lengths = box_lengths(frame.bounds)
        substrate_delta = np.zeros(2)
        if previous_top_positions:
            common_top = sorted(set(top_positions) & set(previous_top_positions))
            top_deltas = np.asarray(
                [
                    minimum_image_vectors(
                        top_positions[atom_id] - previous_top_positions[atom_id], lengths
                    )[:2]
                    for atom_id in common_top
                ]
            )
            substrate_delta = np.mean(top_deltas, axis=0)

        for axis in ("x", "y"):
            axis_index = 0 if axis == "x" else 1
            for edge in ("leading", "trailing"):
                key = (axis, edge)
                members, contact_count, positions, top_oxygen = _edge_members(
                    ids,
                    types,
                    coordinates,
                    frame.bounds,
                    run,
                    type_symbols,
                    axis,
                    edge,
                    contact_height_A,
                    edge_tail_fraction,
                )
                water_pairs, surface_pairs, sioh_pairs, sioh_sites = _hbond_sets(
                    members,
                    top_oxygen,
                    positions,
                    owner_h,
                    frame.bounds,
                    oo_cutoff_A,
                )
                old_members = previous_members.get(key, set())
                persistent = old_members & members
                relative_deltas = np.asarray(
                    [
                        minimum_image_vectors(
                            positions[atom_id] - previous_positions[atom_id], lengths
                        )[axis_index]
                        - substrate_delta[axis_index]
                        for atom_id in sorted(persistent)
                        if atom_id in previous_positions
                    ],
                    dtype=float,
                )
                mean_relative = float(np.mean(relative_deltas)) if len(relative_deltas) else math.nan
                median_relative = float(np.median(relative_deltas)) if len(relative_deltas) else math.nan
                mean_absolute = (
                    mean_relative + substrate_delta[axis_index]
                    if math.isfinite(mean_relative)
                    else math.nan
                )
                median_absolute = (
                    median_relative + substrate_delta[axis_index]
                    if math.isfinite(median_relative)
                    else math.nan
                )
                if math.isfinite(mean_relative):
                    cumulative_slip[key] += mean_relative
                water_changes = classify_pair_changes(
                    previous_water_pairs.get(key, set()),
                    water_pairs,
                    old_members,
                    members,
                    water_pair=True,
                )
                surface_changes = classify_pair_changes(
                    previous_surface_pairs.get(key, set()),
                    surface_pairs,
                    old_members,
                    members,
                    water_pair=False,
                )
                row = {
                    "case_id": run.case_id,
                    "branch_id": run.branch_id,
                    "direction": run.direction,
                    "axis": axis,
                    "edge": edge,
                    "step": frame.timestep,
                    "time_ps": time_ps,
                    "sample_interval_ps": sample_interval,
                    "contact_water_count": contact_count,
                    "edge_member_count": len(members),
                    "persistent_member_count": len(persistent),
                    "member_entered_count": len(members - old_members) if previous_time is not None else 0,
                    "member_exited_count": len(old_members - members) if previous_time is not None else 0,
                    "substrate_delta_A": float(substrate_delta[axis_index]),
                    "persistent_mean_delta_A": mean_absolute,
                    "persistent_median_delta_A": median_absolute,
                    "persistent_mean_relative_delta_A": mean_relative,
                    "persistent_median_relative_delta_A": median_relative,
                    "cumulative_persistent_mean_relative_slip_A": cumulative_slip[key],
                    "current_surface_sioh_count": len(sioh_sites),
                    "water_hbond_count": len(water_pairs),
                    "water_hbond_formed_persistent_count": water_changes["formed_persistent"] if previous_time is not None else 0,
                    "water_hbond_broken_persistent_count": water_changes["broken_persistent"] if previous_time is not None else 0,
                    "water_hbond_formed_membership_count": water_changes["formed_membership"] if previous_time is not None else 0,
                    "water_hbond_broken_membership_count": water_changes["broken_membership"] if previous_time is not None else 0,
                    "surface_hbond_count": len(surface_pairs),
                    "surface_hbond_formed_persistent_count": surface_changes["formed_persistent"] if previous_time is not None else 0,
                    "surface_hbond_broken_persistent_count": surface_changes["broken_persistent"] if previous_time is not None else 0,
                    "surface_hbond_formed_membership_count": surface_changes["formed_membership"] if previous_time is not None else 0,
                    "surface_hbond_broken_membership_count": surface_changes["broken_membership"] if previous_time is not None else 0,
                    "sioh_surface_hbond_count": len(sioh_pairs),
                }
                rows.append(row)
                if previous_time is not None:
                    _update_lifetimes(
                        active_lifetimes,
                        completed_lifetimes,
                        (axis, edge, "water_water"),
                        water_pairs,
                        time_ps,
                        sample_interval,
                    )
                    _update_lifetimes(
                        active_lifetimes,
                        completed_lifetimes,
                        (axis, edge, "surface_water"),
                        surface_pairs,
                        time_ps,
                        sample_interval,
                    )
                previous_members[key] = members
                previous_water_pairs[key] = water_pairs
                previous_surface_pairs[key] = surface_pairs
        previous_positions = all_positions
        previous_top_positions = top_positions
        previous_time = time_ps

    if previous_time is None:
        raise ValueError(f"{run.case_id}/{run.branch_id}: no coordinate frames")
    nominal_interval = min(
        float(row["sample_interval_ps"])
        for row in rows
        if float(row["sample_interval_ps"]) > 0
    )
    for key, tracked in active_lifetimes.items():
        for start, last, samples in tracked.values():
            completed_lifetimes.append(
                {
                    "axis": key[0],
                    "edge": key[1],
                    "pair_kind": key[2],
                    "start_time_ps": start,
                    "end_time_ps": last,
                    "samples": samples,
                    "continuous_lifetime_ps": samples * nominal_interval,
                }
            )
    for row in completed_lifetimes:
        row.update({"case_id": run.case_id, "branch_id": run.branch_id})
    return rows, completed_lifetimes


def summarize_regions(
    rows: Sequence[Mapping[str, object]],
    lifetimes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    lifetime_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["axis"]), str(row["edge"]))].append(row)
    for row in lifetimes:
        lifetime_grouped[(str(row["axis"]), str(row["edge"]), str(row["pair_kind"]))].append(
            float(row["continuous_lifetime_ps"])
        )
    output: list[dict[str, object]] = []
    rate_fields = (
        "member_entered_count",
        "member_exited_count",
        "water_hbond_formed_persistent_count",
        "water_hbond_broken_persistent_count",
        "water_hbond_formed_membership_count",
        "water_hbond_broken_membership_count",
        "surface_hbond_formed_persistent_count",
        "surface_hbond_broken_persistent_count",
        "surface_hbond_formed_membership_count",
        "surface_hbond_broken_membership_count",
    )
    for (axis, edge), sample in sorted(grouped.items()):
        duration = float(sample[-1]["time_ps"]) - float(sample[0]["time_ps"])
        result: dict[str, object] = {
            "case_id": sample[0]["case_id"],
            "branch_id": sample[0]["branch_id"],
            "direction": sample[0]["direction"],
            "axis": axis,
            "edge": edge,
            "frames": len(sample),
            "duration_ps": duration,
            "mean_edge_member_count": float(np.mean([float(row["edge_member_count"]) for row in sample])),
            "mean_water_hbond_count": float(np.mean([float(row["water_hbond_count"]) for row in sample])),
            "mean_surface_hbond_count": float(np.mean([float(row["surface_hbond_count"]) for row in sample])),
            "mean_sioh_surface_hbond_count": float(np.mean([float(row["sioh_surface_hbond_count"]) for row in sample])),
            "final_cumulative_persistent_mean_relative_slip_A": float(sample[-1]["cumulative_persistent_mean_relative_slip_A"]),
            "persistent_mean_relative_slip_rate_A_per_ps": float(sample[-1]["cumulative_persistent_mean_relative_slip_A"]) / duration,
        }
        for field in rate_fields:
            result[field.replace("_count", "_rate_per_ps")] = sum(float(row[field]) for row in sample) / duration
        for kind in ("water_water", "surface_water"):
            values = np.asarray(lifetime_grouped.get((axis, edge, kind), []), dtype=float)
            result[f"{kind}_lifetime_episode_count"] = len(values)
            result[f"{kind}_lifetime_q50_ps"] = float(np.quantile(values, 0.50)) if len(values) else math.nan
            result[f"{kind}_lifetime_q90_ps"] = float(np.quantile(values, 0.90)) if len(values) else math.nan
        output.append(result)
    return output


def paired_region_summary(summary_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["case_id"]), str(row["branch_id"]), str(row["axis"]), str(row["edge"])): row
        for row in summary_rows
    }
    metrics = (
        "persistent_mean_relative_slip_rate_A_per_ps",
        "mean_water_hbond_count",
        "mean_surface_hbond_count",
        "mean_sioh_surface_hbond_count",
        "water_hbond_formed_persistent_rate_per_ps",
        "water_hbond_broken_persistent_rate_per_ps",
        "water_hbond_formed_membership_rate_per_ps",
        "water_hbond_broken_membership_rate_per_ps",
        "surface_hbond_formed_persistent_rate_per_ps",
        "surface_hbond_broken_persistent_rate_per_ps",
        "surface_hbond_formed_membership_rate_per_ps",
        "surface_hbond_broken_membership_rate_per_ps",
    )
    output: list[dict[str, object]] = []
    for (case_id, branch_id, axis, edge), row in sorted(by_key.items()):
        if branch_id == "f0_shared" or branch_id != f"f8e-5_{axis}":
            continue
        baseline = by_key[(case_id, "f0_shared", axis, edge)]
        result: dict[str, object] = {
            "case_id": case_id,
            "branch_id": branch_id,
            "axis": axis,
            "edge": edge,
            "pairing": "forced_minus_same_surface_f0_shared_summary",
        }
        for metric in metrics:
            result[f"forced_{metric}"] = row[metric]
            result[f"f0_{metric}"] = baseline[metric]
            result[f"paired_delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        output.append(result)
    return output


def build_detector_trace(
    base_rows: Sequence[Mapping[str, str]],
    axis: str,
    *,
    amplitude_A: float,
    duration_ps: float,
    center_ps: float,
    mode: str,
) -> list[dict[str, object]]:
    times = np.asarray([float(row["time_ps"]) for row in base_rows])
    leading_raw = np.asarray([float(row[f"leading_{axis}_A"]) for row in base_rows])
    trailing_raw = np.asarray([float(row[f"trailing_{axis}_A"]) for row in base_rows])
    leading = leading_raw - _window_mean(times, leading_raw, 2.0)
    trailing = trailing_raw - _window_mean(times, trailing_raw, 2.0)
    scale = max(duration_ps / 4.0, 0.005)
    argument = np.clip(-(times - center_ps) / scale, -700.0, 700.0)
    step = amplitude_A / (1.0 + np.exp(argument))
    if mode == "coherent":
        leading = leading + step
        trailing = trailing + step
    elif mode == "leading_only":
        leading = leading + step
    elif mode == "trailing_only":
        trailing = trailing + step
    elif mode == "opposing":
        leading = leading + step
        trailing = trailing - step
    elif mode == "retreat":
        leading = leading - step
        trailing = trailing - step
    else:
        raise ValueError(f"unsupported injection mode: {mode}")
    center = 0.5 * (leading + trailing)
    leading_smooth = _window_mean(times, leading, 0.50)
    trailing_smooth = _window_mean(times, trailing, 0.50)
    center_smooth = _window_mean(times, center, 0.50)
    leading_rate = np.gradient(leading_smooth, times)
    trailing_rate = np.gradient(trailing_smooth, times)
    center_rate = np.gradient(center_smooth, times)
    rows: list[dict[str, object]] = []
    for index, (source, time) in enumerate(zip(base_rows, times)):
        rows.append(
            {
                "case_id": source["case_id"],
                "branch_id": "synthetic_injection",
                "direction": axis,
                "step": int(source["step"]),
                "time_ps": float(time),
                "leading_response_rate_A_per_ps": float(leading_rate[index]),
                "trailing_response_rate_A_per_ps": float(trailing_rate[index]),
                "edge_center_response_rate_A_per_ps": float(center_rate[index]),
                "edge_center_response_smooth_A": float(center_smooth[index]),
                "edge_asymmetry_response_A": float(leading[index] - trailing[index]),
            }
        )
    return rows


def qualify_detector(reference_analysis: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    amplitudes = (0.0, 0.25, 0.50, 1.0, 2.0)
    durations = (0.05, 0.10, 0.25, 0.50, 1.0)
    centers = (10.0, 35.0, 65.0, 90.0)
    modes = ("coherent", "leading_only", "trailing_only", "opposing", "retreat")
    for case_id in ("ch3_only", "mixed291"):
        base_rows = _read_tsv(reference_analysis / "02_kinematics" / f"{case_id}__f0_shared.tsv")
        for axis in ("x", "y"):
            for amplitude in amplitudes:
                for duration in durations:
                    for center in centers:
                        for mode in modes:
                            trace = build_detector_trace(
                                base_rows,
                                axis,
                                amplitude_A=amplitude,
                                duration_ps=duration,
                                center_ps=center,
                                mode=mode,
                            )
                            events, _, diagnostic = select_kinematic_events(trace)
                            tolerance = max(1.0, duration)
                            matched = [
                                event
                                for event in events
                                if abs(float(event["peak_time_ps"]) - center) <= tolerance
                            ]
                            trials.append(
                                {
                                    "case_id": case_id,
                                    "axis": axis,
                                    "mode": mode,
                                    "amplitude_A": amplitude,
                                    "duration_ps": duration,
                                    "center_ps": center,
                                    "expected_positive_center_advance": mode in {"coherent", "leading_only", "trailing_only"} and amplitude > 0,
                                    "recovered": bool(matched),
                                    "matched_peak_time_ps": float(matched[0]["peak_time_ps"]) if matched else math.nan,
                                    "detected_event_count": len(events),
                                    "false_positive_count": len(events) - len(matched),
                                    "event_threshold_A_per_ps": diagnostic["event_threshold_A_per_ps"],
                                }
                            )
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in trials:
        grouped[
            (
                row["case_id"],
                row["axis"],
                row["mode"],
                row["amplitude_A"],
                row["duration_ps"],
            )
        ].append(row)
    envelope: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        expected = bool(values[0]["expected_positive_center_advance"])
        envelope.append(
            {
                "case_id": key[0],
                "axis": key[1],
                "mode": key[2],
                "amplitude_A": key[3],
                "duration_ps": key[4],
                "trials": len(values),
                "expected_positive_center_advance": expected,
                "recovery_fraction": float(np.mean([bool(row["recovered"]) for row in values])),
                "mean_false_positive_count": float(np.mean([int(row["false_positive_count"]) for row in values])),
                "maximum_threshold_A_per_ps": max(float(row["event_threshold_A_per_ps"]) for row in values),
                "qualified_at_75pct_recovery": expected and np.mean([bool(row["recovered"]) for row in values]) >= 0.75,
            }
        )
    return trials, envelope


def read_complete_thermo(path: Path, required: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Read all compatible thermo blocks and keep the last copy of shared endpoints."""

    rows: dict[int, list[float]] = {}
    selected_header: list[str] | None = None
    active_header: list[str] | None = None
    active_indices: list[int] | None = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if fields and fields[0] == "Step":
            active_header = fields
            if set(required).issubset(fields):
                selected_header = list(required)
                active_indices = [fields.index(name) for name in required]
            else:
                active_indices = None
            continue
        if active_header is None or active_indices is None or len(fields) != len(active_header):
            continue
        try:
            values = [float(fields[index]) for index in active_indices]
        except ValueError:
            continue
        step = int(round(values[0]))
        rows[step] = values
    if selected_header is None or not rows:
        raise ValueError(f"{path}: no complete thermo block containing {tuple(required)}")
    ordered = np.asarray([rows[step] for step in sorted(rows)], dtype=float)
    return selected_header, ordered


def audit_global_balance(run: RunSpec) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    motion_names, motion = _read_numeric_table(run.run_dir / "motion_energy_stress_0p01ps.dat")
    force_names, forces = _read_numeric_table(run.run_dir / "force_sums_0p01ps.dat")
    thermo_names, thermo = read_complete_thermo(run.run_dir / "lmp.out", ("Step", "TotEng", "f_BATH"))
    motion_index = {name: index for index, name in enumerate(motion_names)}
    force_index = {name: index for index, name in enumerate(force_names)}
    thermo_index = {name: index for index, name in enumerate(thermo_names)}
    steps = motion[:, motion_index["TimeStep"]].astype(np.int64)
    if not np.array_equal(steps, forces[:, force_index["TimeStep"]].astype(np.int64)):
        raise ValueError(f"{run.case_id}/{run.branch_id}: motion/force steps differ")
    thermo_by_step = {int(row[thermo_index["Step"]]): row for row in thermo}
    if any(int(step) not in thermo_by_step for step in steps):
        raise ValueError(f"{run.case_id}/{run.branch_id}: thermo does not cover every table step")
    aligned_thermo = np.asarray([thermo_by_step[int(step)] for step in steps])
    time_ps = (steps - run.start_step) * run.timestep_fs / 1000.0

    work = motion[:, motion_index["v_drivework"]] - motion[0, motion_index["v_drivework"]]
    bath = aligned_thermo[:, thermo_index["f_BATH"]]
    bath = bath - bath[0]
    total = aligned_thermo[:, thermo_index["TotEng"]]
    total = total - total[0]
    energy_residual = total - work + bath

    masses = parse_lammps_data_masses(run.model_data)
    atoms = parse_lammps_data_atoms(run.model_data)
    water_mass = sum(masses[atom_type] for atom_id, (atom_type, _) in atoms.items() if atom_id > run.substrate_atoms)
    series: list[dict[str, object]] = []
    momentum_summary: list[dict[str, object]] = []
    impulses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for axis, component in (("x", 1), ("y", 2)):
        net_force = forces[:, force_index[f"c_FnetWater[{component}]"]]
        dt = np.diff(time_ps)
        integrated = np.concatenate(
            ([0.0], np.cumsum(0.5 * (net_force[1:] + net_force[:-1]) * dt))
        )
        velocity = motion[:, motion_index[f"v_vwater{axis}"]]
        observed = water_mass * (velocity - velocity[0]) / METAL_FORCE_TO_ACCELERATION_A_PER_PS2_PER_AMU
        residual = observed - integrated
        impulses[axis] = (integrated, observed, residual)
        scale = max(float(np.max(np.abs(integrated))), 1.0e-12)
        momentum_summary.append(
            {
                "case_id": run.case_id,
                "branch_id": run.branch_id,
                "axis": axis,
                "water_mass_amu": water_mass,
                "final_integrated_net_force_eV_ps_per_A": float(integrated[-1]),
                "final_observed_momentum_change_eV_ps_per_A": float(observed[-1]),
                "final_momentum_residual_eV_ps_per_A": float(residual[-1]),
                "momentum_residual_rms_eV_ps_per_A": float(np.sqrt(np.mean(residual**2))),
                "maximum_relative_momentum_residual": float(np.max(np.abs(residual)) / scale),
            }
        )
    all_force_closure = np.column_stack(
        [
            forces[:, force_index[f"c_FrawAll[{component}]"]]
            - forces[:, force_index[f"c_FrawSub[{component}]"]]
            - forces[:, force_index[f"c_FrawWater[{component}]"]]
            for component in (1, 2, 3)
        ]
    )
    for index, step in enumerate(steps):
        series.append(
            {
                "case_id": run.case_id,
                "branch_id": run.branch_id,
                "step": int(step),
                "time_ps": float(time_ps[index]),
                "drive_work_eV": float(work[index]),
                "thermostat_removed_eV": float(bath[index]),
                "delta_total_energy_eV": float(total[index]),
                "energy_closure_residual_eV": float(energy_residual[index]),
                "integrated_net_force_x_eV_ps_per_A": float(impulses["x"][0][index]),
                "observed_momentum_change_x_eV_ps_per_A": float(impulses["x"][1][index]),
                "momentum_residual_x_eV_ps_per_A": float(impulses["x"][2][index]),
                "integrated_net_force_y_eV_ps_per_A": float(impulses["y"][0][index]),
                "observed_momentum_change_y_eV_ps_per_A": float(impulses["y"][1][index]),
                "momentum_residual_y_eV_ps_per_A": float(impulses["y"][2][index]),
            }
        )
    summary = {
        "case_id": run.case_id,
        "branch_id": run.branch_id,
        "samples": len(series),
        "duration_ps": float(time_ps[-1]),
        "final_drive_work_eV": float(work[-1]),
        "final_thermostat_removed_eV": float(bath[-1]),
        "final_delta_total_energy_eV": float(total[-1]),
        "final_energy_closure_residual_eV": float(energy_residual[-1]),
        "energy_closure_residual_rms_eV": float(np.sqrt(np.mean(energy_residual**2))),
        "maximum_raw_force_partition_residual_eV_per_A": float(np.max(np.abs(all_force_closure))),
        "scope": "global accounting only; not a unique local or pairwise dissipation decomposition",
    }
    return series, momentum_summary, summary


def _plot_results(
    output: Path,
    detector_envelope: Sequence[Mapping[str, object]],
    paired_summary: Sequence[Mapping[str, object]],
    balance_summary: Sequence[Mapping[str, object]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    qualified = [
        row
        for row in detector_envelope
        if row["mode"] == "coherent" and float(row["amplitude_A"]) > 0
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    labels = [f"{row['case_id']}:{row['axis']} {row['amplitude_A']}A/{row['duration_ps']}ps" for row in qualified]
    axis.bar(np.arange(len(qualified)), [float(row["recovery_fraction"]) for row in qualified], color="#0072B2")
    axis.axhline(0.75, color="#D55E00", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Recovery fraction")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
    figure.tight_layout()
    path = output / "06_figures" / "detector_coherent_injection_recovery.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    labels = [f"{row['case_id']}:{row['axis']}:{row['edge']}" for row in paired_summary]
    axis.bar(
        np.arange(len(paired_summary)),
        [float(row["paired_delta_persistent_mean_relative_slip_rate_A_per_ps"]) for row in paired_summary],
        color=["#009E73" if row["edge"] == "leading" else "#CC79A7" for row in paired_summary],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Forced - F0 persistent-member slip rate (A/ps)")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    figure.tight_layout()
    path = output / "06_figures" / "paired_persistent_member_slip.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    labels = [f"{row['case_id']}:{row['branch_id']}" for row in balance_summary]
    axis.bar(
        np.arange(len(balance_summary)),
        [float(row["final_energy_closure_residual_eV"]) for row in balance_summary],
        color="#E69F00",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Final global energy closure residual (eV)")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    figure.tight_layout()
    path = output / "06_figures" / "global_energy_closure.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def analyze_package(
    package_root: Path,
    reference_analysis: Path,
    output_dir: Path,
    *,
    window_ps: float = 20.0,
    contact_height_A: float = 5.0,
    edge_tail_fraction: float = 0.10,
    oo_cutoff_A: float = 3.5,
) -> dict[str, object]:
    root = Path(package_root).resolve()
    reference = Path(reference_analysis).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"immutable output already exists: {output}")
    subdirectories = (
        "00_contract",
        "01_inputs",
        "02_motion",
        "03_network",
        "04_detector_qualification",
        "05_balance",
        "06_figures",
        "07_review",
        "08_validation",
    )
    for name in subdirectories:
        (output / name).mkdir(parents=True, exist_ok=False)
    contract = {
        "stage": "Stage A: Observable Qualification",
        "coordinate_window_ps": window_ps,
        "coordinate_cadence_ps": 0.01,
        "contact_region": "dynamic height-selected contact population",
        "edge_region": f"{edge_tail_fraction:.3f} axis-tail fraction of contact water",
        "contact_height_A": contact_height_A,
        "oo_cutoff_A": oo_cutoff_A,
        "hbond_angle_deg": 30.0,
        "motion_observables": ["geometric edge", "persistent-member substrate-relative slip", "membership flux"],
        "network_observables": ["persistent-member pair turnover", "membership-caused turnover", "continuous pair lifetime"],
        "detector_qualification": "logistic injections into 2-ps-detrended real F0 edge noise",
        "balance_scope": "global momentum and work/heat accounting; no unique local dissipation partition",
        "independence_warning": "time frames and injection centers are not independent equilibrium replicas",
        "scientific_scope": "observable qualification only; no causal mechanism or event-rate claim",
    }
    (output / "00_contract" / "ANALYSIS-CONTRACT.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    directory_plan = [
        "# Directory plan",
        "",
        "- `00_contract`: frozen definitions and scope.",
        "- `01_inputs`: source identity and immutable input manifest.",
        "- `02_motion`: persistent-member slip and region membership motion.",
        "- `03_network`: high-cadence H-bond turnover with membership separation.",
        "- `04_detector_qualification`: injection/recovery operating envelope.",
        "- `05_balance`: global momentum and work/heat accounting.",
        "- `06_figures`: diagnostic figures only.",
        "- `07_review`: scientific limits and Stage B review gate.",
        "- `08_validation`: machine-readable gate and output checksums.",
    ]
    (output / "00_contract" / "DIRECTORY-PLAN.md").write_text(
        "\n".join(directory_plan) + "\n", encoding="utf-8"
    )

    runs = discover_runs(root)
    input_rows: list[dict[str, object]] = []
    for run in runs:
        for filename in (
            "RUN-RESULT.txt",
            "VALIDATION.json",
            "OUTPUT-SHA256SUMS",
            "tpcl_coordinates.lammpstrj.zst",
            "motion_energy_stress_0p01ps.dat",
            "force_sums_0p01ps.dat",
            "lmp.out",
        ):
            path = run.run_dir / filename
            input_rows.append(
                {
                    "case_id": run.case_id,
                    "branch_id": run.branch_id,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path) if path.stat().st_size < 20_000_000 else "sealed_by_OUTPUT-SHA256SUMS",
                }
            )
    for path in sorted((reference / "02_kinematics").glob("*__f0_shared.tsv")):
        input_rows.append(
            {
                "case_id": "reference",
                "branch_id": "v3_f0_kinematics",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_tsv(output / "01_inputs" / "INPUT-MANIFEST.tsv", input_rows)

    all_region_summaries: list[dict[str, object]] = []
    for run in runs:
        rows, lifetimes = extract_high_cadence_regions(
            run,
            window_ps=window_ps,
            contact_height_A=contact_height_A,
            edge_tail_fraction=edge_tail_fraction,
            oo_cutoff_A=oo_cutoff_A,
        )
        name = f"{run.case_id}__{run.branch_id}.tsv"
        _write_tsv(output / "02_motion" / name, rows, MOTION_FIELDS)
        _write_tsv(output / "03_network" / name, rows, NETWORK_FIELDS)
        _write_tsv(output / "03_network" / name.replace(".tsv", "__lifetimes.tsv"), lifetimes)
        all_region_summaries.extend(summarize_regions(rows, lifetimes))
    _write_tsv(output / "02_motion" / "region_summary.tsv", all_region_summaries)
    _write_tsv(output / "03_network" / "region_summary.tsv", all_region_summaries)
    paired_summary = paired_region_summary(all_region_summaries)
    _write_tsv(output / "02_motion" / "paired_region_summary.tsv", paired_summary)
    _write_tsv(output / "03_network" / "paired_region_summary.tsv", paired_summary)

    detector_trials, detector_envelope = qualify_detector(reference)
    _write_tsv(output / "04_detector_qualification" / "injection_recovery_trials.tsv", detector_trials)
    _write_tsv(output / "04_detector_qualification" / "detector_operating_envelope.tsv", detector_envelope)

    balance_series: list[dict[str, object]] = []
    momentum_summary: list[dict[str, object]] = []
    balance_summary: list[dict[str, object]] = []
    for run in runs:
        series, momentum, summary = audit_global_balance(run)
        balance_series.extend(series)
        momentum_summary.extend(momentum)
        balance_summary.append(summary)
    _write_tsv(output / "05_balance" / "global_balance_timeseries.tsv", balance_series)
    _write_tsv(output / "05_balance" / "momentum_summary.tsv", momentum_summary)
    _write_tsv(output / "05_balance" / "energy_summary.tsv", balance_summary)
    figures = _plot_results(output, detector_envelope, paired_summary, balance_summary)

    qualified_cells = sum(bool(row["qualified_at_75pct_recovery"]) for row in detector_envelope)
    positive_cells = sum(bool(row["expected_positive_center_advance"]) for row in detector_envelope)
    baseline_trials = [row for row in detector_trials if float(row["amplitude_A"]) == 0.0]
    baseline_false_positive_fraction = float(
        np.mean([int(row["detected_event_count"]) > 0 for row in baseline_trials])
    )
    max_energy_residual = max(abs(float(row["final_energy_closure_residual_eV"])) for row in balance_summary)
    max_momentum_relative = max(float(row["maximum_relative_momentum_residual"]) for row in momentum_summary)
    validation = {
        "status": "PASS",
        "run_count": len(runs),
        "coordinate_window_ps": window_ps,
        "region_summary_count": len(all_region_summaries),
        "paired_region_summary_count": len(paired_summary),
        "detector_trial_count": len(detector_trials),
        "detector_positive_operating_cells": positive_cells,
        "detector_qualified_operating_cells": qualified_cells,
        "baseline_false_positive_trial_fraction": baseline_false_positive_fraction,
        "maximum_absolute_final_energy_closure_residual_eV": max_energy_residual,
        "maximum_relative_momentum_residual": max_momentum_relative,
        "network_resolution_gate": "PASS_MEMBERSHIP_AND_PERSISTENT_PAIR_TURNOVER_SEPARATED",
        "detector_gate": "PASS_OPERATING_ENVELOPE_MEASURED_NOT_UNIVERSAL",
        "balance_gate": "PASS_GLOBAL_ACCOUNTING_COMPUTED_NOT_LOCAL_DISSIPATION",
        "scientific_gate": "REVIEW_REQUIRED_BEFORE_STAGE_B_MD",
        "figure_count": len(figures),
    }
    (output / "08_validation" / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Stage A observable-qualification review",
        "",
        f"- Pipeline validation: `{validation['status']}`",
        f"- Network-resolution gate: `{validation['network_resolution_gate']}`",
        f"- Detector gate: `{validation['detector_gate']}`",
        f"- Balance gate: `{validation['balance_gate']}`",
        f"- Scientific gate: `{validation['scientific_gate']}`",
        f"- Qualified detector cells: `{qualified_cells}/{positive_cells}` positive-injection cells.",
        f"- Baseline false-positive trial fraction: `{baseline_false_positive_fraction:.6f}`.",
        "",
        "The high-cadence analysis separates changes among persistent region members from changes caused by molecules entering or leaving the edge region. This repairs a key ambiguity in the previous turnover metric. Detector non-recovery outside the measured operating envelope must not be interpreted as physical absence of depinning.",
        "",
        "Momentum and work/heat tables are global conservation diagnostics. They do not provide a unique atom-wise, hydrogen-bond-wise, or spatially local dissipation decomposition for the many-body potential.",
        "",
        "This package qualifies observables and exposes their supported range. It does not establish water-network causality, a converged event rate, or a transferable friction law. Stage B pulse/reversal MD remains review-gated until the paired summaries and detector envelope are interpreted together.",
    ]
    (output / "07_review" / "STAGE-A-REVIEW.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    next_stage = [
        "# Stage B review gate",
        "",
        "Do not submit the pulse/reversal matrix from this file alone.",
        "",
        "Stage B may be frozen only after review confirms:",
        "",
        "1. the motion estimator has a documented detector operating envelope;",
        "2. membership turnover is not mislabeled as hydrogen-bond chemistry;",
        "3. the selected response exceeds F0 fluctuations in more than one contact history;",
        "4. global accounting is numerically interpretable under the chosen protocol; and",
        "5. the target surface retains a finite, localized contact line.",
    ]
    (output / "07_review" / "NEXT-STAGE-REVIEW-GATE.md").write_text(
        "\n".join(next_stage) + "\n", encoding="utf-8"
    )
    hashed = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "OUTPUT-SHA256SUMS":
            hashed.append(f"{_sha256(path)}  {path.relative_to(output)}")
    (output / "08_validation" / "OUTPUT-SHA256SUMS").write_text(
        "\n".join(hashed) + "\n", encoding="utf-8"
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-ps", type=float, default=20.0)
    parser.add_argument("--contact-height-A", type=float, default=5.0)
    parser.add_argument("--edge-tail-fraction", type=float, default=0.10)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_ps <= 0 or args.edge_tail_fraction <= 0 or args.edge_tail_fraction >= 0.5:
        raise ValueError("window and edge-tail fraction are outside the supported range")
    result = analyze_package(
        args.package_root,
        args.reference_analysis,
        args.output_dir,
        window_ps=args.window_ps,
        contact_height_A=args.contact_height_A,
        edge_tail_fraction=args.edge_tail_fraction,
        oo_cutoff_A=args.oo_cutoff_A,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
