"""Causal-observer audit for high-cadence water-driven nanobubble trajectories.

The analysis deliberately separates three quantities that a moving contact-edge
window otherwise conflates: molecular exchange in a substrate-fixed window,
membership change in a moving window, and the membership change created by
moving the window over frozen coordinates.  Event onsets use past-only
smoothing; future coordinates are used only to confirm persistence. The word
"causal" describes temporal filtering, not identification of causal effects.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from molsimflow.io.lammps_dump import (
    box_lengths,
    iter_lammps_dump_records_selected,
    minimum_image_vectors,
)
from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
    xy_minimum_image,
)
from molsimflow.postprocess.nanobubble_edge_water import (
    _parse_data_atoms,
    _site_registry,
)
from molsimflow.postprocess.nanobubble_flow_response import _molecule_centers
from molsimflow.postprocess.nanobubble_local_flow import (
    _frame_arrays,
    _iter_sampled_frames,
    _range,
    _trajectory_entries,
)
from molsimflow.postprocess.interfacial_water_orientation import assign_hydrogen_neighbors


def causal_rolling_median(values: Sequence[float], window_frames: int) -> np.ndarray:
    """Return a trailing median; values after an index never affect that index."""

    array = np.asarray(values, dtype=float)
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    return np.asarray([
        np.median(array[max(0, index - window_frames + 1): index + 1])
        for index in range(len(array))
    ])


def past_only_changes(
    values: Sequence[float], *, cadence_ps: float, smoothing_ps: float,
    reference_ps: float,
) -> np.ndarray:
    """Past-only displacement relative to the same trailing statistic one lag ago."""

    if cadence_ps <= 0.0 or smoothing_ps <= 0.0 or reference_ps <= 0.0:
        raise ValueError("cadence, smoothing, and reference times must be positive")
    smooth = causal_rolling_median(values, max(1, round(smoothing_ps / cadence_ps)))
    lag = max(1, round(reference_ps / cadence_ps))
    changes = np.full(len(smooth), np.nan, dtype=float)
    changes[lag:] = smooth[lag:] - smooth[:-lag]
    return changes


def detect_past_only_onsets(
    table: pd.DataFrame,
    *,
    threshold_A: float,
    cadence_ps: float,
    smoothing_ps: float,
    reference_ps: float,
    persistence_ps: float,
    persistence_fraction: float,
    refractory_ps: float,
    minimum_time_ps: float | None = None,
    maximum_time_ps: float | None = None,
) -> list[dict[str, object]]:
    """Detect first threshold crossings and confirm their later persistence.

    The onset and displacement sign use only coordinates at or before the onset.
    Coordinates after the onset decide only whether the candidate is persistent.
    """

    if threshold_A <= 0.0:
        raise ValueError("threshold_A must be positive")
    required = {"time_ps", "edge_coordinate_A", "observer_valid"}
    if not required.issubset(table.columns):
        raise ValueError(f"onset table is missing {sorted(required - set(table.columns))}")
    work = table.sort_values("time_ps").reset_index(drop=True)
    times = work.time_ps.to_numpy(float)
    values = work.edge_coordinate_A.to_numpy(float)
    valid = work.observer_valid.to_numpy(int).astype(bool) & np.isfinite(values)
    changes = past_only_changes(
        values, cadence_ps=cadence_ps, smoothing_ps=smoothing_ps,
        reference_ps=reference_ps,
    )
    smooth = causal_rolling_median(values, max(1, round(smoothing_ps / cadence_ps)))
    persist_frames = max(1, round(persistence_ps / cadence_ps))
    refractory_frames = max(1, round(refractory_ps / cadence_ps))
    reference_frames = max(1, round(reference_ps / cadence_ps))
    candidates: list[dict[str, object]] = []
    previous_above = False
    for index in range(reference_frames, len(work) - persist_frames):
        in_range = (
            (minimum_time_ps is None or times[index] >= minimum_time_ps)
            and (maximum_time_ps is None or times[index] <= maximum_time_ps)
        )
        contiguous = bool(np.all(valid[index - reference_frames:index + persist_frames + 1]))
        above = contiguous and math.isfinite(changes[index]) and abs(changes[index]) >= threshold_A
        if not in_range or not above or previous_above:
            previous_above = above
            continue
        previous_above = True
        change = float(changes[index])
        reference = float(smooth[index - reference_frames])
        midpoint = reference + 0.5 * change
        sign = 1.0 if change > 0.0 else -1.0
        future = smooth[index + 1:index + persist_frames + 1]
        persistence = float(np.mean(sign * (future - midpoint) >= 0.0))
        if persistence < persistence_fraction:
            continue
        row = work.iloc[index]
        candidates.append({
            "index": index,
            "step": int(row.step) if "step" in work else index,
            "onset_time_ps": float(times[index]),
            "reference_time_ps": float(times[index - reference_frames]),
            "reference_position_A": reference,
            "onset_position_A": float(smooth[index]),
            "past_only_displacement_A": change,
            "threshold_A": float(threshold_A),
            "persistence_fraction": persistence,
        })
    selected: list[dict[str, object]] = []
    for candidate in candidates:
        if selected and int(candidate["index"]) - int(selected[-1]["index"]) <= refractory_frames:
            continue
        selected.append(candidate)
    return selected


def match_events_one_to_one(
    reference: pd.DataFrame, candidate: pd.DataFrame, *, tolerance_ps: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Greedily match event identities within exact categorical strata."""

    if tolerance_ps <= 0.0:
        raise ValueError("tolerance_ps must be positive")
    keys = ["branch_id", "patch_id", "edge", "process"]
    matches: list[dict[str, object]] = []
    used: set[int] = set()
    for left_index, left in reference.iterrows():
        options = []
        for right_index, right in candidate.iterrows():
            if int(right_index) in used or any(str(left[key]) != str(right[key]) for key in keys):
                continue
            delta = abs(float(left.anchor_time_ps) - float(right.anchor_time_ps))
            if delta <= tolerance_ps:
                options.append((delta, int(right_index), right))
        if not options:
            continue
        delta, right_index, right = min(options, key=lambda item: (item[0], item[1]))
        used.add(right_index)
        matches.append({
            "reference_event_id": left.get("event_id", left_index),
            "candidate_event_id": right.get("event_id", right_index),
            "branch_id": left.branch_id, "patch_id": left.patch_id,
            "edge": left.edge, "process": left.process,
            "reference_time_ps": float(left.anchor_time_ps),
            "candidate_time_ps": float(right.anchor_time_ps),
            "absolute_time_offset_ps": delta,
            "reference_displacement_A": float(left.displacement_A),
            "candidate_displacement_A": float(right.displacement_A),
        })
    count = len(matches)
    denominator = len(reference) + len(candidate) - count
    metrics = {
        "reference_events": float(len(reference)),
        "candidate_events": float(len(candidate)),
        "matched_events": float(count),
        "recall": count / len(reference) if len(reference) else math.nan,
        "precision": count / len(candidate) if len(candidate) else math.nan,
        "jaccard": count / denominator if denominator else math.nan,
    }
    return matches, metrics


def membership_turnover(first: np.ndarray, second: np.ndarray) -> float:
    first_set = set(np.asarray(first, dtype=int).tolist())
    second_set = set(np.asarray(second, dtype=int).tolist())
    denominator = len(first_set) + len(second_set)
    return len(first_set.symmetric_difference(second_set)) / denominator if denominator else math.nan


def _periodic_delta(values: np.ndarray, reference: float, length: float) -> np.ndarray:
    delta = np.asarray(values, dtype=float) - float(reference)
    return delta - length * np.round(delta / length)


def _in_transverse_patch(
    values: np.ndarray, lower: float, upper: float, *, box_lower: float, box_length: float,
) -> np.ndarray:
    wrapped = (np.asarray(values, dtype=float) - box_lower) % box_length + box_lower
    lower_wrapped = (lower - box_lower) % box_length + box_lower
    upper_wrapped = (upper - box_lower) % box_length + box_lower
    if lower_wrapped < upper_wrapped:
        return (wrapped >= lower_wrapped) & (wrapped < upper_wrapped)
    return (wrapped >= lower_wrapped) | (wrapped < upper_wrapped)


def window_membership(
    atom_ids: np.ndarray,
    coordinates: np.ndarray,
    bounds: np.ndarray,
    *,
    drive_index: int,
    transverse_index: int,
    reference_drive_A: float,
    drive_halfwidth_A: float,
    transverse_lo_A: float,
    transverse_hi_A: float,
    surface_top_A: float,
    height_A: float,
) -> np.ndarray:
    lengths = box_lengths(bounds)
    selected = (
        (np.abs(_periodic_delta(
            coordinates[:, drive_index], reference_drive_A, lengths[drive_index]
        )) <= drive_halfwidth_A)
        & _in_transverse_patch(
            coordinates[:, transverse_index], transverse_lo_A, transverse_hi_A,
            box_lower=float(bounds[transverse_index, 0]),
            box_length=float(lengths[transverse_index]),
        )
        & (coordinates[:, 2] >= surface_top_A)
        & (coordinates[:, 2] <= surface_top_A + height_A)
    )
    return np.asarray(atom_ids[selected], dtype=int)


def periodic_site_weights(
    site_xy: np.ndarray,
    point_xy: np.ndarray,
    bounds: np.ndarray,
    *, sigma_A: float, cutoff_A: float,
) -> np.ndarray:
    if sigma_A <= 0.0 or cutoff_A <= 0.0:
        raise ValueError("site-kernel sigma and cutoff must be positive")
    delta = np.asarray(site_xy, dtype=float) - np.asarray(point_xy, dtype=float)
    lengths = box_lengths(bounds)
    delta[:, :2] -= lengths[:2] * np.round(delta[:, :2] / lengths[:2])
    distance = np.linalg.norm(delta[:, :2], axis=1)
    weights = np.exp(-0.5 * (distance / sigma_A) ** 2)
    weights[distance > cutoff_A] = 0.0
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else weights


def _read_tsv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if table.empty:
        raise ValueError(f"No rows in {path}")
    return table


def _load_site_data(model_path: Path, registry_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _bounds, atoms = _parse_data_atoms(model_path)
    registry = _site_registry(registry_path)
    ids = np.asarray(sorted(registry), dtype=int)
    xy = np.asarray([atoms[int(atom_id)][1][:2] for atom_id in ids], dtype=float)
    types = np.asarray([registry[int(atom_id)] for atom_id in ids], dtype=str)
    return ids, xy, types


def _table_for_edge(path: Path, patch_id: str, edge: str) -> pd.DataFrame:
    table = _read_tsv(path)
    selected = table[(table.patch_id == patch_id) & (table.edge == edge)].copy()
    return selected.sort_values("time_ps").reset_index(drop=True)


def eligible_patch_ids(
    driven_patch_ids: Sequence[object], baseline_patch_ids: Sequence[object],
    excluded_patch_ids: Sequence[object],
) -> list[str]:
    """Return common patch IDs after applying an explicit observer exclusion."""

    driven = {str(value) for value in driven_patch_ids}
    baseline = {str(value) for value in baseline_patch_ids}
    excluded = {str(value) for value in excluded_patch_ids}
    return sorted((driven & baseline) - excluded)


def _calibrate_and_detect_onsets(
    raw: Mapping[str, object], cases: Sequence[Mapping[str, object]], base: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cadence = float(raw["sample_interval_ps"])
    calibration_end = float(raw["f0_calibration_end_ps"])
    calibration_rows: list[dict[str, object]] = []
    onset_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    event_index = 1
    for case in cases:
        direction = str(case["direction"]).lower()
        driven_path = resolve_path(case["patch_frames_100fs"], base)
        baseline_path = resolve_path(case["baseline_patch_frames_100fs"], base)
        driven_all = _read_tsv(driven_path)
        baseline_all = _read_tsv(baseline_path)
        patch_ids = eligible_patch_ids(
            driven_all.patch_id, baseline_all.patch_id,
            raw.get("excluded_patch_ids", ()),
        )
        for patch_id in patch_ids:
            for edge in ("front", "rear"):
                driven = driven_all[(driven_all.patch_id == patch_id) & (driven_all.edge == edge)].copy()
                baseline = baseline_all[(baseline_all.patch_id == patch_id) & (baseline_all.edge == edge)].copy()
                if driven.empty or baseline.empty:
                    continue
                driven = driven.sort_values("time_ps").reset_index(drop=True)
                baseline = baseline.sort_values("time_ps").reset_index(drop=True)
                calibration = baseline[baseline.time_ps <= calibration_end].reset_index(drop=True)
                heldout = baseline[baseline.time_ps > calibration_end].reset_index(drop=True)
                coverage = {
                    "driven": float(np.mean(driven.observer_valid.astype(int))),
                    "f0_calibration": float(np.mean(calibration.observer_valid.astype(int))),
                    "f0_heldout": float(np.mean(heldout.observer_valid.astype(int))),
                }
                qualified = all(
                    value >= float(raw["minimum_patch_coverage"])
                    for value in coverage.values()
                )
                changes = past_only_changes(
                    calibration.edge_coordinate_A.to_numpy(float),
                    cadence_ps=cadence,
                    smoothing_ps=float(raw["onset_smoothing_ps"]),
                    reference_ps=float(raw["onset_reference_ps"]),
                )
                finite = np.abs(changes[np.isfinite(changes)])
                threshold = max(
                    float(raw["minimum_displacement_A"]),
                    float(np.quantile(finite, float(raw["f0_quantile"]))) if len(finite) else math.inf,
                )
                key = {
                    "case_id": str(case["case_id"]),
                    "branch_id": str(case["branch_id"]),
                    "direction": direction, "patch_id": patch_id, "edge": edge,
                }
                calibration_rows.append({
                    **key, "baseline_branch_id": str(baseline.branch_id.iloc[0]),
                    "calibration_frames": len(calibration), "heldout_frames": len(heldout),
                    "f0_quantile": float(raw["f0_quantile"]), "threshold_A": threshold,
                    "qualification": "PASS" if qualified else "HOLD_LOW_COVERAGE",
                })
                coverage_rows.append({
                    **key,
                    "driven_valid_fraction": coverage["driven"],
                    "f0_calibration_valid_fraction": coverage["f0_calibration"],
                    "f0_heldout_valid_fraction": coverage["f0_heldout"],
                    "qualification": "PASS" if qualified else "HOLD_LOW_COVERAGE",
                })
                if not qualified or not math.isfinite(threshold):
                    continue
                for source, table, lower in (
                    ("driven", driven, None),
                    ("f0_heldout", baseline, calibration_end + cadence),
                ):
                    detected = detect_past_only_onsets(
                        table,
                        threshold_A=threshold, cadence_ps=cadence,
                        smoothing_ps=float(raw["onset_smoothing_ps"]),
                        reference_ps=float(raw["onset_reference_ps"]),
                        persistence_ps=float(raw["persistence_ps"]),
                        persistence_fraction=float(raw["persistence_fraction"]),
                        refractory_ps=float(raw["refractory_ps"]),
                        minimum_time_ps=lower,
                        maximum_time_ps=float(raw["trajectory_duration_ps"]),
                    )
                    for event in detected:
                        displacement = float(event["past_only_displacement_A"])
                        process = (
                            "gas_advance_liquid_retreat"
                            if (displacement > 0.0) == (edge == "front")
                            else "gas_retreat_liquid_advance"
                        )
                        onset_rows.append({
                            **key, "source": source,
                            "event_id": f"CO{event_index:05d}", "process": process,
                            **{name: value for name, value in event.items() if name != "index"},
                        })
                        event_index += 1
    onset_columns = (
        "case_id", "branch_id", "direction", "patch_id", "edge", "source",
        "event_id", "process", "step", "onset_time_ps", "reference_time_ps",
        "reference_position_A", "onset_position_A", "past_only_displacement_A",
        "threshold_A", "persistence_fraction",
    )
    onset_table = pd.DataFrame(onset_rows)
    if onset_table.empty:
        onset_table = pd.DataFrame(columns=onset_columns)
    return pd.DataFrame(calibration_rows), onset_table, pd.DataFrame(coverage_rows)


def _load_geometry(path: Path) -> pd.DataFrame:
    table = _read_tsv(path)
    return table.set_index("step", drop=False)


def _main_cluster_frame(
    atom_ids: np.ndarray,
    coordinates: np.ndarray,
    bounds: np.ndarray,
    *, nitrogen_range: tuple[int, int], relocated_ids: set[int], cutoff_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = (atom_ids >= nitrogen_range[0]) & (atom_ids <= nitrogen_range[1])
    n_ids = atom_ids[selected]
    n_coord = coordinates[selected]
    if len(n_ids) % 2:
        raise ValueError("Nitrogen atom range is not complete N2 pairs")
    pairs = n_ids.reshape((-1, 2))
    centers = _molecule_centers(n_coord, bounds)
    eligible = np.asarray([
        not bool(relocated_ids.intersection(pair.tolist())) for pair in pairs
    ], dtype=bool)
    eligible_index = np.flatnonzero(eligible)
    components = connected_components(centers[eligible_index], bounds, cutoff_A)
    if not components:
        raise ValueError("No eligible N2 component")
    main = eligible_index[components[0]]
    molecule_ids = pairs[main, 0].astype(int)
    return molecule_ids, centers[main], centers, pairs[:, 0].astype(int)


def _periodic_mean_1d(values: np.ndarray, lower: float, length: float) -> float:
    angles = 2.0 * np.pi * (np.asarray(values) - lower) / length
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) < 1.0e-12:
        return float(values[0])
    return lower + (float(np.angle(vector)) % (2.0 * np.pi)) * length / (2.0 * np.pi)


def _edge_point(
    centers: np.ndarray, bounds: np.ndarray, *, drive_index: int,
    transverse_index: int, edge_coordinate_A: float, transverse_lo_A: float,
    transverse_hi_A: float, surface_top_A: float, contact_cutoff_A: float,
    drive_halfwidth_A: float,
) -> tuple[np.ndarray, str]:
    lengths = box_lengths(bounds)
    selected = (
        (centers[:, 2] <= surface_top_A + contact_cutoff_A)
        & (np.abs(_periodic_delta(
            centers[:, drive_index], edge_coordinate_A, lengths[drive_index]
        )) <= drive_halfwidth_A)
        & _in_transverse_patch(
            centers[:, transverse_index], transverse_lo_A, transverse_hi_A,
            box_lower=float(bounds[transverse_index, 0]),
            box_length=float(lengths[transverse_index]),
        )
    )
    point = np.empty(2, dtype=float)
    point[drive_index] = (
        (edge_coordinate_A - bounds[drive_index, 0]) % lengths[drive_index]
        + bounds[drive_index, 0]
    )
    if np.any(selected):
        point[transverse_index] = _periodic_mean_1d(
            centers[selected, transverse_index], float(bounds[transverse_index, 0]),
            float(lengths[transverse_index]),
        )
        return point, "CONTACT_LOCALIZED"
    point[transverse_index] = (
        (0.5 * (transverse_lo_A + transverse_hi_A) - bounds[transverse_index, 0])
        % lengths[transverse_index] + bounds[transverse_index, 0]
    )
    return point, "PATCH_CENTER_FALLBACK"


def _load_branch_arrays(
    case: Mapping[str, object], raw: Mapping[str, object], base: Path,
) -> dict[str, object]:
    entries = _trajectory_entries(case["state_trajectories"], base)
    origin = int(case["time_origin_step"])
    maximum = int(case["maximum_step"])
    stride = int(raw["sample_stride_steps"])
    water_range = _range(case["water_atom_range"], "water_atom_range")
    nitrogen_range = _range(case["nitrogen_atom_range"], "nitrogen_atom_range")
    oxygen_type = int(raw.get("oxygen_type", 2))
    hydrogen_type = int(raw.get("hydrogen_type", 1))
    relocated = {int(value) for value in case.get("relocated_nitrogen_ids", [])}
    steps: list[int] = []
    water_frames: list[np.ndarray] = []
    main_ids: list[np.ndarray] = []
    main_centers: list[np.ndarray] = []
    all_n2_centers: list[np.ndarray] = []
    all_n2_ids_reference: np.ndarray | None = None
    reference_ids: np.ndarray | None = None
    reference_types: np.ndarray | None = None
    bounds_reference: np.ndarray | None = None
    water_ids_reference: np.ndarray | None = None
    torque_values: list[np.ndarray] = []
    torque_stride = max(1, round(float(raw["torque_stride_ps"]) / float(raw["sample_interval_ps"])))
    drive_index = 0 if str(case["direction"]).lower() == "x" else 1
    for frame in _iter_sampled_frames(
        entries, time_origin_step=origin, maximum_step=maximum, stride_steps=stride
    ):
        atom_ids, atom_types, coordinates = _frame_arrays(frame)
        if reference_ids is None:
            reference_ids = atom_ids.copy(); reference_types = atom_types.copy()
            bounds_reference = frame.bounds.copy()
        elif not np.array_equal(atom_ids, reference_ids) or not np.array_equal(atom_types, reference_types):
            raise ValueError(f"{case['branch_id']} identity changed at step {frame.timestep}")
        water_mask = (
            (atom_ids >= water_range[0]) & (atom_ids <= water_range[1])
            & (atom_types == oxygen_type)
        )
        water_ids = atom_ids[water_mask]
        if water_ids_reference is None:
            water_ids_reference = water_ids.copy()
        elif not np.array_equal(water_ids, water_ids_reference):
            raise ValueError("Water oxygen identity changed")
        ids, centers, all_centers, all_ids = _main_cluster_frame(
            atom_ids, coordinates, frame.bounds, nitrogen_range=nitrogen_range,
            relocated_ids=relocated, cutoff_A=float(raw["cluster_cutoff_A"]),
        )
        if all_n2_ids_reference is None:
            all_n2_ids_reference = all_ids.copy()
        elif not np.array_equal(all_ids, all_n2_ids_reference):
            raise ValueError("N2 molecular identity changed")
        steps.append(frame.timestep)
        water_frames.append(coordinates[water_mask].astype(np.float32))
        main_ids.append(ids); main_centers.append(centers.astype(np.float32))
        all_n2_centers.append(all_centers.astype(np.float32))
        if (len(steps) - 1) % torque_stride == 0:
            water_atom_mask = (atom_ids >= water_range[0]) & (atom_ids <= water_range[1])
            water_hydrogen = coordinates[water_atom_mask & (atom_types == hydrogen_type)]
            water_oxygen = coordinates[water_mask]
            assigned = assign_hydrogen_neighbors(
                water_oxygen, water_hydrogen, frame.bounds, float(raw["oh_cutoff_A"])
            )
            force = np.zeros(3, dtype=float)
            force[drive_index] = float(case.get("force_eV_per_A", 0.0))
            values = []
            lengths = box_lengths(frame.bounds)
            for oxygen, neighbors in zip(water_oxygen, assigned):
                if len(neighbors) != 2:
                    continue
                vectors = minimum_image_vectors(
                    water_hydrogen[neighbors] - oxygen, lengths
                )
                oxygen_from_com = -np.sum(vectors, axis=0) / 18.0
                values.append(float(np.linalg.norm(np.cross(oxygen_from_com, force))))
            if values:
                torque_values.append(np.asarray(values, dtype=float))
    expected = int(round(float(raw["trajectory_duration_ps"]) / float(raw["sample_interval_ps"]))) + 1
    if len(steps) != expected:
        raise ValueError(f"{case['branch_id']}: expected {expected} frames, got {len(steps)}")
    assert (
        water_ids_reference is not None and bounds_reference is not None
        and all_n2_ids_reference is not None
    )
    return {
        "steps": np.asarray(steps, dtype=int),
        "water_ids": water_ids_reference,
        "water": np.stack(water_frames),
        "main_ids": main_ids, "main_centers": main_centers,
        "all_n2_centers": all_n2_centers, "all_n2_ids": all_n2_ids_reference,
        "bounds": bounds_reference,
        "torque_values": np.concatenate(torque_values) if torque_values else np.asarray([]),
    }


def _torque_summary(
    case: Mapping[str, object], arrays: Mapping[str, object], raw: Mapping[str, object]
) -> dict[str, object]:
    values = np.asarray(arrays["torque_values"], dtype=float)
    return {
        "case_id": case["case_id"], "branch_id": case["branch_id"],
        "direction": case["direction"], "force_per_water_oxygen_eV_per_A": case.get("force_eV_per_A", 0.0),
        "sample_stride_ps": raw["torque_stride_ps"], "intact_water_samples": len(values),
        "mean_external_torque_eV": float(np.mean(values)) if len(values) else math.nan,
        "p95_external_torque_eV": float(np.quantile(values, 0.95)) if len(values) else math.nan,
        "maximum_external_torque_eV": float(np.max(values)) if len(values) else math.nan,
        "interpretation": "kinematic oxygen-only-force torque magnitude; not evidence of artifact dominance",
    }


def _gas_transport(
    case: Mapping[str, object], arrays: Mapping[str, object], raw: Mapping[str, object],
) -> list[dict[str, object]]:
    steps = np.asarray(arrays["steps"], dtype=int)
    bounds = np.asarray(arrays["bounds"], dtype=float)
    direction = str(case["direction"]).lower()
    drive_index = 0 if direction == "x" else 1
    lengths = box_lengths(bounds)
    all_ids = np.asarray(arrays["all_n2_ids"], dtype=int)
    previous_all_wrapped: np.ndarray | None = None
    previous_all_unwrapped: np.ndarray | None = None
    previous_members: set[int] | None = None
    previous_total_center: float | None = None
    rows: list[dict[str, object]] = []
    for index, (ids, centers, all_centers) in enumerate(zip(
        arrays["main_ids"], arrays["main_centers"], arrays["all_n2_centers"]
    )):
        ids = np.asarray(ids, dtype=int); centers = np.asarray(centers, dtype=float)
        all_wrapped = np.asarray(all_centers, dtype=float)[:, drive_index]
        if previous_all_wrapped is None or previous_all_unwrapped is None:
            all_unwrapped = all_wrapped.copy()
        else:
            delta = all_wrapped - previous_all_wrapped
            delta -= lengths[drive_index] * np.round(delta / lengths[drive_index])
            all_unwrapped = previous_all_unwrapped + delta
        unwrapped_lookup = {
            int(atom_id): float(value) for atom_id, value in zip(all_ids, all_unwrapped)
        }
        members = set(ids.tolist())
        total_center = float(np.mean([unwrapped_lookup[value] for value in members]))
        if previous_total_center is None or previous_members is None:
            retained: set[int] = set(); gained = members; lost: set[int] = set()
            material = math.nan; total = math.nan; membership = math.nan; jaccard = math.nan
        else:
            retained = members & previous_members
            gained = members - previous_members; lost = previous_members - members
            total = total_center - previous_total_center
            if retained:
                all_index = {int(atom_id): position for position, atom_id in enumerate(all_ids)}
                retained_index = np.asarray([all_index[value] for value in retained], dtype=int)
                material = float(np.mean(
                    all_unwrapped[retained_index] - previous_all_unwrapped[retained_index]
                ))
            else:
                material = math.nan
            membership = total - material if math.isfinite(material) else math.nan
            union = members | previous_members
            jaccard = len(retained) / len(union) if union else math.nan
        rows.append({
            "case_id": case["case_id"], "branch_id": case["branch_id"],
            "direction": direction, "step": int(steps[index]),
            "time_ps": (int(steps[index]) - int(case["time_origin_step"]))
            * float(raw["timestep_fs"]) / 1000.0,
            "main_cluster_n2": len(members), "retained_n2": len(retained),
            "gained_n2": len(gained), "lost_n2": len(lost),
            "membership_jaccard": jaccard,
            "main_cluster_center_unwrapped_A": total_center,
            "total_center_step_A": total, "retained_material_step_A": material,
            "membership_residual_step_A": membership,
        })
        previous_members = members; previous_total_center = total_center
        previous_all_wrapped = all_wrapped.copy()
        previous_all_unwrapped = all_unwrapped.copy()
    return rows


def _risk_rows_for_case(
    case: Mapping[str, object], arrays: Mapping[str, object], raw: Mapping[str, object],
    base: Path, onsets: pd.DataFrame, site_ids: np.ndarray, site_xy: np.ndarray,
    site_types: np.ndarray, calibration: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[int, dict[str, float]], list[dict[str, object]]]:
    patch = _read_tsv(resolve_path(case["patch_frames_100fs"], base))
    geometry = _load_geometry(resolve_path(case["geometry_frames_100fs"], base))
    steps = np.asarray(arrays["steps"], dtype=int)
    water = np.asarray(arrays["water"], dtype=np.float32)
    water_ids = np.asarray(arrays["water_ids"], dtype=int)
    bounds = np.asarray(arrays["bounds"], dtype=float)
    step_to_index = {int(step): index for index, step in enumerate(steps)}
    direction = str(case["direction"]).lower()
    drive_index, transverse_index = (0, 1) if direction == "x" else (1, 0)
    cadence = float(raw["sample_interval_ps"])
    history_frames = round(float(raw["history_ps"]) / cadence)
    gap_frames = round(float(raw["safety_gap_ps"]) / cadence)
    risk_stride_frames = round(float(raw["risk_stride_ps"]) / cadence)
    horizon = float(raw["prediction_horizon_ps"])
    motif: dict[int, dict[str, float]] = {
        int(atom_id): {"opportunity_weight_ps": 0.0, "event_weight": 0.0}
        for atom_id in site_ids
    }
    rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    qualified = onsets[(onsets.source == "driven")]
    for (patch_id, edge), edge_table in patch.groupby(["patch_id", "edge"], sort=True):
        edge_table = edge_table.sort_values("time_ps").reset_index(drop=True)
        qualified_row = calibration[
            (calibration.patch_id == patch_id) & (calibration.edge == edge)
        ]
        if qualified_row.empty or str(qualified_row.qualification.iloc[0]) != "PASS":
            continue
        edge_by_step = edge_table.set_index("step", drop=False)
        local_onsets = qualified[(qualified.patch_id == patch_id) & (qualified.edge == edge)]
        for frame_index in range(history_frames + gap_frames, len(steps), risk_stride_frames):
            start_index = frame_index - gap_frames - history_frames
            cutoff_index = frame_index - gap_frames
            risk_step = int(steps[frame_index]); start_step = int(steps[start_index])
            cutoff_step = int(steps[cutoff_index])
            if any(step not in edge_by_step.index for step in (risk_step, start_step, cutoff_step)):
                continue
            risk_edge = edge_by_step.loc[risk_step]
            start_edge = edge_by_step.loc[start_step]
            cutoff_edge = edge_by_step.loc[cutoff_step]
            if not all(int(row.observer_valid) for row in (risk_edge, start_edge, cutoff_edge)):
                continue
            start_geometry = geometry.loc[start_step]; cutoff_geometry = geometry.loc[cutoff_step]
            transverse_lo = float(risk_edge.patch_transverse_lo_A)
            transverse_hi = float(risk_edge.patch_transverse_hi_A)
            reference = float(start_edge.edge_coordinate_A)
            fixed_start = window_membership(
                water_ids, water[start_index], bounds, drive_index=drive_index,
                transverse_index=transverse_index, reference_drive_A=reference,
                drive_halfwidth_A=float(raw["edge_water_halfwidth_A"]),
                transverse_lo_A=transverse_lo, transverse_hi_A=transverse_hi,
                surface_top_A=float(start_geometry.surface_top_z_A),
                height_A=float(raw["near_surface_height_A"]),
            )
            fixed_cutoff = window_membership(
                water_ids, water[cutoff_index], bounds, drive_index=drive_index,
                transverse_index=transverse_index, reference_drive_A=reference,
                drive_halfwidth_A=float(raw["edge_water_halfwidth_A"]),
                transverse_lo_A=transverse_lo, transverse_hi_A=transverse_hi,
                surface_top_A=float(cutoff_geometry.surface_top_z_A),
                height_A=float(raw["near_surface_height_A"]),
            )
            moving_start = fixed_start
            moving_cutoff = window_membership(
                water_ids, water[cutoff_index], bounds, drive_index=drive_index,
                transverse_index=transverse_index,
                reference_drive_A=float(cutoff_edge.edge_coordinate_A),
                drive_halfwidth_A=float(raw["edge_water_halfwidth_A"]),
                transverse_lo_A=transverse_lo, transverse_hi_A=transverse_hi,
                surface_top_A=float(cutoff_geometry.surface_top_z_A),
                height_A=float(raw["near_surface_height_A"]),
            )
            swept_cutoff = window_membership(
                water_ids, water[start_index], bounds, drive_index=drive_index,
                transverse_index=transverse_index,
                reference_drive_A=float(cutoff_edge.edge_coordinate_A),
                drive_halfwidth_A=float(raw["edge_water_halfwidth_A"]),
                transverse_lo_A=transverse_lo, transverse_hi_A=transverse_hi,
                surface_top_A=float(start_geometry.surface_top_z_A),
                height_A=float(raw["near_surface_height_A"]),
            )
            start_set = set(fixed_start.tolist()); cutoff_set = set(fixed_cutoff.tolist())
            retained = sorted(start_set & cutoff_set)
            if retained:
                lookup = {int(value): idx for idx, value in enumerate(water_ids)}
                index_values = np.asarray([lookup[value] for value in retained], dtype=int)
                displacement = water[cutoff_index, index_values, drive_index] - water[
                    start_index, index_values, drive_index
                ]
                length = box_lengths(bounds)[drive_index]
                displacement -= length * np.round(displacement / length)
                water_advection = float(np.mean(displacement))
            else:
                water_advection = math.nan
            risk_time = float(risk_edge.time_ps)
            future = local_onsets[
                (local_onsets.onset_time_ps > risk_time)
                & (local_onsets.onset_time_ps <= risk_time + horizon + 1.0e-9)
            ].sort_values("onset_time_ps")
            outcome_id = str(future.event_id.iloc[0]) if not future.empty else ""
            edge_prefix = edge
            local_velocity = float(start_geometry[f"{edge_prefix}_water_drive_velocity_mps"])
            local_density = float(start_geometry[f"{edge_prefix}_oxygen_density_A3"])
            contact_point, point_status = _edge_point(
                np.asarray(arrays["main_centers"][frame_index], dtype=float), bounds,
                drive_index=drive_index, transverse_index=transverse_index,
                edge_coordinate_A=float(risk_edge.edge_coordinate_A),
                transverse_lo_A=transverse_lo, transverse_hi_A=transverse_hi,
                surface_top_A=float(geometry.loc[risk_step].surface_top_z_A),
                contact_cutoff_A=float(raw["contact_cutoff_A"]),
                drive_halfwidth_A=float(raw["site_kernel_cutoff_A"]),
            )
            site_weight = periodic_site_weights(
                site_xy, contact_point, bounds,
                sigma_A=float(raw["site_kernel_sigma_A"]),
                cutoff_A=float(raw["site_kernel_cutoff_A"]),
            )
            for atom_id, weight in zip(site_ids, site_weight):
                motif[int(atom_id)]["opportunity_weight_ps"] += (
                    float(weight) * float(raw["risk_stride_ps"])
                )
            ch3_fraction = float(np.sum(site_weight[site_types == "CH3"]))
            row = {
                "case_id": case["case_id"], "branch_id": case["branch_id"],
                "direction": direction, "patch_id": patch_id, "edge": edge,
                "risk_step": risk_step, "risk_time_ps": risk_time,
                "history_start_time_ps": float(start_edge.time_ps),
                "feature_cutoff_time_ps": float(cutoff_edge.time_ps),
                "block_id": int(math.floor(risk_time / float(raw["block_ps"]))),
                "outcome_event": int(bool(outcome_id)), "outcome_event_id": outcome_id,
                "fixed_turnover": membership_turnover(fixed_start, fixed_cutoff),
                "moving_turnover": membership_turnover(moving_start, moving_cutoff),
                "swept_window_turnover": membership_turnover(moving_start, swept_cutoff),
                "fixed_cohort_survival": (
                    len(start_set & cutoff_set) / len(start_set) if start_set else math.nan
                ),
                "fixed_water_start_count": len(fixed_start),
                "fixed_water_cutoff_count": len(fixed_cutoff),
                "retained_water_drive_displacement_A": water_advection,
                "edge_history_displacement_A": (
                    float(cutoff_edge.edge_coordinate_A) - float(start_edge.edge_coordinate_A)
                ),
                "contact_n2_in_patch": int(risk_edge.contact_n2_in_patch),
                "global_film_water_oxygen_count": int(risk_edge.global_film_water_oxygen_count),
                "bubble_height_A": float(start_geometry.bubble_height_A),
                "bubble_drive_width_A": float(start_geometry.bubble_drive_width_A),
                "local_water_velocity_mps": local_velocity,
                "local_oxygen_density_A3": local_density,
                "site_ch3_weight_fraction": ch3_fraction,
                "edge_point_x_A": float(contact_point[0]),
                "edge_point_y_A": float(contact_point[1]),
                "edge_point_status": point_status,
                "all_water_oxygen_ids_present": 1,
            }
            rows.append(row)
            point_rows.append({
                "case_id": case["case_id"], "branch_id": case["branch_id"],
                "direction": direction, "patch_id": patch_id, "edge": edge,
                "risk_step": risk_step, "risk_time_ps": risk_time,
                "edge_point_status": point_status,
            })
    risk = pd.DataFrame(rows)
    if risk.empty:
        raise ValueError(f"No risk opportunities for {case['branch_id']}")
    # Assign one event weight at the last strictly pre-onset risk row.
    for _, event in qualified.iterrows():
        choices = risk[
            (risk.patch_id == event.patch_id) & (risk.edge == event.edge)
            & (risk.risk_time_ps < float(event.onset_time_ps))
        ].sort_values("risk_time_ps")
        if choices.empty:
            continue
        chosen = choices.iloc[-1]
        point = np.asarray([chosen.edge_point_x_A, chosen.edge_point_y_A], dtype=float)
        weights = periodic_site_weights(
            site_xy, point, bounds, sigma_A=float(raw["site_kernel_sigma_A"]),
            cutoff_A=float(raw["site_kernel_cutoff_A"]),
        )
        for atom_id, weight in zip(site_ids, weights):
            motif[int(atom_id)]["event_weight"] += float(weight)
    return rows, motif, point_rows


def _event_feature_rows(risk: pd.DataFrame, onsets: pd.DataFrame, refractory_ps: float) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for _, event in onsets[onsets.source == "driven"].iterrows():
        local = risk[
            (risk.branch_id == event.branch_id) & (risk.patch_id == event.patch_id)
            & (risk.edge == event.edge) & (risk.risk_time_ps < float(event.onset_time_ps))
        ].sort_values("risk_time_ps")
        if local.empty:
            continue
        chosen = local.iloc[-1].to_dict()
        chosen.update({
            "event_id": event.event_id, "process": event.process,
            "onset_time_ps": float(event.onset_time_ps), "row_class": "EVENT",
        })
        output.append(chosen)
        block = int(chosen["block_id"])
        controls = risk[
            (risk.branch_id == event.branch_id) & (risk.patch_id == event.patch_id)
            & (risk.edge == event.edge) & (risk.block_id == block)
            & (risk.outcome_event == 0)
        ].copy()
        event_times = onsets[
            (onsets.source == "driven") & (onsets.branch_id == event.branch_id)
            & (onsets.patch_id == event.patch_id) & (onsets.edge == event.edge)
        ].onset_time_ps.to_numpy(float)
        if len(event_times):
            controls = controls[controls.risk_time_ps.apply(
                lambda value: np.all(np.abs(event_times - float(value)) >= refractory_ps)
            )]
        if controls.empty:
            continue
        scale = {
            "contact_n2_in_patch": max(1.0, float(controls.contact_n2_in_patch.std(ddof=0))),
            "local_water_velocity_mps": max(1.0, float(controls.local_water_velocity_mps.std(ddof=0))),
            "bubble_drive_width_A": max(1.0, float(controls.bubble_drive_width_A.std(ddof=0))),
        }
        distance = sum(
            ((controls[key] - float(chosen[key])) / value) ** 2
            for key, value in scale.items()
        )
        control = controls.loc[distance.idxmin()].to_dict()
        control.update({
            "event_id": event.event_id, "process": event.process,
            "onset_time_ps": float(event.onset_time_ps), "row_class": "MATCHED_CONTROL",
        })
        output.append(control)
    return pd.DataFrame(output)


def _paired_summary(features: pd.DataFrame) -> list[dict[str, object]]:
    metrics = (
        "fixed_turnover", "moving_turnover", "swept_window_turnover",
        "fixed_cohort_survival", "retained_water_drive_displacement_A",
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260922)
    for metric in metrics:
        if features.empty or not {"event_id", "row_class", metric}.issubset(features.columns):
            pivot = pd.DataFrame(columns=["EVENT", "MATCHED_CONTROL"])
        else:
            pivot = features.pivot_table(index="event_id", columns="row_class", values=metric)
            for column in ("EVENT", "MATCHED_CONTROL"):
                if column not in pivot:
                    pivot[column] = np.nan
            pivot = pivot.dropna(subset=["EVENT", "MATCHED_CONTROL"])
        differences = (pivot.EVENT - pivot.MATCHED_CONTROL).to_numpy(float)
        if len(differences):
            boot = np.mean(rng.choice(differences, size=(5000, len(differences)), replace=True), axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
        else:
            low = high = math.nan
        rows.append({
            "metric": metric, "paired_events": len(differences),
            "event_mean": float(pivot.EVENT.mean()) if len(pivot) else math.nan,
            "control_mean": float(pivot.MATCHED_CONTROL.mean()) if len(pivot) else math.nan,
            "paired_mean_difference": float(np.mean(differences)) if len(differences) else math.nan,
            "event_cluster_bootstrap_ci_low": float(low),
            "event_cluster_bootstrap_ci_high": float(high),
            "uncertainty_boundary": "event-cluster only; not independent-parent uncertainty",
        })
    return rows


def _block_shift_null(risk: pd.DataFrame) -> list[dict[str, object]]:
    metrics = ("fixed_turnover", "moving_turnover", "swept_window_turnover")
    groups = list(risk.groupby(["branch_id", "patch_id", "edge", "block_id"], sort=True))
    true_label = risk.outcome_event.to_numpy(int)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260922)
    for metric in metrics:
        values = risk[metric].to_numpy(float)
        finite = np.isfinite(values)
        true_effect = (
            float(np.mean(values[(true_label == 1) & finite]) - np.mean(values[(true_label == 0) & finite]))
            if np.any((true_label == 1) & finite) and np.any((true_label == 0) & finite)
            else math.nan
        )
        null = []
        for _ in range(199):
            shifted = np.zeros(len(risk), dtype=int)
            for _, table in groups:
                indices = table.index.to_numpy(int)
                labels = true_label[indices]
                if len(indices) < 3 or not np.any(labels):
                    shifted[indices] = labels
                    continue
                offset = int(rng.integers(1, len(indices)))
                shifted[indices] = np.roll(labels, offset)
            if np.any((shifted == 1) & finite) and np.any((shifted == 0) & finite):
                null.append(float(
                    np.mean(values[(shifted == 1) & finite])
                    - np.mean(values[(shifted == 0) & finite])
                ))
        p_value = (
            (1 + sum(value >= true_effect for value in null)) / (1 + len(null))
            if null and math.isfinite(true_effect) else math.nan
        )
        rows.append({
            "metric": metric, "true_event_minus_nonevent": true_effect,
            "null_mean": float(np.mean(null)) if null else math.nan,
            "null_q025": float(np.quantile(null, 0.025)) if null else math.nan,
            "null_q975": float(np.quantile(null, 0.975)) if null else math.nan,
            "empirical_one_sided_p": p_value, "null_shifts": len(null),
            "null_definition": "within-branch-patch-edge-20ps-block circular label shifts",
        })
    # Benjamini-Hochberg over the three predeclared observer metrics.
    finite_indices = [index for index, row in enumerate(rows) if math.isfinite(float(row["empirical_one_sided_p"]))]
    ordered = sorted(finite_indices, key=lambda index: float(rows[index]["empirical_one_sided_p"]))
    running = 1.0
    for rank_from_end, index in enumerate(reversed(ordered), 1):
        rank = len(ordered) - rank_from_end + 1
        adjusted = min(1.0, float(rows[index]["empirical_one_sided_p"]) * len(ordered) / rank)
        running = min(running, adjusted)
        rows[index]["bh_q_three_observers"] = running
    return rows


def _model_scores(risk: pd.DataFrame) -> list[dict[str, object]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_features = [
        "edge_history_displacement_A", "contact_n2_in_patch",
        "global_film_water_oxygen_count", "bubble_height_A", "bubble_drive_width_A",
        "local_water_velocity_mps", "local_oxygen_density_A3",
        "fixed_water_start_count", "site_ch3_weight_fraction",
    ]
    models = {
        "M0": base_features,
        "M_FIXED": base_features + ["fixed_turnover"],
        "M_MOVING": base_features + ["moving_turnover"],
        "M_SWEPT": base_features + ["swept_window_turnover"],
    }
    groups = {
        "held_branch": risk.branch_id.astype(str),
        "held_block": risk.branch_id.astype(str) + ":" + risk.block_id.astype(str),
        "held_patch": risk.branch_id.astype(str) + ":" + risk.patch_id.astype(str),
    }
    y = risk.outcome_event.to_numpy(int)
    rows: list[dict[str, object]] = []
    for scheme, labels in groups.items():
        for model_name, features in models.items():
            predictions = np.full(len(risk), np.nan, dtype=float)
            for label in sorted(set(labels)):
                test = labels == label
                train = ~test
                if scheme == "held_block" and np.any(test):
                    branch = str(risk.loc[test, "branch_id"].iloc[0])
                    lower = float(risk.loc[test, "risk_time_ps"].min()) - 2.0
                    upper = float(risk.loc[test, "risk_time_ps"].max()) + 2.0
                    train &= ~(
                        (risk.branch_id.astype(str) == branch)
                        & (risk.risk_time_ps >= lower) & (risk.risk_time_ps <= upper)
                    )
                if len(np.unique(y[train])) < 2 or not np.any(test):
                    continue
                estimator = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=2000),
                )
                estimator.fit(risk.loc[train, features], y[train])
                predictions[test] = estimator.predict_proba(risk.loc[test, features])[:, 1]
            valid = np.isfinite(predictions)
            if not np.any(valid) or len(np.unique(y[valid])) < 2:
                continue
            rows.append({
                "scheme": scheme, "model": model_name, "risk_rows": int(np.sum(valid)),
                "positive_risk_rows": int(np.sum(y[valid])),
                "unique_events": int(risk.loc[valid & (y == 1), "outcome_event_id"].nunique()),
                "log_loss": float(log_loss(y[valid], predictions[valid], labels=[0, 1])),
                "brier_score": float(brier_score_loss(y[valid], predictions[valid])),
                "average_precision": float(average_precision_score(y[valid], predictions[valid])),
                "boundary": "overlapping risk rows; diagnostic, not independent-event inference",
            })
    return rows


def _cadence_identity_audit(cases: Sequence[Mapping[str, object]], base: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs = (("20fs", "40fs"), ("20fs", "100fs"), ("40fs", "100fs"))
    match_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for case in cases:
        tables = {}
        for name in ("20fs", "40fs", "100fs"):
            table = _read_tsv(resolve_path(case[f"events_{name}"], base))
            tables[name] = table[
                (table.validity_status == "VALID")
                & (table.branch_id.astype(str) == str(case["branch_id"]))
            ].reset_index(drop=True)
        for reference_name, candidate_name in pairs:
            matches, metrics = match_events_one_to_one(
                tables[reference_name], tables[candidate_name], tolerance_ps=0.5
            )
            for row in matches:
                row.update({
                    "case_id": case["case_id"],
                    "reference_cadence": reference_name,
                    "candidate_cadence": candidate_name,
                })
                match_rows.append(row)
            metric_rows.append({
                "case_id": case["case_id"], "branch_id": case["branch_id"],
                "reference_cadence": reference_name,
                "candidate_cadence": candidate_name,
                **metrics, "time_tolerance_ps": 0.5,
            })
    return match_rows, metric_rows


def _failure_force_audit(
    raw: Mapping[str, object], base: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    failure = raw.get("failure_force_audit")
    if not isinstance(failure, dict):
        return [], [], {"status": "NOT_REQUESTED"}
    lower = int(failure["audit_start_step"])
    upper = int(failure["failure_step"])
    candidates: list[dict[str, object]] = []
    last_steps: dict[str, int] = {}
    for stream in failure["streams"]:
        path = resolve_path(stream["path"], base)
        role = str(stream["role"])
        def accept(step: int, lo: int = lower, hi: int = upper) -> bool:
            return lo <= step <= hi
        for frame in iter_lammps_dump_records_selected(
            path, accept_timestep=accept, stop_after_timestep=upper
        ):
            required = ("id", "type", "x", "y", "z", "f_RAW[1]", "f_RAW[2]", "f_RAW[3]")
            missing = [name for name in required if name not in frame.atom_fields]
            if missing:
                raise ValueError(f"{role} is missing {missing}")
            column = {name: frame.atom_fields.index(name) for name in required}
            for row in frame.atom_rows:
                force = np.asarray([
                    float(row[column["f_RAW[1]"]]), float(row[column["f_RAW[2]"]]),
                    float(row[column["f_RAW[3]"]]),
                ])
                magnitude = float(np.linalg.norm(force))
                if magnitude < float(failure.get("record_threshold_eV_per_A", 5.0)):
                    continue
                candidates.append({
                    "stream": role, "step": frame.timestep,
                    "time_before_failure_ps": (upper - frame.timestep)
                    * float(raw["timestep_fs"]) / 1000.0,
                    "atom_id": int(row[column["id"]]), "atom_type": int(row[column["type"]]),
                    "x_A": float(row[column["x"]]), "y_A": float(row[column["y"]]),
                    "z_A": float(row[column["z"]]), "raw_force_magnitude_eV_per_A": magnitude,
                })
            last_steps[role] = frame.timestep
    candidates.sort(key=lambda row: (-float(row["raw_force_magnitude_eV_per_A"]), -int(row["step"])))
    candidates = candidates[:200]
    scalar_rows: list[dict[str, object]] = []
    scalar_path = resolve_path(failure["scalar_table"], base)
    with scalar_path.open(encoding="utf-8") as handle:
        first = handle.readline()
        header = handle.readline().lstrip("# ").split()
        if "TimeStep" not in header or "c_FrawMax" not in header:
            raise ValueError(f"Unexpected scalar-table header in {scalar_path}: {first.strip()}")
        step_column = header.index("TimeStep")
        force_column = header.index("c_FrawMax")
        for line in handle:
            parts = line.split()
            if len(parts) != len(header):
                continue
            step = int(float(parts[step_column]))
            if lower <= step <= upper:
                scalar_rows.append({
                    "step": step,
                    "time_before_failure_ps": (upper - step) * float(raw["timestep_fs"]) / 1000.0,
                    "maximum_raw_force_eV_per_A": float(parts[force_column]),
                })
    scalar_rows = scalar_rows[-400:]
    maximum = float(candidates[0]["raw_force_magnitude_eV_per_A"]) if candidates else math.nan
    scalar_maximum = max(
        (float(row["maximum_raw_force_eV_per_A"]) for row in scalar_rows), default=math.nan
    )
    saved_trigger = math.isfinite(maximum) and maximum > float(failure["guard_eV_per_A"])
    summary = {
        "status": "PASS", "failure_step": upper, "audited_start_step": lower,
        "stream_last_steps": last_steps, "maximum_saved_force_eV_per_A": maximum,
        "maximum_scalar_force_before_guard_eV_per_A": scalar_maximum,
        "last_scalar_step": int(scalar_rows[-1]["step"]) if scalar_rows else None,
        "guard_eV_per_A": float(failure["guard_eV_per_A"]),
        "trigger_atom_localized_at_saved_cadence": bool(saved_trigger),
        "decision": "SAVED_TRIGGER_LOCALIZED" if saved_trigger else "NOT_LOCALIZED_AT_SAVED_CADENCE",
        "boundary": (
            "The guard was evaluated every 10 integration steps; a 20 fs dump can miss the "
            "instantaneous trigger. Absence from saved frames does not resolve model validity."
        ),
    }
    return candidates, scalar_rows, summary


def _motif_tables(
    motifs: Sequence[tuple[Mapping[str, object], dict[int, dict[str, float]]]],
    site_ids: np.ndarray, site_types: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    site_rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    type_by_id = {int(atom_id): str(site_type) for atom_id, site_type in zip(site_ids, site_types)}
    for case, values in motifs:
        for atom_id in site_ids:
            record = values[int(atom_id)]
            site_type = type_by_id[int(atom_id)]
            site_rows.append({
                "case_id": case["case_id"], "branch_id": case["branch_id"],
                "direction": case["direction"], "site_id": int(atom_id),
                "site_type": site_type, **record,
            })
            aggregate[site_type]["opportunity_weight_ps"] += record["opportunity_weight_ps"]
            aggregate[site_type]["event_weight"] += record["event_weight"]
    rate_rows = []
    for site_type in ("SiOH", "CH3"):
        opportunity = aggregate[site_type]["opportunity_weight_ps"]
        events = aggregate[site_type]["event_weight"]
        rate_rows.append({
            "site_type": site_type, "weighted_event_count": events,
            "weighted_opportunity_ps": opportunity,
            "weighted_events_per_1000_ps": 1000.0 * events / opportunity if opportunity else math.nan,
        })
    if all(row["weighted_opportunity_ps"] > 0 for row in rate_rows):
        rate = {row["site_type"]: row["weighted_events_per_1000_ps"] for row in rate_rows}
        ratio = rate["CH3"] / rate["SiOH"] if rate["SiOH"] > 0 else math.inf
        for row in rate_rows:
            row["ch3_to_sioh_rate_ratio"] = ratio
            row["claim_boundary"] = "same 2D PBC kernel for event and opportunity; correlated single-parent data"
    return site_rows, rate_rows


def _write_report(output: Path, summary: Mapping[str, object]) -> None:
    decision = summary["decision"]
    fixed = summary.get("fixed_turnover_shift_q")
    moving = summary.get("moving_turnover_shift_q")
    swept = summary.get("swept_turnover_shift_q")
    output.joinpath("REPORT.md").write_text(
        "# High-cadence causal-observer audit\n\n"
        f"Decision: `{decision}`.\n\n"
        "This audit uses past-only event onsets, a common 0.10 ps saved-frame cadence, "
        "a 0.20 ps predictor safety gap, and separate substrate-fixed, moving-edge, and "
        "frozen-water/swept-window observers. Persistence uses future geometry only to "
        "confirm an onset and never to construct predictor features.\n\n"
        f"Multiplicity-aware block-shift q values: fixed={fixed}, moving={moving}, swept={swept}.\n\n"
        "The statistical unit remains one matched parent with two driven branches. "
        "Event-cluster bootstrap intervals and held-out risk models do not estimate "
        "independent-preparation uncertainty or prove causal mediation.\n",
        encoding="utf-8",
    )


def _plot_summary(
    output: Path, paired: Sequence[Mapping[str, object]],
    shift: Sequence[Mapping[str, object]], cadence: Sequence[Mapping[str, object]],
    gas: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    turnover = [row for row in paired if "turnover" in str(row["metric"])]
    labels = [str(row["metric"]).replace("_turnover", "") for row in turnover]
    x = np.arange(len(turnover))
    axes[0, 0].bar(x - 0.18, [float(row["event_mean"]) for row in turnover], 0.36,
                   color="#E45756", label="pre-onset opportunity")
    axes[0, 0].bar(x + 0.18, [float(row["control_mean"]) for row in turnover], 0.36,
                   color="#4C78A8", label="matched non-event")
    axes[0, 0].set_xticks(x, labels); axes[0, 0].set_ylabel("Membership turnover")
    axes[0, 0].set_title("Observer decomposition"); axes[0, 0].legend(frameon=False, fontsize=8)

    shift_map = {str(row["metric"]): row for row in shift}
    true = [float(shift_map[str(row["metric"])]["true_event_minus_nonevent"]) for row in turnover]
    null_mean = [float(shift_map[str(row["metric"])]["null_mean"]) for row in turnover]
    low = [float(shift_map[str(row["metric"])]["null_q025"]) for row in turnover]
    high = [float(shift_map[str(row["metric"])]["null_q975"]) for row in turnover]
    axes[0, 1].errorbar(
        x, null_mean,
        yerr=[np.asarray(null_mean)-np.asarray(low), np.asarray(high)-np.asarray(null_mean)],
        fmt="o", color="#9D9D9D", capsize=4, label="shift null 95% envelope",
    )
    axes[0, 1].scatter(x, true, color="#54A24B", marker="D", label="observed")
    axes[0, 1].axhline(0.0, color="black", lw=0.8)
    axes[0, 1].set_xticks(x, labels); axes[0, 1].set_ylabel("Event − non-event")
    axes[0, 1].set_title("20 ps block-shift null envelope")
    axes[0, 1].legend(frameon=False, fontsize=8)

    cadence_labels = [
        f"{row['branch_id']}\n{row['reference_cadence']}→{row['candidate_cadence']}"
        for row in cadence
    ]
    axes[1, 0].bar(np.arange(len(cadence)), [float(row["jaccard"]) for row in cadence],
                   color="#F58518")
    axes[1, 0].set_xticks(np.arange(len(cadence)), cadence_labels, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_ylim(0.0, 1.0); axes[1, 0].set_ylabel("Matched-event Jaccard")
    axes[1, 0].set_title("Event identity across decimation")

    finite = gas[np.isfinite(gas.total_center_step_A) & np.isfinite(gas.retained_material_step_A)]
    for branch, table in finite.groupby("branch_id"):
        axes[1, 1].scatter(table.retained_material_step_A, table.total_center_step_A,
                           s=6, alpha=0.35, label=str(branch))
    limits = axes[1, 1].get_xlim()
    axes[1, 1].plot(limits, limits, "--", color="black", lw=0.8)
    axes[1, 1].set_xlabel("Retained-material step (Å)")
    axes[1, 1].set_ylabel("Main-cluster COM step (Å)")
    axes[1, 1].set_title("Material motion vs membership effect")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "causal_observer_audit.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    for key in (
        "timestep_fs", "sample_stride_steps", "sample_interval_ps",
        "trajectory_duration_ps", "history_ps", "safety_gap_ps",
        "prediction_horizon_ps", "risk_stride_ps", "onset_smoothing_ps",
        "onset_reference_ps", "persistence_ps", "persistence_fraction",
        "refractory_ps", "minimum_displacement_A", "f0_calibration_end_ps",
        "minimum_patch_coverage", "edge_water_halfwidth_A", "near_surface_height_A",
        "cluster_cutoff_A", "contact_cutoff_A", "site_kernel_sigma_A",
        "site_kernel_cutoff_A", "block_ps", "oh_cutoff_A", "torque_stride_ps",
    ):
        if float(raw.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if not 0.0 < float(raw["f0_quantile"]) < 1.0:
        raise ValueError("f0_quantile must lie in (0, 1)")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    base = contract_path.parent
    model_path = resolve_path(raw["surface_model_data"], base)
    registry_path = resolve_path(raw["surface_site_registry"], base)
    site_ids, site_xy, site_types = _load_site_data(model_path, registry_path)
    calibration, onsets, coverage = _calibrate_and_detect_onsets(raw, cases, base)
    cadence_matches, cadence_metrics = _cadence_identity_audit(cases, base)
    risk_rows: list[dict[str, object]] = []
    gas_rows: list[dict[str, object]] = []
    torque_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    motif_values = []
    manifest = [{
        "role": "contract", "path": str(contract_path), "size_bytes": contract_path.stat().st_size,
        "sha256": sha256(contract_path),
    }]
    for path_role, path in (("surface_model_data", model_path), ("surface_site_registry", registry_path)):
        manifest.append({
            "role": path_role, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for case in cases:
        case_onsets = onsets[
            (onsets.case_id == case["case_id"]) & (onsets.branch_id == case["branch_id"])
        ].copy()
        case_calibration = calibration[
            (calibration.case_id == case["case_id"])
            & (calibration.branch_id == case["branch_id"])
        ].copy()
        arrays = _load_branch_arrays(case, raw, base)
        rows, motif, points = _risk_rows_for_case(
            case, arrays, raw, base, case_onsets, site_ids, site_xy, site_types,
            case_calibration,
        )
        risk_rows.extend(rows); point_rows.extend(points); motif_values.append((case, motif))
        gas_rows.extend(_gas_transport(case, arrays, raw))
        torque_rows.append(_torque_summary(case, arrays, raw))
        for entry in case["state_trajectories"]:
            path = resolve_path(entry["path"] if isinstance(entry, dict) else entry, base)
            manifest.append({
                "role": f"{case['branch_id']}:state_trajectory", "path": str(path),
                "size_bytes": path.stat().st_size, "sha256": str(entry.get("sha256", ""))
                if isinstance(entry, dict) else "",
            })
        for key in (
            "patch_frames_100fs", "baseline_patch_frames_100fs", "geometry_frames_100fs",
            "events_20fs", "events_40fs", "events_100fs",
        ):
            path = resolve_path(case[key], base)
            manifest.append({
                "role": f"{case['branch_id']}:{key}", "path": str(path),
                "size_bytes": path.stat().st_size, "sha256": sha256(path),
            })
    failure_contract = raw.get("failure_force_audit")
    if isinstance(failure_contract, dict):
        for stream in failure_contract["streams"]:
            path = resolve_path(stream["path"], base)
            manifest.append({
                "role": f"failure_stream:{stream['role']}", "path": str(path),
                "size_bytes": path.stat().st_size, "sha256": str(stream.get("sha256", "")),
            })
        scalar_path = resolve_path(failure_contract["scalar_table"], base)
        manifest.append({
            "role": "failure_scalar_table", "path": str(scalar_path),
            "size_bytes": scalar_path.stat().st_size,
            "sha256": str(failure_contract.get("scalar_table_sha256", "")),
        })
    risk = pd.DataFrame(risk_rows).reset_index(drop=True)
    event_features = _event_feature_rows(risk, onsets, float(raw["refractory_ps"]))
    paired = _paired_summary(event_features)
    shift = _block_shift_null(risk)
    model = _model_scores(risk)
    site_rows, rate_rows = _motif_tables(motif_values, site_ids, site_types)
    failure_rows, failure_scalar_rows, failure_summary = _failure_force_audit(raw, base)
    output.mkdir(parents=True)
    calibration.to_csv(output / "PAST-ONLY-CALIBRATION.tsv", sep="\t", index=False)
    onsets.to_csv(output / "PAST-ONLY-ONSETS.tsv", sep="\t", index=False)
    coverage.to_csv(output / "OBSERVER-COVERAGE.tsv", sep="\t", index=False)
    pd.DataFrame(cadence_matches).to_csv(output / "CADENCE-EVENT-MATCHES.tsv", sep="\t", index=False)
    pd.DataFrame(cadence_metrics).to_csv(output / "CADENCE-EVENT-IDENTITY.tsv", sep="\t", index=False)
    risk.to_csv(output / "RISK-OPPORTUNITIES.tsv", sep="\t", index=False)
    event_features.to_csv(output / "EVENT-FIXED-WATER.tsv", sep="\t", index=False)
    write_tsv(output / "OBSERVER-PAIRED-SUMMARY.tsv", paired, tuple(paired[0]))
    write_tsv(output / "BLOCK-SHIFT-NULL.tsv", shift, tuple(shift[0]))
    write_tsv(output / "MODEL-SCORES.tsv", model, tuple(model[0]) if model else ("scheme",))
    write_tsv(output / "GAS-IDENTITY-TRANSPORT.tsv", gas_rows, tuple(gas_rows[0]))
    write_tsv(output / "FORCING-TORQUE.tsv", torque_rows, tuple(torque_rows[0]))
    if failure_rows:
        write_tsv(output / "FAILED-FORCE-CANDIDATES.tsv", failure_rows, tuple(failure_rows[0]))
    if failure_scalar_rows:
        write_tsv(
            output / "FAILED-FORCE-SCALAR-TRACE.tsv", failure_scalar_rows,
            tuple(failure_scalar_rows[0]),
        )
    (output / "FAILED-FORCE-AUDIT.json").write_text(
        json.dumps(failure_summary, indent=2) + "\n", encoding="utf-8"
    )
    write_tsv(output / "EDGE-POINT-COVERAGE.tsv", point_rows, tuple(point_rows[0]))
    write_tsv(output / "SITE-OPPORTUNITY.tsv", site_rows, tuple(site_rows[0]))
    write_tsv(output / "MOTIF-RATES.tsv", rate_rows, tuple(rate_rows[0]))
    write_tsv(output / "input_manifest.tsv", manifest, ("role", "path", "size_bytes", "sha256"))
    shift_by_metric = {row["metric"]: row for row in shift}
    fixed_q = float(shift_by_metric["fixed_turnover"].get("bh_q_three_observers", math.nan))
    moving_q = float(shift_by_metric["moving_turnover"].get("bh_q_three_observers", math.nan))
    swept_q = float(shift_by_metric["swept_window_turnover"].get("bh_q_three_observers", math.nan))
    driven_events = onsets[(onsets.source == "driven")]
    f0_events = onsets[(onsets.source == "f0_heldout")]
    fixed_row = next(row for row in paired if row["metric"] == "fixed_turnover")
    moving_row = next(row for row in paired if row["metric"] == "moving_turnover")
    swept_row = next(row for row in paired if row["metric"] == "swept_window_turnover")
    fixed_signal = (
        math.isfinite(fixed_q) and fixed_q <= 0.10
        and float(fixed_row["paired_mean_difference"]) > 0.0
        and float(fixed_row["event_cluster_bootstrap_ci_low"]) > 0.0
    )
    sweep_dominates = (
        float(swept_row["paired_mean_difference"]) >= 0.8
        * float(moving_row["paired_mean_difference"])
        if math.isfinite(float(moving_row["paired_mean_difference"]))
        else False
    )
    f0_rate = len(f0_events) / max(1.0, len(cases) * (
        float(raw["trajectory_duration_ps"]) - float(raw["f0_calibration_end_ps"])
    ))
    driven_rate = len(driven_events) / max(1.0, len(cases) * float(raw["trajectory_duration_ps"]))
    if not len(driven_events):
        decision = "HOLD_AND_REDEFINE"
    elif fixed_signal and not sweep_dominates and driven_rate > f0_rate:
        decision = "PROCEED_TO_PATHWAY_TEST"
    else:
        decision = "PROCEED_TO_LOADING_ONLY"
    summary = {
        "status": "PASS", "decision": decision,
        "driven_past_only_events": int(len(driven_events)),
        "f0_heldout_past_only_events": int(len(f0_events)),
        "driven_event_rate_per_ps": driven_rate,
        "f0_heldout_event_rate_per_ps": f0_rate,
        "risk_rows": int(len(risk)),
        "positive_risk_rows": int(risk.outcome_event.sum()),
        "unique_positive_events": int(risk.loc[risk.outcome_event == 1, "outcome_event_id"].nunique()),
        "fixed_turnover_shift_q": fixed_q,
        "moving_turnover_shift_q": moving_q,
        "swept_turnover_shift_q": swept_q,
        "fixed_turnover_paired_difference": fixed_row["paired_mean_difference"],
        "moving_turnover_paired_difference": moving_row["paired_mean_difference"],
        "swept_turnover_paired_difference": swept_row["paired_mean_difference"],
        "sweep_dominates_moving": bool(sweep_dominates),
        "failure_force_audit_decision": failure_summary.get("decision", failure_summary.get("status")),
        "claim_boundary": (
            "One matched parent with two driven branches; observer falsification and predictive "
            "diagnostics do not establish causal mediation or independent-preparation uncertainty."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output, summary)
    _plot_summary(output, paired, shift, cadence_metrics, pd.DataFrame(gas_rows))
    write_output_hashes(output)
    return summary
