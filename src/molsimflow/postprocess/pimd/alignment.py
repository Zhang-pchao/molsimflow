"""Strict complete-path assembly, restart selection and time alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .contract import cv_column_names, diagnostic_cv_spec
from .io import field
from .observables import piecewise_derived_coordinate_spec, validate_piecewise_logdistance_printed
from .types import AlignedPaths, PreparedInputs


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


require = _require


def restart_unique_indices(
    frame_ids: Sequence[float] | np.ndarray,
    *,
    policy: str = "keep_first",
) -> Tuple[np.ndarray, int]:
    """Return indices after removing adjacent restart-seam duplicates."""

    ids = np.asarray(frame_ids, dtype=float)
    _require(ids.ndim == 1 and ids.size > 0, "frame ids must be a nonempty vector")
    _require(np.isfinite(ids).all(), "frame ids must be finite")
    _require(np.all(np.diff(ids) >= 0.0), "frame ids decrease across a restart seam")
    _require(policy in {"error", "keep_first", "keep_last"}, "invalid restart policy")
    duplicate = np.r_[False, ids[1:] == ids[:-1]]
    count = int(np.count_nonzero(duplicate))
    if count == 0:
        return np.arange(ids.size), 0
    if policy == "error":
        raise ValueError("duplicate restart-seam frame ids")
    if policy == "keep_first":
        keep = ~duplicate
    else:
        keep = ~np.r_[ids[:-1] == ids[1:], False]
    return np.flatnonzero(keep), count


def assemble_bead_frames(
    frame_ids: Sequence[int] | np.ndarray,
    bead_ids: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    expected_beads: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a long bead table into complete frame-by-bead arrays."""

    frames = np.asarray(frame_ids)
    beads = np.asarray(bead_ids)
    data = np.asarray(values, dtype=float)
    _require(frames.ndim == beads.ndim == 1, "frame and bead ids must be vectors")
    _require(data.ndim in {1, 2}, "invalid bead value table")
    _require(frames.size == beads.size == data.shape[0], "long-table row counts differ")
    _require(frames.size > 0, "invalid bead value table")
    _require(np.isfinite(data).all(), "bead values must be finite")
    _require(
        isinstance(expected_beads, (int, np.integer))
        and not isinstance(expected_beads, (bool, np.bool_))
        and expected_beads > 0,
        "expected_beads must be a positive integer",
    )

    ordered_frames = []
    groups: Dict[object, list[int]] = {}
    for row, frame in enumerate(frames.tolist()):
        if frame not in groups:
            ordered_frames.append(frame)
            groups[frame] = []
        groups[frame].append(row)

    first_rows = groups[ordered_frames[0]]
    bead_labels = np.sort(beads[first_rows])
    _require(len(bead_labels) == int(expected_beads), "first frame has missing beads")
    _require(len(np.unique(bead_labels)) == len(bead_labels), "duplicate bead in first frame")
    assembled = []
    for frame in ordered_frames:
        rows = groups[frame]
        labels = beads[rows]
        _require(len(rows) == int(expected_beads), f"frame {frame} has missing beads")
        _require(len(np.unique(labels)) == len(labels), f"frame {frame} has duplicate beads")
        _require(np.array_equal(np.sort(labels), bead_labels), f"frame {frame} bead ids differ")
        order = np.argsort(labels)
        assembled.append(data[np.asarray(rows)[order]])
    return np.asarray(ordered_frames), bead_labels, np.asarray(assembled)


def time_window_mask(
    values: Sequence[float] | np.ndarray,
    first: float,
    last: float,
    *,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Select a finite closed time interval with an explicit tolerance."""
    times = np.asarray(values, dtype=float)
    require(times.ndim == 1 and times.size > 0, "invalid time axis")
    require(np.isfinite(times).all(), "non-finite time axis")
    require(float(first) <= float(last), "inverted time window")
    require(float(tolerance) >= 0.0, "negative time-window tolerance")
    return (times >= float(first) - float(tolerance)) & (times <= float(last) + float(tolerance))


def time_scale_to_fs(value: object, label: str) -> float:
    """Validate one explicit conversion from an input time column to fs."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {label}") from None
    require(np.isfinite(scale) and scale > 0.0, f"invalid {label}")
    return scale


def time_values_to_fs(values: Sequence[float] | np.ndarray, scale: float) -> np.ndarray:
    """Apply one validated raw-time conversion scale."""
    return np.asarray(values, dtype=float) * scale


def aligned_time_indices(
    source: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Match each target to one distinct source frame within a finite tolerance.

    Missing or ambiguous matches raise instead of interpolating or reusing data.
    """
    source_times = np.asarray(source, dtype=float)
    target_times = np.asarray(target, dtype=float)
    require(source_times.ndim == target_times.ndim == 1, "invalid time-grid rank")
    require(source_times.size > 0 and target_times.size > 0, "empty time grid")
    require(np.isfinite(source_times).all(), "non-finite source time grid")
    require(np.isfinite(target_times).all(), "non-finite target time grid")
    require(np.all(np.diff(source_times) > 0.0), "source time grid is not strictly increasing")
    require(np.all(np.diff(target_times) > 0.0), "target time grid is not strictly increasing")
    tolerance = float(tolerance)
    require(np.isfinite(tolerance) and tolerance >= 0.0, "invalid time-grid tolerance")
    # Find the entire closed tolerance interval, not just its right neighbour.
    indices = np.searchsorted(source_times, target_times - tolerance, side="left")
    stops = np.searchsorted(source_times, target_times + tolerance, side="right")
    matches = stops - indices
    require(np.all(matches > 0), "target time is absent from source grid")
    require(np.all(matches == 1), "ambiguous time-grid match within tolerance")
    require(np.all(np.diff(indices) > 0), "target times map to the same source frame")
    return indices


def align_inputs(
    contract: Mapping[str, object], output: Path, profile: str, inputs: PreparedInputs
) -> AlignedPaths:
    from .reporting import default_cv_label

    source = inputs.source
    selection = inputs.selection
    reweight = inputs.reweight
    bias_mode = inputs.bias_mode
    sampling_time_scale_to_fs = inputs.sampling_time_scale_to_fs
    bead_time_scale_to_fs = inputs.bead_time_scale_to_fs
    bead_offset_fs = inputs.bead_offset_fs
    fields = inputs.fields
    centroid_data = inputs.centroid_data
    bead_tables = inputs.bead_tables
    restart_policy = str(source.get("restart_duplicate_policy", "keep_first"))
    centroid_keep, centroid_restart_duplicates = restart_unique_indices(
        field(centroid_data, fields, "time"), policy=restart_policy
    )
    centroid_data = centroid_data[centroid_keep]
    deduplicated_bead_tables = []
    bead_restart_duplicates = []
    for bead_fields, bead_data in bead_tables:
        bead_keep, duplicate_count = restart_unique_indices(
            field(bead_data, bead_fields, "time"), policy=restart_policy
        )
        deduplicated_bead_tables.append((bead_fields, bead_data[bead_keep]))
        bead_restart_duplicates.append(duplicate_count)
    bead_tables = deduplicated_bead_tables
    time_fs = time_values_to_fs(field(centroid_data, fields, "time"), sampling_time_scale_to_fs)
    first_time_fs = float(selection["first_time_ps"]) * 1000.0
    last_time_fs = float(selection["last_time_ps"]) * 1000.0
    all_steps = np.rint(time_fs / float(selection["timestep_fs"])).astype(int)
    selected_mask = (time_fs >= first_time_fs - 1e-9) & (time_fs <= last_time_fs + 1e-9)
    sample_stride_steps = int(selection.get("sample_stride_steps", 1))
    require(sample_stride_steps > 0, "sample_stride_steps must be positive")
    sample_phase_step = int(selection.get("sample_phase_step", 0))
    selected_mask &= (all_steps - sample_phase_step) % sample_stride_steps == 0
    selected = np.flatnonzero(selected_mask)
    require(selected.size == int(selection["expected_frames"]), "selected frame count mismatch")
    selected_time_ps = time_fs[selected] / 1000.0
    selected_steps = np.rint(time_fs[selected] / float(selection["timestep_fs"])).astype(int)
    cv_names = tuple(reweight["cv_names"])
    require(
        len(cv_names) in ({1, 2} if profile == "core" else {2}),
        "core analysis supports one or two CVs; water-ionization diagnostics require two",
    )
    sampling_cv_names = cv_column_names(reweight, "sampling_cv_names", cv_names)
    bead_cv_names = cv_column_names(reweight, "bead_cv_names", cv_names)
    sampling_label = str(source.get("sampling_label", "Centroid"))
    sampling_slug = str(source.get("sampling_slug", "centroid"))
    require(
        bool(sampling_slug) and Path(sampling_slug).name == sampling_slug, "invalid sampling_slug"
    )
    derived_spec = piecewise_derived_coordinate_spec(contract.get("derived_coordinate"), cv_names)
    require(
        derived_spec is None
        or (
            profile == "water_ionization_opes"
            and len(cv_names) == 2
            and bias_mode != "bead_density_shared"
        ),
        "derived_coordinate is available only in the two-CV water-ionization profile",
    )
    configured_labels = contract["plots"].get("cv_labels", {})
    cv_labels = tuple(str(configured_labels.get(name, default_cv_label(name))) for name in cv_names)
    centroid = np.column_stack(
        [field(centroid_data, fields, name)[selected] for name in sampling_cv_names]
    )
    bead_arrays = []
    bead_selections = []
    for bead_fields, bead_data in bead_tables:
        bead_time_fs = time_values_to_fs(
            field(bead_data, bead_fields, "time"), bead_time_scale_to_fs
        )
        bead_selected = aligned_time_indices(bead_time_fs + bead_offset_fs, time_fs[selected])
        bead_selections.append(bead_selected)
        bead_arrays.append(
            np.column_stack(
                [field(bead_data, bead_fields, name)[bead_selected] for name in bead_cv_names]
            )
        )
    beads = np.stack(bead_arrays, axis=1)
    if bias_mode == "bead_density_shared":
        centroid = np.mean(beads, axis=1)

    def selected_bead_field(column: str) -> np.ndarray:
        return np.stack(
            [
                field(bead_data, bead_fields, column)[bead_selected]
                for (bead_fields, bead_data), bead_selected in zip(bead_tables, bead_selections)
            ],
            axis=1,
        )

    diagnostic_spec = diagnostic_cv_spec(contract.get("diagnostic_cv"))
    diagnostic_sampling = None
    diagnostic_beads = None
    diagnostic_mean_error = None
    if diagnostic_spec is not None:
        diagnostic_beads = selected_bead_field(str(diagnostic_spec["bead_column"]))
        if bias_mode == "bead_density_shared":
            diagnostic_sampling = np.mean(diagnostic_beads, axis=1)
            diagnostic_mean_error = 0.0
        else:
            diagnostic_sampling = field(
                centroid_data, fields, str(diagnostic_spec["sampling_column"])
            )[selected]
            diagnostic_mean_error = float(
                np.max(np.abs(diagnostic_sampling - np.mean(diagnostic_beads, axis=1)))
            )
            require(
                diagnostic_mean_error <= float(diagnostic_spec["mean_tolerance"]),
                "printed diagnostic bead mean mismatch",
            )

    derived_centroid = None
    derived_beads = None
    derived_validation = None
    if derived_spec is not None:
        source_name = str(derived_spec["source"])
        printed_column = str(derived_spec["printed_column"])
        sampling_printed_column = str(derived_spec.get("sampling_printed_column", printed_column))
        bead_printed_column = str(derived_spec.get("bead_printed_column", printed_column))
        source_component = cv_names.index(source_name)
        derived_centroid = field(centroid_data, fields, sampling_printed_column)[selected]
        derived_beads = np.stack(
            [
                field(bead_data, bead_fields, bead_printed_column)[bead_selected]
                for (bead_fields, bead_data), bead_selected in zip(bead_tables, bead_selections)
            ],
            axis=1,
        )
        transform_options = {
            "switch": float(derived_spec["switch"]),
            "offset": float(derived_spec["offset"]),
            "linear_shift": float(derived_spec["linear_shift"]),
            "log_scale": float(derived_spec["log_scale"]),
            "log_reference": float(derived_spec["log_reference"]),
        }
        centroid_validation = validate_piecewise_logdistance_printed(
            centroid[:, source_component],
            derived_centroid,
            tolerance=float(derived_spec["printed_transform_tolerance"]),
            **transform_options,
        )
        bead_validations = [
            validate_piecewise_logdistance_printed(
                beads[:, bead, source_component],
                derived_beads[:, bead],
                tolerance=float(derived_spec["printed_transform_tolerance"]),
                **transform_options,
            )
            for bead in range(beads.shape[1])
        ]
        derived_validation = {
            "centroid": centroid_validation,
            "bead_maximum_absolute_error": max(
                value["maximum_absolute_error"] for value in bead_validations
            ),
            "all_maximum_absolute_error": max(
                [centroid_validation["maximum_absolute_error"]]
                + [value["maximum_absolute_error"] for value in bead_validations]
            ),
        }
    return AlignedPaths(
        restart_policy=restart_policy,
        centroid_restart_duplicates=centroid_restart_duplicates,
        bead_restart_duplicates=bead_restart_duplicates,
        centroid_data=centroid_data,
        time_fs=time_fs,
        first_time_fs=first_time_fs,
        last_time_fs=last_time_fs,
        selected=selected,
        selected_time_ps=selected_time_ps,
        selected_steps=selected_steps,
        cv_names=cv_names,
        sampling_cv_names=sampling_cv_names,
        bead_cv_names=bead_cv_names,
        sampling_label=sampling_label,
        sampling_slug=sampling_slug,
        derived_spec=derived_spec,
        cv_labels=cv_labels,
        centroid=centroid,
        beads=beads,
        selected_bead_field=selected_bead_field,
        diagnostic_spec=diagnostic_spec,
        diagnostic_sampling=diagnostic_sampling,
        diagnostic_beads=diagnostic_beads,
        diagnostic_mean_error=diagnostic_mean_error,
        derived_centroid=derived_centroid,
        derived_beads=derived_beads,
        derived_validation=derived_validation,
    )
