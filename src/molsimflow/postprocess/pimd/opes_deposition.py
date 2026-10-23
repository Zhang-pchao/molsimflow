"""Read selected OPES_METAD KERNELS batches and compare deposition weights.

Time values use the units declared by the producer's PLUMED input. This module
does not assume a particular host, CV name, bead count, or trajectory layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OpesKernelBatch:
    time: float
    cv_names: tuple[str, ...]
    centers: np.ndarray
    sigmas: np.ndarray
    heights: np.ndarray
    logweights: np.ndarray


@dataclass(frozen=True)
class OpesDepositionComparison:
    maximum_cv_match_error: float
    maximum_local_logweight_error: float
    maximum_common_path_logweight_error: float
    logged_logweight_spread: float
    local_logweight_spread: float
    common_path_logweight: float
    logged_by_bead: np.ndarray


def _kernel_cv_names(fields: tuple[str, ...]) -> tuple[str, ...]:
    if (len(fields) < 5 or fields[0] != "time"
            or fields[-2:] != ("height", "logweight")
            or (len(fields) - 3) % 2):
        raise ValueError("unsupported OPES_METAD KERNELS fields")
    count = (len(fields) - 3) // 2
    names = fields[1 : 1 + count]
    if (len(set(names)) != count
            or fields[1 + count : -2] != tuple(f"sigma_{name}" for name in names)):
        raise ValueError("KERNELS CV and sigma fields do not match")
    return names


def read_opes_kernel_batches(
    path: Path, selected_times: Sequence[float], *, expected_rows_per_time: int | None = None,
) -> tuple[OpesKernelBatch, ...]:
    """Stream exact selected deposition times from an append-only KERNELS file."""
    wanted = np.asarray(selected_times, dtype=float)
    if (wanted.ndim != 1 or wanted.size == 0 or not np.isfinite(wanted).all()
            or np.any(np.diff(wanted) <= 0)
            or (expected_rows_per_time is not None and (
                not isinstance(expected_rows_per_time, int)
                or isinstance(expected_rows_per_time, bool)
                or expected_rows_per_time <= 0
            ))):
        raise ValueError("selected times and expected row count must be valid")
    selected = {float(time): [] for time in wanted}
    fields: tuple[str, ...] | None = None
    names: tuple[str, ...] | None = None
    last_time = -np.inf
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#! FIELDS "):
                current = tuple(line.split()[2:])
                current_names = _kernel_cv_names(current)
                if fields is not None and current != fields:
                    raise ValueError("KERNELS fields changed within the file")
                fields, names = current, current_names
                continue
            if line.startswith("#") or not line.strip():
                continue
            if fields is None:
                raise ValueError("KERNELS data precedes fields")
            try:
                values = np.asarray([float(item) for item in line.split()], dtype=float)
            except ValueError as exc:
                raise ValueError("invalid KERNELS numeric row") from exc
            if values.shape != (len(fields),) or not np.isfinite(values).all():
                raise ValueError("invalid KERNELS row width or finite values")
            time = float(values[0])
            if time < last_time:
                raise ValueError("KERNELS times are not nondecreasing")
            last_time = time
            if time > wanted[-1]:
                break
            if time in selected:
                selected[time].append(values)
    if fields is None or names is None or any(not rows for rows in selected.values()):
        raise ValueError("selected KERNELS deposition time is missing")
    count = len(names)
    result = []
    for time, rows in selected.items():
        data = np.vstack(rows)
        if (expected_rows_per_time is not None and len(data) != expected_rows_per_time):
            raise ValueError("selected KERNELS batch has an unexpected row count")
        sigmas = data[:, 1 + count : -2].copy()
        heights = data[:, -2].copy()
        if np.any(sigmas <= 0) or np.any(heights < 0):
            raise ValueError("invalid KERNELS widths or heights")
        result.append(OpesKernelBatch(
            time, names, data[:, 1 : 1 + count].copy(), sigmas,
            heights, data[:, -1].copy(),
        ))
    return tuple(result)


def compare_opes_deposition_weights(
    batch: OpesKernelBatch,
    bead_cv: np.ndarray,
    local_reference_bias_eV: np.ndarray,
    *,
    common_path_bias_eV: float,
    kbt_eV: float,
    cv_tolerance: float = 1e-10,
) -> OpesDepositionComparison:
    """Match logged kernels to beads and compare local and shared-path logweights.

    The caller supplies the actual local deposition bias (OPES plus any
    EXTRA_BIAS) and the physical path bias used for postprocessing. Neither
    convention is silently inferred from a PLUMED input.
    """
    points = np.asarray(bead_cv, dtype=float)
    local = np.asarray(local_reference_bias_eV, dtype=float)
    count = len(batch.logweights)
    if (points.shape != (count, len(batch.cv_names)) or local.shape != (count,)
            or not np.isfinite(points).all() or not np.isfinite(local).all()
            or not np.isfinite(common_path_bias_eV)
            or not np.isfinite(kbt_eV) or kbt_eV <= 0
            or not np.isfinite(cv_tolerance) or cv_tolerance <= 0):
        raise ValueError("invalid bead CV, bias, kBT, or matching tolerance")
    error = np.max(np.abs(batch.centers[:, None, :] - points[None, :, :]), axis=2)
    matched = np.argmin(error, axis=1)
    match_error = float(np.max(error[np.arange(count), matched]))
    if len(set(matched.tolist())) != count or match_error > cv_tolerance:
        raise ValueError("KERNELS centers do not bijectively match bead CVs")
    logged = np.empty(count)
    logged[matched] = batch.logweights
    local_expected = local / kbt_eV
    common_expected = float(common_path_bias_eV / kbt_eV)
    return OpesDepositionComparison(
        match_error,
        float(np.max(np.abs(logged - local_expected))),
        float(np.max(np.abs(logged - common_expected))),
        float(np.ptp(logged)),
        float(np.ptp(local_expected)),
        common_expected,
        logged,
    )
