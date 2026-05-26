"""Water-orientation geometry and table summaries."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.io.lammps_dump import minimum_image_vectors, wrap_point_to_box


METRIC_NAMES = (
    "theta_AB_deg",
    "theta_A_deg",
    "theta_B_deg",
    "cos_theta_AB",
    "cos_theta_A",
    "cos_theta_B",
    "S_AB",
    "S_A",
    "S_B",
)

RADIAL_PROFILE_COLUMNS = (
    "rho_bin_left",
    "rho_bin_right",
    "rho_bin_center",
    "count",
    "theta_AB_deg_mean",
    "theta_AB_deg_std",
    "theta_A_deg_mean",
    "theta_A_deg_std",
    "theta_B_deg_mean",
    "theta_B_deg_std",
    "cos_theta_AB_mean",
    "cos_theta_AB_std",
    "cos_theta_A_mean",
    "cos_theta_A_std",
    "cos_theta_B_mean",
    "cos_theta_B_std",
    "S_AB_mean",
    "S_AB_std",
    "S_A_mean",
    "S_A_std",
    "S_B_mean",
    "S_B_std",
)

SR_MAP_COLUMNS = (
    "s_bin_left",
    "s_bin_right",
    "s_bin_center",
    "rho_bin_left",
    "rho_bin_right",
    "rho_bin_center",
    "count",
    "cos_theta_AB_mean",
    "theta_AB_deg_mean",
    "S_AB_mean",
    "cos_theta_A_mean",
    "theta_A_deg_mean",
    "S_A_mean",
    "cos_theta_B_mean",
    "theta_B_deg_mean",
    "S_B_mean",
)

CV_SUMMARY_COLUMNS = (
    "cv_bin_left",
    "cv_bin_right",
    "cv_bin_center",
    "count",
    "theta_AB_deg_mean",
    "theta_AB_deg_std",
    "cos_theta_AB_mean",
    "cos_theta_AB_std",
    "S_AB_mean",
    "S_AB_std",
    "theta_A_deg_mean",
    "theta_A_deg_std",
    "theta_B_deg_mean",
    "theta_B_deg_std",
)

ANGLE_DISTRIBUTION_COLUMNS = ("theta_bin_left", "theta_bin_right", "theta_bin_center", "count", "density")


@dataclass(frozen=True)
class WaterOrientationSummaryConfig:
    """Settings for water-orientation table summaries."""

    rho_bins: int = 40
    rho_max: Optional[float] = None
    s_bins: int = 40
    s_min: Optional[float] = None
    s_max: Optional[float] = None
    cv_bins: int = 40
    angle_bins: int = 72

    def validate(self) -> None:
        if self.rho_bins <= 0 or self.s_bins <= 0 or self.cv_bins <= 0:
            raise ValueError("rho_bins, s_bins, and cv_bins must be positive")
        if self.angle_bins < 2:
            raise ValueError("angle_bins must be at least 2")
        if self.rho_max is not None and self.rho_max <= 0.0:
            raise ValueError("rho_max must be positive")
        if self.s_min is not None and self.s_max is not None and self.s_max <= self.s_min:
            raise ValueError("s_max must be greater than s_min")


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader], [str(field) for field in reader.fieldnames]


def _write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "frame",
        "time",
        "n_samples",
        "rho_bin_left",
        "rho_bin_right",
        "s_bin_left",
        "s_bin_right",
        "cv_bin_left",
        "cv_bin_right",
        "theta_bin_left",
        "theta_bin_right",
        "count",
        "density",
        "metric",
        "value",
    ]
    ordered = [key for key in preferred if key in keys]
    ordered.extend(key for key in keys if key not in ordered)
    return ordered


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: object, default: int = -1) -> int:
    number = _as_float(value)
    return int(round(number)) if math.isfinite(number) else default


def _format_float(value: float) -> object:
    return value if math.isfinite(value) else ""


def _finite_values(rows: Sequence[Mapping[str, object]], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = _as_float(row.get(column))
        if math.isfinite(value):
            values.append(value)
    return values


def _as_bounds(bounds_or_lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(bounds_or_lengths, dtype=float)
    if values.shape == (3, 2):
        return values
    if values.shape == (3,):
        return np.column_stack((np.zeros(3, dtype=float), values))
    raise ValueError("bounds_or_lengths must have shape (3, 2) or (3,)")


def _box_lengths(bounds_or_lengths: np.ndarray) -> np.ndarray:
    bounds = _as_bounds(bounds_or_lengths)
    return bounds[:, 1] - bounds[:, 0]


def _mean_std(values: Iterable[object]) -> Tuple[float, float]:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), math.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _mean_value(values: Iterable[object]) -> float:
    mean, _ = _mean_std(values)
    return mean


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, or NaN components for a degenerate vector."""

    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        return np.full(3, math.nan, dtype=float)
    return values / norm


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Dot two unit vectors and clip numerical noise to [-1, 1]."""

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if np.isnan(first).any() or np.isnan(second).any():
        return math.nan
    return float(np.clip(float(np.dot(first, second)), -1.0, 1.0))


def cosine_to_angle_deg(value: float) -> float:
    """Convert a cosine value to an angle in degrees."""

    if not math.isfinite(value):
        return math.nan
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def angle_to_axis_deg(unit_vec: np.ndarray, axis_unit: np.ndarray) -> float:
    """Return the angle between two unit vectors in degrees."""

    return cosine_to_angle_deg(safe_cosine(unit_vec, axis_unit))


def nematic_order(cos_theta: float) -> float:
    """Return second-rank orientational order for a cosine value."""

    if not math.isfinite(cos_theta):
        return math.nan
    return float(0.5 * (3.0 * cos_theta * cos_theta - 1.0))


def compute_s_rho(
    point: np.ndarray,
    origin: np.ndarray,
    axis_unit: np.ndarray,
    bounds_or_lengths: np.ndarray,
) -> Tuple[float, float]:
    """Project a point onto an axis and its perpendicular radial distance."""

    lengths = _box_lengths(bounds_or_lengths)
    delta = minimum_image_vectors(np.asarray(point, dtype=float) - np.asarray(origin, dtype=float), lengths)
    s_value = float(np.dot(delta, np.asarray(axis_unit, dtype=float)))
    radial_vec = delta - s_value * np.asarray(axis_unit, dtype=float)
    return s_value, float(np.linalg.norm(radial_vec))


def compute_water_orientation_sample(
    oxygen_position: np.ndarray,
    hydrogen_positions: np.ndarray,
    center_a: np.ndarray,
    center_b: np.ndarray,
    bounds_or_lengths: np.ndarray,
    frame: Optional[int] = None,
    time: Optional[float] = None,
    oxygen_id: Optional[int] = None,
    hydrogen_ids: Optional[Sequence[int]] = None,
    extra_metrics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Compute one water-orientation sample in a two-center reference frame."""

    bounds = _as_bounds(bounds_or_lengths)
    lengths = _box_lengths(bounds)
    oxygen = np.asarray(oxygen_position, dtype=float)
    hydrogens = np.asarray(hydrogen_positions, dtype=float)
    if hydrogens.shape != (2, 3):
        raise ValueError("hydrogen_positions must have shape (2, 3)")
    center_a = np.asarray(center_a, dtype=float)
    center_b = np.asarray(center_b, dtype=float)

    h_vectors = minimum_image_vectors(hydrogens - oxygen, lengths)
    mu = unit_vector(np.mean(h_vectors, axis=0))
    delta_ab = minimum_image_vectors(center_b - center_a, lengths)
    axis_ab = unit_vector(delta_ab)
    center_mid = wrap_point_to_box(center_a + 0.5 * delta_ab, bounds)

    delta_a = minimum_image_vectors(oxygen - center_a, lengths)
    delta_b = minimum_image_vectors(oxygen - center_b, lengths)
    axis_a = unit_vector(delta_a)
    axis_b = unit_vector(delta_b)
    s_a = float(np.dot(delta_a, axis_ab))
    rho_a = float(np.linalg.norm(delta_a - s_a * axis_ab))
    s_value, rho_value = compute_s_rho(oxygen, center_mid, axis_ab, bounds)

    cos_ab = safe_cosine(mu, axis_ab)
    cos_a = safe_cosine(mu, axis_a)
    cos_b = safe_cosine(mu, axis_b)
    theta_ab = cosine_to_angle_deg(cos_ab)
    theta_a = cosine_to_angle_deg(cos_a)
    theta_b = cosine_to_angle_deg(cos_b)

    h_ids = list(hydrogen_ids or [])
    row: Dict[str, object] = {
        "frame": frame if frame is not None else "",
        "time": time if time is not None else "",
        "oxygen_id": oxygen_id if oxygen_id is not None else "",
        "hydrogen1_id": h_ids[0] if len(h_ids) > 0 else "",
        "hydrogen2_id": h_ids[1] if len(h_ids) > 1 else "",
        "C_A_x": float(center_a[0]),
        "C_A_y": float(center_a[1]),
        "C_A_z": float(center_a[2]),
        "C_B_x": float(center_b[0]),
        "C_B_y": float(center_b[1]),
        "C_B_z": float(center_b[2]),
        "C_mid_x": float(center_mid[0]),
        "C_mid_y": float(center_mid[1]),
        "C_mid_z": float(center_mid[2]),
        "e_AB_x": float(axis_ab[0]),
        "e_AB_y": float(axis_ab[1]),
        "e_AB_z": float(axis_ab[2]),
        "s_A": s_a,
        "rho_A": rho_a,
        "s": s_value,
        "rho": rho_value,
        "mu_x": float(mu[0]),
        "mu_y": float(mu[1]),
        "mu_z": float(mu[2]),
        "cos_theta_AB": cos_ab,
        "theta_AB_deg": theta_ab,
        "S_AB": nematic_order(cos_ab),
        "cos_theta_A": cos_a,
        "theta_A_deg": theta_a,
        "S_A": nematic_order(cos_a),
        "cos_theta_B": cos_b,
        "theta_B_deg": theta_b,
        "S_B": nematic_order(cos_b),
        "delta_theta_ABref_deg": theta_a - theta_b if math.isfinite(theta_a) and math.isfinite(theta_b) else math.nan,
    }
    if extra_metrics:
        row.update(extra_metrics)
    return row


def load_orientation_rows(path: Path) -> List[Dict[str, object]]:
    """Load orientation samples from a CSV table."""

    rows, fieldnames = _read_csv_rows(path)
    numeric_columns = set(METRIC_NAMES) | {
        "frame",
        "time",
        "oxygen_id",
        "hydrogen1_id",
        "hydrogen2_id",
        "C_A_x",
        "C_A_y",
        "C_A_z",
        "C_B_x",
        "C_B_y",
        "C_B_z",
        "C_mid_x",
        "C_mid_y",
        "C_mid_z",
        "e_AB_x",
        "e_AB_y",
        "e_AB_z",
        "s_A",
        "rho_A",
        "s",
        "rho",
        "mu_x",
        "mu_y",
        "mu_z",
        "delta_theta_ABref_deg",
        "d3d_all",
        "bridge_cyl_env.sum",
        "bridge_cyl_env.mean",
    }
    out: List[Dict[str, object]] = []
    for raw in rows:
        row: Dict[str, object] = dict(raw)
        for column in fieldnames:
            if column not in numeric_columns:
                continue
            if column in {"frame", "oxygen_id", "hydrogen1_id", "hydrogen2_id"}:
                row[column] = _as_int(raw.get(column))
            else:
                row[column] = _as_float(raw.get(column))
        out.append(row)
    return out


def _bin_edges_from_range(start: float, stop: float, bins: int) -> np.ndarray:
    if not math.isfinite(start) or not math.isfinite(stop):
        start, stop = 0.0, 1.0
    if math.isclose(start, stop):
        start -= 0.5
        stop += 0.5
    return np.linspace(float(start), float(stop), int(bins) + 1)


def _infer_s_range(rows: Sequence[Mapping[str, object]], s_column: str) -> Tuple[float, float]:
    values = _finite_values(rows, s_column)
    if not values:
        return -1.0, 1.0
    return min(values), max(values)


def _infer_rho_max(rows: Sequence[Mapping[str, object]], rho_column: str) -> float:
    values = _finite_values(rows, rho_column)
    if not values:
        return 1.0
    return max(max(values), 1e-6)


def _bin_mask(values: np.ndarray, edges: np.ndarray, index: int) -> np.ndarray:
    if index == len(edges) - 2:
        return ((values >= edges[index]) & (values < edges[index + 1])) | np.isclose(values, edges[-1])
    return (values >= edges[index]) & (values < edges[index + 1])


def _metric_mean_std_row(rows: Sequence[Mapping[str, object]], metrics: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for metric in metrics:
        mean, std = _mean_std(row.get(metric) for row in rows)
        out[f"{metric}_mean"] = _format_float(mean)
        out[f"{metric}_std"] = _format_float(std)
    return out


def build_radial_profile(
    rows: Sequence[Mapping[str, object]],
    rho_bins: int,
    rho_max: float,
    rho_column: str = "rho",
    metrics: Sequence[str] = METRIC_NAMES,
) -> List[Dict[str, object]]:
    """Summarize orientation metrics by radial distance from the bridge axis."""

    edges = np.linspace(0.0, float(rho_max), int(rho_bins) + 1)
    rho_values = np.asarray([_as_float(row.get(rho_column)) for row in rows], dtype=float)
    out: List[Dict[str, object]] = []
    for index in range(len(edges) - 1):
        mask = _bin_mask(rho_values, edges, index)
        chunk = [row for row, keep in zip(rows, mask) if bool(keep)]
        item: Dict[str, object] = {
            "rho_bin_left": float(edges[index]),
            "rho_bin_right": float(edges[index + 1]),
            "rho_bin_center": float(0.5 * (edges[index] + edges[index + 1])),
            "count": len(chunk),
        }
        item.update(_metric_mean_std_row(chunk, metrics))
        out.append(item)
    return out


def build_sr_map(
    rows: Sequence[Mapping[str, object]],
    s_bins: int,
    rho_bins: int,
    s_min: float,
    s_max: float,
    rho_max: float,
    s_column: str = "s",
    rho_column: str = "rho",
) -> List[Dict[str, object]]:
    """Summarize orientation metrics on a bridge-axis/radial grid."""

    s_edges = np.linspace(float(s_min), float(s_max), int(s_bins) + 1)
    rho_edges = np.linspace(0.0, float(rho_max), int(rho_bins) + 1)
    s_values = np.asarray([_as_float(row.get(s_column)) for row in rows], dtype=float)
    rho_values = np.asarray([_as_float(row.get(rho_column)) for row in rows], dtype=float)
    out: List[Dict[str, object]] = []
    for s_index in range(len(s_edges) - 1):
        s_mask = _bin_mask(s_values, s_edges, s_index)
        for rho_index in range(len(rho_edges) - 1):
            rho_mask = _bin_mask(rho_values, rho_edges, rho_index)
            chunk = [row for row, keep in zip(rows, s_mask & rho_mask) if bool(keep)]
            item: Dict[str, object] = {
                "s_bin_left": float(s_edges[s_index]),
                "s_bin_right": float(s_edges[s_index + 1]),
                "s_bin_center": float(0.5 * (s_edges[s_index] + s_edges[s_index + 1])),
                "rho_bin_left": float(rho_edges[rho_index]),
                "rho_bin_right": float(rho_edges[rho_index + 1]),
                "rho_bin_center": float(0.5 * (rho_edges[rho_index] + rho_edges[rho_index + 1])),
                "count": len(chunk),
                "cos_theta_AB_mean": _format_float(_mean_value(row.get("cos_theta_AB") for row in chunk)),
                "theta_AB_deg_mean": _format_float(_mean_value(row.get("theta_AB_deg") for row in chunk)),
                "S_AB_mean": _format_float(_mean_value(row.get("S_AB") for row in chunk)),
                "cos_theta_A_mean": _format_float(_mean_value(row.get("cos_theta_A") for row in chunk)),
                "theta_A_deg_mean": _format_float(_mean_value(row.get("theta_A_deg") for row in chunk)),
                "S_A_mean": _format_float(_mean_value(row.get("S_A") for row in chunk)),
                "cos_theta_B_mean": _format_float(_mean_value(row.get("cos_theta_B") for row in chunk)),
                "theta_B_deg_mean": _format_float(_mean_value(row.get("theta_B_deg") for row in chunk)),
                "S_B_mean": _format_float(_mean_value(row.get("S_B") for row in chunk)),
            }
            out.append(item)
    return out


def build_cv_summary(
    rows: Sequence[Mapping[str, object]],
    cv_bins: int,
    cv_column: str = "d3d_all",
) -> List[Dict[str, object]]:
    """Summarize orientation metrics by a scalar CV column."""

    cv_values_all = np.asarray([_as_float(row.get(cv_column)) for row in rows], dtype=float)
    finite_mask = np.isfinite(cv_values_all)
    usable = [row for row, keep in zip(rows, finite_mask) if bool(keep)]
    cv_values = cv_values_all[finite_mask]
    if cv_values.size == 0:
        return []
    edges = _bin_edges_from_range(float(np.nanmin(cv_values)), float(np.nanmax(cv_values)), int(cv_bins))
    out: List[Dict[str, object]] = []
    for index in range(len(edges) - 1):
        mask = _bin_mask(cv_values, edges, index)
        chunk = [row for row, keep in zip(usable, mask) if bool(keep)]
        item: Dict[str, object] = {
            "cv_bin_left": float(edges[index]),
            "cv_bin_right": float(edges[index + 1]),
            "cv_bin_center": float(0.5 * (edges[index] + edges[index + 1])),
            "count": len(chunk),
        }
        item.update(
            _metric_mean_std_row(
                chunk,
                ("theta_AB_deg", "cos_theta_AB", "S_AB", "theta_A_deg", "theta_B_deg"),
            )
        )
        out.append(item)
    return out


def build_angle_distribution(
    rows: Sequence[Mapping[str, object]],
    angle_bins: int = 72,
    angle_column: str = "theta_AB_deg",
) -> List[Dict[str, object]]:
    """Build a probability-density histogram for water orientation angles."""

    values = np.asarray(_finite_values(rows, angle_column), dtype=float)
    if values.size == 0:
        return []
    counts, edges = np.histogram(values, bins=max(2, int(angle_bins)), range=(0.0, 180.0))
    widths = np.diff(edges)
    total = float(np.sum(counts))
    density = counts / (total * widths) if total > 0.0 else np.zeros_like(counts, dtype=float)
    return [
        {
            "theta_bin_left": float(edges[index]),
            "theta_bin_right": float(edges[index + 1]),
            "theta_bin_center": float(0.5 * (edges[index] + edges[index + 1])),
            "count": int(counts[index]),
            "density": float(density[index]),
        }
        for index in range(len(counts))
    ]


def build_frame_summary(
    rows: Sequence[Mapping[str, object]],
    frame_column: str = "frame",
    time_column: str = "time",
) -> List[Dict[str, object]]:
    """Summarize orientation metrics per frame."""

    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        frame = _as_int(row.get(frame_column))
        grouped.setdefault(frame, []).append(row)
    out: List[Dict[str, object]] = []
    for frame in sorted(grouped):
        chunk = grouped[frame]
        time_values = _finite_values(chunk, time_column)
        item: Dict[str, object] = {
            "frame": frame,
            "time": _format_float(time_values[0] if time_values else math.nan),
            "n_samples": len(chunk),
        }
        item.update(_metric_mean_std_row(chunk, METRIC_NAMES))
        out.append(item)
    return out


def analyze_water_orientation(
    input_csv: Path,
    output_dir: Path,
    config: WaterOrientationSummaryConfig = WaterOrientationSummaryConfig(),
    s_column: str = "s",
    rho_column: str = "rho",
    cv_column: str = "d3d_all",
    angle_column: str = "theta_AB_deg",
    frame_column: str = "frame",
    time_column: str = "time",
) -> Dict[str, Path]:
    """Run water-orientation table summaries and write CSV outputs."""

    config.validate()
    rows = load_orientation_rows(input_csv)
    inferred_s_min, inferred_s_max = _infer_s_range(rows, s_column)
    s_min = inferred_s_min if config.s_min is None else float(config.s_min)
    s_max = inferred_s_max if config.s_max is None else float(config.s_max)
    if math.isclose(s_min, s_max):
        s_min -= 0.5
        s_max += 0.5
    rho_max = _infer_rho_max(rows, rho_column) if config.rho_max is None else float(config.rho_max)

    output_dir = Path(output_dir)
    outputs = {
        "frame_summary": output_dir / "water_orientation_frame_summary.csv",
        "radial_profile": output_dir / "water_orientation_radial_profile.csv",
        "sr_map": output_dir / "water_orientation_sr_map.csv",
        "cv_summary": output_dir / "water_orientation_cv_summary.csv",
        "angle_distribution": output_dir / "water_orientation_angle_distribution.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["frame_summary"], build_frame_summary(rows, frame_column=frame_column, time_column=time_column))
    _write_csv_rows(
        outputs["radial_profile"],
        build_radial_profile(rows, rho_bins=config.rho_bins, rho_max=rho_max, rho_column=rho_column),
        fieldnames=RADIAL_PROFILE_COLUMNS,
    )
    _write_csv_rows(
        outputs["sr_map"],
        build_sr_map(
            rows,
            s_bins=config.s_bins,
            rho_bins=config.rho_bins,
            s_min=s_min,
            s_max=s_max,
            rho_max=rho_max,
            s_column=s_column,
            rho_column=rho_column,
        ),
        fieldnames=SR_MAP_COLUMNS,
    )
    _write_csv_rows(
        outputs["cv_summary"],
        build_cv_summary(rows, cv_bins=config.cv_bins, cv_column=cv_column),
        fieldnames=CV_SUMMARY_COLUMNS,
    )
    _write_csv_rows(
        outputs["angle_distribution"],
        build_angle_distribution(rows, angle_bins=config.angle_bins, angle_column=angle_column),
        fieldnames=ANGLE_DISTRIBUTION_COLUMNS,
    )
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_csv", "value": str(input_csv)},
            {"metric": "n_orientation_samples", "value": len(rows)},
            {"metric": "n_frames", "value": len({_as_int(row.get(frame_column)) for row in rows})},
            {"metric": "s_column", "value": s_column},
            {"metric": "rho_column", "value": rho_column},
            {"metric": "cv_column", "value": cv_column},
            {"metric": "angle_column", "value": angle_column},
            {"metric": "s_min", "value": s_min},
            {"metric": "s_max", "value": s_max},
            {"metric": "rho_max", "value": rho_max},
            {"metric": "s_bins", "value": config.s_bins},
            {"metric": "rho_bins", "value": config.rho_bins},
            {"metric": "cv_bins", "value": config.cv_bins},
            {"metric": "angle_bins", "value": config.angle_bins},
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize water-orientation sample tables")
    parser.add_argument("--input", type=Path, required=True, help="Input water-orientation sample CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--s-column", default="s")
    parser.add_argument("--rho-column", default="rho")
    parser.add_argument("--cv-column", default="d3d_all")
    parser.add_argument("--angle-column", default="theta_AB_deg")
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--rho-bins", type=int, default=40)
    parser.add_argument("--rho-max", type=float)
    parser.add_argument("--s-bins", type=int, default=40)
    parser.add_argument("--s-min", type=float)
    parser.add_argument("--s-max", type=float)
    parser.add_argument("--cv-bins", type=int, default=40)
    parser.add_argument("--angle-bins", type=int, default=72)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_water_orientation(
            input_csv=args.input,
            output_dir=args.output_dir,
            config=WaterOrientationSummaryConfig(
                rho_bins=args.rho_bins,
                rho_max=args.rho_max,
                s_bins=args.s_bins,
                s_min=args.s_min,
                s_max=args.s_max,
                cv_bins=args.cv_bins,
                angle_bins=args.angle_bins,
            ),
            s_column=args.s_column,
            rho_column=args.rho_column,
            cv_column=args.cv_column,
            angle_column=args.angle_column,
            frame_column=args.frame_column,
            time_column=args.time_column,
        )
    except Exception as exc:
        print(f"Water-orientation analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
