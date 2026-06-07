"""Reusable free-energy and barrier analysis utilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_BARRIER_WINDOWS: Tuple[Tuple[str, float, float], ...] = (
    ("all", -math.inf, math.inf),
    ("nucleation_s_lt_210", -math.inf, 210.0),
    ("dissolution_s_gt_50", 50.0, math.inf),
)


@dataclass(frozen=True)
class FesCurveSpec:
    """Input metadata for one 1D FES curve."""

    path: Path
    label: str
    group: str = "default"
    dataset_key: str = ""


@dataclass
class FesCurve:
    """Loaded 1D FES curve data."""

    spec: FesCurveSpec
    cv: np.ndarray
    free_energy: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class Fes2DGrid:
    """Regular 2D FES grid loaded from a PLUMED-style table."""

    source_path: Path
    fields: Tuple[str, ...]
    metadata: Dict[str, str]
    x_name: str
    y_name: str
    z_name: str
    uncertainty_name: Optional[str]
    x_values: np.ndarray
    y_values: np.ndarray
    free_energy: np.ndarray
    uncertainty: Optional[np.ndarray]


def _as_float(value: str) -> float:
    text = str(value).strip()
    if text.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text.lower() in {"-inf", "-infinity"}:
        return -math.inf
    return float(text)


def parse_window(raw: str) -> Tuple[str, float, float]:
    """Parse `name:low:high` window syntax."""

    parts = str(raw).split(":")
    if len(parts) != 3:
        raise ValueError("Barrier windows must use name:low:high syntax")
    name = parts[0].strip()
    if not name:
        raise ValueError("Barrier window name cannot be empty")
    low = _as_float(parts[1])
    high = _as_float(parts[2])
    if high <= low:
        raise ValueError(f"Invalid barrier window bounds for {name}: high <= low")
    return name, low, high


def parse_curve_spec(values: Sequence[str]) -> FesCurveSpec:
    """Parse a CLI curve tuple: PATH LABEL GROUP."""

    if len(values) != 3:
        raise ValueError("--curve requires PATH LABEL GROUP")
    path, label, group = values
    dataset_key = Path(path).stem
    return FesCurveSpec(path=Path(path), label=label, group=group, dataset_key=dataset_key)


def load_curve_manifest(path: Path) -> List[FesCurveSpec]:
    """Load curve specs from a CSV manifest.

    Required columns are `path` and `label`.  Optional columns are `group` and
    `dataset_key`.
    """

    specs: List[FesCurveSpec] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = [column for column in ["path", "label"] if column not in reader.fieldnames]
        if missing:
            raise ValueError("Manifest missing required columns: " + ", ".join(missing))
        for index, row in enumerate(reader):
            raw_path = str(row.get("path", "")).strip()
            label = str(row.get("label", "")).strip()
            if not raw_path or not label:
                continue
            group = str(row.get("group") or "default")
            dataset_key = str(row.get("dataset_key") or Path(raw_path).stem or f"curve_{index}")
            specs.append(FesCurveSpec(path=Path(raw_path), label=label, group=group, dataset_key=dataset_key))
    if not specs:
        raise ValueError(f"No curves found in manifest: {path}")
    return specs


def load_fes_curve(
    spec: FesCurveSpec,
    cv_column: int = 0,
    free_energy_column: int = 1,
    uncertainty_column: Optional[int] = 2,
) -> FesCurve:
    """Load a whitespace FES file with comments ignored."""

    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    data = np.loadtxt(spec.path, comments="#", ndmin=2)
    if data.ndim != 2 or data.shape[1] <= max(cv_column, free_energy_column):
        raise ValueError(f"FES file does not contain requested columns: {spec.path}")
    cv = np.asarray(data[:, cv_column], dtype=float)
    free_energy = np.asarray(data[:, free_energy_column], dtype=float)
    if uncertainty_column is not None and data.shape[1] > uncertainty_column:
        uncertainty = np.asarray(data[:, uncertainty_column], dtype=float)
    else:
        uncertainty = np.full_like(cv, np.nan, dtype=float)
    return FesCurve(spec=spec, cv=cv, free_energy=free_energy, uncertainty=uncertainty)


def parse_fes_header(path: Path) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Parse PLUMED FES header fields and `#! SET` metadata."""

    fields: Tuple[str, ...] = ()
    metadata: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#!"):
                break
            content = line[2:].strip()
            if not content:
                continue
            parts = content.split(None, 2)
            if parts[0] == "FIELDS":
                fields = tuple(content.split()[1:])
            elif parts[0] == "SET" and len(parts) >= 3:
                metadata[parts[1]] = parts[2]
    return fields, metadata


def infer_fes2d_column_names(fields: Sequence[str], n_cols: int) -> Tuple[str, str, str, Optional[str]]:
    """Infer x/y/free-energy/uncertainty column names from a FES header."""

    if len(fields) >= 3:
        return fields[0], fields[1], fields[2], fields[3] if len(fields) >= 4 else None
    if n_cols < 3:
        raise ValueError("Expected at least three numeric columns: x, y, free_energy")
    return "x", "y", "free_energy", "uncertainty" if n_cols >= 4 else None


def load_fes2d_grid(path: Path) -> Fes2DGrid:
    """Load a complete regular 2D FES grid from a whitespace table."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    fields, metadata = parse_fes_header(path)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three numeric columns in {path}")

    x_name, y_name, z_name, uncertainty_name = infer_fes2d_column_names(fields, data.shape[1])
    x_raw = np.asarray(data[:, 0], dtype=float)
    y_raw = np.asarray(data[:, 1], dtype=float)
    z_raw = np.asarray(data[:, 2], dtype=float)
    uncertainty_raw = np.asarray(data[:, 3], dtype=float) if data.shape[1] >= 4 else None

    x_values = np.unique(x_raw)
    y_values = np.unique(y_raw)
    nx = int(x_values.size)
    ny = int(y_values.size)
    if nx * ny != data.shape[0]:
        raise ValueError(
            f"Input is not a complete regular grid: rows={data.shape[0]}, "
            f"unique_x={nx}, unique_y={ny}"
        )

    x_index = {float(value): index for index, value in enumerate(x_values)}
    y_index = {float(value): index for index, value in enumerate(y_values)}
    free_energy = np.full((ny, nx), np.nan, dtype=float)
    uncertainty = np.full((ny, nx), np.nan, dtype=float) if uncertainty_raw is not None else None

    for row_index in range(data.shape[0]):
        ix = x_index[float(x_raw[row_index])]
        iy = y_index[float(y_raw[row_index])]
        free_energy[iy, ix] = z_raw[row_index]
        if uncertainty is not None and uncertainty_raw is not None:
            uncertainty[iy, ix] = uncertainty_raw[row_index]

    return Fes2DGrid(
        source_path=path,
        fields=fields,
        metadata=metadata,
        x_name=x_name,
        y_name=y_name,
        z_name=z_name,
        uncertainty_name=uncertainty_name,
        x_values=x_values,
        y_values=y_values,
        free_energy=free_energy,
        uncertainty=uncertainty,
    )


def clean_series(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Replace non-finite values by linear interpolation."""

    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr), finite
    if not np.all(finite):
        indices = np.arange(arr.size)
        arr[~finite] = np.interp(indices[~finite], indices[finite], arr[finite])
    return arr, finite


def window_mask(cv: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return a mask selecting `low <= cv < high`, with infinite bounds allowed."""

    mask = np.ones_like(cv, dtype=bool)
    if not math.isinf(low):
        mask &= cv >= float(low)
    if not math.isinf(high):
        mask &= cv < float(high)
    return mask


def min_in_window(cv: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    """Return finite minimum in a CV window, falling back to global finite min."""

    mask = window_mask(cv, low, high) & np.isfinite(values)
    if np.any(mask):
        return float(np.min(values[mask]))
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else 0.0


def shift_to_reference_window(
    cv: np.ndarray,
    free_energy: np.ndarray,
    reference_low: float = -math.inf,
    reference_high: float = math.inf,
) -> Tuple[np.ndarray, float]:
    """Shift a FES curve by the minimum in a reference CV window."""

    clean, _ = clean_series(free_energy)
    shift = min_in_window(np.asarray(cv, dtype=float), clean, reference_low, reference_high)
    return clean - shift, shift


def effective_window_length(n_points: int, requested_window: int) -> Optional[int]:
    """Return an odd smoothing window length or None for too-short arrays."""

    if requested_window <= 1 or n_points < 3:
        return None
    window = min(int(requested_window), int(n_points))
    if window % 2 == 0:
        window -= 1
    return window if window >= 3 else None


def moving_average_smooth(values: Sequence[float], window_length: int = 1, passes: int = 1) -> np.ndarray:
    """Smooth with an edge-padded moving average.

    This is intentionally dependency-light.  It is not a direct Savitzky-Golay
    replacement, but gives a stable table-oriented default before plotting
    workflows are migrated.
    """

    clean, _ = clean_series(values)
    window = effective_window_length(clean.size, window_length)
    if window is None:
        return clean
    kernel = np.ones(window, dtype=float) / float(window)
    out = clean.copy()
    pad = window // 2
    for _ in range(max(1, int(passes))):
        padded = np.pad(out, pad_width=pad, mode="edge")
        out = np.convolve(padded, kernel, mode="valid")
    return out


def coordinate_mask(values: np.ndarray, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    """Return an inclusive coordinate mask for optional low/high bounds."""

    arr = np.asarray(values, dtype=float)
    mask = np.ones(arr.shape, dtype=bool)
    if bounds is None:
        return mask
    low, high = float(bounds[0]), float(bounds[1])
    if high < low:
        raise ValueError(f"Invalid coordinate range: high < low ({low}, {high})")
    mask &= arr >= low
    mask &= arr <= high
    return mask


def finite_grid_min(values: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return the finite minimum from a 2D grid, optionally restricted by mask."""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        raise ValueError("No finite FES values found for zeroing")
    return float(np.min(arr[finite]))


def gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel in grid-bin units."""

    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(float(truncate) * sigma + 0.5))
    points = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (points / sigma) ** 2)
    return kernel / float(np.sum(kernel))


def _convolve_along_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = int(kernel.size // 2)
    if radius == 0:
        return np.asarray(values, dtype=float).copy()
    pad_width = [(0, 0)] * values.ndim
    pad_width[int(axis)] = (radius, radius)
    padded = np.pad(values, pad_width=pad_width, mode="constant", constant_values=0.0)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), int(axis), padded)


def normalized_gaussian_smooth_2d(
    values: np.ndarray,
    sigma: float,
    valid_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth a 2D FES grid while normalizing by finite-value support."""

    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if float(sigma) <= 0:
        return arr.copy(), valid.astype(float)

    kernel = gaussian_kernel1d(float(sigma))
    weights = valid.astype(float)
    filled = np.where(valid, arr, 0.0)
    numerator = _convolve_along_axis(_convolve_along_axis(filled, kernel, axis=1), kernel, axis=0)
    denominator = _convolve_along_axis(_convolve_along_axis(weights, kernel, axis=1), kernel, axis=0)

    out = np.full_like(arr, np.nan, dtype=float)
    keep = denominator >= float(valid_threshold)
    out[keep] = numerator[keep] / denominator[keep]
    return out, denominator


def plot_values_for_fes2d(values: np.ndarray, max_fes: float, missing_plot_value: str = "max") -> np.ndarray:
    """Clip finite FES values to plotting range and handle missing points."""

    if missing_plot_value not in {"max", "nan"}:
        raise ValueError("missing_plot_value must be 'max' or 'nan'")
    out = np.where(np.isfinite(values), np.clip(values, 0.0, float(max_fes)), np.nan)
    if missing_plot_value == "max":
        out = np.where(np.isfinite(out), out, float(max_fes))
    return out


def percentile_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
    """Return a compact percentile summary for finite values."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    quantiles = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    if finite.size == 0:
        return {str(q): None for q in quantiles}
    return {str(q): float(np.percentile(finite, q)) for q in quantiles}


def write_fes2d_grid_csv(
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    raw_zeroed: np.ndarray,
    raw_plot: np.ndarray,
    smooth_zeroed: np.ndarray,
    smooth_plot: np.ndarray,
    smooth_support: np.ndarray,
    x_name: str = "x",
    y_name: str = "y",
) -> None:
    """Write a long-form 2D FES grid table for plotting or downstream analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                x_name,
                y_name,
                "free_energy_raw_zeroed_kj_mol",
                "free_energy_raw_plot_kj_mol",
                "free_energy_smooth_zeroed_kj_mol",
                "free_energy_smooth_plot_kj_mol",
                "smooth_support_weight",
                "raw_finite",
                "smooth_finite",
            ]
        )
        for iy, y_value in enumerate(y_values):
            for ix, x_value in enumerate(x_values):
                raw_value = raw_zeroed[iy, ix]
                raw_plot_value = raw_plot[iy, ix]
                smooth_value = smooth_zeroed[iy, ix]
                smooth_plot_value = smooth_plot[iy, ix]
                support_value = smooth_support[iy, ix]
                writer.writerow(
                    [
                        f"{x_value:.10g}",
                        f"{y_value:.10g}",
                        f"{raw_value:.10g}" if np.isfinite(raw_value) else "",
                        f"{raw_plot_value:.10g}" if np.isfinite(raw_plot_value) else "",
                        f"{smooth_value:.10g}" if np.isfinite(smooth_value) else "",
                        f"{smooth_plot_value:.10g}" if np.isfinite(smooth_plot_value) else "",
                        f"{support_value:.10g}" if np.isfinite(support_value) else "",
                        int(np.isfinite(raw_value)),
                        int(np.isfinite(smooth_value)),
                    ]
                )


def _require_matplotlib():
    try:  # pragma: no cover - plotting is smoke-tested only when matplotlib exists.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --write-plots") from exc
    return plt


def plot_fes2d_panel(
    output_path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    values: np.ndarray,
    max_fes: float = 200.0,
    contour_levels: int = 16,
    dpi: int = 300,
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    cmap: str = "viridis",
) -> None:
    """Write one contour plot from a processed 2D FES grid."""

    plt = _require_matplotlib()
    levels = np.linspace(0.0, float(max_fes), int(contour_levels) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=int(dpi), constrained_layout=True)
    filled = ax.contourf(x_values, y_values, values, levels=levels, cmap=cmap, extend="max")
    ax.contour(x_values, y_values, values, levels=levels, colors="k", linewidths=0.35)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(filled, ax=ax)
    cbar.set_label("Delta F (kJ/mol)")
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_fes2d_comparison(
    output_path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    raw_plot: np.ndarray,
    smooth_plot: np.ndarray,
    max_fes: float = 200.0,
    contour_levels: int = 16,
    dpi: int = 300,
    x_label: str = "x",
    y_label: str = "y",
    cmap: str = "viridis",
) -> None:
    """Write a raw-vs-smoothed 2D FES comparison figure."""

    plt = _require_matplotlib()
    levels = np.linspace(0.0, float(max_fes), int(contour_levels) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=int(dpi), constrained_layout=True)
    mappable = None
    for ax, values, title in zip(axes, [raw_plot, smooth_plot], ["Raw clipped", "Smoothed"]):
        mappable = ax.contourf(x_values, y_values, values, levels=levels, cmap=cmap, extend="max")
        ax.contour(x_values, y_values, values, levels=levels, colors="k", linewidths=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.85, pad=0.08)
        cbar.set_label("Delta F (kJ/mol)")
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _metadata_range(bounds: Optional[Tuple[float, float]]) -> Optional[List[float]]:
    if bounds is None:
        return None
    return [float(bounds[0]), float(bounds[1])]


def process_fes2d_grid(
    fes_file: Path,
    output_dir: Path,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    max_fes: float = 200.0,
    smooth_sigma: float = 0.8,
    valid_threshold: float = 0.25,
    prefix: str = "fes2d",
    zero_scope: str = "window",
    missing_plot_value: str = "max",
    write_grid: bool = True,
    write_plots: bool = False,
    write_comparison: bool = False,
    contour_levels: int = 16,
    dpi: int = 300,
    title: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    cmap: str = "viridis",
) -> Dict[str, Path]:
    """Process a 2D FES grid and write path-explicit outputs."""

    if zero_scope not in {"window", "all"}:
        raise ValueError("zero_scope must be 'window' or 'all'")
    grid = load_fes2d_grid(Path(fes_file))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_mask = coordinate_mask(grid.x_values, x_range)
    y_mask = coordinate_mask(grid.y_values, y_range)
    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("The requested x/y range selects no FES grid points")
    selected_mask = np.outer(y_mask, x_mask)

    raw = np.asarray(grid.free_energy, dtype=float).copy()
    raw[~np.isfinite(raw)] = np.nan
    zero_mask = selected_mask if zero_scope == "window" else None
    raw_zero_value = finite_grid_min(raw, zero_mask)
    raw_zeroed = raw - raw_zero_value

    raw_for_smoothing = np.where(np.isfinite(raw_zeroed), np.clip(raw_zeroed, 0.0, float(max_fes)), np.nan)
    smooth, support = normalized_gaussian_smooth_2d(
        raw_for_smoothing,
        sigma=float(smooth_sigma),
        valid_threshold=float(valid_threshold),
    )
    try:
        smooth_zero_value = finite_grid_min(smooth, zero_mask)
    except ValueError:
        smooth_zero_value = finite_grid_min(smooth)
    smooth_zeroed = smooth - smooth_zero_value

    raw_plot = plot_values_for_fes2d(raw_zeroed, max_fes=max_fes, missing_plot_value=missing_plot_value)
    smooth_plot = plot_values_for_fes2d(smooth_zeroed, max_fes=max_fes, missing_plot_value=missing_plot_value)

    x_crop = grid.x_values[x_mask]
    y_crop = grid.y_values[y_mask]
    raw_zeroed_crop = raw_zeroed[np.ix_(y_mask, x_mask)]
    raw_plot_crop = raw_plot[np.ix_(y_mask, x_mask)]
    smooth_zeroed_crop = smooth_zeroed[np.ix_(y_mask, x_mask)]
    smooth_plot_crop = smooth_plot[np.ix_(y_mask, x_mask)]
    support_crop = support[np.ix_(y_mask, x_mask)]

    outputs: Dict[str, Path] = {}
    if write_grid:
        grid_csv = output_dir / f"{prefix}_plot_grid.csv"
        write_fes2d_grid_csv(
            grid_csv,
            x_crop,
            y_crop,
            raw_zeroed_crop,
            raw_plot_crop,
            smooth_zeroed_crop,
            smooth_plot_crop,
            support_crop,
            x_name=grid.x_name,
            y_name=grid.y_name,
        )
        outputs["plot_grid_csv"] = grid_csv

    figure_x_label = x_label or grid.x_name
    figure_y_label = y_label or grid.y_name
    if write_plots or write_comparison:
        smooth_png = output_dir / f"{prefix}_smooth.png"
        plot_fes2d_panel(
            smooth_png,
            x_crop,
            y_crop,
            smooth_plot_crop,
            max_fes=max_fes,
            contour_levels=contour_levels,
            dpi=dpi,
            title=title,
            x_label=figure_x_label,
            y_label=figure_y_label,
            cmap=cmap,
        )
        outputs["smooth_png"] = smooth_png
        if write_comparison:
            comparison_png = output_dir / f"{prefix}_raw_vs_smooth.png"
            plot_fes2d_comparison(
                comparison_png,
                x_crop,
                y_crop,
                raw_plot_crop,
                smooth_plot_crop,
                max_fes=max_fes,
                contour_levels=contour_levels,
                dpi=dpi,
                x_label=figure_x_label,
                y_label=figure_y_label,
                cmap=cmap,
            )
            outputs["raw_vs_smooth_png"] = comparison_png

    metadata = {
        "source_path": str(grid.source_path),
        "fields": list(grid.fields),
        "header_metadata": grid.metadata,
        "x_name": grid.x_name,
        "y_name": grid.y_name,
        "z_name": grid.z_name,
        "uncertainty_name": grid.uncertainty_name,
        "x_range": _metadata_range(x_range),
        "y_range": _metadata_range(y_range),
        "x_bins_total": int(grid.x_values.size),
        "y_bins_total": int(grid.y_values.size),
        "x_bins_selected": int(x_crop.size),
        "y_bins_selected": int(y_crop.size),
        "finite_points_total": int(np.count_nonzero(np.isfinite(raw))),
        "finite_points_selected": int(np.count_nonzero(np.isfinite(raw_zeroed_crop))),
        "zero_scope": zero_scope,
        "raw_zero_value_kj_mol": raw_zero_value,
        "smooth_zero_value_kj_mol": smooth_zero_value,
        "max_fes_kj_mol": float(max_fes),
        "smooth_sigma_bins": float(smooth_sigma),
        "smooth_valid_threshold": float(valid_threshold),
        "missing_plot_value": missing_plot_value,
        "raw_zeroed_selected_percentiles_kj_mol": percentile_summary(raw_zeroed_crop),
        "smooth_zeroed_selected_percentiles_kj_mol": percentile_summary(smooth_zeroed_crop),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata["outputs"]["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    outputs["metadata_json"] = metadata_path
    return outputs


def zero_curve(
    cv: np.ndarray,
    free_energy: np.ndarray,
    zero_low: float = -math.inf,
    zero_high: float = math.inf,
) -> Tuple[np.ndarray, float]:
    """Shift a curve so the minimum in `zero_low:zero_high` is zero."""

    clean, _ = clean_series(free_energy)
    zero = min_in_window(np.asarray(cv, dtype=float), clean, zero_low, zero_high)
    return clean - zero, zero


def process_curve(
    curve: FesCurve,
    reference_low: float = -math.inf,
    reference_high: float = math.inf,
    zero_low: float = -math.inf,
    zero_high: float = math.inf,
    smooth_window: int = 1,
    smooth_passes: int = 1,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    """Build processed rows and return shifted/smoothed/zeroed arrays."""

    shifted, reference_shift = shift_to_reference_window(
        curve.cv,
        curve.free_energy,
        reference_low=reference_low,
        reference_high=reference_high,
    )
    smoothed = moving_average_smooth(shifted, window_length=smooth_window, passes=smooth_passes)
    zeroed, zero_shift = zero_curve(curve.cv, smoothed, zero_low=zero_low, zero_high=zero_high)

    rows: List[Dict[str, object]] = []
    for index, (cv, raw_fe, shifted_fe, smooth_fe, zero_fe, unc) in enumerate(
        zip(curve.cv, curve.free_energy, shifted, smoothed, zeroed, curve.uncertainty)
    ):
        rows.append(
            {
                "dataset_key": curve.spec.dataset_key or curve.spec.path.stem,
                "label": curve.spec.label,
                "group": curve.spec.group,
                "source_path": str(curve.spec.path),
                "point_index": index,
                "cv": float(cv),
                "free_energy_raw_kj_mol": float(raw_fe),
                "free_energy_reference_shifted_kj_mol": float(shifted_fe),
                "free_energy_smooth_kj_mol": float(smooth_fe),
                "free_energy_smooth_zeroed_kj_mol": float(zero_fe),
                "reference_shift_value_kj_mol": reference_shift,
                "smooth_zero_shift_value_kj_mol": zero_shift,
                "uncertainty_kj_mol": float(unc) if np.isfinite(unc) else math.nan,
            }
        )
    return rows, shifted, smoothed, zeroed


def barrier_for_window(cv: np.ndarray, values: np.ndarray, low: float, high: float) -> Tuple[float, float, float, float, int]:
    """Return max-min barrier and extrema metadata for a CV window."""

    mask = window_mask(cv, low, high) & np.isfinite(values)
    if not np.any(mask):
        return math.nan, math.nan, math.nan, math.nan, 0
    x = np.asarray(cv, dtype=float)[mask]
    y = np.asarray(values, dtype=float)[mask]
    min_index = int(np.argmin(y))
    max_index = int(np.argmax(y))
    barrier = float(y[max_index] - y[min_index])
    return barrier, float(x[min_index]), float(y[min_index]), float(x[max_index]), int(mask.sum())


def build_barrier_rows(
    curve: FesCurve,
    shifted: np.ndarray,
    smoothed: np.ndarray,
    windows: Sequence[Tuple[str, float, float]],
) -> List[Dict[str, object]]:
    """Build barrier summary rows for one curve."""

    rows: List[Dict[str, object]] = []
    for name, low, high in windows:
        original_barrier, min_cv, min_fe, max_cv, n_points = barrier_for_window(curve.cv, shifted, low, high)
        smooth_barrier, smooth_min_cv, smooth_min_fe, smooth_max_cv, smooth_n_points = barrier_for_window(
            curve.cv,
            smoothed,
            low,
            high,
        )
        rows.append(
            {
                "dataset_key": curve.spec.dataset_key or curve.spec.path.stem,
                "label": curve.spec.label,
                "group": curve.spec.group,
                "source_path": str(curve.spec.path),
                "barrier_region": name,
                "cv_low": low,
                "cv_high": high,
                "n_points": n_points,
                "smooth_n_points": smooth_n_points,
                "barrier_original_kj_mol": original_barrier,
                "barrier_smooth_kj_mol": smooth_barrier,
                "change_smooth_minus_original_kj_mol": (
                    smooth_barrier - original_barrier
                    if math.isfinite(smooth_barrier) and math.isfinite(original_barrier)
                    else math.nan
                ),
                "original_min_cv": min_cv,
                "original_min_fe_kj_mol": min_fe,
                "original_max_cv": max_cv,
                "smooth_min_cv": smooth_min_cv,
                "smooth_min_fe_kj_mol": smooth_min_fe,
                "smooth_max_cv": smooth_max_cv,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "dataset_key",
        "label",
        "group",
        "barrier_region",
        "point_index",
        "cv",
        "cv_low",
        "cv_high",
        "n_points",
    ]
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend([key for key in fieldnames if key not in ordered])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def analyze_fes_barriers(
    curve_specs: Sequence[FesCurveSpec],
    output_dir: Path,
    windows: Sequence[Tuple[str, float, float]] = DEFAULT_BARRIER_WINDOWS,
    reference_low: float = -math.inf,
    reference_high: float = math.inf,
    zero_low: float = -math.inf,
    zero_high: float = math.inf,
    smooth_window: int = 1,
    smooth_passes: int = 1,
    cv_column: int = 0,
    free_energy_column: int = 1,
    uncertainty_column: Optional[int] = 2,
) -> Dict[str, Path]:
    """Process curves and write processed-curve/barrier CSV outputs."""

    if not curve_specs:
        raise ValueError("At least one FES curve is required")
    output_dir = Path(output_dir)
    processed_rows: List[Dict[str, object]] = []
    barrier_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for spec in curve_specs:
        curve = load_fes_curve(
            spec,
            cv_column=cv_column,
            free_energy_column=free_energy_column,
            uncertainty_column=uncertainty_column,
        )
        rows, shifted, smoothed, _zeroed = process_curve(
            curve,
            reference_low=reference_low,
            reference_high=reference_high,
            zero_low=zero_low,
            zero_high=zero_high,
            smooth_window=smooth_window,
            smooth_passes=smooth_passes,
        )
        processed_rows.extend(rows)
        barrier_rows.extend(build_barrier_rows(curve, shifted, smoothed, windows=windows))
        manifest_rows.append(
            {
                "dataset_key": spec.dataset_key or spec.path.stem,
                "label": spec.label,
                "group": spec.group,
                "path": str(spec.path),
                "n_points": int(len(curve.cv)),
                "cv_min": float(np.nanmin(curve.cv)),
                "cv_max": float(np.nanmax(curve.cv)),
            }
        )

    outputs = {
        "processed_curves": output_dir / "fes_processed_curves.csv",
        "barrier_summary": output_dir / "fes_barrier_summary.csv",
        "manifest": output_dir / "fes_input_manifest.csv",
    }
    _write_csv(outputs["processed_curves"], processed_rows)
    _write_csv(outputs["barrier_summary"], barrier_rows)
    _write_csv(outputs["manifest"], manifest_rows)
    return outputs


def _curve_specs_from_args(args: argparse.Namespace) -> List[FesCurveSpec]:
    specs: List[FesCurveSpec] = []
    if args.manifest is not None:
        specs.extend(load_curve_manifest(args.manifest))
    for values in args.curve or []:
        specs.append(parse_curve_spec(values))
    return specs


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process 1D FES curves and compute barrier summaries")
    parser.add_argument("--manifest", type=Path, help="CSV with path,label,group,dataset_key columns")
    parser.add_argument(
        "--curve",
        nargs=3,
        action="append",
        metavar=("PATH", "LABEL", "GROUP"),
        help="Explicit curve input; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--barrier-window", action="append", help="Window as name:low:high; may be repeated")
    parser.add_argument("--reference-low", type=float, default=-math.inf)
    parser.add_argument("--reference-high", type=float, default=math.inf)
    parser.add_argument("--zero-low", type=float, default=-math.inf)
    parser.add_argument("--zero-high", type=float, default=math.inf)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--cv-column", type=int, default=0, help="Zero-based CV column index")
    parser.add_argument("--free-energy-column", type=int, default=1, help="Zero-based free-energy column index")
    parser.add_argument("--uncertainty-column", type=int, default=2, help="Zero-based uncertainty column index; use -1 to disable")
    return parser.parse_args(argv)


def get_fes2d_grid_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build arguments for the standalone 2D FES grid processor."""

    parser = argparse.ArgumentParser(description="Process a regular 2D FES grid")
    parser.add_argument("--fes-file", type=Path, required=True, help="PLUMED-style 2D FES table")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for processed outputs")
    parser.add_argument("--x-range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Inclusive x range")
    parser.add_argument("--y-range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Inclusive y range")
    parser.add_argument("--max-fes", type=float, default=200.0, help="Maximum plotted FES value in kJ/mol")
    parser.add_argument("--smooth-sigma", type=float, default=0.8, help="Gaussian sigma in grid-bin units")
    parser.add_argument("--smooth-valid-threshold", type=float, default=0.25)
    parser.add_argument("--prefix", default="fes2d", help="Output filename prefix")
    parser.add_argument("--zero-scope", choices=("window", "all"), default="window")
    parser.add_argument("--missing-plot-value", choices=("max", "nan"), default="max")
    parser.add_argument("--no-grid", action="store_true", help="Skip long-form processed grid CSV")
    parser.add_argument("--write-plots", action="store_true", help="Write contour plot PNG outputs")
    parser.add_argument("--write-comparison", action="store_true", help="Also write raw-vs-smoothed comparison")
    parser.add_argument("--contour-levels", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="")
    parser.add_argument("--x-label")
    parser.add_argument("--y-label")
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args(argv)


def run_fes2d_grid(argv: Optional[Sequence[str]] = None) -> int:
    """Run 2D FES grid processing from CLI-style arguments."""

    args = get_fes2d_grid_args(argv)
    x_range = None if args.x_range is None else (float(args.x_range[0]), float(args.x_range[1]))
    y_range = None if args.y_range is None else (float(args.y_range[0]), float(args.y_range[1]))
    try:
        outputs = process_fes2d_grid(
            fes_file=args.fes_file,
            output_dir=args.output_dir,
            x_range=x_range,
            y_range=y_range,
            max_fes=float(args.max_fes),
            smooth_sigma=float(args.smooth_sigma),
            valid_threshold=float(args.smooth_valid_threshold),
            prefix=str(args.prefix),
            zero_scope=str(args.zero_scope),
            missing_plot_value=str(args.missing_plot_value),
            write_grid=not bool(args.no_grid),
            write_plots=bool(args.write_plots or args.write_comparison),
            write_comparison=bool(args.write_comparison),
            contour_levels=int(args.contour_levels),
            dpi=int(args.dpi),
            title=str(args.title),
            x_label=args.x_label,
            y_label=args.y_label,
            cmap=str(args.cmap),
        )
    except Exception as exc:
        print(f"2D FES grid processing failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        specs = _curve_specs_from_args(args)
        windows = [parse_window(item) for item in args.barrier_window] if args.barrier_window else list(DEFAULT_BARRIER_WINDOWS)
        outputs = analyze_fes_barriers(
            specs,
            output_dir=args.output_dir,
            windows=windows,
            reference_low=args.reference_low,
            reference_high=args.reference_high,
            zero_low=args.zero_low,
            zero_high=args.zero_high,
            smooth_window=args.smooth_window,
            smooth_passes=args.smooth_passes,
            cv_column=args.cv_column,
            free_energy_column=args.free_energy_column,
            uncertainty_column=None if int(args.uncertainty_column) < 0 else int(args.uncertainty_column),
        )
    except Exception as exc:
        print(f"FES barrier analysis failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
