"""Reusable 1D free-energy and barrier analysis utilities."""

from __future__ import annotations

import argparse
import csv
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
