"""Gas contact-graph and radius-sum validation utilities."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.centroids import UnionFind


@dataclass(frozen=True)
class GasContactConfig:
    """Thresholds for gas contact-graph classification."""

    contact_cutoff_A: float = 6.0
    min_cross_contacts: int = 1
    connected_lcc_fraction: float = 0.95
    connected_group_fraction: float = 0.80


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _mean(values: Iterable[float]) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else math.nan


def _median(values: Iterable[float]) -> float:
    finite = _finite_values(values)
    return float(np.median(finite)) if finite.size else math.nan


def _probability(values: Iterable[object]) -> float:
    items = list(values)
    return float(np.mean([_as_bool(value) for value in items])) if items else math.nan


def pairwise_contact_edges(coms: np.ndarray, bounds: np.ndarray, cutoff_A: float) -> List[Tuple[int, int]]:
    """Return contact graph edges between gas COMs under orthorhombic PBC."""

    coords = np.asarray(coms, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coms must have shape (n, 3)")
    lengths = box_lengths(np.asarray(bounds, dtype=float))
    edges: List[Tuple[int, int]] = []
    cutoff = float(cutoff_A)
    for i in range(coords.shape[0]):
        deltas = minimum_image_vectors(coords[i + 1 :] - coords[i], lengths)
        distances = np.linalg.norm(deltas, axis=1)
        for offset in np.where(distances <= cutoff)[0]:
            edges.append((i, i + 1 + int(offset)))
    return edges


def component_summary(
    edges: Iterable[Tuple[int, int]],
    groups: Sequence[str],
    n_molecules: Optional[int] = None,
) -> Dict[str, object]:
    """Summarize gas contact-graph components and cross-group contacts."""

    n = int(n_molecules) if n_molecules is not None else len(groups)
    if n != len(groups):
        raise ValueError("n_molecules and groups length differ")
    uf = UnionFind(n)
    cross_contacts = 0
    for i, j in edges:
        i = int(i)
        j = int(j)
        uf.union(i, j)
        if groups[i] != groups[j]:
            cross_contacts += 1
    members: Dict[int, List[int]] = {}
    for index in range(n):
        members.setdefault(uf.find(index), []).append(index)
    components = sorted(members.values(), key=len, reverse=True)
    largest = components[0] if components else []
    second = components[1] if len(components) > 1 else []
    group_names = sorted({str(group) for group in groups})
    row: Dict[str, object] = {
        "n_components": len(components),
        "cross_contacts": int(cross_contacts),
        "largest_component_size": int(len(largest)),
        "second_component_size": int(len(second)),
        "largest_component_fraction": float(len(largest) / n) if n else math.nan,
    }
    for group in group_names:
        group_total = sum(1 for value in groups if str(value) == group)
        largest_count = sum(1 for index in largest if str(groups[index]) == group)
        second_count = sum(1 for index in second if str(groups[index]) == group)
        key = str(group).replace(" ", "_")
        row[f"largest_component_{key}"] = int(largest_count)
        row[f"second_component_{key}"] = int(second_count)
        row[f"largest_{key}_fraction_of_{key}"] = float(largest_count / group_total) if group_total else math.nan
    return row


def minimum_cross_distance(coms: np.ndarray, groups: Sequence[str], bounds: np.ndarray) -> float:
    """Return the minimum distance between molecules from different groups."""

    coords = np.asarray(coms, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coms must have shape (n, 3)")
    lengths = box_lengths(np.asarray(bounds, dtype=float))
    best = math.nan
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            if groups[i] == groups[j]:
                continue
            distance = float(np.linalg.norm(minimum_image_vectors(coords[j] - coords[i], lengths)))
            if not math.isfinite(best) or distance < best:
                best = distance
    return best


def gas_contact_metrics(
    coms: np.ndarray,
    groups: Sequence[str],
    bounds: np.ndarray,
    config: GasContactConfig = GasContactConfig(),
) -> Dict[str, object]:
    """Compute gas contact-graph metrics and boolean connectivity flags."""

    if len(groups) != len(coms):
        raise ValueError("groups length must match number of COMs")
    edges = pairwise_contact_edges(coms, bounds, cutoff_A=config.contact_cutoff_A)
    row = component_summary(edges, groups, n_molecules=len(groups))
    row["n_contact_edges"] = int(len(edges))
    row["min_cross_distance_A"] = minimum_cross_distance(coms, groups, bounds)
    group_names = sorted({str(group).replace(" ", "_") for group in groups})
    group_ok = all(
        _as_float(row.get(f"largest_{group}_fraction_of_{group}")) >= float(config.connected_group_fraction)
        for group in group_names
    )
    row["cross_contact_flag"] = int(row["cross_contacts"]) >= int(config.min_cross_contacts)
    row["gas_connected_flag"] = (
        _as_float(row["largest_component_fraction"]) >= float(config.connected_lcc_fraction)
        and group_ok
    )
    return row


def read_gas_frame_table(path: Path, radius_sum_A: Optional[float] = None, d_column: str = "d3d_all") -> List[Dict[str, object]]:
    """Read a precomputed gas-connectivity frame table."""

    rows: List[Dict[str, object]] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Frame table has no header: {path}")
        required = [d_column, "cross_contact_flag", "gas_connected_flag"]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError("Frame table missing required columns: " + ", ".join(missing))
        for row in reader:
            out: Dict[str, object] = dict(row)
            out[d_column] = _as_float(row.get(d_column))
            if radius_sum_A is not None:
                out["nominal_gap_from_radius_sum_A"] = out[d_column] - float(radius_sum_A)
                out["within_radius_sum_flag"] = out[d_column] <= float(radius_sum_A)
            out["cross_contact_flag"] = _as_bool(row.get("cross_contact_flag"))
            out["gas_connected_flag"] = _as_bool(row.get("gas_connected_flag"))
            rows.append(out)
    return [row for row in rows if math.isfinite(_as_float(row.get(d_column)))]


def parse_window(raw: str) -> Tuple[str, float, float]:
    """Parse `name:low:high` window syntax with inf bounds."""

    parts = str(raw).split(":")
    if len(parts) != 3:
        raise ValueError("Windows must use name:low:high syntax")
    name = parts[0].strip()
    low = _as_float(parts[1], -math.inf)
    high = _as_float(parts[2], math.inf)
    if not name:
        raise ValueError("Window name cannot be empty")
    if high <= low:
        raise ValueError(f"Invalid window bounds for {name}: high <= low")
    return name, low, high


def default_radius_sum_windows(radius_sum_A: float) -> Tuple[Tuple[str, float, float], ...]:
    """Return generic windows around a radius-sum reference."""

    rsum = float(radius_sum_A)
    return (
        ("far_above_radius_sum", rsum + 10.0, math.inf),
        ("outer_onset_radius_sum_plus_2_5", rsum + 2.0, rsum + 5.0),
        ("near_radius_sum_plus_0_2", rsum, rsum + 2.0),
        ("near_radius_sum_minus_0_2", rsum - 2.0, rsum),
        ("below_radius_sum_minus_2_5", rsum - 5.0, rsum - 2.0),
        ("deep_contact_d_le_20", -math.inf, 20.0),
    )


def _select_window(rows: Sequence[Mapping[str, object]], d_column: str, low: float, high: float) -> List[Mapping[str, object]]:
    selected: List[Mapping[str, object]] = []
    for row in rows:
        value = _as_float(row.get(d_column))
        if (math.isinf(low) or value >= low) and (math.isinf(high) or value < high):
            selected.append(row)
    return selected


def _summary_for_rows(rows: Sequence[Mapping[str, object]], d_column: str) -> Dict[str, object]:
    return {
        "n_frames": int(len(rows)),
        "d_min_A": min((_as_float(row.get(d_column)) for row in rows), default=math.nan),
        "d_max_A": max((_as_float(row.get(d_column)) for row in rows), default=math.nan),
        "cross_contact_probability": _probability(row.get("cross_contact_flag") for row in rows),
        "gas_connected_probability": _probability(row.get("gas_connected_flag") for row in rows),
        "cross_contacts_mean": _mean(_as_float(row.get("cross_contacts")) for row in rows),
        "cross_contacts_median": _median(_as_float(row.get("cross_contacts")) for row in rows),
        "largest_component_fraction_mean": _mean(_as_float(row.get("largest_component_fraction")) for row in rows),
        "largest_component_size_mean": _mean(_as_float(row.get("largest_component_size")) for row in rows),
        "second_component_size_mean": _mean(_as_float(row.get("second_component_size")) for row in rows),
        "min_cross_distance_mean_A": _mean(_as_float(row.get("min_cross_distance_A")) for row in rows),
    }


def summarize_distance_bins(
    rows: Sequence[Mapping[str, object]],
    d_column: str = "d3d_all",
    d_range: Tuple[float, float] = (0.0, 70.0),
    bin_width_A: float = 2.0,
    radius_sum_A: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Summarize gas contact probabilities in fixed distance bins."""

    low, high = float(d_range[0]), float(d_range[1])
    width = float(bin_width_A)
    if width <= 0.0 or high <= low:
        raise ValueError("Invalid d_range or bin_width_A")
    out: List[Dict[str, object]] = []
    edges = np.arange(low, high + 0.5 * width, width)
    for left, right in zip(edges[:-1], edges[1:]):
        selected = _select_window(rows, d_column, float(left), float(right))
        if not selected:
            continue
        row = {
            "d_bin_left_A": float(left),
            "d_bin_right_A": float(right),
            "d_bin_center_A": 0.5 * (float(left) + float(right)),
            **_summary_for_rows(selected, d_column),
        }
        if radius_sum_A is not None:
            row["nominal_gap_center_A"] = row["d_bin_center_A"] - float(radius_sum_A)
        out.append(row)
    return out


def summarize_windows(
    rows: Sequence[Mapping[str, object]],
    windows: Sequence[Tuple[str, float, float]],
    d_column: str = "d3d_all",
    radius_sum_A: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Summarize gas contact probabilities in named distance windows."""

    out: List[Dict[str, object]] = []
    for name, low, high in windows:
        selected = _select_window(rows, d_column, low, high)
        if not selected:
            continue
        row = {
            "window": name,
            "d_left_A": low,
            "d_right_A": high,
            **_summary_for_rows(selected, d_column),
        }
        if radius_sum_A is not None:
            row["nominal_gap_left_A"] = low - float(radius_sum_A) if math.isfinite(low) else math.nan
            row["nominal_gap_right_A"] = high - float(radius_sum_A) if math.isfinite(high) else math.nan
        out.append(row)
    return out


def summarize_thresholds(
    bin_rows: Sequence[Mapping[str, object]],
    min_frames: int = 3,
    thresholds: Sequence[float] = (0.25, 0.50, 0.75, 0.90, 0.99),
) -> List[Dict[str, object]]:
    """Find largest distance bins meeting contact-probability thresholds."""

    usable = [row for row in bin_rows if int(_as_float(row.get("n_frames"), 0.0)) >= int(min_frames)]
    out: List[Dict[str, object]] = []
    for metric in ("cross_contact_probability", "gas_connected_probability"):
        for threshold in thresholds:
            hits = [row for row in usable if _as_float(row.get(metric)) >= float(threshold)]
            if not hits:
                out.append(
                    {
                        "metric": metric,
                        "probability_threshold": float(threshold),
                        "largest_d_bin_right_A": math.nan,
                        "largest_d_bin_center_A": math.nan,
                        "nominal_gap_at_center_A": math.nan,
                        "n_frames": 0,
                        "probability": math.nan,
                    }
                )
                continue
            hit = max(hits, key=lambda row: _as_float(row.get("d_bin_center_A")))
            out.append(
                {
                    "metric": metric,
                    "probability_threshold": float(threshold),
                    "largest_d_bin_right_A": _as_float(hit.get("d_bin_right_A")),
                    "largest_d_bin_center_A": _as_float(hit.get("d_bin_center_A")),
                    "nominal_gap_at_center_A": _as_float(hit.get("nominal_gap_center_A")),
                    "n_frames": int(_as_float(hit.get("n_frames"), 0.0)),
                    "probability": _as_float(hit.get(metric)),
                }
            )
    return out


def transition_summary(
    rows: Sequence[Mapping[str, object]],
    radius_sum_A: float,
    d_column: str = "d3d_all",
) -> List[Dict[str, object]]:
    """Summarize contact state above/below a radius-sum reference."""

    rsum = float(radius_sum_A)
    scopes = {
        "radius_sum_window_minus2_plus2": [
            row for row in rows if rsum - 2.0 <= _as_float(row.get(d_column)) < rsum + 2.0
        ],
        "d_le_radius_sum": [row for row in rows if _as_float(row.get(d_column)) <= rsum],
        "d_gt_radius_sum": [row for row in rows if _as_float(row.get(d_column)) > rsum],
        "deep_contact_d_le_20": [row for row in rows if _as_float(row.get(d_column)) <= 20.0],
    }
    out: List[Dict[str, object]] = []
    for name, selected in scopes.items():
        if not selected:
            continue
        summary = _summary_for_rows(selected, d_column)
        out.append(
            {
                "scope": name,
                **summary,
                "nominal_gap_min_A": summary["d_min_A"] - rsum if math.isfinite(summary["d_min_A"]) else math.nan,
                "nominal_gap_max_A": summary["d_max_A"] - rsum if math.isfinite(summary["d_max_A"]) else math.nan,
            }
        )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


BIN_FIELDS = (
    "d_bin_left_A",
    "d_bin_right_A",
    "d_bin_center_A",
    "nominal_gap_center_A",
    "n_frames",
    "d_min_A",
    "d_max_A",
    "cross_contact_probability",
    "gas_connected_probability",
    "cross_contacts_mean",
    "cross_contacts_median",
    "largest_component_fraction_mean",
    "largest_component_size_mean",
    "second_component_size_mean",
    "min_cross_distance_mean_A",
)

WINDOW_FIELDS = (
    "window",
    "d_left_A",
    "d_right_A",
    "nominal_gap_left_A",
    "nominal_gap_right_A",
    "n_frames",
    "d_min_A",
    "d_max_A",
    "cross_contact_probability",
    "gas_connected_probability",
    "cross_contacts_mean",
    "cross_contacts_median",
    "largest_component_fraction_mean",
    "largest_component_size_mean",
    "second_component_size_mean",
    "min_cross_distance_mean_A",
)

THRESHOLD_FIELDS = (
    "metric",
    "probability_threshold",
    "largest_d_bin_right_A",
    "largest_d_bin_center_A",
    "nominal_gap_at_center_A",
    "n_frames",
    "probability",
)

TRANSITION_FIELDS = (
    "scope",
    "n_frames",
    "d_min_A",
    "d_max_A",
    "nominal_gap_min_A",
    "nominal_gap_max_A",
    "cross_contact_probability",
    "gas_connected_probability",
    "cross_contacts_mean",
    "cross_contacts_median",
    "largest_component_fraction_mean",
    "largest_component_size_mean",
    "second_component_size_mean",
    "min_cross_distance_mean_A",
)


def analyze_gas_contact_table(
    input_table: Path,
    output_dir: Path,
    radius_sum_A: float,
    d_column: str = "d3d_all",
    d_range: Tuple[float, float] = (0.0, 70.0),
    bin_width_A: float = 2.0,
    windows: Optional[Sequence[Tuple[str, float, float]]] = None,
    min_bin_frames: int = 3,
) -> Dict[str, Path]:
    """Analyze a precomputed gas-connectivity frame table."""

    rows = read_gas_frame_table(input_table, radius_sum_A=radius_sum_A, d_column=d_column)
    if not rows:
        raise ValueError(f"No valid rows found in {input_table}")
    selected_windows = tuple(windows) if windows is not None else default_radius_sum_windows(radius_sum_A)
    bin_rows = summarize_distance_bins(
        rows,
        d_column=d_column,
        d_range=d_range,
        bin_width_A=bin_width_A,
        radius_sum_A=radius_sum_A,
    )
    window_rows = summarize_windows(rows, selected_windows, d_column=d_column, radius_sum_A=radius_sum_A)
    threshold_rows = summarize_thresholds(bin_rows, min_frames=min_bin_frames)
    transition_rows = transition_summary(rows, radius_sum_A=radius_sum_A, d_column=d_column)

    output_dir = Path(output_dir)
    outputs = {
        "d_bin_summary": output_dir / "gas_contact_d_bin_summary.csv",
        "window_summary": output_dir / "gas_contact_window_summary.csv",
        "threshold_summary": output_dir / "gas_contact_threshold_summary.csv",
        "transition_summary": output_dir / "gas_contact_transition_summary.csv",
    }
    _write_csv(outputs["d_bin_summary"], bin_rows, BIN_FIELDS)
    _write_csv(outputs["window_summary"], window_rows, WINDOW_FIELDS)
    _write_csv(outputs["threshold_summary"], threshold_rows, THRESHOLD_FIELDS)
    _write_csv(outputs["transition_summary"], transition_rows, TRANSITION_FIELDS)
    return outputs


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize gas contact/connectivity around a radius-sum reference")
    parser.add_argument("--input-table", type=Path, required=True, help="gas_connectivity_frame_table.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius-sum-A", type=float, required=True)
    parser.add_argument("--d-column", default="d3d_all")
    parser.add_argument("--d-range", type=float, nargs=2, default=(0.0, 70.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--d-bin-width-A", type=float, default=2.0)
    parser.add_argument("--window", action="append", help="Window as name:low:high; may be repeated")
    parser.add_argument("--min-bin-frames", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        windows = [parse_window(item) for item in args.window] if args.window else None
        outputs = analyze_gas_contact_table(
            input_table=args.input_table,
            output_dir=args.output_dir,
            radius_sum_A=float(args.radius_sum_A),
            d_column=args.d_column,
            d_range=(float(args.d_range[0]), float(args.d_range[1])),
            bin_width_A=float(args.d_bin_width_A),
            windows=windows,
            min_bin_frames=int(args.min_bin_frames),
        )
    except Exception as exc:
        print(f"Gas contact summary failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
