"""Diagnostics for PLUMED CV tests from COLVAR, HILLS, and LAMMPS dumps."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from molsimflow.io.lammps_dump import (
    iter_lammps_dump_frames,
    minimum_image_vectors,
    wrap_point_to_box,
)

BIAS_COLUMNS = ("opes.bias", "opes_e.bias")
DIAGNOSTIC_COLUMNS = {"time", "n2_num", *BIAS_COLUMNS}
PRIMARY_CV_BY_KIND = {
    "nfilm": "nfilm",
    "cgs": "cgs",
    "footprint": "foot_total",
    "dz": "dz.z",
    "sum_cn": "sum_cn.sum",
}


@dataclass(frozen=True)
class PlumedTable:
    """Numeric PLUMED table with the first FIELDS header."""

    path: Path
    columns: tuple[str, ...]
    data: np.ndarray
    header_count: int
    skipped_lines: int

    def has(self, column: str) -> bool:
        return column in self.columns

    def column(self, column: str) -> np.ndarray:
        if column not in self.columns:
            raise KeyError(f"Missing column {column!r} in {self.path}")
        return self.data[:, self.columns.index(column)]

    @property
    def row_count(self) -> int:
        return int(self.data.shape[0])


@dataclass(frozen=True)
class PlumedDefinitions:
    """Selections parsed from the generated PLUMED input."""

    n2_pairs: tuple[tuple[int, int], ...]
    groups: Mapping[str, tuple[int, ...]]
    nfilm_radius_A: Optional[float]
    nfilm_lower_A: Optional[float]
    nfilm_upper_A: Optional[float]


@dataclass(frozen=True)
class DiagnosticConfig:
    run_dir: Path
    output_dir: Path
    case_label: str = ""
    cv_kind: str = "auto"
    colvar_name: str = "COLVAR"
    hills_name: str = "HILLS"
    plumed_name: str = "in.plumed"
    trajectory_name: str = "bubble_1k.lammpstrj"
    fs_per_step: float = 1.0
    hard_contact_cutoff_A: float = 6.0
    target_cv: Optional[str] = None
    nfilm_cutoff_mode: str = "radius"
    max_frames: Optional[int] = None
    colvar_time_tolerance_ps: float = 0.002
    make_plots: bool = True
    dpi: int = 180
    skip_last_data_line: bool = False
    phase_planes: tuple[tuple[str, str], ...] = ()
    plot_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticResult:
    output_dir: Path
    report_path: Path
    summary_path: Path
    physical_checks_path: Path
    geometry_validation_path: Optional[Path]


def read_plumed_table(path: Path, *, skip_last_data_line: bool = False) -> PlumedTable:
    """Read a PLUMED text table and ignore repeated restart headers."""

    header: Optional[list[str]] = None
    header_count = 0
    skipped = 0
    rows: list[list[float]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#! FIELDS"):
            header_count += 1
            if header is None:
                header = line.split()[2:]
            continue
        if line.startswith("#"):
            continue
        if header is None:
            skipped += 1
            continue
        parts = line.split()
        if len(parts) < len(header):
            skipped += 1
            continue
        try:
            rows.append([float(value) for value in parts[: len(header)]])
        except ValueError:
            skipped += 1
    if header is None:
        raise ValueError(f"No '#! FIELDS' header found in {path}")
    if skip_last_data_line and rows:
        rows = rows[:-1]
    data = np.asarray(rows, dtype=float)
    if data.size == 0:
        data = np.empty((0, len(header)), dtype=float)
    return PlumedTable(
        path=Path(path),
        columns=tuple(header),
        data=data,
        header_count=header_count,
        skipped_lines=skipped,
    )


def drop_duplicate_time_rows(table: PlumedTable, time_column: str = "time") -> tuple[PlumedTable, int]:
    """Keep the first row for each time value."""

    if table.row_count == 0 or time_column not in table.columns:
        return table, 0
    time = table.column(time_column)
    _, first_indices = np.unique(time, return_index=True)
    keep = np.sort(first_indices)
    dropped = table.row_count - int(keep.size)
    if dropped == 0:
        return table, 0
    return (
        PlumedTable(
            path=table.path,
            columns=table.columns,
            data=table.data[keep, :],
            header_count=table.header_count,
            skipped_lines=table.skipped_lines,
        ),
        dropped,
    )


def write_numeric_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{digits}g}"
    return str(value)


def _std(values: np.ndarray) -> float:
    finite = _finite(values)
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1))


def summarize_columns(table: PlumedTable) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in table.columns:
        values = table.column(column)
        finite = _finite(values)
        if finite.size == 0:
            rows.append(
                {
                    "column": column,
                    "count": len(values),
                    "finite_count": 0,
                    "finite_fraction": 0.0,
                    "min": math.nan,
                    "q05": math.nan,
                    "mean": math.nan,
                    "median": math.nan,
                    "q95": math.nan,
                    "max": math.nan,
                    "std": math.nan,
                    "span": math.nan,
                    "first": math.nan,
                    "last": math.nan,
                    "mean_abs_step_change": math.nan,
                }
            )
            continue
        diffs = np.diff(finite)
        rows.append(
            {
                "column": column,
                "count": len(values),
                "finite_count": int(finite.size),
                "finite_fraction": float(finite.size / max(1, len(values))),
                "min": float(np.min(finite)),
                "q05": float(np.quantile(finite, 0.05)),
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "q95": float(np.quantile(finite, 0.95)),
                "max": float(np.max(finite)),
                "std": _std(finite),
                "span": float(np.max(finite) - np.min(finite)),
                "first": float(finite[0]),
                "last": float(finite[-1]),
                "mean_abs_step_change": float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
            }
        )
    return rows


def pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 3:
        return math.nan
    x = x_values[mask]
    y = y_values[mask]
    if np.isclose(float(np.std(x)), 0.0) or np.isclose(float(np.std(y)), 0.0):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def infer_cv_kind(table: PlumedTable, requested: str) -> str:
    if requested != "auto":
        return requested
    if table.has("dz.z"):
        return "dz"
    if table.has("nfilm"):
        return "nfilm"
    if table.has("cgs"):
        return "cgs"
    if table.has("foot_total"):
        return "footprint"
    if table.has("sum_cn.sum"):
        return "sum_cn"
    raise ValueError(
        "Cannot infer CV kind from COLVAR columns; pass --cv-kind. "
        f"Columns: {', '.join(table.columns)}"
    )


def resolve_target_cv(table: PlumedTable, cv_kind: str, requested: Optional[str]) -> str:
    """Resolve and validate the target CV column."""

    if cv_kind == "generic" and requested is None:
        raise ValueError("--target-cv is required when --cv-kind=generic")
    target = requested or PRIMARY_CV_BY_KIND[cv_kind]
    if not table.has(target):
        raise KeyError(f"Missing target CV column {target!r} in {table.path}")
    return target


def validate_phase_planes(
    table: PlumedTable,
    phase_planes: Sequence[tuple[str, str]],
) -> None:
    """Validate requested phase-plane columns before writing diagnostics."""

    for x_column, y_column in phase_planes:
        missing = [column for column in (x_column, y_column) if not table.has(column)]
        if missing:
            raise KeyError(f"Missing phase-plane column(s) {missing!r} in {table.path}")


def resolve_plot_columns(table: PlumedTable, requested: Sequence[str]) -> list[str]:
    """Resolve optional plot-column selection and preserve user order."""

    columns = list(dict.fromkeys(requested)) if requested else cv_columns(table)
    missing = [column for column in columns if not table.has(column)]
    if missing:
        raise KeyError(f"Missing plot column(s) {missing!r} in {table.path}")
    return columns


def cv_columns(table: PlumedTable) -> list[str]:
    return [column for column in table.columns if column not in DIAGNOSTIC_COLUMNS]


def bias_total(table: PlumedTable) -> Optional[np.ndarray]:
    parts = [table.column(column) for column in BIAS_COLUMNS if table.has(column)]
    if not parts:
        return None
    total = np.zeros(table.row_count, dtype=float)
    for values in parts:
        total = total + values
    return total


def bias_label(table: PlumedTable) -> str:
    return " + ".join(column for column in BIAS_COLUMNS if table.has(column))


def _check(status: str, name: str, value: object, note: str) -> dict[str, object]:
    return {"status": status, "check": name, "value": value, "note": note}


def build_physical_checks(
    table: PlumedTable,
    hills: Optional[PlumedTable],
    cv_kind: str,
    target: str,
    geometry_rows: Optional[list[dict[str, object]]] = None,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    target_values = table.column(target)
    finite_fraction = float(np.isfinite(target_values).mean()) if target_values.size else 0.0
    checks.append(
        _check(
            "PASS" if np.isclose(finite_fraction, 1.0) else "FAIL",
            "target_cv_finite",
            finite_fraction,
            f"{target} should be finite for every printed step.",
        )
    )
    target_span = float(np.nanmax(target_values) - np.nanmin(target_values)) if target_values.size else math.nan
    checks.append(
        _check(
            "PASS" if target_span > 1.0e-6 else "WARN",
            "target_cv_dynamic_range",
            target_span,
            f"{target} is informative only if it changes over the sampled window.",
        )
    )
    if table.has("n2_num"):
        n2 = table.column("n2_num")
        finite_fraction_n2 = float(np.isfinite(n2).mean()) if n2.size else 0.0
        spread = float(np.nanmax(n2) - np.nanmin(n2)) if n2.size else math.nan
        checks.append(
            _check(
                "PASS" if np.isclose(finite_fraction_n2, 1.0) else "FAIL",
                "n2_num_finite_with_range",
                spread,
                "n2_num is retained from the original PLUMED diagnostics; its range is reported but not interpreted as a fixed molecule count.",
            )
        )
    total_bias = bias_total(table)
    if total_bias is not None and total_bias.size:
        bias_span = float(np.nanmax(total_bias) - np.nanmin(total_bias))
        checks.append(
            _check(
                "PASS" if bias_span > 1.0e-6 else "WARN",
                "bias_changes_after_deposition",
                bias_span,
                "Nonzero bias span indicates OPES stopped being only the initial constant offset.",
            )
        )
    if hills is not None:
        checks.append(
            _check(
                "PASS" if hills.row_count > 0 else "WARN",
                "hills_kernels_written",
                hills.row_count,
                "HILLS rows indicate OPES kernel deposition occurred.",
            )
        )
    if cv_kind == "cgs":
        split_cols = [column for column in ("cgs_ch3", "cgs_oh") if table.has(column)]
        if split_cols:
            split_sum = np.zeros(table.row_count, dtype=float)
            for column in split_cols:
                split_sum += table.column(column)
            err = float(np.nanmax(np.abs(table.column("cgs") - split_sum)))
            checks.append(
                _check(
                    "PASS" if err < 1.0e-5 else "FAIL",
                    "cgs_equals_split_sum",
                    err,
                    "Total cgs should equal cgs_ch3 + cgs_oh for mixed and single surfaces.",
                )
            )
        checks.append(
            _check(
                "PASS" if float(np.nanmax(target_values)) > 1.0e-5 else "WARN",
                "gas_surface_contact_sampled",
                float(np.nanmax(target_values)),
                "A zero maximum means the trajectory did not sample direct gas-terminal contact.",
            )
        )
    if cv_kind == "footprint":
        split_cols = [column for column in ("foot_ch3.morethan", "foot_oh.morethan") if table.has(column)]
        if split_cols:
            split_sum = np.zeros(table.row_count, dtype=float)
            for column in split_cols:
                split_sum += table.column(column)
            err = float(np.nanmax(np.abs(table.column("foot_total") - split_sum)))
            checks.append(
                _check(
                    "PASS" if err < 1.0e-5 else "FAIL",
                    "foot_total_equals_split_sum",
                    err,
                    "foot_total should equal the CH3/OH split coverage terms.",
                )
            )
        if table.has("phi_ch3"):
            phi = table.column("phi_ch3")
            outside = int(np.sum((phi < -1.0e-8) | (phi > 1.0 + 1.0e-8)))
            checks.append(
                _check(
                    "PASS" if outside == 0 else "FAIL",
                    "phi_ch3_between_zero_and_one",
                    outside,
                    "The CH3 footprint fraction should remain in [0, 1].",
                )
            )
        checks.append(
            _check(
                "PASS" if float(np.nanmax(target_values)) > 1.0e-5 else "WARN",
                "gas_footprint_sampled",
                float(np.nanmax(target_values)),
                "A zero maximum means no surface site was identified as gas-covered.",
            )
        )
    if cv_kind == "nfilm":
        checks.append(
            _check(
                "PASS" if float(np.nanmean(target_values)) > 0.0 else "WARN",
                "water_film_occupied",
                float(np.nanmean(target_values)),
                "The bottom water-film cylinder should contain water oxygens for a wet film.",
            )
        )
    if cv_kind == "dz" and table.has("dz.z") and table.has("cb_pos.z") and table.has("csurf_pos.z"):
        dz_from_positions = table.column("cb_pos.z") - table.column("csurf_pos.z")
        err = float(np.nanmax(np.abs(table.column("dz.z") - dz_from_positions)))
        checks.append(
            _check(
                "PASS" if err < 1.0e-4 else "WARN",
                "dz_z_matches_printed_centers",
                err,
                "dz.z should match cb_pos.z - csurf_pos.z for the printed NOPBC reference centers.",
            )
        )
    if cv_kind == "sum_cn":
        checks.append(
            _check(
                "PASS" if float(np.nanmax(target_values)) > 0.0 else "WARN",
                "sum_cn_positive",
                float(np.nanmax(target_values)) if target_values.size else math.nan,
                "sum_cn.sum should be positive when a non-empty N2 cluster is present.",
            )
        )
    ref_cols = [column for column in ("surf_ref_pos.x", "surf_ref_pos.y", "surf_ref_pos.z") if table.has(column)]
    if ref_cols:
        spans = []
        finite_ok = True
        for column in ref_cols:
            values = table.column(column)
            finite_ok = finite_ok and bool(np.isfinite(values).all())
            spans.append(f"{column}={float(np.nanmax(values) - np.nanmin(values)):.6g}")
        checks.append(
            _check(
                "PASS" if finite_ok else "FAIL",
                "surf_ref_position_finite_with_span",
                ";".join(spans),
                "surf_ref_pos is printed to diagnose dynamic cylinder-reference drift; x/y spans may include the chosen PBC image.",
            )
        )
    if geometry_rows:
        for key in ("hard_nfilm_count", "hard_cgs_total", "hard_foot_total"):
            if key not in geometry_rows[0]:
                continue
            hard = np.asarray([float(row[key]) for row in geometry_rows], dtype=float)
            cv = np.asarray([float(row.get(target, math.nan)) for row in geometry_rows], dtype=float)
            corr = pearson(cv, hard)
            hard_span = float(np.nanmax(hard) - np.nanmin(hard)) if hard.size else math.nan
            status = "PASS"
            note = "Independent hard-cutoff geometry should move in the same direction as the smooth CV."
            if not math.isfinite(corr):
                status = "WARN"
                note = "The hard geometry check was constant or too short for a Pearson correlation."
            elif corr < 0.3:
                status = "WARN"
            checks.append(_check(status, f"geometry_correlation_{key}", corr, note))
            checks.append(
                _check(
                    "PASS" if hard_span > 0.0 else "WARN",
                    f"geometry_dynamic_range_{key}",
                    hard_span,
                    "The hard geometric proxy needs sampled variation to validate sampling behavior.",
                )
            )
    return checks


def _expand_atom_expression(expr: str) -> tuple[int, ...]:
    ids: list[int] = []
    for chunk in expr.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            if ":" in right:
                stop_text, step_text = right.split(":", 1)
                step = int(step_text)
            else:
                stop_text = right
                step = 1
            start = int(left)
            stop = int(stop_text)
            if step == 0:
                raise ValueError(f"Invalid zero stride in atom expression {expr!r}")
            if start <= stop:
                ids.extend(range(start, stop + 1, abs(step)))
            else:
                ids.extend(range(start, stop - 1, -abs(step)))
        else:
            ids.append(int(item))
    return tuple(ids)


def parse_plumed_definitions(path: Path) -> PlumedDefinitions:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    groups: dict[str, tuple[int, ...]] = {}
    n2_pairs: list[tuple[int, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        group = re.match(r"^([A-Za-z0-9_.]+):\s+GROUP\s+ATOMS=([^\s]+)", line)
        if group:
            label, expr = group.groups()
            if re.fullmatch(r"[0-9,:\-]+", expr):
                groups[label] = _expand_atom_expression(expr)
            continue
        com = re.match(r"^c\d+:\s+COM\s+ATOMS=([0-9]+),([0-9]+)\b", line)
        if com:
            n2_pairs.append((int(com.group(1)), int(com.group(2))))
    radius = lower = upper = None
    incyl = re.search(
        r"nfilm:\s+INCYLINDER\b.*?RADIUS=\{(?:TANH|RATIONAL)\s+R_0=([0-9.eE+\-]+).*?"
        r"LOWER=([0-9.eE+\-]+)\s+UPPER=([0-9.eE+\-]+)",
        text,
    )
    if incyl:
        radius = float(incyl.group(1))
        lower = float(incyl.group(2))
        upper = float(incyl.group(3))
    return PlumedDefinitions(
        n2_pairs=tuple(n2_pairs),
        groups=groups,
        nfilm_radius_A=radius,
        nfilm_lower_A=lower,
        nfilm_upper_A=upper,
    )


def _pair_centers(
    positions: Mapping[int, np.ndarray],
    pairs: Sequence[tuple[int, int]],
    bounds: np.ndarray,
) -> np.ndarray:
    centers = []
    lengths = bounds[:, 1] - bounds[:, 0]
    for atom_a, atom_b in pairs:
        if atom_a not in positions or atom_b not in positions:
            continue
        a = positions[atom_a]
        b = positions[atom_b]
        delta = minimum_image_vectors(np.asarray([b - a]), lengths)[0]
        centers.append(wrap_point_to_box(a + 0.5 * delta, bounds))
    return np.asarray(centers, dtype=float)


def _group_coords(positions: Mapping[int, np.ndarray], atom_ids: Sequence[int]) -> np.ndarray:
    coords = [positions[atom_id] for atom_id in atom_ids if atom_id in positions]
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _pairwise_distances(a: np.ndarray, b: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0] if a.ndim == 2 else 0, b.shape[0] if b.ndim == 2 else 0))
    lengths = bounds[:, 1] - bounds[:, 0]
    deltas = a[:, None, :] - b[None, :, :]
    flat = minimum_image_vectors(deltas.reshape((-1, 3)), lengths).reshape(deltas.shape)
    return np.sqrt(np.sum(flat * flat, axis=2))


def _hard_contact_count(
    n2_centers: np.ndarray,
    surface_coords: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> tuple[int, float]:
    distances = _pairwise_distances(n2_centers, surface_coords, bounds)
    if distances.size == 0:
        return 0, math.nan
    return int(np.sum(distances <= cutoff_A)), float(np.min(distances))


def _hard_covered_sites(
    n2_centers: np.ndarray,
    surface_coords: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> int:
    distances = _pairwise_distances(surface_coords, n2_centers, bounds)
    if distances.size == 0:
        return 0
    return int(np.sum(np.min(distances, axis=1) <= cutoff_A))


def _periodic_xy_center(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    center = np.mean(coords, axis=0)
    lengths = bounds[:, 1] - bounds[:, 0]
    for dim in (0, 1):
        length = float(lengths[dim])
        if length <= 0.0:
            continue
        lo = float(bounds[dim, 0])
        scaled = (coords[:, dim] - lo) / length
        angles = 2.0 * np.pi * scaled
        mean_vector = np.exp(1j * angles).mean()
        if np.isclose(abs(mean_vector), 0.0):
            continue
        angle = np.angle(mean_vector)
        if angle < 0.0:
            angle += 2.0 * np.pi
        center[dim] = lo + (angle / (2.0 * np.pi)) * length
    return center


def _hard_nfilm_count(
    water_coords: np.ndarray,
    surf_coords: np.ndarray,
    bounds: np.ndarray,
    radius_A: float,
    lower_A: float,
    upper_A: float,
) -> int:
    if water_coords.size == 0 or surf_coords.size == 0:
        return 0
    center = _periodic_xy_center(surf_coords, bounds)
    lengths = bounds[:, 1] - bounds[:, 0]
    deltas = water_coords - center
    deltas[:, :2] = minimum_image_vectors(
        np.column_stack([deltas[:, 0], deltas[:, 1], np.zeros(len(deltas))]),
        lengths,
    )[:, :2]
    radial = np.sqrt(deltas[:, 0] * deltas[:, 0] + deltas[:, 1] * deltas[:, 1])
    axial = water_coords[:, 2] - center[2]
    mask = (radial <= radius_A) & (axial >= lower_A) & (axial <= upper_A)
    return int(np.sum(mask))


def _nearest_colvar_row(table: PlumedTable, time_ps: float, tolerance_ps: float) -> Optional[dict[str, float]]:
    if not table.has("time") or table.row_count == 0:
        return None
    times = table.column("time")
    idx = int(np.searchsorted(times, time_ps))
    candidates = []
    if idx < len(times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if idx + 1 < len(times):
        candidates.append(idx + 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(float(times[i]) - time_ps))
    if abs(float(times[best]) - time_ps) > tolerance_ps:
        return None
    return {column: float(table.data[best, col_idx]) for col_idx, column in enumerate(table.columns)}


def geometry_validation(
    config: DiagnosticConfig,
    table: PlumedTable,
    cv_kind: str,
    definitions: PlumedDefinitions,
) -> list[dict[str, object]]:
    if cv_kind not in {"nfilm", "cgs", "footprint"}:
        return []
    traj_path = config.run_dir / config.trajectory_name
    if not traj_path.exists() or not definitions.n2_pairs:
        return []
    needed: set[int] = set(atom for pair in definitions.n2_pairs for atom in pair)
    for group_name in ("surface_terminal_atoms", "surface_ch3", "surface_oh", "water_o", "surf_si"):
        needed.update(definitions.groups.get(group_name, ()))
    rows: list[dict[str, object]] = []
    for frame in iter_lammps_dump_frames(traj_path, needed_atom_ids=needed, max_frames=config.max_frames):
        time_ps = frame.timestep * config.fs_per_step / 1000.0
        matched = _nearest_colvar_row(table, time_ps, config.colvar_time_tolerance_ps)
        if matched is None:
            continue
        n2_centers = _pair_centers(frame.selected_positions, definitions.n2_pairs, frame.bounds)
        row: dict[str, object] = {
            "frame_index": frame.frame_index,
            "timestep": frame.timestep,
            "time_ps": time_ps,
        }
        for column in cv_columns(table):
            row[column] = matched.get(column, math.nan)
        if cv_kind in {"cgs", "footprint"}:
            total = _group_coords(
                frame.selected_positions,
                definitions.groups.get("surface_terminal_atoms", ()),
            )
            ch3 = _group_coords(frame.selected_positions, definitions.groups.get("surface_ch3", ()))
            oh = _group_coords(frame.selected_positions, definitions.groups.get("surface_oh", ()))
            total_count, total_min = _hard_contact_count(
                n2_centers,
                total,
                frame.bounds,
                config.hard_contact_cutoff_A,
            )
            row["hard_cgs_total"] = total_count
            row["min_n2_surface_distance_A"] = total_min
            if ch3.size:
                row["hard_cgs_ch3"] = _hard_contact_count(
                    n2_centers,
                    ch3,
                    frame.bounds,
                    config.hard_contact_cutoff_A,
                )[0]
            if oh.size:
                row["hard_cgs_oh"] = _hard_contact_count(
                    n2_centers,
                    oh,
                    frame.bounds,
                    config.hard_contact_cutoff_A,
                )[0]
            row["hard_foot_total"] = _hard_covered_sites(
                n2_centers,
                total,
                frame.bounds,
                config.hard_contact_cutoff_A,
            )
            if ch3.size:
                row["hard_foot_ch3"] = _hard_covered_sites(
                    n2_centers,
                    ch3,
                    frame.bounds,
                    config.hard_contact_cutoff_A,
                )
            if oh.size:
                row["hard_foot_oh"] = _hard_covered_sites(
                    n2_centers,
                    oh,
                    frame.bounds,
                    config.hard_contact_cutoff_A,
                )
        if cv_kind == "nfilm":
            water = _group_coords(frame.selected_positions, definitions.groups.get("water_o", ()))
            surf = _group_coords(frame.selected_positions, definitions.groups.get("surf_si", ()))
            if (
                definitions.nfilm_radius_A is not None
                and definitions.nfilm_lower_A is not None
                and definitions.nfilm_upper_A is not None
            ):
                row["hard_nfilm_count"] = _hard_nfilm_count(
                    water,
                    surf,
                    frame.bounds,
                    definitions.nfilm_radius_A,
                    definitions.nfilm_lower_A,
                    definitions.nfilm_upper_A,
                )
        rows.append(row)
    return rows


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _plot_phase_plane(
    plt,
    output_dir: Path,
    x_column: str,
    y_column: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color_values: np.ndarray,
    color_name: str,
    color_label: str,
    dpi: int,
) -> Path:
    mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(color_values)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=dpi)
    scatter = ax.scatter(
        x_values[mask],
        y_values[mask],
        c=color_values[mask],
        s=4,
        alpha=0.5,
        cmap="cividis",
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"{x_column} vs {y_column} colored by {color_label}")
    fig.colorbar(scatter, ax=ax, label=color_label)
    out = output_dir / (
        f"phase_plane_{_safe_filename(x_column)}_vs_{_safe_filename(y_column)}"
        f"_by_{color_name}.png"
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_plots(
    output_dir: Path,
    table: PlumedTable,
    cv_kind: str,
    target: str,
    geometry_rows: Sequence[Mapping[str, object]],
    dpi: int,
    phase_planes: Sequence[tuple[str, str]] = (),
    plot_columns: Sequence[str] = (),
) -> list[Path]:
    plt = _load_matplotlib()
    written: list[Path] = []
    time = table.column("time") if table.has("time") else np.arange(table.row_count)
    total_bias = bias_total(table)
    total_bias_label = bias_label(table)
    columns = resolve_plot_columns(table, plot_columns)
    for column in columns:
        values = table.column(column)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        ax.plot(time, values, lw=1.0)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(column)
        ax.set_title(f"{column} vs time")
        out = output_dir / f"timeseries_{_safe_filename(column)}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi)
        ax.hist(_finite(values), bins=60, color="#3b82f6", alpha=0.85)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"{column} distribution")
        out = output_dir / f"hist_{_safe_filename(column)}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

        if total_bias is not None:
            fig, ax = plt.subplots(figsize=(6, 4.8), dpi=dpi)
            scatter = ax.scatter(values, total_bias, s=4, alpha=0.45, c=time, cmap="cividis")
            ax.set_xlabel(column)
            ax.set_ylabel(total_bias_label)
            ax.set_title(f"Bias vs {column}")
            fig.colorbar(scatter, ax=ax, label="Time (ps)")
            out = output_dir / f"bias_vs_{_safe_filename(column)}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    if total_bias is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        ax.plot(time, total_bias, lw=1.0)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(total_bias_label)
        ax.set_title("Total OPES bias vs time")
        out = output_dir / "timeseries_bias_total.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    for x_column, y_column in phase_planes:
        x_values = table.column(x_column)
        y_values = table.column(y_column)
        if total_bias is not None:
            written.append(
                _plot_phase_plane(
                    plt,
                    output_dir,
                    x_column,
                    y_column,
                    x_values,
                    y_values,
                    total_bias,
                    "bias_total",
                    total_bias_label,
                    dpi,
                )
            )
        written.append(
            _plot_phase_plane(
                plt,
                output_dir,
                x_column,
                y_column,
                x_values,
                y_values,
                time,
                "time",
                "Time (ps)",
                dpi,
            )
        )

    if len(columns) >= 2:
        corr = np.asarray(
            [[pearson(table.column(x), table.column(y)) for y in columns] for x in columns],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(max(5, len(columns) * 0.8), 4.8), dpi=dpi)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        ax.set_yticks(range(len(columns)), columns)
        fig.colorbar(im, ax=ax, label="Pearson r")
        ax.set_title("CV correlation matrix")
        out = output_dir / "cv_correlation_heatmap.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    if geometry_rows:
        x = np.asarray([float(row.get(target, math.nan)) for row in geometry_rows], dtype=float)
        for hard_key in ("hard_nfilm_count", "hard_cgs_total", "hard_foot_total"):
            if hard_key not in geometry_rows[0]:
                continue
            y = np.asarray([float(row[hard_key]) for row in geometry_rows], dtype=float)
            fig, ax = plt.subplots(figsize=(6, 4.8), dpi=dpi)
            ax.scatter(x, y, s=18, alpha=0.75)
            ax.set_xlabel(target)
            ax.set_ylabel(hard_key)
            ax.set_title(f"Smooth CV vs hard geometry: {hard_key}")
            out = output_dir / f"geometry_{_safe_filename(target)}_vs_{hard_key}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    return written


def write_report(
    path: Path,
    config: DiagnosticConfig,
    cv_kind: str,
    target: str,
    table: PlumedTable,
    hills: Optional[PlumedTable],
    summary_rows: Sequence[Mapping[str, object]],
    checks: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    plot_paths: Sequence[Path],
) -> None:
    summary_by_column = {str(row["column"]): row for row in summary_rows}
    target_summary = summary_by_column[target]
    lines = [
        f"# PLUMED CV diagnostics: {config.case_label or config.run_dir.name}",
        "",
        f"Run directory: `{config.run_dir}`",
        f"CV kind: `{cv_kind}`; target column: `{target}`",
        f"COLVAR rows: {table.row_count}; time span: {_fmt(table.column('time')[0])} to {_fmt(table.column('time')[-1])} ps",
        f"COLVAR headers seen: {table.header_count}; skipped data lines: {table.skipped_lines}",
    ]
    if hills is not None:
        lines.append(f"HILLS kernels read: {hills.row_count}")
    lines.extend(
        [
            "",
            "## Target CV range",
            "",
            "| min | q05 | mean | median | q95 | max | span |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| "
            + " | ".join(
                _fmt(target_summary[key])
                for key in ("min", "q05", "mean", "median", "q95", "max", "span")
            )
            + " |",
            "",
            "## Physical checks",
            "",
            "| status | check | value | note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in checks:
        lines.append(
            "| {status} | {check} | {value} | {note} |".format(
                status=row["status"],
                check=row["check"],
                value=_fmt(row["value"]),
                note=str(row["note"]).replace("|", "/"),
            )
        )
    lines.extend(["", "## Interpretation", ""])
    if cv_kind == "nfilm":
        lines.append(
            "`nfilm` is a smooth water-oxygen occupancy in the dynamic Si-patch cylinder. "
            "The hard-count geometry validation reports how many water oxygens fall in the same cylinder."
        )
    elif cv_kind == "cgs":
        lines.append(
            "`cgs` is a smooth N2-terminal contact count. The split columns show whether contacts are "
            "assigned to CH3 C or OH O sites, and the hard-count validation counts N2 COM-terminal "
            f"pairs within {config.hard_contact_cutoff_A:g} A."
        )
    elif cv_kind == "footprint":
        lines.append(
            "`foot_total` is a smooth count of terminal sites covered by nearby N2. The hard-count "
            "validation counts terminal sites with at least one nearby N2 COM."
        )
    elif cv_kind == "dz":
        lines.append(
            "`dz.z` is the NOPBC z-separation between the printed surface reference center and "
            "the bubble center. It is an approach/proximity coordinate rather than a contact count."
        )
    elif cv_kind == "sum_cn":
        lines.append(
            "`sum_cn.sum` is the printed N2 cluster-property monitor used as the OPES target in "
            "this run. Interpret it together with `n2_num`, `foot_total`, and the bias trace."
        )
    else:
        lines.append(
            f"`{target}` is treated as a generic PLUMED CV target. Interpret it from the PLUMED "
            "definition and its correlations with the other printed CVs."
        )
    warn_fail = [row for row in checks if row["status"] != "PASS"]
    if warn_fail:
        lines.append("")
        lines.append("Warnings indicate either a possible definition issue or that the sampled trajectory did not contain the relevant physical event.")
    if geometry_rows:
        lines.extend(["", f"Geometry validation frames: {len(geometry_rows)}"])
    if plot_paths:
        lines.extend(["", "## Figures", ""])
        for plot_path in plot_paths:
            lines.append(f"- `{plot_path.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(config: DiagnosticConfig) -> DiagnosticResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    colvar = read_plumed_table(
        config.run_dir / config.colvar_name,
        skip_last_data_line=config.skip_last_data_line,
    )
    colvar, dropped_time = drop_duplicate_time_rows(colvar)
    hills_path = config.run_dir / config.hills_name
    hills = read_plumed_table(hills_path, skip_last_data_line=False) if hills_path.exists() else None
    cv_kind = infer_cv_kind(colvar, config.cv_kind)
    target = resolve_target_cv(colvar, cv_kind, config.target_cv)
    validate_phase_planes(colvar, config.phase_planes)
    resolve_plot_columns(colvar, config.plot_columns)
    definitions = parse_plumed_definitions(config.run_dir / config.plumed_name)

    clean_rows = colvar.data.tolist()
    write_numeric_csv(config.output_dir / "colvar_clean.csv", colvar.columns, clean_rows)
    summary_rows = summarize_columns(colvar)
    summary_path = config.output_dir / "cv_summary.csv"
    write_numeric_csv(
        summary_path,
        [
            "column",
            "count",
            "finite_count",
            "finite_fraction",
            "min",
            "q05",
            "mean",
            "median",
            "q95",
            "max",
            "std",
            "span",
            "first",
            "last",
            "mean_abs_step_change",
        ],
        ([row[column] for column in [
            "column",
            "count",
            "finite_count",
            "finite_fraction",
            "min",
            "q05",
            "mean",
            "median",
            "q95",
            "max",
            "std",
            "span",
            "first",
            "last",
            "mean_abs_step_change",
        ]] for row in summary_rows),
    )
    corr_rows = []
    columns = cv_columns(colvar)
    for x_column in columns:
        for y_column in columns:
            corr_rows.append([x_column, y_column, pearson(colvar.column(x_column), colvar.column(y_column))])
    write_numeric_csv(config.output_dir / "cv_correlations.csv", ["x", "y", "pearson_r"], corr_rows)

    geometry_rows = geometry_validation(config, colvar, cv_kind, definitions)
    geometry_path = None
    if geometry_rows:
        geometry_path = config.output_dir / "geometry_validation.csv"
        geometry_columns = list(geometry_rows[0].keys())
        write_numeric_csv(
            geometry_path,
            geometry_columns,
            ([row.get(column, "") for column in geometry_columns] for row in geometry_rows),
        )
    checks = build_physical_checks(colvar, hills, cv_kind, target, geometry_rows)
    checks.append(
        _check(
            "PASS" if dropped_time == 0 else "WARN",
            "duplicate_time_rows_dropped",
            dropped_time,
            "Dropped duplicate time rows are usually restart-boundary repeats.",
        )
    )
    physical_path = config.output_dir / "physical_checks.csv"
    write_numeric_csv(
        physical_path,
        ["status", "check", "value", "note"],
        ([row["status"], row["check"], row["value"], row["note"]] for row in checks),
    )
    plot_paths: list[Path] = []
    if config.make_plots:
        plot_paths = make_plots(
            config.output_dir,
            colvar,
            cv_kind,
            target,
            geometry_rows,
            config.dpi,
            config.phase_planes,
            config.plot_columns,
        )
    report_path = config.output_dir / "analysis_report.md"
    write_report(report_path, config, cv_kind, target, colvar, hills, summary_rows, checks, geometry_rows, plot_paths)
    return DiagnosticResult(
        output_dir=config.output_dir,
        report_path=report_path,
        summary_path=summary_path,
        physical_checks_path=physical_path,
        geometry_validation_path=geometry_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing COLVAR/HILLS/in.plumed")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostics")
    parser.add_argument("--case-label", default="")
    parser.add_argument(
        "--cv-kind",
        choices=["auto", "generic", "nfilm", "cgs", "footprint", "dz", "sum_cn"],
        default="auto",
    )
    parser.add_argument("--target-cv", help="Override target CV column for generic diagnostics")
    parser.add_argument("--colvar-name", default="COLVAR")
    parser.add_argument("--hills-name", default="HILLS")
    parser.add_argument("--plumed-name", default="in.plumed")
    parser.add_argument("--trajectory-name", default="bubble_1k.lammpstrj")
    parser.add_argument("--fs-per-step", type=float, default=1.0)
    parser.add_argument("--hard-contact-cutoff-A", type=float, default=6.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--colvar-time-tolerance-ps", type=float, default=0.002)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-column",
        action="append",
        help="Plot only this COLVAR column; may be repeated",
    )
    parser.add_argument(
        "--phase-plane",
        action="append",
        nargs=2,
        metavar=("X_COLUMN", "Y_COLUMN"),
        help="Plot X_COLUMN versus Y_COLUMN colored by total bias and time; may be repeated",
    )
    parser.add_argument(
        "--skip-last-data-line",
        action="store_true",
        help="Drop the final parsed COLVAR data row, useful while COLVAR is still being written",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_diagnostics(
        DiagnosticConfig(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            case_label=args.case_label,
            cv_kind=args.cv_kind,
            colvar_name=args.colvar_name,
            hills_name=args.hills_name,
            plumed_name=args.plumed_name,
            trajectory_name=args.trajectory_name,
            fs_per_step=args.fs_per_step,
            hard_contact_cutoff_A=args.hard_contact_cutoff_A,
            target_cv=args.target_cv,
            max_frames=args.max_frames,
            colvar_time_tolerance_ps=args.colvar_time_tolerance_ps,
            make_plots=not args.no_plots,
            dpi=args.dpi,
            skip_last_data_line=args.skip_last_data_line,
            phase_planes=tuple(tuple(pair) for pair in (args.phase_plane or [])),
            plot_columns=tuple(args.plot_column or []),
        )
    )
    print(f"output_dir={result.output_dir}")
    print(f"report={result.report_path}")
    print(f"summary={result.summary_path}")
    print(f"physical_checks={result.physical_checks_path}")
    if result.geometry_validation_path is not None:
        print(f"geometry_validation={result.geometry_validation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
