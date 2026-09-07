"""Summarize restart-aware nanobubble geometry and whole-box thermo diagnostics.

The input core table is produced by ``nanobubble-attachment``.  Thermo
segments are selected explicitly from a TSV manifest; a higher ``priority``
replaces duplicate timesteps from an earlier restart segment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

KB_J_PER_K = 1.380649e-23
BAR_PER_PA = 1.0e-5
ANGSTROM3_TO_M3 = 1.0e-30

CORE_REQUIRED = {
    "step",
    "largest_cluster_n2_count",
    "dissolved_or_disconnected_n2_count",
    "bubble_height_q05_q95_A",
    "footprint_convex_hull_area_A2",
    "relative_shape_anisotropy",
    "bubble_lateral_displacement_A",
}
THERMO_COLUMNS = (
    "Temp",
    "Density",
    "PotEng",
    "KinEng",
    "TotEng",
    "Volume",
    "Press",
    "Pxx",
    "Pyy",
    "Pzz",
    "Pxy",
    "Pxz",
    "Pyz",
)
UNITS = {
    "largest_cluster_n2_count": "molecules",
    "dissolved_or_disconnected_n2_count": "molecules",
    "bubble_height_q05_q95_A": "A",
    "footprint_equivalent_radius_A": "A",
    "sphere_radius_candidate_A": "A",
    "gas_side_angle_candidate_deg": "degree",
    "spherical_cap_volume_candidate_A3": "A3",
    "gas_liquid_area_candidate_A2": "A2",
    "ideal_gas_pressure_cap_bar": "bar",
    "bubble_lateral_displacement_A": "A",
    "relative_shape_anisotropy": "1",
    "Temp": "K",
    "Density": "g cm-3",
    "PotEng": "eV",
    "KinEng": "eV",
    "TotEng": "eV",
    "Volume": "A3",
    "Press": "bar",
    "Pxx": "bar",
    "Pyy": "bar",
    "Pzz": "bar",
    "Pxy": "bar",
    "Pxz": "bar",
    "Pyz": "bar",
    "normal_minus_tangential_bar": "bar",
}


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_core(path: Path, timestep_fs: float, min_cluster: int, temperature_K: float) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or CORE_REQUIRED.difference(reader.fieldnames):
            raise ValueError(f"{path}: missing required core columns")
        rows: list[dict[str, object]] = []
        previous = -1
        for raw in reader:
            step = int(raw["step"])
            if step <= previous:
                raise ValueError(f"{path}: non-increasing core step {step}")
            previous = step
            cluster = int(float(raw["largest_cluster_n2_count"]))
            height = _float(raw, "bubble_height_q05_q95_A")
            footprint = _float(raw, "footprint_convex_hull_area_A2")
            fragmented = cluster < min_cluster
            radius = math.sqrt(footprint / math.pi) if footprint > 0.0 else math.nan
            valid = not fragmented and height > 0.0 and math.isfinite(radius) and radius > 0.0
            sphere_radius = (radius * radius + height * height) / (2.0 * height) if valid else math.nan
            gas_angle = math.degrees(2.0 * math.atan(height / radius)) if valid else math.nan
            volume = math.pi * height * (3.0 * radius * radius + height * height) / 6.0 if valid else math.nan
            area = math.pi * (radius * radius + height * height) if valid else math.nan
            pressure = (
                cluster * KB_J_PER_K * temperature_K / (volume * ANGSTROM3_TO_M3) * BAR_PER_PA
                if valid and volume > 0.0
                else math.nan
            )
            row: dict[str, object] = dict(raw)
            row.update(
                {
                    "time_ns": step * timestep_fs / 1.0e6,
                    "fragmented": fragmented,
                    "geometry_valid": valid,
                    "footprint_equivalent_radius_A": radius,
                    "sphere_radius_candidate_A": sphere_radius,
                    "mean_curvature_candidate_A-1": 1.0 / sphere_radius if valid else math.nan,
                    "gas_side_angle_candidate_deg": gas_angle,
                    "spherical_cap_volume_candidate_A3": volume,
                    "gas_liquid_area_candidate_A2": area,
                    "solid_gas_area_candidate_A2": math.pi * radius * radius if valid else math.nan,
                    "tpcl_length_candidate_A": 2.0 * math.pi * radius if valid else math.nan,
                    "n2_number_density_cap_A-3": cluster / volume if valid else math.nan,
                    "ideal_gas_pressure_cap_bar": pressure,
                }
            )
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least two core rows")
    return rows


def _read_thermo_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"path", "min_step", "max_step", "priority"}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{path}: expected TSV columns {sorted(required)}")
        rows = []
        for raw in reader:
            source = Path(raw["path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            lower, upper = int(raw["min_step"]), int(raw["max_step"])
            if lower < 0 or upper < lower:
                raise ValueError(f"{path}: invalid step range for {source}")
            rows.append({"path": source, "min_step": lower, "max_step": upper, "priority": int(raw["priority"])})
    if not rows:
        raise ValueError(f"{path}: no thermo sources")
    return sorted(rows, key=lambda row: int(row["priority"]))


def _read_one_thermo(path: Path, lower: int, upper: int, timestep_fs: float) -> dict[int, dict[str, object]]:
    header: list[str] | None = None
    selected: dict[int, dict[str, object]] = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        fields = line.split()
        if fields and fields[0] == "Step":
            header = fields
            continue
        if header is None or len(fields) != len(header):
            continue
        try:
            numeric = [float(value) for value in fields]
        except ValueError:
            continue
        step = int(numeric[0])
        if numeric[0] != step or not lower <= step <= upper:
            continue
        row: dict[str, object] = {"step": step, "time_ns": step * timestep_fs / 1.0e6}
        for name, value in zip(header[1:], numeric[1:]):
            if math.isfinite(value):
                row[name] = value
        selected[step] = row
    return selected


def _read_thermo(manifest: Path, timestep_fs: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    merged: dict[int, dict[str, object]] = {}
    inventory: list[dict[str, object]] = []
    for source in _read_thermo_manifest(manifest):
        rows = _read_one_thermo(source["path"], int(source["min_step"]), int(source["max_step"]), timestep_fs)
        inventory.append({**source, "selected_rows": len(rows)})
        merged.update(rows)
    result = [merged[step] for step in sorted(merged)]
    if len(result) < 2:
        raise ValueError(f"{manifest}: fewer than two merged thermo rows")
    if any(right["step"] <= left["step"] for left, right in zip(result, result[1:])):
        raise ValueError(f"{manifest}: non-increasing merged thermo")
    for row in result:
        if {"Pxx", "Pyy", "Pzz"}.issubset(row):
            row["normal_minus_tangential_bar"] = float(row["Pzz"]) - 0.5 * (float(row["Pxx"]) + float(row["Pyy"]))
    return result, inventory


def _finite(values: Iterable[object]) -> list[float]:
    result = []
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            result.append(candidate)
    return result


def _block_rows(case_id: str, source: str, rows: Sequence[dict[str, object]], metrics: Sequence[str], block_ns: float) -> list[dict[str, object]]:
    if block_ns <= 0:
        raise ValueError("block_ns must be positive")
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        time = float(row["time_ns"])
        index = math.floor(time / block_ns)
        for metric in metrics:
            values = _finite([row.get(metric)])
            if values:
                grouped[(metric, index)].append((time, values[0]))
    output = []
    for (metric, index), samples in sorted(grouped.items()):
        values = [value for _, value in samples]
        output.append(
            {
                "case_id": case_id,
                "source": source,
                "metric": metric,
                "unit": UNITS.get(metric, "native"),
                "block_index": index,
                "block_start_ns": index * block_ns,
                "block_end_ns": (index + 1) * block_ns,
                "sample_count": len(values),
                "mean": fmean(values),
                "std": stdev(values) if len(values) > 1 else math.nan,
            }
        )
    return output


def _summary(rows: Sequence[dict[str, object]], metrics: Sequence[str], lower: float, upper: float) -> list[dict[str, object]]:
    selected = [row for row in rows if lower < float(row["time_ns"]) <= upper]
    result = []
    for metric in metrics:
        values = _finite(row.get(metric) for row in selected)
        result.append(
            {
                "metric": metric,
                "unit": UNITS.get(metric, "native"),
                "window_start_ns_exclusive": lower,
                "window_end_ns_inclusive": upper,
                "sample_count": len(values),
                "mean": fmean(values) if values else math.nan,
                "std": stdev(values) if len(values) > 1 else math.nan,
            }
        )
    return result


def _plot(rows: Sequence[dict[str, object]], thermo: Sequence[dict[str, object]], output: Path, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figures = output / "figures"
    figures.mkdir()
    time = np.asarray([float(row["time_ns"]) for row in rows])
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(time, [float(row["largest_cluster_n2_count"]) for row in rows], lw=0.8, label="largest N2 cluster")
    axes[0].plot(time, [float(row["dissolved_or_disconnected_n2_count"]) for row in rows], lw=0.8, label="disconnected N2")
    axes[0].set_ylabel("N2 molecules"); axes[0].legend(frameon=False)
    axes[1].plot(time, [float(row["bubble_height_q05_q95_A"]) for row in rows], lw=0.8, label="height")
    axes[1].plot(time, [float(row["footprint_equivalent_radius_A"]) for row in rows], lw=0.8, label="footprint radius")
    axes[1].set(xlabel="Time (ns)", ylabel="Length (A)"); axes[1].legend(frameon=False)
    figure.tight_layout(); figure.savefig(figures / "01_integrity_geometry.png", dpi=300); plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(time, [float(row["sphere_radius_candidate_A"]) for row in rows], lw=0.8)
    axes[0].set_ylabel("Sphere radius proxy (A)")
    axes[1].plot(time, [float(row["gas_side_angle_candidate_deg"]) for row in rows], lw=0.8)
    axes[1].set(xlabel="Time (ns)", ylabel="Gas-side angle proxy (degree)")
    figure.tight_layout(); figure.savefig(figures / "02_capillary_proxies.png", dpi=300); plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(time, [float(row["bubble_lateral_displacement_A"]) for row in rows], lw=0.8)
    axes[0].set_ylabel("XY displacement (A)")
    axes[1].plot(time, [float(row["relative_shape_anisotropy"]) for row in rows], lw=0.8)
    axes[1].set(xlabel="Time (ns)", ylabel="Shape anisotropy")
    figure.tight_layout(); figure.savefig(figures / "03_motion_shape.png", dpi=300); plt.close(figure)
    thermo_time = np.asarray([float(row["time_ns"]) for row in thermo])
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(thermo_time, [float(row.get("Temp", math.nan)) for row in thermo], lw=0.6)
    axes[0].set_ylabel("Temperature (K)")
    axes[1].plot(thermo_time, [float(row.get("Press", math.nan)) for row in thermo], lw=0.6, label="whole-box P")
    axes[1].plot(thermo_time, [float(row.get("normal_minus_tangential_bar", math.nan)) for row in thermo], lw=0.6, label="Pnormal-Ptangent")
    axes[1].set(xlabel="Time (ns)", ylabel="Pressure diagnostic (bar)"); axes[1].legend(frameon=False)
    figure.tight_layout(); figure.savefig(figures / "04_whole_box_thermo.png", dpi=300); plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    geometry = _read_core(Path(args.core_metrics), args.timestep_fs, args.min_geometry_cluster_n2, args.temperature_K)
    thermo, inventory = _read_thermo(Path(args.thermo_manifest), args.timestep_fs)
    geometry_metrics = (
        "largest_cluster_n2_count", "dissolved_or_disconnected_n2_count", "bubble_height_q05_q95_A",
        "footprint_equivalent_radius_A", "sphere_radius_candidate_A", "gas_side_angle_candidate_deg",
        "spherical_cap_volume_candidate_A3", "ideal_gas_pressure_cap_bar", "bubble_lateral_displacement_A",
        "relative_shape_anisotropy",
    )
    thermo_metrics = tuple(metric for metric in (*THERMO_COLUMNS, "normal_minus_tangential_bar") if any(metric in row for row in thermo))
    _write_csv(output / "geometry_timeseries.csv", geometry)
    _write_csv(output / "thermo_timeseries.csv", thermo)
    _write_csv(output / "block_statistics.csv", _block_rows(args.case_id, "geometry", geometry, geometry_metrics, args.block_ns) + _block_rows(args.case_id, "thermo", thermo, thermo_metrics, args.block_ns))
    late = _summary(geometry, geometry_metrics, args.late_start_ns, args.late_end_ns) + _summary(thermo, thermo_metrics, args.late_start_ns, args.late_end_ns)
    _write_csv(output / "late_window_summary.csv", late)
    if not args.no_plots:
        if args.font_path is None or not args.font_path.is_file():
            raise FileNotFoundError("a readable --font-path is required when writing figures")
        _plot(geometry, thermo, output, args.font_path)
    validation = {
        "status": "PASS",
        "case_id": args.case_id,
        "checks": {
            "core_steps_strictly_increasing": True,
            "thermo_steps_strictly_increasing": True,
            "core_first_step": int(geometry[0]["step"]),
            "core_last_step": int(geometry[-1]["step"]),
            "thermo_first_step": int(thermo[0]["step"]),
            "thermo_last_step": int(thermo[-1]["step"]),
            "geometry_rows": len(geometry),
            "thermo_rows": len(thermo),
            "fragmented_geometry_rows": sum(bool(row["fragmented"]) for row in geometry),
            "geometry_valid_rows": sum(bool(row["geometry_valid"]) for row in geometry),
            "thermo_source_inventory": [{**row, "path": str(row["path"])} for row in inventory],
        },
        "claim_boundary": [
            "Spherical-cap values are molecular-center proxies, not density-dividing-surface contact angles.",
            "Ideal-gas pressure is a reference, not measured bubble pressure.",
            "Whole-box pressure and stress are not local surface tension or Laplace pressure.",
            "All comparisons are descriptive single-trajectory observations, not causal ion effects or replicate uncertainty.",
        ],
    }
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"case_id": args.case_id, "late_window": late, **validation["checks"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--core-metrics", type=Path, required=True)
    parser.add_argument("--thermo-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-K", type=float, default=330.0)
    parser.add_argument("--min-geometry-cluster-n2", type=int, default=250)
    parser.add_argument("--block-ns", type=float, default=0.5)
    parser.add_argument("--late-start-ns", type=float, default=8.0)
    parser.add_argument("--late-end-ns", type=float, default=10.0)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    validation = run(build_parser().parse_args(argv))
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
