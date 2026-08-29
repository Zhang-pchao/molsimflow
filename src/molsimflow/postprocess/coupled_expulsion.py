"""Coupled confined-film water and mobile-ion expulsion analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


MOBILE_SPECIES = ("H3O_plus", "OH_minus_bulk", "Na_plus", "Cl_minus")
PARTITIONS = ("film", "A_facing_shell", "B_facing_shell", "other_liquid")


def _pandas():
    import pandas as pd

    return pd


def film_axial_bounds(
    gap_A: np.ndarray | float,
    rho_A: np.ndarray | float,
    radius_a_A: float,
    radius_b_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the facing inner-surface bounds for cylindrical coordinates."""

    gap = np.asarray(gap_A, dtype=float)
    rho = np.asarray(rho_A, dtype=float)
    valid = rho <= min(radius_a_A, radius_b_A)
    d = gap + radius_a_A + radius_b_A
    left = -0.5 * d + np.sqrt(np.maximum(radius_a_A**2 - rho**2, 0.0))
    right = 0.5 * d - np.sqrt(np.maximum(radius_b_A**2 - rho**2, 0.0))
    return left, right, valid & (right >= left)


def in_confined_film(
    s_A: np.ndarray | float,
    rho_A: np.ndarray | float,
    gap_A: np.ndarray | float,
    radius_a_A: float,
    radius_b_A: float,
    rho_core_A: float,
) -> np.ndarray:
    """Select points inside the liquid film between the facing bubble surfaces."""

    s = np.asarray(s_A, dtype=float)
    rho = np.asarray(rho_A, dtype=float)
    left, right, valid = film_axial_bounds(gap_A, rho, radius_a_A, radius_b_A)
    return valid & (rho <= rho_core_A) & (s >= left) & (s <= right)


def confined_film_volume_A3(gap_A: np.ndarray | float, radius_a_A: float, radius_b_A: float, rho_core_A: float) -> np.ndarray:
    """Return the exact axisymmetric volume between two nominal spheres."""

    if rho_core_A <= 0 or rho_core_A >= min(radius_a_A, radius_b_A):
        raise ValueError("rho_core_A must lie between zero and both radii")
    gap = np.asarray(gap_A, dtype=float)
    d = gap + radius_a_A + radius_b_A

    def cap_integral(radius: float) -> float:
        return (radius**3 - (radius**2 - rho_core_A**2) ** 1.5) / 3.0

    return math.pi * d * rho_core_A**2 - 2.0 * math.pi * (
        cap_integral(radius_a_A) + cap_integral(radius_b_A)
    )


def classify_mobile_partition(
    s_A: float,
    gap_A: float,
    surface_delta_a_A: float,
    surface_delta_b_A: float,
    in_film: bool,
    radius_a_A: float,
    radius_b_A: float,
    shell_width_A: float,
) -> str:
    """Assign one mutually exclusive spatial partition to a mobile ion."""

    if in_film:
        return "film"
    d = gap_A + radius_a_A + radius_b_A
    candidates: list[tuple[float, str]] = []
    if 0.0 <= surface_delta_a_A <= shell_width_A and s_A + 0.5 * d >= 0.0:
        candidates.append((surface_delta_a_A, "A_facing_shell"))
    if 0.0 <= surface_delta_b_A <= shell_width_A and s_A - 0.5 * d <= 0.0:
        candidates.append((surface_delta_b_A, "B_facing_shell"))
    return min(candidates)[1] if candidates else "other_liquid"


def _gap_label(gap_A: float, width_A: float) -> str:
    left = math.floor(gap_A / width_A) * width_A
    return f"{left:g}-{left + width_A:g}A"


def _time_key(values):
    return np.round(np.asarray(values, dtype=float), 9)


def _read_manifest_row(path: Path, case_label: str) -> Mapping[str, str]:
    with Path(path).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("case_label") == case_label]
    if len(rows) != 1:
        raise ValueError(f"Expected one case_label={case_label!r}, found {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _artifact_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def _source_rows(cases: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        sources = [("water_trace", case["trace_csv"])]
        for item in case["segment_specs"].split(";"):
            label, path = item.split("=", 1)
            sources.append((f"trajectory:{label}", path))
        for role in ("ion_frame_csv", "ion_samples_csv"):
            if case.get(role, "").strip():
                sources.append((role, case[role]))
        for role, raw_path in sources:
            path = Path(raw_path)
            rows.append(
                {
                    "case_label": case["case_label"],
                    "role": role,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
    return rows


def _metric_values(data, weights) -> dict[str, float]:
    wet = data["n_water_film"].to_numpy(float) > 0
    water_weight = float(np.sum(weights * data["n_water_film"].to_numpy(float)))
    ion_weight = float(np.sum(weights * data["n_mobile_ions_film"].to_numpy(float)))
    ion_free = data["ion_free_wet"].to_numpy(float)
    ion_free_valid = wet & np.isfinite(ion_free)
    ion_free_weight = float(np.sum(weights[ion_free_valid]))
    return {
        "water_density_A-3": float(np.average(data["water_density_A-3"], weights=weights)),
        "mobile_ion_density_A-3": float(np.average(data["mobile_ion_density_A-3"], weights=weights)),
        "mobile_ions_per_100_water": 100.0 * ion_weight / water_weight if water_weight else math.nan,
        "ion_free_wet_probability": float(np.sum(weights[ion_free_valid] * ion_free[ion_free_valid]) / ion_free_weight) if ion_free_weight else math.nan,
    }


def _summaries(frame, block_ns: float, bootstrap_samples: int, random_seed: int, wide_left_A: float, wide_right_A: float):
    pd = _pandas()
    data = frame.copy()
    data["block_id"] = np.floor(data["time_ns"].to_numpy(float) / block_ns + 1.0e-10).astype(int)
    blocks = np.sort(data["block_id"].unique())
    bins = sorted(data["gap_bin"].unique(), key=lambda value: float(value.split("-", 1)[0]))
    wide = data[(data.gap_A >= wide_left_A) & (data.gap_A < wide_right_A)]
    if wide.empty:
        raise ValueError(f"No frames in wide reference {wide_left_A:g}-{wide_right_A:g} A")

    point_wide = _metric_values(wide, np.ones(len(wide)))
    point: dict[tuple[str, str], float] = {}
    support: dict[str, tuple[int, int]] = {}
    for gap_bin in bins:
        chunk = data[data.gap_bin == gap_bin]
        values = _metric_values(chunk, np.ones(len(chunk)))
        values["water_density_wide_normalized"] = values["water_density_A-3"] / point_wide["water_density_A-3"]
        values["ion_density_wide_normalized"] = (
            values["mobile_ion_density_A-3"] / point_wide["mobile_ion_density_A-3"]
            if point_wide["mobile_ion_density_A-3"] > 0
            else math.nan
        )
        values["ion_water_selectivity_wide_normalized"] = (
            values["mobile_ions_per_100_water"] / point_wide["mobile_ions_per_100_water"]
            if point_wide["mobile_ions_per_100_water"] > 0
            else math.nan
        )
        point.update({(gap_bin, metric): value for metric, value in values.items()})
        support[gap_bin] = (len(chunk), int(chunk.block_id.nunique()))

    draws: dict[tuple[str, str], list[float]] = {key: [] for key in point}
    rng = np.random.default_rng(random_seed)
    for _ in range(bootstrap_samples):
        multiplicity = Counter(rng.choice(blocks, size=len(blocks), replace=True).tolist())
        weights = data["block_id"].map(multiplicity).to_numpy(float)
        wide_mask = (data.gap_A >= wide_left_A) & (data.gap_A < wide_right_A) & (weights > 0)
        if not np.any(wide_mask):
            continue
        wide_values = _metric_values(data[wide_mask], weights[wide_mask])
        for gap_bin in bins:
            mask = data.gap_bin.eq(gap_bin).to_numpy() & (weights > 0)
            if not np.any(mask):
                continue
            values = _metric_values(data[mask], weights[mask])
            values["water_density_wide_normalized"] = values["water_density_A-3"] / wide_values["water_density_A-3"]
            values["ion_density_wide_normalized"] = (
                values["mobile_ion_density_A-3"] / wide_values["mobile_ion_density_A-3"]
                if wide_values["mobile_ion_density_A-3"] > 0
                else math.nan
            )
            values["ion_water_selectivity_wide_normalized"] = (
                values["mobile_ions_per_100_water"] / wide_values["mobile_ions_per_100_water"]
                if wide_values["mobile_ions_per_100_water"] > 0
                else math.nan
            )
            for metric, value in values.items():
                if math.isfinite(value):
                    draws[(gap_bin, metric)].append(value)

    rows = []
    for (gap_bin, metric), value in point.items():
        samples = np.asarray(draws[(gap_bin, metric)], dtype=float)
        left = float(gap_bin.split("-", 1)[0])
        rows.append(
            {
                "case_label": str(data.case_label.iloc[0]),
                "gap_bin": gap_bin,
                "gap_center_A": left + 1.0,
                "frame_count": support[gap_bin][0],
                "effective_block_count": support[gap_bin][1],
                "metric": metric,
                "mean": value,
                "ci95_low": float(np.quantile(samples, 0.025)) if len(samples) >= 20 else math.nan,
                "ci95_high": float(np.quantile(samples, 0.975)) if len(samples) >= 20 else math.nan,
                "bootstrap_draw_count": len(samples),
            }
        )
    return pd.DataFrame(rows)


def _partition_summary(frame, block_ns: float, bootstrap_samples: int, random_seed: int):
    windows = ((2.0, 6.0, "near 2-6A"), (6.0, 10.0, "intermediate 6-10A"), (10.0, 14.0, "wide 10-14A"))
    data = frame.copy()
    data["block_id"] = np.floor(data.time_ns.to_numpy(float) / block_ns + 1.0e-10).astype(int)
    rng = np.random.default_rng(random_seed + 11)
    rows = []
    for left, right, label in windows:
        chunk = data[(data.gap_A >= left) & (data.gap_A < right)]
        blocks = np.sort(chunk.block_id.unique())
        totals = {name: float(chunk[f"mobile_{name}"].sum()) for name in PARTITIONS}
        denominator = sum(totals.values())
        draws = {name: [] for name in PARTITIONS}
        for _ in range(bootstrap_samples):
            multiplicity = Counter(rng.choice(blocks, size=len(blocks), replace=True).tolist())
            weights = chunk.block_id.map(multiplicity).to_numpy(float)
            sampled = {name: float(np.sum(weights * chunk[f"mobile_{name}"].to_numpy(float))) for name in PARTITIONS}
            sampled_total = sum(sampled.values())
            if sampled_total:
                for name in PARTITIONS:
                    draws[name].append(sampled[name] / sampled_total)
        for name in PARTITIONS:
            samples = np.asarray(draws[name])
            rows.append(
                {
                    "case_label": str(data.case_label.iloc[0]),
                    "window": label,
                    "window_left_A": left,
                    "window_right_A": right,
                    "frame_count": len(chunk),
                    "effective_block_count": len(blocks),
                    "partition": name,
                    "mobile_ion_observations": denominator,
                    "fraction": totals[name] / denominator if denominator else math.nan,
                    "ci95_low": float(np.quantile(samples, 0.025)) if len(samples) >= 20 else math.nan,
                    "ci95_high": float(np.quantile(samples, 0.975)) if len(samples) >= 20 else math.nan,
                }
            )
    return rows


def analyze_case(args) -> int:
    pd = _pandas()
    case = _read_manifest_row(Path(args.case_manifest), args.case_label)
    radius_a = float(case["nominal_radius_a_A"])
    radius_b = float(case["nominal_radius_b_A"])
    water_root = Path(case["water_output_dir"])
    water_frames = pd.read_csv(water_root / "frame_summary.csv")
    water_samples = pd.read_csv(water_root / "water_orientation_samples.csv.gz")
    water_frames["time_key"] = _time_key(water_frames.time_ns)
    water_samples["time_key"] = _time_key(water_samples.time_ns)
    water_frames["gap_A"] = water_frames.nominal_gap_A.astype(float)
    water_frames = water_frames[(water_frames.gap_A >= args.gap_min_A) & (water_frames.gap_A < args.gap_max_A)].copy()

    has_ions = bool(case.get("ion_samples_csv", "").strip())
    ion_samples = pd.DataFrame()
    ion_frame_lookup: dict[float, int] = {}
    if has_ions:
        ion_frames = pd.read_csv(case["ion_frame_csv"])
        ion_frames["time_key"] = _time_key(ion_frames.time_ns)
        ion_frame_lookup = dict(zip(ion_frames.time_key, ion_frames.global_frame.astype(int)))
        water_frames = water_frames[water_frames.time_key.isin(ion_frame_lookup)].copy()
        water_samples = water_samples[water_samples.time_key.isin(set(water_frames.time_key))].copy()
        ion_samples = pd.read_csv(case["ion_samples_csv"])
        ion_samples["time_key"] = _time_key(ion_samples.time_ns)
        ion_samples = ion_samples[ion_samples.time_key.isin(set(water_frames.time_key))].copy()

    water_mask = in_confined_film(
        water_samples.s_A,
        water_samples.rho_A,
        water_samples.nominal_gap_A,
        radius_a,
        radius_b,
        args.rho_core_A,
    )
    water_counts = water_samples.loc[water_mask].groupby("time_key").size()
    water_frames["n_water_film"] = water_frames.time_key.map(water_counts).fillna(0).astype(int)
    water_frames["film_volume_A3"] = confined_film_volume_A3(water_frames.gap_A, radius_a, radius_b, args.rho_core_A)
    water_frames["water_density_A-3"] = water_frames.n_water_film / water_frames.film_volume_A3
    water_frames["global_frame"] = water_frames.time_key.map(ion_frame_lookup).fillna(np.rint(water_frames.time_ns * 1000)).astype(int)
    water_frames["gap_bin"] = [_gap_label(value, args.gap_bin_width_A) for value in water_frames.gap_A]
    slab_intersects = case.get("has_tio2", "0") in {"1", "true", "True"} and (
        water_frames.surface_z_mid_A.fillna(-math.inf) >= -args.rho_core_A
    ).any()
    if slab_intersects:
        raise ValueError("TiO2 surface intersects the nominal film core; axisymmetric volume is invalid")

    for column in ["n_mobile_ions_film", "n_surface_markers"] + [f"mobile_{name}" for name in PARTITIONS]:
        water_frames[column] = math.nan if not has_ions else 0
    inside_bubble_count = 0
    if has_ions and not ion_samples.empty:
        ion_mask = in_confined_film(
            ion_samples.s_A,
            ion_samples.rho_A,
            ion_samples.gap_A,
            radius_a,
            radius_b,
            args.rho_core_A,
        )
        ion_samples["strict_film"] = ion_mask
        mobile = ion_samples[ion_samples.species.isin(args.mobile_species)].copy()
        inside_bubble_count = int(((mobile.r_A_A < radius_a) | (mobile.r_B_A < radius_b)).sum())
        mobile = mobile[(mobile.r_A_A >= radius_a) & (mobile.r_B_A >= radius_b)].copy()
        mobile["partition"] = [
            classify_mobile_partition(
                row.s_A,
                row.gap_A,
                row.surface_delta_A_A,
                row.surface_delta_B_A,
                bool(row.strict_film),
                radius_a,
                radius_b,
                args.shell_width_A,
            )
            for row in mobile.itertuples(index=False)
        ]
        film_counts = mobile[mobile.strict_film].groupby("time_key").size()
        surface_counts = ion_samples[~ion_samples.species.isin(args.mobile_species)].groupby("time_key").size()
        water_frames["n_mobile_ions_film"] = water_frames.time_key.map(film_counts).fillna(0).astype(int)
        water_frames["n_surface_markers"] = water_frames.time_key.map(surface_counts).fillna(0).astype(int)
        for name in PARTITIONS:
            counts = mobile[mobile.partition.eq(name)].groupby("time_key").size()
            water_frames[f"mobile_{name}"] = water_frames.time_key.map(counts).fillna(0).astype(int)

    water_frames["mobile_ion_density_A-3"] = water_frames.n_mobile_ions_film / water_frames.film_volume_A3
    water_frames["ion_free_wet"] = np.where(
        has_ions & (water_frames.n_water_film > 0),
        (water_frames.n_mobile_ions_film == 0).astype(float),
        math.nan,
    )
    water_frames["case_label"] = args.case_label
    keep = [
        "case_label", "segment", "local_frame", "global_frame", "timestep", "time_ns", "gap_A", "gap_bin",
        "film_volume_A3", "n_water_film", "water_density_A-3", "n_mobile_ions_film", "mobile_ion_density_A-3",
        "ion_free_wet", "n_surface_markers", *[f"mobile_{name}" for name in PARTITIONS], "surface_z_mid_A",
    ]
    frame_output = water_frames[keep].sort_values("time_ns")
    output = Path(args.output_root) / args.case_label
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "coupled_frame_summary.csv"
    summary_path = output / "coupled_gap_summary.csv"
    partition_path = output / "mobile_ion_partition_summary.csv"
    frame_output.to_csv(frame_path, index=False)
    _summaries(frame_output, args.block_ns, args.bootstrap_samples, args.random_seed, args.wide_left_A, args.wide_right_A).to_csv(summary_path, index=False)
    _write_csv(partition_path, _partition_summary(frame_output, args.block_ns, args.bootstrap_samples, args.random_seed) if has_ions else [])
    stats = [
        {"metric": "case_label", "value": args.case_label},
        {"metric": "canonical_frame_count", "value": len(frame_output)},
        {"metric": "water_sample_count", "value": len(water_samples)},
        {"metric": "charged_sample_count", "value": len(ion_samples)},
        {"metric": "mobile_rows_inside_nominal_bubbles_excluded", "value": inside_bubble_count},
        {"metric": "rho_core_A", "value": args.rho_core_A},
        {"metric": "shell_width_A", "value": args.shell_width_A},
        {"metric": "wide_reference_A", "value": f"{args.wide_left_A:g}-{args.wide_right_A:g}"},
        {"metric": "mobile_species", "value": ";".join(args.mobile_species)},
        {"metric": "formal_charge_scope", "value": "classified species only; not exact electrostatics"},
    ]
    stats_path = output / "state_statistics.csv"
    _write_csv(stats_path, stats)
    manifest_path = output / "artifact_manifest.csv"
    _write_csv(manifest_path, _artifact_rows((frame_path, summary_path, partition_path, stats_path)))
    print(f"case={args.case_label} frames={len(frame_output)} output={output}")
    return 0


def _plot_series(ax, summary, metric: str, labels: Sequence[str], colors: Mapping[str, str], ylabel: str) -> None:
    for label in labels:
        q = summary[(summary.case_label == label) & (summary.metric == metric) & (summary.gap_center_A < 14)].sort_values("gap_center_A")
        if q.empty or not q["mean"].notna().any():
            continue
        supported = q.effective_block_count >= 4
        ax.plot(q.gap_center_A, q["mean"], color=colors[label], lw=1.5, label=label)
        ax.fill_between(q.gap_center_A, q.ci95_low, q.ci95_high, color=colors[label], alpha=0.10)
        ax.scatter(q.loc[supported, "gap_center_A"], q.loc[supported, "mean"], color=colors[label], s=28, zorder=3)
        ax.scatter(q.loc[~supported, "gap_center_A"], q.loc[~supported, "mean"], facecolor="white", edgecolor=colors[label], s=34, zorder=3)
    ax.set_xlabel("Nominal gap h (Å)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7)


def assemble(args) -> int:
    pd = _pandas()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(args.case_manifest).open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    labels = [row["case_label"] for row in cases]
    outputs = Path(args.output_root)
    summaries = pd.concat([pd.read_csv(outputs / label / "coupled_gap_summary.csv") for label in labels], ignore_index=True)
    frames = pd.concat([pd.read_csv(outputs / label / "coupled_frame_summary.csv") for label in labels], ignore_index=True)
    partition_parts = []
    for label in labels:
        path = outputs / label / "mobile_ion_partition_summary.csv"
        if path.exists() and path.stat().st_size > 1:
            try:
                part = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if not part.empty:
                partition_parts.append(part)
    partitions = pd.concat(partition_parts, ignore_index=True) if partition_parts else pd.DataFrame()

    figure_dir = Path(args.figure_dir)
    study_root = figure_dir.parent.parent
    plot_data = figure_dir / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(plot_data / "multicase_coupled_gap_summary.csv", index=False)
    frames.to_csv(plot_data / "multicase_coupled_frame_summary.csv", index=False)
    partitions.to_csv(plot_data / "multicase_mobile_ion_partition_summary.csv", index=False)

    colors = {
        "Bulk-water-S": "#666666",
        "TiO2-water-S": "#2c7fb8",
        "TiO2-NaCl-S": "#756bb1",
        "TiO2-NaOH-S": "#8c8c2f",
        "TiO2-HCl-S": "#d95f8d",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.78, hspace=0.34, wspace=0.22)
    _plot_series(axes[0, 0], summaries, "water_density_wide_normalized", labels, colors, "Film water density / wide-gap value")
    axes[0, 0].axhline(1.0, color="#333333", lw=0.8, ls="--")
    axes[0, 0].set_title("a  Confined-film water density")

    ion_labels = [label for label in labels if label != "Bulk-water-S"]
    _plot_series(axes[0, 1], summaries, "mobile_ions_per_100_water", ion_labels, colors, "Mobile ions per 100 film waters")
    axes[0, 1].set_title("b  Film ion-to-water composition")
    _plot_series(axes[1, 0], summaries, "ion_free_wet_probability", ion_labels, colors, "P(ion-free film | water remains)")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_title("c  Ion-free wet-film probability")

    ax = axes[1, 1]
    partition_colors = {"film": "#2c7fb8", "A_facing_shell": "#d8a52b", "B_facing_shell": "#e1812c", "other_liquid": "#bdbdbd"}
    plot_rows = partitions[(partitions.mobile_ion_observations > 0) & partitions.case_label.ne("TiO2-water-S")].copy()
    order = [(case, window) for case in ["TiO2-NaCl-S", "TiO2-NaOH-S", "TiO2-HCl-S"] for window in ["near 2-6A", "intermediate 6-10A", "wide 10-14A"]]
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for name in PARTITIONS:
        values = []
        for case, window in order:
            q = plot_rows[(plot_rows.case_label == case) & (plot_rows.window == window) & (plot_rows.partition == name)]
            values.append(float(q.fraction.iloc[0]) if len(q) else 0.0)
        ax.bar(x, values, bottom=bottom, color=partition_colors[name], edgecolor="white", linewidth=0.4, label=name.replace("_", " "))
        bottom += np.asarray(values)
    short = {"TiO2-NaCl-S": "NaCl", "TiO2-NaOH-S": "NaOH", "TiO2-HCl-S": "HCl"}
    window_short = {"near 2-6A": "2–6", "intermediate 6-10A": "6–10", "wide 10-14A": "10–14"}
    ax.set_xticks(x, [f"{short[case]}\n{window_short[window]}" for case, window in order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of classified mobile-ion observations")
    ax.set_title("d  Mobile-ion spatial partition")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#d9d9d9", lw=0.7)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Coupled confined-film water and mobile-ion organization\nS systems; 20 ps joint block bootstrap; open markers have <4 effective blocks", fontsize=14, y=0.985)
    png = figure_dir / "candidate_fig01_coupled_water_ion_expulsion.png"
    pdf = figure_dir / "candidate_fig01_coupled_water_ion_expulsion.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    support = summaries[["case_label", "gap_bin", "frame_count", "effective_block_count"]].drop_duplicates()
    support.to_csv(plot_data / "frame_block_support.csv", index=False)
    source_manifest = study_root / "manifests" / "source_manifest.csv"
    _write_csv(source_manifest, _source_rows(cases))
    validation = study_root / "reports" / "VALIDATION.md"
    validation.parent.mkdir(parents=True, exist_ok=True)
    low = support[support.effective_block_count < 4]
    validation.write_text(
        "# Figure 1 validation\n\n"
        f"- Cases: {', '.join(labels)}\n"
        f"- Canonical frames: {len(frames)}\n"
        f"- Gap-summary rows: {len(summaries)}\n"
        f"- Low-support case/bins (<4 blocks): {len(low)}\n"
        "- Bulk ion quantities are N/A, not zero.\n"
        "- Mobile-ion partitions use classified formal species and are spatial observation fractions, not atom-conserved reaction fluxes.\n"
        "- Gap bins are structural conditional ensembles, not a kinetic time sequence.\n"
        "- Figure status: candidate for review; manuscript untouched.\n",
        encoding="utf-8",
    )
    manifest = study_root / "manifests" / "artifact_manifest.csv"
    artifacts = [png, pdf, validation, *plot_data.glob("*.csv")]
    _write_csv(manifest, _artifact_rows(artifacts))
    print(f"FIG01_VALIDATION_OK cases={len(labels)} frames={len(frames)} figure={png}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    analyze = sub.add_parser("analyze-case")
    analyze.add_argument("--case-manifest", required=True)
    analyze.add_argument("--case-label", required=True)
    analyze.add_argument("--output-root", required=True)
    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--case-manifest", required=True)
    assemble_parser.add_argument("--output-root", required=True)
    assemble_parser.add_argument("--figure-dir", required=True)
    for target in (analyze, assemble_parser):
        target.add_argument("--block-ns", type=float, default=0.020)
        target.add_argument("--bootstrap-samples", type=int, default=1000)
        target.add_argument("--random-seed", type=int, default=20260822)
    analyze.add_argument("--gap-min-A", type=float, default=0.0)
    analyze.add_argument("--gap-max-A", type=float, default=18.0)
    analyze.add_argument("--gap-bin-width-A", type=float, default=2.0)
    analyze.add_argument("--rho-core-A", type=float, default=6.0)
    analyze.add_argument("--shell-width-A", type=float, default=4.0)
    analyze.add_argument("--wide-left-A", type=float, default=10.0)
    analyze.add_argument("--wide-right-A", type=float, default=14.0)
    analyze.add_argument("--mobile-species", nargs="+", default=MOBILE_SPECIES)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return analyze_case(args) if args.operation == "analyze-case" else assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
