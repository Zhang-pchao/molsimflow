"""Assemble a manuscript-scale three-dimensional dual-interface ion-access figure.

The upstream water and ion analyses use every atom in a bubble-comoving
cylindrical frame.  This module combines their volume-normalized ``(s, rho)``
maps with block-bootstrap confined-film summaries; it does not treat gap bins
as a kinetic trajectory or formal species charges as an exact electric field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


MOBILE_CHARGE = {
    "H3O_plus": 1,
    "Na_plus": 1,
    "OH_minus_bulk": -1,
    "Cl_minus": -1,
}

CASE_DISPLAY = {
    "Bulk-water-S": "Bulk water",
    "TiO2-water-S": r"TiO$_2$–water",
    "TiO2-NaCl-S": r"TiO$_2$–NaCl",
    "TiO2-NaOH-S": r"TiO$_2$–NaOH",
    "TiO2-HCl-S": r"TiO$_2$–HCl",
}


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-ion-access requires pandas") from exc
    return pd


def _read_cases(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["case_label"] for row in rows}) != len(rows):
        raise ValueError("Case manifest must contain unique case_label rows")
    return rows


def _gap_bounds(label: str) -> tuple[float, float]:
    value = str(label).strip().removesuffix("A")
    left, right = value.split("-", 1)
    return float(left), float(right)


def _inside_window(labels, left_A: float, right_A: float):
    return np.asarray(
        [left >= left_A and right <= right_A for left, right in map(_gap_bounds, labels)],
        dtype=bool,
    )


def aggregate_water_map(path: Path, case_label: str, left_A: float, right_A: float):
    """Combine adjacent gap-bin cylindrical water maps using frame-volume exposure."""

    pd = _pandas()
    data = pd.read_csv(path)
    data = data[_inside_window(data.gap_window, left_A, right_A)].copy()
    if data.empty:
        raise ValueError(f"No water map rows for {case_label} in {left_A:g}-{right_A:g} A")
    data["exposure_A3_frame"] = data.n_frames.astype(float) * data.bin_volume_A3.astype(float)
    grouped = (
        data.groupby(
            ["s_left_A", "s_right_A", "s_center_A", "rho_left_A", "rho_right_A", "rho_center_A"],
            as_index=False,
        )
        .agg(count=("count", "sum"), exposure_A3_frame=("exposure_A3_frame", "sum"))
    )
    grouped["water_number_density_A-3"] = grouped["count"] / grouped["exposure_A3_frame"]
    grouped.insert(0, "case_label", case_label)
    grouped.insert(1, "gap_window", f"{left_A:g}-{right_A:g}A")
    return grouped


def aggregate_ion_map(
    sample_path: Path,
    frame_path: Path,
    case_label: str,
    left_A: float,
    right_A: float,
    radius_A: float = 19.0,
):
    """Bin mobile ions outside both nominal bubbles into cation/anion densities."""

    pd = _pandas()
    frames = pd.read_csv(frame_path)
    frames = frames[(frames.gap_A >= left_A) & (frames.gap_A < right_A)]
    if frames.empty:
        raise ValueError(f"No ion frames for {case_label} in {left_A:g}-{right_A:g} A")
    data = pd.read_csv(sample_path)
    data = data[
        (data.gap_A >= left_A)
        & (data.gap_A < right_A)
        & data.species.isin(MOBILE_CHARGE)
        & (data.r_A_A >= radius_A)
        & (data.r_B_A >= radius_A)
    ].copy()
    data["charge_group"] = data.species.map({name: "cation" if charge > 0 else "anion" for name, charge in MOBILE_CHARGE.items()})
    s_edges = np.arange(-24.0, 24.0 + 1.0, 2.0)
    rho_edges = np.arange(0.0, 18.0 + 1.0, 2.0)
    rows = []
    for group in ("cation", "anion"):
        q = data[data.charge_group == group]
        counts = np.histogram2d(q.s_A, q.rho_A, bins=(s_edges, rho_edges))[0]
        for i in range(len(s_edges) - 1):
            for j in range(len(rho_edges) - 1):
                volume = (s_edges[i + 1] - s_edges[i]) * math.pi * (rho_edges[j + 1] ** 2 - rho_edges[j] ** 2)
                exposure = len(frames) * volume
                rows.append(
                    {
                        "charge_group": group,
                        "s_left_A": s_edges[i],
                        "s_right_A": s_edges[i + 1],
                        "s_center_A": 0.5 * (s_edges[i] + s_edges[i + 1]),
                        "rho_left_A": rho_edges[j],
                        "rho_right_A": rho_edges[j + 1],
                        "rho_center_A": 0.5 * (rho_edges[j] + rho_edges[j + 1]),
                        "count": counts[i, j],
                        "exposure_A3_frame": exposure,
                        "number_density_A-3": counts[i, j] / exposure,
                    }
                )
    grouped = pd.DataFrame(rows)
    grouped.insert(0, "case_label", case_label)
    grouped.insert(1, "gap_window", f"{left_A:g}-{right_A:g}A")
    return grouped


def _window_metrics(data, weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    water = data["water_density_A-3"].to_numpy(float)
    water_density = float(np.average(water, weights=weights)) if weights.sum() else math.nan
    n_water = data["n_water_film"].to_numpy(float)
    n_ion = data["n_mobile_ions_film"].to_numpy(float)
    valid_ion = np.isfinite(n_ion)
    wet = (n_water > 0) & valid_ion
    wet_weight = float(weights[wet].sum())
    access = float(np.sum(weights[wet] * (n_ion[wet] > 0)) / wet_weight) if wet_weight else math.nan
    water_count = float(np.sum(weights[valid_ion] * n_water[valid_ion]))
    ion_count = float(np.sum(weights[valid_ion] * n_ion[valid_ion]))
    per_100 = 100.0 * ion_count / water_count if water_count else math.nan
    return {
        "water_density_A-3": water_density,
        "mobile_ion_access_probability": access,
        "mobile_ions_per_100_water": per_100,
    }


def summarize_windows(
    frame,
    case_label: str,
    near: tuple[float, float],
    wide: tuple[float, float],
    block_ns: float,
    bootstrap_samples: int,
    random_seed: int,
):
    """Jointly bootstrap near and wide structural ensembles by continuous-time blocks."""

    pd = _pandas()
    data = frame.copy()
    data["block_id"] = np.floor(data.time_ns.to_numpy(float) / block_ns + 1.0e-10).astype(int)
    windows = {"near": near, "wide": wide}
    masks = {
        name: (data.gap_A.to_numpy(float) >= bounds[0]) & (data.gap_A.to_numpy(float) < bounds[1])
        for name, bounds in windows.items()
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError(f"Both near and wide windows require frames for {case_label}")
    point = {name: _window_metrics(data[mask], np.ones(int(mask.sum()))) for name, mask in masks.items()}
    ratio = point["near"]["water_density_A-3"] / point["wide"]["water_density_A-3"]
    draws: dict[tuple[str, str], list[float]] = {
        (window, metric): [] for window in windows for metric in point[window]
    }
    ratio_draws: list[float] = []
    blocks = np.sort(data.block_id.unique())
    rng = np.random.default_rng(random_seed)
    for _ in range(bootstrap_samples):
        multiplicity = Counter(rng.choice(blocks, size=len(blocks), replace=True).tolist())
        weights = data.block_id.map(multiplicity).to_numpy(float)
        sampled = {}
        for name, mask in masks.items():
            selected = mask & (weights > 0)
            if not selected.any():
                sampled = {}
                break
            sampled[name] = _window_metrics(data[selected], weights[selected])
        if not sampled:
            continue
        for name in windows:
            for metric, value in sampled[name].items():
                if math.isfinite(value):
                    draws[(name, metric)].append(value)
        near_density = sampled["near"]["water_density_A-3"]
        wide_density = sampled["wide"]["water_density_A-3"]
        if math.isfinite(near_density) and wide_density > 0:
            ratio_draws.append(near_density / wide_density)

    rows = []
    for name, (left_A, right_A) in windows.items():
        mask = masks[name]
        for metric, value in point[name].items():
            values = np.asarray(draws[(name, metric)], dtype=float)
            rows.append(
                {
                    "case_label": case_label,
                    "window": name,
                    "window_left_A": left_A,
                    "window_right_A": right_A,
                    "frame_count": int(mask.sum()),
                    "effective_block_count": int(data.loc[mask, "block_id"].nunique()),
                    "metric": metric,
                    "mean": value,
                    "ci95_low": float(np.quantile(values, 0.025)) if len(values) >= 20 else math.nan,
                    "ci95_high": float(np.quantile(values, 0.975)) if len(values) >= 20 else math.nan,
                    "bootstrap_draw_count": len(values),
                }
            )
    ratio_values = np.asarray(ratio_draws, dtype=float)
    rows.append(
        {
            "case_label": case_label,
            "window": "near/wide",
            "window_left_A": near[0],
            "window_right_A": near[1],
            "frame_count": int(masks["near"].sum() + masks["wide"].sum()),
            "effective_block_count": int(data.loc[masks["near"] | masks["wide"], "block_id"].nunique()),
            "metric": "water_density_near_wide_ratio",
            "mean": ratio,
            "ci95_low": float(np.quantile(ratio_values, 0.025)) if len(ratio_values) >= 20 else math.nan,
            "ci95_high": float(np.quantile(ratio_values, 0.975)) if len(ratio_values) >= 20 else math.nan,
            "bootstrap_draw_count": len(ratio_values),
        }
    )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _artifact_manifest(paths: Sequence[Path]):
    pd = _pandas()
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _source_manifest(cases: Sequence[Mapping[str, str]], ion_cases: Sequence[Mapping[str, str]]):
    pd = _pandas()
    rows = []
    for case in cases:
        sources = [("water_trace", case["trace_csv"])]
        for item in case["segment_specs"].split(";"):
            label, path = item.split("=", 1)
            sources.append((f"trajectory:{label}", path))
        for role, value in sources:
            path = Path(value)
            rows.append({"case_label": case["case_label"], "role": role, "path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    for case in ion_cases:
        for role in ("frame_csv", "position_csv"):
            path = Path(case[role])
            rows.append({"case_label": case["case_label"], "role": f"classified_ion_{role}", "path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    return pd.DataFrame(rows)


def _draw_bubbles(ax, gap_A: float, radius_A: float = 19.0) -> None:
    theta = np.linspace(0.0, math.pi, 300)
    distance = 2.0 * radius_A + gap_A
    for center in (-0.5 * distance, 0.5 * distance):
        ax.plot(center + radius_A * np.cos(theta), radius_A * np.sin(theta), color="#262626", lw=0.8)


def _errorbar(ax, x: float, low: float, high: float, y: float, **kwargs) -> None:
    if math.isfinite(low) and math.isfinite(high):
        kwargs["xerr"] = [[max(0.0, x - low)], [max(0.0, high - x)]]
    ax.errorbar(x, y, **kwargs)


def assemble(args) -> int:
    pd = _pandas()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = _read_cases(Path(args.case_manifest))
    ion_cases = _read_cases(Path(args.ion_case_manifest))
    labels = [row["case_label"] for row in cases]
    map_labels = args.map_case_label or ["TiO2-NaOH-S", "TiO2-NaCl-S", "TiO2-HCl-S"]
    if not set(map_labels).issubset({row["case_label"] for row in ion_cases}):
        raise ValueError("Every map case must be present in the ion manifest")

    water_maps = []
    ion_maps = []
    for label in map_labels:
        water_maps.append(
            aggregate_water_map(Path(args.water_root) / label / "water_orientation_sr_map.csv", label, args.near_left_A, args.near_right_A)
        )
        ion_maps.append(
            aggregate_ion_map(
                Path(args.ion_root) / label / "ion3d_samples.csv.gz",
                Path(args.ion_root) / label / "ion3d_frame_summary.csv",
                label,
                args.near_left_A,
                args.near_right_A,
            )
        )
    water_map = pd.concat(water_maps, ignore_index=True)
    ion_map = pd.concat(ion_maps, ignore_index=True)

    summaries = []
    frames = {}
    for index, label in enumerate(labels):
        frame = pd.read_csv(Path(args.coupled_root) / label / "coupled_frame_summary.csv")
        frames[label] = frame
        summaries.append(
            summarize_windows(
                frame,
                label,
                (args.near_left_A, args.near_right_A),
                (args.wide_left_A, args.wide_right_A),
                args.block_ns,
                args.bootstrap_samples,
                args.random_seed + index,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)

    figure_dir = Path(args.figure_dir)
    plot_data = figure_dir / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    water_csv = plot_data / "figure6_water_sr_density.csv"
    ion_csv = plot_data / "figure6_mobile_ion_sr_density.csv"
    summary_csv = plot_data / "figure6_window_summary.csv"
    _write_csv(water_csv, water_map)
    _write_csv(ion_csv, ion_map)
    _write_csv(summary_csv, summary)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    fig = plt.figure(figsize=(10.8, 7.4))
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=(1.0, 0.08, 0.82),
        left=0.07,
        right=0.98,
        bottom=0.09,
        top=0.82,
        hspace=0.42,
        wspace=0.55,
    )
    map_axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6])]
    colorbar_axis = fig.add_subplot(grid[1, 0:6])
    ax_water = fig.add_subplot(grid[2, 0:3])
    ax_access = fig.add_subplot(grid[2, 3:6])
    panel_letters = "abc"
    ion_nonzero = ion_map.loc[ion_map["number_density_A-3"] > 0, "number_density_A-3"].to_numpy(float)
    ion_cap = float(np.quantile(ion_nonzero, 0.98)) if len(ion_nonzero) else 1.0
    image = None
    for panel, ax, label in zip(panel_letters, map_axes, map_labels):
        water = water_map[water_map.case_label == label]
        s_values = np.sort(water.s_center_A.unique())
        rho_values = np.sort(water.rho_center_A.unique())
        density = water.pivot(index="rho_center_A", columns="s_center_A", values="water_number_density_A-3").reindex(index=rho_values, columns=s_values).to_numpy(float)
        image = ax.pcolormesh(s_values, rho_values, density, shading="nearest", cmap="Blues", vmin=0.0, vmax=0.040, rasterized=True)
        ions = ion_map[(ion_map.case_label == label) & (ion_map["number_density_A-3"] > 0)]
        for group, marker, color in (("cation", "o", "#D28E00"), ("anion", "s", "#B23A70")):
            q = ions[ions.charge_group == group]
            size = 5.0 + 145.0 * np.clip(q["number_density_A-3"].to_numpy(float) / ion_cap, 0.0, 1.0)
            ax.scatter(q.s_center_A, q.rho_center_A, s=size, marker=marker, facecolors="none", edgecolors=color, linewidths=0.75, alpha=0.80)
        near_frame = frames[label][(frames[label].gap_A >= args.near_left_A) & (frames[label].gap_A < args.near_right_A)]
        _draw_bubbles(ax, float(near_frame.gap_A.mean()))
        ax.axhline(6.0, color="#4D4D4D", lw=0.7, ls="--")
        ax.set_xlim(-24.0, 24.0)
        ax.set_ylim(0.0, 18.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Bubble-axis coordinate $s$ (Å)")
        if ax is map_axes[0]:
            ax.set_ylabel(r"Radial coordinate $\rho$ (Å)")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{panel}  {CASE_DISPLAY[label]}\n2 ≤ h < 6 Å; n = {len(near_frame)} frames", loc="left")
    if image is not None:
        colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label(r"Water O number density (Å$^{-3}$)")
    ion_legend = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#D28E00", markerfacecolor="none", label="mobile cation", markersize=7),
        Line2D([0], [0], marker="s", color="none", markeredgecolor="#B23A70", markerfacecolor="none", label="mobile anion", markersize=7),
        Line2D([0], [0], color="#262626", lw=0.8, label="nominal bubble surface"),
        Line2D([0], [0], color="#4D4D4D", lw=0.7, ls="--", label=r"strict core $\rho\leq6$ Å"),
    ]
    fig.legend(handles=ion_legend, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.915))

    order = labels
    y = np.arange(len(order))
    ratio_rows = summary[(summary.metric == "water_density_near_wide_ratio") & (summary.window == "near/wide")].set_index("case_label")
    for yi, label in enumerate(order):
        row = ratio_rows.loc[label]
        _errorbar(
            ax_water,
            float(row["mean"]),
            float(row["ci95_low"]),
            float(row["ci95_high"]),
            yi,
            fmt="o",
            ms=5.5,
            color="#2C7FB8",
            ecolor="#2C7FB8",
            elinewidth=1.0,
            capsize=2.5,
        )
    ax_water.axvline(1.0, color="#333333", lw=0.8, ls="--")
    ax_water.set_yticks(y, [CASE_DISPLAY[label] for label in order])
    ax_water.set_ylim(len(order) - 0.5, -0.5)
    finite_ratio = ratio_rows[["ci95_low", "ci95_high"]].to_numpy(float)
    low = max(0.0, float(np.nanmin(finite_ratio)) - 0.08)
    high = float(np.nanmax(finite_ratio)) + 0.08
    ax_water.set_xlim(low, high)
    ax_water.set_xlabel(r"$\rho_{\mathrm{water}}(2$–$6\ \mathrm{Å})/\rho_{\mathrm{water}}(10$–$14\ \mathrm{Å})$")
    ax_water.set_title("d  Confined-water density response", loc="left", pad=8)
    ax_water.grid(axis="x", color="#DDDDDD", lw=0.6)

    access = summary[summary.metric == "mobile_ion_access_probability"].copy()
    max_access = 0.0
    for yi, label in enumerate(order):
        rows = access[access.case_label == label].set_index("window")
        if rows["mean"].notna().sum() == 0:
            ax_access.text(0.01, yi, "N/A (no ionic species)", va="center", ha="left", fontsize=8, color="#666666")
            continue
        wide_row = rows.loc["wide"]
        near_row = rows.loc["near"]
        values = [float(wide_row["mean"]), float(near_row["mean"])]
        ax_access.plot(values, [yi - 0.10, yi + 0.10], color="#BDBDBD", lw=1.0, zorder=1)
        _errorbar(ax_access, values[0], float(wide_row["ci95_low"]), float(wide_row["ci95_high"]), yi - 0.10, fmt="s", ms=5.2, mfc="white", mec="#333333", ecolor="#777777", elinewidth=0.9, capsize=2.0, zorder=2)
        _errorbar(ax_access, values[1], float(near_row["ci95_low"]), float(near_row["ci95_high"]), yi + 0.10, fmt="o", ms=5.5, mfc="#B23A70", mec="#B23A70", ecolor="#B23A70", elinewidth=0.9, capsize=2.0, zorder=3)
        max_access = max(max_access, float(wide_row["ci95_high"]), float(near_row["ci95_high"]))
    ax_access.set_xlim(-0.01, min(1.0, max(0.12, max_access * 1.10)))
    ax_access.set_yticks(y, [CASE_DISPLAY[label] for label in order])
    ax_access.set_ylim(len(order) - 0.5, -0.5)
    ax_access.set_xlabel("P(mobile ion in strict wet film)")
    ax_access.set_title("e  Mobile-ion access: wide → compressed", loc="left", pad=8)
    ax_access.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax_access.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markeredgecolor="#333333", markerfacecolor="white", label="10–14 Å", markersize=6),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#B23A70", markerfacecolor="#B23A70", label="2–6 Å", markersize=6),
        ],
        frameon=False,
        ncol=2,
        loc="upper right",
    )

    fig.suptitle("Three-dimensional ion access to the dual-interface wet film", fontsize=14, y=0.992)
    fig.text(0.5, 0.955, "S systems; cylindrical volume statistics; 20 ps block-bootstrap 95% CI", ha="center", va="top", fontsize=9, color="#4D4D4D")
    png = figure_dir / "candidate_figure6_3d_ion_access.png"
    pdf = figure_dir / "candidate_figure6_3d_ion_access.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    expected_cases = {"Bulk-water-S", "TiO2-water-S", "TiO2-NaCl-S", "TiO2-NaOH-S", "TiO2-HCl-S"}
    if set(labels) != expected_cases:
        raise ValueError(f"Expected five S cases, found {labels}")
    if set(water_map.case_label) != set(map_labels) or set(ion_map.case_label) != set(map_labels):
        raise ValueError("Map case coverage is incomplete")
    if (water_map["water_number_density_A-3"] < 0).any() or (ion_map["number_density_A-3"] < 0).any():
        raise ValueError("Negative number density detected")
    if png.stat().st_size < 100_000 or pdf.stat().st_size < 20_000:
        raise ValueError("Figure export is unexpectedly small")

    study_root = figure_dir.parent.parent
    source_csv = study_root / "manifests" / "source_manifest.csv"
    validation = study_root / "reports" / "VALIDATION.md"
    artifact_csv = study_root / "manifests" / "artifact_manifest.csv"
    _write_csv(source_csv, _source_manifest(cases, ion_cases))
    validation.parent.mkdir(parents=True, exist_ok=True)
    frame_total = sum(len(frame) for frame in frames.values())
    low_support = summary[(summary.effective_block_count < 4) & summary.window.isin(["near", "wide"])]
    validation.write_text(
        "# Candidate Figure 6 validation\n\n"
        f"- Cases: {', '.join(labels)}\n"
        f"- Canonical frames: {frame_total}\n"
        f"- Compressed map cases: {', '.join(map_labels)}\n"
        f"- Near window: {args.near_left_A:g}-{args.near_right_A:g} A; wide window: {args.wide_left_A:g}-{args.wide_right_A:g} A\n"
        f"- Low-support near/wide rows (<4 blocks): {len(low_support)}\n"
        f"- Global ion-symbol density cap (98th percentile): {ion_cap:.8g} A^-3\n"
        "- Water and ion maps are three-dimensional cylindrical volume statistics, not center-plane slices.\n"
        "- Mobile ions are classified H3O_plus, OH_minus_bulk, Na_plus, and Cl_minus; surface markers are excluded.\n"
        "- Bulk-water ion access is N/A, not a fabricated zero.\n"
        "- Gap windows are structural conditional ensembles, not a kinetic time sequence.\n"
        "- Formal species are not dynamic polarization charges or an exact electric field.\n"
        "- Figure status: candidate for review; manuscript and formal figures untouched.\n",
        encoding="utf-8",
    )
    artifacts = [png, pdf, water_csv, ion_csv, summary_csv, source_csv, validation]
    _write_csv(artifact_csv, _artifact_manifest(artifacts))
    print(f"FIG06_3D_VALIDATION_OK cases={len(labels)} frames={frame_total} figure={png}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble a three-dimensional dual-interface ion-access candidate figure")
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--ion-case-manifest", required=True)
    parser.add_argument("--water-root", required=True)
    parser.add_argument("--ion-root", required=True)
    parser.add_argument("--coupled-root", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--map-case-label", action="append", default=[])
    parser.add_argument("--near-left-A", type=float, default=2.0)
    parser.add_argument("--near-right-A", type=float, default=6.0)
    parser.add_argument("--wide-left-A", type=float, default=10.0)
    parser.add_argument("--wide-right-A", type=float, default=14.0)
    parser.add_argument("--block-ns", type=float, default=0.020)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260823)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return assemble(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
