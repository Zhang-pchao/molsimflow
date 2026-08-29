"""Three-dimensional ion organization in a two-bubble coordinate frame.

The input is the existing classified ion-position table plus its frame-level
two-bubble geometry table.  Positions are transformed for every selected
separated frame into ``(s, rho, phi)`` coordinates.  Cylindrical densities are
volume normalized, while phi-sector and first-harmonic summaries retain a
compact measure of genuinely three-dimensional azimuthal organization.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.dual_interface_ion import (
    ALLOWED_CHARGES,
    local_basis,
    minimum_image,
    validate_species_charge,
)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("\n")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def _write_gzip_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with gzip.open(path, "wt") as handle:
            handle.write("\n")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-ion3d requires pandas") from exc
    return pd


def _bootstrap(values: np.ndarray, times: np.ndarray, block_ns: float, samples: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    blocks = np.floor(times / block_ns + 1.0e-10).astype(int)
    unique = np.unique(blocks)
    block_means = np.asarray([values[blocks == block_id].mean() for block_id in unique])
    mean = float(values.mean())
    if len(block_means) < 2 or samples <= 0:
        return mean, float("nan"), float("nan"), int(len(block_means))
    draws = rng.choice(block_means, size=(samples, len(block_means)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high), int(len(block_means))


def _gap_label(left: float, right: float) -> str:
    return f"{left:g}-{right:g}A"


def _case_rows(path: Path, case_label: str) -> Mapping[str, str]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("case_label") == case_label]
    if len(matches) != 1:
        raise ValueError(f"Expected one case_label={case_label!r}, found {len(matches)}")
    return matches[0]


def _load_frames(case: Mapping[str, str], config) -> Tuple[object, Dict[int, Mapping[str, object]]]:
    pd = _pandas()
    frame = pd.read_csv(case["frame_csv"])
    required = {"global_frame", "time_ns", "d3d_all", "coalescence_state", "bubble_A_center_x_A", "bubble_A_center_y_A", "bubble_A_center_z_A", "bubble_B_center_x_A", "bubble_B_center_y_A", "bubble_B_center_z_A"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing frame columns: {sorted(missing)}")
    frame = frame[frame["coalescence_state"].astype(str).eq(config.state)].copy()
    frame["gap_A"] = pd.to_numeric(frame["d3d_all"], errors="coerce") - float(case.get("nominal_radius_a_A") or 19.0) - float(case.get("nominal_radius_b_A") or 19.0)
    frame = frame[(frame.gap_A >= config.gap_min_A) & (frame.gap_A < config.gap_max_A)].copy()
    frame["case_label"] = case["case_label"]
    frame["gap_left_A"] = np.floor((frame.gap_A - config.gap_min_A) / config.gap_bin_width_A) * config.gap_bin_width_A + config.gap_min_A
    frame["gap_right_A"] = frame.gap_left_A + config.gap_bin_width_A
    frame["gap_bin"] = [_gap_label(float(a), float(b)) for a, b in zip(frame.gap_left_A, frame.gap_right_A)]
    frame = frame.sort_values("time_ns")
    if frame.empty:
        raise ValueError(f"No separated frames in {config.gap_min_A}:{config.gap_max_A} A")
    lookup = {int(row.global_frame): row._asdict() for row in frame.itertuples(index=False)}
    return frame, lookup


def _read_samples(case: Mapping[str, str], lookup: Mapping[int, Mapping[str, object]], config) -> List[Dict[str, object]]:
    pd = _pandas()
    selected = set(lookup)
    usecols = ["global_frame", "atom_id", "species_canonical", "species_charge_e", "x_A", "y_A", "z_A", "in_bridge", "in_trace_region", "in_bridge_region", "in_slab"]
    lengths = np.asarray(config.box_lengths_A, dtype=float)
    radius_a = float(case.get("nominal_radius_a_A") or 19.0)
    radius_b = float(case.get("nominal_radius_b_A") or 19.0)
    output: List[Dict[str, object]] = []
    for chunk in pd.read_csv(case["position_csv"], usecols=usecols, chunksize=200000):
        charge = pd.to_numeric(chunk["species_charge_e"], errors="coerce").fillna(0.0)
        chunk = chunk[chunk["global_frame"].isin(selected) & charge.ne(0.0)]
        for row in chunk.itertuples(index=False):
            frame = lookup[int(row.global_frame)]
            q = validate_species_charge(row.species_canonical, row.species_charge_e)
            center_a = np.asarray([frame["bubble_A_center_x_A"], frame["bubble_A_center_y_A"], frame["bubble_A_center_z_A"]], dtype=float)
            center_b = np.asarray([frame["bubble_B_center_x_A"], frame["bubble_B_center_y_A"], frame["bubble_B_center_z_A"]], dtype=float)
            midpoint, e_s, e_u, e_z = local_basis(center_a, center_b, lengths)
            position = np.asarray([row.x_A, row.y_A, row.z_A], dtype=float)
            disp = minimum_image(position - midpoint, lengths)
            s = float(disp @ e_s)
            u = float(disp @ e_u)
            z = float(disp @ e_z)
            rho = float(math.hypot(u, z))
            d = float(frame["d3d_all"])
            r_a = math.hypot(s + 0.5 * d, rho)
            r_b = math.hypot(s - 0.5 * d, rho)
            core = int(r_a >= radius_a and r_b >= radius_b and rho <= config.core_rho_max_A)
            shell_a = int(r_a >= radius_a and r_a - radius_a <= config.surface_shell_width_A and rho <= config.shell_rho_max_A)
            shell_b = int(r_b >= radius_b and r_b - radius_b <= config.surface_shell_width_A and rho <= config.shell_rho_max_A)
            output.append({
                "case_label": case["case_label"], "global_frame": int(row.global_frame), "time_ns": float(frame["time_ns"]),
                "gap_A": float(frame["gap_A"]), "gap_bin": str(frame["gap_bin"]), "atom_id": int(row.atom_id),
                "species": str(row.species_canonical), "formal_charge_e": q, "s_A": s, "u_A": u, "z_A": z,
                "rho_A": rho, "phi_rad": float(math.atan2(z, u)), "r_A_A": r_a, "r_B_A": r_b,
                "surface_delta_A_A": r_a - radius_a, "surface_delta_B_A": r_b - radius_b,
                "in_dual_core": core, "in_A_surface_shell": shell_a, "in_B_surface_shell": shell_b,
                "in_bridge": int(row.in_bridge), "in_trace_region": int(row.in_trace_region),
                "in_bridge_region": int(row.in_bridge_region), "in_slab": int(row.in_slab),
            })
    return output


def _frame_summary(frame, samples: Sequence[Mapping[str, object]], config) -> List[Dict[str, object]]:
    by_frame: Dict[int, List[Mapping[str, object]]] = {int(row.global_frame): [] for row in frame.itertuples(index=False)}
    for row in samples:
        by_frame[int(row["global_frame"])].append(row)
    rows: List[Dict[str, object]] = []
    for meta in frame.itertuples(index=False):
        values = by_frame[int(meta.global_frame)]
        core = [row for row in values if int(row["in_dual_core"])]
        q = np.asarray([float(row["formal_charge_e"]) for row in values], dtype=float)
        cq = np.asarray([float(row["formal_charge_e"]) for row in core], dtype=float)
        cphi = np.asarray([float(row["phi_rad"]) for row in core], dtype=float)
        qden = float(np.abs(cq).sum())
        rows.append({
            "case_label": str(meta.case_label), "global_frame": int(meta.global_frame), "time_ns": float(meta.time_ns),
            "gap_A": float(meta.gap_A), "gap_bin": str(meta.gap_bin), "n_ions": len(values),
            "net_charge_e": float(q.sum()), "abs_charge_e": float(np.abs(q).sum()), "core_n_ions": len(core),
            "core_net_charge_e": float(cq.sum()), "core_abs_charge_e": qden,
            "core_phi_m1_count": float(abs(np.exp(1j * cphi).sum()) / len(core)) if core else 0.0,
            "core_phi_m1_charge": float(abs((cq * np.exp(1j * cphi)).sum()) / qden) if qden else 0.0,
            "A_shell_n_ions": sum(int(row["in_A_surface_shell"]) for row in values),
            "B_shell_n_ions": sum(int(row["in_B_surface_shell"]) for row in values),
        })
    species = sorted({str(row["species"]) for row in samples})
    for row in rows:
        values = by_frame[int(row["global_frame"])]
        for name in species:
            row[f"n_{name}"] = sum(str(item["species"]) == name for item in values)
            row[f"core_n_{name}"] = sum(str(item["species"]) == name and int(item["in_dual_core"]) for item in values)
    return rows


def _cylindrical_density(samples: Sequence[Mapping[str, object]], frame, config) -> List[Dict[str, object]]:
    s_edges = np.arange(config.s_min_A, config.s_max_A + config.s_bin_width_A * 0.5, config.s_bin_width_A)
    rho_edges = np.arange(0.0, config.rho_max_A + config.rho_bin_width_A * 0.5, config.rho_bin_width_A)
    species = sorted({str(row["species"]) for row in samples})
    output: List[Dict[str, object]] = []
    for gap_bin, frame_chunk in frame.groupby("gap_bin", sort=True):
        n_frames = len(frame_chunk)
        for name in species:
            rows = [row for row in samples if row["gap_bin"] == gap_bin and row["species"] == name]
            s = np.asarray([float(row["s_A"]) for row in rows], dtype=float)
            rho = np.asarray([float(row["rho_A"]) for row in rows], dtype=float)
            q = np.asarray([float(row["formal_charge_e"]) for row in rows], dtype=float)
            counts = np.histogram2d(s, rho, bins=(s_edges, rho_edges))[0]
            charges = np.histogram2d(s, rho, bins=(s_edges, rho_edges), weights=q)[0]
            abs_charges = np.histogram2d(s, rho, bins=(s_edges, rho_edges), weights=np.abs(q))[0]
            for i in range(len(s_edges) - 1):
                for j in range(len(rho_edges) - 1):
                    volume = (s_edges[i + 1] - s_edges[i]) * math.pi * (rho_edges[j + 1] ** 2 - rho_edges[j] ** 2)
                    output.append({
                        "case_label": str(frame_chunk.case_label.iloc[0]), "gap_bin": gap_bin, "species": name,
                        "s_left_A": s_edges[i], "s_right_A": s_edges[i + 1], "s_center_A": 0.5 * (s_edges[i] + s_edges[i + 1]),
                        "rho_left_A": rho_edges[j], "rho_right_A": rho_edges[j + 1], "rho_center_A": 0.5 * (rho_edges[j] + rho_edges[j + 1]),
                        "n_frames": n_frames, "volume_A3": volume,
                        "number_density_A-3": float(counts[i, j] / (n_frames * volume)),
                        "charge_density_e_A-3": float(charges[i, j] / (n_frames * volume)),
                        "abs_charge_density_e_A-3": float(abs_charges[i, j] / (n_frames * volume)),
                    })
    return output


def _phi_summary(samples: Sequence[Mapping[str, object]], frame, config) -> List[Dict[str, object]]:
    output = []
    edges = np.linspace(-math.pi, math.pi, config.phi_bins + 1)
    for (gap_bin, species), group in _group_samples(samples, ("gap_bin", "species")):
        phi = np.asarray([float(row["phi_rad"]) for row in group])
        count = np.histogram(phi, bins=edges)[0]
        charge = np.histogram(phi, bins=edges, weights=[float(row["formal_charge_e"]) for row in group])[0]
        n_frames = int((frame.gap_bin == gap_bin).sum())
        for index in range(config.phi_bins):
            output.append({
                "case_label": str(frame.case_label.iloc[0]), "gap_bin": gap_bin, "species": species,
                "phi_left_rad": edges[index], "phi_right_rad": edges[index + 1], "phi_center_rad": 0.5 * (edges[index] + edges[index + 1]),
                "n_frames": n_frames, "count_per_frame": float(count[index] / n_frames), "charge_e_per_frame": float(charge[index] / n_frames),
            })
    return output


def _group_samples(samples: Sequence[Mapping[str, object]], keys: Sequence[str]):
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in samples:
        key = tuple(row[key_name] for key_name in keys)
        groups.setdefault(key, []).append(row)
    return groups.items()


def _gap_summary(frame_rows: Sequence[Mapping[str, object]], config, rng) -> List[Dict[str, object]]:
    pd = _pandas()
    data = pd.DataFrame(frame_rows)
    metrics = ["n_ions", "net_charge_e", "abs_charge_e", "core_n_ions", "core_net_charge_e", "core_abs_charge_e", "core_phi_m1_count", "core_phi_m1_charge", "A_shell_n_ions", "B_shell_n_ions"]
    metrics.extend(column for column in data.columns if column.startswith("core_n_") or column.startswith("n_"))
    metrics = list(dict.fromkeys(metrics))
    output = []
    for gap_bin, group in data.groupby("gap_bin", sort=True):
        for metric in metrics:
            mean, low, high, blocks = _bootstrap(group[metric].to_numpy(float), group.time_ns.to_numpy(float), config.block_ns, config.bootstrap_samples, rng)
            output.append({
                "case_label": str(group.case_label.iloc[0]), "gap_bin": gap_bin, "frame_count": len(group), "effective_block_count": blocks,
                "metric": metric, "mean": mean, "ci95_low": low, "ci95_high": high,
            })
    return output


def _manifest_rows(paths: Sequence[Path]) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest})
    return rows


def analyze_case(args) -> int:
    case = _case_rows(Path(args.case_manifest), args.case_label)
    config = args
    frame, lookup = _load_frames(case, config)
    samples = _read_samples(case, lookup, config)
    frame_rows = _frame_summary(frame, samples, config)
    output = Path(args.output_root) / case["case_label"]
    output.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(output / "ion3d_samples.csv.gz", samples)
    _write_csv(output / "ion3d_frame_summary.csv", frame_rows)
    _write_csv(output / "ion3d_gap_summary.csv", _gap_summary(frame_rows, config, np.random.default_rng(config.random_seed)))
    _write_csv(output / "ion3d_cylindrical_density.csv", _cylindrical_density(samples, frame, config))
    _write_csv(output / "ion3d_phi_sector.csv", _phi_summary(samples, frame, config))
    statistics = [
        {"metric": "case_label", "value": case["case_label"]},
        {"metric": "selected_separated_frames", "value": len(frame)},
        {"metric": "transformed_charged_rows", "value": len(samples)},
        {"metric": "gap_range_A", "value": f"{config.gap_min_A:g}-{config.gap_max_A:g}"},
        {"metric": "gap_bin_width_A", "value": config.gap_bin_width_A},
        {"metric": "coordinate_frame", "value": "s=bubble axis; rho=sqrt(u^2+z^2); phi=atan2(z,u)"},
        {"metric": "volume_normalization", "value": "pi*(rho_hi^2-rho_lo^2)*ds per frame"},
        {"metric": "core_definition", "value": f"outside both nominal spheres and rho <= {config.core_rho_max_A:g} A"},
        {"metric": "interpretation", "value": "classified formal-charge distribution; not DP partial charge or exact electric field"},
    ]
    _write_csv(output / "state_statistics.csv", statistics)
    artifacts = [p for p in output.iterdir() if p.is_file()]
    _write_csv(output / "artifact_manifest.csv", _manifest_rows(artifacts))
    print(f"case={case['case_label']} frames={len(frame)} samples={len(samples)} output={output}")
    return 0


def _read_csv(path: Path):
    return _pandas().read_csv(path)


def _case_colors(labels):
    palette = ["#2c7fb8", "#41ab5d", "#d95f02", "#8c6bb1", "#c51b8a"]
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}


def assemble(args) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    root = Path(args.output_root)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.case_manifest).open(newline="") as handle:
        cases = [row for row in csv.DictReader(handle)]
    cases = [row for row in cases if (root / row["case_label"] / "ion3d_cylindrical_density.csv").exists()]
    labels = [row["case_label"] for row in cases]
    colors = _case_colors(labels)
    plot_data = figure_dir / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    density = {row["case_label"]: _read_csv(root / row["case_label"] / "ion3d_cylindrical_density.csv") for row in cases}
    summaries = {row["case_label"]: _read_csv(root / row["case_label"] / "ion3d_gap_summary.csv") for row in cases}
    frames = {row["case_label"]: _read_csv(root / row["case_label"] / "ion3d_frame_summary.csv") for row in cases}
    _write_csv(plot_data / "multicase_ion3d_cylindrical_density.csv", [record for value in density.values() for record in value.to_dict("records")])
    _write_csv(plot_data / "multicase_ion3d_gap_summary.csv", [record for value in summaries.values() for record in value.to_dict("records")])
    _write_csv(plot_data / "multicase_ion3d_frame_summary.csv", [record for value in frames.values() for record in value.to_dict("records")])

    windows = ["4-6A", "12-14A"]
    fig, axes = plt.subplots(len(labels), 2, figsize=(12, max(7.0, 2.8 * len(labels))), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    all_values = np.concatenate([d["charge_density_e_A-3"].to_numpy(float) for d in density.values()])
    vmax = max(abs(float(np.nanpercentile(all_values, 99))), abs(float(np.nanpercentile(all_values, 1))), 1.0e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None
    for i, label in enumerate(labels):
        d = density[label]
        for j, window in enumerate(windows):
            ax = axes[i, j]
            chunk = d[d.gap_bin == window]
            if chunk.empty:
                ax.text(0.5, 0.5, "no separated frames", ha="center", va="center", transform=ax.transAxes)
                continue
            pivot = chunk.pivot_table(index="rho_center_A", columns="s_center_A", values="charge_density_e_A-3", aggfunc="sum").sort_index()
            image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()], cmap="RdBu_r", norm=norm)
            ax.set_title(f"{label}\n{window}, n={int(chunk.n_frames.iloc[0])} frames")
            if i == len(labels) - 1:
                ax.set_xlabel("s (Å)")
            if j == 0:
                ax.set_ylabel("rho (Å)")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="formal-charge density (e Å⁻³)", shrink=0.8)
    fig.suptitle("Three-dimensional cylindrical formal-charge density", fontsize=15)
    fig.savefig(figure_dir / "candidate_ion3d_cylindrical_density.png", dpi=220)
    plt.close(fig)

    metric_names = ["core_n_ions", "core_net_charge_e", "core_phi_m1_count", "A_shell_n_ions", "B_shell_n_ions"]
    titles = ["Dual-core ion occupancy", "Dual-core net formal charge", "Core azimuthal m=1 anisotropy", "A-side surface-shell occupancy", "B-side surface-shell occupancy"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, metric, title in zip(axes, metric_names[:3], titles[:3]):
        for label in labels:
            q = summaries[label]
            q = q[q.metric == metric].copy()
            if q.empty:
                continue
            q["gap_center_A"] = q.gap_bin.str.extract(r"([0-9.]+)-([0-9.]+)")[0].astype(float) + args.gap_bin_width_A / 2.0
            q = q.sort_values("gap_center_A")
            x = q["gap_center_A"].to_numpy(float)
            ax.plot(x, q["mean"], marker="o", label=label, color=colors[label])
            if q.ci95_low.notna().any():
                ax.fill_between(x, q.ci95_low, q.ci95_high, color=colors[label], alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("nominal gap h (Å)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean per frame")
    axes[1].axhline(0.0, color="#444444", lw=0.8)
    axes[2].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Three-dimensional ion organization during approach", fontsize=15)
    fig.savefig(figure_dir / "candidate_ion3d_approach_evolution.png", dpi=220)
    plt.close(fig)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:  # pragma: no cover
        pass
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    species_colors = {"H3O_plus": "#d95f02", "Na_plus": "#2c7fb8", "Cl_minus": "#756bb1", "OH_minus_bulk": "#31a354", "OH_minus_surf": "#74c476", "H_plus_surf": "#c51b8a"}
    for index, label in enumerate(labels, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        sample_path = root / label / "ion3d_samples.csv.gz"
        data = _read_csv(sample_path)
        data = data[data.gap_bin == "4-6A"]
        if len(data) > args.cloud_points:
            data = data.sample(args.cloud_points, random_state=20260821)
        for species, group in data.groupby("species"):
            ax.scatter(group.s_A, group.u_A, group.z_A, s=4, alpha=0.35, color=species_colors.get(species, "#555555"), label=species)
        ax.set_title(f"{label}\n4–6 Å, n={len(data)} points")
        ax.set_xlabel("s (Å)")
        ax.set_ylabel("u (Å)")
        ax.set_zlabel("z (Å)")
        ax.view_init(elev=22, azim=-55)
    handles, labels_legend = axes[0].get_legend_handles_labels() if False else ([], [])
    fig.suptitle("Ensemble 3D ion clouds in the dual-bubble frame", fontsize=15)
    fig.savefig(figure_dir / "candidate_ion3d_ensemble_cloud.png", dpi=220)
    plt.close(fig)
    print(f"figures={figure_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["analyze-case", "assemble"])
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--case-label")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--figure-dir")
    parser.add_argument("--gap-min-A", type=float, default=0.0)
    parser.add_argument("--gap-max-A", type=float, default=18.0)
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--s-min-A", type=float, default=-24.0)
    parser.add_argument("--s-max-A", type=float, default=24.0)
    parser.add_argument("--s-bin-width-A", type=float, default=2.0)
    parser.add_argument("--rho-max-A", type=float, default=24.0)
    parser.add_argument("--rho-bin-width-A", type=float, default=2.0)
    parser.add_argument("--core-rho-max-A", type=float, default=6.0)
    parser.add_argument("--surface-shell-width-A", type=float, default=4.0)
    parser.add_argument("--shell-rho-max-A", type=float, default=10.0)
    parser.add_argument("--phi-bins", type=int, default=12)
    parser.add_argument("--block-ns", type=float, default=0.020)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--box-lengths-A", type=float, nargs=3, default=(113.65520574, 75.6507936, 220.39056371))
    parser.add_argument("--state", default="separated")
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--cloud-points", type=int, default=1800)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "analyze-case":
        if not args.case_label:
            raise SystemExit("--case-label is required for analyze-case")
        return analyze_case(args)
    if not args.figure_dir:
        raise SystemExit("--figure-dir is required for assemble")
    return assemble(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
