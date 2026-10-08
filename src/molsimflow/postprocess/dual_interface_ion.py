"""Matched-gap formal-charge and ionic-field proxies in a two-bubble frame.

This module consumes existing, classified ion-position tables.  It does not
infer Deep-Potential charges.  The resulting fields are deliberately labelled
as softened formal-charge proxies rather than exact electrostatic fields.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.dual_interface_water import (
    _bootstrap_ratio,
    _write_csv,
    gap_window_label,
    parse_gap_window,
)


COULOMB_V_A_PER_E = 14.399645478
ALLOWED_CHARGES = {
    "H3O_plus": 1.0,
    "Na_plus": 1.0,
    "H_plus_surf": 1.0,
    "Cl_minus": -1.0,
    "OH_minus_bulk": -1.0,
    "OH_minus_surf": -1.0,
}


@dataclass(frozen=True)
class IonConfig:
    gap_windows_A: Tuple[Tuple[float, float], ...] = ((4.0, 6.0), (12.0, 14.0))
    nominal_radius_a_A: float = 19.0
    nominal_radius_b_A: float = 19.0
    state: str = "separated"
    box_lengths_A: Tuple[float, float, float] = (113.65520574, 75.6507936, 220.39056371)
    s_min_A: float = -24.0
    s_max_A: float = 24.0
    z_min_A: float = -24.0
    z_max_A: float = 20.0
    transverse_half_width_A: float = 6.0
    s_bins: int = 48
    z_bins: int = 44
    epsilon_r: float = 78.5
    softening_A: float = 2.0
    smoothing_sigma_bins: float = 1.0
    block_ns: float = 0.020
    bootstrap_samples: int = 1000
    random_seed: int = 20260821
    max_frames_per_window: int = 0

    def validate(self) -> None:
        if not self.gap_windows_A or any(b <= a for a, b in self.gap_windows_A):
            raise ValueError("Invalid gap windows")
        if self.epsilon_r <= 0 or self.softening_A <= 0 or self.block_ns <= 0:
            raise ValueError("epsilon_r, softening_A, and block_ns must be positive")
        if min(self.box_lengths_A) <= 0 or min(self.s_bins, self.z_bins) <= 0:
            raise ValueError("Box lengths and bin counts must be positive")


@dataclass(frozen=True)
class IonCaseSpec:
    case_label: str
    frame_csv: Path
    position_csv: Path
    water_map_csv: Path
    nominal_radius_a_A: float = 19.0
    nominal_radius_b_A: float = 19.0


def read_case_manifest(path: Path, case_index: Optional[int] = None, case_label: Optional[str] = None) -> IonCaseSpec:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if case_label is not None:
        rows = [row for row in rows if row.get("case_label") == case_label]
        if len(rows) != 1:
            raise ValueError(f"Expected one case_label={case_label!r}, found {len(rows)}")
        row = rows[0]
    else:
        index = 0 if case_index is None else int(case_index)
        if index < 0 or index >= len(rows):
            raise IndexError(f"Case index {index} is outside 0..{len(rows) - 1}")
        row = rows[index]
    return IonCaseSpec(
        case_label=str(row["case_label"]),
        frame_csv=Path(row["frame_csv"]).expanduser(),
        position_csv=Path(row["position_csv"]).expanduser(),
        water_map_csv=Path(row["water_map_csv"]).expanduser(),
        nominal_radius_a_A=float(row.get("nominal_radius_a_A") or 19.0),
        nominal_radius_b_A=float(row.get("nominal_radius_b_A") or 19.0),
    )


def minimum_image(delta: np.ndarray, lengths: Sequence[float]) -> np.ndarray:
    lengths_array = np.asarray(lengths, dtype=float)
    return np.asarray(delta, dtype=float) - lengths_array * np.round(np.asarray(delta, dtype=float) / lengths_array)


def local_basis(center_a: np.ndarray, center_b: np.ndarray, lengths: Sequence[float]) -> Tuple[np.ndarray, ...]:
    delta = minimum_image(np.asarray(center_b) - np.asarray(center_a), lengths)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        raise ValueError("Bubble centers define a degenerate axis")
    e_s = delta / norm
    surface_normal = np.array([0.0, 0.0, 1.0])
    e_u = np.cross(surface_normal, e_s)
    if float(np.linalg.norm(e_u)) <= 1e-8:
        e_u = np.cross(np.array([0.0, 1.0, 0.0]), e_s)
    e_u /= np.linalg.norm(e_u)
    e_z = np.cross(e_s, e_u)
    e_z /= np.linalg.norm(e_z)
    if float(np.dot(e_z, surface_normal)) < 0:
        e_u *= -1
        e_z *= -1
    midpoint = np.asarray(center_a) + 0.5 * delta
    return midpoint, e_s, e_u, e_z


def field_proxy(
    evaluation_points: np.ndarray,
    charge_positions: np.ndarray,
    charges_e: np.ndarray,
    epsilon_r: float,
    softening_A: float,
) -> np.ndarray:
    """Return a softened Coulomb field proxy in V/A."""

    points = np.asarray(evaluation_points, dtype=float)
    positions = np.asarray(charge_positions, dtype=float)
    charges = np.asarray(charges_e, dtype=float)
    if len(positions) == 0:
        return np.zeros_like(points)
    displacement = points[:, None, :] - positions[None, :, :]
    radius2 = np.einsum("mni,mni->mn", displacement, displacement) + float(softening_A) ** 2
    weighted = charges[None, :, None] * displacement / np.power(radius2[:, :, None], 1.5)
    return (COULOMB_V_A_PER_E / float(epsilon_r)) * weighted.sum(axis=1)


def validate_species_charge(species: object, charge: object) -> float:
    name = str(species)
    value = float(charge)
    expected = ALLOWED_CHARGES.get(name)
    if expected is None:
        raise ValueError(f"Nonzero charge for unsupported species {name!r}")
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Charge mismatch for {name}: observed {value}, expected {expected}")
    return value


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-ion requires pandas") from exc
    return pd


def _select_frames(case: IonCaseSpec, config: IonConfig):
    pd = _require_pandas()
    frame = pd.read_csv(case.frame_csv)
    frame["nominal_gap_A"] = (
        pd.to_numeric(frame["d3d_all"], errors="coerce")
        - case.nominal_radius_a_A
        - case.nominal_radius_b_A
    )
    frame = frame[frame["coalescence_state"].astype(str).eq(config.state)].copy()
    selected = []
    for window in config.gap_windows_A:
        chunk = frame[(frame.nominal_gap_A >= window[0]) & (frame.nominal_gap_A < window[1])].sort_values("time_ns")
        if config.max_frames_per_window > 0 and len(chunk) > config.max_frames_per_window:
            indices = np.linspace(0, len(chunk) - 1, config.max_frames_per_window, dtype=int)
            chunk = chunk.iloc[indices]
        chunk = chunk.copy()
        chunk["gap_window"] = gap_window_label(window)
        selected.append(chunk)
    result = pd.concat(selected, ignore_index=True)
    if result.empty:
        raise ValueError(f"No frames match requested windows for {case.case_label}")
    if result.global_frame.duplicated().any():
        raise ValueError("A frame was assigned to more than one gap window")
    return result


def _read_charge_samples(case: IonCaseSpec, frames, config: IonConfig) -> List[Dict[str, object]]:
    pd = _require_pandas()
    frame_lookup = {int(row.global_frame): row for row in frames.itertuples(index=False)}
    selected_ids = set(frame_lookup)
    usecols = [
        "global_frame", "atom_id", "species_canonical", "species_charge_e",
        "x_A", "y_A", "z_A", "in_bridge", "in_trace_region", "in_bridge_region", "in_slab",
    ]
    output: List[Dict[str, object]] = []
    for chunk in pd.read_csv(case.position_csv, usecols=usecols, chunksize=200000):
        charge = pd.to_numeric(chunk["species_charge_e"], errors="coerce").fillna(0.0)
        chunk = chunk[chunk["global_frame"].isin(selected_ids) & charge.ne(0.0)]
        for row in chunk.itertuples(index=False):
            q = validate_species_charge(row.species_canonical, row.species_charge_e)
            meta = frame_lookup[int(row.global_frame)]
            center_a = np.asarray([meta.bubble_A_center_x_A, meta.bubble_A_center_y_A, meta.bubble_A_center_z_A], dtype=float)
            center_b = np.asarray([meta.bubble_B_center_x_A, meta.bubble_B_center_y_A, meta.bubble_B_center_z_A], dtype=float)
            midpoint, e_s, e_u, e_z = local_basis(center_a, center_b, config.box_lengths_A)
            disp = minimum_image(np.asarray([row.x_A, row.y_A, row.z_A], dtype=float) - midpoint, config.box_lengths_A)
            s, u, z = float(disp @ e_s), float(disp @ e_u), float(disp @ e_z)
            output.append(
                {
                    "case_label": case.case_label,
                    "global_frame": int(row.global_frame),
                    "time_ns": float(meta.time_ns),
                    "gap_window": str(meta.gap_window),
                    "nominal_gap_A": float(meta.nominal_gap_A),
                    "atom_id": int(row.atom_id),
                    "species": str(row.species_canonical),
                    "formal_charge_e": q,
                    "s_A": s,
                    "z_mid_A": z,
                    "u_A": u,
                    "rho_A": float(math.hypot(u, z)),
                    "in_bridge": int(row.in_bridge),
                    "in_trace_region": int(row.in_trace_region),
                    "in_bridge_region": int(row.in_bridge_region),
                    "in_slab": int(row.in_slab),
                }
            )
    return output


def _frame_field_arrays(samples, frames, window_label: str, s_centers: np.ndarray, config: IonConfig, softening_A: float, epsilon_r: float):
    frame_ids = [int(row.global_frame) for row in frames.itertuples(index=False) if row.gap_window == window_label]
    by_frame: Dict[int, List[Mapping[str, object]]] = {frame_id: [] for frame_id in frame_ids}
    for row in samples:
        if row["gap_window"] == window_label and abs(float(row["u_A"])) < config.transverse_half_width_A:
            by_frame[int(row["global_frame"])].append(row)
    values = np.zeros((len(frame_ids), len(s_centers)), dtype=float)
    points = np.column_stack([s_centers, np.zeros(len(s_centers)), np.zeros(len(s_centers))])
    for index, frame_id in enumerate(frame_ids):
        rows = by_frame[frame_id]
        positions = np.asarray([[float(r["s_A"]), float(r["u_A"]), float(r["z_mid_A"])] for r in rows], dtype=float).reshape((-1, 3))
        charges = np.asarray([float(r["formal_charge_e"]) for r in rows], dtype=float)
        values[index] = field_proxy(points, positions, charges, epsilon_r, softening_A)[:, 0]
    times = np.asarray([float(next(row.time_ns for row in frames.itertuples(index=False) if int(row.global_frame) == fid)) for fid in frame_ids])
    return frame_ids, times, values


def _build_map(samples, frames, config: IonConfig) -> List[Dict[str, object]]:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-ion requires scipy") from exc
    s_edges = np.linspace(config.s_min_A, config.s_max_A, config.s_bins + 1)
    z_edges = np.linspace(config.z_min_A, config.z_max_A, config.z_bins + 1)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    ds, dz = np.diff(s_edges)[0], np.diff(z_edges)[0]
    volume = float(ds * dz * 2.0 * config.transverse_half_width_A)
    output = []
    for window in config.gap_windows_A:
        label = gap_window_label(window)
        n_frames = int(sum(1 for row in frames.itertuples(index=False) if row.gap_window == label))
        rows = [row for row in samples if row["gap_window"] == label and abs(float(row["u_A"])) < config.transverse_half_width_A]
        s = np.asarray([float(row["s_A"]) for row in rows])
        z = np.asarray([float(row["z_mid_A"]) for row in rows])
        q = np.asarray([float(row["formal_charge_e"]) for row in rows])
        raw = np.histogram2d(s, z, bins=(s_edges, z_edges), weights=q)[0] / (n_frames * volume)
        absolute = np.histogram2d(s, z, bins=(s_edges, z_edges), weights=np.abs(q))[0] / (n_frames * volume)
        smoothed = gaussian_filter(raw, sigma=config.smoothing_sigma_bins, mode="constant")
        ss, zz = np.meshgrid(s_centers, z_centers, indexing="ij")
        points = np.column_stack([ss.ravel(), np.zeros(ss.size), zz.ravel()])
        positions = np.asarray([[float(r["s_A"]), float(r["u_A"]), float(r["z_mid_A"])] for r in rows], dtype=float).reshape((-1, 3))
        charges = np.asarray([float(r["formal_charge_e"]) for r in rows], dtype=float)
        field = field_proxy(points, positions, charges, config.epsilon_r, config.softening_A) / n_frames
        field = field.reshape((config.s_bins, config.z_bins, 3))
        for i, sc in enumerate(s_centers):
            for j, zc in enumerate(z_centers):
                output.append(
                    {
                        "gap_window": label,
                        "s_left_A": s_edges[i], "s_right_A": s_edges[i + 1], "s_center_A": sc,
                        "z_left_A": z_edges[j], "z_right_A": z_edges[j + 1], "z_center_A": zc,
                        "n_frames": n_frames,
                        "bin_volume_A3": volume,
                        "formal_charge_density_e_A3": raw[i, j],
                        "formal_abs_charge_density_e_A3": absolute[i, j],
                        "formal_charge_density_smoothed_e_A3": smoothed[i, j],
                        "E_s_ion_proxy_V_A": field[i, j, 0],
                        "E_z_ion_proxy_V_A": field[i, j, 2],
                        "epsilon_r": config.epsilon_r,
                        "softening_A": config.softening_A,
                    }
                )
    return output


def _build_profiles(samples, frames, config: IonConfig):
    s_edges = np.linspace(config.s_min_A, config.s_max_A, config.s_bins + 1)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    rng = np.random.default_rng(config.random_seed)
    profiles, sensitivity = [], []
    for window in config.gap_windows_A:
        label = gap_window_label(window)
        frame_ids, times, values = _frame_field_arrays(samples, frames, label, s_centers, config, config.softening_A, config.epsilon_r)
        block_ids = np.floor(times / config.block_ns + 1e-10).astype(int)
        blocks = sorted(set(block_ids.tolist()))
        for j, center in enumerate(s_centers):
            sums = np.asarray([values[block_ids == block, j].sum() for block in blocks])
            counts = np.asarray([np.count_nonzero(block_ids == block) for block in blocks], dtype=float)
            low, high = _bootstrap_ratio(sums, counts, config.bootstrap_samples, rng)
            profiles.append(
                {
                    "gap_window": label, "s_center_A": center, "n_frames": len(frame_ids),
                    "effective_block_count": len(blocks), "E_s_ion_proxy_mean_V_A": float(values[:, j].mean()),
                    "E_s_ion_proxy_ci95_low_V_A": low, "E_s_ion_proxy_ci95_high_V_A": high,
                    "epsilon_r": config.epsilon_r, "softening_A": config.softening_A,
                }
            )
        for softening in (1.0, 2.0, 3.0):
            for epsilon in (40.0, 78.5):
                _, _, trial = _frame_field_arrays(samples, frames, label, np.asarray([-8.0, 0.0, 8.0]), config, softening, epsilon)
                for column, location in enumerate(("A_entrance", "center", "B_entrance")):
                    sensitivity.append(
                        {
                            "gap_window": label, "location": location, "s_A": (-8.0, 0.0, 8.0)[column],
                            "softening_A": softening, "epsilon_r": epsilon, "n_frames": len(frame_ids),
                            "E_s_ion_proxy_mean_V_A": float(trial[:, column].mean()),
                        }
                    )
    return profiles, sensitivity


def analyze_case(case: IonCaseSpec, output_dir: Path, config: IonConfig) -> Dict[str, Path]:
    config = IonConfig(**{**config.__dict__, "nominal_radius_a_A": case.nominal_radius_a_A, "nominal_radius_b_A": case.nominal_radius_b_A})
    config.validate()
    for path in (case.frame_csv, case.position_csv, case.water_map_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    frames = _select_frames(case, config)
    samples = _read_charge_samples(case, frames, config)
    maps = _build_map(samples, frames, config)
    profiles, sensitivity = _build_profiles(samples, frames, config)
    frame_rows = []
    for row in frames.itertuples(index=False):
        charge_rows = [item for item in samples if int(item["global_frame"]) == int(row.global_frame)]
        frame_rows.append(
            {
                "case_label": case.case_label, "global_frame": int(row.global_frame), "time_ns": float(row.time_ns),
                "gap_window": str(row.gap_window), "nominal_gap_A": float(row.nominal_gap_A),
                "charged_position_count": len(charge_rows),
                "formal_net_charge_e": sum(float(item["formal_charge_e"]) for item in charge_rows),
                "formal_abs_charge_e": sum(abs(float(item["formal_charge_e"])) for item in charge_rows),
            }
        )
    output_dir = Path(output_dir)
    outputs = {
        "samples": output_dir / "formal_charge_samples.csv.gz",
        "frames": output_dir / "formal_charge_frame_summary.csv",
        "map": output_dir / "formal_charge_field_sz_map.csv",
        "profile": output_dir / "ionic_field_axial_profile.csv",
        "sensitivity": output_dir / "ionic_field_sensitivity.csv",
        "statistics": output_dir / "state_statistics.csv",
        "manifest": output_dir / "artifact_manifest.csv",
    }
    _write_csv(outputs["samples"], samples)
    _write_csv(outputs["frames"], frame_rows)
    _write_csv(outputs["map"], maps)
    _write_csv(outputs["profile"], profiles)
    _write_csv(outputs["sensitivity"], sensitivity)
    _write_csv(
        outputs["statistics"],
        [
            {"metric": "case_label", "value": case.case_label},
            {"metric": "frame_csv", "value": str(case.frame_csv)},
            {"metric": "position_csv", "value": str(case.position_csv)},
            {"metric": "selected_frames", "value": len(frames)},
            {"metric": "charged_position_rows", "value": len(samples)},
            {"metric": "charge_model", "value": "classified species formal charges only"},
            {"metric": "epsilon_r", "value": config.epsilon_r},
            {"metric": "softening_A", "value": config.softening_A},
            {"metric": "interpretation", "value": "ionic field proxy; not exact electric field"},
        ],
    )
    manifest = []
    for name, path in outputs.items():
        if name == "manifest":
            continue
        manifest.append({"artifact": name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_csv(outputs["manifest"], manifest)
    return outputs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def assemble_figures(case_manifest: Path, output_root: Path, figure_dir: Path, config: IonConfig) -> Dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with Path(case_manifest).open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    cases = [row["case_label"] for row in manifest]
    maps = {case: _read_csv(Path(output_root) / case / "formal_charge_field_sz_map.csv") for case in cases}
    profiles = {case: _read_csv(Path(output_root) / case / "ionic_field_axial_profile.csv") for case in cases}
    sensitivities = {case: _read_csv(Path(output_root) / case / "ionic_field_sensitivity.csv") for case in cases}
    water_maps = {row["case_label"]: _read_csv(Path(row["water_map_csv"])) for row in manifest}
    plot_dir = Path(figure_dir) / "plot_data"
    combined_maps = [{"case_label": case, **row} for case in cases for row in maps[case]]
    combined_profiles = [{"case_label": case, **row} for case in cases for row in profiles[case]]
    combined_sensitivity = [{"case_label": case, **row} for case in cases for row in sensitivities[case]]
    map_csv = plot_dir / "multicase_formal_charge_field_sz_map.csv"
    profile_csv = plot_dir / "multicase_ionic_field_axial_profile.csv"
    sensitivity_csv = plot_dir / "multicase_ionic_field_sensitivity.csv"
    _write_csv(map_csv, combined_maps)
    _write_csv(profile_csv, combined_profiles)
    _write_csv(sensitivity_csv, combined_sensitivity)

    windows = [gap_window_label(item) for item in config.gap_windows_A]
    magnitudes = np.asarray([abs(float(row["formal_charge_density_smoothed_e_A3"])) for row in combined_maps])
    vmax = float(np.nanpercentile(magnitudes, 99.5)) if np.any(np.isfinite(magnitudes)) else 1e-4
    field_magnitudes = np.asarray(
        [
            math.hypot(float(row["E_s_ion_proxy_V_A"]), float(row["E_z_ion_proxy_V_A"]))
            for row in combined_maps
        ]
    )
    field_reference = float(np.nanpercentile(field_magnitudes, 99.0))
    field_display_threshold = 0.05 * field_reference
    fig, axes = plt.subplots(len(windows), len(cases), figsize=(3.35 * len(cases), 3.25 * len(windows)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    image = None
    for i, window in enumerate(windows):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            rows = [row for row in maps[case] if row["gap_window"] == window]
            waters = [row for row in water_maps[case] if row["gap_window"] == window]
            svals = sorted({float(row["s_center_A"]) for row in rows})
            zvals = sorted({float(row["z_center_A"]) for row in rows})
            lookup = {(float(row["s_center_A"]), float(row["z_center_A"])): row for row in rows}
            wlookup = {(float(row["s_center_A"]), float(row["z_center_A"])): row for row in waters}
            charge = np.asarray([[float(lookup[(s, z)]["formal_charge_density_smoothed_e_A3"]) for s in svals] for z in zvals])
            es = np.asarray([[float(lookup[(s, z)]["E_s_ion_proxy_V_A"]) for s in svals] for z in zvals])
            ez = np.asarray([[float(lookup[(s, z)]["E_z_ion_proxy_V_A"]) for s in svals] for z in zvals])
            mu_s = np.asarray([[float(wlookup[(s, z)]["mu_s_mean"]) if wlookup[(s, z)]["mu_s_mean"] else np.nan for s in svals] for z in zvals])
            mu_z = np.asarray([[float(wlookup[(s, z)]["mu_z_mean"]) if wlookup[(s, z)]["mu_z_mean"] else np.nan for s in svals] for z in zvals])
            wcount = np.asarray([[int(wlookup[(s, z)]["count"]) for s in svals] for z in zvals])
            image = ax.pcolormesh(svals, zvals, charge, shading="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ss, zz = np.meshgrid(svals, zvals)
            field_norm = np.hypot(es, ez)
            safe_es = np.divide(es, field_norm, out=np.zeros_like(es), where=field_norm > 1e-12)
            safe_ez = np.divide(ez, field_norm, out=np.zeros_like(ez), where=field_norm > 1e-12)
            field_mask = field_norm >= field_display_threshold
            ax.quiver(
                ss[::5, ::5], zz[::5, ::5],
                np.where(field_mask, safe_es, np.nan)[::5, ::5],
                np.where(field_mask, safe_ez, np.nan)[::5, ::5],
                color="black", alpha=0.65, scale=18, width=0.004,
            )
            mask = wcount >= 10
            ax.quiver(ss[::5, ::5], zz[::5, ::5], np.where(mask, mu_s, np.nan)[::5, ::5], np.where(mask, mu_z, np.nan)[::5, ::5], color="#f4c430", scale=4.5, width=0.006)
            ax.axvline(0, color="0.35", linewidth=0.5)
            if i == 0:
                ax.set_title(case)
            if j == 0:
                ax.set_ylabel(f"$h={window[:-1]}$ Å\n$z_{{mid}}$ (Å)")
            if i == len(windows) - 1:
                ax.set_xlabel("$s$ (Å)")
    cax = fig.add_axes([0.915, 0.18, 0.014, 0.64])
    fig.colorbar(image, cax=cax, label="Smoothed formal-charge density (e Å$^{-3}$)")
    fig.legend(
        handles=[Line2D([0], [0], color="black", label="ionic-field proxy direction"), Line2D([0], [0], color="#f4c430", label="water geometric dipole")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.49, 0.005),
    )
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    atlas = figure_dir / "candidate_ion_effect_p0.png"
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.10, top=0.92, wspace=0.08, hspace=0.12)
    fig.savefig(atlas, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(windows), figsize=(6.0 * len(windows), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(cases)))
    for ax, window in zip(axes, windows):
        for color, case in zip(colors, cases):
            rows = [row for row in profiles[case] if row["gap_window"] == window]
            x = np.asarray([float(row["s_center_A"]) for row in rows])
            mean = np.asarray([float(row["E_s_ion_proxy_mean_V_A"]) for row in rows])
            low = np.asarray([float(row["E_s_ion_proxy_ci95_low_V_A"]) for row in rows])
            high = np.asarray([float(row["E_s_ion_proxy_ci95_high_V_A"]) for row in rows])
            ax.plot(x, mean, color=color, label=case, linewidth=1.5)
            ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.axhline(0, color="0.4", linewidth=0.7)
        ax.axvline(0, color="0.4", linewidth=0.7)
        ax.set_title(f"$h={window[:-1]}$ Å")
        ax.set_xlabel("$s$ (Å)")
    axes[0].set_ylabel("Mean axial ionic-field proxy (V Å$^{-1}$)")
    axes[-1].legend(frameon=False, fontsize=8)
    profile_figure = figure_dir / "candidate_ionic_field_axial_proxy.png"
    fig.tight_layout()
    fig.savefig(profile_figure, dpi=300)
    plt.close(fig)
    return {"atlas": atlas, "axial_profile": profile_figure, "atlas_plot_data": map_csv, "profile_plot_data": profile_csv, "sensitivity_plot_data": sensitivity_csv}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched-gap formal-charge and ionic-field proxy analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze-case")
    analyze.add_argument("--case-manifest", type=Path, required=True)
    group = analyze.add_mutually_exclusive_group()
    group.add_argument("--case-index", type=int)
    group.add_argument("--case-label")
    analyze.add_argument("--output-root", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--case-manifest", type=Path, required=True)
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--figure-dir", type=Path, required=True)
    for target in (analyze, assemble):
        target.add_argument("--gap-window", action="append", default=[])
        target.add_argument("--epsilon-r", type=float, default=78.5)
        target.add_argument("--softening-A", type=float, default=2.0)
        target.add_argument("--block-ns", type=float, default=0.020)
        target.add_argument("--bootstrap-samples", type=int, default=1000)
    analyze.add_argument("--max-frames-per-window", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    windows = tuple(parse_gap_window(item) for item in args.gap_window) or ((4.0, 6.0), (12.0, 14.0))
    config = IonConfig(
        gap_windows_A=windows, epsilon_r=args.epsilon_r, softening_A=args.softening_A,
        block_ns=args.block_ns, bootstrap_samples=args.bootstrap_samples,
        max_frames_per_window=getattr(args, "max_frames_per_window", 0),
    )
    if args.command == "analyze-case":
        case = read_case_manifest(args.case_manifest, args.case_index, args.case_label)
        outputs = analyze_case(case, Path(args.output_root) / case.case_label, config)
    else:
        outputs = assemble_figures(args.case_manifest, args.output_root, args.figure_dir, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
