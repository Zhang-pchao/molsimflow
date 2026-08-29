"""Matched-gap water-density and dipole maps in a two-bubble frame."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    iter_lammps_dump_records,
    minimum_image_vectors,
    wrap_point_to_box,
)


@dataclass(frozen=True)
class DualInterfaceConfig:
    """Geometry, topology, and binning choices for one analysis run."""

    gap_windows_A: Tuple[Tuple[float, float], ...] = ((4.0, 6.0), (12.0, 14.0))
    nominal_radius_a_A: float = 19.0
    nominal_radius_b_A: float = 19.0
    state: str = "separated"
    oxygen_type: int = 2
    hydrogen_type: int = 1
    titanium_type: int = 6
    oh_cutoff_A: float = 1.25
    s_min_A: float = -24.0
    s_max_A: float = 24.0
    z_min_A: float = -24.0
    z_max_A: float = 20.0
    transverse_half_width_A: float = 6.0
    rho_max_A: float = 24.0
    s_bins: int = 48
    z_bins: int = 44
    rho_bins: int = 24
    min_bin_count: int = 10
    surface_quantile: float = 95.0
    block_ns: float = 0.020
    bootstrap_samples: int = 1000
    random_seed: int = 20260821
    max_frames_per_window: int = 0

    def validate(self) -> None:
        if not self.gap_windows_A:
            raise ValueError("At least one gap window is required")
        if any(right <= left for left, right in self.gap_windows_A):
            raise ValueError("Gap-window upper bounds must exceed lower bounds")
        if self.s_max_A <= self.s_min_A or self.z_max_A <= self.z_min_A:
            raise ValueError("Map upper bounds must exceed lower bounds")
        if self.transverse_half_width_A <= 0 or self.rho_max_A <= 0:
            raise ValueError("Transverse cutoffs must be positive")
        if min(self.s_bins, self.z_bins, self.rho_bins) <= 0:
            raise ValueError("Bin counts must be positive")
        if self.oh_cutoff_A <= 0 or self.block_ns <= 0:
            raise ValueError("Topology cutoff and block duration must be positive")


@dataclass(frozen=True)
class CaseSpec:
    case_label: str
    trace_csv: Path
    segments: Tuple[Tuple[str, Path], ...]
    has_tio2: bool
    nominal_radius_a_A: float = 19.0
    nominal_radius_b_A: float = 19.0


def parse_gap_window(raw: str) -> Tuple[float, float]:
    """Parse ``left:right`` into a numeric gap window."""

    parts = raw.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid gap window {raw!r}; expected left:right")
    left, right = (float(value) for value in parts)
    if right <= left:
        raise ValueError(f"Invalid gap window {raw!r}; right must exceed left")
    return left, right


def gap_window_label(window: Tuple[float, float]) -> str:
    return f"{window[0]:g}-{window[1]:g}A"


def parse_segment_specs(raw: str) -> Tuple[Tuple[str, Path], ...]:
    """Parse semicolon-separated ``label=trajectory`` entries."""

    segments: List[Tuple[str, Path]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid segment entry {item!r}; expected label=trajectory")
        label, path = item.split("=", 1)
        segments.append((label.strip(), Path(path.strip()).expanduser()))
    if not segments:
        raise ValueError("No trajectory segments were configured")
    return tuple(segments)


def read_case_manifest(path: Path, case_index: Optional[int] = None, case_label: Optional[str] = None) -> CaseSpec:
    """Read one case from a CSV manifest."""

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Case manifest is empty: {path}")
    if case_label is not None:
        matches = [row for row in rows if row.get("case_label") == case_label]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one case_label={case_label!r}, found {len(matches)}")
        row = matches[0]
    else:
        index = 0 if case_index is None else int(case_index)
        if index < 0 or index >= len(rows):
            raise IndexError(f"Case index {index} is outside 0..{len(rows) - 1}")
        row = rows[index]
    return CaseSpec(
        case_label=str(row["case_label"]),
        trace_csv=Path(row["trace_csv"]).expanduser(),
        segments=parse_segment_specs(str(row["segment_specs"])),
        has_tio2=str(row.get("has_tio2", "0")).strip().lower() in {"1", "true", "yes"},
        nominal_radius_a_A=float(row.get("nominal_radius_a_A") or 19.0),
        nominal_radius_b_A=float(row.get("nominal_radius_b_A") or 19.0),
    )


def local_basis(center_a: np.ndarray, center_b: np.ndarray, bounds: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return midpoint and an orthonormal ``s,u,z`` basis."""

    lengths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    delta = minimum_image_vectors(np.asarray(center_b) - np.asarray(center_a), lengths)
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
    if float(np.dot(e_z, surface_normal)) < 0.0:
        e_u *= -1.0
        e_z *= -1.0
    midpoint = wrap_point_to_box(np.asarray(center_a) + 0.5 * delta, np.asarray(bounds, dtype=float))
    return midpoint, e_s, e_u, e_z


def assign_intact_waters(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Assign neutral H2O while rejecting shared-H and non-twofold oxygen sites."""

    if len(oxygen_positions) == 0 or len(hydrogen_positions) == 0:
        return [], np.zeros(len(oxygen_positions), dtype=int)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - analysis extra is required on HPC
        raise RuntimeError("dual-interface-water requires scipy") from exc
    bounds = np.asarray(bounds, dtype=float)
    lengths = bounds[:, 1] - bounds[:, 0]
    oxy = (np.asarray(oxygen_positions) - bounds[:, 0]) % lengths
    hyd = (np.asarray(hydrogen_positions) - bounds[:, 0]) % lengths
    adjacency = cKDTree(oxy, boxsize=lengths).query_ball_tree(cKDTree(hyd, boxsize=lengths), r=float(cutoff_A))
    h_degree = np.zeros(len(hydrogen_positions), dtype=int)
    for neighbors in adjacency:
        for index in neighbors:
            h_degree[index] += 1
    assignments: List[Tuple[int, int]] = []
    status = np.zeros(len(adjacency), dtype=int)
    for oxygen_index, neighbors in enumerate(adjacency):
        if len(neighbors) != 2:
            status[oxygen_index] = len(neighbors)
            assignments.append((-1, -1))
            continue
        if any(h_degree[index] != 1 for index in neighbors):
            status[oxygen_index] = -1
            assignments.append((-1, -1))
            continue
        status[oxygen_index] = 2
        assignments.append((int(neighbors[0]), int(neighbors[1])))
    return assignments, status


def _coord_index(fields: Sequence[str], dim: str) -> Tuple[int, bool]:
    for name in (dim, dim + "u", dim + "s"):
        if name in fields:
            return fields.index(name), name.endswith("s")
    raise ValueError(f"LAMMPS dump is missing {dim}/{dim}u/{dim}s")


def _positions_by_type(frame: LammpsDumpFrame, needed_types: Iterable[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    fields = frame.atom_fields
    if "id" not in fields or "type" not in fields:
        raise ValueError("LAMMPS dump requires id and type columns")
    id_index, type_index = fields.index("id"), fields.index("type")
    coord_info = [_coord_index(fields, dim) for dim in "xyz"]
    lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
    wanted = set(int(value) for value in needed_types)
    ids: Dict[int, List[int]] = {value: [] for value in wanted}
    coords: Dict[int, List[List[float]]] = {value: [] for value in wanted}
    for row in frame.atom_rows:
        atom_type = int(row[type_index])
        if atom_type not in wanted:
            continue
        xyz = [float(row[index]) for index, _ in coord_info]
        for dim, (_, scaled) in enumerate(coord_info):
            if scaled:
                xyz[dim] = frame.bounds[dim, 0] + xyz[dim] * lengths[dim]
        ids[atom_type].append(int(row[id_index]))
        coords[atom_type].append(xyz)
    return {
        atom_type: (np.asarray(ids[atom_type], dtype=int), np.asarray(coords[atom_type], dtype=float).reshape((-1, 3)))
        for atom_type in wanted
    }


def _trace_rows(path: Path, config: DualInterfaceConfig) -> Dict[Tuple[str, int], Dict[str, object]]:
    candidates: Dict[str, List[Tuple[Tuple[str, int], Dict[str, object]]]] = {
        gap_window_label(window): [] for window in config.gap_windows_A
    }
    with Path(path).open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("state", "")) != config.state:
                continue
            distance = float(raw["bubble_center_distance_A"])
            gap = distance - config.nominal_radius_a_A - config.nominal_radius_b_A
            window = next((item for item in config.gap_windows_A if item[0] <= gap < item[1]), None)
            if window is None:
                continue
            row: Dict[str, object] = dict(raw)
            row["nominal_gap_A"] = gap
            row["gap_window"] = gap_window_label(window)
            key = (str(raw["segment"]), int(raw["local_frame"]))
            candidates[str(row["gap_window"])].append((key, row))
    selected: Dict[Tuple[str, int], Dict[str, object]] = {}
    for rows in candidates.values():
        rows.sort(key=lambda item: float(item[1]["time_ns"]))
        if config.max_frames_per_window > 0 and len(rows) > config.max_frames_per_window:
            indices = np.linspace(0, len(rows) - 1, config.max_frames_per_window, dtype=int)
            rows = [rows[int(index)] for index in indices]
        selected.update(rows)
    return selected


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _histogram_sum(x: np.ndarray, y: np.ndarray, weights: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    return np.histogram2d(x, y, bins=(x_edges, y_edges), weights=weights)[0]


def build_sz_map(
    sample_rows: Sequence[Mapping[str, object]],
    frame_rows: Sequence[Mapping[str, object]],
    config: DualInterfaceConfig,
) -> List[Dict[str, object]]:
    """Build number-density and mean-dipole maps in ``s,z_mid``."""

    s_edges = np.linspace(config.s_min_A, config.s_max_A, config.s_bins + 1)
    z_edges = np.linspace(config.z_min_A, config.z_max_A, config.z_bins + 1)
    ds, dz = float(np.diff(s_edges)[0]), float(np.diff(z_edges)[0])
    volume = ds * dz * (2.0 * config.transverse_half_width_A)
    output: List[Dict[str, object]] = []
    for window in config.gap_windows_A:
        label = gap_window_label(window)
        chunk = [row for row in sample_rows if row["gap_window"] == label]
        frames = [row for row in frame_rows if row["gap_window"] == label]
        n_frames = len(frames)
        s = np.asarray([float(row["s_A"]) for row in chunk])
        z = np.asarray([float(row["z_mid_A"]) for row in chunk])
        count = _histogram_sum(s, z, np.ones(len(chunk)), s_edges, z_edges)
        sums = {
            name: _histogram_sum(s, z, np.asarray([float(row[name]) for row in chunk]), s_edges, z_edges)
            for name in ("mu_s", "mu_z", "mu_u")
        }
        surface_values = [float(row["surface_z_mid_A"]) for row in frames if math.isfinite(float(row["surface_z_mid_A"]))]
        surface_mean = float(np.mean(surface_values)) if surface_values else math.nan
        for i in range(config.s_bins):
            for j in range(config.z_bins):
                n = int(count[i, j])
                output.append(
                    {
                        "gap_window": label,
                        "s_left_A": s_edges[i],
                        "s_right_A": s_edges[i + 1],
                        "s_center_A": 0.5 * (s_edges[i] + s_edges[i + 1]),
                        "z_left_A": z_edges[j],
                        "z_right_A": z_edges[j + 1],
                        "z_center_A": 0.5 * (z_edges[j] + z_edges[j + 1]),
                        "count": n,
                        "n_frames": n_frames,
                        "bin_volume_A3": volume,
                        "water_number_density_A3": n / (n_frames * volume) if n_frames else math.nan,
                        "mu_s_mean": sums["mu_s"][i, j] / n if n else math.nan,
                        "mu_z_mean": sums["mu_z"][i, j] / n if n else math.nan,
                        "mu_u_mean": sums["mu_u"][i, j] / n if n else math.nan,
                        "surface_z_mid_A_mean": surface_mean,
                    }
                )
    return output


def build_sr_map(
    sample_rows: Sequence[Mapping[str, object]],
    frame_rows: Sequence[Mapping[str, object]],
    config: DualInterfaceConfig,
) -> List[Dict[str, object]]:
    """Build the cylindrical ``s,rho`` audit map."""

    s_edges = np.linspace(config.s_min_A, config.s_max_A, config.s_bins + 1)
    rho_edges = np.linspace(0.0, config.rho_max_A, config.rho_bins + 1)
    output: List[Dict[str, object]] = []
    for window in config.gap_windows_A:
        label = gap_window_label(window)
        chunk = [row for row in sample_rows if row["gap_window"] == label and float(row["rho_A"]) < config.rho_max_A]
        n_frames = sum(1 for row in frame_rows if row["gap_window"] == label)
        s = np.asarray([float(row["s_A"]) for row in chunk])
        rho = np.asarray([float(row["rho_A"]) for row in chunk])
        count = _histogram_sum(s, rho, np.ones(len(chunk)), s_edges, rho_edges)
        mu_s_sum = _histogram_sum(s, rho, np.asarray([float(row["mu_s"]) for row in chunk]), s_edges, rho_edges)
        for i in range(config.s_bins):
            for j in range(config.rho_bins):
                n = int(count[i, j])
                volume = (s_edges[i + 1] - s_edges[i]) * math.pi * (rho_edges[j + 1] ** 2 - rho_edges[j] ** 2)
                output.append(
                    {
                        "gap_window": label,
                        "s_left_A": s_edges[i],
                        "s_right_A": s_edges[i + 1],
                        "s_center_A": 0.5 * (s_edges[i] + s_edges[i + 1]),
                        "rho_left_A": rho_edges[j],
                        "rho_right_A": rho_edges[j + 1],
                        "rho_center_A": 0.5 * (rho_edges[j] + rho_edges[j + 1]),
                        "count": n,
                        "n_frames": n_frames,
                        "bin_volume_A3": volume,
                        "water_number_density_A3": n / (n_frames * volume) if n_frames else math.nan,
                        "mu_s_mean": mu_s_sum[i, j] / n if n else math.nan,
                    }
                )
    return output


def _bootstrap_ratio(
    sums: np.ndarray,
    counts: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    usable = counts > 0
    sums, counts = sums[usable], counts[usable]
    if len(counts) < 3:
        return math.nan, math.nan
    draws = rng.integers(0, len(counts), size=(int(samples), len(counts)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def build_axial_profile(sample_rows: Sequence[Mapping[str, object]], config: DualInterfaceConfig) -> List[Dict[str, object]]:
    """Build axial polarization profiles with time-block bootstrap intervals."""

    edges = np.linspace(config.s_min_A, config.s_max_A, config.s_bins + 1)
    rng = np.random.default_rng(config.random_seed)
    output: List[Dict[str, object]] = []
    for window in config.gap_windows_A:
        label = gap_window_label(window)
        chunk = [row for row in sample_rows if row["gap_window"] == label]
        blocks = sorted({int(math.floor(float(row["time_ns"]) / config.block_ns + 1e-10)) for row in chunk})
        for index in range(config.s_bins):
            rows = [row for row in chunk if edges[index] <= float(row["s_A"]) < edges[index + 1]]
            block_sums = np.zeros(len(blocks), dtype=float)
            block_counts = np.zeros(len(blocks), dtype=float)
            lookup = {block: i for i, block in enumerate(blocks)}
            for row in rows:
                block = int(math.floor(float(row["time_ns"]) / config.block_ns + 1e-10))
                j = lookup[block]
                block_sums[j] += float(row["mu_s"])
                block_counts[j] += 1.0
            total = float(block_counts.sum())
            mean = float(block_sums.sum() / total) if total else math.nan
            low, high = _bootstrap_ratio(block_sums, block_counts, config.bootstrap_samples, rng)
            output.append(
                {
                    "gap_window": label,
                    "s_left_A": edges[index],
                    "s_right_A": edges[index + 1],
                    "s_center_A": 0.5 * (edges[index] + edges[index + 1]),
                    "count": int(total),
                    "effective_block_count": int(np.count_nonzero(block_counts)),
                    "mu_s_mean": mean,
                    "mu_s_ci95_low": low,
                    "mu_s_ci95_high": high,
                }
            )
    return output


def _plot_case_map(case_label: str, rows: Sequence[Mapping[str, object]], output_path: Path, config: DualInterfaceConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [gap_window_label(item) for item in config.gap_windows_A]
    fig, axes = plt.subplots(1, len(windows), figsize=(5.2 * len(windows), 4.1), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    densities = np.asarray([float(row["water_number_density_A3"]) for row in rows], dtype=float)
    vmax = float(np.nanpercentile(densities, 99.0)) if np.isfinite(densities).any() else 0.04
    image = None
    for ax, label in zip(axes, windows):
        chunk = [row for row in rows if row["gap_window"] == label]
        s_values = sorted({float(row["s_center_A"]) for row in chunk})
        z_values = sorted({float(row["z_center_A"]) for row in chunk})
        lookup = {(float(row["s_center_A"]), float(row["z_center_A"])): row for row in chunk}
        density = np.asarray([[float(lookup[(s, z)]["water_number_density_A3"]) for s in s_values] for z in z_values])
        mu_s = np.asarray([[float(lookup[(s, z)]["mu_s_mean"]) for s in s_values] for z in z_values])
        mu_z = np.asarray([[float(lookup[(s, z)]["mu_z_mean"]) for s in s_values] for z in z_values])
        count = np.asarray([[int(lookup[(s, z)]["count"]) for s in s_values] for z in z_values])
        image = ax.pcolormesh(s_values, z_values, density, shading="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
        step_s, step_z = max(1, len(s_values) // 12), max(1, len(z_values) // 11)
        ss, zz = np.meshgrid(s_values, z_values)
        mask = count >= config.min_bin_count
        ax.quiver(
            ss[::step_z, ::step_s],
            zz[::step_z, ::step_s],
            np.where(mask, mu_s, np.nan)[::step_z, ::step_s],
            np.where(mask, mu_z, np.nan)[::step_z, ::step_s],
            color="#b2182b",
            pivot="mid",
            scale=4.0,
            width=0.004,
        )
        surface = [float(row["surface_z_mid_A_mean"]) for row in chunk if math.isfinite(float(row["surface_z_mid_A_mean"]))]
        if surface:
            ax.axhline(float(np.mean(surface)), color="black", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="0.4", linewidth=0.6)
        ax.set_title(f"{case_label}\n$h={label[:-1]}$ Å")
        ax.set_xlabel("$s$ (Å)")
    axes[0].set_ylabel("$z_{mid}$ (Å)")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Water number density (Å$^{-3}$)", shrink=0.88)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.85, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_case(case: CaseSpec, output_dir: Path, config: DualInterfaceConfig) -> Dict[str, Path]:
    """Analyze one case and write auditable plot data and figures."""

    config = DualInterfaceConfig(**{**config.__dict__, "nominal_radius_a_A": case.nominal_radius_a_A, "nominal_radius_b_A": case.nominal_radius_b_A})
    config.validate()
    if not case.trace_csv.exists():
        raise FileNotFoundError(case.trace_csv)
    trace_lookup = _trace_rows(case.trace_csv, config)
    if not trace_lookup:
        raise ValueError(f"No trace frames match requested gap windows for {case.case_label}")
    sample_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    seen_trace_keys = set()
    for segment_label, trajectory in case.segments:
        if not trajectory.exists():
            raise FileNotFoundError(trajectory)
        for frame in iter_lammps_dump_records(trajectory):
            key = (segment_label, frame.frame_index)
            trace = trace_lookup.get(key)
            if trace is None:
                continue
            seen_trace_keys.add(key)
            needed = {config.oxygen_type, config.hydrogen_type}
            if case.has_tio2:
                needed.add(config.titanium_type)
            positions = _positions_by_type(frame, needed)
            oxygen_ids, oxygen = positions[config.oxygen_type]
            hydrogen_ids, hydrogen = positions[config.hydrogen_type]
            center_a = np.asarray([float(trace[f"bubble_A_center_{dim}_A"]) for dim in "xyz"])
            center_b = np.asarray([float(trace[f"bubble_B_center_{dim}_A"]) for dim in "xyz"])
            midpoint, e_s, e_u, e_z = local_basis(center_a, center_b, frame.bounds)
            lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
            delta_o = minimum_image_vectors(oxygen - midpoint, lengths)
            s = delta_o @ e_s
            u = delta_o @ e_u
            z = delta_o @ e_z
            rho = np.sqrt(u * u + z * z)
            in_map = (
                (s >= config.s_min_A)
                & (s < config.s_max_A)
                & (z >= config.z_min_A)
                & (z < config.z_max_A)
                & (np.abs(u) < config.transverse_half_width_A)
            )
            assignments, status = assign_intact_waters(oxygen, hydrogen, frame.bounds, config.oh_cutoff_A)
            surface = math.nan
            if case.has_tio2:
                _, titanium = positions[config.titanium_type]
                delta_ti = minimum_image_vectors(titanium - midpoint, lengths)
                surface = float(np.percentile(delta_ti @ e_z, config.surface_quantile))
            valid = 0
            for oxygen_index in np.where(in_map)[0]:
                h1_index, h2_index = assignments[int(oxygen_index)]
                if h1_index < 0:
                    continue
                h_vectors = minimum_image_vectors(
                    hydrogen[[h1_index, h2_index]] - oxygen[oxygen_index], lengths
                )
                mu = np.mean(h_vectors, axis=0)
                mu_norm = float(np.linalg.norm(mu))
                if mu_norm <= 1e-12:
                    continue
                mu /= mu_norm
                delta_a = minimum_image_vectors(oxygen[oxygen_index] - center_a, lengths)
                delta_b = minimum_image_vectors(oxygen[oxygen_index] - center_b, lengths)
                sample_rows.append(
                    {
                        "case_label": case.case_label,
                        "segment": segment_label,
                        "local_frame": frame.frame_index,
                        "timestep": frame.timestep,
                        "time_ns": float(trace["time_ns"]),
                        "state": trace["state"],
                        "gap_window": trace["gap_window"],
                        "nominal_gap_A": float(trace["nominal_gap_A"]),
                        "oxygen_id": int(oxygen_ids[oxygen_index]),
                        "hydrogen1_id": int(hydrogen_ids[h1_index]),
                        "hydrogen2_id": int(hydrogen_ids[h2_index]),
                        "s_A": float(s[oxygen_index]),
                        "z_mid_A": float(z[oxygen_index]),
                        "u_A": float(u[oxygen_index]),
                        "rho_A": float(rho[oxygen_index]),
                        "mu_s": float(np.dot(mu, e_s)),
                        "mu_z": float(np.dot(mu, e_z)),
                        "mu_u": float(np.dot(mu, e_u)),
                        "cos_theta_A": float(np.dot(mu, delta_a / np.linalg.norm(delta_a))),
                        "cos_theta_B": float(np.dot(mu, delta_b / np.linalg.norm(delta_b))),
                    }
                )
                valid += 1
            candidate_status = status[in_map]
            frame_rows.append(
                {
                    "case_label": case.case_label,
                    "segment": segment_label,
                    "local_frame": frame.frame_index,
                    "timestep": frame.timestep,
                    "time_ns": float(trace["time_ns"]),
                    "state": trace["state"],
                    "gap_window": trace["gap_window"],
                    "nominal_gap_A": float(trace["nominal_gap_A"]),
                    "n_oxygen_candidates": int(np.count_nonzero(in_map)),
                    "n_valid_intact_water": valid,
                    "n_non_twofold_oxygen": int(np.count_nonzero(candidate_status != 2)),
                    "n_shared_h_rejected": int(np.count_nonzero(candidate_status == -1)),
                    "surface_z_mid_A": surface,
                }
            )
    missing = sorted(set(trace_lookup) - seen_trace_keys)
    if missing:
        raise ValueError(f"{len(missing)} selected trace frames were not found in configured trajectories; first={missing[0]}")
    output_dir = Path(output_dir)
    outputs = {
        "samples": output_dir / "water_orientation_samples.csv.gz",
        "frames": output_dir / "frame_summary.csv",
        "sz_map": output_dir / "water_orientation_sz_map.csv",
        "sr_map": output_dir / "water_orientation_sr_map.csv",
        "axial_profile": output_dir / "water_orientation_axial_profile.csv",
        "figure": output_dir / "dual_interface_water_sz_map.png",
        "statistics": output_dir / "state_statistics.csv",
        "manifest": output_dir / "artifact_manifest.csv",
    }
    sz_rows = build_sz_map(sample_rows, frame_rows, config)
    _write_csv(outputs["samples"], sample_rows)
    _write_csv(outputs["frames"], frame_rows)
    _write_csv(outputs["sz_map"], sz_rows)
    _write_csv(outputs["sr_map"], build_sr_map(sample_rows, frame_rows, config))
    _write_csv(outputs["axial_profile"], build_axial_profile(sample_rows, config))
    _plot_case_map(case.case_label, sz_rows, outputs["figure"], config)
    stats = [
        {"metric": "case_label", "value": case.case_label},
        {"metric": "trace_csv", "value": str(case.trace_csv)},
        {"metric": "selected_frames", "value": len(frame_rows)},
        {"metric": "valid_water_samples", "value": len(sample_rows)},
        {"metric": "gap_windows_A", "value": ";".join(gap_window_label(item) for item in config.gap_windows_A)},
        {"metric": "coordinate_frame", "value": "s=bubble_axis; z_mid=orthogonalized_TiO2_normal; u=transverse"},
        {"metric": "water_definition", "value": f"O with exactly two unique H within {config.oh_cutoff_A:g} A"},
        {"metric": "block_ns", "value": config.block_ns},
        {"metric": "bootstrap_samples", "value": config.bootstrap_samples},
    ]
    _write_csv(outputs["statistics"], stats)
    manifest_rows = []
    for name, path in outputs.items():
        if name == "manifest":
            continue
        manifest_rows.append(
            {
                "artifact": name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_csv(outputs["manifest"], manifest_rows)
    return outputs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def assemble_figures(case_manifest: Path, output_root: Path, figure_dir: Path, config: DualInterfaceConfig) -> Dict[str, Path]:
    """Assemble the multicase density/dipole atlas and axial profiles."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(case_manifest).open(newline="") as handle:
        case_rows = list(csv.DictReader(handle))
    cases = [str(row["case_label"]) for row in case_rows]
    maps = {case: _read_csv(Path(output_root) / case / "water_orientation_sz_map.csv") for case in cases}
    profiles = {case: _read_csv(Path(output_root) / case / "water_orientation_axial_profile.csv") for case in cases}
    plot_data_dir = Path(figure_dir) / "plot_data"
    combined_maps = [{"case_label": case, **row} for case in cases for row in maps[case]]
    combined_profiles = [{"case_label": case, **row} for case in cases for row in profiles[case]]
    map_plot_data = plot_data_dir / "multicase_water_orientation_sz_map.csv"
    profile_plot_data = plot_data_dir / "multicase_water_orientation_axial_profile.csv"
    _write_csv(map_plot_data, combined_maps)
    _write_csv(profile_plot_data, combined_profiles)
    windows = [gap_window_label(item) for item in config.gap_windows_A]
    densities = np.asarray(
        [float(row["water_number_density_A3"]) for case in cases for row in maps[case]], dtype=float
    )
    vmax = float(np.nanpercentile(densities, 99.0))
    fig, axes = plt.subplots(len(windows), len(cases), figsize=(3.2 * len(cases), 3.05 * len(windows)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    image = None
    for row_index, window in enumerate(windows):
        for col_index, case in enumerate(cases):
            ax = axes[row_index, col_index]
            chunk = [row for row in maps[case] if row["gap_window"] == window]
            s_values = sorted({float(row["s_center_A"]) for row in chunk})
            z_values = sorted({float(row["z_center_A"]) for row in chunk})
            lookup = {(float(row["s_center_A"]), float(row["z_center_A"])): row for row in chunk}
            density = np.asarray([[float(lookup[(s, z)]["water_number_density_A3"]) for s in s_values] for z in z_values])
            mu_s = np.asarray([[float(lookup[(s, z)]["mu_s_mean"]) for s in s_values] for z in z_values])
            mu_z = np.asarray([[float(lookup[(s, z)]["mu_z_mean"]) for s in s_values] for z in z_values])
            count = np.asarray([[int(lookup[(s, z)]["count"]) for s in s_values] for z in z_values])
            image = ax.pcolormesh(s_values, z_values, density, shading="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
            ss, zz = np.meshgrid(s_values, z_values)
            mask = count >= config.min_bin_count
            ax.quiver(
                ss[::4, ::4], zz[::4, ::4], np.where(mask, mu_s, np.nan)[::4, ::4], np.where(mask, mu_z, np.nan)[::4, ::4],
                color="#b2182b", pivot="mid", scale=4.0, width=0.005,
            )
            surface = [float(item["surface_z_mid_A_mean"]) for item in chunk if item["surface_z_mid_A_mean"]]
            if surface and np.isfinite(surface).any():
                ax.axhline(float(np.nanmean(surface)), color="black", linestyle="--", linewidth=0.8)
            ax.axvline(0.0, color="0.45", linewidth=0.5)
            if row_index == 0:
                ax.set_title(case)
            if col_index == 0:
                ax.set_ylabel(f"$h={window[:-1]}$ Å\n$z_{{mid}}$ (Å)")
            if row_index == len(windows) - 1:
                ax.set_xlabel("$s$ (Å)")
    if image is not None:
        colorbar_axis = fig.add_axes([0.925, 0.16, 0.012, 0.68])
        fig.colorbar(image, cax=colorbar_axis, label="Water number density (Å$^{-3}$)")
    figure_dir.mkdir(parents=True, exist_ok=True)
    atlas = figure_dir / "candidate_dual_interface_water_polarization.png"
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.09, top=0.92, wspace=0.08, hspace=0.12)
    fig.savefig(atlas, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(windows), figsize=(6.0 * len(windows), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(cases)))
    for ax, window in zip(axes, windows):
        for color, case in zip(colors, cases):
            rows = [row for row in profiles[case] if row["gap_window"] == window]
            x = np.asarray([float(row["s_center_A"]) for row in rows])
            mean = np.asarray([float(row["mu_s_mean"]) if row["mu_s_mean"] else np.nan for row in rows])
            low = np.asarray([float(row["mu_s_ci95_low"]) if row["mu_s_ci95_low"] else np.nan for row in rows])
            high = np.asarray([float(row["mu_s_ci95_high"]) if row["mu_s_ci95_high"] else np.nan for row in rows])
            ax.plot(x, mean, label=case, color=color, linewidth=1.5)
            ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.axhline(0.0, color="0.4", linewidth=0.7)
        ax.axvline(0.0, color="0.4", linewidth=0.7)
        ax.set_title(f"$h={window[:-1]}$ Å")
        ax.set_xlabel("$s$ (Å)")
    axes[0].set_ylabel(r"Mean axial dipole component $\langle\mu_s\rangle$")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    profile_figure = figure_dir / "candidate_dual_interface_axial_polarization.png"
    fig.tight_layout()
    fig.savefig(profile_figure, dpi=300)
    plt.close(fig)
    return {
        "atlas": atlas,
        "axial_profile": profile_figure,
        "atlas_plot_data": map_plot_data,
        "axial_profile_plot_data": profile_plot_data,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched-gap dual-interface water polarization analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze-case", help="Analyze one case selected from a CSV manifest")
    analyze.add_argument("--case-manifest", type=Path, required=True)
    group = analyze.add_mutually_exclusive_group()
    group.add_argument("--case-index", type=int)
    group.add_argument("--case-label")
    analyze.add_argument("--output-root", type=Path, required=True)
    assemble = subparsers.add_parser("assemble", help="Assemble completed case outputs")
    assemble.add_argument("--case-manifest", type=Path, required=True)
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--figure-dir", type=Path, required=True)
    for target in (analyze, assemble):
        target.add_argument("--gap-window", action="append", default=[])
        target.add_argument("--block-ns", type=float, default=0.020)
        target.add_argument("--bootstrap-samples", type=int, default=1000)
        target.add_argument("--min-bin-count", type=int, default=10)
    analyze.add_argument("--oh-cutoff-A", type=float, default=1.25)
    analyze.add_argument("--max-frames-per-window", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    windows = tuple(parse_gap_window(item) for item in args.gap_window) or ((4.0, 6.0), (12.0, 14.0))
    config = DualInterfaceConfig(
        gap_windows_A=windows,
        block_ns=args.block_ns,
        bootstrap_samples=args.bootstrap_samples,
        min_bin_count=args.min_bin_count,
        oh_cutoff_A=getattr(args, "oh_cutoff_A", 1.25),
        max_frames_per_window=getattr(args, "max_frames_per_window", 0),
    )
    if args.command == "analyze-case":
        case = read_case_manifest(args.case_manifest, case_index=args.case_index, case_label=args.case_label)
        outputs = analyze_case(case, Path(args.output_root) / case.case_label, config)
    else:
        outputs = assemble_figures(args.case_manifest, args.output_root, args.figure_dir, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
