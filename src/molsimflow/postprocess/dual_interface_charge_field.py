"""Charge-source and electric-field proxies for approaching nanobubble pairs.

The Deep-Potential trajectories do not contain electronic or dynamically
polarized atomic charges.  This module therefore constructs an explicit,
auditable proxy from (i) fixed partial charges on intact water molecules,
(ii) formal charges of coordination defects and salt ions, and (iii) formal
surface-protonation markers.  The charge density is deposited on the full
periodic simulation cell and Poisson's equation is solved by FFT.  Reported
fields are structural descriptors, not exact electrostatic observables or
unbiased kinetic trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records, minimum_image_vectors
from molsimflow.postprocess.dual_interface_water import _positions_by_type, local_basis


COULOMB_V_A_PER_E = 14.399645478
EPS0_F_PER_M = 8.8541878128e-12
SPCE_Q_O = -0.8476
SPCE_Q_H = 0.4238
TIP3P_Q_O = -0.834
TIP3P_Q_H = 0.417
WATER_MODELS = {
    "SPC/E": (SPCE_Q_O, SPCE_Q_H),
    "TIP3P": (TIP3P_Q_O, TIP3P_Q_H),
}
SOURCE_GROUPS = ("water", "mobile_ions", "oxide_defects")
FIELD_SOURCES = (*SOURCE_GROUPS, "combined")
FIELD_METRICS = (
    "film_Es_V_A",
    "film_Ez_V_A",
    "film_field_magnitude_V_A",
    "inner_normal_field_V_A",
    "outer_normal_field_V_A",
    "inner_outer_normal_contrast_V_A",
    "tio2_normal_field_V_A",
)


@dataclass(frozen=True)
class ChargeFieldConfig:
    """Numerical and physical definitions for one analysis run."""

    gap_min_A: float = 0.0
    gap_max_A: float = 18.0
    gap_bin_width_A: float = 2.0
    oh_cutoff_A: float = 1.35
    max_free_protons: int = 8
    surface_ti_cutoff_A: float = 3.5
    grid_spacing_A: float = 2.0
    smoothing_sigma_A: float = 2.0
    probe_offset_A: float = 2.0
    cap_cosine: float = 0.5
    surface_probe_count: int = 384
    film_probe_radius_A: float = 4.0
    film_probe_step_A: float = 2.0
    block_ns: float = 0.020
    bootstrap_samples: int = 2000
    random_seed: int = 20260823
    max_frames_per_gap: int = 0
    map_s_min_A: float = -48.0
    map_s_max_A: float = 48.0
    map_z_min_A: float = -26.0
    map_z_max_A: float = 28.0
    map_step_A: float = 2.0
    map_u_half_width_A: float = 20.0
    map_u_step_A: float = 2.0
    max_map_frames_per_window: int = 80
    near_window_A: tuple[float, float] = (2.0, 6.0)
    wide_window_A: tuple[float, float] = (10.0, 14.0)

    def validate(self) -> None:
        if self.gap_max_A <= self.gap_min_A or self.gap_bin_width_A <= 0:
            raise ValueError("Invalid gap range or bin width")
        if min(self.oh_cutoff_A, self.surface_ti_cutoff_A, self.grid_spacing_A) <= 0:
            raise ValueError("Topology and grid cutoffs must be positive")
        if self.smoothing_sigma_A <= 0 or self.probe_offset_A <= 0:
            raise ValueError("Smoothing and probe offsets must be positive")
        if min(self.film_probe_radius_A, self.film_probe_step_A) <= 0:
            raise ValueError("Film-probe dimensions must be positive")
        if min(self.map_u_half_width_A, self.map_u_step_A) <= 0:
            raise ValueError("Map projection dimensions must be positive")
        if not 0.0 < self.cap_cosine < 1.0:
            raise ValueError("cap_cosine must lie between zero and one")
        if self.surface_probe_count < 24 or self.block_ns <= 0:
            raise ValueError("Too few surface probes or invalid block duration")


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-charge-field requires pandas") from exc
    return pd


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["case_label"] for row in rows}) != len(rows):
        raise ValueError("Case manifest must contain unique case_label rows")
    return rows


def _manifest_case(path: Path, index: int) -> dict[str, str]:
    rows = _read_manifest(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Case index {index} is outside 0..{len(rows) - 1}")
    return rows[index]


def parse_segment_specs(raw: str) -> tuple[tuple[str, Path], ...]:
    output = []
    for item in str(raw).split(";"):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        output.append((label.strip(), Path(value.strip()).expanduser()))
    if not output:
        raise ValueError("No trajectory segments were configured")
    return tuple(output)


def _gap_label(value: float, width: float) -> str:
    left = math.floor(value / width) * width
    return f"{left:g}-{left + width:g}A"


def _map_window(value: float, config: ChargeFieldConfig) -> Optional[str]:
    if config.near_window_A[0] <= value < config.near_window_A[1]:
        return "near_2-6A"
    if config.wide_window_A[0] <= value < config.wide_window_A[1]:
        return "wide_10-14A"
    return None


def _select_trace_rows(case: Mapping[str, str], config: ChargeFieldConfig):
    pd = _pandas()
    trace = pd.read_csv(case["trace_csv"])
    required = {
        "segment",
        "local_frame",
        "time_ns",
        "bubble_center_distance_A",
        "bubble_A_center_x_A",
        "bubble_A_center_y_A",
        "bubble_A_center_z_A",
        "bubble_B_center_x_A",
        "bubble_B_center_y_A",
        "bubble_B_center_z_A",
    }
    missing = required.difference(trace.columns)
    if missing:
        raise ValueError(f"Missing trace columns: {sorted(missing)}")
    radius_a = float(case["nominal_radius_a_A"])
    radius_b = float(case["nominal_radius_b_A"])
    trace["gap_A"] = trace.bubble_center_distance_A.astype(float) - radius_a - radius_b
    trace = trace[(trace.gap_A >= config.gap_min_A) & (trace.gap_A < config.gap_max_A)].copy()
    trace["gap_bin"] = [_gap_label(value, config.gap_bin_width_A) for value in trace.gap_A]
    trace["local_frame"] = trace.local_frame.astype(int)
    if config.max_frames_per_gap > 0:
        selected = []
        for _, group in trace.groupby("gap_bin", sort=True):
            group = group.sort_values("time_ns")
            n = min(config.max_frames_per_gap, len(group))
            indices = np.linspace(0, len(group) - 1, n, dtype=int)
            selected.append(group.iloc[np.unique(indices)])
        trace = pd.concat(selected, ignore_index=True) if selected else trace.iloc[0:0]
    trace = trace.sort_values("time_ns").drop_duplicates(["segment", "local_frame"])
    if trace.empty:
        raise ValueError(f"No trace rows in configured gap range for {case['case_label']}")
    return trace


def _nearest_oxygen_assignments(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> tuple[list[list[int]], np.ndarray]:
    """Assign each hydrogen to its nearest oxygen under orthorhombic PBC."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-charge-field requires scipy") from exc
    lengths = bounds[:, 1] - bounds[:, 0]
    oxygen = (np.asarray(oxygen_positions) - bounds[:, 0]) % lengths
    hydrogen = (np.asarray(hydrogen_positions) - bounds[:, 0]) % lengths
    distances, nearest = cKDTree(oxygen, boxsize=lengths).query(
        hydrogen, k=1, distance_upper_bound=float(cutoff_A)
    )
    assignments: list[list[int]] = [[] for _ in range(len(oxygen_positions))]
    valid = np.isfinite(distances) & (nearest < len(oxygen_positions))
    for h_index, o_index in zip(np.where(valid)[0], nearest[valid]):
        assignments[int(o_index)].append(int(h_index))
    return assignments, np.where(~valid)[0]


def classify_charge_sources(frame, has_tio2: bool, config: ChargeFieldConfig):
    """Return complete water, mobile-ion, and oxide-defect charge sources."""

    wanted = {1, 2, 4, 5}
    if has_tio2:
        wanted.add(6)
    positions = _positions_by_type(frame, wanted)
    hydrogen_ids, hydrogen = positions[1]
    oxygen_ids, oxygen = positions[2]
    assignments, unassigned_h_indices = _nearest_oxygen_assignments(
        oxygen, hydrogen, frame.bounds, config.oh_cutoff_A
    )
    if len(unassigned_h_indices) > config.max_free_protons:
        raise ValueError(
            f"Too many free protons for reliable topology: {len(unassigned_h_indices)}"
        )
    coordination = np.asarray([len(values) for values in assignments], dtype=int)

    if has_tio2:
        titanium_ids, titanium = positions[6]
        lattice_oxygen = oxygen_ids < titanium_ids.min()
        solution_oxygen = oxygen_ids > titanium_ids.max()
        if np.count_nonzero(~(lattice_oxygen | solution_oxygen)):
            raise ValueError("O/Ti atom ordering does not separate lattice and solution oxygen")
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("dual-interface-charge-field requires scipy") from exc
        lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
        o_wrapped = (oxygen - frame.bounds[:, 0]) % lengths
        ti_wrapped = (titanium - frame.bounds[:, 0]) % lengths
        ti_distance, _ = cKDTree(ti_wrapped, boxsize=lengths).query(o_wrapped, k=1)
    else:
        lattice_oxygen = np.zeros(len(oxygen), dtype=bool)
        solution_oxygen = np.ones(len(oxygen), dtype=bool)
        ti_distance = np.full(len(oxygen), np.inf)

    water_indices = np.where(solution_oxygen & (coordination == 2))[0]
    water_o = oxygen[water_indices]
    water_h_indices = [h for o_index in water_indices for h in assignments[int(o_index)]]
    water_h = hydrogen[np.asarray(water_h_indices, dtype=int)] if water_h_indices else np.empty((0, 3))
    water_positions = np.vstack([water_o, water_h])
    water_charges = np.concatenate(
        [
            np.full(len(water_o), SPCE_Q_O),
            np.full(len(water_h), SPCE_Q_H),
        ]
    )

    nonwater_solution = np.where(solution_oxygen & (coordination != 2))[0]
    formal_q = coordination[nonwater_solution].astype(float) - 2.0
    oxide_associated = ti_distance[nonwater_solution] <= config.surface_ti_cutoff_A
    mobile_o = nonwater_solution[~oxide_associated]
    oxide_o = nonwater_solution[oxide_associated]
    mobile_q = formal_q[~oxide_associated]
    oxide_q = formal_q[oxide_associated]

    na_positions = positions.get(4, (np.empty(0, dtype=int), np.empty((0, 3))))[1]
    cl_positions = positions.get(5, (np.empty(0, dtype=int), np.empty((0, 3))))[1]
    free_h = hydrogen[np.asarray(unassigned_h_indices, dtype=int)]
    mobile_positions = np.vstack([oxygen[mobile_o], free_h, na_positions, cl_positions])
    mobile_charges = np.concatenate(
        [
            mobile_q,
            np.ones(len(free_h)),
            np.ones(len(na_positions)),
            -np.ones(len(cl_positions)),
        ]
    )

    lattice_h_indices = [
        h_index
        for o_index in np.where(lattice_oxygen)[0]
        for h_index in assignments[int(o_index)]
    ]
    lattice_h = hydrogen[np.asarray(lattice_h_indices, dtype=int)] if lattice_h_indices else np.empty((0, 3))
    oxide_positions = np.vstack([oxygen[oxide_o], lattice_h])
    oxide_charges = np.concatenate([oxide_q, np.ones(len(lattice_h))])

    sources = {
        "water": (water_positions, water_charges),
        "mobile_ions": (mobile_positions, mobile_charges),
        "oxide_defects": (oxide_positions, oxide_charges),
    }
    counts = {
        "n_intact_water": int(len(water_indices)),
        "n_H3O_plus": int(np.count_nonzero(solution_oxygen & (coordination == 3))),
        "n_OH_minus_total": int(np.count_nonzero(solution_oxygen & (coordination == 1))),
        "n_solution_coord0": int(np.count_nonzero(solution_oxygen & (coordination == 0))),
        "n_solution_coord4plus": int(np.count_nonzero(solution_oxygen & (coordination >= 4))),
        "n_oxide_associated_nonwater_O": int(len(oxide_o)),
        "n_H_plus_surf": int(len(lattice_h)),
        "n_H_plus_free": int(len(free_h)),
        "n_Na_plus": int(len(na_positions)),
        "n_Cl_minus": int(len(cl_positions)),
        "n_unassigned_H": int(len(unassigned_h_indices)),
        "formal_net_charge_e": float(sum(charges.sum() for _, charges in sources.values())),
        "water_net_charge_e": float(water_charges.sum()),
        "mobile_net_charge_e": float(mobile_charges.sum()),
        "oxide_defect_net_charge_e": float(oxide_charges.sum()),
    }
    if not math.isclose(counts["water_net_charge_e"], 0.0, abs_tol=1e-9):
        raise ValueError("Fixed-charge water assignment is not neutral")
    if not math.isclose(counts["formal_net_charge_e"], 0.0, abs_tol=1e-9):
        raise ValueError(f"Formal charge conservation failed: {counts['formal_net_charge_e']}")
    return sources, counts


def grid_shape(lengths_A: np.ndarray, spacing_A: float) -> tuple[int, int, int]:
    return tuple(max(8, int(math.ceil(float(length) / spacing_A))) for length in lengths_A)


def deposit_cic(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    bounds_A: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Cloud-in-cell charge deposition with exact PBC charge conservation."""

    grid = np.zeros(shape, dtype=float)
    if len(positions_A) == 0:
        return grid
    lengths = bounds_A[:, 1] - bounds_A[:, 0]
    fractional = ((positions_A - bounds_A[:, 0]) / lengths) % 1.0
    coordinates = fractional * np.asarray(shape)
    lower = np.floor(coordinates).astype(int)
    delta = coordinates - lower
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (delta[:, 0] if dx else 1.0 - delta[:, 0])
                    * (delta[:, 1] if dy else 1.0 - delta[:, 1])
                    * (delta[:, 2] if dz else 1.0 - delta[:, 2])
                )
                indices = (lower + np.asarray([dx, dy, dz])) % np.asarray(shape)
                np.add.at(grid, (indices[:, 0], indices[:, 1], indices[:, 2]), charges_e * weight)
    if not math.isclose(float(grid.sum()), float(np.sum(charges_e)), abs_tol=1e-8):
        raise ValueError("CIC deposition did not conserve charge")
    return grid


def solve_periodic_field(
    charge_grid_e: np.ndarray,
    lengths_A: np.ndarray,
    smoothing_sigma_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve periodic Poisson equation and return smoothed rho and E in V/A."""

    shape = np.asarray(charge_grid_e.shape)
    spacing = lengths_A / shape
    cell_volume = float(np.prod(spacing))
    rho = np.asarray(charge_grid_e, dtype=float) / cell_volume
    rho_k = np.fft.fftn(rho)
    k_axes = [2.0 * math.pi * np.fft.fftfreq(int(n), d=float(dx)) for n, dx in zip(shape, spacing)]
    kx, ky, kz = np.meshgrid(*k_axes, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    gaussian = np.exp(-0.5 * smoothing_sigma_A**2 * k2)
    phi_k = np.zeros_like(rho_k, dtype=complex)
    nonzero = k2 > 0
    phi_k[nonzero] = 4.0 * math.pi * COULOMB_V_A_PER_E * rho_k[nonzero] * gaussian[nonzero] / k2[nonzero]
    fields = tuple(np.fft.ifftn(-1j * axis * phi_k).real for axis in (kx, ky, kz))
    rho_smooth = np.fft.ifftn(rho_k * gaussian).real
    return rho_smooth, fields[0], fields[1], fields[2]


def interpolate_periodic(
    values: np.ndarray,
    points_A: np.ndarray,
    bounds_A: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation on a cell-centered periodic grid."""

    if len(points_A) == 0:
        return np.empty(0)
    shape = np.asarray(values.shape)
    lengths = bounds_A[:, 1] - bounds_A[:, 0]
    coordinates = (((points_A - bounds_A[:, 0]) / lengths) % 1.0) * shape
    lower = np.floor(coordinates).astype(int)
    delta = coordinates - lower
    result = np.zeros(len(points_A), dtype=float)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (delta[:, 0] if dx else 1.0 - delta[:, 0])
                    * (delta[:, 1] if dy else 1.0 - delta[:, 1])
                    * (delta[:, 2] if dz else 1.0 - delta[:, 2])
                )
                indices = (lower + np.asarray([dx, dy, dz])) % shape
                result += values[indices[:, 0], indices[:, 1], indices[:, 2]] * weight
    return result


def _fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = indices * math.pi * (3.0 - math.sqrt(5.0))
    # Local coordinate order is s, u, z.
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def _lab_vectors(local: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    e_s, e_u, e_z = basis
    return local[:, [0]] * e_s + local[:, [1]] * e_u + local[:, [2]] * e_z


def _sample_geometry(trace, frame, case, config: ChargeFieldConfig, include_map: bool = False):
    center_a = np.asarray([float(trace[f"bubble_A_center_{dim}_A"]) for dim in "xyz"])
    center_b = np.asarray([float(trace[f"bubble_B_center_{dim}_A"]) for dim in "xyz"])
    midpoint, e_s, e_u, e_z = local_basis(center_a, center_b, frame.bounds)
    basis = (e_s, e_u, e_z)
    directions_local = _fibonacci_sphere(config.surface_probe_count)
    directions_lab = _lab_vectors(directions_local, basis)
    radius_a = float(case["nominal_radius_a_A"])
    radius_b = float(case["nominal_radius_b_A"])
    points_a = center_a + (radius_a + config.probe_offset_A) * directions_lab
    points_b = center_b + (radius_b + config.probe_offset_A) * directions_lab
    surface_points = np.vstack([points_a, points_b])
    surface_normals = np.vstack([directions_lab, directions_lab])
    inner = np.concatenate(
        [directions_local[:, 0] >= config.cap_cosine, directions_local[:, 0] <= -config.cap_cosine]
    )
    outer = np.concatenate(
        [directions_local[:, 0] <= -config.cap_cosine, directions_local[:, 0] >= config.cap_cosine]
    )
    surface_z = math.nan
    if str(case["has_tio2"]).lower() in {"1", "true", "yes"}:
        _, titanium = _positions_by_type(frame, {6})[6]
        lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
        delta_ti = minimum_image_vectors(titanium - midpoint, lengths)
        surface_z = float(np.percentile(delta_ti @ e_z, 95.0))
        delta_surface = minimum_image_vectors(surface_points - midpoint, lengths)
        accessible = delta_surface @ e_z >= surface_z + 0.5
        inner &= accessible
        outer &= accessible

    film_axis = np.arange(
        -config.film_probe_radius_A,
        config.film_probe_radius_A + 0.5 * config.film_probe_step_A,
        config.film_probe_step_A,
    )
    uu, zz = np.meshgrid(film_axis, film_axis, indexing="xy")
    film_local = np.column_stack([np.zeros(uu.size), uu.ravel(), zz.ravel()])
    film_local = film_local[np.linalg.norm(film_local[:, 1:], axis=1) <= config.film_probe_radius_A + 1e-9]
    if math.isfinite(surface_z):
        film_local = film_local[film_local[:, 2] >= surface_z + 0.5]
    if not len(film_local):
        film_local = np.zeros((1, 3))
    probes = list(midpoint + _lab_vectors(film_local, basis))
    film_probe_count = len(probes)
    if math.isfinite(surface_z):
        probes.append(midpoint + (surface_z + config.probe_offset_A) * e_z)

    map_points = np.empty((0, 3))
    map_local_2d = np.empty((0, 3))
    map_shape = (0, 0, 0)
    map_valid_2d = np.zeros(0, dtype=bool)
    if include_map:
        s_values = np.arange(
            config.map_s_min_A,
            config.map_s_max_A + 0.5 * config.map_step_A,
            config.map_step_A,
        )
        z_values = np.arange(
            config.map_z_min_A,
            config.map_z_max_A + 0.5 * config.map_step_A,
            config.map_step_A,
        )
        u_values = np.arange(
            -config.map_u_half_width_A,
            config.map_u_half_width_A + 0.5 * config.map_u_step_A,
            config.map_u_step_A,
        )
        zz3, uu3, ss3 = np.meshgrid(z_values, u_values, s_values, indexing="ij")
        local_map = np.column_stack([ss3.ravel(), uu3.ravel(), zz3.ravel()])
        map_points = midpoint + _lab_vectors(local_map, basis)
        ss2, zz2 = np.meshgrid(s_values, z_values, indexing="xy")
        map_local_2d = np.column_stack([ss2.ravel(), np.zeros(ss2.size), zz2.ravel()])
        map_valid_2d = np.ones(len(map_local_2d), dtype=bool)
        if math.isfinite(surface_z):
            map_valid_2d &= map_local_2d[:, 2] >= surface_z + 0.5
        map_shape = (len(z_values), len(u_values), len(s_values))
    return {
        "midpoint": midpoint,
        "basis": basis,
        "surface_points": surface_points,
        "surface_normals": surface_normals,
        "inner_mask": inner,
        "outer_mask": outer,
        "surface_z_A": surface_z,
        "map_points": map_points,
        "map_local_2d": map_local_2d,
        "map_shape": map_shape,
        "map_valid_2d": map_valid_2d,
        "probes": np.asarray(probes),
        "film_probe_count": film_probe_count,
    }


def _sample_field(field_grids, points, bounds):
    return np.column_stack([interpolate_periodic(values, points, bounds) for values in field_grids])


def _field_metrics(field_samples, geometry) -> dict[str, float]:
    e_s, _, e_z = geometry["basis"]
    surface_count = len(geometry["surface_points"])
    surface = field_samples[:surface_count]
    probes = field_samples[surface_count:]
    normal = np.einsum("ij,ij->i", surface, geometry["surface_normals"])
    inner = geometry["inner_mask"]
    outer = geometry["outer_mask"]
    film_count = geometry["film_probe_count"]
    film = probes[:film_count].mean(axis=0)
    inner_mean = float(np.mean(normal[inner])) if np.any(inner) else math.nan
    outer_mean = float(np.mean(normal[outer])) if np.any(outer) else math.nan
    return {
        "film_Es_V_A": float(film @ e_s),
        "film_Ez_V_A": float(film @ e_z),
        "film_field_magnitude_V_A": float(np.linalg.norm(film)),
        "inner_normal_field_V_A": inner_mean,
        "outer_normal_field_V_A": outer_mean,
        "inner_outer_normal_contrast_V_A": inner_mean - outer_mean,
        "tio2_normal_field_V_A": (
            float(probes[film_count] @ e_z)
            if math.isfinite(geometry["surface_z_A"])
            else math.nan
        ),
    }


def _bootstrap_mean(values, times, block_ns, samples, seed):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(values) & np.isfinite(times)
    values, times = values[valid], times[valid]
    if not len(values):
        return math.nan, math.nan, math.nan, 0
    blocks = np.floor(times / block_ns + 1e-10).astype(int)
    unique = np.unique(blocks)
    block_means = np.asarray([values[blocks == block].mean() for block in unique])
    if len(block_means) < 2 or samples <= 0:
        return float(values.mean()), math.nan, math.nan, int(len(unique))
    rng = np.random.default_rng(seed)
    draws = rng.choice(block_means, size=(samples, len(block_means)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), int(len(unique))


def summarize_frames(frame, config: ChargeFieldConfig):
    pd = _pandas()
    rows = []
    numeric = [column for column in frame.columns if any(column.startswith(source + "_") for source in FIELD_SOURCES)]
    for gap_bin, group in frame.groupby("gap_bin", sort=True):
        for column in numeric:
            mean, low, high, blocks = _bootstrap_mean(
                group[column], group.time_ns, config.block_ns, config.bootstrap_samples,
                config.random_seed + sum(map(ord, column + str(gap_bin))),
            )
            rows.append(
                {
                    "case_label": str(group.case_label.iloc[0]),
                    "gap_bin": gap_bin,
                    "gap_center_A": float(group.gap_A.mean()),
                    "metric": column,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "frame_count": len(group),
                    "effective_block_count": blocks,
                }
            )
    return pd.DataFrame(rows)


def block_summary(frame, config: ChargeFieldConfig):
    pd = _pandas()
    data = frame.copy()
    data["block_id"] = np.floor(data.time_ns.to_numpy(float) / config.block_ns + 1e-10).astype(int)
    numeric = ["time_ns", "gap_A", *[column for column in data.columns if column.startswith("combined_")]]
    result = data.groupby(["case_label", "gap_bin", "block_id"], as_index=False)[numeric].mean()
    result["frame_count"] = data.groupby(["case_label", "gap_bin", "block_id"]).size().to_numpy()
    return result


def _write_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_case(args) -> int:
    pd = _pandas()
    config = ChargeFieldConfig(
        gap_min_A=args.gap_min_A,
        gap_max_A=args.gap_max_A,
        gap_bin_width_A=args.gap_bin_width_A,
        oh_cutoff_A=args.oh_cutoff_A,
        grid_spacing_A=args.grid_spacing_A,
        smoothing_sigma_A=args.smoothing_sigma_A,
        block_ns=args.block_ns,
        bootstrap_samples=args.bootstrap_samples,
        max_frames_per_gap=args.max_frames_per_gap,
    )
    config.validate()
    case = _manifest_case(Path(args.case_manifest), args.case_index)
    trace = _select_trace_rows(case, config)
    trace_lookup = {
        (str(row.segment), int(row.local_frame)): row._asdict()
        for row in trace.itertuples(index=False)
    }
    segments = parse_segment_specs(case["segment_specs"])
    has_tio2 = str(case["has_tio2"]).lower() in {"1", "true", "yes"}
    frame_rows = []
    seen = set()
    map_accumulator: dict[tuple[str, str, str], np.ndarray] = {}
    map_exposure: dict[str, np.ndarray] = {}
    map_local_reference = None
    make_maps = str(case.get("make_maps", "0")).lower() in {"1", "true", "yes"}
    map_key_window = {}
    if make_maps:
        map_trace = trace.assign(map_window=[_map_window(value, config) for value in trace.gap_A])
        for window, group in map_trace.dropna(subset=["map_window"]).groupby("map_window"):
            count = min(config.max_map_frames_per_window, len(group))
            indices = np.linspace(0, len(group) - 1, count, dtype=int)
            for row in group.sort_values("time_ns").iloc[np.unique(indices)].itertuples():
                map_key_window[(str(row.segment), int(row.local_frame))] = str(window)

    active_segments = {segment for segment, _ in trace_lookup}

    for segment_label, trajectory in segments:
        if segment_label not in active_segments:
            continue
        if not trajectory.exists():
            raise FileNotFoundError(trajectory)
        last_needed_frame = max(
            local_frame for segment, local_frame in trace_lookup if segment == segment_label
        )
        for frame in iter_lammps_dump_records(trajectory):
            if frame.frame_index > last_needed_frame:
                break
            key = (segment_label, frame.frame_index)
            trace_row = trace_lookup.get(key)
            if trace_row is None:
                continue
            seen.add(key)
            sources, counts = classify_charge_sources(frame, has_tio2, config)
            window = map_key_window.get(key)
            geometry = _sample_geometry(
                trace_row,
                frame,
                case,
                config,
                include_map=window is not None,
            )
            lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
            shape = grid_shape(lengths, config.grid_spacing_A)
            all_points = np.vstack(
                [geometry["surface_points"], geometry["probes"], geometry["map_points"]]
            )
            field_by_source = {}
            rho_by_source = {}
            for source in SOURCE_GROUPS:
                positions_A, charges_e = sources[source]
                charge_grid = deposit_cic(positions_A, charges_e, frame.bounds, shape)
                rho, ex, ey, ez = solve_periodic_field(charge_grid, lengths, config.smoothing_sigma_A)
                sampled = _sample_field((ex, ey, ez), all_points, frame.bounds)
                field_by_source[source] = sampled
                rho_by_source[source] = interpolate_periodic(
                    rho, geometry["map_points"], frame.bounds
                )

            field_by_source["combined"] = sum(field_by_source[source] for source in SOURCE_GROUPS)
            rho_by_source["combined"] = sum(rho_by_source[source] for source in SOURCE_GROUPS)
            water_scale = TIP3P_Q_H / SPCE_Q_H
            field_tip3p = water_scale * field_by_source["water"] + field_by_source["mobile_ions"] + field_by_source["oxide_defects"]

            row = {
                "case_label": case["case_label"],
                "family": case["family"],
                "chemistry": case["chemistry"],
                "segment": segment_label,
                "local_frame": frame.frame_index,
                "timestep": frame.timestep,
                "time_ns": float(trace_row["time_ns"]),
                "source_state": str(trace_row.get("state", "")),
                "gap_A": float(trace_row["gap_A"]),
                "gap_bin": str(trace_row["gap_bin"]),
                "surface_z_mid_A": geometry["surface_z_A"],
                "grid_nx": shape[0],
                "grid_ny": shape[1],
                "grid_nz": shape[2],
                **counts,
            }
            surface_count = len(geometry["surface_points"])
            metric_points = slice(0, surface_count + len(geometry["probes"]))
            for source in FIELD_SOURCES:
                for name, value in _field_metrics(field_by_source[source][metric_points], geometry).items():
                    row[f"{source}_{name}"] = value
            tip3p_metrics = _field_metrics(field_tip3p[metric_points], geometry)
            for name, value in tip3p_metrics.items():
                row[f"TIP3P_combined_{name}"] = value
            row["combined_film_maxwell_stress_MPa"] = (
                0.5 * EPS0_F_PER_M * (row["combined_film_field_magnitude_V_A"] * 1.0e10) ** 2 / 1.0e6
            )
            frame_rows.append(row)

            if window:
                map_start = surface_count + len(geometry["probes"])
                valid = geometry["map_valid_2d"]
                map_exposure.setdefault(window, np.zeros(len(valid), dtype=int))[valid] += 1
                e_s, _, e_z = geometry["basis"]
                for source in FIELD_SOURCES:
                    sampled = field_by_source[source][map_start:]
                    values = {
                        "rho_e_A3": rho_by_source[source],
                        "Es_V_A": sampled @ e_s,
                        "Ez_V_A": sampled @ e_z,
                    }
                    for metric, array in values.items():
                        projected = np.asarray(array).reshape(geometry["map_shape"]).mean(axis=1).ravel()
                        map_accumulator.setdefault(
                            (window, source, metric), np.zeros(len(valid), dtype=float)
                        )[valid] += projected[valid]
                map_local_reference = geometry["map_local_2d"]

    missing = sorted(set(trace_lookup) - seen)
    if missing:
        raise ValueError(f"{len(missing)} selected trace frames were not found; first={missing[0]}")
    frames = pd.DataFrame(frame_rows).sort_values("time_ns")
    output = Path(args.output_root) / case["case_label"]
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "charge_field_frame_summary.csv"
    gap_path = output / "charge_field_gap_summary.csv"
    block_path = output / "charge_field_block_summary.csv"
    map_path = output / "charge_field_maps.csv"
    _write_csv(frame_path, frames)
    _write_csv(gap_path, summarize_frames(frames, config))
    _write_csv(block_path, block_summary(frames, config))

    map_rows = []
    if map_local_reference is not None:
        for window, exposure in map_exposure.items():
            for index, local in enumerate(map_local_reference):
                if exposure[index] <= 0:
                    continue
                row = {
                    "case_label": case["case_label"],
                    "window": window,
                    "s_A": local[0],
                    "u_A": local[1],
                    "z_A": local[2],
                    "frame_count": exposure[index],
                }
                for source in FIELD_SOURCES:
                    for metric in ("rho_e_A3", "Es_V_A", "Ez_V_A"):
                        row[f"{source}_{metric}"] = (
                            map_accumulator[(window, source, metric)][index] / exposure[index]
                        )
                map_rows.append(row)
    map_columns = ["case_label", "window", "s_A", "u_A", "z_A", "frame_count"] + [
        f"{source}_{metric}"
        for source in FIELD_SOURCES
        for metric in ("rho_e_A3", "Es_V_A", "Ez_V_A")
    ]
    _write_csv(map_path, pd.DataFrame(map_rows, columns=map_columns))

    source_paths = [Path(case["trace_csv"]), *[path for _, path in segments]]
    source_manifest = pd.DataFrame(
        [
            {
                "case_label": case["case_label"],
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in source_paths
        ]
    )
    _write_csv(output / "source_manifest.csv", source_manifest)
    stats = {
        "case_label": case["case_label"],
        "frame_count": len(frames),
        "time_min_ns": float(frames.time_ns.min()),
        "time_max_ns": float(frames.time_ns.max()),
        "gap_min_A": float(frames.gap_A.min()),
        "gap_max_A": float(frames.gap_A.max()),
        "max_abs_formal_net_charge_e": float(frames.formal_net_charge_e.abs().max()),
        "water_model_primary": "SPC/E",
        "water_model_sensitivity": "TIP3P",
        "water_charges_e": {"SPC/E": [SPCE_Q_O, SPCE_Q_H], "TIP3P": [TIP3P_Q_O, TIP3P_Q_H]},
        "epsilon_r": 1.0,
        "pbc_solver": "orthorhombic FFT Poisson with k=0 removed",
        "grid_spacing_A": config.grid_spacing_A,
        "smoothing_sigma_A": config.smoothing_sigma_A,
        "oh_cutoff_A": config.oh_cutoff_A,
        "max_free_protons": config.max_free_protons,
        "film_probe_definition": (
            f"mid-film disk radius {config.film_probe_radius_A:g} A, "
            f"spacing {config.film_probe_step_A:g} A"
        ),
        "map_projection_definition": (
            f"transverse mean over |u| <= {config.map_u_half_width_A:g} A, "
            f"spacing {config.map_u_step_A:g} A"
        ),
        "map_frame_limit_per_window": config.max_map_frames_per_window,
        "interpretation": "fixed-charge, explicit-water structural field proxy; not dynamic polarization or kinetics",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(stats, indent=2) + "\n")
    artifacts = [frame_path, gap_path, block_path, map_path, output / "analysis_metadata.json"]
    artifact_manifest = pd.DataFrame(
        [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _hash_file(path)}
            for path in artifacts
        ]
    )
    _write_csv(output / "artifact_manifest.csv", artifact_manifest)
    print(
        f"case={case['case_label']} frames={len(frames)} gap={frames.gap_A.min():.3f}:{frames.gap_A.max():.3f} "
        f"max_abs_q={frames.formal_net_charge_e.abs().max():.3g} output={output}"
    )
    return 0


def _case_display(label: str) -> str:
    return label.replace("TiO2", r"TiO$_2$").replace("-", "–")


def _save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def assemble(args) -> int:
    pd = _pandas()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = _read_manifest(Path(args.case_manifest))
    root = Path(args.output_root)
    figure_dir = Path(args.figure_dir)
    plot_dir = figure_dir / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frames = pd.concat(
        [pd.read_csv(root / case["case_label"] / "charge_field_frame_summary.csv") for case in cases],
        ignore_index=True,
    )
    gaps = pd.concat(
        [pd.read_csv(root / case["case_label"] / "charge_field_gap_summary.csv") for case in cases],
        ignore_index=True,
    )
    blocks = pd.concat(
        [pd.read_csv(root / case["case_label"] / "charge_field_block_summary.csv") for case in cases],
        ignore_index=True,
    )
    map_frames = [
        pd.read_csv(root / case["case_label"] / "charge_field_maps.csv") for case in cases
    ]
    maps = pd.concat([frame for frame in map_frames if not frame.empty], ignore_index=True)
    _write_csv(plot_dir / "all_case_frame_metrics.csv", frames)
    _write_csv(plot_dir / "all_case_gap_summary.csv", gaps)
    _write_csv(plot_dir / "all_case_block_summary.csv", blocks)
    _write_csv(plot_dir / "all_case_field_maps.csv", maps)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    palette = {"water": "#2C6BA0", "mobile_ions": "#D28E00", "oxide_defects": "#B23A70", "combined": "#252525"}
    markers = {"S": "o", "A": "s", "S16": "^"}

    # Figure 1: gap-conditioned field evolution for all three geometry families.
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True)
    metrics = [
        ("combined_film_field_magnitude_V_A", "Film-center $|E|$ (V Å$^{-1}$)"),
        ("combined_inner_outer_normal_contrast_V_A", "$E_n^{inner}-E_n^{outer}$ (V Å$^{-1}$)"),
        ("combined_tio2_normal_field_V_A", "TiO$_2$-proximal $E_z$ (V Å$^{-1}$)"),
        ("combined_film_maxwell_stress_MPa", "Film Maxwell-stress proxy (MPa)"),
    ]
    chemistry_colors = {"water": "#6B6B6B", "NaCl": "#2C6BA0", "NaOH": "#D28E00", "HCl": "#B23A70"}
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        subset = gaps[gaps.metric.eq(metric) & gaps.effective_block_count.ge(4)].copy()
        for case in cases:
            group = subset[subset.case_label.eq(case["case_label"])].sort_values("gap_center_A")
            if group.empty:
                continue
            color = chemistry_colors[case["chemistry"]]
            marker = markers[case["family"]]
            ax.plot(group.gap_center_A, group["mean"], color=color, marker=marker, ms=3.5, lw=1.0, alpha=0.9)
            if np.isfinite(group.ci95_low).any():
                ax.fill_between(group.gap_center_A, group.ci95_low, group.ci95_high, color=color, alpha=0.09)
        ax.axhline(0.0, color="#777777", lw=0.6, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("Nominal surface gap $h$ (Å)")
    handles = [Line2D([0], [0], color=color, lw=2, label=name) for name, color in chemistry_colors.items()]
    handles += [
        Line2D([0], [0], color="#333333", marker=markers[fam], lw=0, label=fam)
        for fam in ("S", "A", "S16")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Gap-conditioned electric-field proxies around approaching nanobubbles", y=1.02, fontsize=12)
    fig.text(0.5, 0.96, "SPC/E water charges; full-cell PBC Poisson solver; 20 ps block-bootstrap 95% CI", ha="center", color="#4D4D4D")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, figure_dir / "candidate_field_evolution_SA")
    plt.close(fig)

    # Figure 2: source decomposition in compressed and wide ensembles.
    selected_cases = ["Bulk-water-S", "TiO2-water-S", "TiO2-NaCl-S", "TiO2-NaOH-S", "TiO2-HCl-S"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    windows = [(2.0, 6.0, "Compressed: 2 ≤ h < 6 Å"), (10.0, 14.0, "Wide: 10 ≤ h < 14 Å")]
    summary_config = ChargeFieldConfig()
    for ax, (left, right, title) in zip(axes, windows):
        positions = np.arange(len(selected_cases), dtype=float)
        width = 0.18
        for offset, source in enumerate(FIELD_SOURCES):
            values = []
            errors = []
            for label in selected_cases:
                group = frames[frames.case_label.eq(label) & frames.gap_A.ge(left) & frames.gap_A.lt(right)]
                if len(group):
                    mean, low, high, _ = _bootstrap_mean(
                        group[f"{source}_film_Es_V_A"],
                        group.time_ns,
                        summary_config.block_ns,
                        summary_config.bootstrap_samples,
                        summary_config.random_seed + offset,
                    )
                    values.append(mean)
                    errors.append([mean - low, high - mean] if math.isfinite(low) else [math.nan, math.nan])
                else:
                    values.append(math.nan)
                    errors.append([math.nan, math.nan])
            ax.bar(
                positions + (offset - 1.5) * width,
                values,
                width=width,
                color=palette[source],
                label=source.replace("_", " "),
                yerr=np.asarray(errors).T,
                error_kw={"elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6},
            )
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_xticks(positions, [_case_display(label).replace("–S", "") for label in selected_cases], rotation=24, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    axes[0].set_ylabel("Signed axial field at film center $E_s$ (V Å$^{-1}$)")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle("Water, mobile-ion, and oxide-defect contributions", y=1.02, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "candidate_field_source_decomposition_S")
    plt.close(fig)

    # Figure 3: time-order diagnostic, retaining h as the structural coordinate.
    diagnostic_cases = ["TiO2-NaCl-S", "TiO2-NaOH-S", "TiO2-HCl-S"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), sharey=True)
    scatter = None
    for ax, label in zip(axes, diagnostic_cases):
        group = blocks[blocks.case_label.eq(label)].copy()
        scatter = ax.scatter(
            group.gap_A,
            group.combined_inner_outer_normal_contrast_V_A,
            c=group.time_ns,
            cmap="viridis",
            s=18 + 2 * group.frame_count,
            alpha=0.75,
            edgecolors="none",
        )
        ax.axhline(0, color="#666666", lw=0.6, ls="--")
        ax.set_title(_case_display(label))
        ax.set_xlabel("Nominal gap $h$ (Å)")
        ax.grid(color="#ECECEC", lw=0.5)
    axes[0].set_ylabel("20 ps block $E_n^{inner}-E_n^{outer}$ (V Å$^{-1}$)")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="OPES trajectory time (ns)", shrink=0.85)
    fig.suptitle("Trajectory-order diagnostic for the directional field contrast", y=1.03, fontsize=12)
    fig.text(0.5, 0.96, "Color shows sampling order; it is not interpreted as unbiased coalescence kinetics", ha="center", color="#4D4D4D")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.83, wspace=0.20)
    _save_figure(fig, figure_dir / "candidate_field_time_order_diagnostic")
    plt.close(fig)

    # Figure 4: near/wide HCl-S field maps from full 3D charge sources.
    target = maps[maps.case_label.eq("TiO2-HCl-S")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), sharex=True, sharey=True)
    sources = ("water", "mobile_ions", "combined")
    windows_order = ("wide_10-14A", "near_2-6A")
    max_field = float(np.nanquantile(np.abs(target[[f"{source}_Es_V_A" for source in sources]].to_numpy()), 0.98))
    max_field = max(max_field, 1e-6)
    image = None
    for i, window in enumerate(windows_order):
        for j, source in enumerate(sources):
            ax = axes[i, j]
            group = target[target.window.eq(window)]
            pivot = group.pivot(index="z_A", columns="s_A", values=f"{source}_Es_V_A").sort_index()
            image = ax.imshow(
                pivot.to_numpy(), origin="lower", aspect="auto",
                extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                cmap="coolwarm", vmin=-max_field, vmax=max_field,
            )
            ax.set_title(f"{source.replace('_', ' ')}\n{window.replace('_', ' ')}")
            ax.axhline(-20, color="black", lw=0.7, ls="--")
            if i == 1:
                ax.set_xlabel("Bubble-axis coordinate $s$ (Å)")
            if j == 0:
                ax.set_ylabel("Surface-normal coordinate $z$ (Å)")
    if image is not None:
        fig.colorbar(image, ax=axes, label="$E_s$ (V Å$^{-1}$)", shrink=0.84)
    fig.suptitle("TiO$_2$–HCl–S axial field from full three-dimensional charge sources", y=1.01, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.10, top=0.88, wspace=0.10, hspace=0.25)
    _save_figure(fig, figure_dir / "candidate_HCl_S_full3d_field_maps")
    plt.close(fig)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    max_charge_error = float(frames.formal_net_charge_e.abs().max())
    low_support = gaps[gaps.effective_block_count.lt(4)]
    validation = [
        "# Validation report",
        "",
        f"- Cases: {frames.case_label.nunique()} / {len(cases)}",
        f"- Frames: {len(frames)}",
        f"- Maximum absolute formal net charge: {max_charge_error:.3e} e",
        f"- Gap-summary rows with fewer than four 20 ps blocks: {len(low_support)}",
        f"- Charge conservation: {'PASS' if max_charge_error < 1e-9 else 'FAIL'}",
        "- Water model: SPC/E primary; TIP3P fixed-charge sensitivity retained in frame data.",
        "- Interpretation: PBC fixed-charge structural field proxy; no dynamic polarization and no unbiased kinetic claim.",
    ]
    (report_dir / "VALIDATION.md").write_text("\n".join(validation) + "\n")
    if max_charge_error >= 1e-9:
        raise ValueError("Charge-conservation validation failed")

    artifact_paths = sorted([*figure_dir.glob("candidate_*.png"), *figure_dir.glob("candidate_*.pdf"), *plot_dir.glob("*.csv"), report_dir / "VALIDATION.md"])
    artifact_manifest = pd.DataFrame(
        [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _hash_file(path)}
            for path in artifact_paths
        ]
    )
    _write_csv(Path(args.manifest_dir) / "artifact_manifest.csv", artifact_manifest)
    print(f"assembled cases={frames.case_label.nunique()} frames={len(frames)} figure_dir={figure_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze-case")
    analyze.add_argument("--case-manifest", required=True)
    analyze.add_argument("--case-index", type=int, required=True)
    analyze.add_argument("--output-root", required=True)
    analyze.add_argument("--gap-min-A", type=float, default=0.0)
    analyze.add_argument("--gap-max-A", type=float, default=18.0)
    analyze.add_argument("--gap-bin-width-A", type=float, default=2.0)
    analyze.add_argument("--oh-cutoff-A", type=float, default=1.35)
    analyze.add_argument("--grid-spacing-A", type=float, default=2.0)
    analyze.add_argument("--smoothing-sigma-A", type=float, default=2.0)
    analyze.add_argument("--block-ns", type=float, default=0.020)
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument("--max-frames-per-gap", type=int, default=0)
    analyze.set_defaults(func=analyze_case)

    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--case-manifest", required=True)
    assemble_parser.add_argument("--output-root", required=True)
    assemble_parser.add_argument("--figure-dir", required=True)
    assemble_parser.add_argument("--report-dir", required=True)
    assemble_parser.add_argument("--manifest-dir", required=True)
    assemble_parser.set_defaults(func=assemble)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
