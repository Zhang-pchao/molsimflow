"""Particle flotation analysis for silica-particle/nanobubble trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ATOMIC_MASSES = {
    "H": 1.0079999997,
    "O": 15.9989999959,
    "N": 14.0069999964,
    "Na": 22.9897692741,
    "Cl": 35.4499999909,
    "Ti": 47.8669999877,
    "C": 12.0109999969,
    "Si": 28.0849999928,
}
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660

FRAME_COLUMNS = (
    "source_file",
    "frame_index",
    "step",
    "time_ps",
    "particle_com_x_A",
    "particle_com_y_A",
    "particle_com_z_A",
    "particle_lift_z_A",
    "particle_z_min_A",
    "particle_z_max_A",
    "particle_z_span_A",
    "slab_com_z_A",
    "slab_z_max_A",
    "particle_slab_gap_A",
    "n2_molecule_count",
    "n2_contact_count",
    "n2_lower_contact_count",
    "n2_upper_contact_count",
    "n2_contact_fraction",
    "n2_angular_coverage_fraction",
    "n2_projected_coverage_fraction",
    "n2_min_particle_distance_A",
    "n2_distance_q10_A",
    "n2_distance_q50_A",
    "n2_distance_q90_A",
    "n2_relative_z_mean_A",
    "n2_relative_z_q10_A",
    "n2_relative_z_q90_A",
    "n2_voxel_volume_A3",
    "n2_number_density_per_A3",
    "n2_mass_density_g_cm3",
)
RADIAL_COLUMNS = (
    "step",
    "time_ps",
    "r_bin_min_A",
    "r_bin_max_A",
    "r_bin_center_A",
    "n2_count",
    "shell_volume_A3",
    "number_density_per_A3",
)
FORCE_VELOCITY_COLUMNS = (
    "step",
    "time_ps",
    "particle_force_x",
    "particle_force_y",
    "particle_force_z",
    "particle_force_norm",
    "particle_force_z_per_atom",
    "particle_velocity_x_A_ps",
    "particle_velocity_y_A_ps",
    "particle_velocity_z_A_ps",
    "particle_speed_A_ps",
)


@dataclass(frozen=True)
class AtomRange:
    start: int
    end: int

    def contains(self, atom_id: int) -> bool:
        return self.start <= atom_id <= self.end

    @property
    def count(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class AnalysisSpec:
    slab: AtomRange
    particle: AtomRange
    particle_core: AtomRange
    n2: AtomRange
    type_to_symbol: Mapping[int, str]
    particle_radius_A: float
    timestep_ps: float


@dataclass(frozen=True)
class DumpFrame:
    source_file: Path
    source_frame_index: int
    step: int
    box: np.ndarray
    columns: Tuple[str, ...]
    groups: Mapping[str, Mapping[str, np.ndarray]]


@dataclass
class RunningCenter:
    wrapped: Optional[np.ndarray] = None
    unwrapped: Optional[np.ndarray] = None

    def update(self, wrapped_center: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        if self.wrapped is None or self.unwrapped is None:
            self.wrapped = np.array(wrapped_center, dtype=float)
            self.unwrapped = np.array(wrapped_center, dtype=float)
            return self.unwrapped.copy()
        delta = minimum_image_delta(wrapped_center - self.wrapped, lengths)
        self.unwrapped = self.unwrapped + delta
        self.wrapped = np.array(wrapped_center, dtype=float)
        return self.unwrapped.copy()


@dataclass(frozen=True)
class Snapshot:
    step: int
    time_ps: float
    particle_rel: np.ndarray
    n2_rel: np.ndarray


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _format_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.8g}"
        return "nan"
    return value


def parse_atom_range(text: str) -> AtomRange:
    parts = str(text).replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Atom range must be START:END, got {text!r}")
    start, end = int(parts[0]), int(parts[1])
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError(f"Invalid atom range: {text!r}")
    return AtomRange(start=start, end=end)


def parse_type_map(entries: Optional[Sequence[str]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(f"Type map entries must be TYPE=ELEMENT, got {entry!r}")
        raw_type, symbol = entry.split("=", 1)
        mapping[int(raw_type)] = symbol
    return mapping


def invert_lammps_type_order(order: Mapping[str, object]) -> Dict[int, str]:
    return {int(value): str(key) for key, value in order.items()}


def load_spec_from_summary(
    summary_path: Path,
    *,
    timestep_ps: float,
    type_map: Optional[Mapping[int, str]] = None,
    particle_radius_A: Optional[float] = None,
) -> AnalysisSpec:
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    counts = data.get("counts", {})
    geometry = data.get("geometry", {})
    particle_stats = data.get("particle_stats", {})
    slab_atoms = int(counts["slab_atoms"])
    particle_atoms = int(counts["particle_atoms"])
    particle_core_atoms = int(particle_stats.get("core_atoms", particle_atoms))
    fixed_atoms = int(counts.get("fixed_atoms", slab_atoms + particle_atoms))
    n2_molecules = int(counts["N2"])
    if fixed_atoms != slab_atoms + particle_atoms:
        raise ValueError(
            "Summary counts are not ordered as slab+particle fixed atoms: "
            f"fixed_atoms={fixed_atoms}, slab_atoms={slab_atoms}, particle_atoms={particle_atoms}"
        )
    inferred_type_map = invert_lammps_type_order(data.get("lammps_atom_type_order", {}))
    merged_type_map = dict(inferred_type_map)
    merged_type_map.update(dict(type_map or {}))
    radius = (
        particle_radius_A
        if particle_radius_A is not None
        else _as_float(geometry.get("particle_outer_radius_A"), default=math.nan)
    )
    if not math.isfinite(radius):
        radius = _as_float(geometry.get("particle_core_radius_A"), default=math.nan)
    if not math.isfinite(radius):
        raise ValueError("Could not infer particle radius; pass --particle-radius-A")
    return AnalysisSpec(
        slab=AtomRange(1, slab_atoms),
        particle=AtomRange(slab_atoms + 1, fixed_atoms),
        particle_core=AtomRange(slab_atoms + 1, slab_atoms + particle_core_atoms),
        n2=AtomRange(fixed_atoms + 1, fixed_atoms + 2 * n2_molecules),
        type_to_symbol=merged_type_map,
        particle_radius_A=float(radius),
        timestep_ps=float(timestep_ps),
    )


def load_spec_from_args(args: argparse.Namespace) -> AnalysisSpec:
    type_map = parse_type_map(args.type_map)
    if args.model_summary is not None:
        return load_spec_from_summary(
            args.model_summary,
            timestep_ps=args.timestep_ps,
            type_map=type_map,
            particle_radius_A=args.particle_radius_A,
        )
    if args.slab_range is None or args.particle_range is None or args.n2_range is None:
        raise ValueError(
            "Either --model-summary or all of --slab-range, --particle-range, and --n2-range are required"
        )
    if args.particle_radius_A is None:
        raise ValueError("--particle-radius-A is required when --model-summary is not provided")
    return AnalysisSpec(
        slab=args.slab_range,
        particle=args.particle_range,
        particle_core=args.particle_range,
        n2=args.n2_range,
        type_to_symbol=type_map,
        particle_radius_A=float(args.particle_radius_A),
        timestep_ps=float(args.timestep_ps),
    )


def group_masses(types: np.ndarray, type_to_symbol: Mapping[int, str]) -> np.ndarray:
    masses = []
    for atom_type in types:
        symbol = type_to_symbol.get(int(atom_type))
        masses.append(float(ATOMIC_MASSES.get(str(symbol), 1.0)))
    return np.array(masses, dtype=float)


def box_lengths(box: np.ndarray) -> np.ndarray:
    return box[:, 1] - box[:, 0]


def minimum_image_delta(delta: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.array(delta, dtype=float)
    for axis, length in enumerate(lengths):
        if math.isfinite(float(length)) and length > 0:
            out[..., axis] -= length * np.rint(out[..., axis] / length)
    return out


def circular_group_center(coords: np.ndarray, masses: np.ndarray, box: np.ndarray) -> np.ndarray:
    lengths = box_lengths(box)
    center = np.zeros(3, dtype=float)
    weights = masses / np.sum(masses) if np.sum(masses) > 0 else np.ones(len(coords)) / len(coords)
    for axis, length in enumerate(lengths):
        if not math.isfinite(float(length)) or length <= 0:
            center[axis] = float(np.average(coords[:, axis], weights=weights))
            continue
        values = (coords[:, axis] - box[axis, 0]) / length
        angles = values * 2.0 * math.pi
        sin_mean = float(np.sum(weights * np.sin(angles)))
        cos_mean = float(np.sum(weights * np.cos(angles)))
        angle = math.atan2(sin_mean, cos_mean)
        if angle < 0:
            angle += 2.0 * math.pi
        center[axis] = box[axis, 0] + angle * length / (2.0 * math.pi)
    deltas = minimum_image_delta(coords - center, lengths)
    return center + np.average(deltas, axis=0, weights=weights)


def unwrap_group(
    coords: np.ndarray,
    masses: np.ndarray,
    box: np.ndarray,
    running_center: RunningCenter,
) -> Tuple[np.ndarray, np.ndarray]:
    lengths = box_lengths(box)
    wrapped_center = circular_group_center(coords, masses, box)
    center_unwrapped = running_center.update(wrapped_center, lengths)
    deltas = minimum_image_delta(coords - wrapped_center, lengths)
    return center_unwrapped, center_unwrapped + deltas


def iter_lammps_dump(
    paths: Sequence[Path],
    spec: AnalysisSpec,
    *,
    groups: Sequence[str],
    required_columns: Sequence[str],
) -> Iterator[DumpFrame]:
    group_ranges = {
        "slab": spec.slab,
        "particle": spec.particle,
        "particle_core": spec.particle_core,
        "n2": spec.n2,
    }
    selected_groups = tuple(groups)
    required = tuple(required_columns)
    for path in paths:
        path = Path(path)
        frame_index = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.startswith("ITEM: TIMESTEP"):
                    raise ValueError(f"Expected TIMESTEP in {path}, got {line!r}")
                step = int(handle.readline().strip())
                if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError(f"Expected NUMBER OF ATOMS after step {step} in {path}")
                atom_count = int(handle.readline().strip())
                if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(f"Expected BOX BOUNDS after step {step} in {path}")
                bounds = []
                for _ in range(3):
                    parts = handle.readline().split()
                    bounds.append([float(parts[0]), float(parts[1])])
                atoms_header = handle.readline().strip()
                if not atoms_header.startswith("ITEM: ATOMS"):
                    raise ValueError(f"Expected ATOMS after step {step} in {path}")
                columns = tuple(atoms_header.split()[2:])
                column_index = {name: index for index, name in enumerate(columns)}
                missing = [name for name in ("id", "type", *required) if name not in column_index]
                if missing:
                    raise ValueError(f"{path} step {step} is missing dump columns: {missing}")
                raw: Dict[str, List[List[float]]] = {name: [] for name in selected_groups}
                for _ in range(atom_count):
                    parts = handle.readline().split()
                    atom_id = int(parts[column_index["id"]])
                    for name in selected_groups:
                        atom_range = group_ranges[name]
                        if atom_range.contains(atom_id):
                            row = [
                                float(atom_id),
                                float(parts[column_index["type"]]),
                                *[float(parts[column_index[column]]) for column in required],
                            ]
                            raw[name].append(row)
                            break
                parsed: Dict[str, Mapping[str, np.ndarray]] = {}
                for name, rows in raw.items():
                    if not rows:
                        raise ValueError(f"No atoms selected for group {name!r} in {path} step {step}")
                    array = np.array(rows, dtype=float)
                    order = np.argsort(array[:, 0])
                    array = array[order]
                    parsed[name] = {
                        "ids": array[:, 0].astype(int),
                        "types": array[:, 1].astype(int),
                        "values": array[:, 2:],
                    }
                yield DumpFrame(
                    source_file=path,
                    source_frame_index=frame_index,
                    step=step,
                    box=np.array(bounds, dtype=float),
                    columns=required,
                    groups=parsed,
                )
                frame_index += 1


def compute_n2_molecule_coms(n2_coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    if len(n2_coords) % 2 != 0:
        raise ValueError(f"N2 atom count must be even, got {len(n2_coords)}")
    lengths = box_lengths(box)
    first = n2_coords[0::2]
    second = n2_coords[1::2]
    delta = minimum_image_delta(second - first, lengths)
    return first + 0.5 * delta


def angular_coverage(vectors: np.ndarray, z_bins: int, phi_bins: int) -> float:
    if len(vectors) == 0:
        return 0.0
    radii = np.linalg.norm(vectors, axis=1)
    valid = radii > 1.0e-12
    if not np.any(valid):
        return 0.0
    unit = vectors[valid] / radii[valid, None]
    cos_theta = np.clip(unit[:, 2], -1.0, 1.0)
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * math.pi)
    z_index = np.clip(((cos_theta + 1.0) * 0.5 * z_bins).astype(int), 0, z_bins - 1)
    phi_index = np.clip((phi / (2.0 * math.pi) * phi_bins).astype(int), 0, phi_bins - 1)
    occupied = set(zip(z_index.tolist(), phi_index.tolist()))
    return len(occupied) / float(z_bins * phi_bins)


def occupied_voxel_volume(
    coords: np.ndarray,
    *,
    voxel_A: float,
    probe_radius_A: float,
) -> float:
    if len(coords) == 0:
        return 0.0
    occupied = set()
    radius_cells = max(0, int(math.ceil(probe_radius_A / voxel_A)))
    diagonal_pad = 0.5 * math.sqrt(3.0) * voxel_A
    for coord in coords:
        base = np.floor(coord / voxel_A).astype(int)
        for i in range(base[0] - radius_cells, base[0] + radius_cells + 1):
            for j in range(base[1] - radius_cells, base[1] + radius_cells + 1):
                for k in range(base[2] - radius_cells, base[2] + radius_cells + 1):
                    center = (np.array([i, j, k], dtype=float) + 0.5) * voxel_A
                    if np.linalg.norm(center - coord) <= probe_radius_A + diagonal_pad:
                        occupied.add((int(i), int(j), int(k)))
    return len(occupied) * voxel_A**3


def radial_density_rows(
    distances: np.ndarray,
    *,
    step: int,
    time_ps: float,
    bin_width_A: float,
    r_max_A: float,
) -> List[Dict[str, object]]:
    edges = np.arange(0.0, r_max_A + bin_width_A, bin_width_A)
    if len(edges) < 2:
        edges = np.array([0.0, bin_width_A], dtype=float)
    counts, _ = np.histogram(distances, bins=edges)
    rows = []
    for index, count in enumerate(counts):
        r_min = float(edges[index])
        r_max = float(edges[index + 1])
        volume = 4.0 / 3.0 * math.pi * (r_max**3 - r_min**3)
        rows.append(
            {
                "step": step,
                "time_ps": time_ps,
                "r_bin_min_A": r_min,
                "r_bin_max_A": r_max,
                "r_bin_center_A": 0.5 * (r_min + r_max),
                "n2_count": int(count),
                "shell_volume_A3": volume,
                "number_density_per_A3": int(count) / volume if volume > 0 else math.nan,
            }
        )
    return rows


def process_position_trajectories(
    paths: Sequence[Path],
    spec: AnalysisSpec,
    *,
    start_time_ps: Optional[float],
    end_time_ps: Optional[float],
    max_frames: Optional[int],
    keep_duplicate_steps: bool,
    contact_shell_A: float,
    coverage_probe_radius_A: float,
    coverage_z_bins: int,
    coverage_phi_bins: int,
    voxel_A: float,
    voxel_probe_radius_A: float,
    radial_bin_width_A: float,
    radial_max_A: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Snapshot]]:
    frame_rows: List[Dict[str, object]] = []
    radial_rows: List[Dict[str, object]] = []
    snapshots: List[Snapshot] = []
    seen_steps = set()
    processed = 0
    particle_center_state = RunningCenter()
    slab_center_state = RunningCenter()
    initial_particle_z: Optional[float] = None
    n2_molecule_mass_amu = 2.0 * ATOMIC_MASSES["N"]

    for frame in iter_lammps_dump(
        paths,
        spec,
        groups=("slab", "particle_core", "n2"),
        required_columns=("x", "y", "z"),
    ):
        time_ps = frame.step * spec.timestep_ps
        if start_time_ps is not None and time_ps < start_time_ps:
            continue
        if end_time_ps is not None and time_ps > end_time_ps:
            continue
        if not keep_duplicate_steps and frame.step in seen_steps:
            continue
        seen_steps.add(frame.step)
        if max_frames is not None and processed >= max_frames:
            break

        particle = frame.groups["particle_core"]
        slab = frame.groups["slab"]
        n2 = frame.groups["n2"]
        particle_coords = particle["values"]
        slab_coords = slab["values"]
        particle_masses = group_masses(particle["types"], spec.type_to_symbol)
        slab_masses = group_masses(slab["types"], spec.type_to_symbol)
        particle_center, particle_unwrapped = unwrap_group(
            particle_coords,
            particle_masses,
            frame.box,
            particle_center_state,
        )
        slab_center, slab_unwrapped = unwrap_group(slab_coords, slab_masses, frame.box, slab_center_state)
        if initial_particle_z is None:
            initial_particle_z = float(particle_center[2])

        n2_coms = compute_n2_molecule_coms(n2["values"], frame.box)
        n2_rel = minimum_image_delta(n2_coms - particle_center, box_lengths(frame.box))
        n2_distances = np.linalg.norm(n2_rel, axis=1)
        contact_mask = n2_distances <= spec.particle_radius_A + contact_shell_A
        lower_mask = contact_mask & (n2_rel[:, 2] < 0.0)
        upper_mask = contact_mask & (n2_rel[:, 2] >= 0.0)
        contact_count = int(np.count_nonzero(contact_mask))
        contact_vectors = n2_rel[contact_mask]
        projected_surface_area = 4.0 * math.pi * spec.particle_radius_A**2
        projected_coverage = (
            min(1.0, contact_count * math.pi * coverage_probe_radius_A**2 / projected_surface_area)
            if projected_surface_area > 0
            else math.nan
        )
        voxel_volume = occupied_voxel_volume(
            n2_rel,
            voxel_A=voxel_A,
            probe_radius_A=voxel_probe_radius_A,
        )
        total_n2_mass_amu = len(n2_coms) * n2_molecule_mass_amu
        n2_number_density = len(n2_coms) / voxel_volume if voxel_volume > 0 else math.nan
        n2_mass_density = (
            total_n2_mass_amu * AMU_PER_A3_TO_G_PER_CM3 / voxel_volume if voxel_volume > 0 else math.nan
        )
        particle_z_min = float(np.min(particle_unwrapped[:, 2]))
        particle_z_max = float(np.max(particle_unwrapped[:, 2]))
        slab_z_max = float(np.max(slab_unwrapped[:, 2]))
        row = {
            "source_file": str(frame.source_file),
            "frame_index": processed,
            "step": frame.step,
            "time_ps": time_ps,
            "particle_com_x_A": float(particle_center[0]),
            "particle_com_y_A": float(particle_center[1]),
            "particle_com_z_A": float(particle_center[2]),
            "particle_lift_z_A": float(particle_center[2] - initial_particle_z),
            "particle_z_min_A": particle_z_min,
            "particle_z_max_A": particle_z_max,
            "particle_z_span_A": particle_z_max - particle_z_min,
            "slab_com_z_A": float(slab_center[2]),
            "slab_z_max_A": slab_z_max,
            "particle_slab_gap_A": particle_z_min - slab_z_max,
            "n2_molecule_count": len(n2_coms),
            "n2_contact_count": contact_count,
            "n2_lower_contact_count": int(np.count_nonzero(lower_mask)),
            "n2_upper_contact_count": int(np.count_nonzero(upper_mask)),
            "n2_contact_fraction": contact_count / len(n2_coms) if len(n2_coms) else math.nan,
            "n2_angular_coverage_fraction": angular_coverage(
                contact_vectors,
                coverage_z_bins,
                coverage_phi_bins,
            ),
            "n2_projected_coverage_fraction": projected_coverage,
            "n2_min_particle_distance_A": float(np.min(n2_distances)) if len(n2_distances) else math.nan,
            "n2_distance_q10_A": float(np.quantile(n2_distances, 0.10)) if len(n2_distances) else math.nan,
            "n2_distance_q50_A": float(np.quantile(n2_distances, 0.50)) if len(n2_distances) else math.nan,
            "n2_distance_q90_A": float(np.quantile(n2_distances, 0.90)) if len(n2_distances) else math.nan,
            "n2_relative_z_mean_A": float(np.mean(n2_rel[:, 2])) if len(n2_rel) else math.nan,
            "n2_relative_z_q10_A": float(np.quantile(n2_rel[:, 2], 0.10)) if len(n2_rel) else math.nan,
            "n2_relative_z_q90_A": float(np.quantile(n2_rel[:, 2], 0.90)) if len(n2_rel) else math.nan,
            "n2_voxel_volume_A3": voxel_volume,
            "n2_number_density_per_A3": n2_number_density,
            "n2_mass_density_g_cm3": n2_mass_density,
        }
        frame_rows.append(row)
        radial_rows.extend(
            radial_density_rows(
                n2_distances,
                step=frame.step,
                time_ps=time_ps,
                bin_width_A=radial_bin_width_A,
                r_max_A=radial_max_A,
            )
        )
        snapshots.append(
            Snapshot(
                step=frame.step,
                time_ps=time_ps,
                particle_rel=particle_unwrapped - particle_center,
                n2_rel=n2_rel,
            )
        )
        processed += 1
    return frame_rows, radial_rows, snapshots


def _process_vector_file(
    paths: Sequence[Path],
    spec: AnalysisSpec,
    *,
    required_columns: Sequence[str],
    value_prefix: str,
) -> Dict[int, Dict[str, object]]:
    rows_by_step: Dict[int, Dict[str, object]] = {}
    for frame in iter_lammps_dump(
        paths,
        spec,
        groups=("particle_core",),
        required_columns=required_columns,
    ):
        particle = frame.groups["particle_core"]
        values = particle["values"]
        masses = group_masses(particle["types"], spec.type_to_symbol)
        row = rows_by_step.setdefault(
            frame.step,
            {
                "step": frame.step,
                "time_ps": frame.step * spec.timestep_ps,
            },
        )
        if value_prefix == "force":
            force = np.sum(values, axis=0)
            row["particle_force_x"] = float(force[0])
            row["particle_force_y"] = float(force[1])
            row["particle_force_z"] = float(force[2])
            row["particle_force_norm"] = float(np.linalg.norm(force))
            row["particle_force_z_per_atom"] = float(force[2] / len(values))
        elif value_prefix == "velocity":
            velocity = np.average(values, axis=0, weights=masses)
            row["particle_velocity_x_A_ps"] = float(velocity[0])
            row["particle_velocity_y_A_ps"] = float(velocity[1])
            row["particle_velocity_z_A_ps"] = float(velocity[2])
            row["particle_speed_A_ps"] = float(np.linalg.norm(velocity))
        else:
            raise ValueError(f"Unknown value prefix: {value_prefix}")
    return rows_by_step


def process_force_velocity(
    force_paths: Sequence[Path],
    velocity_paths: Sequence[Path],
    spec: AnalysisSpec,
) -> List[Dict[str, object]]:
    merged: Dict[int, Dict[str, object]] = {}
    if force_paths:
        for step, row in _process_vector_file(
            force_paths,
            spec,
            required_columns=("fx", "fy", "fz"),
            value_prefix="force",
        ).items():
            merged.setdefault(step, {"step": step, "time_ps": step * spec.timestep_ps}).update(row)
    if velocity_paths:
        for step, row in _process_vector_file(
            velocity_paths,
            spec,
            required_columns=("vx", "vy", "vz"),
            value_prefix="velocity",
        ).items():
            merged.setdefault(step, {"step": step, "time_ps": step * spec.timestep_ps}).update(row)
    rows = [merged[step] for step in sorted(merged)]
    for row in rows:
        for column in FORCE_VELOCITY_COLUMNS:
            row.setdefault(column, math.nan)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in fieldnames})


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("matplotlib is required for plotting particle flotation outputs") from exc
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "ytick.major.size": 2.5,
            "xtick.major.size": 2.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _series(rows: Sequence[Mapping[str, object]], column: str) -> np.ndarray:
    return np.array([_as_float(row.get(column)) for row in rows], dtype=float)


def plot_lift(frame_rows: Sequence[Mapping[str, object]], path: Path, dpi: int) -> None:
    plt = _load_pyplot()
    time = _series(frame_rows, "time_ps")
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 5.6), sharex=True)
    axes[0].plot(time, _series(frame_rows, "particle_com_z_A"), color="#4C78A8", linewidth=1.2)
    axes[0].set_ylabel("COM z (A)")
    axes[1].plot(time, _series(frame_rows, "particle_lift_z_A"), color="#F58518", linewidth=1.2)
    axes[1].set_ylabel("lift z (A)")
    axes[2].plot(time, _series(frame_rows, "particle_slab_gap_A"), color="#54A24B", linewidth=1.2)
    axes[2].set_ylabel("particle-slab gap (A)")
    axes[2].set_xlabel("time (ps)")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_n2_contact(frame_rows: Sequence[Mapping[str, object]], path: Path, dpi: int) -> None:
    plt = _load_pyplot()
    time = _series(frame_rows, "time_ps")
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 3.8), sharex=True)
    axes[0].plot(time, _series(frame_rows, "n2_contact_count"), label="all", color="#4C78A8")
    axes[0].plot(time, _series(frame_rows, "n2_lower_contact_count"), label="lower", color="#F58518")
    axes[0].plot(time, _series(frame_rows, "n2_upper_contact_count"), label="upper", color="#54A24B")
    axes[0].set_ylabel("N2 near particle")
    axes[0].legend(frameon=False, ncols=3, loc="upper right")
    axes[1].plot(
        time,
        _series(frame_rows, "n2_angular_coverage_fraction"),
        label="angular bins",
        color="#4C78A8",
    )
    axes[1].plot(
        time,
        _series(frame_rows, "n2_projected_coverage_fraction"),
        label="projected area",
        color="#E45756",
    )
    axes[1].set_ylabel("coverage fraction")
    axes[1].set_xlabel("time (ps)")
    axes[1].legend(frameon=False, ncols=2, loc="upper right")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_n2_volume_density(frame_rows: Sequence[Mapping[str, object]], path: Path, dpi: int) -> None:
    plt = _load_pyplot()
    time = _series(frame_rows, "time_ps")
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 3.8), sharex=True)
    axes[0].plot(time, _series(frame_rows, "n2_voxel_volume_A3"), color="#4C78A8", linewidth=1.1)
    axes[0].set_ylabel("voxel volume (A3)")
    axes[1].plot(time, _series(frame_rows, "n2_mass_density_g_cm3"), color="#F58518", linewidth=1.1)
    axes[1].set_ylabel("N2 density (g/cm3)")
    axes[1].set_xlabel("time (ps)")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_radial_heatmap(radial_rows: Sequence[Mapping[str, object]], path: Path, dpi: int) -> None:
    if not radial_rows:
        return
    plt = _load_pyplot()
    times = sorted({_as_float(row["time_ps"]) for row in radial_rows})
    centers = sorted({_as_float(row["r_bin_center_A"]) for row in radial_rows})
    time_index = {value: index for index, value in enumerate(times)}
    r_index = {value: index for index, value in enumerate(centers)}
    grid = np.full((len(centers), len(times)), np.nan, dtype=float)
    for row in radial_rows:
        grid[r_index[_as_float(row["r_bin_center_A"])], time_index[_as_float(row["time_ps"])]] = _as_float(
            row["number_density_per_A3"]
        )
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    extent = [min(times), max(times), min(centers), max(centers)]
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("distance from particle COM (A)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("N2 number density (1/A3)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_force_velocity(rows: Sequence[Mapping[str, object]], path: Path, dpi: int) -> None:
    if not rows:
        return
    plt = _load_pyplot()
    time = _series(rows, "time_ps")
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 3.8), sharex=True)
    axes[0].plot(time, _series(rows, "particle_force_z"), marker="o", color="#4C78A8")
    axes[0].set_ylabel("net Fz")
    axes[1].plot(time, _series(rows, "particle_velocity_z_A_ps"), marker="o", color="#F58518")
    axes[1].set_ylabel("COM vz (A/ps)")
    axes[1].set_xlabel("time (ps)")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_snapshots(snapshots: Sequence[Snapshot], path: Path, dpi: int) -> None:
    if not snapshots:
        return
    plt = _load_pyplot()
    indices = sorted(set([0, len(snapshots) // 2, len(snapshots) - 1]))
    fig, axes = plt.subplots(1, len(indices), figsize=(3.1 * len(indices), 3.0), sharex=True, sharey=True)
    if len(indices) == 1:
        axes = [axes]
    for ax, index in zip(axes, indices):
        snap = snapshots[index]
        ax.scatter(
            snap.particle_rel[:, 0],
            snap.particle_rel[:, 2],
            s=2,
            color="#6E6E6E",
            alpha=0.45,
            linewidths=0,
            label="particle",
        )
        ax.scatter(
            snap.n2_rel[:, 0],
            snap.n2_rel[:, 2],
            s=7,
            color="#4C78A8",
            alpha=0.78,
            linewidths=0,
            label="N2 COM",
        )
        ax.axhline(0.0, color="#C7C7C7", linewidth=0.6)
        ax.axvline(0.0, color="#C7C7C7", linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{snap.time_ps:.1f} ps")
        ax.set_xlabel("x - particle COM (A)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("z - particle COM (A)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_rows(frame_rows: Sequence[Mapping[str, object]], force_velocity_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not frame_rows:
        return {"frame_count": 0}
    lift = _series(frame_rows, "particle_lift_z_A")
    gap = _series(frame_rows, "particle_slab_gap_A")
    contact = _series(frame_rows, "n2_contact_count")
    coverage = _series(frame_rows, "n2_angular_coverage_fraction")
    density = _series(frame_rows, "n2_mass_density_g_cm3")
    return {
        "frame_count": len(frame_rows),
        "time_start_ps": float(_series(frame_rows, "time_ps")[0]),
        "time_end_ps": float(_series(frame_rows, "time_ps")[-1]),
        "step_start": int(frame_rows[0]["step"]),
        "step_end": int(frame_rows[-1]["step"]),
        "particle_lift_final_A": float(lift[-1]),
        "particle_lift_max_A": float(np.nanmax(lift)),
        "particle_slab_gap_initial_A": float(gap[0]),
        "particle_slab_gap_final_A": float(gap[-1]),
        "n2_contact_initial": int(contact[0]),
        "n2_contact_final": int(contact[-1]),
        "n2_contact_max": int(np.nanmax(contact)),
        "n2_angular_coverage_final": float(coverage[-1]),
        "n2_density_initial_g_cm3": float(density[0]),
        "n2_density_final_g_cm3": float(density[-1]),
        "force_velocity_frame_count": len(force_velocity_rows),
    }


def write_report(
    path: Path,
    *,
    spec: AnalysisSpec,
    summary: Mapping[str, object],
    position_paths: Sequence[Path],
    force_paths: Sequence[Path],
    velocity_paths: Sequence[Path],
    output_dir: Path,
    figure_dir: Path,
) -> None:
    lines = [
        "# Particle flotation analysis",
        "",
        "## Inputs",
    ]
    for item in position_paths:
        lines.append(f"- trajectory: {item}")
    for item in force_paths:
        lines.append(f"- force: {item}")
    for item in velocity_paths:
        lines.append(f"- velocity: {item}")
    lines.extend(
        [
            "",
            "## Atom selections",
            f"- slab atom ids: {spec.slab.start}-{spec.slab.end} ({spec.slab.count})",
            f"- full particle atom ids: {spec.particle.start}-{spec.particle.end} ({spec.particle.count})",
            f"- particle core atom ids used for COM/force: {spec.particle_core.start}-{spec.particle_core.end} ({spec.particle_core.count})",
            f"- N2 atom ids: {spec.n2.start}-{spec.n2.end} ({spec.n2.count}; {spec.n2.count // 2} molecules)",
            f"- particle radius used for surface metrics: {spec.particle_radius_A:.3f} A",
            "",
            "## Summary",
        ]
    )
    for key, value in summary.items():
        lines.append(f"- {key}: {_format_value(value)}")
    lines.extend(
        [
            "",
            "## Outputs",
            f"- frame metrics: {output_dir / 'particle_flotation_frame_metrics.csv'}",
            f"- radial density: {output_dir / 'n2_radial_density.csv'}",
            f"- force/velocity metrics: {output_dir / 'particle_force_velocity_metrics.csv'}",
            f"- summary JSON: {output_dir / 'particle_flotation_summary.json'}",
            "",
            "## Figures",
            f"- {figure_dir / 'particle_lift_timeseries.png'}",
            f"- {figure_dir / 'n2_contact_coverage_timeseries.png'}",
            f"- {figure_dir / 'n2_volume_density_timeseries.png'}",
            f"- {figure_dir / 'n2_radial_density_heatmap.png'}",
            f"- {figure_dir / 'particle_n2_snapshots_xz.png'}",
        ]
    )
    if force_paths or velocity_paths:
        lines.append(f"- {figure_dir / 'particle_force_velocity_timeseries.png'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Dict[str, Path]:
    spec = load_spec_from_args(args)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    position_paths = [Path(item) for item in args.trajectory]
    force_paths = [Path(item) for item in args.force_dump or []]
    velocity_paths = [Path(item) for item in args.velocity_dump or []]
    frame_rows, radial_rows, snapshots = process_position_trajectories(
        position_paths,
        spec,
        start_time_ps=args.start_time_ps,
        end_time_ps=args.end_time_ps,
        max_frames=args.max_frames,
        keep_duplicate_steps=args.keep_duplicate_steps,
        contact_shell_A=args.contact_shell_A,
        coverage_probe_radius_A=args.coverage_probe_radius_A,
        coverage_z_bins=args.coverage_z_bins,
        coverage_phi_bins=args.coverage_phi_bins,
        voxel_A=args.n2_volume_voxel_A,
        voxel_probe_radius_A=args.n2_volume_probe_radius_A,
        radial_bin_width_A=args.radial_bin_width_A,
        radial_max_A=args.radial_max_A,
    )
    force_velocity_rows = process_force_velocity(force_paths, velocity_paths, spec)
    paths = {
        "frame_metrics": output_dir / "particle_flotation_frame_metrics.csv",
        "radial_density": output_dir / "n2_radial_density.csv",
        "force_velocity": output_dir / "particle_force_velocity_metrics.csv",
        "summary_json": output_dir / "particle_flotation_summary.json",
        "report": output_dir / "particle_flotation_report.md",
    }
    write_csv(paths["frame_metrics"], frame_rows, FRAME_COLUMNS)
    write_csv(paths["radial_density"], radial_rows, RADIAL_COLUMNS)
    write_csv(paths["force_velocity"], force_velocity_rows, FORCE_VELOCITY_COLUMNS)
    summary = summarize_rows(frame_rows, force_velocity_rows)
    payload = {
        "summary": summary,
        "atom_selections": {
            "slab": [spec.slab.start, spec.slab.end],
            "particle": [spec.particle.start, spec.particle.end],
            "particle_core": [spec.particle_core.start, spec.particle_core.end],
            "n2": [spec.n2.start, spec.n2.end],
            "type_to_symbol": {str(key): value for key, value in spec.type_to_symbol.items()},
            "particle_radius_A": spec.particle_radius_A,
            "timestep_ps": spec.timestep_ps,
        },
        "parameters": {
            "contact_shell_A": args.contact_shell_A,
            "coverage_probe_radius_A": args.coverage_probe_radius_A,
            "coverage_z_bins": args.coverage_z_bins,
            "coverage_phi_bins": args.coverage_phi_bins,
            "n2_volume_voxel_A": args.n2_volume_voxel_A,
            "n2_volume_probe_radius_A": args.n2_volume_probe_radius_A,
            "radial_bin_width_A": args.radial_bin_width_A,
            "radial_max_A": args.radial_max_A,
        },
        "inputs": {
            "trajectory": [str(path) for path in position_paths],
            "force_dump": [str(path) for path in force_paths],
            "velocity_dump": [str(path) for path in velocity_paths],
            "model_summary": str(args.model_summary) if args.model_summary else "",
        },
    }
    paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.no_plots and frame_rows:
        plot_lift(frame_rows, figure_dir / "particle_lift_timeseries.png", args.dpi)
        plot_n2_contact(frame_rows, figure_dir / "n2_contact_coverage_timeseries.png", args.dpi)
        plot_n2_volume_density(frame_rows, figure_dir / "n2_volume_density_timeseries.png", args.dpi)
        plot_radial_heatmap(radial_rows, figure_dir / "n2_radial_density_heatmap.png", args.dpi)
        plot_snapshots(snapshots, figure_dir / "particle_n2_snapshots_xz.png", args.dpi)
        if force_velocity_rows:
            plot_force_velocity(force_velocity_rows, figure_dir / "particle_force_velocity_timeseries.png", args.dpi)
    write_report(
        paths["report"],
        spec=spec,
        summary=summary,
        position_paths=position_paths,
        force_paths=force_paths,
        velocity_paths=velocity_paths,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--force-dump", type=Path, action="append")
    parser.add_argument("--velocity-dump", type=Path, action="append")
    parser.add_argument("--model-summary", type=Path)
    parser.add_argument("--slab-range", type=parse_atom_range)
    parser.add_argument("--particle-range", type=parse_atom_range)
    parser.add_argument("--n2-range", type=parse_atom_range)
    parser.add_argument("--type-map", action="append", help="LAMMPS type mapping such as 1=H")
    parser.add_argument("--particle-radius-A", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--timestep-ps", type=float, default=0.0005)
    parser.add_argument("--start-time-ps", type=float)
    parser.add_argument("--end-time-ps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--keep-duplicate-steps", action="store_true")
    parser.add_argument("--contact-shell-A", type=float, default=5.0)
    parser.add_argument("--coverage-probe-radius-A", type=float, default=2.0)
    parser.add_argument("--coverage-z-bins", type=int, default=8)
    parser.add_argument("--coverage-phi-bins", type=int, default=16)
    parser.add_argument("--n2-volume-voxel-A", type=float, default=3.0)
    parser.add_argument("--n2-volume-probe-radius-A", type=float, default=3.0)
    parser.add_argument("--radial-bin-width-A", type=float, default=2.0)
    parser.add_argument("--radial-max-A", type=float, default=80.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = run_analysis(args)
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
