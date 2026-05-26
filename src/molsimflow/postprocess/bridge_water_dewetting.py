"""Bridge-water dewetting metrics from two-bubble trajectories.

This module migrates the reusable geometry and connectivity logic from the
legacy bridge-water dewetting workflow.  Inputs are explicit: a LAMMPS dump,
water oxygen atom ids, and two bubble atom groups either from CLI ranges or a
PLUMED file containing `bubA_all` and `bubB_all` definitions.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.coalescence_state import read_plumed_table


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

FRAME_COLUMNS: Tuple[str, ...] = (
    "frame",
    "timestep",
    "time_ns",
    "Nw_bridge",
    "Nw_expected",
    "rho_bridge_per_A3",
    "DeltaN_dewet",
    "dewet_fraction",
    "largest_water_cluster_size_bridge",
    "water_bridge_connected_flag",
    "d3d_all",
    "bridge_cyl_env.sum",
    "bridge_cyl_env.mean",
    "n2A_num",
    "n2B_num",
    "sumA_cn.sum",
    "sumB_cn.sum",
    "wallA.bias",
    "wallB.bias",
)

CV_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "cv_bin_center",
    "cv_bin_left",
    "cv_bin_right",
    "count",
    "Nw_bridge_mean",
    "Nw_bridge_std",
    "DeltaN_dewet_mean",
    "DeltaN_dewet_std",
    "dewet_fraction_mean",
    "dewet_fraction_std",
    "water_bridge_connected_probability",
    "bridge_cyl_env_sum_mean",
    "bridge_cyl_env_mean_mean",
)

COLVAR_COLUMNS: Tuple[str, ...] = ("d3d_all", "bridge_cyl_env.sum")
POST_COLUMNS: Tuple[str, ...] = (
    "n2A_num",
    "n2B_num",
    "sumA_cn.sum",
    "sumB_cn.sum",
    "bridge_cyl_env.mean",
    "wallA.bias",
    "wallB.bias",
)


@dataclass(frozen=True)
class PlumedAtomDefinition:
    """One PLUMED label and its optional ATOMS expression."""

    action: str
    atoms_expr: Optional[str]


@dataclass(frozen=True)
class LammpsFrame:
    """Selected atom positions from one LAMMPS dump frame."""

    frame_index: int
    timestep: int
    bounds: np.ndarray
    selected_positions: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class BridgeWaterDewettingConfig:
    """Geometry and matching settings for bridge-water dewetting."""

    axis: str = "z"
    radius_A: float = 6.5
    lower_A: float = -8.0
    upper_A: float = 8.0
    oo_cutoff_A: float = 3.5
    connect_side_thickness_A: float = 2.0
    connect_min_water: int = 2
    time_tolerance_ns: float = 0.00051
    dump_time_scale_ns: Optional[float] = None
    bulk_number_density_per_A3: Optional[float] = None
    cv_bins: int = 40
    max_frames: Optional[int] = None

    @property
    def axis_index(self) -> int:
        if self.axis not in AXIS_TO_INDEX:
            raise ValueError(f"Unsupported bridge axis: {self.axis}")
        return AXIS_TO_INDEX[self.axis]

    @property
    def cylinder_volume_A3(self) -> float:
        if self.radius_A <= 0:
            raise ValueError("radius_A must be positive")
        if self.upper_A <= self.lower_A:
            raise ValueError("upper_A must be greater than lower_A")
        return math.pi * float(self.radius_A) ** 2 * (float(self.upper_A) - float(self.lower_A))


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Iterable[object]) -> float:
    finite = [_as_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _std(values: Iterable[object]) -> float:
    finite = [_as_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) == 1 else math.nan


def parse_atom_selection(expression: str) -> List[int]:
    """Parse a PLUMED-like 1-based atom expression such as `1,4-10:2`."""

    text = str(expression).strip()
    if not text:
        raise ValueError("Atom expression is empty")
    atom_ids: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"\d+", token):
            atom_ids.append(int(token))
            continue
        match = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", token)
        if match is None:
            raise ValueError(f"Unsupported atom token: {token}")
        start = int(match.group(1))
        stop = int(match.group(2))
        step = int(match.group(3)) if match.group(3) else 1
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid atom range: {token}")
        atom_ids.extend(range(start, stop + 1, step))
    out: List[int] = []
    seen = set()
    for atom_id in atom_ids:
        if atom_id not in seen:
            seen.add(atom_id)
            out.append(atom_id)
    if not out:
        raise ValueError(f"No atom ids parsed from expression: {expression}")
    return out


def parse_plumed_atom_definitions(plumed_path: Path) -> Dict[str, PlumedAtomDefinition]:
    """Parse PLUMED labels with ATOMS expressions."""

    label_pattern = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\b(.*)$")
    atoms_pattern = re.compile(r"\bATOMS=([^\s]+)")
    definitions: Dict[str, PlumedAtomDefinition] = {}
    with Path(plumed_path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            match = label_pattern.match(line)
            if match is None:
                continue
            label, action, suffix = match.groups()
            atoms_match = atoms_pattern.search(suffix)
            definitions[label] = PlumedAtomDefinition(
                action=action.upper(),
                atoms_expr=atoms_match.group(1).strip() if atoms_match else None,
            )
    return definitions


def _token_is_numeric_atom_expr(token: str) -> bool:
    return bool(re.fullmatch(r"\d+", token) or re.fullmatch(r"\d+-\d+(?::\d+)?", token))


def resolve_plumed_atoms_expression(
    expression: str,
    definitions: Mapping[str, PlumedAtomDefinition],
    stack: Optional[Sequence[str]] = None,
) -> List[int]:
    """Resolve mixed PLUMED atom expressions to concrete atom ids."""

    resolved: List[int] = []
    stack = tuple(stack or ())
    for token in [item.strip() for item in str(expression).split(",") if item.strip()]:
        if _token_is_numeric_atom_expr(token):
            resolved.extend(parse_atom_selection(token))
            continue
        definition = definitions.get(token)
        if definition is None:
            raise ValueError(f"Unknown PLUMED label in ATOMS expression: {token}")
        if definition.atoms_expr is None:
            raise ValueError(f"PLUMED label {token} has no ATOMS expression")
        if token in stack:
            raise ValueError(f"Recursive PLUMED label reference detected: {token}")
        resolved.extend(resolve_plumed_atoms_expression(definition.atoms_expr, definitions, stack=(*stack, token)))
    if not resolved:
        raise ValueError(f"Resolved empty atom set for expression: {expression}")
    return resolved


def extract_bubble_atom_groups_from_plumed(plumed_path: Path) -> Tuple[List[int], List[int]]:
    """Resolve `bubA_all` and `bubB_all` from a PLUMED file."""

    definitions = parse_plumed_atom_definitions(plumed_path)
    missing = [label for label in ("bubA_all", "bubB_all") if label not in definitions]
    if missing:
        raise ValueError(f"Missing PLUMED bubble labels: {', '.join(missing)}")
    return (
        resolve_plumed_atoms_expression(definitions["bubA_all"].atoms_expr or "", definitions),
        resolve_plumed_atoms_expression(definitions["bubB_all"].atoms_expr or "", definitions),
    )


def _choose_coord_field(fields: Sequence[str], dim: str) -> Tuple[int, bool]:
    for name in (dim, dim + "u", dim + "s"):
        if name in fields:
            return fields.index(name), name.endswith("s")
    raise ValueError(f"LAMMPS dump is missing {dim}/{dim}u/{dim}s coordinate column")


def iter_lammps_dump_frames(
    dump_path: Path,
    needed_atom_ids: Optional[Iterable[int]] = None,
    max_frames: Optional[int] = None,
) -> Iterator[LammpsFrame]:
    """Iterate LAMMPS dump frames while retaining selected atom positions."""

    needed = set(needed_atom_ids) if needed_atom_ids is not None else None
    with Path(dump_path).open(encoding="utf-8") as handle:
        frame_index = 0
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError("Unexpected dump format: expected ITEM: TIMESTEP")
            timestep = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected dump format: missing ITEM: NUMBER OF ATOMS")
            n_atoms = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: missing ITEM: BOX BOUNDS")
            bounds = np.zeros((3, 2), dtype=float)
            for dim in range(3):
                parts = handle.readline().split()
                bounds[dim, 0] = float(parts[0])
                bounds[dim, 1] = float(parts[1])
            atoms_header = handle.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: missing ITEM: ATOMS")
            fields = atoms_header.split()[2:]
            if "id" not in fields:
                raise ValueError("LAMMPS dump ATOMS line must contain id")
            id_index = fields.index("id")
            x_index, x_scaled = _choose_coord_field(fields, "x")
            y_index, y_scaled = _choose_coord_field(fields, "y")
            z_index, z_scaled = _choose_coord_field(fields, "z")
            lengths = bounds[:, 1] - bounds[:, 0]
            selected: Dict[int, np.ndarray] = {}
            for _ in range(n_atoms):
                parts = handle.readline().split()
                atom_id = int(parts[id_index])
                if needed is not None and atom_id not in needed:
                    continue
                coords = np.asarray([float(parts[x_index]), float(parts[y_index]), float(parts[z_index])], dtype=float)
                for dim, scaled in enumerate((x_scaled, y_scaled, z_scaled)):
                    if scaled:
                        coords[dim] = bounds[dim, 0] + coords[dim] * lengths[dim]
                selected[atom_id] = coords
            yield LammpsFrame(frame_index=frame_index, timestep=timestep, bounds=bounds, selected_positions=selected)
            frame_index += 1
            if max_frames is not None and frame_index >= int(max_frames):
                break


def box_lengths(bounds: np.ndarray) -> np.ndarray:
    return np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]


def minimum_image_vectors(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    return values - np.asarray(lengths, dtype=float) * np.round(values / np.asarray(lengths, dtype=float))


def wrap_point_to_box(point: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lengths = box_lengths(bounds)
    wrapped = np.empty(3, dtype=float)
    for dim in range(3):
        lo = float(bounds[dim, 0])
        wrapped[dim] = ((float(point[dim]) - lo) % float(lengths[dim])) + lo
    return wrapped


def periodic_center(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Compute a periodic center robustly when coordinates straddle boundaries."""

    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        raise ValueError("Cannot compute periodic center for empty coordinate array")
    lengths = box_lengths(bounds)
    center = np.empty(3, dtype=float)
    for dim in range(3):
        lo = float(bounds[dim, 0])
        scaled = (coords[:, dim] - lo) / float(lengths[dim])
        angles = 2.0 * np.pi * scaled
        complex_mean = np.exp(1j * angles).mean()
        if np.isclose(abs(complex_mean), 0.0):
            center[dim] = float(np.mean(coords[:, dim]))
            continue
        angle = np.angle(complex_mean)
        if angle < 0:
            angle += 2.0 * np.pi
        center[dim] = lo + (angle / (2.0 * np.pi)) * float(lengths[dim])
    return center


def midpoint_minimum_image(center_a: np.ndarray, center_b: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lengths = box_lengths(bounds)
    delta = minimum_image_vectors(np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float), lengths)
    return wrap_point_to_box(np.asarray(center_a, dtype=float) + 0.5 * delta, bounds)


def cylinder_membership(
    coords: np.ndarray,
    center: np.ndarray,
    bounds: np.ndarray,
    axis_index: int,
    radius_A: float,
    lower_A: float,
    upper_A: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bridge mask, local axial coordinate, and radial distance."""

    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    lengths = box_lengths(bounds)
    deltas = minimum_image_vectors(coords - np.asarray(center, dtype=float), lengths)
    axial = deltas[:, axis_index]
    perp = [index for index in range(3) if index != axis_index]
    radial = np.sqrt(np.sum(deltas[:, perp] ** 2, axis=1))
    mask = (axial >= float(lower_A)) & (axial <= float(upper_A)) & (radial <= float(radius_A))
    return mask, axial, radial


def _union_find_root(parents: np.ndarray, index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = int(parents[index])
    return int(index)


def _union_find_union(parents: np.ndarray, sizes: np.ndarray, i: int, j: int) -> None:
    root_i = _union_find_root(parents, i)
    root_j = _union_find_root(parents, j)
    if root_i == root_j:
        return
    if sizes[root_i] < sizes[root_j]:
        root_i, root_j = root_j, root_i
    parents[root_j] = root_i
    sizes[root_i] += sizes[root_j]


def analyze_bridge_connectivity(
    bridge_coords: np.ndarray,
    bridge_axial: np.ndarray,
    bounds: np.ndarray,
    oo_cutoff_A: float,
    lower_A: float,
    upper_A: float,
    side_thickness_A: float,
    min_bridge_waters_for_connectivity: int = 2,
) -> Tuple[int, bool]:
    """Return largest bridge-water cluster size and spanning flag."""

    n_atoms = int(np.asarray(bridge_coords).shape[0])
    if n_atoms == 0:
        return 0, False
    parents = np.arange(n_atoms)
    sizes = np.ones(n_atoms, dtype=int)
    lengths = box_lengths(bounds)
    cutoff_sq = float(oo_cutoff_A) ** 2
    for i in range(n_atoms - 1):
        deltas = np.asarray(bridge_coords)[i + 1 :] - np.asarray(bridge_coords)[i]
        deltas = minimum_image_vectors(deltas, lengths)
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        for offset in np.where(distances_sq <= cutoff_sq)[0]:
            _union_find_union(parents, sizes, i, i + 1 + int(offset))
    clusters: Dict[int, List[int]] = {}
    for atom_index in range(n_atoms):
        clusters.setdefault(_union_find_root(parents, atom_index), []).append(atom_index)
    largest = max((len(indices) for indices in clusters.values()), default=0)
    if n_atoms < int(min_bridge_waters_for_connectivity):
        return largest, False
    lower_mask = np.asarray(bridge_axial) <= (float(lower_A) + float(side_thickness_A))
    upper_mask = np.asarray(bridge_axial) >= (float(upper_A) - float(side_thickness_A))
    for indices in clusters.values():
        cluster = np.asarray(indices, dtype=int)
        if lower_mask[cluster].any() and upper_mask[cluster].any():
            return largest, True
    return largest, False


def _nearest_row_index(rows: Sequence[Mapping[str, object]], time_ns: float, tolerance_ns: float) -> Optional[int]:
    if not rows:
        return None
    times = np.asarray([_as_float(row.get("time_ns")) for row in rows], dtype=float)
    index = int(np.searchsorted(times, float(time_ns)))
    candidates = []
    if index < times.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(float(times[item]) - float(time_ns)))
    return best if abs(float(times[best]) - float(time_ns)) <= float(tolerance_ns) else None


def infer_dump_time_scale_ns(
    timesteps: Sequence[int],
    colvar_rows: Sequence[Mapping[str, object]],
    tolerance_ns: float,
) -> float:
    """Infer timestep-to-ns scale from COLVAR matching."""

    if not timesteps or not colvar_rows:
        return 1.0
    candidates = (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
    best_scale = 1.0
    best_matches = -1
    best_delta = math.inf
    for scale in candidates:
        matches = 0
        deltas = []
        for timestep in timesteps:
            frame_time = float(timestep) * scale
            index = _nearest_row_index(colvar_rows, frame_time, tolerance_ns)
            if index is not None:
                matches += 1
                deltas.append(abs(_as_float(colvar_rows[index].get("time_ns")) - frame_time))
        mean_delta = float(np.mean(deltas)) if deltas else math.inf
        if matches > best_matches or (matches == best_matches and mean_delta < best_delta):
            best_scale = scale
            best_matches = matches
            best_delta = mean_delta
    if best_matches <= 0:
        max_step = max(abs(float(item)) for item in timesteps)
        max_time = max(abs(_as_float(row.get("time_ns"))) for row in colvar_rows)
        if max_step > 0 and max_time > 0:
            return max_time / max_step
    return best_scale


def compute_cv_binned_summary(rows: Sequence[Mapping[str, object]], bins: int) -> List[Dict[str, object]]:
    """Compute d3d_all-binned dewetting summaries."""

    valid = [row for row in rows if math.isfinite(_as_float(row.get("d3d_all")))]
    if not valid:
        return []
    values = np.asarray([_as_float(row.get("d3d_all")) for row in valid], dtype=float)
    if float(np.nanmax(values)) == float(np.nanmin(values)):
        edges = np.asarray([float(values[0]) - 0.5, float(values[0]) + 0.5], dtype=float)
    else:
        edges = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), max(1, int(bins)) + 1)
    out: List[Dict[str, object]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        chunk = [
            row
            for row in valid
            if _as_float(row.get("d3d_all")) >= float(left)
            and (_as_float(row.get("d3d_all")) < float(right) or right == edges[-1])
        ]
        if not chunk:
            continue
        out.append(
            {
                "cv_bin_center": 0.5 * (float(left) + float(right)),
                "cv_bin_left": float(left),
                "cv_bin_right": float(right),
                "count": len(chunk),
                "Nw_bridge_mean": _mean(row.get("Nw_bridge") for row in chunk),
                "Nw_bridge_std": _std(row.get("Nw_bridge") for row in chunk),
                "DeltaN_dewet_mean": _mean(row.get("DeltaN_dewet") for row in chunk),
                "DeltaN_dewet_std": _std(row.get("DeltaN_dewet") for row in chunk),
                "dewet_fraction_mean": _mean(row.get("dewet_fraction") for row in chunk),
                "dewet_fraction_std": _std(row.get("dewet_fraction") for row in chunk),
                "water_bridge_connected_probability": _mean(
                    1.0 if row.get("water_bridge_connected_flag") else 0.0 for row in chunk
                ),
                "bridge_cyl_env_sum_mean": _mean(row.get("bridge_cyl_env.sum") for row in chunk),
                "bridge_cyl_env_mean_mean": _mean(row.get("bridge_cyl_env.mean") for row in chunk),
            }
        )
    return out


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_statistics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv_rows(path, rows, ["metric", "value"])


def analyze_bridge_water_dewetting(
    dump: Path,
    output_dir: Path,
    water_oxygen_atoms: Sequence[int],
    bubble_a_atoms: Sequence[int],
    bubble_b_atoms: Sequence[int],
    config: BridgeWaterDewettingConfig = BridgeWaterDewettingConfig(),
    colvar: Optional[Path] = None,
    colvar_post: Optional[Path] = None,
    colvar_time_unit: str = "ns",
) -> Dict[str, Path]:
    """Compute bridge-water dewetting frame and CV summary tables."""

    if not water_oxygen_atoms:
        raise ValueError("At least one water oxygen atom id is required")
    if not bubble_a_atoms or not bubble_b_atoms:
        raise ValueError("Both bubble atom groups are required")
    colvar_rows = read_plumed_table(colvar, time_unit=colvar_time_unit, optional_columns=COLVAR_COLUMNS) if colvar else []
    post_rows = read_plumed_table(colvar_post, time_unit=colvar_time_unit, optional_columns=POST_COLUMNS) if colvar_post else []
    needed = set(water_oxygen_atoms) | set(bubble_a_atoms) | set(bubble_b_atoms)
    frames = list(iter_lammps_dump_frames(dump, needed_atom_ids=needed, max_frames=config.max_frames))
    if not frames:
        raise ValueError(f"No frames were processed from {dump}")
    timesteps = [frame.timestep for frame in frames]
    time_scale = config.dump_time_scale_ns
    if time_scale is None:
        time_scale = infer_dump_time_scale_ns(timesteps, colvar_rows, tolerance_ns=config.time_tolerance_ns)
    volume_A3 = config.cylinder_volume_A3
    rows: List[Dict[str, object]] = []
    density_samples = []
    matched_colvar = 0
    matched_post = 0
    for frame in frames:
        missing = needed.difference(frame.selected_positions.keys())
        if missing:
            raise ValueError(
                f"Frame {frame.frame_index} missing {len(missing)} required selected atoms; first missing {min(missing)}"
            )
        bounds = frame.bounds
        box_volume_A3 = float(np.prod(box_lengths(bounds)))
        density_samples.append(len(water_oxygen_atoms) / box_volume_A3)
        water_coords = np.asarray([frame.selected_positions[atom_id] for atom_id in water_oxygen_atoms], dtype=float)
        bubble_a_coords = np.asarray([frame.selected_positions[atom_id] for atom_id in bubble_a_atoms], dtype=float)
        bubble_b_coords = np.asarray([frame.selected_positions[atom_id] for atom_id in bubble_b_atoms], dtype=float)
        center_a = periodic_center(bubble_a_coords, bounds)
        center_b = periodic_center(bubble_b_coords, bounds)
        midpoint = midpoint_minimum_image(center_a, center_b, bounds)
        inside, axial, _radial = cylinder_membership(
            water_coords,
            center=midpoint,
            bounds=bounds,
            axis_index=config.axis_index,
            radius_A=config.radius_A,
            lower_A=config.lower_A,
            upper_A=config.upper_A,
        )
        bridge_coords = water_coords[inside]
        bridge_axial = axial[inside]
        n_bridge = int(np.count_nonzero(inside))
        largest, connected = analyze_bridge_connectivity(
            bridge_coords,
            bridge_axial,
            bounds,
            oo_cutoff_A=config.oo_cutoff_A,
            lower_A=config.lower_A,
            upper_A=config.upper_A,
            side_thickness_A=config.connect_side_thickness_A,
            min_bridge_waters_for_connectivity=config.connect_min_water,
        )
        time_ns = float(frame.timestep) * float(time_scale)
        row: Dict[str, object] = {
            "frame": frame.frame_index,
            "timestep": frame.timestep,
            "time_ns": time_ns,
            "Nw_bridge": n_bridge,
            "rho_bridge_per_A3": n_bridge / volume_A3,
            "largest_water_cluster_size_bridge": largest,
            "water_bridge_connected_flag": bool(connected),
            "d3d_all": math.nan,
            "bridge_cyl_env.sum": math.nan,
            "bridge_cyl_env.mean": math.nan,
            "n2A_num": math.nan,
            "n2B_num": math.nan,
            "sumA_cn.sum": math.nan,
            "sumB_cn.sum": math.nan,
            "wallA.bias": math.nan,
            "wallB.bias": math.nan,
        }
        colvar_index = _nearest_row_index(colvar_rows, time_ns, config.time_tolerance_ns)
        if colvar_index is not None:
            matched_colvar += 1
            row["d3d_all"] = _as_float(colvar_rows[colvar_index].get("d3d_all"))
            row["bridge_cyl_env.sum"] = _as_float(colvar_rows[colvar_index].get("bridge_cyl_env.sum"))
        post_index = _nearest_row_index(post_rows, time_ns, config.time_tolerance_ns)
        if post_index is not None:
            matched_post += 1
            for column in POST_COLUMNS:
                if column in post_rows[post_index]:
                    row[column] = _as_float(post_rows[post_index].get(column))
        rows.append(row)
    if config.bulk_number_density_per_A3 is not None:
        bulk_density = float(config.bulk_number_density_per_A3)
        bulk_density_source = "user_specified"
    else:
        bulk_density = float(np.nanmean(np.asarray(density_samples, dtype=float)))
        bulk_density_source = "trajectory_estimate"
    expected = bulk_density * volume_A3
    for row in rows:
        row["Nw_expected"] = expected
        delta = expected - _as_float(row.get("Nw_bridge"))
        row["DeltaN_dewet"] = delta
        row["dewet_fraction"] = delta / expected if math.isfinite(expected) and expected > 0 else math.nan
    binned = compute_cv_binned_summary(rows, config.cv_bins)
    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "bridge_water_dewetting.csv",
        "cv_summary": output_dir / "bridge_water_dewetting_by_cv.csv",
        "statistics": output_dir / "bridge_water_dewetting_statistics.csv",
    }
    _write_csv_rows(outputs["frame_table"], rows, FRAME_COLUMNS)
    _write_csv_rows(outputs["cv_summary"], binned, CV_SUMMARY_COLUMNS)
    _write_statistics(
        outputs["statistics"],
        [
            {"metric": "total_frames", "value": len(rows)},
            {"metric": "matched_colvar_frames", "value": matched_colvar},
            {"metric": "matched_colvar_post_frames", "value": matched_post},
            {"metric": "bulk_number_density_per_A3", "value": bulk_density},
            {"metric": "bulk_density_source", "value": bulk_density_source},
            {"metric": "dump_time_scale_ns", "value": time_scale},
            {"metric": "bridge_cylinder_volume_A3", "value": volume_A3},
        ],
    )
    return outputs


def _resolve_bubble_groups(args: argparse.Namespace) -> Tuple[List[int], List[int]]:
    if args.plumed is not None:
        return extract_bubble_atom_groups_from_plumed(args.plumed)
    if args.bubble_a_atoms and args.bubble_b_atoms:
        return parse_atom_selection(args.bubble_a_atoms), parse_atom_selection(args.bubble_b_atoms)
    raise ValueError("Provide --plumed or both --bubble-a-atoms and --bubble-b-atoms")


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bridge-water dewetting metrics from a LAMMPS dump")
    parser.add_argument("--dump", type=Path, required=True, help="LAMMPS dump trajectory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--water-oxygen-atoms", required=True, help="PLUMED-style atom id expression for water oxygens")
    parser.add_argument("--plumed", type=Path, help="PLUMED file containing bubA_all and bubB_all labels")
    parser.add_argument("--bubble-a-atoms", help="Explicit atom expression for bubble A")
    parser.add_argument("--bubble-b-atoms", help="Explicit atom expression for bubble B")
    parser.add_argument("--colvar", type=Path, help="Optional COLVAR table")
    parser.add_argument("--colvar-post", type=Path, help="Optional secondary COLVAR table")
    parser.add_argument("--colvar-time-unit", choices=["fs", "ps", "ns"], default="ns")
    parser.add_argument("--axis", choices=sorted(AXIS_TO_INDEX), default="z")
    parser.add_argument("--radius-A", type=float, default=6.5)
    parser.add_argument("--lower-A", type=float, default=-8.0)
    parser.add_argument("--upper-A", type=float, default=8.0)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    parser.add_argument("--connect-side-thickness-A", type=float, default=2.0)
    parser.add_argument("--connect-min-water", type=int, default=2)
    parser.add_argument("--time-tolerance-ns", type=float, default=0.00051)
    parser.add_argument("--dump-time-scale-ns", type=float)
    parser.add_argument("--bulk-number-density-per-A3", type=float)
    parser.add_argument("--cv-bins", type=int, default=40)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        bubble_a, bubble_b = _resolve_bubble_groups(args)
        outputs = analyze_bridge_water_dewetting(
            dump=args.dump,
            output_dir=args.output_dir,
            water_oxygen_atoms=parse_atom_selection(args.water_oxygen_atoms),
            bubble_a_atoms=bubble_a,
            bubble_b_atoms=bubble_b,
            colvar=args.colvar,
            colvar_post=args.colvar_post,
            colvar_time_unit=args.colvar_time_unit,
            config=BridgeWaterDewettingConfig(
                axis=args.axis,
                radius_A=args.radius_A,
                lower_A=args.lower_A,
                upper_A=args.upper_A,
                oo_cutoff_A=args.oo_cutoff_A,
                connect_side_thickness_A=args.connect_side_thickness_A,
                connect_min_water=args.connect_min_water,
                time_tolerance_ns=args.time_tolerance_ns,
                dump_time_scale_ns=args.dump_time_scale_ns,
                bulk_number_density_per_A3=args.bulk_number_density_per_A3,
                cv_bins=args.cv_bins,
                max_frames=args.max_frames,
            ),
        )
    except Exception as exc:
        print(f"Bridge-water dewetting failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
