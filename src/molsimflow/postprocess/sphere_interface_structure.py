"""Analyze terminated spherical interfaces, solvent orientation, and H-bond structure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

@dataclass(frozen=True)
class CaseSpec:
    label: str
    run_dir: Path


@dataclass(frozen=True)
class AtomTypeMap:
    """LAMMPS atom-type IDs needed by the interface analysis.

    LAMMPS type numbers are system-specific, so they must be supplied by the
    caller instead of being embedded in the library.
    """

    framework: int
    oxygen: int
    carbon: int
    hydrogen: int

    def validate(self) -> None:
        values = {
            "framework": self.framework,
            "oxygen": self.oxygen,
            "carbon": self.carbon,
            "hydrogen": self.hydrogen,
        }
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError("LAMMPS atom-type IDs must be positive")
        if len(set(values.values())) != len(values):
            raise ValueError("framework, oxygen, carbon, and hydrogen atom types must be distinct")


@dataclass(frozen=True)
class SegmentSpec:
    label: str
    segment_dir: Path
    dump_path: Path
    plumed_path: Path
    frame_count: int


@dataclass(frozen=True)
class SelectionMap:
    surf_si_top: Tuple[int, ...]
    surface_ch3: Tuple[int, ...]
    surface_oh: Tuple[int, ...]
    water_o: Tuple[int, ...]
    n2_atoms: Tuple[int, ...]


@dataclass(frozen=True)
class AnalysisConfig:
    last_frames_per_case: int = 200
    frame_stride: int = 2
    timestep_ps: float = 0.001
    interface_z_min_A: float = 0.0
    interface_z_max_A: float = 12.0
    bubble_radius_A: float = 21.0
    outside_margin_A: float = 5.0
    si_o_cutoff_A: float = 2.20
    si_c_cutoff_A: float = 2.35
    covalent_oh_cutoff_A: float = 1.30
    water_oh_valid_cutoff_A: float = 1.35
    hbond_oo_cutoff_A: float = 3.50
    hbond_angle_deg: float = 30.0
    dpi: int = 220
    make_plots: bool = True

    def validate(self) -> None:
        if self.last_frames_per_case < 1:
            raise ValueError("last_frames_per_case must be positive")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if self.interface_z_max_A <= self.interface_z_min_A:
            raise ValueError("interface_z_max_A must exceed interface_z_min_A")
        for name in (
            "bubble_radius_A",
            "si_o_cutoff_A",
            "si_c_cutoff_A",
            "covalent_oh_cutoff_A",
            "water_oh_valid_cutoff_A",
            "hbond_oo_cutoff_A",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ReferenceState:
    ligand_edges: Set[Tuple[int, int, str]]
    backbone_o_ids: Set[int]
    surf_si_positions: Dict[int, np.ndarray]
    terminal_positions: Dict[int, np.ndarray]
    terminal_si_by_id: Dict[int, int]
    surface_oh_h_by_id: Dict[int, int]


@dataclass(frozen=True)
class DumpFrame:
    global_index: int
    segment_label: str
    local_index: int
    timestep: int
    bounds: np.ndarray
    positions: Mapping[int, np.ndarray]


def parse_case(value: str) -> CaseSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Case must be LABEL=RUN_DIR")
    label, run_dir = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Case label must not be empty")
    return CaseSpec(label=label, run_dir=Path(run_dir).expanduser())


def parse_atom_selection(selection: str) -> Tuple[int, ...]:
    atoms: List[int] = []
    for token in selection.replace(" ", "").split(","):
        if not token:
            continue
        stride = 1
        if ":" in token:
            token, stride_text = token.split(":", 1)
            stride = int(stride_text)
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            atoms.extend(range(int(start_text), int(stop_text) + 1, stride))
        else:
            atoms.append(int(token))
    return tuple(atoms)


def _parse_group(plumed_text: str, name: str, required: bool = False) -> Tuple[int, ...]:
    match = re.search(rf"^{re.escape(name)}:\s+GROUP\s+ATOMS=([^\n]+)", plumed_text, re.MULTILINE)
    if match:
        return parse_atom_selection(match.group(1))
    if required:
        raise ValueError(f"Missing PLUMED group {name!r}")
    return ()


def parse_plumed_selections(path: Path) -> SelectionMap:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    n2_atoms: List[int] = []
    for match in re.finditer(r"^c\d+:\s+COM\s+ATOMS=([0-9,:-]+)", text, re.MULTILINE):
        n2_atoms.extend(parse_atom_selection(match.group(1)))
    if not n2_atoms:
        raise ValueError(f"No N2 atom pairs found in {path}")
    return SelectionMap(
        surf_si_top=_parse_group(text, "surf_si_top", required=True),
        surface_ch3=_parse_group(text, "surface_ch3"),
        surface_oh=_parse_group(text, "surface_oh"),
        water_o=_parse_group(text, "water_o", required=True),
        n2_atoms=tuple(sorted(set(n2_atoms))),
    )


def parse_lammps_atom_types(path: Path) -> Dict[int, int]:
    types: Dict[int, int] = {}
    in_atoms = False
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not in_atoms:
                if line.startswith("Atoms"):
                    in_atoms = True
                continue
            if not line:
                continue
            parts = line.split()
            if not parts[0].lstrip("+-").isdigit():
                if types:
                    break
                continue
            if len(parts) < 2:
                continue
            types[int(parts[0])] = int(parts[1])
    if not types:
        raise ValueError(f"No atom types parsed from {path}")
    return types


def _segment_sort_key(path: Path) -> Tuple[int, float, str]:
    try:
        return (0, float(path.name), path.name)
    except ValueError:
        match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", path.name)
        if match:
            return (1, float(match.group(1)), path.name)
        return (2, math.inf, path.name)


def count_dump_frames(path: Path) -> int:
    count = 0
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ITEM: TIMESTEP"):
                count += 1
    return count


def discover_segments(run_dir: Path, dump_name: str, plumed_name: str) -> List[SegmentSpec]:
    candidates = [
        child
        for child in Path(run_dir).iterdir()
        if child.is_dir() and (child / dump_name).is_file() and (child / plumed_name).is_file()
    ]
    candidates.sort(key=_segment_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No trajectory segments with {dump_name} and {plumed_name} under {run_dir}")
    return [
        SegmentSpec(
            label=child.name,
            segment_dir=child,
            dump_path=child / dump_name,
            plumed_path=child / plumed_name,
            frame_count=count_dump_frames(child / dump_name),
        )
        for child in candidates
    ]


def _choose_data_file(run_dir: Path, data_name: Optional[str] = None) -> Path:
    if data_name:
        candidate = Path(run_dir) / data_name
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing requested LAMMPS data file: {candidate}")
        return candidate
    candidates = sorted(Path(run_dir).glob("*.data"))
    if candidates:
        return candidates[-1]
    for child in sorted(Path(run_dir).iterdir(), key=_segment_sort_key):
        if child.is_dir():
            nested = sorted(child.glob("*.data"))
            if nested:
                return nested[-1]
    raise FileNotFoundError(f"No LAMMPS data file found under {run_dir}")


def _coord_field(fields: Sequence[str], dim: str) -> Tuple[int, bool]:
    for name in (dim, dim + "u", dim + "s"):
        if name in fields:
            return fields.index(name), name.endswith("s")
    raise ValueError(f"Missing {dim} coordinate in dump")


def iter_selected_frames(
    segments: Sequence[SegmentSpec],
    selected_global_indices: Set[int],
    needed_atom_ids: Set[int],
) -> Iterator[DumpFrame]:
    global_offset = 0
    for segment in segments:
        with segment.dump_path.open(encoding="utf-8", errors="replace") as handle:
            local_index = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.startswith("ITEM: TIMESTEP"):
                    raise ValueError(f"Unexpected dump format in {segment.dump_path}")
                timestep = int(handle.readline().strip())
                if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError("Missing NUMBER OF ATOMS")
                n_atoms = int(handle.readline().strip())
                if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                    raise ValueError("Missing BOX BOUNDS")
                bounds = np.zeros((3, 2), dtype=float)
                for dim in range(3):
                    parts = handle.readline().split()
                    bounds[dim, :] = [float(parts[0]), float(parts[1])]
                header = handle.readline().strip()
                fields = header.split()[2:]
                global_index = global_offset + local_index
                selected = global_index in selected_global_indices
                if not selected:
                    for _ in range(n_atoms):
                        handle.readline()
                    local_index += 1
                    continue
                id_index = fields.index("id")
                x_index, x_scaled = _coord_field(fields, "x")
                y_index, y_scaled = _coord_field(fields, "y")
                z_index, z_scaled = _coord_field(fields, "z")
                lengths = box_lengths(bounds)
                positions: Dict[int, np.ndarray] = {}
                for _ in range(n_atoms):
                    parts = handle.readline().split()
                    atom_id = int(parts[id_index])
                    if atom_id not in needed_atom_ids:
                        continue
                    coords = np.asarray(
                        [float(parts[x_index]), float(parts[y_index]), float(parts[z_index])],
                        dtype=float,
                    )
                    for dim, scaled in enumerate((x_scaled, y_scaled, z_scaled)):
                        if scaled:
                            coords[dim] = bounds[dim, 0] + coords[dim] * lengths[dim]
                    positions[atom_id] = coords
                yield DumpFrame(
                    global_index=global_index,
                    segment_label=segment.label,
                    local_index=local_index,
                    timestep=timestep,
                    bounds=bounds,
                    positions=positions,
                )
                local_index += 1
        global_offset += segment.frame_count


def _coords(positions: Mapping[int, np.ndarray], atom_ids: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
    ids = [atom_id for atom_id in atom_ids if atom_id in positions]
    if not ids:
        return np.empty((0, 3), dtype=float), []
    return np.asarray([positions[atom_id] for atom_id in ids], dtype=float), ids


def _wrap(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lengths = box_lengths(bounds)
    return np.mod(np.asarray(coords, dtype=float) - bounds[:, 0], lengths)


def _cross_pairs(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    cutoff_A: float,
    bounds: np.ndarray,
) -> List[Tuple[int, int, float]]:
    if left_coords.size == 0 or right_coords.size == 0:
        return []
    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    tree = cKDTree(_wrap(right_coords, bounds), boxsize=lengths)
    neighborhoods = tree.query_ball_point(_wrap(left_coords, bounds), cutoff_A)
    pairs: List[Tuple[int, int, float]] = []
    for left_index, right_indices in enumerate(neighborhoods):
        if not right_indices:
            continue
        vectors = minimum_image_vectors(
            right_coords[np.asarray(right_indices, dtype=int)] - left_coords[left_index],
            lengths,
        )
        distances = np.linalg.norm(vectors, axis=1)
        pairs.extend(
            (left_index, int(right_index), float(distance))
            for right_index, distance in zip(right_indices, distances)
            if distance <= cutoff_A
        )
    return pairs


def _periodic_xy_center(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    center = np.zeros(2, dtype=float)
    for dim in (0, 1):
        lo = float(bounds[dim, 0])
        length = float(bounds[dim, 1] - bounds[dim, 0])
        angles = 2.0 * np.pi * ((coords[:, dim] - lo) / length)
        mean_complex = np.exp(1j * angles).mean()
        angle = np.angle(mean_complex)
        if angle < 0.0:
            angle += 2.0 * np.pi
        center[dim] = lo + angle / (2.0 * np.pi) * length
    return center


def _angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if norm <= 0.0:
        return math.nan
    cosine = float(np.clip(np.dot(vector_a, vector_b) / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if array.size == 0:
        return math.nan, math.nan
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def _fraction_equal(values: np.ndarray, target: int) -> float:
    return float(np.mean(values == target)) if values.size else math.nan


def _graph_metrics(nodes: Sequence[int], edges: Set[Tuple[int, int]]) -> Dict[str, float]:
    node_set = set(nodes)
    adjacency: Dict[int, List[int]] = {node: [] for node in node_set}
    for first, second in edges:
        if first in adjacency and second in adjacency:
            adjacency[first].append(second)
            adjacency[second].append(first)
    components: List[Set[int]] = []
    seen: Set[int] = set()
    for node in node_set:
        if node in seen:
            continue
        component: Set[int] = set()
        queue: deque[int] = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    largest = max((len(component) for component in components), default=0)
    cycle_rank = max(0, len(edges) - len(node_set) + len(components))
    return {
        "network_component_count": float(len(components)),
        "network_largest_component_fraction": largest / max(len(node_set), 1),
        "network_cycle_rank": float(cycle_rank),
        "network_cycle_rank_per_node": cycle_rank / max(len(node_set), 1),
    }


def _subgraph_metrics(node_ids: Sequence[int], edges: Set[Tuple[int, int]]) -> Dict[str, float]:
    nodes = set(node_ids)
    selected_edges = {edge for edge in edges if edge[0] in nodes and edge[1] in nodes}
    graph = _graph_metrics(sorted(nodes), selected_edges)
    degrees = defaultdict(int)
    for first, second in selected_edges:
        degrees[first] += 1
        degrees[second] += 1
    graph["avg_degree"] = 2.0 * len(selected_edges) / max(len(nodes), 1)
    graph["isolated_fraction"] = (
        sum(1 for node in nodes if degrees[node] == 0) / max(len(nodes), 1)
    )
    graph["edge_count"] = float(len(selected_edges))
    graph["node_count"] = float(len(nodes))
    return graph


def _nearest_mapping(
    anchor_ids: Sequence[int],
    neighbor_ids: Sequence[int],
    positions: Mapping[int, np.ndarray],
    bounds: np.ndarray,
    cutoff_A: float,
) -> Dict[int, int]:
    anchor_coords, kept_anchors = _coords(positions, anchor_ids)
    neighbor_coords, kept_neighbors = _coords(positions, neighbor_ids)
    mapping: Dict[int, int] = {}
    by_anchor: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for left, right, distance in _cross_pairs(anchor_coords, neighbor_coords, cutoff_A, bounds):
        by_anchor[left].append((right, distance))
    for left, candidates in by_anchor.items():
        right = min(candidates, key=lambda item: item[1])[0]
        mapping[kept_anchors[left]] = kept_neighbors[right]
    return mapping


def _surface_rigid_shift(
    frame: DumpFrame,
    reference: ReferenceState,
) -> np.ndarray:
    lengths = box_lengths(frame.bounds)
    displacements = [
        minimum_image_vectors(frame.positions[atom_id] - reference_position, lengths)
        for atom_id, reference_position in reference.surf_si_positions.items()
        if atom_id in frame.positions
    ]
    if not displacements:
        return np.zeros(3, dtype=float)
    return np.mean(np.asarray(displacements, dtype=float), axis=0)


def _build_reference(
    frame: DumpFrame,
    selections: SelectionMap,
    si_ids: Sequence[int],
    o_ids: Sequence[int],
    c_ids: Sequence[int],
    substrate_h_ids: Sequence[int],
    config: AnalysisConfig,
) -> ReferenceState:
    si_coords, kept_si = _coords(frame.positions, si_ids)
    o_coords, kept_o = _coords(frame.positions, o_ids)
    c_coords, kept_c = _coords(frame.positions, c_ids)
    ligand_edges: Set[Tuple[int, int, str]] = set()
    o_coord = defaultdict(int)
    for si_index, o_index, _ in _cross_pairs(si_coords, o_coords, config.si_o_cutoff_A, frame.bounds):
        ligand_edges.add((kept_si[si_index], kept_o[o_index], "O"))
        o_coord[kept_o[o_index]] += 1
    for si_index, c_index, _ in _cross_pairs(si_coords, c_coords, config.si_c_cutoff_A, frame.bounds):
        ligand_edges.add((kept_si[si_index], kept_c[c_index], "C"))
    terminal_ids = tuple(selections.surface_ch3) + tuple(selections.surface_oh)
    terminal_si = _nearest_mapping(
        terminal_ids,
        si_ids,
        frame.positions,
        frame.bounds,
        max(config.si_o_cutoff_A, config.si_c_cutoff_A),
    )
    surface_oh_h = _nearest_mapping(
        selections.surface_oh,
        substrate_h_ids,
        frame.positions,
        frame.bounds,
        config.covalent_oh_cutoff_A,
    )
    return ReferenceState(
        ligand_edges=ligand_edges,
        backbone_o_ids={atom_id for atom_id, count in o_coord.items() if count >= 2},
        surf_si_positions={
            atom_id: np.asarray(frame.positions[atom_id], dtype=float)
            for atom_id in selections.surf_si_top
            if atom_id in frame.positions
        },
        terminal_positions={
            atom_id: np.asarray(frame.positions[atom_id], dtype=float)
            for atom_id in terminal_ids
            if atom_id in frame.positions
        },
        terminal_si_by_id=terminal_si,
        surface_oh_h_by_id=surface_oh_h,
    )


def _bond_angle_samples(
    si_coords: np.ndarray,
    si_ids: Sequence[int],
    o_coords: np.ndarray,
    o_ids: Sequence[int],
    pairs: Sequence[Tuple[int, int, float]],
    bounds: np.ndarray,
) -> Tuple[List[float], List[float]]:
    lengths = box_lengths(bounds)
    o_by_si: Dict[int, List[int]] = defaultdict(list)
    si_by_o: Dict[int, List[int]] = defaultdict(list)
    for si_index, o_index, _ in pairs:
        o_by_si[si_index].append(o_index)
        si_by_o[o_index].append(si_index)
    o_si_o: List[float] = []
    for si_index, neighbors in o_by_si.items():
        vectors = minimum_image_vectors(o_coords[np.asarray(neighbors)] - si_coords[si_index], lengths)
        for first in range(len(vectors)):
            for second in range(first + 1, len(vectors)):
                o_si_o.append(_angle_deg(vectors[first], vectors[second]))
    si_o_si: List[float] = []
    for o_index, neighbors in si_by_o.items():
        vectors = minimum_image_vectors(si_coords[np.asarray(neighbors)] - o_coords[o_index], lengths)
        for first in range(len(vectors)):
            for second in range(first + 1, len(vectors)):
                si_o_si.append(_angle_deg(vectors[first], vectors[second]))
    return o_si_o, si_o_si


def compute_topology_frame_metrics(
    frame: DumpFrame,
    selections: SelectionMap,
    si_ids: Sequence[int],
    o_ids: Sequence[int],
    c_ids: Sequence[int],
    reference: ReferenceState,
    config: AnalysisConfig,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    si_coords, kept_si = _coords(frame.positions, si_ids)
    o_coords, kept_o = _coords(frame.positions, o_ids)
    c_coords, kept_c = _coords(frame.positions, c_ids)
    si_index = {atom_id: index for index, atom_id in enumerate(kept_si)}
    o_pairs = _cross_pairs(si_coords, o_coords, config.si_o_cutoff_A, frame.bounds)
    c_pairs = _cross_pairs(si_coords, c_coords, config.si_c_cutoff_A, frame.bounds)
    si_o_coord = np.zeros(len(kept_si), dtype=int)
    si_c_coord = np.zeros(len(kept_si), dtype=int)
    o_si_coord = np.zeros(len(kept_o), dtype=int)
    current_edges: Set[Tuple[int, int, str]] = set()
    bond_lengths: List[float] = []
    for si_local, o_local, distance in o_pairs:
        si_o_coord[si_local] += 1
        o_si_coord[o_local] += 1
        current_edges.add((kept_si[si_local], kept_o[o_local], "O"))
        bond_lengths.append(distance)
    si_c_lengths: List[float] = []
    for si_local, c_local, distance in c_pairs:
        si_c_coord[si_local] += 1
        current_edges.add((kept_si[si_local], kept_c[c_local], "C"))
        si_c_lengths.append(distance)
    ligand_coord = si_o_coord + si_c_coord
    surface_indices = np.asarray(
        [si_index[atom_id] for atom_id in selections.surf_si_top if atom_id in si_index],
        dtype=int,
    )
    initial = reference.ligand_edges
    survival = len(current_edges & initial) / max(len(initial), 1)
    new_fraction = len(current_edges - initial) / max(len(current_edges), 1)
    backbone_o_local = [
        index for index, atom_id in enumerate(kept_o) if atom_id in reference.backbone_o_ids
    ]
    backbone_coord = o_si_coord[np.asarray(backbone_o_local, dtype=int)] if backbone_o_local else np.asarray([])
    network_edges = {
        (si_id, ligand_id)
        for si_id, ligand_id, ligand_type in current_edges
        if ligand_type == "O" and ligand_id in reference.backbone_o_ids
    }
    network_nodes = list(kept_si) + list(reference.backbone_o_ids)
    graph = _graph_metrics(network_nodes, network_edges)
    o_si_o, si_o_si = _bond_angle_samples(
        si_coords, kept_si, o_coords, kept_o, o_pairs, frame.bounds
    )
    lengths = box_lengths(frame.bounds)
    rigid_shift = _surface_rigid_shift(frame, reference)
    surface_displacements = []
    surface_z = []
    for atom_id, reference_position in reference.surf_si_positions.items():
        if atom_id not in frame.positions:
            continue
        delta = minimum_image_vectors(frame.positions[atom_id] - reference_position, lengths)
        delta = delta - rigid_shift
        surface_displacements.append(float(np.linalg.norm(delta)))
        surface_z.append(float(frame.positions[atom_id][2]))
    metrics = {
        "si_ligand_coord4_fraction": _fraction_equal(ligand_coord, 4),
        "si_ligand_undercoord_fraction": float(np.mean(ligand_coord < 4)),
        "si_ligand_overcoord_fraction": float(np.mean(ligand_coord > 4)),
        "surface_si_ligand_coord4_fraction": (
            _fraction_equal(ligand_coord[surface_indices], 4) if surface_indices.size else math.nan
        ),
        "surface_si_o_coord_mean": (
            float(np.mean(si_o_coord[surface_indices])) if surface_indices.size else math.nan
        ),
        "surface_si_c_coord_mean": (
            float(np.mean(si_c_coord[surface_indices])) if surface_indices.size else math.nan
        ),
        "bridging_o_coord2_fraction": _fraction_equal(backbone_coord, 2),
        "si_o_bond_length_mean_A": _mean_std(bond_lengths)[0],
        "si_o_bond_length_std_A": _mean_std(bond_lengths)[1],
        "si_c_bond_length_mean_A": _mean_std(si_c_lengths)[0],
        "o_si_o_angle_mean_deg": _mean_std(o_si_o)[0],
        "o_si_o_angle_std_deg": _mean_std(o_si_o)[1],
        "si_o_si_angle_mean_deg": _mean_std(si_o_si)[0],
        "si_o_si_angle_std_deg": _mean_std(si_o_si)[1],
        "ligand_edge_survival_fraction": survival,
        "ligand_new_edge_fraction": new_fraction,
        "surface_si_displacement_rms_A": (
            float(np.sqrt(np.mean(np.square(surface_displacements))))
            if surface_displacements
            else math.nan
        ),
        "surface_rigid_shift_A": float(np.linalg.norm(rigid_shift)),
        "surface_si_z_roughness_A": float(np.std(surface_z, ddof=1)) if len(surface_z) > 1 else 0.0,
        **graph,
    }
    samples = {
        "si_o_bond_length_A": bond_lengths,
        "si_c_bond_length_A": si_c_lengths,
        "o_si_o_angle_deg": o_si_o,
        "si_o_si_angle_deg": si_o_si,
    }
    return metrics, samples


def compute_termination_frame_metrics(
    frame: DumpFrame,
    selections: SelectionMap,
    reference: ReferenceState,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    lengths = box_lengths(frame.bounds)
    normal = np.asarray([0.0, 0.0, 1.0])
    rigid_shift = _surface_rigid_shift(frame, reference)
    ch3_tilts: List[float] = []
    oh_tilts: List[float] = []
    oh_bond_tilts: List[float] = []
    displacements: List[float] = []
    terminal_z: List[float] = []
    for atom_id, si_id in reference.terminal_si_by_id.items():
        if atom_id not in frame.positions or si_id not in frame.positions:
            continue
        vector = minimum_image_vectors(frame.positions[atom_id] - frame.positions[si_id], lengths)
        tilt = _angle_deg(vector, normal)
        if atom_id in selections.surface_ch3:
            ch3_tilts.append(tilt)
        elif atom_id in selections.surface_oh:
            oh_tilts.append(tilt)
        terminal_z.append(float(frame.positions[atom_id][2]))
        if atom_id in reference.terminal_positions:
            delta = minimum_image_vectors(
                frame.positions[atom_id] - reference.terminal_positions[atom_id], lengths
            )
            delta = delta - rigid_shift
            displacements.append(float(np.linalg.norm(delta)))
    for oxygen_id, hydrogen_id in reference.surface_oh_h_by_id.items():
        if oxygen_id not in frame.positions or hydrogen_id not in frame.positions:
            continue
        vector = minimum_image_vectors(
            frame.positions[hydrogen_id] - frame.positions[oxygen_id], lengths
        )
        oh_bond_tilts.append(_angle_deg(vector, normal))
    metrics = {
        "ch3_tilt_mean_deg": _mean_std(ch3_tilts)[0],
        "ch3_tilt_std_deg": _mean_std(ch3_tilts)[1],
        "oh_anchor_tilt_mean_deg": _mean_std(oh_tilts)[0],
        "oh_anchor_tilt_std_deg": _mean_std(oh_tilts)[1],
        "surface_oh_bond_tilt_mean_deg": _mean_std(oh_bond_tilts)[0],
        "surface_oh_bond_tilt_std_deg": _mean_std(oh_bond_tilts)[1],
        "terminal_anchor_displacement_rms_A": (
            float(np.sqrt(np.mean(np.square(displacements)))) if displacements else math.nan
        ),
        "terminal_anchor_z_roughness_A": (
            float(np.std(terminal_z, ddof=1)) if len(terminal_z) > 1 else 0.0
        ),
    }
    return metrics, {
        "ch3_tilt_deg": ch3_tilts,
        "oh_anchor_tilt_deg": oh_tilts,
        "surface_oh_bond_tilt_deg": oh_bond_tilts,
    }


def _water_orientation_and_regions(
    frame: DumpFrame,
    selections: SelectionMap,
    z_ref: float,
    hydrogen_assignment: Mapping[int, Sequence[int]],
    config: AnalysisConfig,
) -> Dict[str, object]:
    lengths = box_lengths(frame.bounds)
    n2_coords, _ = _coords(frame.positions, selections.n2_atoms)
    center_xy = _periodic_xy_center(n2_coords, frame.bounds)
    records: List[Dict[str, object]] = []
    for oxygen_id in selections.water_o:
        if oxygen_id not in frame.positions:
            continue
        hydrogen_ids = tuple(hydrogen_assignment.get(oxygen_id, ()))
        oxygen = frame.positions[oxygen_id]
        z_rel = float(oxygen[2] - z_ref)
        if not (config.interface_z_min_A <= z_rel <= config.interface_z_max_A):
            continue
        radial_delta = minimum_image_vectors(
            np.asarray([oxygen[0] - center_xy[0], oxygen[1] - center_xy[1], 0.0]),
            lengths,
        )
        radius = float(np.linalg.norm(radial_delta[:2]))
        if radius <= config.bubble_radius_A:
            region = "under_bubble"
        elif radius >= config.bubble_radius_A + config.outside_margin_A:
            region = "outside_bubble"
        else:
            region = "contact_line"
        valid = len(hydrogen_ids) == 2
        if valid:
            h_coords = np.asarray([frame.positions[atom_id] for atom_id in hydrogen_ids])
            oh_vectors = minimum_image_vectors(h_coords - oxygen, lengths)
            dipole = np.mean(oh_vectors, axis=0)
            dipole_norm = float(np.linalg.norm(dipole))
            cos_theta = float(dipole[2] / dipole_norm) if dipole_norm > 0.0 else math.nan
        else:
            oh_vectors = np.empty((0, 3), dtype=float)
            cos_theta = math.nan
        records.append(
            {
                "oxygen_id": oxygen_id,
                "region": region,
                "z_rel_A": z_rel,
                "radius_A": radius,
                "cos_theta": cos_theta,
                "S": 0.5 * (3.0 * cos_theta * cos_theta - 1.0)
                if math.isfinite(cos_theta)
                else math.nan,
                "valid": valid,
                "h_count": len(hydrogen_ids),
                "oxygen": oxygen,
                "hydrogen_ids": hydrogen_ids,
                "oh_vectors": oh_vectors,
            }
        )
    return {"records": records, "n2_center_xy": center_xy}


def _assign_hydrogens_to_nearest_oxygen(
    frame: DumpFrame,
    oxygen_ids: Sequence[int],
    hydrogen_ids: Sequence[int],
    cutoff_A: float,
) -> Dict[int, Tuple[int, ...]]:
    from scipy.spatial import cKDTree

    oxygen_coords, kept_oxygen = _coords(frame.positions, oxygen_ids)
    hydrogen_coords, kept_hydrogen = _coords(frame.positions, hydrogen_ids)
    if oxygen_coords.size == 0 or hydrogen_coords.size == 0:
        return {}
    lengths = box_lengths(frame.bounds)
    tree = cKDTree(_wrap(oxygen_coords, frame.bounds), boxsize=lengths)
    distances, nearest = tree.query(
        _wrap(hydrogen_coords, frame.bounds),
        k=1,
        distance_upper_bound=cutoff_A,
    )
    grouped: Dict[int, List[int]] = defaultdict(list)
    for hydrogen_index, (distance, oxygen_index) in enumerate(zip(distances, nearest)):
        if not math.isfinite(float(distance)) or int(oxygen_index) >= len(kept_oxygen):
            continue
        grouped[kept_oxygen[int(oxygen_index)]].append(kept_hydrogen[hydrogen_index])
    return {oxygen_id: tuple(hydrogens) for oxygen_id, hydrogens in grouped.items()}


def _water_hbond_edges(
    frame: DumpFrame,
    records: Sequence[Mapping[str, object]],
    config: AnalysisConfig,
) -> Set[Tuple[int, int]]:
    if not records:
        return set()
    from scipy.spatial import cKDTree

    valid_records = [record for record in records if bool(record["valid"])]
    if not valid_records:
        return set()
    oxygen_coords = np.asarray([record["oxygen"] for record in valid_records], dtype=float)
    oxygen_ids = [int(record["oxygen_id"]) for record in valid_records]
    lengths = box_lengths(frame.bounds)
    tree = cKDTree(_wrap(oxygen_coords, frame.bounds), boxsize=lengths)
    candidate_pairs = tree.query_pairs(config.hbond_oo_cutoff_A)
    cosine_cutoff = math.cos(math.radians(config.hbond_angle_deg))
    edges: Set[Tuple[int, int]] = set()
    for first, second in candidate_pairs:
        vector_ij = minimum_image_vectors(oxygen_coords[second] - oxygen_coords[first], lengths)
        distance = float(np.linalg.norm(vector_ij))
        if distance <= 0.0:
            continue
        unit_ij = vector_ij / distance
        first_vectors = np.asarray(valid_records[first]["oh_vectors"], dtype=float)
        second_vectors = np.asarray(valid_records[second]["oh_vectors"], dtype=float)
        first_unit = first_vectors / np.linalg.norm(first_vectors, axis=1)[:, None]
        second_unit = second_vectors / np.linalg.norm(second_vectors, axis=1)[:, None]
        donor_first = bool(np.any(first_unit @ unit_ij >= cosine_cutoff))
        donor_second = bool(np.any(second_unit @ (-unit_ij) >= cosine_cutoff))
        if donor_first or donor_second:
            edges.add(tuple(sorted((oxygen_ids[first], oxygen_ids[second]))))
    return edges


def _surface_water_hbonds(
    frame: DumpFrame,
    records: Sequence[Mapping[str, object]],
    surface_oh_ids: Sequence[int],
    hydrogen_assignment: Mapping[int, Sequence[int]],
    config: AnalysisConfig,
) -> Tuple[int, int]:
    valid_records = [record for record in records if bool(record["valid"])]
    if not valid_records or not surface_oh_ids:
        return 0, 0
    water_coords = np.asarray([record["oxygen"] for record in valid_records], dtype=float)
    lengths = box_lengths(frame.bounds)
    cosine_cutoff = math.cos(math.radians(config.hbond_angle_deg))
    surface_donor = 0
    water_donor = 0
    for surface_o in surface_oh_ids:
        assigned_h = tuple(hydrogen_assignment.get(surface_o, ()))
        if surface_o not in frame.positions:
            continue
        oxygen = frame.positions[surface_o]
        candidates = _cross_pairs(
            np.asarray([oxygen]), water_coords, config.hbond_oo_cutoff_A, frame.bounds
        )
        surface_oh = None
        if len(assigned_h) == 1 and assigned_h[0] in frame.positions:
            surface_oh = minimum_image_vectors(frame.positions[assigned_h[0]] - oxygen, lengths)
            surface_oh /= max(float(np.linalg.norm(surface_oh)), 1e-12)
        for _, water_index, _ in candidates:
            vector = minimum_image_vectors(water_coords[water_index] - oxygen, lengths)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            if surface_oh is not None and float(np.dot(surface_oh, vector)) >= cosine_cutoff:
                surface_donor += 1
            water_vectors = np.asarray(valid_records[water_index]["oh_vectors"], dtype=float)
            water_unit = water_vectors / np.linalg.norm(water_vectors, axis=1)[:, None]
            if bool(np.any(water_unit @ (-vector) >= cosine_cutoff)):
                water_donor += 1
    return surface_donor, water_donor


def compute_water_hbond_metrics(
    frame: DumpFrame,
    selections: SelectionMap,
    reference: ReferenceState,
    oxygen_candidate_ids: Sequence[int],
    hydrogen_candidate_ids: Sequence[int],
    z_ref: float,
    config: AnalysisConfig,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    hydrogen_assignment = _assign_hydrogens_to_nearest_oxygen(
        frame,
        oxygen_candidate_ids,
        hydrogen_candidate_ids,
        config.water_oh_valid_cutoff_A,
    )
    payload = _water_orientation_and_regions(
        frame,
        selections,
        z_ref,
        hydrogen_assignment,
        config,
    )
    records = payload["records"]
    edges = _water_hbond_edges(frame, records, config)
    surface_donor, water_donor = _surface_water_hbonds(
        frame,
        records,
        selections.surface_oh,
        hydrogen_assignment,
        config,
    )
    water_h_counts = np.asarray([int(record["h_count"]) for record in records], dtype=int)
    surface_oh_protonated = sum(
        len(hydrogen_assignment.get(oxygen_id, ())) == 1
        for oxygen_id in selections.surface_oh
    )
    metrics: Dict[str, float] = {
        "interface_water_count": float(len(records)),
        "water_h2o_fraction": float(np.mean(water_h_counts == 2)) if water_h_counts.size else math.nan,
        "water_oh_fraction": float(np.mean(water_h_counts == 1)) if water_h_counts.size else math.nan,
        "water_h3o_fraction": float(np.mean(water_h_counts == 3)) if water_h_counts.size else math.nan,
        "water_unprotonated_o_fraction": (
            float(np.mean(water_h_counts == 0)) if water_h_counts.size else math.nan
        ),
        "surface_oh_protonated_fraction": (
            surface_oh_protonated / len(selections.surface_oh)
            if selections.surface_oh
            else math.nan
        ),
        "water_water_hbond_count": float(len(edges)),
        "surfaceOH_donor_water_hbond_count": float(surface_donor),
        "water_donor_surfaceOH_hbond_count": float(water_donor),
        "surface_water_hbond_total": float(surface_donor + water_donor),
        "surface_water_hbond_per_surfaceOH": (
            (surface_donor + water_donor) / max(len(selections.surface_oh), 1)
            if selections.surface_oh
            else 0.0
        ),
    }
    orientation_samples: Dict[str, List[float]] = defaultdict(list)
    region_by_id = {int(record["oxygen_id"]): str(record["region"]) for record in records}
    for region in ("all", "under_bubble", "contact_line", "outside_bubble"):
        selected = records if region == "all" else [record for record in records if record["region"] == region]
        cos_values = [
            float(record["cos_theta"])
            for record in selected
            if math.isfinite(float(record["cos_theta"]))
        ]
        s_values = [
            float(record["S"]) for record in selected if math.isfinite(float(record["S"]))
        ]
        metrics[f"water_{region}_count"] = float(len(selected))
        metrics[f"water_{region}_cos_mean"] = _mean_std(cos_values)[0]
        metrics[f"water_{region}_S_mean"] = _mean_std(s_values)[0]
        orientation_samples[region].extend(cos_values)
        node_ids = [int(record["oxygen_id"]) for record in selected]
        graph = _subgraph_metrics(node_ids, edges)
        metrics[f"hbond_{region}_avg_degree"] = graph["avg_degree"]
        metrics[f"hbond_{region}_largest_component_fraction"] = graph[
            "network_largest_component_fraction"
        ]
        metrics[f"hbond_{region}_isolated_fraction"] = graph["isolated_fraction"]
        metrics[f"hbond_{region}_edge_count"] = graph["edge_count"]
    under_nodes = {atom_id for atom_id, region in region_by_id.items() if region == "under_bubble"}
    outside_nodes = {atom_id for atom_id, region in region_by_id.items() if region == "outside_bubble"}
    metrics["hbond_edges_crossing_under_boundary"] = float(
        sum(
            1
            for first, second in edges
            if (first in under_nodes and second not in under_nodes)
            or (second in under_nodes and first not in under_nodes)
        )
    )
    metrics["hbond_edges_crossing_outside_boundary"] = float(
        sum(
            1
            for first, second in edges
            if (first in outside_nodes and second not in outside_nodes)
            or (second in outside_nodes and first not in outside_nodes)
        )
    )
    return metrics, dict(orientation_samples)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _ordered_fields(rows: Sequence[Mapping[str, object]], preferred: Sequence[str]) -> List[str]:
    keys = sorted({key for row in rows for key in row})
    return list(preferred) + [key for key in keys if key not in preferred]


def _aggregate_frame_rows(frame_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row["case"])].append(row)
    summaries: List[Dict[str, object]] = []
    metadata_fields = {"case", "run_dir", "ch3_fraction", "global_frame", "segment", "timestep", "time_ps"}
    for case, rows in grouped.items():
        summary: Dict[str, object] = {
            "case": case,
            "run_dir": rows[0]["run_dir"],
            "ch3_fraction": rows[0]["ch3_fraction"],
            "frames_used": len(rows),
            "time_first_ps": min(float(row["time_ps"]) for row in rows),
            "time_last_ps": max(float(row["time_ps"]) for row in rows),
        }
        numeric_fields = sorted(set(rows[0]) - metadata_fields)
        for field in numeric_fields:
            values = np.asarray(
                [
                    float(row[field])
                    for row in rows
                    if field in row and math.isfinite(float(row[field]))
                ],
                dtype=float,
            )
            summary[f"{field}_mean"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else math.nan
            )
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (float(row["ch3_fraction"]), str(row["case"])),
    )


def _histogram_rows(
    samples: Mapping[Tuple[str, str], List[float]],
    definitions: Mapping[str, Tuple[float, float, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (case, metric), values in sorted(samples.items()):
        if metric not in definitions or not values:
            continue
        lower, upper, bins = definitions[metric]
        counts, edges = np.histogram(values, bins=bins, range=(lower, upper), density=False)
        total = max(int(np.sum(counts)), 1)
        widths = np.diff(edges)
        density = counts / total / widths
        for index, count in enumerate(counts):
            rows.append(
                {
                    "case": case,
                    "metric": metric,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "bin_center": 0.5 * (edges[index] + edges[index + 1]),
                    "count": int(count),
                    "density": density[index],
                }
            )
    return rows


def _use_chart_theme() -> None:
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
        },
    )


def _chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, 78, break_long_words=False)
    subtitle = textwrap.fill(subtitle, 112, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    subtitle_y = 0.925 - 0.052 * (title_lines - 1)
    axes_top = subtitle_y - 0.095 - 0.026 * (subtitle_lines - 1)
    fig.subplots_adjust(top=max(0.58, axes_top))
    left = ax.get_position().x0
    fig.text(left, 0.98, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, subtitle_y, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def _make_plots(
    output_dir: Path,
    frame_rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    distribution_rows: Sequence[Mapping[str, object]],
    config: AnalysisConfig,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    _use_chart_theme()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    summary_df = pd.DataFrame(summaries)
    frame_df = pd.DataFrame(frame_rows)
    dist_df = pd.DataFrame(distribution_rows)
    case_order = [row["case"] for row in summaries]
    colors = sns.color_palette("colorblind", n_colors=max(len(case_order), 1))
    palette = {case: colors[index] for index, case in enumerate(case_order)}

    topology_metrics = [
        ("surface_si_ligand_coord4_fraction_mean", "Surface Si four-ligand fraction"),
        ("bridging_o_coord2_fraction_mean", "Bridging O two-Si fraction"),
        ("ligand_edge_survival_fraction_mean", "Initial ligand-edge survival"),
    ]
    long_rows = [
        {"case": row["case"], "metric": label, "value": row.get(field, math.nan)}
        for row in summaries
        for field, label in topology_metrics
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.barplot(
        data=pd.DataFrame(long_rows),
        x="case",
        y="value",
        hue="metric",
        order=case_order,
        palette=["#A3BEFA", "#A3D576", "#FFE15B"],
        edgecolor=TOKENS["ink"],
        linewidth=0.8,
        ax=ax,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Termination case")
    ax.set_ylabel("Fraction")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3)
    _chart_header(
        fig,
        ax,
        "Silica network integrity should be judged with O and C ligands together",
        "Late-window averages; Si-C terminations are counted as valid surface ligands rather than missing Si-O bonds.",
    )
    path = figure_dir / "surface_topology_integrity.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    comparisons = [
        ("surface_si_displacement_rms_A_mean", "Surface Si displacement (A)"),
        ("surface_si_z_roughness_A_mean", "Surface Si z roughness (A)"),
        ("terminal_anchor_displacement_rms_A_mean", "Terminal displacement (A)"),
    ]
    for ax, (field, ylabel) in zip(axes, comparisons):
        sns.barplot(
            data=summary_df,
            x="case",
            y=field,
            order=case_order,
            palette=palette,
            edgecolor=TOKENS["ink"],
            linewidth=0.8,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
    _chart_header(
        fig,
        axes[0],
        "Separate silica backbone motion from flexible terminal-group disorder",
        "RMS displacement is referenced to the first available analyzed trajectory frame; terminal motion can be large while surface Si remains stable.",
    )
    path = figure_dir / "surface_motion_and_roughness.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    orientation = dist_df[dist_df["metric"].astype(str).str.startswith("water_cos_")]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for case in case_order:
        part = orientation[
            (orientation["case"] == case) & (orientation["metric"] == "water_cos_under_bubble")
        ]
        if part.empty:
            continue
        sns.lineplot(
            data=part,
            x="bin_center",
            y="density",
            color=palette[case],
            linewidth=1.2,
            label=case,
            ax=ax,
        )
    ax.set_xlabel("Water dipole cos(theta) relative to outward surface normal")
    ax.set_ylabel("Probability density")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=4)
    _chart_header(
        fig,
        ax,
        "Under-bubble water orientation resolves termination-dependent interfacial ordering",
        f"Water oxygens at z={config.interface_z_min_A:g}-{config.interface_z_max_A:g} A relative to the top-Si reference; latest selected frames.",
    )
    path = figure_dir / "under_bubble_water_orientation.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    hbond_metrics = [
        ("hbond_under_bubble_avg_degree_mean", "Under-bubble average degree"),
        ("hbond_outside_bubble_avg_degree_mean", "Outside-bubble average degree"),
        ("surface_water_hbond_per_surfaceOH_mean", "Surface-water H-bond per OH"),
    ]
    long_rows = [
        {"case": row["case"], "metric": label, "value": row.get(field, math.nan)}
        for row in summaries
        for field, label in hbond_metrics
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    sns.barplot(
        data=pd.DataFrame(long_rows),
        x="case",
        y="value",
        hue="metric",
        order=case_order,
        palette=["#A3BEFA", "#A3D576", "#F0986E"],
        edgecolor=TOKENS["ink"],
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Termination case")
    ax.set_ylabel("Network metric")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3)
    _chart_header(
        fig,
        ax,
        "Hydrogen-bond connectivity distinguishes surface stabilization from bubble disruption",
        "Geometric criterion uses O-O distance and donor O-H alignment; surface-water counts are normalized by available top-surface OH sites.",
    )
    path = figure_dir / "hbond_network_comparison.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    coupling = [
        ("surface_si_displacement_rms_A_mean", "Surface Si residual RMS (A)"),
        ("terminal_anchor_displacement_rms_A_mean", "Terminal residual RMS (A)"),
        ("water_under_bubble_S_mean_mean", "Under-bubble order S"),
        ("hbond_under_bubble_avg_degree_mean", "Under-bubble H-bond degree"),
    ]
    for ax, (field, ylabel) in zip(axes.flat, coupling):
        sns.scatterplot(
            data=summary_df,
            x="ch3_fraction",
            y=field,
            hue="case",
            palette=palette,
            s=70,
            edgecolor=TOKENS["ink"],
            linewidth=0.7,
            legend=False,
            ax=ax,
        )
        sns.lineplot(
            data=summary_df.sort_values("ch3_fraction"),
            x="ch3_fraction",
            y=field,
            color="#7A828F",
            linewidth=0.9,
            ax=ax,
        )
        for _, row in summary_df.iterrows():
            ax.annotate(str(row["case"]), (row["ch3_fraction"], row[field]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("CH3 fraction")
        ax.set_ylabel(ylabel)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, wspace=0.34, hspace=0.38)
    _chart_header(
        fig,
        axes[0, 0],
        "Interface coupling map links termination chemistry, structural motion, water order, and H-bond connectivity",
        "Four-case comparison is descriptive rather than a statistical regression; each point is a late-window block average.",
    )
    path = figure_dir / "interface_coupling_map.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.4), sharex=False)
    time_metrics = [
        ("surface_si_ligand_coord4_fraction", "Surface Si coord4 fraction"),
        ("surface_si_displacement_rms_A", "Surface Si residual RMS (A)"),
        ("hbond_under_bubble_avg_degree", "Under-bubble H-bond degree"),
    ]
    for ax, (field, ylabel) in zip(axes, time_metrics):
        for case in case_order:
            part = frame_df[frame_df["case"] == case].sort_values("time_ps")
            sns.lineplot(
                data=part,
                x="time_ps",
                y=field,
                color=palette[case],
                linewidth=1.0,
                label=case if ax is axes[0] else None,
                ax=ax,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (ps)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=4)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.08, hspace=0.38)
    _chart_header(
        fig,
        axes[0],
        "Late-window structural and hydration metrics remain traceable frame by frame",
        f"Last {config.last_frames_per_case} available frames per case sampled every {config.frame_stride} frame(s).",
    )
    path = figure_dir / "interface_structure_timeseries.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


def _write_report(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
    config: AnalysisConfig,
    figure_paths: Sequence[Path],
) -> Path:
    report_path = output_dir / "reports" / "interface_structure_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SiO2 Sphere Interface Structure Analysis",
        "",
        "## Scientific Question",
        "",
        "Does CH3 termination disrupt the silica backbone, or does it mainly increase terminal-group flexibility and surface roughness while the Si-(O+C) network remains intact?",
        "",
        "## Analysis Layers",
        "",
        "1. `surface_topology`: Si-O/Si-C ligand coordination, bond survival, angle distributions, connected-network metrics, and top-surface displacement.",
        "2. `termination_structure`: CH3/OH anchor tilt, terminal displacement, terminal roughness, and surface-OH orientation.",
        "3. `interfacial_water`: water dipole orientation relative to the outward surface normal, separated into under-bubble, contact-line, and outside-bubble regions.",
        "4. `hbond_network`: water-water connectivity and surface-OH/water hydrogen bonds.",
        "5. `interface_coupling`: cross-case comparison against CH3 surface fraction.",
        "",
        "## Method Guardrails",
        "",
        "- CH3 termination creates Si-C ligands. Therefore Si-O coordination alone is not a valid network-integrity metric; the primary metric is total Si-(O+C) ligand coordination.",
        "- The first available analyzed trajectory frame is used as the bond/topology reference. Edge survival is not a free-energy observable.",
        "- Water and surface-OH hydrogens are reassigned each frame to the nearest oxygen within the covalent O-H cutoff, so H2O/OH/H3O changes are tracked rather than discarded.",
        f"- Hydrogen bonds use O-O <= {config.hbond_oo_cutoff_A:g} A and donor O-H deviation <= {config.hbond_angle_deg:g} degrees.",
        "- Water-orientation distributions must be read together with the under-bubble water count; nearly dry cases provide too few molecules for a stable orientation distribution.",
        "- Four termination compositions establish trends but do not support high-confidence regression claims.",
        "",
        "## Late-Window Summary",
        "",
        "| case | CH3 frac. | surface Si coord4 | edge survival | surface Si displacement / A | terminal displacement / A | under-bubble S | under-bubble HB degree | surface HB/OH |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def value(field: str) -> float:
            try:
                return float(row.get(field, math.nan))
            except (TypeError, ValueError):
                return math.nan

        lines.append(
            "| {case} | {ch3:.3f} | {coord:.3f} | {survival:.3f} | {si_disp:.3f} | {term_disp:.3f} | {order:.3f} | {degree:.3f} | {surf_hb:.3f} |".format(
                case=row["case"],
                ch3=value("ch3_fraction"),
                coord=value("surface_si_ligand_coord4_fraction_mean"),
                survival=value("ligand_edge_survival_fraction_mean"),
                si_disp=value("surface_si_displacement_rms_A_mean"),
                term_disp=value("terminal_anchor_displacement_rms_A_mean"),
                order=value("water_under_bubble_S_mean_mean"),
                degree=value("hbond_under_bubble_avg_degree_mean"),
                surf_hb=value("surface_water_hbond_per_surfaceOH_mean"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Decision Tree",
            "",
            "- If Si-(O+C) coord4, ligand-edge survival, and bridging-O coord2 remain high while terminal displacement/tilt broadens, the apparent CH3 disorder is mainly terminal flexibility and surface corrugation.",
            "- If Si-(O+C) coord4 and ligand-edge survival fall together, with new-edge fraction and Si/O angle broadening increasing, the silica network is genuinely reconstructing.",
            "- If the network remains intact but under-bubble H-bond degree and water orientational order fall, the bubble primarily disrupts hydration rather than silica topology.",
            "- If OH-rich surfaces retain high surface-water H-bond counts and stronger water orientation while the bubble footprint is smaller, hydration stabilization is the likely wetting-control mechanism.",
            "",
            "## Trajectory Scope",
            "",
        ]
    )
    for row in segments:
        lines.append(
            f"- `{row['case']}` segment `{row['segment']}`: `{row['dump_path']}`; frames={row['frame_count']}"
        )
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Next Analyses",
            "",
            "1. Add block-bootstrap uncertainty for the reported late-window means.",
            "2. Couple these metrics to the existing wetting outputs (`foot_total`, `nfilm`, rough contact angle).",
            "3. Validate dynamic H2O/OH/H3O assignments against a focused proton-transfer trajectory inspection before using species fractions as a central claim.",
            "4. Add water residence times and site-resolved CH3/OH contact-line enrichment after the structural baseline is accepted.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_analysis(
    cases: Sequence[CaseSpec],
    output_dir: Path,
    config: AnalysisConfig = AnalysisConfig(),
    atom_type_map: Optional[AtomTypeMap] = None,
    dump_name: str = "trajectory.lammpstrj",
    plumed_name: str = "in.plumed",
    data_name: Optional[str] = None,
) -> Dict[str, Path]:
    config.validate()
    if atom_type_map is None:
        raise ValueError("atom_type_map is required; provide the LAMMPS type IDs for the system")
    atom_type_map.validate()
    output_dir = Path(output_dir)
    for name in (
        "surface_topology",
        "termination_structure",
        "interfacial_water",
        "hbond_network",
        "interface_coupling",
        "reports",
        "figures",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    frame_rows: List[Dict[str, object]] = []
    segment_rows: List[Dict[str, object]] = []
    samples: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    case_metadata: List[Dict[str, object]] = []

    ordered_cases = list(cases)
    for case in ordered_cases:
        segments = discover_segments(case.run_dir, dump_name, plumed_name)
        selections = parse_plumed_selections(segments[-1].plumed_path)
        atom_types = parse_lammps_atom_types(_choose_data_file(case.run_dir, data_name))
        substrate_stop = min(selections.n2_atoms)
        substrate_ids = [atom_id for atom_id in atom_types if atom_id < substrate_stop]
        si_ids = [atom_id for atom_id in substrate_ids if atom_types[atom_id] == atom_type_map.framework]
        o_ids = [atom_id for atom_id in substrate_ids if atom_types[atom_id] == atom_type_map.oxygen]
        c_ids = [atom_id for atom_id in substrate_ids if atom_types[atom_id] == atom_type_map.carbon]
        substrate_h_ids = [atom_id for atom_id in substrate_ids if atom_types[atom_id] == atom_type_map.hydrogen]
        termination_count = len(selections.surface_ch3) + len(selections.surface_oh)
        carbon_termination_fraction = (
            len(selections.surface_ch3) / termination_count if termination_count else math.nan
        )
        water_h_ids = [hydrogen_id for oxygen_id in selections.water_o for hydrogen_id in (oxygen_id + 1, oxygen_id + 2)]
        needed = (
            set(substrate_ids)
            | set(selections.n2_atoms)
            | set(selections.water_o)
            | set(water_h_ids)
        )
        total_frames = sum(segment.frame_count for segment in segments)
        start = max(0, total_frames - config.last_frames_per_case)
        analysis_indices = set(range(start, total_frames, config.frame_stride))
        selected_indices = set(analysis_indices) | {0}
        global_offset = 0
        for segment in segments:
            segment_rows.append(
                {
                    "case": case.label,
                    "segment": segment.label,
                    "segment_dir": str(segment.segment_dir),
                    "dump_path": str(segment.dump_path),
                    "frame_count": segment.frame_count,
                    "global_first_frame": global_offset,
                    "global_last_frame": global_offset + segment.frame_count - 1,
                    "analysis_start_global": start,
                    "analysis_stop_global_exclusive": total_frames,
                }
            )
            global_offset += segment.frame_count
        case_metadata.append(
            {
                "case": case.label,
                "run_dir": str(case.run_dir),
                "ch3_fraction": carbon_termination_fraction,
                "total_frames": total_frames,
                "analysis_frames_requested": len(analysis_indices),
                "substrate_si_count": len(si_ids),
                "substrate_o_count": len(o_ids),
                "substrate_c_count": len(c_ids),
                "top_surface_si_count": len(selections.surf_si_top),
                "top_surface_ch3_count": len(selections.surface_ch3),
                "top_surface_oh_count": len(selections.surface_oh),
                "water_count": len(selections.water_o),
                "n2_atom_count": len(selections.n2_atoms),
            }
        )
        reference: Optional[ReferenceState] = None
        seen_analysis_timesteps: Set[int] = set()
        for frame in iter_selected_frames(segments, selected_indices, needed):
            if reference is None:
                reference = _build_reference(
                    frame,
                    selections,
                    si_ids,
                    o_ids,
                    c_ids,
                    substrate_h_ids,
                    config,
                )
            if frame.global_index not in analysis_indices:
                continue
            if frame.timestep in seen_analysis_timesteps:
                continue
            seen_analysis_timesteps.add(frame.timestep)
            surf_coords, _ = _coords(frame.positions, selections.surf_si_top)
            z_ref = float(np.mean(surf_coords[:, 2]))
            topology_metrics, topology_samples = compute_topology_frame_metrics(
                frame,
                selections,
                si_ids,
                o_ids,
                c_ids,
                reference,
                config,
            )
            termination_metrics, termination_samples = compute_termination_frame_metrics(
                frame, selections, reference
            )
            water_metrics, orientation_samples = compute_water_hbond_metrics(
                frame,
                selections,
                reference,
                oxygen_candidate_ids=tuple(o_ids) + tuple(selections.water_o),
                hydrogen_candidate_ids=tuple(substrate_h_ids) + tuple(water_h_ids),
                z_ref=z_ref,
                config=config,
            )
            row: Dict[str, object] = {
                "case": case.label,
                "run_dir": str(case.run_dir),
                "ch3_fraction": carbon_termination_fraction,
                "global_frame": frame.global_index,
                "segment": frame.segment_label,
                "timestep": frame.timestep,
                "time_ps": frame.timestep * config.timestep_ps,
                "surface_reference_z_A": z_ref,
                **topology_metrics,
                **termination_metrics,
                **water_metrics,
            }
            frame_rows.append(row)
            for metric, values in topology_samples.items():
                samples[(case.label, metric)].extend(values)
            for metric, values in termination_samples.items():
                samples[(case.label, metric)].extend(values)
            for region, values in orientation_samples.items():
                samples[(case.label, f"water_cos_{region}")].extend(values)

    summaries = _aggregate_frame_rows(frame_rows)
    histogram_definitions = {
        "si_o_bond_length_A": (1.3, 2.4, 55),
        "si_c_bond_length_A": (1.5, 2.8, 52),
        "o_si_o_angle_deg": (50.0, 150.0, 50),
        "si_o_si_angle_deg": (70.0, 180.0, 55),
        "ch3_tilt_deg": (0.0, 180.0, 60),
        "oh_anchor_tilt_deg": (0.0, 180.0, 60),
        "surface_oh_bond_tilt_deg": (0.0, 180.0, 60),
        "water_cos_all": (-1.0, 1.0, 50),
        "water_cos_under_bubble": (-1.0, 1.0, 50),
        "water_cos_contact_line": (-1.0, 1.0, 50),
        "water_cos_outside_bubble": (-1.0, 1.0, 50),
    }
    distribution_rows = _histogram_rows(samples, histogram_definitions)

    metadata_fields = _ordered_fields(
        case_metadata,
        ["case", "run_dir", "ch3_fraction", "total_frames", "analysis_frames_requested"],
    )
    _write_csv(output_dir / "reports" / "case_metadata.csv", case_metadata, metadata_fields)
    _write_csv(
        output_dir / "reports" / "segments.csv",
        segment_rows,
        _ordered_fields(segment_rows, ["case", "segment", "segment_dir", "dump_path", "frame_count"]),
    )
    frame_fields = _ordered_fields(
        frame_rows,
        ["case", "run_dir", "ch3_fraction", "global_frame", "segment", "timestep", "time_ps"],
    )
    _write_csv(output_dir / "reports" / "interface_frame_metrics.csv", frame_rows, frame_fields)
    summary_fields = _ordered_fields(
        summaries,
        ["case", "run_dir", "ch3_fraction", "frames_used", "time_first_ps", "time_last_ps"],
    )
    _write_csv(
        output_dir / "reports" / "interface_structure_summary.csv",
        summaries,
        summary_fields,
    )
    distribution_fields = [
        "case",
        "metric",
        "bin_left",
        "bin_right",
        "bin_center",
        "count",
        "density",
    ]
    _write_csv(
        output_dir / "reports" / "interface_distributions.csv",
        distribution_rows,
        distribution_fields,
    )

    topology_fields = [
        field
        for field in frame_fields
        if field in {"case", "global_frame", "time_ps"}
        or field.startswith(("si_", "surface_si_", "bridging_", "ligand_", "network_", "o_si_o_", "si_o_si_"))
    ]
    _write_csv(output_dir / "surface_topology" / "frame_metrics.csv", frame_rows, topology_fields)
    termination_fields = [
        field
        for field in frame_fields
        if field in {"case", "global_frame", "time_ps"} or field.startswith(("ch3_", "oh_", "surface_oh_", "terminal_"))
    ]
    _write_csv(
        output_dir / "termination_structure" / "frame_metrics.csv",
        frame_rows,
        termination_fields,
    )
    water_fields = [
        field
        for field in frame_fields
        if field in {"case", "global_frame", "time_ps"}
        or field.startswith(("water_", "interface_water_"))
    ]
    _write_csv(
        output_dir / "interfacial_water" / "frame_metrics.csv",
        frame_rows,
        water_fields,
    )
    hbond_fields = [
        field
        for field in frame_fields
        if field in {"case", "global_frame", "time_ps"}
        or "hbond" in field.lower()
    ]
    _write_csv(
        output_dir / "hbond_network" / "frame_metrics.csv",
        frame_rows,
        hbond_fields,
    )
    _write_csv(
        output_dir / "interface_coupling" / "case_summary.csv",
        summaries,
        summary_fields,
    )

    figures: List[Path] = []
    if config.make_plots:
        figures = _make_plots(output_dir, frame_rows, summaries, distribution_rows, config)
    report = _write_report(output_dir, summaries, segment_rows, config, figures)
    metadata = {
        "cases": [{"label": case.label, "run_dir": str(case.run_dir)} for case in ordered_cases],
        "config": config.__dict__,
        "outputs": {
            "summary": str(output_dir / "reports" / "interface_structure_summary.csv"),
            "frame_metrics": str(output_dir / "reports" / "interface_frame_metrics.csv"),
            "distributions": str(output_dir / "reports" / "interface_distributions.csv"),
            "report": str(report),
            "figures": [str(path) for path in figures],
        },
    }
    metadata_path = output_dir / "reports" / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "summary": output_dir / "reports" / "interface_structure_summary.csv",
        "frame_metrics": output_dir / "reports" / "interface_frame_metrics.csv",
        "distributions": output_dir / "reports" / "interface_distributions.csv",
        "report": report,
        "metadata": metadata_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dump-name", default="trajectory.lammpstrj")
    parser.add_argument("--plumed-name", default="in.plumed")
    parser.add_argument("--data-name", help="Optional LAMMPS data filename relative to each case directory")
    parser.add_argument("--framework-atom-type", type=int, required=True)
    parser.add_argument("--oxygen-atom-type", type=int, required=True)
    parser.add_argument("--carbon-atom-type", type=int, required=True)
    parser.add_argument("--hydrogen-atom-type", type=int, required=True)
    parser.add_argument("--last-frames-per-case", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--timestep-ps", type=float, default=0.001)
    parser.add_argument("--interface-z-min-A", type=float, default=0.0)
    parser.add_argument("--interface-z-max-A", type=float, default=12.0)
    parser.add_argument("--bubble-radius-A", type=float, default=21.0)
    parser.add_argument("--outside-margin-A", type=float, default=5.0)
    parser.add_argument("--si-o-cutoff-A", type=float, default=2.20)
    parser.add_argument("--si-c-cutoff-A", type=float, default=2.35)
    parser.add_argument("--hbond-oo-cutoff-A", type=float, default=3.50)
    parser.add_argument("--hbond-angle-deg", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run_analysis(
            cases=args.case,
            output_dir=args.output_dir,
            atom_type_map=AtomTypeMap(
                framework=args.framework_atom_type,
                oxygen=args.oxygen_atom_type,
                carbon=args.carbon_atom_type,
                hydrogen=args.hydrogen_atom_type,
            ),
            config=AnalysisConfig(
                last_frames_per_case=args.last_frames_per_case,
                frame_stride=args.frame_stride,
                timestep_ps=args.timestep_ps,
                interface_z_min_A=args.interface_z_min_A,
                interface_z_max_A=args.interface_z_max_A,
                bubble_radius_A=args.bubble_radius_A,
                outside_margin_A=args.outside_margin_A,
                si_o_cutoff_A=args.si_o_cutoff_A,
                si_c_cutoff_A=args.si_c_cutoff_A,
                hbond_oo_cutoff_A=args.hbond_oo_cutoff_A,
                hbond_angle_deg=args.hbond_angle_deg,
                dpi=args.dpi,
                make_plots=not args.no_plots,
            ),
            dump_name=args.dump_name,
            plumed_name=args.plumed_name,
            data_name=args.data_name,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Sphere interface-structure analysis failed: {exc}")
        return 1
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
