"""Water order and instantaneous H-bond networks in TPCL-comoving coordinates.

The analysis uses water-oxygen geometry, explicit O-H assignment, and an
accepted time-dependent contact contour. Outputs are single-trajectory
structural diagnostics, not free energies, rates, or causal mechanisms.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    box_lengths,
    iter_lammps_dump_records,
    minimum_image_vectors,
)
from molsimflow.postprocess.interfacial_water_orientation import assign_hydrogen_neighbors

SCIENTIFIC_STATUS = (
    "SINGLE_TRAJECTORY_TPCL_COMOVING_WATER_STRUCTURE_AND_INSTANTANEOUS_HBOND_DIAGNOSTIC"
    "_NOT_FREE_ENERGY_RATE_OR_CAUSAL_MECHANISM"
)
SAMPLE_FIELDS = (
    "step",
    "time_ns",
    "oxygen_id",
    "arc_index",
    "theta_deg",
    "contour_distance_A",
    "normal_distance_A",
    "tangential_offset_A",
    "surface_distance_A",
    "q_tet",
    "lsi_A2",
    "lsi_neighbor_count",
    "lsi_neighbor_cap_reached",
    "oo_coordination",
    "oo_coordination_nonfour",
    "h_coordination",
    "h_coordination_defect",
    "hbond_donor_count",
    "hbond_acceptor_count",
    "hbond_degree",
    "hbond_internal_tpcl_degree",
)
FRAME_FIELDS = (
    "step",
    "time_ns",
    "tpcl_water_count",
    "qtet_valid_count",
    "mean_q_tet",
    "lsi_valid_count",
    "mean_lsi_A2",
    "mean_oo_coordination",
    "oo_coordination_nonfour_fraction",
    "h_coordination_defect_fraction",
    "hbond_incident_edge_count",
    "hbond_induced_edge_count",
    "mean_hbond_degree",
    "hbond_component_count",
    "hbond_largest_component_fraction",
    "lsi_neighbor_cap_reached_count",
)
ARC_FIELDS = (
    "step",
    "time_ns",
    "arc_index",
    "theta_deg",
    "water_count",
    "mean_normal_distance_A",
    "mean_surface_distance_A",
    "mean_q_tet",
    "mean_lsi_A2",
    "mean_oo_coordination",
    "oo_coordination_nonfour_fraction",
    "h_coordination_defect_fraction",
    "mean_hbond_degree",
    "mean_hbond_internal_tpcl_degree",
)


@dataclass(frozen=True)
class ReferenceFrame:
    center_xy: np.ndarray
    theta_rad: np.ndarray
    radii_A: np.ndarray


@dataclass(frozen=True)
class SelectedFrame:
    step: int
    bounds: np.ndarray
    surface: np.ndarray
    water_oxygen_ids: np.ndarray
    water_oxygen: np.ndarray
    candidate_oxygen_ids: np.ndarray
    candidate_oxygen: np.ndarray
    hydrogen: np.ndarray


def parse_range(raw: str) -> tuple[int, int]:
    parts = str(raw).replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be START:END")
    start, end = (int(value) for value in parts)
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("range must satisfy 1 <= START <= END")
    return start, end


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite value, got {value!r}")
    return number


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    missing = required.difference(fields)
    if missing or not rows:
        raise ValueError(f"{path}: missing rows or required columns {sorted(missing)}")
    return rows


def load_reference_frames(
    p1_path: Path,
    geometry_path: Path,
    arc_path: Path,
) -> dict[int, ReferenceFrame]:
    """Load a complete contour and corrected center for every admitted step."""

    p1_rows = _read_csv(
        p1_path,
        {"step", "bubble_center_x_A", "bubble_center_y_A"},
    )
    geometry_rows = _read_csv(
        geometry_path,
        {
            "step",
            "geometry_quality_pass",
            "contact_contour_centroid_local_x_A",
            "contact_contour_centroid_local_y_A",
        },
    )
    arc_rows = _read_csv(
        arc_path,
        {"step", "arc_index", "theta_deg", "local_radius_A"},
    )

    def by_step(rows: Sequence[Mapping[str, str]], source: Path) -> dict[int, Mapping[str, str]]:
        output: dict[int, Mapping[str, str]] = {}
        for row in rows:
            step = round(_finite(row["step"]))
            if step in output:
                raise ValueError(f"{source}: duplicate step {step}")
            output[step] = row
        return output

    p1 = by_step(p1_rows, p1_path)
    geometry = by_step(geometry_rows, geometry_path)
    if set(p1) != set(geometry):
        raise ValueError("P1 and geometry timestep supports differ")

    grouped: dict[int, list[Mapping[str, str]]] = {}
    seen: set[tuple[int, int]] = set()
    for row in arc_rows:
        step = round(_finite(row["step"]))
        arc = round(_finite(row["arc_index"]))
        key = (step, arc)
        if key in seen:
            raise ValueError(f"{arc_path}: duplicate (step, arc_index) {key}")
        seen.add(key)
        grouped.setdefault(step, []).append(row)
    if set(grouped) != set(p1):
        raise ValueError("arc and center-reference timestep supports differ")

    reference: dict[int, ReferenceFrame] = {}
    expected_arcs: list[int] | None = None
    previous_step: int | None = None
    for step in sorted(p1):
        if previous_step is not None and step <= previous_step:
            raise ValueError("reference steps are not strictly increasing")
        previous_step = step
        geo = geometry[step]
        if str(geo["geometry_quality_pass"]).lower() not in {"true", "1", "yes"}:
            raise ValueError(f"geometry is not quality-passing at step {step}")
        ordered = sorted(grouped[step], key=lambda row: round(_finite(row["arc_index"])))
        arcs = [round(_finite(row["arc_index"])) for row in ordered]
        if expected_arcs is None:
            expected_arcs = list(range(len(arcs)))
            if len(arcs) < 8:
                raise ValueError("at least eight contiguous contour arcs are required")
        if arcs != expected_arcs:
            raise ValueError(f"incomplete contour arcs at step {step}")
        theta = np.radians([_finite(row["theta_deg"]) for row in ordered])
        expected_theta = np.arange(len(arcs), dtype=float) * 2.0 * math.pi / len(arcs)
        if not np.allclose(np.mod(theta, 2.0 * math.pi), expected_theta, atol=1.0e-8):
            raise ValueError(f"nonuniform contour angles at step {step}")
        center = np.asarray(
            [
                _finite(p1[step]["bubble_center_x_A"])
                + _finite(geo["contact_contour_centroid_local_x_A"]),
                _finite(p1[step]["bubble_center_y_A"])
                + _finite(geo["contact_contour_centroid_local_y_A"]),
            ]
        )
        reference[step] = ReferenceFrame(
            center_xy=center,
            theta_rad=theta,
            radii_A=np.asarray([_finite(row["local_radius_A"]) for row in ordered]),
        )
    return reference


def _coordinate_columns(fields: Sequence[str]) -> tuple[tuple[int, bool], ...]:
    result = []
    for dimension in "xyz":
        for candidate in (dimension, dimension + "u", dimension + "s"):
            if candidate in fields:
                result.append((fields.index(candidate), candidate.endswith("s")))
                break
        else:
            raise ValueError(f"LAMMPS dump is missing {dimension}/{dimension}u/{dimension}s")
    return tuple(result)


def select_frame(
    frame: LammpsDumpFrame,
    surface_range: tuple[int, int],
    water_range: tuple[int, int],
    *,
    oxygen_type: int,
    hydrogen_type: int,
) -> SelectedFrame:
    """Select surface, target-water O, all candidate O, and all H atoms."""

    fields = frame.atom_fields
    if "id" not in fields or "type" not in fields:
        raise ValueError(f"step {frame.timestep}: missing id/type columns")
    id_index, type_index = fields.index("id"), fields.index("type")
    coordinates = _coordinate_columns(fields)
    lengths = box_lengths(frame.bounds)
    surface: list[tuple[int, np.ndarray]] = []
    water_oxygen: list[tuple[int, np.ndarray]] = []
    candidate_oxygen: list[tuple[int, np.ndarray]] = []
    hydrogen: list[tuple[int, np.ndarray]] = []
    surface_seen: set[int] = set()
    water_seen: set[int] = set()
    for row in frame.atom_rows:
        atom_id = int(row[id_index])
        atom_type = int(row[type_index])
        xyz = np.asarray([float(row[index]) for index, _ in coordinates], dtype=float)
        for dimension, (_, scaled) in enumerate(coordinates):
            if scaled:
                xyz[dimension] = frame.bounds[dimension, 0] + xyz[dimension] * lengths[dimension]
        if surface_range[0] <= atom_id <= surface_range[1]:
            surface.append((atom_id, xyz))
            surface_seen.add(atom_id)
        if water_range[0] <= atom_id <= water_range[1]:
            water_seen.add(atom_id)
            if atom_type == oxygen_type:
                water_oxygen.append((atom_id, xyz))
        if atom_type == oxygen_type:
            candidate_oxygen.append((atom_id, xyz))
        if atom_type == hydrogen_type:
            hydrogen.append((atom_id, xyz))
    if len(surface_seen) != surface_range[1] - surface_range[0] + 1:
        raise ValueError(f"step {frame.timestep}: incomplete surface range")
    if len(water_seen) != water_range[1] - water_range[0] + 1:
        raise ValueError(f"step {frame.timestep}: incomplete water range")
    if len(water_oxygen) < 5 or not candidate_oxygen or not hydrogen:
        raise ValueError(f"step {frame.timestep}: incomplete water chemistry selection")
    surface.sort(key=lambda item: item[0])
    water_oxygen.sort(key=lambda item: item[0])
    candidate_oxygen.sort(key=lambda item: item[0])
    hydrogen.sort(key=lambda item: item[0])
    return SelectedFrame(
        step=frame.timestep,
        bounds=frame.bounds,
        surface=np.asarray([item[1] for item in surface]),
        water_oxygen_ids=np.asarray([item[0] for item in water_oxygen], dtype=int),
        water_oxygen=np.asarray([item[1] for item in water_oxygen]),
        candidate_oxygen_ids=np.asarray([item[0] for item in candidate_oxygen], dtype=int),
        candidate_oxygen=np.asarray([item[1] for item in candidate_oxygen]),
        hydrogen=np.asarray([item[1] for item in hydrogen]),
    )


def tetrahedral_order(vectors: np.ndarray) -> float:
    """Return the standard four-nearest-neighbor tetrahedral order parameter."""

    if vectors.shape != (4, 3):
        return math.nan
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        return math.nan
    unit = vectors / norms[:, None]
    penalty = 0.0
    for left in range(3):
        for right in range(left + 1, 4):
            penalty += (float(unit[left] @ unit[right]) + 1.0 / 3.0) ** 2
    return 1.0 - 3.0 * penalty / 8.0


def local_structure_index(
    sorted_neighbor_distances_A: np.ndarray,
    cutoff_A: float,
) -> tuple[float, int]:
    """Return LSI from shell spacings through the first neighbor beyond cutoff."""

    distances = np.asarray(sorted_neighbor_distances_A, dtype=float)
    if len(distances) < 3 or np.any(np.diff(distances) < 0.0):
        return math.nan, 0
    inside = int(np.count_nonzero(distances <= cutoff_A))
    if inside < 2 or inside >= len(distances):
        return math.nan, inside
    gaps = np.diff(distances[: inside + 1])
    return float(np.mean((gaps - np.mean(gaps)) ** 2)), inside


def _donates(oh_vectors: np.ndarray, donor_acceptor: np.ndarray, angle_deg: float) -> bool:
    distance = float(np.linalg.norm(donor_acceptor))
    if distance <= 0.0 or len(oh_vectors) == 0:
        return False
    norms = np.linalg.norm(oh_vectors, axis=1)
    valid = norms > 0.0
    if not np.any(valid):
        return False
    cosine = (oh_vectors[valid] / norms[valid, None]) @ (donor_acceptor / distance)
    return bool(np.any(cosine >= math.cos(math.radians(angle_deg))))


def assigned_water_oh_vectors(
    frame: SelectedFrame,
    oh_cutoff_A: float,
) -> list[np.ndarray]:
    """Assign hydrogens and return O-H vectors for each declared water oxygen."""

    lengths = box_lengths(frame.bounds)
    assigned = assign_hydrogen_neighbors(
        frame.candidate_oxygen,
        frame.hydrogen,
        frame.bounds,
        oh_cutoff_A,
    )
    candidate_index = {
        int(atom_id): index for index, atom_id in enumerate(frame.candidate_oxygen_ids)
    }
    water_assignments = [
        assigned[candidate_index[int(atom_id)]] for atom_id in frame.water_oxygen_ids
    ]
    return [
        minimum_image_vectors(frame.hydrogen[items] - frame.water_oxygen[index], lengths)
        for index, items in enumerate(water_assignments)
    ]


def water_hbond_edges(
    oxygen: np.ndarray,
    oh_vectors: Sequence[np.ndarray],
    selected_indices: np.ndarray,
    bounds: np.ndarray,
    *,
    oo_cutoff_A: float,
    angle_cutoff_deg: float,
) -> list[tuple[int, int]]:
    """Return directed water-water H bonds incident to selected oxygens."""

    from scipy.spatial import cKDTree

    if not len(selected_indices):
        return []
    lengths = box_lengths(bounds)
    tree = cKDTree((oxygen - bounds[:, 0]) % lengths, boxsize=lengths)
    pairs: set[tuple[int, int]] = set()
    for oxygen_index in selected_indices:
        neighbors = tree.query_ball_point(
            (oxygen[int(oxygen_index)] - bounds[:, 0]) % lengths,
            oo_cutoff_A,
        )
        for neighbor in neighbors:
            if int(neighbor) != int(oxygen_index):
                pairs.add(tuple(sorted((int(oxygen_index), int(neighbor)))))

    edges: list[tuple[int, int]] = []
    for left, right in sorted(pairs):
        vector = minimum_image_vectors(oxygen[right] - oxygen[left], lengths)
        if _donates(oh_vectors[left], vector, angle_cutoff_deg):
            edges.append((left, right))
        if _donates(oh_vectors[right], -vector, angle_cutoff_deg):
            edges.append((right, left))
    return edges


def hbond_network_metrics(
    oxygen: np.ndarray,
    oh_vectors: Sequence[np.ndarray],
    selected_indices: np.ndarray,
    bounds: np.ndarray,
    *,
    oo_cutoff_A: float,
    angle_cutoff_deg: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Measure edges incident to selected water and its induced TPCL network."""

    count = len(selected_indices)
    donors = np.zeros(count, dtype=int)
    acceptors = np.zeros(count, dtype=int)
    degree = np.zeros(count, dtype=int)
    internal_degree = np.zeros(count, dtype=int)
    if count == 0:
        return (
            {
                "donor": donors,
                "acceptor": acceptors,
                "degree": degree,
                "internal_degree": internal_degree,
            },
            {
                "incident_edges": 0,
                "induced_edges": 0,
                "component_count": 0,
                "largest_component_fraction": math.nan,
            },
        )
    selected_lookup = {int(index): local for local, index in enumerate(selected_indices)}
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    directed_edges = water_hbond_edges(
        oxygen,
        oh_vectors,
        selected_indices,
        bounds,
        oo_cutoff_A=oo_cutoff_A,
        angle_cutoff_deg=angle_cutoff_deg,
    )
    incident_edges = {tuple(sorted(edge)) for edge in directed_edges}
    induced_edges = {
        edge
        for edge in incident_edges
        if edge[0] in selected_lookup and edge[1] in selected_lookup
    }
    for donor, acceptor in directed_edges:
        if donor in selected_lookup:
            donors[selected_lookup[donor]] += 1
        if acceptor in selected_lookup:
            acceptors[selected_lookup[acceptor]] += 1
    for left, right in incident_edges:
        if left in selected_lookup:
            degree[selected_lookup[left]] += 1
        if right in selected_lookup:
            degree[selected_lookup[right]] += 1
    for left, right in induced_edges:
        left_local = selected_lookup[left]
        right_local = selected_lookup[right]
        internal_degree[left_local] += 1
        internal_degree[right_local] += 1
        union(left_local, right_local)
    component_sizes: dict[int, int] = {}
    for index in range(count):
        root = find(index)
        component_sizes[root] = component_sizes.get(root, 0) + 1
    return (
        {
            "donor": donors,
            "acceptor": acceptors,
            "degree": degree,
            "internal_degree": internal_degree,
        },
        {
            "incident_edges": len(incident_edges),
            "induced_edges": len(induced_edges),
            "component_count": len(component_sizes),
            "largest_component_fraction": max(component_sizes.values()) / count,
        },
    )


def _mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _column_values(rows: Sequence[Mapping[str, object]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=float)


def analyze_selected_frame(
    frame: SelectedFrame,
    reference: ReferenceFrame,
    *,
    timestep_fs: float,
    tpcl_half_width_A: float,
    surface_layer_max_A: float,
    oh_cutoff_A: float,
    oo_cutoff_A: float,
    hbond_angle_deg: float,
    lsi_cutoff_A: float,
    lsi_neighbor_cap: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Analyze one trajectory frame against one accepted moving contour."""

    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    contour = reference.center_xy + reference.radii_A[:, None] * np.column_stack(
        (np.cos(reference.theta_rad), np.sin(reference.theta_rad))
    )
    center_displacement = minimum_image_vectors(
        frame.water_oxygen[:, :2] - reference.center_xy,
        lengths[:2],
    )
    polar_angle = np.mod(
        np.arctan2(center_displacement[:, 1], center_displacement[:, 0]),
        2.0 * math.pi,
    )
    delta_theta = 2.0 * math.pi / len(reference.theta_rad)
    nearest_arc = np.mod(
        np.floor(polar_angle / delta_theta + 0.5).astype(int),
        len(reference.theta_rad),
    )
    radial_distance = np.linalg.norm(center_displacement, axis=1)
    normal_distance = radial_distance - reference.radii_A[nearest_arc]
    contour_displacement = minimum_image_vectors(
        frame.water_oxygen[:, :2] - contour[nearest_arc],
        lengths[:2],
    )
    contour_distance = np.linalg.norm(contour_displacement, axis=1)
    surface_tree = cKDTree(
        (frame.surface - frame.bounds[:, 0]) % lengths,
        boxsize=lengths,
    )
    surface_distance = surface_tree.query(
        (frame.water_oxygen - frame.bounds[:, 0]) % lengths
    )[0]
    selected = np.flatnonzero(
        (np.abs(normal_distance) <= tpcl_half_width_A)
        & (surface_distance <= surface_layer_max_A)
    )
    if not len(selected):
        raise ValueError(f"step {frame.step}: no water samples in the TPCL window")
    water_tree = cKDTree(
        (frame.water_oxygen - frame.bounds[:, 0]) % lengths,
        boxsize=lengths,
    )
    k = min(lsi_neighbor_cap + 2, len(frame.water_oxygen))
    distances, neighbors = water_tree.query(
        (frame.water_oxygen[selected] - frame.bounds[:, 0]) % lengths,
        k=k,
    )
    if len(selected) == 1:
        distances = distances.reshape((1, -1))
        neighbors = neighbors.reshape((1, -1))

    oh_vectors = assigned_water_oh_vectors(frame, oh_cutoff_A)
    network_rows, network_summary = hbond_network_metrics(
        frame.water_oxygen,
        oh_vectors,
        selected,
        frame.bounds,
        oo_cutoff_A=oo_cutoff_A,
        angle_cutoff_deg=hbond_angle_deg,
    )

    sample_rows: list[dict[str, object]] = []
    time_ns = frame.step * timestep_fs / 1.0e6
    for local_index, oxygen_index in enumerate(selected):
        ordered = [
            (float(distance), int(neighbor))
            for distance, neighbor in zip(distances[local_index], neighbors[local_index])
            if int(neighbor) != int(oxygen_index) and math.isfinite(float(distance))
        ]
        ordered.sort()
        ordered_distances = np.asarray([item[0] for item in ordered])
        q_tet = math.nan
        if len(ordered) >= 4:
            nearest = np.asarray([item[1] for item in ordered[:4]], dtype=int)
            vectors = minimum_image_vectors(
                frame.water_oxygen[nearest] - frame.water_oxygen[int(oxygen_index)],
                lengths,
            )
            q_tet = tetrahedral_order(vectors)
        lsi, lsi_neighbors = local_structure_index(ordered_distances, lsi_cutoff_A)
        cap_reached = bool(len(ordered_distances) and ordered_distances[-1] <= lsi_cutoff_A)
        oo_coordination = int(np.count_nonzero(ordered_distances <= oo_cutoff_A))
        arc = int(nearest_arc[int(oxygen_index)])
        displacement = minimum_image_vectors(
            frame.water_oxygen[int(oxygen_index), :2] - contour[arc], lengths[:2]
        )
        normal = np.asarray(
            [math.cos(reference.theta_rad[arc]), math.sin(reference.theta_rad[arc])]
        )
        tangent = np.asarray([-normal[1], normal[0]])
        h_coordination = len(oh_vectors[int(oxygen_index)])
        sample_rows.append(
            {
                "step": frame.step,
                "time_ns": time_ns,
                "oxygen_id": int(frame.water_oxygen_ids[int(oxygen_index)]),
                "arc_index": arc,
                "theta_deg": math.degrees(float(reference.theta_rad[arc])),
                "contour_distance_A": float(contour_distance[int(oxygen_index)]),
                "normal_distance_A": float(normal_distance[int(oxygen_index)]),
                "tangential_offset_A": float(displacement @ tangent),
                "surface_distance_A": float(surface_distance[int(oxygen_index)]),
                "q_tet": q_tet,
                "lsi_A2": lsi,
                "lsi_neighbor_count": lsi_neighbors,
                "lsi_neighbor_cap_reached": cap_reached,
                "oo_coordination": oo_coordination,
                "oo_coordination_nonfour": oo_coordination != 4,
                "h_coordination": h_coordination,
                "h_coordination_defect": h_coordination != 2,
                "hbond_donor_count": int(network_rows["donor"][local_index]),
                "hbond_acceptor_count": int(network_rows["acceptor"][local_index]),
                "hbond_degree": int(network_rows["degree"][local_index]),
                "hbond_internal_tpcl_degree": int(
                    network_rows["internal_degree"][local_index]
                ),
            }
        )

    q_tet_values = np.asarray([row["q_tet"] for row in sample_rows], dtype=float)
    lsi_values = np.asarray([row["lsi_A2"] for row in sample_rows], dtype=float)
    oo_values = np.asarray([row["oo_coordination"] for row in sample_rows], dtype=float)
    oo_nonfour = np.asarray([row["oo_coordination_nonfour"] for row in sample_rows], dtype=float)
    h_defects = np.asarray([row["h_coordination_defect"] for row in sample_rows], dtype=float)
    hbond_degree = np.asarray([row["hbond_degree"] for row in sample_rows], dtype=float)
    frame_row: dict[str, object] = {
        "step": frame.step,
        "time_ns": time_ns,
        "tpcl_water_count": len(sample_rows),
        "qtet_valid_count": int(np.count_nonzero(np.isfinite(q_tet_values))),
        "mean_q_tet": _mean(q_tet_values),
        "lsi_valid_count": int(np.count_nonzero(np.isfinite(lsi_values))),
        "mean_lsi_A2": _mean(lsi_values),
        "mean_oo_coordination": _mean(oo_values),
        "oo_coordination_nonfour_fraction": _mean(oo_nonfour),
        "h_coordination_defect_fraction": _mean(h_defects),
        "hbond_incident_edge_count": network_summary["incident_edges"],
        "hbond_induced_edge_count": network_summary["induced_edges"],
        "mean_hbond_degree": _mean(hbond_degree),
        "hbond_component_count": network_summary["component_count"],
        "hbond_largest_component_fraction": network_summary["largest_component_fraction"],
        "lsi_neighbor_cap_reached_count": sum(
            bool(row["lsi_neighbor_cap_reached"]) for row in sample_rows
        ),
    }
    arc_rows: list[dict[str, object]] = []
    for arc in range(len(reference.theta_rad)):
        rows = [row for row in sample_rows if int(row["arc_index"]) == arc]
        arc_rows.append(
            {
                "step": frame.step,
                "time_ns": time_ns,
                "arc_index": arc,
                "theta_deg": math.degrees(float(reference.theta_rad[arc])),
                "water_count": len(rows),
                "mean_normal_distance_A": _mean(_column_values(rows, "normal_distance_A")),
                "mean_surface_distance_A": _mean(
                    _column_values(rows, "surface_distance_A")
                ),
                "mean_q_tet": _mean(_column_values(rows, "q_tet")),
                "mean_lsi_A2": _mean(_column_values(rows, "lsi_A2")),
                "mean_oo_coordination": _mean(_column_values(rows, "oo_coordination")),
                "oo_coordination_nonfour_fraction": _mean(
                    _column_values(rows, "oo_coordination_nonfour")
                ),
                "h_coordination_defect_fraction": _mean(
                    _column_values(rows, "h_coordination_defect")
                ),
                "mean_hbond_degree": _mean(_column_values(rows, "hbond_degree")),
                "mean_hbond_internal_tpcl_degree": _mean(
                    _column_values(rows, "hbond_internal_tpcl_degree")
                ),
            }
        )
    return sample_rows, frame_row, arc_rows


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    if len(args.trajectory) != len(args.trajectory_end_step):
        raise ValueError("each trajectory requires one declared end step")
    reference = load_reference_frames(args.p1_timeseries, args.geometry, args.arc_kinematics)
    steps = sorted(reference)
    selected_steps = set(steps[:: args.frame_stride])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame_rows: list[dict[str, object]] = []
    processed: set[int] = set()
    stop = False
    with gzip.open(output / "water_order_samples.csv.gz", "wt", newline="") as sample_handle, (
        output / "water_order_by_frame.csv"
    ).open("w", newline="") as frame_handle, gzip.open(
        output / "water_order_by_arc.csv.gz", "wt", newline=""
    ) as arc_handle:
        sample_writer = csv.DictWriter(sample_handle, fieldnames=list(SAMPLE_FIELDS))
        frame_writer = csv.DictWriter(frame_handle, fieldnames=list(FRAME_FIELDS))
        arc_writer = csv.DictWriter(arc_handle, fieldnames=list(ARC_FIELDS))
        sample_writer.writeheader()
        frame_writer.writeheader()
        arc_writer.writeheader()
        for segment_index, (trajectory, end_step) in enumerate(
            zip(args.trajectory, args.trajectory_end_step)
        ):
            reached_end = False
            for raw_frame in iter_lammps_dump_records(trajectory):
                step = raw_frame.timestep
                if step > end_step:
                    raise ValueError(f"{trajectory}: passed declared end step {end_step}")
                if step == end_step:
                    reached_end = True
                replace_with_later_segment = (
                    segment_index < len(args.trajectory) - 1 and step == end_step
                )
                if (
                    step in selected_steps
                    and not replace_with_later_segment
                    and step not in processed
                ):
                    selected_frame = select_frame(
                        raw_frame,
                        args.surface_range,
                        args.water_range,
                        oxygen_type=args.oxygen_type,
                        hydrogen_type=args.hydrogen_type,
                    )
                    samples, frame_row, arc_rows = analyze_selected_frame(
                        selected_frame,
                        reference[step],
                        timestep_fs=args.timestep_fs,
                        tpcl_half_width_A=args.tpcl_half_width_A,
                        surface_layer_max_A=args.surface_layer_max_A,
                        oh_cutoff_A=args.oh_cutoff_A,
                        oo_cutoff_A=args.oo_cutoff_A,
                        hbond_angle_deg=args.hbond_angle_deg,
                        lsi_cutoff_A=args.lsi_cutoff_A,
                        lsi_neighbor_cap=args.lsi_neighbor_cap,
                    )
                    if not samples:
                        raise ValueError(f"step {step}: no water samples in the TPCL window")
                    sample_writer.writerows(samples)
                    frame_writer.writerow(frame_row)
                    arc_writer.writerows(arc_rows)
                    frame_rows.append(frame_row)
                    processed.add(step)
                    if args.max_frames is not None and len(processed) >= args.max_frames:
                        stop = True
                        break
                if step == end_step:
                    break
            if stop:
                break
            if not reached_end:
                raise ValueError(f"{trajectory}: declared end step {end_step} was not found")
    if not frame_rows:
        raise ValueError("no reference-aligned trajectory frames were analyzed")
    if args.max_frames is None and processed != selected_steps:
        missing = len(selected_steps.difference(processed))
        extra = len(processed.difference(selected_steps))
        raise ValueError(f"trajectory/reference support mismatch: missing={missing}, extra={extra}")

    def frame_mean(name: str) -> float | None:
        values = np.asarray([row[name] for row in frame_rows], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if len(finite) else None

    summary: dict[str, object] = {
        "status": "PASS",
        "case_id": args.case_id,
        "analyzed_frames": len(frame_rows),
        "first_step": int(frame_rows[0]["step"]),
        "last_step": int(frame_rows[-1]["step"]),
        "frame_stride": args.frame_stride,
        "total_tpcl_water_frame_samples": sum(
            int(row["tpcl_water_count"]) for row in frame_rows
        ),
        "mean_tpcl_water_count": frame_mean("tpcl_water_count"),
        "mean_q_tet": frame_mean("mean_q_tet"),
        "mean_lsi_A2": frame_mean("mean_lsi_A2"),
        "mean_oo_coordination": frame_mean("mean_oo_coordination"),
        "mean_oo_coordination_nonfour_fraction": frame_mean(
            "oo_coordination_nonfour_fraction"
        ),
        "mean_h_coordination_defect_fraction": frame_mean(
            "h_coordination_defect_fraction"
        ),
        "mean_hbond_degree": frame_mean("mean_hbond_degree"),
        "mean_hbond_largest_component_fraction": frame_mean(
            "hbond_largest_component_fraction"
        ),
        "lsi_neighbor_cap_reached_total": sum(
            int(row["lsi_neighbor_cap_reached_count"]) for row in frame_rows
        ),
        "scientific_status": SCIENTIFIC_STATUS,
    }
    manifest = {
        "case_id": args.case_id,
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "trajectory_end_steps": args.trajectory_end_step,
        "p1_timeseries": str(args.p1_timeseries.resolve()),
        "geometry": str(args.geometry.resolve()),
        "arc_kinematics": str(args.arc_kinematics.resolve()),
        "surface_atom_range": list(args.surface_range),
        "water_atom_range": list(args.water_range),
        "oxygen_type": args.oxygen_type,
        "hydrogen_type": args.hydrogen_type,
        "tpcl_half_width_A": args.tpcl_half_width_A,
        "tpcl_selection_definition": (
            "absolute_radial_normal_distance_to_nearest_uniform_polar_arc"
        ),
        "surface_layer_definition": "minimum_distance_to_any_declared_surface_atom",
        "surface_layer_max_A": args.surface_layer_max_A,
        "qtet_definition": "four_nearest_water_oxygen_neighbors",
        "lsi_cutoff_A": args.lsi_cutoff_A,
        "lsi_neighbor_cap": args.lsi_neighbor_cap,
        "lsi_definition": "variance_of_sorted_OO_gaps_through_first_neighbor_beyond_cutoff",
        "oo_coordination_cutoff_A": args.oo_cutoff_A,
        "oo_coordination_nonfour_definition": (
            "geometric_neighbor_count_not_equal_to_four_not_a_chemical_defect_label"
        ),
        "oh_assignment_cutoff_A": args.oh_cutoff_A,
        "h_coordination_defect_definition": "assigned_H_count_not_equal_to_two",
        "hbond_oo_cutoff_A": args.oo_cutoff_A,
        "hbond_angle_cutoff_deg": args.hbond_angle_deg,
        "hbond_network_definition": "water_water_edges_incident_to_tpcl_selected_oxygen",
        "restart_policy": "later_segment_replaces_duplicate_boundary_step",
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "\n".join(
            (
                "# TPCL-comoving local water order",
                "",
                f"- Case: {args.case_id}",
                f"- Analyzed frames: {len(frame_rows)}",
                f"- Water-frame samples: {summary['total_tpcl_water_frame_samples']}",
                f"- Mean q_tet: {summary['mean_q_tet']}",
                f"- Mean LSI (A^2): {summary['mean_lsi_A2']}",
                f"- Mean instantaneous H-bond degree: {summary['mean_hbond_degree']}",
                "",
                (
                    "These are single-trajectory structural and instantaneous network diagnostics. "
                    "They do not establish free energies, physical rates, event ordering, or causality."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-end-step", type=int, action="append", required=True)
    parser.add_argument("--p1-timeseries", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--arc-kinematics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--tpcl-half-width-A", type=float, default=4.0)
    parser.add_argument("--surface-layer-max-A", type=float, default=6.0)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.25)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    parser.add_argument("--hbond-angle-deg", type=float, default=30.0)
    parser.add_argument("--lsi-cutoff-A", type=float, default=3.7)
    parser.add_argument("--lsi-neighbor-cap", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = (
        args.timestep_fs,
        args.tpcl_half_width_A,
        args.surface_layer_max_A,
        args.oh_cutoff_A,
        args.oo_cutoff_A,
        args.hbond_angle_deg,
        args.lsi_cutoff_A,
    )
    if min(positive) <= 0.0 or args.frame_stride < 1 or args.lsi_neighbor_cap < 5:
        raise ValueError("cutoffs must be positive, stride >= 1, and LSI cap >= 5")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
