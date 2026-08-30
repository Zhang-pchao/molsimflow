"""Resolve local TPCL dwell--jump candidates from restart-aware trajectories.

The workflow separates whole-object translation, mean-radius change, low-order
shape modes, and local contour residuals.  Reported events are operational
kinetic candidates, not free-energy barriers or causal proof.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors, periodic_center
from molsimflow.postprocess.interfacial_water_hbond import donor_points_to
from molsimflow.postprocess.interfacial_water_orientation import assign_hydrogen_neighbors
from molsimflow.postprocess.nanobubble_attachment import largest_cluster, molecule_centers
from molsimflow.postprocess.surface_reference import load_surface_reference
from molsimflow.postprocess.surface_site_enrichment import identify_surface_sites

Range = tuple[int, int]


def _range(value: object, name: str) -> Range:
    if isinstance(value, str):
        parts = value.replace("-", ":").split(":")
        if len(parts) != 2:
            raise ValueError(f"{name} must be START:END")
        start, end = map(int, parts)
    else:
        start, end = map(int, value)  # type: ignore[arg-type]
    if start < 1 or end < start:
        raise ValueError(f"{name} must satisfy 1 <= START <= END")
    return start, end


@dataclass(frozen=True)
class TpclConfig:
    kind: str
    trajectories: tuple[Path, ...]
    initial_xyz: Path
    surface_range: Range
    phase_range: Range
    water_range: Range
    surface_z_A: float
    timestep_fs: float = 0.5
    oxygen_type: int = 2
    hydrogen_type: int = 1
    cluster_cutoff_A: float = 5.5
    contact_cutoff_A: float = 4.0
    arc_bins: int = 36
    minimum_contact_points: int = 8
    surface_depth_A: float = 3.0
    site_bond_cutoff_A: float = 1.25
    local_site_radius_A: float = 6.0
    chemical_boundary_A: float = 2.0
    tpcl_half_width_A: float = 4.0
    hydration_z_min_A: float = 0.0
    hydration_z_max_A: float = 6.0
    oh_cutoff_A: float = 1.25
    oo_cutoff_A: float = 3.5
    hbond_angle_deg: float = 30.0
    angle_fit_z_max_A: float = 12.0
    angle_fit_bin_A: float = 2.0
    minimum_dwell_frames: int = 4
    post_confirm_frames: int = 2
    noise_floor_A: float = 0.5
    minimum_dwell_tolerance_A: float = 1.0
    minimum_jump_A: float = 2.0
    jump_noise_multiplier: float = 4.0
    residual_noise_multiplier: float = 2.0
    maximum_translation_fraction: float = 0.7
    minimum_site_stability_fraction: float = 0.75

    @classmethod
    def load(cls, path: Path) -> TpclConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown TPCL config keys: {unknown}")
        raw["trajectories"] = tuple(Path(item) for item in raw["trajectories"])
        raw["initial_xyz"] = Path(raw["initial_xyz"])
        for name in ("surface_range", "phase_range", "water_range"):
            raw[name] = _range(raw[name], name)
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.kind not in {"nanobubble", "nanodroplet"}:
            raise ValueError("kind must be nanobubble or nanodroplet")
        if not self.trajectories:
            raise ValueError("At least one trajectory is required")
        if self.arc_bins < 8 or self.minimum_contact_points < 3:
            raise ValueError("arc_bins must be >= 8 and minimum_contact_points >= 3")
        positive = (
            self.timestep_fs,
            self.cluster_cutoff_A,
            self.contact_cutoff_A,
            self.surface_depth_A,
            self.site_bond_cutoff_A,
            self.local_site_radius_A,
            self.chemical_boundary_A,
            self.tpcl_half_width_A,
            self.oh_cutoff_A,
            self.oo_cutoff_A,
            self.hbond_angle_deg,
            self.angle_fit_z_max_A,
            self.angle_fit_bin_A,
            self.noise_floor_A,
            self.minimum_dwell_tolerance_A,
            self.minimum_jump_A,
            self.jump_noise_multiplier,
            self.residual_noise_multiplier,
        )
        if min(positive) <= 0:
            raise ValueError("All distance, time, and threshold parameters must be positive")
        if self.hydration_z_max_A <= self.hydration_z_min_A:
            raise ValueError("Hydration z range is invalid")
        if self.minimum_dwell_frames < 3 or self.post_confirm_frames < 2:
            raise ValueError("Dwell and post-state confirmation are under-resolved")
        if not 0 <= self.maximum_translation_fraction < 1:
            raise ValueError("maximum_translation_fraction must lie in [0, 1)")
        if not 0 < self.minimum_site_stability_fraction <= 1:
            raise ValueError("minimum_site_stability_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class TpclFrame:
    source: Path
    source_frame: int
    segment_index: int
    step: int
    bounds: np.ndarray
    surface_ids: np.ndarray
    surface_types: np.ndarray
    surface: np.ndarray
    phase_ids: np.ndarray
    phase_types: np.ndarray
    phase: np.ndarray
    water_ids: np.ndarray
    water_types: np.ndarray
    water: np.ndarray


def _coord_indices(fields: Sequence[str]) -> tuple[tuple[int, bool], ...]:
    result = []
    for dim in "xyz":
        for candidate in (dim, dim + "u", dim + "s"):
            if candidate in fields:
                result.append((fields.index(candidate), candidate.endswith("s")))
                break
        else:
            raise ValueError(f"LAMMPS dump is missing {dim}/{dim}u/{dim}s")
    return tuple(result)


def _selected_array(rows: list[tuple[int, int, float, float, float]], expected: Range, step: int):
    rows.sort(key=lambda item: item[0])
    count = expected[1] - expected[0] + 1
    if len(rows) != count or (rows and (rows[0][0], rows[-1][0]) != expected):
        raise ValueError(f"Incomplete atom selection {expected} at step {step}")
    values = np.asarray(rows, dtype=float)
    return values[:, 0].astype(int), values[:, 1].astype(int), values[:, 2:]


def iter_tpcl_frames(path: Path, config: TpclConfig, segment_index: int) -> Iterator[TpclFrame]:
    """Stream sorted surface, phase, and water selections from an orthorhombic dump."""

    with Path(path).open(encoding="utf-8") as handle:
        frame_index = 0
        while True:
            line = handle.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Expected TIMESTEP in {path}, got {line!r}")
            step = int(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Missing atom count at step {step}")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing orthorhombic box at step {step}")
            bounds = np.asarray(
                [list(map(float, handle.readline().split()[:2])) for _ in range(3)]
            )
            fields = handle.readline().split()[2:]
            if "id" not in fields or "type" not in fields:
                raise ValueError(f"LAMMPS dump lacks id/type at step {step}")
            id_index, type_index = fields.index("id"), fields.index("type")
            coord_indices = _coord_indices(fields)
            lengths = box_lengths(bounds)
            surface: list[tuple[int, int, float, float, float]] = []
            phase: list[tuple[int, int, float, float, float]] = []
            water: list[tuple[int, int, float, float, float]] = []
            for _ in range(atom_count):
                values = handle.readline().split()
                atom_id, atom_type = int(values[id_index]), int(values[type_index])
                xyz = np.asarray([float(values[index]) for index, _ in coord_indices])
                for dim, (_, scaled) in enumerate(coord_indices):
                    if scaled:
                        xyz[dim] = bounds[dim, 0] + xyz[dim] * lengths[dim]
                row = (atom_id, atom_type, float(xyz[0]), float(xyz[1]), float(xyz[2]))
                if config.surface_range[0] <= atom_id <= config.surface_range[1]:
                    surface.append(row)
                if config.phase_range[0] <= atom_id <= config.phase_range[1]:
                    phase.append(row)
                if config.water_range[0] <= atom_id <= config.water_range[1]:
                    water.append(row)
            surface_ids, surface_types, surface_xyz = _selected_array(
                surface, config.surface_range, step
            )
            phase_ids, phase_types, phase_xyz = _selected_array(phase, config.phase_range, step)
            water_ids, water_types, water_xyz = _selected_array(water, config.water_range, step)
            yield TpclFrame(
                Path(path),
                frame_index,
                segment_index,
                step,
                bounds,
                surface_ids,
                surface_types,
                surface_xyz,
                phase_ids,
                phase_types,
                phase_xyz,
                water_ids,
                water_types,
                water_xyz,
            )
            frame_index += 1


def ray_polygon_radii(boundary: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Intersect rays from the origin with a closed convex polygon."""

    following = np.roll(boundary, -1, axis=0)
    segment = following - boundary
    radii = np.full(len(theta), np.nan)
    for index, angle in enumerate(theta):
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        denominator = direction[0] * segment[:, 1] - direction[1] * segment[:, 0]
        usable = np.abs(denominator) > 1.0e-12
        cross_point_segment = boundary[:, 0] * segment[:, 1] - boundary[:, 1] * segment[:, 0]
        cross_point_direction = boundary[:, 0] * direction[1] - boundary[:, 1] * direction[0]
        ray_distance = np.divide(
            cross_point_segment,
            denominator,
            out=np.full(len(boundary), np.nan),
            where=usable,
        )
        segment_fraction = np.divide(
            cross_point_direction,
            denominator,
            out=np.full(len(boundary), np.nan),
            where=usable,
        )
        valid = usable & (ray_distance >= -1.0e-10) & (segment_fraction >= -1.0e-10) & (
            segment_fraction <= 1.0 + 1.0e-10
        )
        if np.any(valid):
            radii[index] = float(np.nanmax(ray_distance[valid]))
    return radii


def contour_differential(points: np.ndarray, delta_theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return outward normals, signed curvature magnitude, and local arc lengths."""

    first = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / (2 * delta_theta)
    second = (
        np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
    ) / delta_theta**2
    speed = np.linalg.norm(first, axis=1)
    tangent = np.divide(first, speed[:, None], out=np.zeros_like(first), where=speed[:, None] > 0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    flip = np.sum(normal * points, axis=1) < 0
    normal[flip] *= -1
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    curvature = np.divide(
        np.abs(cross), speed**3, out=np.full(len(points), np.nan), where=speed > 0
    )
    arc_length = 0.5 * (
        np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        + np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    )
    return normal, curvature, arc_length


def _point_inside_convex_hull(hull) -> bool:
    return bool(np.all(hull.equations[:, :-1] @ np.zeros(2) + hull.equations[:, -1] <= 1e-9))


def phase_contour(frame: TpclFrame, config: TpclConfig) -> tuple[dict, dict | None]:
    """Build one PBC-aware particle-level phase contour."""

    from scipy.spatial import ConvexHull, QhullError, cKDTree

    lengths = box_lengths(frame.bounds)
    if config.kind == "nanobubble":
        centers = molecule_centers(frame.phase, frame.bounds)
        members = largest_cluster(centers, frame.bounds, config.cluster_cutoff_A)
        cluster = centers[members]
        center = periodic_center(cluster, frame.bounds)
        shifted_surface = (frame.surface - frame.bounds[:, 0]) % lengths
        shifted_phase = (frame.phase - frame.bounds[:, 0]) % lengths
        atom_distances = cKDTree(shifted_surface, boxsize=lengths).query(shifted_phase, k=1)[0]
        molecule_distances = atom_distances.reshape(-1, 2).min(axis=1)
        contact_cluster_indices = members[molecule_distances[members] <= config.contact_cutoff_A]
        contact_points = centers[contact_cluster_indices]
        n2_contact_count = len(contact_points)
    else:
        oxygen = frame.phase[frame.phase_types == config.oxygen_type]
        members = largest_cluster(oxygen, frame.bounds, config.cluster_cutoff_A)
        cluster = oxygen[members]
        center = periodic_center(cluster, frame.bounds)
        shifted_surface = (frame.surface - frame.bounds[:, 0]) % lengths
        shifted_oxygen = (oxygen - frame.bounds[:, 0]) % lengths
        distances = cKDTree(shifted_surface, boxsize=lengths).query(shifted_oxygen, k=1)[0]
        contact_points = oxygen[members[distances[members] <= config.contact_cutoff_A]]
        n2_contact_count = 0
    base = {
        "phase_center_x_A": float(center[0]),
        "phase_center_y_A": float(center[1]),
        "phase_center_z_A": float(center[2]),
        "largest_cluster_size": len(cluster),
        "contact_phase_point_count": len(contact_points),
        "bubble_contact_n2_count": int(n2_contact_count),
    }
    if len(contact_points) < config.minimum_contact_points:
        return base, {"reason": "too_few_contact_phase_points"}
    local_cluster = minimum_image_vectors(cluster - center, lengths)
    local_contact = minimum_image_vectors(contact_points - center, lengths)[:, :2]
    try:
        hull = ConvexHull(local_contact)
    except QhullError:
        return base, {"reason": "degenerate_contact_hull"}
    boundary = local_contact[hull.vertices]
    if not _point_inside_convex_hull(hull):
        return base, {"reason": "phase_center_outside_contact_hull"}
    theta = 2 * math.pi * np.arange(config.arc_bins) / config.arc_bins
    radii = ray_polygon_radii(boundary, theta)
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        return base, {"reason": "incomplete_ray_intersections"}
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    contour_local = radii[:, None] * directions
    normals, curvature, arc_length = contour_differential(
        contour_local, 2 * math.pi / config.arc_bins
    )
    absolute = contour_local + center[:2]
    absolute = (absolute - frame.bounds[:2, 0]) % lengths[:2] + frame.bounds[:2, 0]
    edge_lengths = np.linalg.norm(boundary - np.roll(boundary, 1, axis=0), axis=1)
    base.update(
        {
            "contact_line_area_A2": float(hull.volume),
            "contact_line_perimeter_A": float(hull.area),
            "contact_line_mean_radius_A": float(np.mean(radii)),
            "contact_line_radius_std_A": float(np.std(radii, ddof=1)),
            "contact_line_radius_min_A": float(np.min(radii)),
            "contact_line_radius_max_A": float(np.max(radii)),
            "contact_line_circularity": float(4 * math.pi * hull.volume / hull.area**2),
            "contact_line_hull_vertex_count": len(boundary),
            "contact_line_mean_hull_edge_A": float(np.mean(edge_lengths)),
        }
    )
    return base, {
        "reason": None,
        "center": center,
        "cluster_local": local_cluster,
        "contact_points": contact_points,
        "boundary_local": boundary,
        "theta": theta,
        "directions": directions,
        "radii": radii,
        "contour_local": contour_local,
        "contour_absolute": absolute,
        "normal": normals,
        "curvature": curvature,
        "arc_length": arc_length,
    }


def local_contact_angles(
    cluster_local: np.ndarray,
    center_z_A: float,
    surface_z_A: float,
    box_z_A: float,
    theta: np.ndarray,
    config: TpclConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit local outer-envelope slopes and return phase-side contact angles."""

    radial = np.linalg.norm(cluster_local[:, :2], axis=1)
    point_theta = np.arctan2(cluster_local[:, 1], cluster_local[:, 0]) % (2 * math.pi)
    z = (center_z_A + cluster_local[:, 2] - surface_z_A) % box_z_A
    z = np.where(z > 0.5 * box_z_A, z - box_z_A, z)
    half_width = max(1.5 * math.pi * 2 / config.arc_bins, math.radians(15.0))
    edges = np.arange(0.0, config.angle_fit_z_max_A + config.angle_fit_bin_A, config.angle_fit_bin_A)
    angles = np.full(config.arc_bins, np.nan)
    rmse = np.full(config.arc_bins, np.nan)
    fit_points = np.zeros(config.arc_bins, dtype=int)
    for arc, value in enumerate(theta):
        delta = np.abs((point_theta - value + math.pi) % (2 * math.pi) - math.pi)
        selected = delta <= half_width
        envelope_z, envelope_r = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = selected & (z >= low) & (z < high)
            if np.count_nonzero(mask) >= 2:
                envelope_z.append(0.5 * (low + high))
                envelope_r.append(float(np.quantile(radial[mask], 0.9)))
        if len(envelope_z) < 3:
            continue
        coefficients = np.polyfit(envelope_z, envelope_r, 1)
        predicted = np.polyval(coefficients, envelope_z)
        angles[arc] = math.degrees(math.atan2(1.0, -float(coefficients[0])))
        rmse[arc] = float(np.sqrt(np.mean((np.asarray(envelope_r) - predicted) ** 2)))
        fit_points[arc] = len(envelope_z)
    return angles, rmse, fit_points


@dataclass(frozen=True)
class SurfaceSites:
    atom_ids: np.ndarray
    site_types: np.ndarray
    sioh_hydrogen_ids: Mapping[int, int]
    ch3_count: int
    sioh_count: int


def load_surface_sites(config: TpclConfig) -> SurfaceSites:
    elements, coordinates, lengths = read_extxyz_positions(config.initial_xyz)
    sites = identify_surface_sites(
        elements,
        coordinates,
        lengths,
        slab_range=config.surface_range,
        surface_z=config.surface_z_A,
        surface_depth=config.surface_depth_A,
        bond_cutoff=config.site_bond_cutoff_A,
    )
    atom_ids = np.asarray([row["atom_id"] for row in sites], dtype=int)
    site_types = np.asarray([row["site_type"] for row in sites])
    slab_start, slab_end = config.surface_range[0] - 1, config.surface_range[1]
    slab_elements = elements[slab_start:slab_end]
    slab_coordinates = coordinates[slab_start:slab_end]
    hydrogen_local = np.flatnonzero(slab_elements == "H")
    hydrogen_ids = slab_start + hydrogen_local + 1
    hydrogen_coordinates = slab_coordinates[hydrogen_local]
    sioh_hydrogen_ids: dict[int, int] = {}
    if len(hydrogen_coordinates):
        from scipy.spatial import cKDTree

        tree = cKDTree(hydrogen_coordinates % lengths, boxsize=lengths)
        for atom_id, site_type in zip(atom_ids, site_types):
            if site_type != "SiOH":
                continue
            oxygen = coordinates[atom_id - 1]
            distance, index = tree.query(
                oxygen % lengths, distance_upper_bound=config.site_bond_cutoff_A
            )
            if np.isfinite(distance) and index < len(hydrogen_ids):
                sioh_hydrogen_ids[int(atom_id)] = int(hydrogen_ids[int(index)])
    ch3_count = int(np.count_nonzero(site_types == "CH3"))
    sioh_count = int(np.count_nonzero(site_types == "SiOH"))
    return SurfaceSites(atom_ids, site_types, sioh_hydrogen_ids, ch3_count, sioh_count)


def current_site_coordinates(frame: TpclFrame, sites: SurfaceSites, config: TpclConfig) -> np.ndarray:
    indices = sites.atom_ids - config.surface_range[0]
    if np.any(indices < 0) or np.any(indices >= len(frame.surface)):
        raise ValueError("A surface site ID lies outside the configured surface range")
    if not np.array_equal(frame.surface_ids[indices], sites.atom_ids):
        raise ValueError("Sorted surface IDs do not match the surface-site definition")
    return frame.surface[indices]


def local_site_metrics(
    contour_xy: np.ndarray,
    site_coordinates: np.ndarray,
    sites: SurfaceSites,
    bounds: np.ndarray,
    config: TpclConfig,
) -> list[dict]:
    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)[:2]
    shifted_sites = (site_coordinates[:, :2] - bounds[:2, 0]) % lengths
    shifted_contour = (contour_xy - bounds[:2, 0]) % lengths
    tree = cKDTree(shifted_sites, boxsize=lengths)
    nearest_distance, nearest_index = tree.query(shifted_contour)
    neighborhoods = tree.query_ball_point(shifted_contour, config.local_site_radius_A)
    ch3_mask = sites.site_types == "CH3"
    ch3_tree = cKDTree(shifted_sites[ch3_mask], boxsize=lengths) if np.any(ch3_mask) else None
    oh_tree = cKDTree(shifted_sites[~ch3_mask], boxsize=lengths) if np.any(~ch3_mask) else None
    d_ch3 = (
        ch3_tree.query(shifted_contour)[0] if ch3_tree is not None else np.full(len(contour_xy), np.nan)
    )
    d_oh = oh_tree.query(shifted_contour)[0] if oh_tree is not None else np.full(len(contour_xy), np.nan)
    rows = []
    for arc, (distance, site_index, neighbors) in enumerate(
        zip(nearest_distance, nearest_index, neighborhoods)
    ):
        local_types = sites.site_types[np.asarray(neighbors, dtype=int)]
        fraction = float(np.mean(local_types == "CH3")) if len(local_types) else math.nan
        boundary_proxy = (
            0.5 * abs(float(d_ch3[arc]) - float(d_oh[arc]))
            if np.isfinite(d_ch3[arc]) and np.isfinite(d_oh[arc])
            else math.nan
        )
        rows.append(
            {
                "nearest_site_id": int(sites.atom_ids[int(site_index)]),
                "nearest_site_type": str(sites.site_types[int(site_index)]),
                "nearest_site_distance_A": float(distance),
                "local_site_count": len(local_types),
                "local_ch3_fraction": fraction,
                "chemical_boundary_proxy_A": boundary_proxy,
                "near_chemical_boundary": bool(
                    np.isfinite(boundary_proxy) and boundary_proxy <= config.chemical_boundary_A
                ),
            }
        )
    return rows


def local_water_metrics(
    frame: TpclFrame,
    contour_xy: np.ndarray,
    arc_length: np.ndarray,
    surface_z_A: float,
    site_coordinates: np.ndarray,
    sites: SurfaceSites,
    config: TpclConfig,
) -> dict[str, np.ndarray]:
    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    oxygen = frame.water[frame.water_types == config.oxygen_type]
    hydrogen = frame.water[frame.water_types == config.hydrogen_type]
    z = minimum_image_vectors(
        np.column_stack((np.zeros(len(oxygen)), np.zeros(len(oxygen)), oxygen[:, 2] - surface_z_A)),
        lengths,
    )[:, 2]
    layer = (z >= config.hydration_z_min_A) & (z < config.hydration_z_max_A)
    shifted_contour = (contour_xy - frame.bounds[:2, 0]) % lengths[:2]
    shifted_oxygen = (oxygen[:, :2] - frame.bounds[:2, 0]) % lengths[:2]
    contour_tree = cKDTree(shifted_contour, boxsize=lengths[:2])
    line_distance, nearest_arc = contour_tree.query(shifted_oxygen)
    tpcl = layer & (line_distance <= config.tpcl_half_width_A)
    hydration_count = np.bincount(nearest_arc[tpcl], minlength=config.arc_bins).astype(int)
    area = 2.0 * config.tpcl_half_width_A * arc_length
    hydration_density = np.divide(
        hydration_count, area, out=np.full(config.arc_bins, np.nan), where=area > 0
    )

    assigned = assign_hydrogen_neighbors(oxygen, hydrogen, frame.bounds, config.oh_cutoff_A)
    molecular = np.asarray([len(items) == 2 for items in assigned])
    selected_indices = np.flatnonzero(tpcl & molecular)
    selected_oxygen = oxygen[selected_indices]
    selected_arc = nearest_arc[selected_indices]
    oh_vectors = [
        minimum_image_vectors(hydrogen[assigned[index]] - oxygen[index], lengths)
        for index in selected_indices
    ]
    water_count = np.bincount(selected_arc, minlength=config.arc_bins).astype(int)
    water_degree_sum = np.zeros(config.arc_bins)
    if len(selected_oxygen) >= 2:
        shifted_selected = (selected_oxygen - frame.bounds[:, 0]) % lengths
        for left, right in cKDTree(shifted_selected, boxsize=lengths).query_pairs(
            config.oo_cutoff_A
        ):
            vector = minimum_image_vectors(selected_oxygen[right] - selected_oxygen[left], lengths)
            if donor_points_to(oh_vectors[left], vector, config.hbond_angle_deg) or donor_points_to(
                oh_vectors[right], -vector, config.hbond_angle_deg
            ):
                water_degree_sum[int(selected_arc[left])] += 1.0
                water_degree_sum[int(selected_arc[right])] += 1.0
    water_hbond_degree = np.divide(
        water_degree_sum,
        water_count,
        out=np.full(config.arc_bins, np.nan),
        where=water_count > 0,
    )

    surface_hbond_count = np.zeros(config.arc_bins, dtype=int)
    sioh_pairs = [
        (oxygen_id, hydrogen_id)
        for oxygen_id, hydrogen_id in sites.sioh_hydrogen_ids.items()
        if config.surface_range[0] <= hydrogen_id <= config.surface_range[1]
    ]
    if len(selected_oxygen) and sioh_pairs:
        oxygen_indices = np.asarray([item[0] - config.surface_range[0] for item in sioh_pairs])
        hydrogen_indices = np.asarray([item[1] - config.surface_range[0] for item in sioh_pairs])
        valid_index = (
            (oxygen_indices >= 0)
            & (oxygen_indices < len(frame.surface))
            & (hydrogen_indices >= 0)
            & (hydrogen_indices < len(frame.surface))
        )
        oxygen_indices, hydrogen_indices = oxygen_indices[valid_index], hydrogen_indices[valid_index]
        surface_oxygen = frame.surface[oxygen_indices]
        surface_oh = minimum_image_vectors(
            frame.surface[hydrogen_indices] - surface_oxygen, lengths
        )
        protonated = np.linalg.norm(surface_oh, axis=1) <= config.oh_cutoff_A
        surface_oxygen, surface_oh = surface_oxygen[protonated], surface_oh[protonated]
        if len(surface_oxygen):
            surface_tree = cKDTree(
                (surface_oxygen - frame.bounds[:, 0]) % lengths, boxsize=lengths
            )
            for water_index, neighbors in enumerate(
                surface_tree.query_ball_point(
                    (selected_oxygen - frame.bounds[:, 0]) % lengths, config.oo_cutoff_A
                )
            ):
                for surface_index in neighbors:
                    vector = minimum_image_vectors(
                        surface_oxygen[surface_index] - selected_oxygen[water_index], lengths
                    )
                    if donor_points_to(
                        oh_vectors[water_index], vector, config.hbond_angle_deg
                    ) or donor_points_to(
                        surface_oh[surface_index][None, :], -vector, config.hbond_angle_deg
                    ):
                        surface_hbond_count[int(selected_arc[water_index])] += 1
    surface_hbond_per_water = np.divide(
        surface_hbond_count,
        water_count,
        out=np.full(config.arc_bins, np.nan),
        where=water_count > 0,
    )
    return {
        "local_hydration_oxygen_count": hydration_count,
        "local_hydration_areal_density_A-2": hydration_density,
        "local_molecular_water_count": water_count,
        "local_water_water_hbond_degree": water_hbond_degree,
        "local_surface_water_hbond_per_h2o": surface_hbond_per_water,
    }


def analyze_tpcl_frame(
    frame: TpclFrame,
    surface_z_A: float,
    sites: SurfaceSites,
    config: TpclConfig,
) -> tuple[dict, list[dict], dict | None]:
    geometry, contour = phase_contour(frame, config)
    row = {
        "source_file": str(frame.source),
        "source_frame": frame.source_frame,
        "segment_index": frame.segment_index,
        "step": frame.step,
        "time_ns": frame.step * config.timestep_fs / 1.0e6,
        "surface_reference_z_A": surface_z_A,
        "box_x_A": float(box_lengths(frame.bounds)[0]),
        "box_y_A": float(box_lengths(frame.bounds)[1]),
        "box_z_A": float(box_lengths(frame.bounds)[2]),
        **geometry,
    }
    if contour is None or contour["reason"] is not None:
        row["contour_valid"] = False
        row["contour_invalid_reason"] = (
            "unknown_contour_failure" if contour is None else contour["reason"]
        )
        return row, [], None
    row["contour_valid"] = True
    row["contour_invalid_reason"] = ""
    site_coordinates = current_site_coordinates(frame, sites, config)
    site_rows = local_site_metrics(
        contour["contour_absolute"], site_coordinates, sites, frame.bounds, config
    )
    angles, angle_rmse, angle_points = local_contact_angles(
        contour["cluster_local"],
        float(contour["center"][2]),
        surface_z_A,
        float(box_lengths(frame.bounds)[2]),
        contour["theta"],
        config,
    )
    water = local_water_metrics(
        frame,
        contour["contour_absolute"],
        contour["arc_length"],
        surface_z_A,
        site_coordinates,
        sites,
        config,
    )
    n2_count = np.zeros(config.arc_bins, dtype=int)
    if config.kind == "nanobubble" and len(contour["contact_points"]):
        from scipy.spatial import cKDTree

        lengths = box_lengths(frame.bounds)[:2]
        tree = cKDTree(
            (contour["contour_absolute"] - frame.bounds[:2, 0]) % lengths,
            boxsize=lengths,
        )
        nearest = tree.query(
            (contour["contact_points"][:, :2] - frame.bounds[:2, 0]) % lengths
        )[1]
        n2_count = np.bincount(nearest, minlength=config.arc_bins).astype(int)
    local_rows = []
    for arc in range(config.arc_bins):
        local = {
            "source_file": str(frame.source),
            "source_frame": frame.source_frame,
            "segment_index": frame.segment_index,
            "step": frame.step,
            "time_ns": frame.step * config.timestep_fs / 1.0e6,
            "arc_index": arc,
            "theta_deg": math.degrees(float(contour["theta"][arc])),
            "contour_x_A": float(contour["contour_absolute"][arc, 0]),
            "contour_y_A": float(contour["contour_absolute"][arc, 1]),
            "contour_z_A": surface_z_A,
            "local_radius_A": float(contour["radii"][arc]),
            "local_normal_x": float(contour["normal"][arc, 0]),
            "local_normal_y": float(contour["normal"][arc, 1]),
            "local_curvature_A-1": float(contour["curvature"][arc]),
            "local_arc_length_A": float(contour["arc_length"][arc]),
            "local_contact_angle_deg": float(angles[arc]),
            "local_contact_angle_fit_rmse_A": float(angle_rmse[arc]),
            "local_contact_angle_fit_points": int(angle_points[arc]),
            "local_n2_contact_count": int(n2_count[arc]),
            **site_rows[arc],
        }
        for name, values in water.items():
            value = values[arc]
            local[name] = int(value) if name.endswith("_count") else float(value)
        local_rows.append(local)
    row["valid_local_contact_angle_count"] = int(np.count_nonzero(np.isfinite(angles)))
    snapshot = {
        "step": frame.step,
        "bounds": frame.bounds.copy(),
        "center": contour["center"].copy(),
        "contour": contour["contour_absolute"].copy(),
        "contact_points": contour["contact_points"].copy(),
        "site_coordinates": site_coordinates.copy(),
        "site_types": sites.site_types.copy(),
    }
    return row, local_rows, snapshot


def _fourier_components(radii: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        (
            np.ones(len(theta)),
            np.cos(theta),
            np.sin(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
        )
    )
    coefficients = np.linalg.lstsq(design, radii, rcond=None)[0]
    return coefficients, design @ coefficients


def finalize_kinematics(
    frame_rows: list[dict], local_by_step: dict[int, list[dict]], config: TpclConfig
) -> tuple[float, float, float, int]:
    """Add unwrapped coordinates and motion decomposition in place."""

    if not frame_rows:
        raise ValueError("No frame metrics to finalize")
    lengths = np.asarray([frame_rows[0][f"box_{dim}_A"] for dim in "xyz"])
    previous_wrapped = None
    unwrapped = None
    origin = None
    for row in frame_rows:
        current = np.asarray([row[f"phase_center_{dim}_A"] for dim in "xyz"], dtype=float)
        if previous_wrapped is None:
            unwrapped = current.copy()
            origin = current.copy()
        else:
            unwrapped += minimum_image_vectors(current - previous_wrapped, lengths)
        for dim, value in zip("xyz", unwrapped):
            row[f"phase_center_{dim}_unwrapped_A"] = float(value)
        row["phase_center_lateral_displacement_A"] = float(np.linalg.norm(unwrapped[:2] - origin[:2]))
        previous_wrapped = current

    theta = 2 * math.pi * np.arange(config.arc_bins) / config.arc_bins
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    frame_by_step = {int(row["step"]): row for row in frame_rows}
    for step, local_rows in local_by_step.items():
        frame = frame_by_step[step]
        radii = np.asarray([row["local_radius_A"] for row in local_rows])
        coefficients, fitted = _fourier_components(radii, theta)
        mean_radius = float(coefficients[0])
        low_shape = fitted - mean_radius
        residual = radii - fitted
        frame["decomposed_mean_radius_A"] = mean_radius
        frame["shape_mode_1_amplitude_A"] = float(np.hypot(coefficients[1], coefficients[2]))
        frame["shape_mode_2_amplitude_A"] = float(np.hypot(coefficients[3], coefficients[4]))
        center = np.asarray(
            [frame["phase_center_x_unwrapped_A"], frame["phase_center_y_unwrapped_A"]]
        )
        for arc, local in enumerate(local_rows):
            position = center + radii[arc] * directions[arc]
            local["contour_x_unwrapped_A"] = float(position[0])
            local["contour_y_unwrapped_A"] = float(position[1])
            local["mean_radius_component_A"] = mean_radius
            local["low_order_shape_component_A"] = float(low_shape[arc])
            local["local_residual_A"] = float(residual[arc])
            local["absolute_normal_position_A"] = float(position @ directions[arc])
            local["phase_center_normal_position_A"] = float(center @ directions[arc])
            local["consecutive_from_previous"] = False
            for name in (
                "absolute_normal_displacement_A",
                "phase_center_normal_displacement_A",
                "mean_radius_displacement_A",
                "low_order_shape_displacement_A",
                "local_residual_displacement_A",
                "local_normal_velocity_A_per_ps",
            ):
                local[name] = math.nan

    steps = [int(row["step"]) for row in frame_rows]
    positive_deltas = np.diff(sorted(set(steps)))
    if len(positive_deltas) == 0:
        raise ValueError("At least two unique frames are required")
    expected_step = int(np.median(positive_deltas))
    interval_ps = expected_step * config.timestep_fs / 1000.0
    residual_changes = []
    for previous_frame, current_frame in zip(frame_rows[:-1], frame_rows[1:]):
        previous_step, current_step = int(previous_frame["step"]), int(current_frame["step"])
        if previous_step not in local_by_step or current_step not in local_by_step:
            continue
        if (
            current_step - previous_step != expected_step
            or current_frame["segment_index"] != previous_frame["segment_index"]
        ):
            continue
        previous_rows, current_rows = local_by_step[previous_step], local_by_step[current_step]
        for arc, (previous, current) in enumerate(zip(previous_rows, current_rows)):
            normal = np.asarray(
                [
                    previous["local_normal_x"] + current["local_normal_x"],
                    previous["local_normal_y"] + current["local_normal_y"],
                ]
            )
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else directions[arc]
            radial_projection = float(directions[arc] @ normal)
            current_position = np.asarray(
                [current["contour_x_unwrapped_A"], current["contour_y_unwrapped_A"]]
            )
            previous_position = np.asarray(
                [previous["contour_x_unwrapped_A"], previous["contour_y_unwrapped_A"]]
            )
            current_center = np.asarray(
                [
                    current_frame["phase_center_x_unwrapped_A"],
                    current_frame["phase_center_y_unwrapped_A"],
                ]
            )
            previous_center = np.asarray(
                [
                    previous_frame["phase_center_x_unwrapped_A"],
                    previous_frame["phase_center_y_unwrapped_A"],
                ]
            )
            actual = float((current_position - previous_position) @ normal)
            center_change = float((current_center - previous_center) @ normal)
            mean_change = float(
                (current["mean_radius_component_A"] - previous["mean_radius_component_A"])
                * radial_projection
            )
            low_change = float(
                (
                    current["low_order_shape_component_A"]
                    - previous["low_order_shape_component_A"]
                )
                * radial_projection
            )
            residual_change = float(
                (current["local_residual_A"] - previous["local_residual_A"])
                * radial_projection
            )
            current.update(
                {
                    "consecutive_from_previous": True,
                    "absolute_normal_displacement_A": actual,
                    "phase_center_normal_displacement_A": center_change,
                    "mean_radius_displacement_A": mean_change,
                    "low_order_shape_displacement_A": low_change,
                    "local_residual_displacement_A": residual_change,
                    "local_normal_velocity_A_per_ps": actual / interval_ps,
                }
            )
            residual_changes.append(residual_change)
    changes = np.asarray(residual_changes, dtype=float)
    if len(changes):
        median = float(np.median(changes))
        mad = float(np.median(np.abs(changes - median)))
        noise = max(config.noise_floor_A, 1.4826 * mad / math.sqrt(2.0))
    else:
        noise = config.noise_floor_A
    dwell_tolerance = max(config.minimum_dwell_tolerance_A, 2.0 * noise)
    jump_threshold = max(config.minimum_jump_A, config.jump_noise_multiplier * noise)
    for rows in local_by_step.values():
        for row in rows:
            row["localization_noise_A"] = noise
            row["dwell_tolerance_A"] = dwell_tolerance
            row["jump_threshold_A"] = jump_threshold
    return noise, dwell_tolerance, jump_threshold, expected_step


EVENT_METRICS = (
    "local_ch3_fraction",
    "chemical_boundary_proxy_A",
    "local_hydration_oxygen_count",
    "local_hydration_areal_density_A-2",
    "local_water_water_hbond_degree",
    "local_surface_water_hbond_per_h2o",
    "local_n2_contact_count",
    "local_curvature_A-1",
    "local_contact_angle_deg",
)


def _mean(rows: Sequence[dict], name: str) -> float:
    values = np.asarray([row[name] for row in rows], dtype=float)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else math.nan


def _mode(values: Sequence[object]) -> object:
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _consecutive_groups(frame_rows: Sequence[dict], local_by_step: Mapping[int, list[dict]]):
    groups: list[list[int]] = []
    current: list[int] = []
    for row in frame_rows:
        step = int(row["step"])
        if step not in local_by_step:
            if current:
                groups.append(current)
                current = []
            continue
        first_local = local_by_step[step][0]
        if current and not first_local["consecutive_from_previous"]:
            groups.append(current)
            current = []
        current.append(step)
    if current:
        groups.append(current)
    return groups


def detect_events(
    frame_rows: Sequence[dict],
    local_by_step: Mapping[int, list[dict]],
    config: TpclConfig,
    dwell_tolerance: float,
    jump_threshold: float,
    expected_step: int,
) -> list[dict]:
    events: list[dict] = []
    interval_ps = expected_step * config.timestep_fs / 1000.0
    for steps in _consecutive_groups(frame_rows, local_by_step):
        if len(steps) < config.minimum_dwell_frames + config.post_confirm_frames:
            continue
        for arc in range(config.arc_bins):
            position = np.asarray(
                [local_by_step[step][arc]["absolute_normal_position_A"] for step in steps]
            )
            residual = np.asarray([local_by_step[step][arc]["local_residual_A"] for step in steps])
            center = np.asarray(
                [local_by_step[step][arc]["phase_center_normal_position_A"] for step in steps]
            )
            start = 0
            while start + config.minimum_dwell_frames + config.post_confirm_frames <= len(steps):
                dwell_end = start + config.minimum_dwell_frames
                if np.ptp(position[start:dwell_end]) > dwell_tolerance:
                    start += 1
                    continue
                while dwell_end < len(steps) and np.ptp(position[start : dwell_end + 1]) <= dwell_tolerance:
                    dwell_end += 1
                post_end = dwell_end + config.post_confirm_frames
                if post_end > len(steps):
                    break
                pre_slice, post_slice = slice(start, dwell_end), slice(dwell_end, post_end)
                jump = float(np.median(position[post_slice]) - np.median(position[pre_slice]))
                if abs(jump) < config.minimum_jump_A:
                    start = max(start + 1, dwell_end)
                    continue
                post_stable = float(np.ptp(position[post_slice])) <= dwell_tolerance
                residual_jump = float(
                    np.median(residual[post_slice]) - np.median(residual[pre_slice])
                )
                center_jump = float(np.median(center[post_slice]) - np.median(center[pre_slice]))
                translation_fraction = abs(center_jump) / abs(jump) if jump else math.inf
                neighbor_jumps = []
                for neighbor in ((arc - 1) % config.arc_bins, (arc + 1) % config.arc_bins):
                    neighbor_position = np.asarray(
                        [
                            local_by_step[step][neighbor]["absolute_normal_position_A"]
                            for step in steps
                        ]
                    )
                    neighbor_jumps.append(
                        float(
                            np.median(neighbor_position[post_slice])
                            - np.median(neighbor_position[pre_slice])
                        )
                    )
                coherent = any(
                    np.sign(value) == np.sign(jump) and abs(value) >= 0.5 * jump_threshold
                    for value in neighbor_jumps
                )
                pre_rows = [local_by_step[step][arc] for step in steps[pre_slice]]
                post_rows = [local_by_step[step][arc] for step in steps[post_slice]]
                site_ids = [row["nearest_site_id"] for row in pre_rows]
                dominant_site = _mode(site_ids)
                site_stability = site_ids.count(dominant_site) / len(site_ids)
                reasons = []
                if abs(jump) < jump_threshold:
                    reasons.append("below_noise_scaled_jump_threshold")
                if not post_stable:
                    reasons.append("post_state_not_stable")
                if abs(residual_jump) < config.residual_noise_multiplier * pre_rows[0]["localization_noise_A"]:
                    reasons.append("local_residual_change_below_threshold")
                if translation_fraction > config.maximum_translation_fraction:
                    reasons.append("whole_object_translation_dominated")
                if not coherent:
                    reasons.append("no_adjacent_arc_coherence")
                if site_stability < config.minimum_site_stability_fraction:
                    reasons.append("dwell_not_stable_relative_to_surface_site")
                boundary = _mean(pre_rows, "chemical_boundary_proxy_A")
                mechanism_class = (
                    "boundary"
                    if np.isfinite(boundary) and boundary <= config.chemical_boundary_A
                    else str(_mode([row["nearest_site_type"] for row in pre_rows]))
                )
                event = {
                    "event_id": len(events) + 1,
                    "arc_index": arc,
                    "theta_deg": pre_rows[0]["theta_deg"],
                    "start_step": steps[start],
                    "end_step": steps[dwell_end - 1],
                    "transition_step": steps[dwell_end],
                    "post_end_step": steps[post_end - 1],
                    "start_time_ns": steps[start] * config.timestep_fs / 1.0e6,
                    "end_time_ns": steps[dwell_end - 1] * config.timestep_fs / 1.0e6,
                    "transition_time_ns": steps[dwell_end] * config.timestep_fs / 1.0e6,
                    "dwell_frames": dwell_end - start,
                    "dwell_time_ps": (dwell_end - start) * interval_ps,
                    "jump_signed_A": jump,
                    "jump_distance_A": abs(jump),
                    "local_residual_jump_A": residual_jump,
                    "phase_center_jump_A": center_jump,
                    "translation_fraction": translation_fraction,
                    "left_neighbor_jump_A": neighbor_jumps[0],
                    "right_neighbor_jump_A": neighbor_jumps[1],
                    "dwell_site_stability_fraction": site_stability,
                    "pre_nearest_site_id": dominant_site,
                    "pre_nearest_site_type": _mode(
                        [row["nearest_site_type"] for row in pre_rows]
                    ),
                    "post_nearest_site_id": _mode(
                        [row["nearest_site_id"] for row in post_rows]
                    ),
                    "post_nearest_site_type": _mode(
                        [row["nearest_site_type"] for row in post_rows]
                    ),
                    "mechanism_class": mechanism_class,
                    "quality_status": "preliminary_candidate" if not reasons else "rejected",
                    "rejection_reason": ";".join(reasons),
                }
                for metric in EVENT_METRICS:
                    pre_value, post_value = _mean(pre_rows, metric), _mean(post_rows, metric)
                    event[f"pre_{metric}"] = pre_value
                    event[f"post_{metric}"] = post_value
                    event[f"delta_{metric}"] = post_value - pre_value
                events.append(event)
                start = post_end
    preliminary = [
        index for index, event in enumerate(events) if event["quality_status"] == "preliminary_candidate"
    ]
    parent = {index: index for index in preliminary}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def circular_arc_distance(left: int, right: int) -> int:
        direct = abs(left - right)
        return min(direct, config.arc_bins - direct)

    for position, left in enumerate(preliminary):
        for right in preliminary[position + 1 :]:
            if (
                abs(int(events[left]["transition_step"]) - int(events[right]["transition_step"]))
                <= config.post_confirm_frames * expected_step
                and circular_arc_distance(int(events[left]["arc_index"]), int(events[right]["arc_index"]))
                <= 1
            ):
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[right_root] = left_root
    clusters: dict[int, list[int]] = {}
    for index in preliminary:
        clusters.setdefault(find(index), []).append(index)
    class_clusters: dict[str, set[int]] = {}
    for cluster_id, indices in enumerate(clusters.values(), start=1):
        mechanism_class = str(_mode([events[index]["mechanism_class"] for index in indices]))
        class_clusters.setdefault(mechanism_class, set()).add(cluster_id)
        for index in indices:
            events[index]["event_cluster_id"] = cluster_id
            events[index]["event_cluster_arc_count"] = len(indices)
            events[index]["cluster_mechanism_class"] = mechanism_class
    for event in events:
        if event["quality_status"] != "preliminary_candidate":
            event["event_cluster_id"] = ""
            event["event_cluster_arc_count"] = 0
            event["cluster_mechanism_class"] = ""
            continue
        cluster_class = str(event["cluster_mechanism_class"])
        if len(class_clusters[cluster_class]) >= 2:
            event["quality_status"] = "candidate_stick_slip"
        else:
            event["quality_status"] = "insufficient_repetition"
            event["rejection_reason"] = "insufficient_distinct_same_class_event_clusters"
    return events


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    names: list[str] = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames=None) -> None:
    names = list(fieldnames or _fieldnames(rows))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames=None) -> None:
    names = list(fieldnames or _fieldnames(rows))
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _configure_matplotlib(font_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams.update(
        {
            "font.family": properties.get_name(),
            "mathtext.fontset": "custom",
            "mathtext.rm": properties.get_name(),
            "mathtext.it": f"{properties.get_name()}:italic",
            "mathtext.bf": f"{properties.get_name()}:bold",
            "mathtext.cal": properties.get_name(),
            "mathtext.sf": properties.get_name(),
            "mathtext.tt": properties.get_name(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return properties


def _save_figure(figure, path_without_suffix: Path) -> None:
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")


def write_figures(
    frame_rows: Sequence[dict],
    local_by_step: Mapping[int, list[dict]],
    events: Sequence[dict],
    snapshots: Mapping[int, dict],
    output: Path,
    font_path: Path,
    config: TpclConfig,
    expected_step: int,
) -> None:
    _configure_matplotlib(font_path)
    from matplotlib import pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    for index, step in enumerate(sorted(snapshots)):
        snapshot = snapshots[step]
        lengths = box_lengths(snapshot["bounds"])[:2]
        center = snapshot["center"][:2]
        contour = minimum_image_vectors(snapshot["contour"] - center, lengths)
        contact = minimum_image_vectors(snapshot["contact_points"][:, :2] - center, lengths)
        sites = minimum_image_vectors(snapshot["site_coordinates"][:, :2] - center, lengths)
        ch3 = snapshot["site_types"] == "CH3"
        figure, axis = plt.subplots(figsize=(6.2, 5.6))
        axis.scatter(
            sites[ch3, 0], sites[ch3, 1], s=13, facecolors="none", edgecolors="tab:blue", label=r"CH$_3$"
        )
        axis.scatter(
            sites[~ch3, 0], sites[~ch3, 1], s=13, facecolors="none", edgecolors="tab:orange", label="SiOH"
        )
        axis.scatter(contact[:, 0], contact[:, 1], s=10, color="0.45", alpha=0.55, label="Contact phase")
        closed = np.vstack((contour, contour[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="black", lw=1.2, label="TPCL contour")
        axis.scatter([0.0], [0.0], marker="+", color="tab:red", s=60, label="Phase center")
        axis.set_aspect("equal")
        axis.set_xlabel(r"$x-x_c$ (Å)")
        axis.set_ylabel(r"$y-y_c$ (Å)")
        axis.set_title(f"PBC contour QA, step {step}")
        axis.legend(frameon=False, fontsize=8, ncol=2)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        _save_figure(figure, figures / f"01_contour_qa_{index + 1}_step_{step}")
        plt.close(figure)

    first_step, last_step = int(frame_rows[0]["step"]), int(frame_rows[-1]["step"])
    grid_steps = np.arange(first_step, last_step + expected_step, expected_step, dtype=int)
    step_index = {step: index for index, step in enumerate(grid_steps)}
    radius = np.full((len(grid_steps), config.arc_bins), np.nan)
    residual = np.full_like(radius, np.nan)
    velocity = np.full_like(radius, np.nan)
    for step, rows in local_by_step.items():
        if step not in step_index:
            continue
        index = step_index[step]
        radius[index] = [row["local_radius_A"] for row in rows]
        residual[index] = [row["local_residual_A"] for row in rows]
        velocity[index] = [row["local_normal_velocity_A_per_ps"] for row in rows]
    time = grid_steps * config.timestep_fs / 1.0e6
    theta_edges = np.linspace(0.0, 360.0, config.arc_bins + 1)
    time_edges = np.concatenate(
        (
            [time[0] - 0.5 * expected_step * config.timestep_fs / 1.0e6],
            0.5 * (time[:-1] + time[1:]),
            [time[-1] + 0.5 * expected_step * config.timestep_fs / 1.0e6],
        )
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    for axis, values, label, cmap in (
        (axes[0], radius, r"Local radius (Å)", "viridis"),
        (axes[1], residual, r"Local residual (Å)", "RdBu_r"),
    ):
        mesh = axis.pcolormesh(time_edges, theta_edges, np.ma.masked_invalid(values.T), shading="flat", cmap=cmap)
        figure.colorbar(mesh, ax=axis, label=label)
        axis.set_ylabel(r"Polar angle $\theta$ (deg)")
    axes[1].set_xlabel("Time (ns)")
    figure.tight_layout()
    _save_figure(figure, figures / "02_radius_residual_kymograph")
    plt.close(figure)

    times = np.asarray([row["time_ns"] for row in frame_rows])
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].plot(times, [row["phase_center_lateral_displacement_A"] for row in frame_rows], lw=0.9)
    axes[0].set_ylabel("Center shift (Å)")
    axes[1].plot(times, [row.get("decomposed_mean_radius_A", math.nan) for row in frame_rows], lw=0.9)
    axes[1].set_ylabel(r"Mean $R_{CL}$ (Å)")
    axes[2].plot(times, [row.get("shape_mode_1_amplitude_A", math.nan) for row in frame_rows], label="Mode 1")
    axes[2].plot(times, [row.get("shape_mode_2_amplitude_A", math.nan) for row in frame_rows], label="Mode 2")
    axes[2].set_ylabel("Mode amplitude (Å)")
    axes[2].set_xlabel("Time (ns)")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, figures / "03_motion_shape_separation")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    limit = np.nanquantile(np.abs(velocity), 0.98) if np.any(np.isfinite(velocity)) else 1.0
    mesh = axis.pcolormesh(
        time_edges,
        theta_edges,
        np.ma.masked_invalid(velocity.T),
        shading="flat",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    for event in events:
        if event["quality_status"] == "candidate_stick_slip":
            axis.scatter(
                event["transition_time_ns"],
                event["theta_deg"],
                marker="o",
                facecolors="none",
                edgecolors="black",
                s=35,
            )
    figure.colorbar(mesh, ax=axis, label=r"Local normal velocity (Å ps$^{-1}$)")
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel(r"Polar angle $\theta$ (deg)")
    figure.tight_layout()
    _save_figure(figure, figures / "04_local_velocity_event_timeline")
    plt.close(figure)


def _serializable_config(config: TpclConfig) -> dict:
    value = asdict(config)
    value["trajectories"] = [str(path.resolve()) for path in config.trajectories]
    value["initial_xyz"] = str(config.initial_xyz.resolve())
    return value


def run_analysis(
    config: TpclConfig,
    output_dir: Path,
    *,
    font_path: Path | None = None,
    start_ns: float | None = None,
    end_ns: float | None = None,
    max_frames: int | None = None,
    drop_first_frame: bool = True,
    make_plots: bool = True,
) -> dict:
    config.validate()
    if make_plots and font_path is None:
        raise ValueError("font_path is required when plots are enabled")
    if start_ns is not None and end_ns is not None and end_ns < start_ns:
        raise ValueError("end_ns precedes start_ns")
    if max_frames is not None and max_frames < 2:
        raise ValueError("max_frames must be at least two")
    start_step = None if start_ns is None else round(start_ns * 1.0e6 / config.timestep_fs)
    end_step = None if end_ns is None else round(end_ns * 1.0e6 / config.timestep_fs)
    surface_reference = load_surface_reference(
        config.initial_xyz, config.surface_range, config.surface_z_A
    )
    sites = load_surface_sites(config)
    records: dict[int, tuple[dict, list[dict]]] = {}
    snapshots: dict[int, dict] = {}
    raw_frames = 0
    window_frames = 0
    reference_bounds = None
    stop = False
    for segment_index, trajectory in enumerate(config.trajectories):
        for frame in iter_tpcl_frames(trajectory, config, segment_index):
            raw_frames += 1
            if reference_bounds is None:
                reference_bounds = frame.bounds
            elif not np.allclose(frame.bounds, reference_bounds):
                raise ValueError("TPCL analysis requires a constant orthorhombic box")
            if (start_step is not None and frame.step < start_step) or (
                end_step is not None and frame.step > end_step
            ):
                continue
            surface_z = surface_reference.plane_z(frame.surface, frame.bounds)
            frame_row, local_rows, snapshot = analyze_tpcl_frame(frame, surface_z, sites, config)
            records[frame.step] = frame_row, local_rows
            if snapshot is not None:
                snapshots[frame.step] = snapshot
                if len(snapshots) > 2:
                    for obsolete in sorted(snapshots)[1:-1]:
                        del snapshots[obsolete]
            window_frames += 1
            if max_frames is not None and window_frames >= max_frames:
                stop = True
                break
        if stop:
            break
    ordered_steps = sorted(records)
    if drop_first_frame and ordered_steps and ordered_steps[0] == 0:
        del records[0]
        snapshots.pop(0, None)
        ordered_steps = sorted(records)
    if len(ordered_steps) < 2:
        raise ValueError("Fewer than two analyzed frames remain")
    frame_rows = [records[step][0] for step in ordered_steps]
    local_by_step = {step: records[step][1] for step in ordered_steps if records[step][1]}
    if len(local_by_step) < 2:
        raise ValueError("Fewer than two frames have a valid TPCL contour")
    noise, dwell_tolerance, jump_threshold, expected_step = finalize_kinematics(
        frame_rows, local_by_step, config
    )
    events = detect_events(
        frame_rows,
        local_by_step,
        config,
        dwell_tolerance,
        jump_threshold,
        expected_step,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "frame_metrics.csv", frame_rows)
    local_rows = [row for step in ordered_steps for row in local_by_step.get(step, [])]
    contour_fields = (
        "source_file",
        "source_frame",
        "segment_index",
        "step",
        "time_ns",
        "arc_index",
        "theta_deg",
        "contour_x_A",
        "contour_y_A",
        "contour_z_A",
        "contour_x_unwrapped_A",
        "contour_y_unwrapped_A",
        "local_radius_A",
        "local_normal_x",
        "local_normal_y",
        "local_curvature_A-1",
        "local_contact_angle_deg",
        "local_contact_angle_fit_rmse_A",
        "local_contact_angle_fit_points",
    )
    _write_gzip_csv(output / "contour_points.csv.gz", local_rows, contour_fields)
    _write_gzip_csv(output / "local_arc_metrics.csv.gz", local_rows)
    event_fields = _fieldnames(events) if events else [
        "event_id",
        "arc_index",
        "start_step",
        "end_step",
        "transition_step",
        "dwell_time_ps",
        "jump_distance_A",
        "mechanism_class",
        "quality_status",
        "rejection_reason",
    ]
    _write_csv(output / "candidate_events.csv", events, event_fields)
    valid_count = len(local_by_step)
    valid_steps = sorted(local_by_step)
    first_valid_index = next(
        index for index, row in enumerate(frame_rows) if int(row["step"]) == valid_steps[0]
    )
    post_first_valid_frames = len(frame_rows) - first_valid_index
    invalid_reasons = Counter(
        str(row["contour_invalid_reason"])
        for row in frame_rows
        if not row["contour_valid"]
    )
    candidate_count = sum(event["quality_status"] == "candidate_stick_slip" for event in events)
    candidate_clusters = {
        int(event["event_cluster_id"])
        for event in events
        if event["quality_status"] == "candidate_stick_slip"
    }
    insufficient_count = sum(
        event["quality_status"] == "insufficient_repetition" for event in events
    )
    frame_interval_ps = expected_step * config.timestep_fs / 1000.0
    summary = {
        "status": "PASS",
        "kind": config.kind,
        "raw_frames": raw_frames,
        "unique_window_frames_before_drop": len(records),
        "analyzed_frames": len(frame_rows),
        "valid_contour_frames": valid_count,
        "invalid_contour_frames": len(frame_rows) - valid_count,
        "contour_valid_fraction": valid_count / len(frame_rows),
        "first_valid_contour_step": valid_steps[0],
        "last_valid_contour_step": valid_steps[-1],
        "contour_valid_fraction_after_first_valid": valid_count / post_first_valid_frames,
        "invalid_contour_reason_counts": dict(invalid_reasons),
        "first_step": int(frame_rows[0]["step"]),
        "last_step": int(frame_rows[-1]["step"]),
        "expected_step_interval": expected_step,
        "frame_interval_ps": frame_interval_ps,
        "surface_reference_mode": "dynamic_slab_translation",
        "arc_bins": config.arc_bins,
        "top_surface_ch3_sites": sites.ch3_count,
        "top_surface_sioh_sites": sites.sioh_count,
        "localization_noise_A": noise,
        "dwell_tolerance_A": dwell_tolerance,
        "jump_threshold_A": jump_threshold,
        "minimum_dwell_frames": config.minimum_dwell_frames,
        "minimum_resolved_dwell_ps": config.minimum_dwell_frames * frame_interval_ps,
        "post_confirm_frames": config.post_confirm_frames,
        "candidate_event_count": len(candidate_clusters),
        "candidate_arc_record_count": candidate_count,
        "insufficient_repetition_event_count": insufficient_count,
        "rejected_event_count": sum(event["quality_status"] == "rejected" for event in events),
        "time_resolution_assessment": (
            f"resolves only dwell >= {config.minimum_dwell_frames * frame_interval_ps:g} ps "
            f"with >= {config.post_confirm_frames * frame_interval_ps:g} ps confirmed post state; "
            f"sub-{frame_interval_ps:g} ps motion is unresolved"
        ),
        "scientific_classification": (
            "REPEATED_CANDIDATE_STICK_SLIP_NOT_CAUSAL_PROOF"
            if candidate_clusters
            else "NO_REPEATED_CANDIDATE_AT_AVAILABLE_TIME_RESOLUTION"
        ),
        "events_are_effective_kinetic_landscape_proxies_not_free_energy_barriers": True,
        "no_interpolation_or_cross_gap_smoothing": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "config": _serializable_config(config),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "max_frames": max_frames,
        "drop_first_frame": drop_first_frame,
        "restart_policy": "later trajectory replaces earlier duplicate timestep",
        "event_windows_may_not_cross_segment_or_missing_or_invalid_frame": True,
        "contour_definition": (
            "fixed-angle ray intersections with the convex hull of largest-cluster "
            "near-surface N2-pair centers or water oxygens"
        ),
        "contact_angle_definition": "local phase-envelope slope; phase-side angle",
        "thresholds_frozen_before_smoke": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    if make_plots:
        write_figures(
            frame_rows,
            local_by_step,
            events,
            snapshots,
            output,
            Path(font_path),
            config,
            expected_step,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--start-ns", type=float)
    parser.add_argument("--end-ns", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--drop-first-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        TpclConfig.load(args.config),
        args.output_dir,
        font_path=args.font_path,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        max_frames=args.max_frames,
        drop_first_frame=args.drop_first_frame,
        make_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
