"""Patch-resolved footprint geometry, water renewal, and H-bond rewiring.

This existing-trajectory analysis uses fixed substrate transverse patches.  It
keeps oxygen-identity renewal separate from H-bond changes among retained
oxygen nodes and reports only discrete persistence at the saved-frame cadence.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
    xy_minimum_image,
)
from molsimflow.postprocess.interfacial_water_hbond import (
    donor_points_to,
    water_surface_hbond_counts,
)
from molsimflow.postprocess.interfacial_water_orientation import (
    assign_hydrogen_neighbors,
)
from molsimflow.postprocess.nanobubble_flow_response import (
    _molecule_centers,
    _tracked_periodic_center,
)
from molsimflow.postprocess.nanobubble_local_flow import (
    _frame_arrays,
    _iter_sampled_frames,
    _iter_sampled_synchronized_frames,
    _periodic_nearest_distance,
    _range,
    _select_present_range,
    _select_range,
    _synchronized_trajectory_entries,
    _trajectory_entries,
)

FRAME_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "patch_id",
    "patch_index", "patch_transverse_lo_A", "patch_transverse_hi_A", "edge",
    "edge_coordinate_A", "contact_n2_in_patch", "observer_valid",
    "global_pbc_spanning_drive", "global_pbc_spanning_transverse",
    "water_oxygen_count", "intact_water_count", "intact_water_fraction",
    "water_water_hbond_count", "water_water_hbond_degree",
    "surface_water_hbond_count", "surface_water_hbond_per_intact_water",
    "mean_water_dipole_drive_cos", "mean_water_dipole_z_cos",
    "sum_local_waterO_raw_force_drive_eV_per_A",
    "mean_local_waterO_raw_force_drive_eV_per_A",
    "sum_exposed_site_raw_force_drive_eV_per_A",
    "exposed_site_count", "exposed_sioh_count", "exposed_ch3_count",
    "exposed_ch3_fraction", "mean_local_ch3_fraction",
    "mean_local_boundary_edge_fraction", "nearest_site_id", "nearest_site_type",
    "nearest_site_drive_distance_A", "global_film_water_oxygen_count",
)
INTERVAL_FIELDS = (
    "case_id", "branch_id", "direction", "start_step", "end_step",
    "start_time_ps", "end_time_ps", "interval_ps", "patch_id", "patch_index",
    "edge", "edge_displacement_A", "previous_water_oxygen_count",
    "current_water_oxygen_count", "retained_water_oxygen_count",
    "lost_water_oxygen_count", "gained_water_oxygen_count",
    "node_survival_fraction", "gross_membership_turnover",
    "previous_retained_hbond_edges", "current_retained_hbond_edges",
    "surviving_retained_hbond_edges", "retained_node_edge_survival_fraction",
    "broken_retained_hbond_edges", "formed_retained_hbond_edges",
    "hbond_break_rate_per_ps", "hbond_formation_rate_per_ps",
    "retained_dipole_count", "retained_dipole_mean_cosine",
    "retained_dipole_mean_rotation_deg", "retained_dipole_rotated_gt30_fraction",
    "local_waterO_raw_force_impulse_eV_ps_per_A",
    "surface_water_hbond_change", "observer_contiguous",
)
SITE_FIELDS = (
    "case_id", "branch_id", "direction", "patch_id", "patch_index", "edge",
    "site_id", "site_type", "local_ch3_fraction", "local_boundary_edge_fraction",
    "exposure_frames", "valid_patch_frames", "exposure_fraction",
)
RESIDENCE_FIELDS = (
    "case_id", "branch_id", "direction", "patch_id", "patch_index", "edge",
    "observed_exit_episodes", "observer_censored_episodes", "end_censored_episodes",
    "mean_observed_residence_ps", "median_observed_residence_ps",
    "p95_observed_residence_ps", "saved_frame_interval_ps",
)
HBOND_LIFETIME_FIELDS = (
    "case_id", "branch_id", "direction", "patch_id", "patch_index", "edge",
    "observed_break_episodes", "observer_censored_episodes", "end_censored_episodes",
    "mean_observed_lifetime_ps", "median_observed_lifetime_ps",
    "p95_observed_lifetime_ps", "saved_frame_interval_ps",
)


def _parse_data_atoms(path: Path) -> tuple[np.ndarray, dict[int, tuple[int, np.ndarray]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    bounds = np.zeros((3, 2), dtype=float)
    labels = {"xlo": 0, "ylo": 1, "zlo": 2}
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[2] in labels:
            bounds[labels[parts[2]]] = [float(parts[0]), float(parts[1])]
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith("Atoms"))
    except StopIteration as exc:
        raise ValueError(f"No Atoms section in {path}") from exc
    atoms: dict[int, tuple[int, np.ndarray]] = {}
    for line in lines[start + 1:]:
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts[0].isdigit():
            break
        atoms[int(parts[0])] = (
            int(parts[1]), np.asarray([float(value) for value in parts[2:5]], dtype=float)
        )
    if not atoms or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"Incomplete data file parse: {path}")
    return bounds, atoms


def surface_site_descriptors(
    model_path: Path,
    registry_path: Path,
    *,
    neighbor_cutoff_A: float,
    local_radius_A: float,
) -> dict[int, dict[str, object]]:
    from scipy.spatial import cKDTree

    bounds, atoms = _parse_data_atoms(model_path)
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    if not registry:
        raise ValueError("Empty surface-site registry")
    ids = np.asarray([int(row["atom_id"]) for row in registry], dtype=int)
    types = np.asarray([row["site_type"] for row in registry], dtype=str)
    if not set(types).issubset({"SiOH", "CH3"}):
        raise ValueError("Unsupported surface-site type")
    missing = [atom_id for atom_id in ids if atom_id not in atoms]
    if missing:
        raise ValueError(f"Surface registry IDs missing from model: {missing[:5]}")
    expected_types = {"SiOH": 2, "CH3": 7}
    for atom_id, site_type in zip(ids, types):
        if atoms[int(atom_id)][0] != expected_types[site_type]:
            raise ValueError(f"Site {atom_id} type disagrees with current model")
    xy = np.asarray([atoms[int(atom_id)][1][:2] for atom_id in ids], dtype=float)
    lengths = bounds[:2, 1] - bounds[:2, 0]
    shifted = (xy - bounds[:2, 0]) % lengths
    tree = cKDTree(shifted, boxsize=lengths)
    graph_edges = sorted(tree.query_pairs(neighbor_cutoff_A))
    output: dict[int, dict[str, object]] = {}
    for index, atom_id in enumerate(ids):
        local = np.asarray(tree.query_ball_point(shifted[index], local_radius_A), dtype=int)
        local_set = set(local.tolist())
        local_edges = [(a, b) for a, b in graph_edges if a in local_set and b in local_set]
        boundary = sum(types[a] != types[b] for a, b in local_edges)
        output[int(atom_id)] = {
            "site_id": int(atom_id), "site_type": str(types[index]),
            "local_ch3_fraction": float(np.mean(types[local] == "CH3")),
            "local_boundary_edge_fraction": (
                boundary / len(local_edges) if local_edges else 0.0
            ),
        }
    return output


def water_hbond_edges(
    oxygen_ids: np.ndarray,
    oxygen: np.ndarray,
    oh_vectors: Sequence[np.ndarray],
    bounds: np.ndarray,
    *,
    oo_cutoff_A: float,
    angle_cutoff_deg: float,
) -> set[tuple[int, int]]:
    from scipy.spatial import cKDTree

    if len(oxygen) < 2:
        return set()
    lengths = box_lengths(bounds)
    shifted = (oxygen - bounds[:, 0]) % lengths
    output: set[tuple[int, int]] = set()
    for first, second in cKDTree(shifted, boxsize=lengths).query_pairs(oo_cutoff_A):
        vector = minimum_image_vectors(oxygen[second] - oxygen[first], lengths)
        if donor_points_to(oh_vectors[first], vector, angle_cutoff_deg) or donor_points_to(
            oh_vectors[second], -vector, angle_cutoff_deg
        ):
            output.add(tuple(sorted((int(oxygen_ids[first]), int(oxygen_ids[second])))))
    return output


def _mean_orientation(
    oh_vectors: Sequence[np.ndarray], drive_index: int
) -> tuple[float, float]:
    drive, vertical = [], []
    for vectors in oh_vectors:
        if len(vectors) != 2:
            continue
        unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        bisector = np.sum(unit, axis=0)
        norm = float(np.linalg.norm(bisector))
        if norm <= 0.0:
            continue
        direction = bisector / norm
        drive.append(direction[drive_index]); vertical.append(direction[2])
    return (
        float(np.mean(drive)) if drive else math.nan,
        float(np.mean(vertical)) if vertical else math.nan,
    )


def _dipole_directions(
    oxygen_ids: np.ndarray, oh_vectors: Sequence[np.ndarray]
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for atom_id, vectors in zip(oxygen_ids, oh_vectors):
        if len(vectors) != 2:
            continue
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 0.0):
            continue
        bisector = np.sum(vectors / lengths[:, None], axis=0)
        norm = float(np.linalg.norm(bisector))
        if norm > 0.0:
            output[int(atom_id)] = bisector / norm
    return output


def _optional_frame_vectors(
    frame: object, fields: tuple[str, str, str]
) -> tuple[np.ndarray, np.ndarray] | None:
    atom_fields = tuple(getattr(frame, "atom_fields"))
    if not all(field in atom_fields for field in fields) or "id" not in atom_fields:
        return None
    id_column = atom_fields.index("id")
    columns = [atom_fields.index(field) for field in fields]
    rows = tuple(getattr(frame, "atom_rows"))
    atom_ids = np.asarray([int(row[id_column]) for row in rows], dtype=int)
    order = np.argsort(atom_ids)
    vectors = np.asarray(
        [[float(row[column]) for column in columns] for row in rows], dtype=float
    )[order]
    return atom_ids[order], vectors


def _geometry_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No geometry rows in {path}")
    return {int(row["step"]): row for row in rows}


def _site_registry(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["atom_id"]): row["site_type"] for row in rows}


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array[np.isfinite(array)])) if np.any(np.isfinite(array)) else math.nan


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    for key in (
        "timestep_fs", "sample_stride_steps", "cluster_cutoff_A", "contact_cutoff_A",
        "edge_quantile", "transverse_patch_count", "edge_water_halfwidth_A",
        "near_surface_height_A", "site_exposure_halfwidth_A", "oh_cutoff_A",
        "oo_cutoff_A", "angle_cutoff_deg", "site_neighbor_cutoff_A",
        "site_local_radius_A",
    ):
        if float(raw.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if not 0.5 < float(raw["edge_quantile"]) < 1.0:
        raise ValueError("edge_quantile must lie in (0.5, 1)")
    case = raw.get("case")
    if not isinstance(case, dict):
        raise TypeError("case must be an object")
    base = contract_path.parent
    model_path = resolve_path(case["surface_model_data"], base)
    registry_path = resolve_path(case["surface_site_registry"], base)
    geometry_path = resolve_path(case["geometry_frames"], base)
    synchronized = "synchronized_state_trajectories" in case
    if synchronized:
        synchronized_entries = _synchronized_trajectory_entries(
            case["synchronized_state_trajectories"], base
        )
        state_entries: list[dict[str, object]] = []
    else:
        synchronized_entries = []
        state_entries = _trajectory_entries(case["state_trajectories"], base)
    geometry = _geometry_rows(geometry_path)
    registry = _site_registry(registry_path)
    descriptors = surface_site_descriptors(
        model_path, registry_path,
        neighbor_cutoff_A=float(raw["site_neighbor_cutoff_A"]),
        local_radius_A=float(raw["site_local_radius_A"]),
    )
    surface_range = _range(case["surface_atom_range"], "surface_atom_range")
    nitrogen_range = _range(case["nitrogen_atom_range"], "nitrogen_atom_range")
    water_range = _range(case["water_atom_range"], "water_atom_range")
    relocated_ids = {int(value) for value in case.get("relocated_nitrogen_ids", [])}
    origin = int(case["time_origin_step"]); maximum = int(case["maximum_step"])
    stride = int(raw["sample_stride_steps"])
    direction = str(case["direction"]).lower()
    drive_index, transverse_index = (0, 1) if direction == "x" else (1, 0)
    patch_count = int(raw["transverse_patch_count"])
    oxygen_type = int(raw.get("oxygen_type", 2)); hydrogen_type = int(raw.get("hydrogen_type", 1))
    site_ids = np.asarray(sorted(registry), dtype=int)
    frame_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    site_exposure: dict[tuple[int, str, int], int] = defaultdict(int)
    valid_patch_frames: dict[tuple[int, str], int] = defaultdict(int)
    previous: dict[tuple[int, str], dict[str, object]] = {}
    active: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    residence_observed: dict[tuple[int, str], list[float]] = defaultdict(list)
    residence_observer_censored: dict[tuple[int, str], list[float]] = defaultdict(list)
    active_hbonds: dict[tuple[int, str], dict[tuple[int, int], float]] = defaultdict(dict)
    hbond_lifetime_observed: dict[tuple[int, str], list[float]] = defaultdict(list)
    hbond_lifetime_observer_censored: dict[
        tuple[int, str], list[float]
    ] = defaultdict(list)
    previous_center: np.ndarray | None = None
    previous_wrapped: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    origin_unwrapped: np.ndarray | None = None
    last_time = 0.0
    frame_count = 0
    if synchronized:
        states = _iter_sampled_synchronized_frames(
            synchronized_entries,
            time_origin_step=origin,
            maximum_step=maximum,
            stride_steps=stride,
        )
    else:
        states = _iter_sampled_frames(
            state_entries,
            time_origin_step=origin,
            maximum_step=maximum,
            stride_steps=stride,
        )
    for frame in states:
        if frame.timestep not in geometry:
            raise ValueError(f"Geometry is missing step {frame.timestep}")
        global_row = geometry[frame.timestep]
        atom_ids, atom_types, coordinates = _frame_arrays(frame)
        force_data = _optional_frame_vectors(
            frame, ("f_RAW[1]", "f_RAW[2]", "f_RAW[3]")
        )
        raw_forces: np.ndarray | None = None
        if force_data is not None:
            force_ids, raw_forces = force_data
            if not np.array_equal(force_ids, atom_ids):
                raise ValueError(f"step {frame.timestep}: raw-force atom IDs differ")
        if synchronized:
            surface_ids, _surface_types, surface = _select_present_range(
                atom_ids, atom_types, coordinates, surface_range,
                name="surface_atom_range",
            )
        else:
            surface_ids, _surface_types, surface = _select_range(
                atom_ids, atom_types, coordinates, surface_range
            )
        nitrogen_ids, _, nitrogen = _select_range(
            atom_ids, atom_types, coordinates, nitrogen_range
        )
        if synchronized:
            water_ids_all, water_types, water_all = _select_present_range(
                atom_ids, atom_types, coordinates, water_range,
                name="water_atom_range",
            )
        else:
            water_ids_all, water_types, water_all = _select_range(
                atom_ids, atom_types, coordinates, water_range
            )
        water_mask = water_types == oxygen_type
        water_ids = water_ids_all[water_mask]; water = water_all[water_mask]
        surface_raw_forces = (
            raw_forces[(atom_ids >= surface_range[0]) & (atom_ids <= surface_range[1])]
            if raw_forces is not None else None
        )
        water_raw_all = (
            raw_forces[(atom_ids >= water_range[0]) & (atom_ids <= water_range[1])]
            if raw_forces is not None else None
        )
        water_raw_forces = water_raw_all[water_mask] if water_raw_all is not None else None
        candidate_mask = atom_types == oxygen_type
        candidate_ids = atom_ids[candidate_mask]; candidate_oxygen = coordinates[candidate_mask]
        hydrogen = coordinates[atom_types == hydrogen_type]
        assigned = assign_hydrogen_neighbors(
            candidate_oxygen, hydrogen, frame.bounds, float(raw["oh_cutoff_A"])
        )
        candidate_lookup = {int(atom_id): index for index, atom_id in enumerate(candidate_ids)}
        water_candidate_indices = np.asarray([candidate_lookup[int(atom_id)] for atom_id in water_ids])
        water_assigned = [assigned[index] for index in water_candidate_indices]
        lengths = box_lengths(frame.bounds)
        water_vectors = [
            minimum_image_vectors(hydrogen[items] - water[index], lengths)
            for index, items in enumerate(water_assigned)
        ]
        if len(nitrogen_ids) % 2:
            raise ValueError("Nitrogen range does not contain complete N2 pairs")
        pairs = nitrogen_ids.reshape((-1, 2))
        centers = _molecule_centers(nitrogen, frame.bounds)
        relocated = np.asarray([
            bool(relocated_ids.intersection(pair.tolist())) for pair in pairs
        ], dtype=bool)
        candidates = np.flatnonzero(~relocated)
        components = connected_components(
            centers[candidates], frame.bounds, float(raw["cluster_cutoff_A"])
        )
        if not components:
            raise ValueError(f"No bubble component at step {frame.timestep}")
        members = candidates[components[0]]
        bubble = centers[members]
        center = _tracked_periodic_center(bubble, frame.bounds, previous_center)
        vectors = bubble - center
        vectors[:, :2] = xy_minimum_image(vectors, frame.bounds)[:, :2]
        nitrogen_distance = _periodic_nearest_distance(
            nitrogen, surface, frame.bounds
        ).reshape((-1, 2)).min(axis=1)
        contact_vectors = vectors[nitrogen_distance[members] <= float(raw["contact_cutoff_A"])]
        if previous_wrapped is None:
            unwrapped = center.copy()
            origin_unwrapped = center.copy()
        else:
            assert unwrapped is not None
            unwrapped = unwrapped + xy_minimum_image(center - previous_wrapped, frame.bounds)
            unwrapped[2] = center[2]
        assert unwrapped is not None and origin_unwrapped is not None
        previous_center = center.copy(); previous_wrapped = center.copy()
        if not math.isclose(
            float(unwrapped[drive_index] - origin_unwrapped[drive_index]),
            float(global_row["bubble_drive_unwrapped_A"]),
            abs_tol=1.0e-5,
        ):
            raise ValueError(f"Bubble tracking differs from geometry table at {frame.timestep}")
        transverse_lo = frame.bounds[transverse_index, 0]
        transverse_length = lengths[transverse_index]
        patch_width = transverse_length / patch_count
        contact_transverse = (
            center[transverse_index] + contact_vectors[:, transverse_index] - transverse_lo
        ) % transverse_length + transverse_lo
        contact_patch = np.minimum(
            ((contact_transverse - transverse_lo) / patch_width).astype(int), patch_count - 1
        )
        contact_drive = unwrapped[drive_index] + contact_vectors[:, drive_index]
        water_delta = water - center
        water_delta[:, :2] = xy_minimum_image(water_delta, frame.bounds)[:, :2]
        water_drive = unwrapped[drive_index] + water_delta[:, drive_index]
        water_transverse = (water[:, transverse_index] - transverse_lo) % transverse_length + transverse_lo
        water_patch = np.minimum(
            ((water_transverse - transverse_lo) / patch_width).astype(int), patch_count - 1
        )
        surface_top = float(np.quantile(surface[:, 2], float(raw.get("surface_top_quantile", 0.99))))
        water_near = (water[:, 2] >= surface_top) & (
            water[:, 2] <= surface_top + float(raw["near_surface_height_A"])
        )
        surface_lookup = {int(atom_id): index for index, atom_id in enumerate(surface_ids)}
        if any(int(atom_id) not in surface_lookup for atom_id in site_ids):
            raise ValueError("Site registry is not contained in surface range")
        dynamic_site = np.asarray([surface[surface_lookup[int(atom_id)]] for atom_id in site_ids])
        site_delta = dynamic_site - center
        site_delta[:, :2] = xy_minimum_image(site_delta, frame.bounds)[:, :2]
        site_drive = unwrapped[drive_index] + site_delta[:, drive_index]
        site_transverse = (
            dynamic_site[:, transverse_index] - transverse_lo
        ) % transverse_length + transverse_lo
        site_patch = np.minimum(
            ((site_transverse - transverse_lo) / patch_width).astype(int), patch_count - 1
        )
        frame_time = (frame.timestep - origin) * float(raw["timestep_fs"]) / 1000.0
        last_time = frame_time; present_keys: set[tuple[int, str]] = set()
        for patch in range(patch_count):
            selected_contact = contact_patch == patch
            contact_count = int(np.count_nonzero(selected_contact))
            drive_values = contact_drive[selected_contact]
            quantile = float(raw["edge_quantile"])
            edge_values = {
                "front": float(np.quantile(drive_values, quantile)) if contact_count else math.nan,
                "rear": float(np.quantile(drive_values, 1.0 - quantile)) if contact_count else math.nan,
            }
            for edge, edge_coordinate in edge_values.items():
                key = (patch, edge)
                patch_id = f"P{patch:02d}"
                valid = math.isfinite(edge_coordinate)
                if valid:
                    present_keys.add(key); valid_patch_frames[key] += 1
                    water_selected = (
                        (water_patch == patch) & water_near
                        & (np.abs(water_drive - edge_coordinate) <= float(raw["edge_water_halfwidth_A"]))
                    )
                    selected_indices = np.flatnonzero(water_selected)
                    node_ids = {int(value) for value in water_ids[selected_indices]}
                    intact_indices = np.asarray([
                        index for index in selected_indices if len(water_assigned[index]) == 2
                    ], dtype=int)
                    intact_ids = water_ids[intact_indices]
                    intact_coords = water[intact_indices]
                    intact_vectors = [water_vectors[index] for index in intact_indices]
                    dipoles = _dipole_directions(intact_ids, intact_vectors)
                    hbond_edges = water_hbond_edges(
                        intact_ids, intact_coords, intact_vectors, frame.bounds,
                        oo_cutoff_A=float(raw["oo_cutoff_A"]),
                        angle_cutoff_deg=float(raw["angle_cutoff_deg"]),
                    )
                    exposed = (site_patch == patch) & (
                        np.abs(site_drive - edge_coordinate) <= float(raw["site_exposure_halfwidth_A"])
                    )
                    exposed_indices = np.flatnonzero(exposed)
                    exposed_ids = site_ids[exposed_indices]
                    for atom_id in exposed_ids:
                        site_exposure[(patch, edge, int(atom_id))] += 1
                    sioh_indices = [
                        candidate_lookup[int(site_ids[index])]
                        for index in exposed_indices
                        if registry[int(site_ids[index])] == "SiOH"
                        and len(assigned[candidate_lookup[int(site_ids[index])]]) == 1
                    ]
                    surface_oxygen = candidate_oxygen[sioh_indices]
                    surface_vectors = [
                        minimum_image_vectors(
                            hydrogen[assigned[index]] - candidate_oxygen[index], lengths
                        )
                        for index in sioh_indices
                    ]
                    water_donor, surface_donor = water_surface_hbond_counts(
                        intact_coords, intact_vectors, surface_oxygen, surface_vectors,
                        frame.bounds, oo_cutoff=float(raw["oo_cutoff_A"]),
                        angle_cutoff_deg=float(raw["angle_cutoff_deg"]),
                    )
                    surface_hbonds = water_donor + surface_donor
                    orientation_drive, orientation_z = _mean_orientation(
                        intact_vectors, drive_index
                    )
                    water_force_sum = (
                        float(np.sum(water_raw_forces[selected_indices, drive_index]))
                        if water_raw_forces is not None else math.nan
                    )
                    water_force_mean = (
                        water_force_sum / len(selected_indices)
                        if selected_indices.size and math.isfinite(water_force_sum) else math.nan
                    )
                    exposed_surface_indices = np.asarray(
                        [surface_lookup[int(atom_id)] for atom_id in exposed_ids], dtype=int
                    )
                    exposed_site_force_sum = (
                        float(np.sum(surface_raw_forces[exposed_surface_indices, drive_index]))
                        if surface_raw_forces is not None and exposed_surface_indices.size
                        else math.nan
                    )
                    exposed_descriptors = [descriptors[int(atom_id)] for atom_id in exposed_ids]
                    nearest_site_id: int | str = ""
                    nearest_site_type = ""
                    nearest_distance = math.nan
                    if len(exposed_indices):
                        distances = np.abs(site_drive[exposed_indices] - edge_coordinate)
                        nearest_local = int(np.argmin(distances))
                        nearest_site_id = int(exposed_ids[nearest_local])
                        nearest_site_type = registry[int(nearest_site_id)]
                        nearest_distance = float(distances[nearest_local])
                    row = {
                        "case_id": case["case_id"], "branch_id": case["branch_id"],
                        "direction": direction, "step": frame.timestep, "time_ps": frame_time,
                        "patch_id": patch_id, "patch_index": patch,
                        "patch_transverse_lo_A": transverse_lo + patch * patch_width,
                        "patch_transverse_hi_A": transverse_lo + (patch + 1) * patch_width,
                        "edge": edge, "edge_coordinate_A": edge_coordinate,
                        "contact_n2_in_patch": contact_count, "observer_valid": 1,
                        "global_pbc_spanning_drive": global_row["pbc_spanning_drive"],
                        "global_pbc_spanning_transverse": global_row["pbc_spanning_transverse"],
                        "water_oxygen_count": len(node_ids), "intact_water_count": len(intact_ids),
                        "intact_water_fraction": len(intact_ids) / len(node_ids) if node_ids else math.nan,
                        "water_water_hbond_count": len(hbond_edges),
                        "water_water_hbond_degree": 2.0 * len(hbond_edges) / len(intact_ids) if len(intact_ids) else math.nan,
                        "surface_water_hbond_count": surface_hbonds,
                        "surface_water_hbond_per_intact_water": surface_hbonds / len(intact_ids) if len(intact_ids) else math.nan,
                        "mean_water_dipole_drive_cos": orientation_drive,
                        "mean_water_dipole_z_cos": orientation_z,
                        "sum_local_waterO_raw_force_drive_eV_per_A": water_force_sum,
                        "mean_local_waterO_raw_force_drive_eV_per_A": water_force_mean,
                        "sum_exposed_site_raw_force_drive_eV_per_A": exposed_site_force_sum,
                        "exposed_site_count": len(exposed_ids),
                        "exposed_sioh_count": sum(registry[int(value)] == "SiOH" for value in exposed_ids),
                        "exposed_ch3_count": sum(registry[int(value)] == "CH3" for value in exposed_ids),
                        "exposed_ch3_fraction": (
                            np.mean([registry[int(value)] == "CH3" for value in exposed_ids])
                            if len(exposed_ids) else math.nan
                        ),
                        "mean_local_ch3_fraction": _finite_mean([
                            float(value["local_ch3_fraction"]) for value in exposed_descriptors
                        ]),
                        "mean_local_boundary_edge_fraction": _finite_mean([
                            float(value["local_boundary_edge_fraction"]) for value in exposed_descriptors
                        ]),
                        "nearest_site_id": nearest_site_id,
                        "nearest_site_type": nearest_site_type,
                        "nearest_site_drive_distance_A": nearest_distance,
                        "global_film_water_oxygen_count": global_row["film_water_oxygen_count"],
                    }
                    old = previous.get(key)
                    contiguous = False
                    if old is not None:
                        contiguous = int(frame.timestep) - int(old["step"]) == stride
                        if contiguous:
                            old_nodes = set(old["nodes"]); retained = old_nodes & node_ids
                            old_edges = {
                                value for value in set(old["hbond_edges"])
                                if value[0] in retained and value[1] in retained
                            }
                            current_edges = {
                                value for value in hbond_edges
                                if value[0] in retained and value[1] in retained
                            }
                            broken_edges = old_edges - current_edges
                            formed_edges = current_edges - old_edges
                            retained_dipoles = sorted(
                                set(old["dipoles"]).intersection(dipoles)
                            )
                            dipole_cosines = np.asarray([
                                np.clip(
                                    float(np.dot(old["dipoles"][atom_id], dipoles[atom_id])),
                                    -1.0, 1.0,
                                )
                                for atom_id in retained_dipoles
                            ], dtype=float)
                            dipole_angles = np.degrees(np.arccos(dipole_cosines))
                            interval_ps = frame_time - float(old["time_ps"])
                            force_impulse = (
                                0.5 * (float(old["water_force_sum"]) + water_force_sum)
                                * interval_ps
                                if math.isfinite(float(old["water_force_sum"]))
                                and math.isfinite(water_force_sum) else math.nan
                            )
                            denominator = len(old_nodes) + len(node_ids)
                            interval_rows.append({
                                "case_id": case["case_id"], "branch_id": case["branch_id"],
                                "direction": direction, "start_step": old["step"],
                                "end_step": frame.timestep, "start_time_ps": old["time_ps"],
                                "end_time_ps": frame_time,
                                "interval_ps": interval_ps,
                                "patch_id": patch_id, "patch_index": patch, "edge": edge,
                                "edge_displacement_A": edge_coordinate - float(old["edge_coordinate_A"]),
                                "previous_water_oxygen_count": len(old_nodes),
                                "current_water_oxygen_count": len(node_ids),
                                "retained_water_oxygen_count": len(retained),
                                "lost_water_oxygen_count": len(old_nodes - node_ids),
                                "gained_water_oxygen_count": len(node_ids - old_nodes),
                                "node_survival_fraction": len(retained) / len(old_nodes) if old_nodes else math.nan,
                                "gross_membership_turnover": (
                                    len(old_nodes.symmetric_difference(node_ids)) / denominator
                                    if denominator else math.nan
                                ),
                                "previous_retained_hbond_edges": len(old_edges),
                                "current_retained_hbond_edges": len(current_edges),
                                "surviving_retained_hbond_edges": len(old_edges & current_edges),
                                "retained_node_edge_survival_fraction": (
                                    len(old_edges & current_edges) / len(old_edges)
                                    if old_edges else math.nan
                                ),
                                "broken_retained_hbond_edges": len(broken_edges),
                                "formed_retained_hbond_edges": len(formed_edges),
                                "hbond_break_rate_per_ps": len(broken_edges) / interval_ps,
                                "hbond_formation_rate_per_ps": len(formed_edges) / interval_ps,
                                "retained_dipole_count": len(retained_dipoles),
                                "retained_dipole_mean_cosine": (
                                    float(np.mean(dipole_cosines)) if dipole_cosines.size else math.nan
                                ),
                                "retained_dipole_mean_rotation_deg": (
                                    float(np.mean(dipole_angles)) if dipole_angles.size else math.nan
                                ),
                                "retained_dipole_rotated_gt30_fraction": (
                                    float(np.mean(dipole_angles > 30.0))
                                    if dipole_angles.size else math.nan
                                ),
                                "local_waterO_raw_force_impulse_eV_ps_per_A": force_impulse,
                                "surface_water_hbond_change": surface_hbonds - int(old["surface_hbonds"]),
                                "observer_contiguous": 1,
                            })
                    active_edges = active_hbonds[key]
                    if old is None or not contiguous:
                        for start in active_edges.values():
                            hbond_lifetime_observer_censored[key].append(frame_time - start)
                        active_edges.clear()
                    else:
                        for hbond, start in list(active_edges.items()):
                            if hbond in hbond_edges:
                                continue
                            duration = frame_time - start
                            if hbond[0] in node_ids and hbond[1] in node_ids:
                                hbond_lifetime_observed[key].append(duration)
                            else:
                                hbond_lifetime_observer_censored[key].append(duration)
                            active_edges.pop(hbond)
                    for hbond in hbond_edges:
                        active_edges.setdefault(hbond, frame_time)
                    previous[key] = {
                        "step": frame.timestep, "time_ps": frame_time,
                        "edge_coordinate_A": edge_coordinate, "nodes": node_ids,
                        "hbond_edges": hbond_edges, "surface_hbonds": surface_hbonds,
                        "dipoles": dipoles, "water_force_sum": water_force_sum,
                    }
                    active_nodes = active[key]
                    for atom_id in list(active_nodes):
                        if atom_id not in node_ids:
                            residence_observed[key].append(frame_time - active_nodes.pop(atom_id))
                    for atom_id in node_ids:
                        active_nodes.setdefault(atom_id, frame_time)
                else:
                    row = {
                        field: math.nan for field in FRAME_FIELDS
                    }
                    row.update({
                        "case_id": case["case_id"], "branch_id": case["branch_id"],
                        "direction": direction, "step": frame.timestep, "time_ps": frame_time,
                        "patch_id": patch_id, "patch_index": patch,
                        "patch_transverse_lo_A": transverse_lo + patch * patch_width,
                        "patch_transverse_hi_A": transverse_lo + (patch + 1) * patch_width,
                        "edge": edge, "edge_coordinate_A": math.nan,
                        "contact_n2_in_patch": 0, "observer_valid": 0,
                        "global_pbc_spanning_drive": global_row["pbc_spanning_drive"],
                        "global_pbc_spanning_transverse": global_row["pbc_spanning_transverse"],
                        "water_oxygen_count": 0, "intact_water_count": 0,
                        "exposed_site_count": 0, "exposed_sioh_count": 0,
                        "exposed_ch3_count": 0,
                        "global_film_water_oxygen_count": global_row["film_water_oxygen_count"],
                    })
                frame_rows.append(row)
        for key in set(previous).difference(present_keys):
            previous.pop(key, None)
            active_nodes = active[key]
            for atom_id, start in active_nodes.items():
                residence_observer_censored[key].append(frame_time - start)
            active_nodes.clear()
            for start in active_hbonds[key].values():
                hbond_lifetime_observer_censored[key].append(frame_time - start)
            active_hbonds[key].clear()
        frame_count += 1
    if set(geometry).difference({int(row["step"]) for row in frame_rows}):
        raise ValueError("Not all geometry steps were consumed")
    residence_rows = []
    hbond_lifetime_rows = []
    for patch in range(patch_count):
        for edge in ("front", "rear"):
            key = (patch, edge)
            observed = residence_observed[key]
            end_censored = [last_time - start for start in active[key].values()]
            residence_rows.append({
                "case_id": case["case_id"], "branch_id": case["branch_id"],
                "direction": direction, "patch_id": f"P{patch:02d}",
                "patch_index": patch, "edge": edge,
                "observed_exit_episodes": len(observed),
                "observer_censored_episodes": len(residence_observer_censored[key]),
                "end_censored_episodes": len(end_censored),
                "mean_observed_residence_ps": float(np.mean(observed)) if observed else math.nan,
                "median_observed_residence_ps": float(np.median(observed)) if observed else math.nan,
                "p95_observed_residence_ps": float(np.quantile(observed, 0.95)) if observed else math.nan,
                "saved_frame_interval_ps": stride * float(raw["timestep_fs"]) / 1000.0,
            })
            hbond_observed = hbond_lifetime_observed[key]
            hbond_end_censored = [last_time - start for start in active_hbonds[key].values()]
            hbond_lifetime_rows.append({
                "case_id": case["case_id"], "branch_id": case["branch_id"],
                "direction": direction, "patch_id": f"P{patch:02d}",
                "patch_index": patch, "edge": edge,
                "observed_break_episodes": len(hbond_observed),
                "observer_censored_episodes": len(
                    hbond_lifetime_observer_censored[key]
                ),
                "end_censored_episodes": len(hbond_end_censored),
                "mean_observed_lifetime_ps": (
                    float(np.mean(hbond_observed)) if hbond_observed else math.nan
                ),
                "median_observed_lifetime_ps": (
                    float(np.median(hbond_observed)) if hbond_observed else math.nan
                ),
                "p95_observed_lifetime_ps": (
                    float(np.quantile(hbond_observed, 0.95)) if hbond_observed else math.nan
                ),
                "saved_frame_interval_ps": stride * float(raw["timestep_fs"]) / 1000.0,
            })
    site_rows = []
    for (patch, edge, atom_id), exposure in sorted(site_exposure.items()):
        descriptor = descriptors[atom_id]
        valid_frames = valid_patch_frames[(patch, edge)]
        site_rows.append({
            "case_id": case["case_id"], "branch_id": case["branch_id"],
            "direction": direction, "patch_id": f"P{patch:02d}",
            "patch_index": patch, "edge": edge, "site_id": atom_id,
            "site_type": descriptor["site_type"],
            "local_ch3_fraction": descriptor["local_ch3_fraction"],
            "local_boundary_edge_fraction": descriptor["local_boundary_edge_fraction"],
            "exposure_frames": exposure, "valid_patch_frames": valid_frames,
            "exposure_fraction": exposure / valid_frames if valid_frames else math.nan,
        })
    output.mkdir(parents=True)
    write_tsv(output / "patch_edge_frames.tsv", frame_rows, FRAME_FIELDS)
    write_tsv(output / "patch_edge_intervals.tsv", interval_rows, INTERVAL_FIELDS)
    write_tsv(output / "site_exposure.tsv", site_rows, SITE_FIELDS)
    write_tsv(output / "residence_summary.tsv", residence_rows, RESIDENCE_FIELDS)
    write_tsv(
        output / "hbond_lifetime_summary.tsv",
        hbond_lifetime_rows,
        HBOND_LIFETIME_FIELDS,
    )
    inputs = [contract_path, model_path, registry_path, geometry_path]
    if synchronized:
        inputs.extend(
            Path(path) for entry in synchronized_entries for path in entry["paths"]
        )
    else:
        inputs.extend(Path(entry["path"]) for entry in state_entries)
    write_tsv(
        output / "input_manifest.tsv",
        [{"path": str(path), "size_bytes": path.stat().st_size,
          "sha256": sha256(path) if bool(raw.get("hash_inputs", True)) else "NOT_COMPUTED"}
         for path in inputs],
        ("path", "size_bytes", "sha256"),
    )
    summary = {
        "status": "PASS", "case_id": case["case_id"],
        "branch_id": case["branch_id"], "direction": direction,
        "frames": frame_count, "patches": patch_count,
        "patch_edge_frame_rows": len(frame_rows),
        "patch_edge_interval_rows": len(interval_rows),
        "site_exposure_rows": len(site_rows),
        "saved_frame_interval_ps": stride * float(raw["timestep_fs"]) / 1000.0,
        "spatially_filtered_synchronized_streams": synchronized,
        "retained_water_dipole_rotation_resolved": True,
        "retained_node_hbond_break_form_resolved": True,
        "local_raw_force_impulse_resolved": force_data is not None,
        "hbond_lifetime_resolved": True,
        "claim_boundary": (
            "Identity renewal, retained-node H-bond rewiring, and residence are "
            "discrete saved-frame observables in moving edge windows. RAW forces are "
            "many-body conservative atom forces, not water-substrate pair forces."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Patch-resolved edge water\n\n"
        f"Status: `PASS`; `{frame_count}` coordinate frames and `{patch_count}` fixed "
        "substrate transverse patches. Oxygen identity renewal is separated from "
        "H-bond rewiring among retained oxygen nodes. Residence is discrete at the "
        "saved-frame cadence; observer gaps and terminal censoring are counted separately.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
