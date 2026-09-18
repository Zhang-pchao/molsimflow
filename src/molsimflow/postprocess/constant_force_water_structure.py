"""Morphology-aware water structure for constant-force trajectories.

The contract supports water assemblies split into oxygen-connectivity islands
and laterally spread films split into height layers.  The analysis reports
water and surface hydrogen bonds, one-frame edge turnover, water order, and
region residence.  All results are single-trajectory diagnostics.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    box_lengths,
    iter_lammps_dump_records,
    minimum_image_vectors,
)
from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)
from molsimflow.postprocess.interfacial_water_orientation import assign_hydrogen_neighbors
from molsimflow.postprocess.local_water_order import local_structure_index, tetrahedral_order


FRAME_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "region",
    "water_count", "water_fraction", "physical_component_count",
    "physical_largest_component_fraction", "mean_q_tet", "mean_lsi_A2",
    "mean_oo_coordination", "h_coordination_defect_fraction",
    "water_water_hbond_edges", "water_surface_hbond_edges",
    "water_donor_surface_edges", "surface_donor_water_edges",
    "hbond_component_count", "hbond_largest_component_fraction",
    "water_water_edge_jaccard", "water_water_edge_turnover",
    "water_surface_edge_jaccard", "water_surface_edge_turnover",
)
EXCHANGE_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps",
    "from_region", "to_region", "water_count", "rate_per_ps",
)
RESIDENCE_FIELDS = (
    "case_id", "branch_id", "direction", "region", "episodes",
    "mean_residence_ps", "median_residence_ps", "p95_residence_ps",
    "right_censored_episodes",
)
LIFETIME_FIELDS = (
    "case_id", "branch_id", "direction", "edge_type", "episodes",
    "mean_persistence_ps", "median_persistence_ps", "p95_persistence_ps",
    "right_censored_episodes", "sampling_interval_ps",
)


@dataclass(frozen=True)
class ChemistryFrame:
    """Identity-sorted surface and water chemistry for one frame."""

    step: int
    bounds: np.ndarray
    water_ids: np.ndarray
    water: np.ndarray
    surface_oxygen_ids: np.ndarray
    surface_oxygen: np.ndarray
    water_oh: tuple[np.ndarray, ...]
    surface_oh: tuple[np.ndarray, ...]
    water_h_coordination: np.ndarray


def _range(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [start, end]")
    start, end = (int(item) for item in value)
    if start < 1 or end < start:
        raise ValueError(f"{name} must satisfy 1 <= start <= end")
    return start, end


def _validate_contract(raw: Mapping[str, object]) -> None:
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    for key in (
        "timestep_fs", "oh_cutoff_A", "oo_cutoff_A", "hbond_angle_deg",
        "lsi_cutoff_A", "cluster_cutoff_A",
    ):
        if float(raw.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    _range(raw.get("surface_atom_range"), "surface_atom_range")
    _range(raw.get("water_atom_range"), "water_atom_range")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    identities = []
    for entry in cases:
        if not isinstance(entry, dict):
            raise ValueError("Each case entry must be an object")
        for key in ("case_id", "branch_id", "direction", "region_mode", "trajectories"):
            if key not in entry:
                raise ValueError(f"Case entry is missing {key}")
        direction = str(entry["direction"]).lower()
        if direction not in {"none", "x", "y"}:
            raise ValueError("direction must be none, x, or y")
        mode = str(entry["region_mode"]).lower()
        if mode not in {"islands", "layers"}:
            raise ValueError("region_mode must be islands or layers")
        if not isinstance(entry["trajectories"], list) or not entry["trajectories"]:
            raise ValueError("trajectories must be a non-empty list")
        if mode == "layers":
            if "surface_z_A" not in entry:
                raise ValueError("layer entries require surface_z_A")
            edges = np.asarray(entry.get("z_edges_A", []), dtype=float)
            if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
                raise ValueError("z_edges_A must be strictly increasing")
        identities.append((str(entry["case_id"]), str(entry["branch_id"])))
    if len(identities) != len(set(identities)):
        raise ValueError("case_id/branch_id pairs must be unique")


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


def select_chemistry_frame(
    frame: LammpsDumpFrame,
    surface_range: tuple[int, int],
    water_range: tuple[int, int],
    *,
    oxygen_type: int,
    hydrogen_type: int,
    oh_cutoff_A: float,
) -> ChemistryFrame:
    """Select declared water and surface atoms and assign O-H neighbors."""

    fields = frame.atom_fields
    if "id" not in fields or "type" not in fields:
        raise ValueError(f"step {frame.timestep}: missing id/type columns")
    id_column, type_column = fields.index("id"), fields.index("type")
    coordinate_columns = _coordinate_columns(fields)
    lengths = box_lengths(frame.bounds)
    candidates: list[tuple[int, np.ndarray, bool, bool]] = []
    hydrogen: list[tuple[int, np.ndarray]] = []
    surface_seen: set[int] = set()
    water_seen: set[int] = set()
    for row in frame.atom_rows:
        atom_id, atom_type = int(row[id_column]), int(row[type_column])
        xyz = np.asarray([float(row[index]) for index, _ in coordinate_columns])
        for dimension, (_, scaled) in enumerate(coordinate_columns):
            if scaled:
                xyz[dimension] = frame.bounds[dimension, 0] + xyz[dimension] * lengths[dimension]
        in_surface = surface_range[0] <= atom_id <= surface_range[1]
        in_water = water_range[0] <= atom_id <= water_range[1]
        if in_surface:
            surface_seen.add(atom_id)
        if in_water:
            water_seen.add(atom_id)
        if atom_type == oxygen_type and (in_surface or in_water):
            candidates.append((atom_id, xyz, in_surface, in_water))
        elif atom_type == hydrogen_type and (in_surface or in_water):
            hydrogen.append((atom_id, xyz))
    expected_surface = surface_range[1] - surface_range[0] + 1
    expected_water = water_range[1] - water_range[0] + 1
    if len(surface_seen) != expected_surface or len(water_seen) != expected_water:
        raise ValueError(f"step {frame.timestep}: incomplete declared atom ranges")
    candidates.sort(key=lambda item: item[0])
    hydrogen.sort(key=lambda item: item[0])
    oxygen = np.asarray([item[1] for item in candidates], dtype=float)
    hydrogen_coordinates = np.asarray([item[1] for item in hydrogen], dtype=float)
    if not len(oxygen) or not len(hydrogen_coordinates):
        raise ValueError(f"step {frame.timestep}: empty oxygen or hydrogen selection")
    assignments = assign_hydrogen_neighbors(
        oxygen, hydrogen_coordinates, frame.bounds, oh_cutoff_A
    )
    water_indices = [index for index, item in enumerate(candidates) if item[3]]
    surface_indices = [index for index, item in enumerate(candidates) if item[2]]

    def vectors(index: int) -> np.ndarray:
        return minimum_image_vectors(
            hydrogen_coordinates[assignments[index]] - oxygen[index], lengths
        )

    return ChemistryFrame(
        step=frame.timestep,
        bounds=frame.bounds,
        water_ids=np.asarray([candidates[index][0] for index in water_indices], dtype=int),
        water=oxygen[water_indices],
        surface_oxygen_ids=np.asarray(
            [candidates[index][0] for index in surface_indices], dtype=int
        ),
        surface_oxygen=oxygen[surface_indices],
        water_oh=tuple(vectors(index) for index in water_indices),
        surface_oh=tuple(vectors(index) for index in surface_indices),
        water_h_coordination=np.asarray(
            [len(assignments[index]) for index in water_indices], dtype=int
        ),
    )


def iter_chemistry_frames(
    paths: Sequence[Path],
    surface_range: tuple[int, int],
    water_range: tuple[int, int],
    *,
    oxygen_type: int,
    hydrogen_type: int,
    oh_cutoff_A: float,
) -> Iterator[ChemistryFrame]:
    """Stream restart segments and replace their shared endpoint with the later frame."""

    previous_step: int | None = None
    previous_ids: np.ndarray | None = None
    for segment_index, path in enumerate(paths):
        frames = 0
        for raw in iter_lammps_dump_records(path):
            frames += 1
            if previous_step is not None and raw.timestep == previous_step:
                continue
            if previous_step is not None and raw.timestep < previous_step:
                raise ValueError(f"Non-increasing timestep {raw.timestep} in {path}")
            selected = select_chemistry_frame(
                raw, surface_range, water_range, oxygen_type=oxygen_type,
                hydrogen_type=hydrogen_type, oh_cutoff_A=oh_cutoff_A,
            )
            if previous_ids is None:
                previous_ids = selected.water_ids.copy()
            elif not np.array_equal(previous_ids, selected.water_ids):
                raise ValueError(f"Water oxygen identity changed at step {raw.timestep}")
            previous_step = raw.timestep
            yield selected
        if frames == 0:
            raise ValueError(f"No complete frames in segment {segment_index}: {path}")


def _donates(vectors: np.ndarray, donor_acceptor: np.ndarray, angle_deg: float) -> bool:
    distance = float(np.linalg.norm(donor_acceptor))
    if distance <= 0.0 or not len(vectors):
        return False
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0.0
    if not np.any(valid):
        return False
    cosine = (vectors[valid] / norms[valid, None]) @ (donor_acceptor / distance)
    return bool(np.any(cosine >= math.cos(math.radians(angle_deg))))


def water_hbond_edges(
    frame: ChemistryFrame,
    *,
    oo_cutoff_A: float,
    angle_deg: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return water-water, water-donor-surface, and surface-donor-water edges."""

    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    shifted_water = (frame.water - frame.bounds[:, 0]) % lengths
    water_tree = cKDTree(shifted_water, boxsize=lengths)
    water_edges: set[tuple[int, int]] = set()
    for left, right in water_tree.query_pairs(oo_cutoff_A):
        vector = minimum_image_vectors(frame.water[right] - frame.water[left], lengths)
        if _donates(frame.water_oh[left], vector, angle_deg) or _donates(
            frame.water_oh[right], -vector, angle_deg
        ):
            water_edges.add(
                tuple(sorted((int(frame.water_ids[left]), int(frame.water_ids[right]))))
            )

    shifted_surface = (frame.surface_oxygen - frame.bounds[:, 0]) % lengths
    surface_tree = cKDTree(shifted_surface, boxsize=lengths)
    water_donor: set[tuple[int, int]] = set()
    surface_donor: set[tuple[int, int]] = set()
    neighborhoods = surface_tree.query_ball_point(shifted_water, oo_cutoff_A)
    for water_index, surface_neighbors in enumerate(neighborhoods):
        for surface_index in surface_neighbors:
            vector = minimum_image_vectors(
                frame.surface_oxygen[surface_index] - frame.water[water_index], lengths
            )
            edge = (int(frame.water_ids[water_index]), int(frame.surface_oxygen_ids[surface_index]))
            if _donates(frame.water_oh[water_index], vector, angle_deg):
                water_donor.add(edge)
            if _donates(frame.surface_oh[surface_index], -vector, angle_deg):
                surface_donor.add(edge)
    return water_edges, water_donor, surface_donor


def edge_similarity(
    previous: set[tuple[int, int]] | None,
    current: set[tuple[int, int]],
) -> tuple[float, float]:
    """Return Jaccard similarity and symmetric edge turnover."""

    if previous is None:
        return math.nan, math.nan
    union = previous | current
    if not union:
        return math.nan, math.nan
    jaccard = len(previous & current) / len(union)
    denominator = len(previous) + len(current)
    turnover = len(previous ^ current) / denominator if denominator else math.nan
    return jaccard, turnover


def region_labels(
    frame: ChemistryFrame,
    entry: Mapping[str, object],
    cluster_cutoff_A: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], int, float]:
    """Classify each water oxygen by island membership or height layer."""

    mode = str(entry["region_mode"]).lower()
    count = len(frame.water_ids)
    if mode == "islands":
        components = connected_components(frame.water, frame.bounds, cluster_cutoff_A)
        labels = np.full(count, "satellites", dtype=object)
        labels[components[0]] = "largest_island"
        regions = {
            "all": np.arange(count, dtype=int),
            "largest_island": np.asarray(components[0], dtype=int),
            "satellites": np.asarray(
                [index for component in components[1:] for index in component], dtype=int
            ),
        }
        return labels, regions, len(components), len(components[0]) / count
    edges = np.asarray(entry["z_edges_A"], dtype=float)
    relative_z = frame.water[:, 2] - float(entry["surface_z_A"])
    indices = np.digitize(relative_z, edges, right=False) - 1
    indices[(relative_z < edges[0]) | (relative_z >= edges[-1])] = -1
    labels = np.asarray(
        ["outside" if index < 0 else f"layer_{index}" for index in indices], dtype=object
    )
    regions = {"all": np.arange(count, dtype=int)}
    for layer in range(len(edges) - 1):
        regions[f"layer_{layer}"] = np.flatnonzero(indices == layer)
    regions["outside"] = np.flatnonzero(indices < 0)
    components = connected_components(frame.water, frame.bounds, cluster_cutoff_A)
    return labels, regions, len(components), len(components[0]) / count


def water_order_metrics(
    frame: ChemistryFrame,
    *,
    oo_cutoff_A: float,
    lsi_cutoff_A: float,
    lsi_neighbor_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q_tet, LSI, and O-O coordination for every water oxygen."""

    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    tree = cKDTree((frame.water - frame.bounds[:, 0]) % lengths, boxsize=lengths)
    k = min(lsi_neighbor_cap + 2, len(frame.water))
    distances, neighbors = tree.query(
        (frame.water - frame.bounds[:, 0]) % lengths, k=k
    )
    if len(frame.water) == 1:
        distances = distances.reshape((1, -1))
        neighbors = neighbors.reshape((1, -1))
    qtet = np.full(len(frame.water), np.nan)
    lsi = np.full(len(frame.water), np.nan)
    coordination = np.zeros(len(frame.water), dtype=int)
    for oxygen_index in range(len(frame.water)):
        ordered = sorted(
            (float(distance), int(neighbor))
            for distance, neighbor in zip(distances[oxygen_index], neighbors[oxygen_index])
            if int(neighbor) != oxygen_index and math.isfinite(float(distance))
        )
        ordered_distances = np.asarray([item[0] for item in ordered])
        coordination[oxygen_index] = int(np.count_nonzero(ordered_distances <= oo_cutoff_A))
        if coordination[oxygen_index] >= 4:
            nearest = np.asarray([item[1] for item in ordered[:4]], dtype=int)
            vectors = minimum_image_vectors(
                frame.water[nearest] - frame.water[oxygen_index], lengths
            )
            qtet[oxygen_index] = tetrahedral_order(vectors)
        lsi[oxygen_index] = local_structure_index(ordered_distances, lsi_cutoff_A)[0]
    return qtet, lsi, coordination


def hbond_component_metrics(
    selected_ids: set[int], edges: set[tuple[int, int]]
) -> tuple[int, float]:
    """Return component count and largest fraction in an induced H-bond graph."""

    if not selected_ids:
        return 0, math.nan
    parent = {atom_id: atom_id for atom_id in selected_ids}

    def find(atom_id: int) -> int:
        while parent[atom_id] != atom_id:
            parent[atom_id] = parent[parent[atom_id]]
            atom_id = parent[atom_id]
        return atom_id

    for left, right in edges:
        if left not in parent or right not in parent:
            continue
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    sizes: dict[int, int] = defaultdict(int)
    for atom_id in selected_ids:
        sizes[find(atom_id)] += 1
    return len(sizes), max(sizes.values()) / len(selected_ids)


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _record_edge_episodes(
    active: dict[tuple[int, int], tuple[float, float]],
    current: set[tuple[int, int]],
    time_ps: float,
    interval_ps: float,
    completed: list[tuple[float, bool]],
) -> None:
    for edge in set(active).difference(current):
        start, last = active.pop(edge)
        completed.append((last - start + interval_ps, False))
    for edge in current:
        if edge in active:
            active[edge] = (active[edge][0], time_ps)
        else:
            active[edge] = (time_ps, time_ps)


def _summary(values: Sequence[tuple[float, bool]]) -> tuple[int, float, float, float, int]:
    durations = np.asarray([value for value, _ in values], dtype=float)
    if not len(durations):
        return 0, math.nan, math.nan, math.nan, 0
    return (
        len(values), float(np.mean(durations)), float(np.median(durations)),
        float(np.quantile(durations, 0.95)), sum(censored for _, censored in values),
    )


def _plot(frame_rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = sorted({(str(row["case_id"]), str(row["branch_id"])) for row in frame_rows})
    figure, axes = plt.subplots(4, len(keys), figsize=(4.0 * len(keys), 9.2), squeeze=False)
    for column, key in enumerate(keys):
        rows = [
            row for row in frame_rows
            if (row["case_id"], row["branch_id"]) == key and row["region"] != "all"
        ]
        for region in sorted({str(row["region"]) for row in rows}):
            selected = [row for row in rows if row["region"] == region]
            time = np.asarray([row["time_ps"] for row in selected]) / 1000.0
            axes[0, column].plot(time, [row["mean_q_tet"] for row in selected], label=region)
            axes[1, column].plot(time, [row["mean_lsi_A2"] for row in selected], label=region)
            axes[2, column].plot(
                time, [row["hbond_largest_component_fraction"] for row in selected], label=region
            )
            axes[3, column].plot(
                time, [row["water_water_edge_turnover"] for row in selected], label=region
            )
        axes[0, column].set_title(f"{key[0]} / {key[1]}")
        axes[0, column].set_ylabel("q_tet")
        axes[1, column].set_ylabel("LSI (A^2)")
        axes[2, column].set_ylabel("Largest H-bond network")
        axes[3, column].set_ylabel("H-bond turnover")
        axes[3, column].set_xlabel("Time (ns)")
        axes[0, column].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "water_structure_overview.png", dpi=220)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Run a JSON contract and write auditable tabular outputs."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    output.mkdir(parents=True)
    base = contract_path.parent
    timestep_fs = float(raw["timestep_fs"])
    origin = int(raw.get("time_origin_step", 0))
    surface_range = _range(raw["surface_atom_range"], "surface_atom_range")
    water_range = _range(raw["water_atom_range"], "water_atom_range")
    oxygen_type = int(raw.get("oxygen_type", 2))
    hydrogen_type = int(raw.get("hydrogen_type", 1))
    oh_cutoff = float(raw["oh_cutoff_A"])
    oo_cutoff = float(raw["oo_cutoff_A"])
    angle = float(raw["hbond_angle_deg"])
    lsi_cutoff = float(raw["lsi_cutoff_A"])
    lsi_cap = int(raw.get("lsi_neighbor_cap", 24))
    cluster_cutoff = float(raw["cluster_cutoff_A"])
    frame_rows: list[dict[str, object]] = []
    exchange_rows: list[dict[str, object]] = []
    residence_values: dict[tuple[str, str, str, str], list[tuple[float, bool]]] = defaultdict(list)
    lifetime_values: dict[tuple[str, str, str, str], list[tuple[float, bool]]] = defaultdict(list)
    input_rows = [{
        "path": str(contract_path), "size_bytes": contract_path.stat().st_size,
        "sha256": sha256(contract_path),
    }]

    for entry in raw["cases"]:
        case_id, branch_id = str(entry["case_id"]), str(entry["branch_id"])
        direction = str(entry["direction"]).lower()
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        input_rows.extend(
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        )
        previous_labels: dict[int, str] = {}
        residence_state: dict[int, tuple[str, float]] = {}
        previous_region_edges: dict[str, tuple[set[tuple[int, int]], set[tuple[int, int]]]] = {}
        active_edges = {"water_water": {}, "water_surface": {}}
        previous_time: float | None = None
        sampling_interval: float | None = None
        final_time = math.nan
        for frame in iter_chemistry_frames(
            paths, surface_range, water_range, oxygen_type=oxygen_type,
            hydrogen_type=hydrogen_type, oh_cutoff_A=oh_cutoff,
        ):
            time_ps = (frame.step - origin) * timestep_fs / 1000.0
            final_time = time_ps
            if previous_time is not None:
                interval = time_ps - previous_time
                if interval <= 0.0:
                    raise ValueError("Frame times must be strictly increasing")
                if sampling_interval is None:
                    sampling_interval = interval
                elif not math.isclose(sampling_interval, interval, rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError("A constant frame interval is required for persistence")
            labels, regions, component_count, largest_fraction = region_labels(
                frame, entry, cluster_cutoff
            )
            qtet, lsi, coordination = water_order_metrics(
                frame, oo_cutoff_A=oo_cutoff, lsi_cutoff_A=lsi_cutoff,
                lsi_neighbor_cap=lsi_cap,
            )
            water_edges, water_donor, surface_donor = water_hbond_edges(
                frame, oo_cutoff_A=oo_cutoff, angle_deg=angle
            )
            surface_edges = water_donor | surface_donor
            current_labels = {
                int(atom_id): str(label) for atom_id, label in zip(frame.water_ids, labels)
            }
            if previous_time is not None:
                dt_ps = time_ps - previous_time
                transitions: dict[tuple[str, str], int] = defaultdict(int)
                for atom_id, label in current_labels.items():
                    old = previous_labels.get(atom_id)
                    if old is not None and old != label:
                        transitions[(old, label)] += 1
                for (old, new), count in sorted(transitions.items()):
                    exchange_rows.append({
                        "case_id": case_id, "branch_id": branch_id, "direction": direction,
                        "step": frame.step, "time_ps": time_ps, "from_region": old,
                        "to_region": new, "water_count": count, "rate_per_ps": count / dt_ps,
                    })
            for atom_id, label in current_labels.items():
                old = residence_state.get(atom_id)
                if old is None:
                    residence_state[atom_id] = (label, time_ps)
                elif old[0] != label:
                    residence_values[(case_id, branch_id, direction, old[0])].append(
                        (time_ps - old[1], False)
                    )
                    residence_state[atom_id] = (label, time_ps)

            if previous_time is not None and sampling_interval is not None:
                _record_edge_episodes(
                    active_edges["water_water"], water_edges, time_ps, sampling_interval,
                    lifetime_values[(case_id, branch_id, direction, "water_water")],
                )
                _record_edge_episodes(
                    active_edges["water_surface"], surface_edges, time_ps, sampling_interval,
                    lifetime_values[(case_id, branch_id, direction, "water_surface")],
                )
            else:
                active_edges["water_water"] = {
                    edge: (time_ps, time_ps) for edge in water_edges
                }
                active_edges["water_surface"] = {
                    edge: (time_ps, time_ps) for edge in surface_edges
                }

            for region, selected in regions.items():
                selected_ids = {int(frame.water_ids[index]) for index in selected}
                induced_water = {
                    edge
                    for edge in water_edges
                    if edge[0] in selected_ids and edge[1] in selected_ids
                }
                selected_water_donor = {edge for edge in water_donor if edge[0] in selected_ids}
                selected_surface_donor = {edge for edge in surface_donor if edge[0] in selected_ids}
                selected_surface = selected_water_donor | selected_surface_donor
                previous = previous_region_edges.get(region)
                ww_jaccard, ww_turnover = edge_similarity(
                    previous[0] if previous is not None else None, induced_water
                )
                ws_jaccard, ws_turnover = edge_similarity(
                    previous[1] if previous is not None else None, selected_surface
                )
                network_components, network_largest = hbond_component_metrics(
                    selected_ids, induced_water
                )
                selected_array = np.asarray(selected, dtype=int)
                frame_rows.append({
                    "case_id": case_id, "branch_id": branch_id, "direction": direction,
                    "step": frame.step, "time_ps": time_ps, "region": region,
                    "water_count": len(selected),
                    "water_fraction": len(selected) / len(frame.water),
                    "physical_component_count": component_count,
                    "physical_largest_component_fraction": largest_fraction,
                    "mean_q_tet": _finite_mean(qtet[selected_array]),
                    "mean_lsi_A2": _finite_mean(lsi[selected_array]),
                    "mean_oo_coordination": _finite_mean(coordination[selected_array]),
                    "h_coordination_defect_fraction": _finite_mean(
                        frame.water_h_coordination[selected_array] != 2
                    ),
                    "water_water_hbond_edges": len(induced_water),
                    "water_surface_hbond_edges": len(selected_surface),
                    "water_donor_surface_edges": len(selected_water_donor),
                    "surface_donor_water_edges": len(selected_surface_donor),
                    "hbond_component_count": network_components,
                    "hbond_largest_component_fraction": network_largest,
                    "water_water_edge_jaccard": ww_jaccard,
                    "water_water_edge_turnover": ww_turnover,
                    "water_surface_edge_jaccard": ws_jaccard,
                    "water_surface_edge_turnover": ws_turnover,
                })
                previous_region_edges[region] = (induced_water, selected_surface)
            previous_labels = current_labels
            previous_time = time_ps

        if previous_time is None or sampling_interval is None:
            raise ValueError(f"At least two frames are required for {case_id}/{branch_id}")
        for _, (region, start) in residence_state.items():
            residence_values[(case_id, branch_id, direction, region)].append(
                (final_time - start + sampling_interval, True)
            )
        for edge_type, active in active_edges.items():
            values = lifetime_values[(case_id, branch_id, direction, edge_type)]
            for start, last in active.values():
                values.append((last - start + sampling_interval, True))

    residence_rows = []
    for key, values in sorted(residence_values.items()):
        count, mean, median, p95, censored = _summary(values)
        residence_rows.append({
            "case_id": key[0], "branch_id": key[1], "direction": key[2], "region": key[3],
            "episodes": count, "mean_residence_ps": mean, "median_residence_ps": median,
            "p95_residence_ps": p95, "right_censored_episodes": censored,
        })
    lifetime_rows = []
    for key, values in sorted(lifetime_values.items()):
        count, mean, median, p95, censored = _summary(values)
        case_rows = [
            row for row in frame_rows
            if (row["case_id"], row["branch_id"], row["direction"]) == key[:3]
            and row["region"] == "all"
        ]
        interval = float(case_rows[1]["time_ps"]) - float(case_rows[0]["time_ps"])
        lifetime_rows.append({
            "case_id": key[0], "branch_id": key[1], "direction": key[2],
            "edge_type": key[3], "episodes": count, "mean_persistence_ps": mean,
            "median_persistence_ps": median, "p95_persistence_ps": p95,
            "right_censored_episodes": censored, "sampling_interval_ps": interval,
        })

    write_tsv(output / "water_structure_by_frame.tsv", frame_rows, FRAME_FIELDS)
    write_tsv(output / "region_exchange.tsv", exchange_rows, EXCHANGE_FIELDS)
    write_tsv(output / "region_residence_summary.tsv", residence_rows, RESIDENCE_FIELDS)
    write_tsv(output / "hbond_persistence_summary.tsv", lifetime_rows, LIFETIME_FIELDS)
    write_tsv(
        output / "input_manifest.tsv", input_rows,
        ("path", "size_bytes", "sha256"),
    )
    if bool(raw.get("write_plots", True)):
        _plot(frame_rows, output)
    summary = {
        "status": "PASS",
        "scientific_status": (
            "SINGLE_TRAJECTORY_MORPHOLOGY_AWARE_WATER_STRUCTURE_DIAGNOSTIC_"
            "NOT_FREE_ENERGY_RATE_OR_CAUSAL_MECHANISM"
        ),
        "case_branches": len(raw["cases"]),
        "frame_region_rows": len(frame_rows),
        "exchange_rows": len(exchange_rows),
        "residence_rows": len(residence_rows),
        "persistence_rows": len(lifetime_rows),
        "persistence_resolution": (
            "Edges are sampled at the reported frame interval; shorter events are unresolved."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_output_hashes(output)
    return summary
