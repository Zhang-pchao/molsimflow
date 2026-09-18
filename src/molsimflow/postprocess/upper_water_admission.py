"""Audit whether relocated slab water is an admissible enhanced-sampling parent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import LammpsDumpFrame, box_lengths, iter_lammps_dump_records

PERIODIC_SLAB = (True, True, False)


def read_lammps_atomic_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read IDs, types, coordinates, and orthorhombic bounds from atomic data."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    atom_count = None
    bounds = []
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[1] == "atoms":
            atom_count = int(fields[0])
        if line.strip().endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            bounds.append((float(fields[0]), float(fields[1])))
        if line.strip().endswith(("xy xz yz", "xz yz xy", "yz xy xz")):
            raise ValueError("triclinic LAMMPS data files are not supported")
    if atom_count is None or len(bounds) != 3:
        raise ValueError(f"{path} is missing an atom count or orthorhombic bounds")
    try:
        header = next(index for index, line in enumerate(lines) if line.strip().startswith("Atoms"))
    except StopIteration as exc:
        raise ValueError(f"{path} is missing an Atoms section") from exc
    records = []
    for line in lines[header + 1 :]:
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if not fields[0].lstrip("+-").isdigit():
            if records:
                break
            continue
        if len(fields) < 5:
            raise ValueError(f"invalid atomic row in {path}: {line!r}")
        records.append(
            (int(fields[0]), int(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]))
        )
        if len(records) == atom_count:
            break
    if len(records) != atom_count:
        raise ValueError(f"expected {atom_count} atoms in {path}, found {len(records)}")
    records.sort(key=lambda row: row[0])
    ids = np.asarray([row[0] for row in records], dtype=np.int64)
    if len(set(ids.tolist())) != atom_count:
        raise ValueError(f"{path} contains duplicate atom IDs")
    return (
        ids,
        np.asarray([row[1] for row in records], dtype=np.int64),
        np.asarray([row[2:] for row in records], dtype=float),
        np.asarray(bounds, dtype=float),
    )


def maximum_slab_coordinate_mismatch(
    reference: np.ndarray, observed: np.ndarray, bounds: np.ndarray
) -> float:
    """Return maximum coordinate mismatch using x/y minimum images and direct z."""

    delta = np.asarray(observed, dtype=float) - np.asarray(reference, dtype=float)
    lengths = box_lengths(bounds)
    delta[:, :2] -= lengths[:2] * np.rint(delta[:, :2] / lengths[:2])
    return float(np.max(np.linalg.norm(delta, axis=1)))


def _frame_arrays(frame: LammpsDumpFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {name: position for position, name in enumerate(frame.atom_fields)}
    missing = {"id", "type", "x", "y", "z"}.difference(index)
    if missing:
        raise ValueError(f"timestep {frame.timestep} is missing columns: {sorted(missing)}")
    ids = np.fromiter(
        (int(row[index["id"]]) for row in frame.atom_rows), dtype=np.int64, count=frame.atom_count
    )
    types = np.fromiter(
        (int(row[index["type"]]) for row in frame.atom_rows),
        dtype=np.int64,
        count=frame.atom_count,
    )
    coordinates = np.asarray(
        [[float(row[index[name]]) for name in ("x", "y", "z")] for row in frame.atom_rows],
        dtype=float,
    )
    if len(set(ids.tolist())) != frame.atom_count:
        raise ValueError(f"timestep {frame.timestep} contains duplicate atom IDs")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"timestep {frame.timestep} contains non-finite coordinates")
    return ids, types, coordinates


def contact_components(
    coordinates: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
    periodic: Sequence[bool] = PERIODIC_SLAB,
) -> list[np.ndarray]:
    """Return distance-connected components under selected periodic dimensions."""

    from scipy.spatial import cKDTree

    coords = np.asarray(coordinates, dtype=float)
    limits = np.asarray(bounds, dtype=float)
    periodic_mask = np.asarray(periodic, dtype=bool)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if limits.shape != (3, 2):
        raise ValueError("bounds must have shape (3, 2)")
    if periodic_mask.shape != (3,):
        raise ValueError("periodic must have shape (3,)")
    if cutoff_A <= 0.0:
        raise ValueError("cutoff_A must be positive")
    if len(coords) == 0:
        return []

    lengths = box_lengths(limits)
    normalized = coords.copy()
    normalized[:, periodic_mask] = (
        normalized[:, periodic_mask] - limits[periodic_mask, 0]
    ) % lengths[periodic_mask]
    pairs = cKDTree(
        normalized,
        boxsize=np.where(periodic_mask, lengths, 0.0),
    ).query_pairs(float(cutoff_A), output_type="ndarray")

    parent = np.arange(len(coords), dtype=int)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for left, right in pairs:
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    members: dict[int, list[int]] = defaultdict(list)
    for item in range(len(coords)):
        members[find(item)].append(item)
    return [
        np.asarray(component, dtype=int)
        for component in sorted(members.values(), key=lambda value: (-len(value), value[0]))
    ]


def periodic_center_2d(coordinates: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Return a circular x/y center and an arithmetic z center."""

    coords = np.asarray(coordinates, dtype=float)
    limits = np.asarray(bounds, dtype=float)
    lengths = box_lengths(limits)
    center = np.empty(3, dtype=float)
    for dimension in (0, 1):
        angles = 2.0 * np.pi * (coords[:, dimension] - limits[dimension, 0]) / lengths[dimension]
        value = np.mean(np.exp(1j * angles))
        angle = np.angle(value) % (2.0 * np.pi)
        center[dimension] = limits[dimension, 0] + lengths[dimension] * angle / (2.0 * np.pi)
    center[2] = float(np.mean(coords[:, 2]))
    return center


def shortest_periodic_arc(
    values: np.ndarray, lower: float, length: float, fraction: float = 0.95
) -> float:
    """Return the shortest periodic arc containing at least ``fraction`` of points."""

    wrapped = np.sort((np.asarray(values, dtype=float) - lower) % length)
    if len(wrapped) <= 1:
        return 0.0
    count = max(1, min(len(wrapped), math.ceil(fraction * len(wrapped))))
    extended = np.concatenate([wrapped, wrapped + length])
    widths = extended[np.arange(len(wrapped)) + count - 1] - extended[: len(wrapped)]
    return float(np.min(widths))


def frame_metrics(
    frame: LammpsDumpFrame,
    *,
    nsub: int,
    oxygen_type: int,
    lower_cut_z_A: float,
    cluster_cutoff_A: float,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray], list[np.ndarray]]:
    ids, types, coordinates = _frame_arrays(frame)
    water_mask = (ids > nsub) & (types == oxygen_type)
    water = coordinates[water_mask]
    if len(water) == 0:
        raise ValueError(f"timestep {frame.timestep} contains no selected water oxygen")
    components = contact_components(water, frame.bounds, cluster_cutoff_A)
    sizes = [len(component) for component in components]
    center = periodic_center_2d(water, frame.bounds)
    lengths = box_lengths(frame.bounds)
    resultants = []
    arcs = []
    for dimension in (0, 1):
        angles = 2.0 * np.pi * (
            (water[:, dimension] - frame.bounds[dimension, 0]) / lengths[dimension]
        )
        resultants.append(float(abs(np.mean(np.exp(1j * angles)))))
        arcs.append(
            shortest_periodic_arc(
                water[:, dimension], frame.bounds[dimension, 0], lengths[dimension]
            )
        )
    substrate = coordinates[ids <= nsub]
    row: dict[str, object] = {
        "frame_index": int(frame.frame_index),
        "step": int(frame.timestep),
        "water_oxygen_count": len(water),
        "lower_water_oxygen_count": int(np.count_nonzero(water[:, 2] < lower_cut_z_A)),
        "largest_component_size": int(sizes[0]),
        "largest_component_fraction": float(sizes[0] / len(water)),
        "second_component_size": int(sizes[1] if len(sizes) > 1 else 0),
        "component_count": len(sizes),
        "water_oxygen_z_min_A": float(np.min(water[:, 2])),
        "water_oxygen_z_q05_A": float(np.quantile(water[:, 2], 0.05)),
        "water_oxygen_z_mean_A": float(center[2]),
        "water_oxygen_z_q95_A": float(np.quantile(water[:, 2], 0.95)),
        "water_oxygen_z_max_A": float(np.max(water[:, 2])),
        "water_oxygen_center_x_A": float(center[0]),
        "water_oxygen_center_y_A": float(center[1]),
        "substrate_z_max_A": float(np.max(substrate[:, 2])),
        "water_substrate_min_z_gap_A": float(np.min(water[:, 2]) - np.max(substrate[:, 2])),
        "all_atom_top_clearance_A": float(frame.bounds[2, 1] - np.max(coordinates[:, 2])),
        "water_oxygen_top_clearance_A": float(frame.bounds[2, 1] - np.max(water[:, 2])),
        "xy_first_harmonic_resultant_x": resultants[0],
        "xy_first_harmonic_resultant_y": resultants[1],
        "xy_shortest_arc95_x_A": arcs[0],
        "xy_shortest_arc95_y_A": arcs[1],
        "xy_shortest_arc95_x_fraction": float(arcs[0] / lengths[0]),
        "xy_shortest_arc95_y_fraction": float(arcs[1] / lengths[1]),
    }
    return row, (ids, types, coordinates), components


def _count_distribution(values: Sequence[int]) -> dict[str, int]:
    counter = Counter(int(value) for value in values)
    return {
        "0H": counter.get(0, 0),
        "1H": counter.get(1, 0),
        "2H": counter.get(2, 0),
        "3H": counter.get(3, 0),
        "ge4H": sum(count for value, count in counter.items() if value >= 4),
    }


def assign_hydrogens_to_heavy_atoms(
    oxygen_ids: np.ndarray,
    oxygen: np.ndarray,
    carbon_ids: np.ndarray,
    carbon: np.ndarray,
    hydrogen_ids: np.ndarray,
    hydrogen: np.ndarray,
    bounds: np.ndarray,
    *,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
    periodic: Sequence[bool] = PERIODIC_SLAB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest valid O/C owners for H under selected periodic axes."""

    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    origin = np.asarray(bounds, dtype=float)[:, 0]
    periodic_mask = np.asarray(periodic, dtype=bool)
    if periodic_mask.shape != (3,):
        raise ValueError("periodic must have shape (3,)")
    normalized_h = np.asarray(hydrogen, dtype=float).copy()
    normalized_h[:, periodic_mask] = (
        normalized_h[:, periodic_mask] - origin[periodic_mask]
    ) % lengths[periodic_mask]

    def nearest(ids: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(ids) == 0:
            return np.full(len(hydrogen), np.inf), np.full(len(hydrogen), -1, dtype=int)
        normalized = np.asarray(coordinates, dtype=float).copy()
        normalized[:, periodic_mask] = (
            normalized[:, periodic_mask] - origin[periodic_mask]
        ) % lengths[periodic_mask]
        distance, local_index = cKDTree(
            normalized,
            boxsize=np.where(periodic_mask, lengths, 0.0),
        ).query(normalized_h)
        return np.asarray(distance), np.asarray(local_index, dtype=int)

    oxygen_distance, oxygen_index = nearest(oxygen_ids, oxygen)
    carbon_distance, carbon_index = nearest(carbon_ids, carbon)
    valid_oxygen = oxygen_distance <= oh_cutoff_A
    valid_carbon = carbon_distance <= ch_cutoff_A
    choose_oxygen = valid_oxygen & (~valid_carbon | (oxygen_distance <= carbon_distance))
    choose_carbon = valid_carbon & (~valid_oxygen | (carbon_distance < oxygen_distance))
    owner_ids = np.full(len(hydrogen), -1, dtype=int)
    owner_elements = np.full(len(hydrogen), "", dtype="U1")
    distances = np.full(len(hydrogen), np.inf)
    owner_ids[choose_oxygen] = oxygen_ids[oxygen_index[choose_oxygen]]
    owner_elements[choose_oxygen] = "O"
    distances[choose_oxygen] = oxygen_distance[choose_oxygen]
    owner_ids[choose_carbon] = carbon_ids[carbon_index[choose_carbon]]
    owner_elements[choose_carbon] = "C"
    distances[choose_carbon] = carbon_distance[choose_carbon]
    return owner_ids, owner_elements, distances


def species_metrics(
    ids: np.ndarray,
    types: np.ndarray,
    coordinates: np.ndarray,
    bounds: np.ndarray,
    *,
    nsub: int,
    hydrogen_type: int,
    oxygen_type: int,
    carbon_type: int,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
) -> tuple[dict[str, object], dict[int, tuple[int, ...]]]:
    oxygen_mask = types == oxygen_type
    carbon_mask = types == carbon_type
    hydrogen_mask = types == hydrogen_type
    hydrogen_ids = ids[hydrogen_mask]
    owner_ids, _owner_elements, owner_distances = assign_hydrogens_to_heavy_atoms(
        ids[oxygen_mask],
        coordinates[oxygen_mask],
        ids[carbon_mask],
        coordinates[carbon_mask],
        hydrogen_ids,
        coordinates[hydrogen_mask],
        bounds,
        oh_cutoff_A=oh_cutoff_A,
        ch_cutoff_A=ch_cutoff_A,
        periodic=PERIODIC_SLAB,
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for hydrogen_id, owner_id in zip(hydrogen_ids, owner_ids):
        if int(owner_id) >= 0:
            grouped[int(owner_id)].append(int(hydrogen_id))
    owned = {owner_id: tuple(sorted(values)) for owner_id, values in grouped.items()}
    water_oxygen_ids = ids[(ids > nsub) & oxygen_mask]
    substrate_oxygen_ids = ids[(ids <= nsub) & oxygen_mask]
    carbon_ids = ids[carbon_mask]
    water_counts = {int(atom_id): len(owned.get(int(atom_id), ())) for atom_id in water_oxygen_ids}
    substrate_counts = [len(owned.get(int(atom_id), ())) for atom_id in substrate_oxygen_ids]
    carbon_counts = [len(owned.get(int(atom_id), ())) for atom_id in carbon_ids]
    water_set = set(map(int, water_oxygen_ids))
    water_hydrogen_ids = set(map(int, ids[(ids > nsub) & hydrogen_mask]))
    transferred_substrate_h = sorted(
        int(hydrogen_id)
        for hydrogen_id, owner_id in zip(hydrogen_ids, owner_ids)
        if int(hydrogen_id) <= nsub and int(owner_id) in water_set
    )
    water_h_to_substrate = sorted(
        int(hydrogen_id)
        for hydrogen_id, owner_id in zip(hydrogen_ids, owner_ids)
        if int(hydrogen_id) in water_hydrogen_ids and 0 < int(owner_id) <= nsub
    )
    unassigned = sorted(
        int(hydrogen_id)
        for hydrogen_id, owner_id in zip(hydrogen_ids, owner_ids)
        if int(owner_id) < 0
    )
    anomalous = {
        f"{count}H": sorted(atom_id for atom_id, value in water_counts.items() if value == count)
        for count in (0, 1, 3)
    }
    anomalous["ge4H"] = sorted(atom_id for atom_id, value in water_counts.items() if value >= 4)
    report: dict[str, object] = {
        "water_oxygen_count": len(water_oxygen_ids),
        "water_oxygen_hydrogen_count_distribution": _count_distribution(list(water_counts.values())),
        "water_oxygen_anomalous_ids": anomalous,
        "substrate_oxygen_count": len(substrate_oxygen_ids),
        "substrate_oxygen_hydrogen_count_distribution": _count_distribution(substrate_counts),
        "carbon_count": len(carbon_ids),
        "carbon_hydrogen_count_distribution": _count_distribution(carbon_counts),
        "hydrogen_count": int(np.count_nonzero(hydrogen_mask)),
        "unassigned_hydrogen_count": len(unassigned),
        "unassigned_hydrogen_ids": unassigned,
        "substrate_id_hydrogen_owned_by_water_oxygen_count": len(transferred_substrate_h),
        "substrate_id_hydrogen_owned_by_water_oxygen_ids": transferred_substrate_h,
        "water_id_hydrogen_owned_by_substrate_heavy_count": len(water_h_to_substrate),
        "water_id_hydrogen_owned_by_substrate_heavy_ids": water_h_to_substrate,
        "water_oxygen_owned_hydrogen_count": sum(water_counts.values()),
        "substrate_oxygen_owned_hydrogen_count": sum(substrate_counts),
        "carbon_owned_hydrogen_count": sum(carbon_counts),
        "assignment_distance_max_A": float(
            np.max(owner_distances[np.isfinite(owner_distances)])
        ),
        "oh_cutoff_A": float(oh_cutoff_A),
        "ch_cutoff_A": float(ch_cutoff_A),
        "periodic_axes": "xy",
    }
    return report, owned


def read_monitor(path: Path) -> tuple[list[str], np.ndarray]:
    header: list[str] | None = None
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("# TimeStep"):
                header = stripped[2:].split()
            elif stripped and not stripped.startswith("#"):
                rows.append([float(value) for value in stripped.split()])
    if header is None or not rows:
        raise ValueError(f"monitor has no header or data: {path}")
    data = np.asarray(rows, dtype=float)
    if data.shape[1] != len(header):
        raise ValueError(f"monitor column mismatch: expected {len(header)}, got {data.shape[1]}")
    return header, data


def monitor_summary(path: Path, late_start_step: int) -> dict[str, object]:
    header, data = read_monitor(path)
    index = {name: position for position, name in enumerate(header)}
    required = {"TimeStep", "v_topclear", "c_ZminO", "c_ZmaxO"}
    missing = required.difference(index)
    if missing:
        raise ValueError(f"monitor is missing columns: {sorted(missing)}")
    steps = data[:, index["TimeStep"]].astype(np.int64)
    clearance = data[:, index["v_topclear"]]
    late = steps >= late_start_step
    minimum_index = int(np.argmin(clearance))
    result: dict[str, object] = {
        "sample_count": len(data),
        "minimum_top_clearance_A": float(clearance[minimum_index]),
        "minimum_top_clearance_step": int(steps[minimum_index]),
        "samples_below_10A": int(np.count_nonzero(clearance < 10.0)),
        "samples_below_1A": int(np.count_nonzero(clearance < 1.0)),
        "samples_below_0p5A": int(np.count_nonzero(clearance < 0.5)),
        "samples_below_0p2A": int(np.count_nonzero(clearance < 0.2)),
        "first_step_below_10A": (
            int(steps[np.flatnonzero(clearance < 10.0)[0]]) if np.any(clearance < 10.0) else None
        ),
        "last_step_below_10A": (
            int(steps[np.flatnonzero(clearance < 10.0)[-1]]) if np.any(clearance < 10.0) else None
        ),
        "last_step_below_1A": (
            int(steps[np.flatnonzero(clearance < 1.0)[-1]]) if np.any(clearance < 1.0) else None
        ),
        "late_sample_count": int(np.count_nonzero(late)),
        "late_minimum_top_clearance_A": float(np.min(clearance[late])),
        "late_lower_water_oxygen_z_min_A": float(np.min(data[late, index["c_ZminO"]])),
        "late_water_oxygen_z_max_A": float(np.max(data[late, index["c_ZmaxO"]])),
    }
    return result


def _linear_slope(rows: Sequence[Mapping[str, object]], column: str, timestep_fs: float) -> float:
    if len(rows) < 2:
        return math.nan
    step = np.asarray([float(row["step"]) for row in rows])
    time_ps = (step - step[0]) * timestep_fs / 1000.0
    values = np.asarray([float(row[column]) for row in rows])
    return float(np.polyfit(time_ps, values, 1)[0])


def _read_status(path: Path) -> str:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("status="):
            return line.split("=", 1)[1].strip()
    return "UNKNOWN"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _morphology_label(final: Mapping[str, object]) -> str:
    arc_x = float(final["xy_shortest_arc95_x_fraction"])
    arc_y = float(final["xy_shortest_arc95_y_fraction"])
    result_x = float(final["xy_first_harmonic_resultant_x"])
    result_y = float(final["xy_first_harmonic_resultant_y"])
    if arc_x >= 0.85 and arc_y >= 0.85:
        return "laterally_spread_film_candidate"
    if arc_x <= 0.80 and arc_y <= 0.80 and max(result_x, result_y) >= 0.25:
        return "finite_laterally_localized_body_candidate"
    return "intermediate_or_anisotropic_candidate"


def compact_species_row(step: int, report: Mapping[str, object]) -> dict[str, object]:
    distribution = report["water_oxygen_hydrogen_count_distribution"]
    assert isinstance(distribution, Mapping)
    return {
        "step": int(step),
        "water_0H_count": int(distribution["0H"]),
        "water_1H_OH_like_count": int(distribution["1H"]),
        "water_2H_H2O_count": int(distribution["2H"]),
        "water_3H_H3O_like_count": int(distribution["3H"]),
        "water_ge4H_count": int(distribution["ge4H"]),
        "water_owned_hydrogen_count": int(report["water_oxygen_owned_hydrogen_count"]),
        "substrate_oxygen_owned_hydrogen_count": int(
            report["substrate_oxygen_owned_hydrogen_count"]
        ),
        "carbon_owned_hydrogen_count": int(report["carbon_owned_hydrogen_count"]),
        "unassigned_hydrogen_count": int(report["unassigned_hydrogen_count"]),
        "substrate_id_hydrogen_owned_by_water_oxygen_count": int(
            report["substrate_id_hydrogen_owned_by_water_oxygen_count"]
        ),
        "water_id_hydrogen_owned_by_substrate_heavy_count": int(
            report["water_id_hydrogen_owned_by_substrate_heavy_count"]
        ),
    }


def write_plot(
    frame_rows: Sequence[Mapping[str, object]], monitor_path: Path, output: Path, timestep_fs: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    step0 = float(frame_rows[0]["step"])
    time_ps = np.asarray(
        [(float(row["step"]) - step0) * timestep_fs / 1000.0 for row in frame_rows]
    )
    header, monitor = read_monitor(monitor_path)
    index = {name: position for position, name in enumerate(header)}
    monitor_time = (monitor[:, index["TimeStep"]] - step0) * timestep_fs / 1000.0
    figure, axes = plt.subplots(4, 1, figsize=(8.0, 9.5), sharex=True)
    axes[0].plot(time_ps, [row["largest_component_fraction"] for row in frame_rows], marker="o")
    axes[0].set_ylabel("Largest O cluster")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(time_ps, [row["lower_water_oxygen_count"] for row in frame_rows], marker="o")
    axes[1].set_ylabel("Lower-side O")
    for key, label in (
        ("water_oxygen_z_min_A", "min"),
        ("water_oxygen_z_mean_A", "mean"),
        ("water_oxygen_z_max_A", "max"),
    ):
        axes[2].plot(time_ps, [row[key] for row in frame_rows], label=label)
    axes[2].set_ylabel("Water O z (A)")
    axes[2].legend(frameon=False, ncol=3)
    axes[3].plot(monitor_time, monitor[:, index["v_topclear"]], linewidth=0.8)
    axes[3].axhline(10.0, color="tab:red", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Top clearance (A)")
    axes[3].set_xlabel("Production time (ps)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame_rows = []
    late_span_steps = round(args.late_window_ps * 1000.0 / args.timestep_fs)
    late_start_step = args.expected_final_step - late_span_steps
    identity: tuple[tuple[int, int], ...] | None = None
    first_arrays = None
    final_arrays = None
    final_bounds = None
    final_components = None
    final_water_ids = None
    for frame in iter_lammps_dump_records(args.trajectory):
        row, arrays, components = frame_metrics(
            frame,
            nsub=args.nsub,
            oxygen_type=args.oxygen_type,
            lower_cut_z_A=args.lower_cut_z_A,
            cluster_cutoff_A=args.cluster_cutoff_A,
        )
        current_identity = tuple(zip(arrays[0].tolist(), arrays[1].tolist()))
        if identity is None:
            identity = current_identity
            first_arrays = arrays
        elif current_identity != identity:
            raise ValueError(f"atom ID/type identity changed at timestep {frame.timestep}")
        frame_rows.append(row)
        final_arrays = arrays
        final_bounds = frame.bounds.copy()
        final_components = components
        final_water_ids = arrays[0][(arrays[0] > args.nsub) & (arrays[1] == args.oxygen_type)]
    if not frame_rows or first_arrays is None or final_arrays is None or final_bounds is None:
        raise ValueError("trajectory contains no complete frames")
    if int(frame_rows[-1]["step"]) != args.expected_final_step:
        raise ValueError(
            f"last timestep {frame_rows[-1]['step']} != expected {args.expected_final_step}"
        )
    if any(int(row["water_oxygen_count"]) != args.expected_water_oxygen_count for row in frame_rows):
        raise ValueError("water oxygen count changed or differs from expectation")

    data_ids, data_types, data_xyz, data_bounds = read_lammps_atomic_data(args.final_data)
    final_ids, final_types, final_xyz = final_arrays
    final_order = np.argsort(final_ids)
    terminal_identity_pass = bool(
        np.array_equal(final_ids[final_order], data_ids)
        and np.array_equal(final_types[final_order], data_types)
    )
    terminal_bounds_pass = bool(np.allclose(final_bounds, data_bounds, atol=1.0e-8, rtol=0.0))
    terminal_coordinate_mismatch_A = (
        maximum_slab_coordinate_mismatch(final_xyz[final_order], data_xyz, final_bounds)
        if terminal_identity_pass and terminal_bounds_pass
        else math.inf
    )
    terminal_coordinate_pass = terminal_coordinate_mismatch_A <= args.coordinate_tolerance_A

    late_rows = [row for row in frame_rows if int(row["step"]) >= late_start_step]
    late_observed_span_ps = (
        (int(late_rows[-1]["step"]) - int(late_rows[0]["step"]))
        * args.timestep_fs
        / 1000.0
    )
    monitor = monitor_summary(args.monitor, late_start_step)
    species_rows = []
    first_species = None
    for frame in iter_lammps_dump_records(args.trajectory):
        if frame.frame_index != 0 and frame.timestep < late_start_step:
            continue
        ids, types, coordinates = _frame_arrays(frame)
        report, _ = species_metrics(
            ids,
            types,
            coordinates,
            frame.bounds,
            nsub=args.nsub,
            hydrogen_type=args.hydrogen_type,
            oxygen_type=args.oxygen_type,
            carbon_type=args.carbon_type,
            oh_cutoff_A=args.oh_cutoff_A,
            ch_cutoff_A=args.ch_cutoff_A,
        )
        if frame.frame_index == 0:
            first_species = report
        species_rows.append(compact_species_row(frame.timestep, report))
    if first_species is None:
        raise ValueError("failed to analyze first-frame species")
    final_species, final_owned = species_metrics(
        data_ids,
        data_types,
        data_xyz,
        data_bounds,
        nsub=args.nsub,
        hydrogen_type=args.hydrogen_type,
        oxygen_type=args.oxygen_type,
        carbon_type=args.carbon_type,
        oh_cutoff_A=args.oh_cutoff_A,
        ch_cutoff_A=args.ch_cutoff_A,
    )
    strict_neutral_water_pass = final_species["water_oxygen_hydrogen_count_distribution"] == {
        "0H": 0,
        "1H": 0,
        "2H": args.expected_water_oxygen_count,
        "3H": 0,
        "ge4H": 0,
    }
    final_distribution = final_species["water_oxygen_hydrogen_count_distribution"]
    species_inventory_pass = bool(
        final_distribution["0H"] == 0
        and final_distribution["ge4H"] == 0
        and final_species["unassigned_hydrogen_count"] == 0
    )
    owner_delta = sum(
        int(final_species[column]) - int(first_species[column])
        for column in (
            "water_oxygen_owned_hydrogen_count",
            "substrate_oxygen_owned_hydrogen_count",
            "carbon_owned_hydrogen_count",
        )
    )
    species_inventory_pass = species_inventory_pass and owner_delta == 0
    reactive_water_species_present = bool(
        final_distribution["1H"] > 0 or final_distribution["3H"] > 0
    )
    late_z_mean_slope = _linear_slope(late_rows, "water_oxygen_z_mean_A", args.timestep_fs)
    late_z_q95_slope = _linear_slope(late_rows, "water_oxygen_z_q95_A", args.timestep_fs)
    late_z_mean_fitted_change = late_z_mean_slope * late_observed_span_ps
    late_z_q95_fitted_change = late_z_q95_slope * late_observed_span_ps
    late_z_trend_pass = bool(
        abs(late_z_mean_fitted_change) <= args.late_mean_z_trend_limit_A
        and abs(late_z_q95_fitted_change) <= args.late_q95_z_trend_limit_A
    )
    final_row = frame_rows[-1]
    final_upper_pass = int(final_row["lower_water_oxygen_count"]) == 0
    final_cluster_pass = int(final_row["largest_component_size"]) == args.expected_water_oxygen_count
    late_upper_pass = max(int(row["lower_water_oxygen_count"]) for row in late_rows) == 0
    late_cluster_pass = min(int(row["largest_component_size"]) for row in late_rows) == (
        args.expected_water_oxygen_count
    )
    late_wall_pass = float(monitor["late_minimum_top_clearance_A"]) >= args.wall_buffer_A
    full_wall_pass = float(monitor["minimum_top_clearance_A"]) >= args.wall_buffer_A
    structure_pass = all(
        (
            terminal_identity_pass,
            terminal_bounds_pass,
            terminal_coordinate_pass,
            species_inventory_pass,
            final_upper_pass,
            final_cluster_pass,
            late_upper_pass,
            late_cluster_pass,
        )
    )
    if not structure_pass or not late_wall_pass:
        admission = "REJECT"
    else:
        conditions = []
        if not full_wall_pass:
            conditions.append("WALL_HISTORY")
        if reactive_water_species_present:
            conditions.append("REACTIVE_CHEMISTRY")
        if not late_z_trend_pass:
            conditions.append("LATE_Z_RELAXATION")
        admission = "CONDITIONAL_" + "_AND_".join(conditions) if conditions else "ADMIT_CANDIDATE"

    gate = json.loads(Path(args.output_gate).read_text(encoding="utf-8"))
    run_status = _read_status(args.run_result)
    cutoff_sensitivity = {}
    final_water_mask = (final_ids > args.nsub) & (final_types == args.oxygen_type)
    for cutoff in args.cluster_sensitivity_A:
        components = contact_components(final_xyz[final_water_mask], final_bounds, cutoff)
        cutoff_sensitivity[f"{cutoff:g}"] = {
            "largest_component_size": len(components[0]),
            "second_component_size": len(components[1]) if len(components) > 1 else 0,
            "component_count": len(components),
        }

    summary: dict[str, object] = {
        "analysis_status": "PASS",
        "case_label": args.case_label,
        "trajectory_frames": len(frame_rows),
        "first_step": int(frame_rows[0]["step"]),
        "last_step": int(frame_rows[-1]["step"]),
        "frame_interval_ps": (
            (int(frame_rows[1]["step"]) - int(frame_rows[0]["step"]))
            * args.timestep_fs
            / 1000.0
            if len(frame_rows) > 1
            else None
        ),
        "scheduler_or_wrapper_status": run_status,
        "terminal_output_gate_status": gate.get("status", "UNKNOWN"),
        "terminal_output_gate_failure_is_historical_wall_buffer": bool(
            gate.get("status") == "FAIL"
            and gate.get("lammps", {}).get("minimum_top_clearance_A", math.inf)
            < args.wall_buffer_A
        ),
        "terminal_data_vs_last_dump": {
            "identity_pass": terminal_identity_pass,
            "bounds_pass": terminal_bounds_pass,
            "maximum_minimum_image_xy_direct_z_displacement_A": terminal_coordinate_mismatch_A,
            "coordinate_tolerance_A": args.coordinate_tolerance_A,
            "coordinate_pass": terminal_coordinate_pass,
        },
        "selection": {
            "water_oxygen": f"id > {args.nsub} and type == {args.oxygen_type}",
            "expected_water_oxygen_count": args.expected_water_oxygen_count,
            "periodic_axes": "xy",
            "nonperiodic_axis": "z",
            "cluster_cutoff_A": args.cluster_cutoff_A,
            "lower_cut_z_A": args.lower_cut_z_A,
            "late_window_ps": args.late_window_ps,
            "wall_buffer_A": args.wall_buffer_A,
        },
        "first_species": first_species,
        "final_species": final_species,
        "species_inventory_pass": bool(species_inventory_pass),
        "strict_neutral_water_pass": bool(strict_neutral_water_pass),
        "reactive_water_species_present": reactive_water_species_present,
        "heavy_atom_owner_inventory_delta_H": owner_delta,
        "late_species": {
            "sample_count": len(species_rows) - 1,
            "H2O_count_min": min(
                int(row["water_2H_H2O_count"]) for row in species_rows[1:]
            ),
            "H2O_count_max": max(
                int(row["water_2H_H2O_count"]) for row in species_rows[1:]
            ),
            "OH_like_count_min": min(
                int(row["water_1H_OH_like_count"]) for row in species_rows[1:]
            ),
            "OH_like_count_max": max(
                int(row["water_1H_OH_like_count"]) for row in species_rows[1:]
            ),
            "H3O_like_count_min": min(
                int(row["water_3H_H3O_like_count"]) for row in species_rows[1:]
            ),
            "H3O_like_count_max": max(
                int(row["water_3H_H3O_like_count"]) for row in species_rows[1:]
            ),
        },
        "final_frame": final_row,
        "final_cluster_cutoff_sensitivity": cutoff_sensitivity,
        "late_window": {
            "first_step": int(late_rows[0]["step"]),
            "last_step": int(late_rows[-1]["step"]),
            "frame_count": len(late_rows),
            "largest_component_size_min": min(
                int(row["largest_component_size"]) for row in late_rows
            ),
            "largest_component_size_max": max(
                int(row["largest_component_size"]) for row in late_rows
            ),
            "lower_water_oxygen_count_max": max(
                int(row["lower_water_oxygen_count"]) for row in late_rows
            ),
            "water_oxygen_z_min_A": min(
                float(row["water_oxygen_z_min_A"]) for row in late_rows
            ),
            "water_oxygen_z_max_A": max(
                float(row["water_oxygen_z_max_A"]) for row in late_rows
            ),
            "water_oxygen_z_mean_slope_A_per_ps": _linear_slope(
                late_rows, "water_oxygen_z_mean_A", args.timestep_fs
            ),
            "water_oxygen_z_q95_slope_A_per_ps": _linear_slope(
                late_rows, "water_oxygen_z_q95_A", args.timestep_fs
            ),
            "observed_span_ps": late_observed_span_ps,
            "water_oxygen_z_mean_fitted_change_A": late_z_mean_fitted_change,
            "water_oxygen_z_q95_fitted_change_A": late_z_q95_fitted_change,
            "mean_z_fitted_change_limit_A": args.late_mean_z_trend_limit_A,
            "q95_z_fitted_change_limit_A": args.late_q95_z_trend_limit_A,
            "z_trend_pass": late_z_trend_pass,
        },
        "monitor": monitor,
        "morphology_label": _morphology_label(final_row),
        "gates": {
            "terminal_data_identity_pass": terminal_identity_pass,
            "terminal_data_bounds_pass": terminal_bounds_pass,
            "terminal_data_coordinates_match_last_dump_pass": terminal_coordinate_pass,
            "final_species_inventory_pass": bool(species_inventory_pass),
            "final_strict_neutral_water_pass": bool(strict_neutral_water_pass),
            "final_all_water_oxygen_upper_pass": bool(final_upper_pass),
            "final_all_water_oxygen_in_largest_cluster_pass": bool(final_cluster_pass),
            "late_all_water_oxygen_upper_pass": bool(late_upper_pass),
            "late_all_water_oxygen_in_largest_cluster_pass": bool(late_cluster_pass),
            "late_wall_buffer_pass": bool(late_wall_pass),
            "full_history_wall_buffer_pass": bool(full_wall_pass),
            "late_z_trend_pass": late_z_trend_pass,
        },
        "enhanced_sampling_parent_admission": admission,
        "scientific_scope": (
            "Coordinate-parent admission only; this does not validate a CV, bias gradient, "
            "equilibration, PMF convergence, kinetics, or friction."
        ),
    }

    with (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "species_first_last.json").write_text(
        json.dumps({"first": first_species, "last": final_species}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "species_frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(species_rows[0]))
        writer.writeheader()
        writer.writerows(species_rows)

    assert final_components is not None and final_water_ids is not None
    component_rank = {}
    component_size = {}
    for rank, members in enumerate(final_components, start=1):
        for member in members:
            atom_id = int(final_water_ids[int(member)])
            component_rank[atom_id] = rank
            component_size[atom_id] = len(members)
    final_by_id = {int(atom_id): xyz for atom_id, xyz in zip(final_ids, final_xyz)}
    with (output / "final_water_oxygen_membership.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["atom_id", "component_rank", "component_size", "hydrogen_count", "x_A", "y_A", "z_A"]
        )
        for atom_id in sorted(map(int, final_water_ids)):
            xyz = final_by_id[atom_id]
            writer.writerow(
                [
                    atom_id,
                    component_rank[atom_id],
                    component_size[atom_id],
                    len(final_owned.get(atom_id, ())),
                    f"{xyz[0]:.10f}",
                    f"{xyz[1]:.10f}",
                    f"{xyz[2]:.10f}",
                ]
            )

    sources = [
        args.trajectory,
        args.final_data,
        args.monitor,
        args.output_gate,
        args.run_result,
    ]
    manifest = {
        "sources": [
            {"path": str(Path(path).resolve()), "size_bytes": Path(path).stat().st_size, "sha256": _sha256(path)}
            for path in sources
        ],
        "module_path": str(Path(__file__).resolve()),
        "module_sha256": _sha256(Path(__file__)),
        "parameters": vars(args),
    }
    manifest["parameters"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        write_plot(frame_rows, args.monitor, output / "admission_diagnostics.png", args.timestep_fs)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--final-data", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--nsub", type=int, required=True)
    parser.add_argument("--lower-cut-z-A", type=float, required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--expected-water-oxygen-count", type=int, default=1297)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--carbon-type", type=int, default=7)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.3)
    parser.add_argument("--ch-cutoff-A", type=float, default=1.3)
    parser.add_argument("--cluster-cutoff-A", type=float, default=3.5)
    parser.add_argument(
        "--cluster-sensitivity-A", type=float, nargs="+", default=(3.3, 3.5, 3.8)
    )
    parser.add_argument("--late-window-ps", type=float, default=200.0)
    parser.add_argument("--late-mean-z-trend-limit-A", type=float, default=0.5)
    parser.add_argument("--late-q95-z-trend-limit-A", type=float, default=1.0)
    parser.add_argument("--wall-buffer-A", type=float, default=10.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--coordinate-tolerance-A", type=float, default=1.0e-6)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_analysis(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
