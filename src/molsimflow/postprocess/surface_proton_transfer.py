"""Track persistent surface-H exchange and solution ion candidates in reactive MD."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.interfacial_water_density import REGIONS, classify_regions
from molsimflow.postprocess.interfacial_water_orientation import iter_orientation_frames
from molsimflow.postprocess.nanodroplet_spreading import parse_range
from molsimflow.postprocess.surface_reference import load_surface_reference
from molsimflow.postprocess.surface_site_enrichment import read_contact_lines


@dataclass(frozen=True)
class HeavyAtomAssignment:
    """Nearest valid O/C owner for each hydrogen."""

    hydrogen_ids: np.ndarray
    owner_ids: np.ndarray
    owner_elements: np.ndarray
    distances_A: np.ndarray


def assign_hydrogens_to_oxygen_or_carbon(
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
) -> HeavyAtomAssignment:
    """Assign each H to the closest valid O or C under orthorhombic PBC."""

    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    origin = bounds[:, 0]
    wrapped_h = (hydrogen - origin) % lengths

    def nearest(ids: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(ids) == 0:
            return np.full(len(hydrogen), np.inf), np.full(len(hydrogen), -1, dtype=int)
        distance, local_index = cKDTree(
            (coordinates - origin) % lengths, boxsize=lengths
        ).query(wrapped_h)
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
    return HeavyAtomAssignment(
        np.asarray(hydrogen_ids, dtype=int), owner_ids, owner_elements, distances
    )


def hydrogen_ids_by_owner(assignment: HeavyAtomAssignment) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for hydrogen_id, owner_id in zip(assignment.hydrogen_ids, assignment.owner_ids):
        if owner_id >= 0:
            grouped[int(owner_id)].append(int(hydrogen_id))
    return {owner_id: tuple(sorted(values)) for owner_id, values in grouped.items()}


def identify_initial_donor_sites(
    elements: np.ndarray,
    coordinates: np.ndarray,
    lengths: np.ndarray,
    *,
    slab_range: tuple[int, int],
    surface_z_A: float,
    surface_depth_A: float,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
) -> list[dict]:
    """Identify top SiOH/CH3 donors and their initial H atom identities."""

    atom_ids = np.arange(1, len(elements) + 1)
    oxygen_mask = elements == "O"
    carbon_mask = elements == "C"
    hydrogen_mask = elements == "H"
    bounds = np.column_stack((np.zeros(3), lengths))
    assignment = assign_hydrogens_to_oxygen_or_carbon(
        atom_ids[oxygen_mask],
        coordinates[oxygen_mask],
        atom_ids[carbon_mask],
        coordinates[carbon_mask],
        atom_ids[hydrogen_mask],
        coordinates[hydrogen_mask],
        bounds,
        oh_cutoff_A=oh_cutoff_A,
        ch_cutoff_A=ch_cutoff_A,
    )
    grouped = hydrogen_ids_by_owner(assignment)
    sites = []
    start, end = slab_range
    for atom_id in range(start, end + 1):
        index = atom_id - 1
        if coordinates[index, 2] < surface_z_A - surface_depth_A:
            continue
        hydrogen_ids = grouped.get(atom_id, ())
        if elements[index] == "O" and len(hydrogen_ids) == 1:
            site_type, expected = "SiOH", 1
        elif elements[index] == "C" and len(hydrogen_ids) == 3:
            site_type, expected = "CH3", 3
        else:
            continue
        sites.append(
            {
                "atom_id": atom_id,
                "site_type": site_type,
                "expected_h_count": expected,
                "initial_hydrogen_ids": ";".join(map(str, hydrogen_ids)),
                "x_A": float(coordinates[index, 0]),
                "y_A": float(coordinates[index, 1]),
                "z_A": float(coordinates[index, 2]),
                "_initial_hydrogen_ids": hydrogen_ids,
            }
        )
    if not sites:
        raise ValueError("No top SiOH or CH3 donor sites were identified")
    return sites


def classify_site_state(
    site_type: str, expected_h_count: int, initial_hydrogen_ids: Sequence[int], current_hydrogen_ids: Sequence[int]
) -> str:
    count = len(current_hydrogen_ids)
    if count < expected_h_count:
        return "deprotonated_candidate" if site_type == "SiOH" else "c_h_loss_candidate"
    if count > expected_h_count:
        return "hyperprotonated_candidate" if site_type == "SiOH" else "c_h_gain_candidate"
    if set(current_hydrogen_ids) != set(initial_hydrogen_ids):
        return "proton_exchanged_candidate" if site_type == "SiOH" else "hydrogen_exchanged_candidate"
    return "nominal"


def classify_ion_pair_state(h3o_count: int, oh_count: int) -> str:
    if h3o_count == 0 and oh_count == 0:
        return "neutral"
    if h3o_count > 0 and oh_count > 0:
        return "paired_candidate"
    return "unbalanced_candidate"


def point_regions(
    points_xy: np.ndarray,
    boundary_xy: np.ndarray | None,
    center_xy: np.ndarray | None,
    lengths_xy: np.ndarray,
    *,
    tpcl_half_width_A: float,
) -> np.ndarray:
    labels = np.full(len(points_xy), "unmapped", dtype="U16")
    if boundary_xy is None or center_xy is None or len(boundary_xy) < 3:
        return labels
    masks = classify_regions(
        points_xy,
        boundary_xy,
        center_xy,
        lengths_xy,
        tpcl_half_width=tpcl_half_width_A,
    )
    for region in REGIONS:
        labels[masks[region]] = region
    return labels


def extract_episodes(
    rows: Sequence[dict], *, key_columns: Sequence[str], min_persistence_frames: int
) -> tuple[list[dict], list[dict]]:
    """Split candidate rows into consecutive same-key sampled episodes."""

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[column] for column in key_columns)].append(row)
    episodes = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: row["frame_index"])
        chunks, chunk = [], [values[0]]
        for row in values[1:]:
            if row["frame_index"] == chunk[-1]["frame_index"] + 1:
                chunk.append(row)
            else:
                chunks.append(chunk); chunk = [row]
        chunks.append(chunk)
        for index, part in enumerate(chunks):
            first, last = part[0], part[-1]
            event = {column: value for column, value in zip(key_columns, key)}
            event.update(
                {
                    "episode_index": index,
                    "start_step": first["step"],
                    "end_step": last["step"],
                    "start_time_ns": first["time_ns"],
                    "end_time_ns": last["time_ns"],
                    "sample_count": len(part),
                    "observed_span_ps": round(
                        1000.0 * (last["time_ns"] - first["time_ns"]), 6
                    ),
                    "regions": ";".join(dict.fromkeys(row.get("region", "unmapped") for row in part)),
                    "max_original_h_in_water_count": max(
                        int(row.get("original_h_in_water_count", 0)) for row in part
                    ),
                    "contact_line_radius_start_A": first.get("contact_line_radius_A", ""),
                    "contact_line_radius_end_A": last.get("contact_line_radius_A", ""),
                    "contact_line_center_displacement_start_A": first.get(
                        "contact_line_center_displacement_A", ""
                    ),
                    "contact_line_center_displacement_end_A": last.get(
                        "contact_line_center_displacement_A", ""
                    ),
                    "persistent_candidate": len(part) >= min_persistence_frames,
                }
            )
            episodes.append(event)
    return episodes, [row for row in episodes if row["persistent_candidate"]]


def _write_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _contact_values(contact: dict | None) -> dict:
    if contact is None:
        return {"contact_line_radius_A": "", "contact_line_center_displacement_A": ""}
    return {
        "contact_line_radius_A": float(contact["contact_line_equivalent_radius_A"]),
        "contact_line_center_displacement_A": float(contact["contact_line_center_displacement_A"]),
    }


def write_plot(rows: Sequence[dict], output: Path, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    time = [row["time_ns"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.3), sharex=True)
    axes[0].plot(
        time,
        [row["top_sioh_deprotonated_candidate_count"] for row in rows],
        label="SiOH H-loss",
    )
    axes[0].plot(
        time,
        [row["top_sioh_proton_exchanged_candidate_count"] for row in rows],
        "--",
        label="SiOH H-exchange",
    )
    axes[0].plot(
        time,
        [row["top_ch3_c_h_loss_candidate_count"] for row in rows],
        ":",
        label=r"CH$_3$ H-loss",
    )
    axes[0].set_ylabel("Surface-site candidates")
    axes[1].plot(
        time,
        [row["solution_h3o_candidate_count"] for row in rows],
        label=r"H$_3$O-like",
    )
    axes[1].plot(
        time,
        [row["solution_oh_candidate_count"] for row in rows],
        "--",
        label=r"OH-like",
    )
    axes[1].set_ylabel("Solution ion candidates")
    axes[1].set_xlabel("Time (ns)")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    elements, initial_coordinates, initial_lengths = read_extxyz_positions(args.initial_xyz)
    sites = identify_initial_donor_sites(
        elements,
        initial_coordinates,
        initial_lengths,
        slab_range=args.surface_range,
        surface_z_A=args.surface_z_A,
        surface_depth_A=args.surface_depth_A,
        oh_cutoff_A=args.oh_cutoff_A,
        ch_cutoff_A=args.ch_cutoff_A,
    )
    site_by_id = {int(site["atom_id"]): site for site in sites}
    initial_origin_by_h = {
        int(hydrogen_id): site["site_type"]
        for site in sites
        for hydrogen_id in site["_initial_hydrogen_ids"]
    }
    contact_rows, boundaries = read_contact_lines(args.contact_line, args.contact_line_points)
    contacts = {int(row["step"]): row for row in contact_rows}
    surface_reference = load_surface_reference(
        args.initial_xyz, args.surface_range, args.surface_z_A
    )
    slab_start, slab_end = args.surface_range
    slab_elements = elements[slab_start - 1 : slab_end]
    surface_ids = np.arange(slab_start, slab_end + 1)
    carbon_mask = slab_elements == "C"
    carbon_ids = surface_ids[carbon_mask]
    records = {}
    raw_frames = 0
    for trajectory in args.trajectory:
        for frame in iter_orientation_frames(
            trajectory,
            args.surface_range,
            args.water_range,
            oxygen_type=args.oxygen_type,
            hydrogen_type=args.hydrogen_type,
        ):
            raw_frames += 1
            carbon = frame.surface[carbon_mask]
            assignment = assign_hydrogens_to_oxygen_or_carbon(
                frame.candidate_oxygen_ids,
                frame.candidate_oxygen,
                carbon_ids,
                carbon,
                frame.hydrogen_ids,
                frame.hydrogen,
                frame.bounds,
                oh_cutoff_A=args.oh_cutoff_A,
                ch_cutoff_A=args.ch_cutoff_A,
            )
            grouped = hydrogen_ids_by_owner(assignment)
            owner_by_hydrogen_id = {
                int(hydrogen_id): int(owner_id)
                for hydrogen_id, owner_id in zip(
                    assignment.hydrogen_ids, assignment.owner_ids
                )
            }
            contact = contacts.get(frame.step)
            boundary = boundaries.get(frame.step)
            center = None if contact is None else np.asarray(
                [float(contact["contact_line_center_x_A"]), float(contact["contact_line_center_y_A"])]
            )
            lengths = box_lengths(frame.bounds)
            site_ids = np.asarray(sorted(site_by_id), dtype=int)
            site_points = frame.surface[site_ids - slab_start]
            site_regions = point_regions(
                site_points[:, :2], boundary, center, lengths[:2],
                tpcl_half_width_A=args.tpcl_half_width_A,
            )
            water_id_set = set(map(int, frame.oxygen_ids))
            contact_values = _contact_values(contact)
            site_candidates = []
            site_state_counts: dict[str, int] = defaultdict(int)
            site_region_counts: dict[str, int] = defaultdict(int)
            for site_id, point, region in zip(site_ids, site_points, site_regions):
                site = site_by_id[int(site_id)]
                current_h = grouped.get(int(site_id), ())
                state = classify_site_state(
                    site["site_type"],
                    int(site["expected_h_count"]),
                    site["_initial_hydrogen_ids"],
                    current_h,
                )
                site_state_counts[f"{site['site_type']}_{state}"] += 1
                site_region_counts[f"{region}_{site['site_type']}_all"] += 1
                site_region_counts[f"{region}_{site['site_type']}_{state}"] += 1
                if state == "nominal":
                    continue
                row = {
                    "step": frame.step,
                    "time_ns": frame.step * args.timestep_fs / 1.0e6,
                    "site_atom_id": int(site_id),
                    "site_type": site["site_type"],
                    "state": state,
                    "current_h_count": len(current_h),
                    "initial_hydrogen_ids": site["initial_hydrogen_ids"],
                    "current_hydrogen_ids": ";".join(map(str, current_h)),
                    "original_h_in_water_count": sum(
                        owner_by_hydrogen_id[hydrogen_id] in water_id_set
                        for hydrogen_id in site["_initial_hydrogen_ids"]
                    ),
                    "region": str(region),
                    "x_A": float(point[0]),
                    "y_A": float(point[1]),
                    "z_A": float(point[2]),
                    **contact_values,
                }
                site_candidates.append(row)

            owner_count = {owner_id: len(values) for owner_id, values in grouped.items()}
            water_counts = np.asarray([owner_count.get(int(atom_id), 0) for atom_id in frame.oxygen_ids])
            ion_mask = water_counts != 2
            ion_ids = frame.oxygen_ids[ion_mask]
            ion_points = frame.oxygen[ion_mask]
            ion_counts = water_counts[ion_mask]
            ion_regions = point_regions(
                ion_points[:, :2], boundary, center, lengths[:2],
                tpcl_half_width_A=args.tpcl_half_width_A,
            )
            surface_z = surface_reference.plane_z(frame.surface, frame.bounds)
            ion_candidates = []
            for atom_id, point, count, region in zip(ion_ids, ion_points, ion_counts, ion_regions):
                current_h = grouped.get(int(atom_id), ())
                origin_types = [initial_origin_by_h.get(hydrogen_id, "other") for hydrogen_id in current_h]
                relative_z = minimum_image_vectors(
                    np.asarray([[0.0, 0.0, point[2] - surface_z]]), lengths
                )[0, 2]
                ion_candidates.append(
                    {
                        "step": frame.step,
                        "time_ns": frame.step * args.timestep_fs / 1.0e6,
                        "oxygen_atom_id": int(atom_id),
                        "assigned_h_count": int(count),
                        "species_candidate": (
                            "OH-like" if count == 1 else "H3O-like" if count == 3 else "other"
                        ),
                        "hydrogen_ids": ";".join(map(str, current_h)),
                        "initial_sioh_h_count": origin_types.count("SiOH"),
                        "initial_ch3_h_count": origin_types.count("CH3"),
                        "region": str(region),
                        "x_A": float(point[0]),
                        "y_A": float(point[1]),
                        "surface_relative_z_A": float(relative_z),
                        **contact_values,
                    }
                )
            h3o_count = int(np.count_nonzero(water_counts == 3))
            oh_count = int(np.count_nonzero(water_counts == 1))
            frame_row = {
                "step": frame.step,
                "time_ns": frame.step * args.timestep_fs / 1.0e6,
                "surface_reference_z_A": surface_z,
                "top_sioh_nominal_count": site_state_counts["SiOH_nominal"],
                "top_sioh_deprotonated_candidate_count": site_state_counts[
                    "SiOH_deprotonated_candidate"
                ],
                "top_sioh_proton_exchanged_candidate_count": site_state_counts[
                    "SiOH_proton_exchanged_candidate"
                ],
                "top_sioh_hyperprotonated_candidate_count": site_state_counts[
                    "SiOH_hyperprotonated_candidate"
                ],
                "top_ch3_nominal_count": site_state_counts["CH3_nominal"],
                "top_ch3_c_h_loss_candidate_count": site_state_counts[
                    "CH3_c_h_loss_candidate"
                ],
                "top_ch3_hydrogen_exchanged_candidate_count": site_state_counts[
                    "CH3_hydrogen_exchanged_candidate"
                ],
                "top_ch3_c_h_gain_candidate_count": site_state_counts[
                    "CH3_c_h_gain_candidate"
                ],
                "solution_h2o_candidate_count": int(np.count_nonzero(water_counts == 2)),
                "solution_oh_candidate_count": oh_count,
                "solution_h3o_candidate_count": h3o_count,
                "solution_zero_h_oxygen_count": int(np.count_nonzero(water_counts == 0)),
                "solution_four_plus_h_oxygen_count": int(np.count_nonzero(water_counts >= 4)),
                "solution_excess_proton_count": int(np.sum(water_counts - 2)),
                "surface_origin_h_in_water_count": int(
                    sum(
                        initial_origin_by_h.get(int(hydrogen_id)) in {"SiOH", "CH3"}
                        and int(owner_id) in water_id_set
                        for hydrogen_id, owner_id in zip(
                            assignment.hydrogen_ids, assignment.owner_ids
                        )
                    )
                ),
                "unassigned_hydrogen_count": int(np.count_nonzero(assignment.owner_ids < 0)),
                "ion_pair_state": classify_ion_pair_state(h3o_count, oh_count),
                **contact_values,
            }
            for region in REGIONS:
                sioh_total = site_region_counts[f"{region}_SiOH_all"]
                deprotonated = site_region_counts[
                    f"{region}_SiOH_deprotonated_candidate"
                ]
                exchanged = site_region_counts[
                    f"{region}_SiOH_proton_exchanged_candidate"
                ]
                frame_row[f"{region}_sioh_site_count"] = sioh_total
                frame_row[f"{region}_sioh_deprotonated_candidate_count"] = deprotonated
                frame_row[f"{region}_sioh_proton_exchanged_candidate_count"] = exchanged
                frame_row[f"{region}_sioh_deprotonated_candidate_fraction"] = (
                    deprotonated / sioh_total if sioh_total else math.nan
                )
                frame_row[f"{region}_h3o_candidate_count"] = sum(
                    row["species_candidate"] == "H3O-like" and row["region"] == region
                    for row in ion_candidates
                )
                frame_row[f"{region}_oh_candidate_count"] = sum(
                    row["species_candidate"] == "OH-like" and row["region"] == region
                    for row in ion_candidates
                )
            records[frame.step] = frame_row, site_candidates, ion_candidates

    steps = sorted(records)
    if args.drop_first_frame and steps:
        steps = steps[1:]
    if not steps:
        raise ValueError("No trajectory frames remain")
    frames, site_candidates, ion_candidates = [], [], []
    for frame_index, step in enumerate(steps):
        frame, site_rows, ion_rows = records[step]
        frame["frame_index"] = frame_index; frames.append(frame)
        for row in site_rows:
            row["frame_index"] = frame_index; site_candidates.append(row)
        for row in ion_rows:
            row["frame_index"] = frame_index; ion_candidates.append(row)
    site_episodes, persistent_site_episodes = extract_episodes(
        site_candidates,
        key_columns=("site_atom_id", "site_type", "state"),
        min_persistence_frames=args.min_persistence_frames,
    ) if site_candidates else ([], [])
    solution_ion_rows = [
        {
            **row,
            "original_h_in_water_count": row["initial_sioh_h_count"]
            + row["initial_ch3_h_count"],
        }
        for row in ion_candidates
        if row["species_candidate"] in {"H3O-like", "OH-like"}
    ]
    ion_episodes, persistent_ion_episodes = extract_episodes(
        solution_ion_rows,
        key_columns=("oxygen_atom_id", "species_candidate"),
        min_persistence_frames=args.min_persistence_frames,
    ) if solution_ion_rows else ([], [])
    pair_rows = [
        {**row, "region": "system", "original_h_in_water_count": row["surface_origin_h_in_water_count"]}
        for row in frames if row["ion_pair_state"] != "neutral"
    ]
    pair_episodes, persistent_pair_episodes = extract_episodes(
        pair_rows,
        key_columns=("ion_pair_state",),
        min_persistence_frames=args.min_persistence_frames,
    ) if pair_rows else ([], [])

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    public_sites = [{key: value for key, value in site.items() if not key.startswith("_")} for site in sites]
    _write_rows(output / "initial_surface_donor_sites.csv", public_sites, public_sites[0].keys())
    _write_rows(output / "proton_transfer_by_frame.csv", frames, frames[0].keys())
    site_fields = (
        "frame_index", "step", "time_ns", "site_atom_id", "site_type", "state",
        "current_h_count", "initial_hydrogen_ids", "current_hydrogen_ids",
        "original_h_in_water_count", "region", "x_A", "y_A", "z_A",
        "contact_line_radius_A", "contact_line_center_displacement_A",
    )
    ion_fields = (
        "frame_index", "step", "time_ns", "oxygen_atom_id", "assigned_h_count",
        "species_candidate", "hydrogen_ids", "initial_sioh_h_count", "initial_ch3_h_count",
        "region", "x_A", "y_A", "surface_relative_z_A", "contact_line_radius_A",
        "contact_line_center_displacement_A",
    )
    episode_fields = tuple(site_episodes[0].keys()) if site_episodes else (
        "site_atom_id", "site_type", "state", "episode_index", "start_step", "end_step",
        "start_time_ns", "end_time_ns", "sample_count", "observed_span_ps", "regions",
        "max_original_h_in_water_count", "contact_line_radius_start_A",
        "contact_line_radius_end_A", "contact_line_center_displacement_start_A",
        "contact_line_center_displacement_end_A", "persistent_candidate",
    )
    pair_episode_fields = tuple(pair_episodes[0].keys()) if pair_episodes else (
        "ion_pair_state", "episode_index", "start_step", "end_step", "start_time_ns",
        "end_time_ns", "sample_count", "observed_span_ps", "regions",
        "max_original_h_in_water_count", "contact_line_radius_start_A",
        "contact_line_radius_end_A", "contact_line_center_displacement_start_A",
        "contact_line_center_displacement_end_A", "persistent_candidate",
    )
    ion_episode_fields = tuple(ion_episodes[0].keys()) if ion_episodes else (
        "oxygen_atom_id", "species_candidate", "episode_index", "start_step",
        "end_step", "start_time_ns", "end_time_ns", "sample_count",
        "observed_span_ps", "regions", "max_original_h_in_water_count",
        "contact_line_radius_start_A", "contact_line_radius_end_A",
        "contact_line_center_displacement_start_A",
        "contact_line_center_displacement_end_A", "persistent_candidate",
    )
    _write_rows(output / "surface_site_candidates.csv", site_candidates, site_fields)
    _write_rows(output / "solution_ion_candidates.csv", ion_candidates, ion_fields)
    _write_rows(output / "surface_site_episodes.csv", site_episodes, episode_fields)
    _write_rows(output / "persistent_surface_site_events.csv", persistent_site_episodes, episode_fields)
    _write_rows(output / "solution_ion_episodes.csv", ion_episodes, ion_episode_fields)
    _write_rows(
        output / "persistent_solution_ion_events.csv",
        persistent_ion_episodes,
        ion_episode_fields,
    )
    _write_rows(output / "solution_ion_pair_episodes.csv", pair_episodes, pair_episode_fields)
    _write_rows(output / "persistent_solution_ion_pair_events.csv", persistent_pair_episodes, pair_episode_fields)

    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "analyzed_frames": len(frames),
        "first_step": frames[0]["step"],
        "last_step": frames[-1]["step"],
        "frame_interval_ps": (
            (frames[1]["step"] - frames[0]["step"]) * args.timestep_fs / 1000.0
            if len(frames) > 1 else None
        ),
        "top_sioh_site_count": sum(site["site_type"] == "SiOH" for site in sites),
        "top_ch3_site_count": sum(site["site_type"] == "CH3" for site in sites),
        "frames_with_surface_site_candidate": len({row["step"] for row in site_candidates}),
        "surface_site_candidate_episode_count": len(site_episodes),
        "persistent_surface_site_event_count": len(persistent_site_episodes),
        "frames_with_h3o_candidate": sum(row["solution_h3o_candidate_count"] > 0 for row in frames),
        "frames_with_oh_candidate": sum(row["solution_oh_candidate_count"] > 0 for row in frames),
        "solution_ion_candidate_episode_count": len(ion_episodes),
        "persistent_solution_ion_candidate_event_count": len(persistent_ion_episodes),
        "system_ion_pair_occupancy_episode_count": len(pair_episodes),
        "persistent_system_ion_pair_occupancy_episode_count": len(
            persistent_pair_episodes
        ),
        "solution_ion_pair_episode_count": len(pair_episodes),
        "persistent_solution_ion_pair_event_count": len(persistent_pair_episodes),
        "maximum_h3o_candidate_count": max(row["solution_h3o_candidate_count"] for row in frames),
        "maximum_oh_candidate_count": max(row["solution_oh_candidate_count"] for row in frames),
        "maximum_surface_origin_h_in_water_count": max(
            row["surface_origin_h_in_water_count"] for row in frames
        ),
        "min_persistence_frames": args.min_persistence_frames,
        "persistent_event_is_sampled_geometry_candidate_not_chemical_identity": True,
        "sub_frame_proton_transfer_not_resolved": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "initial_xyz": str(Path(args.initial_xyz).resolve()),
        "surface_atom_range": list(args.surface_range),
        "water_atom_range": list(args.water_range),
        "oxygen_type": args.oxygen_type,
        "hydrogen_type": args.hydrogen_type,
        "oh_cutoff_A": args.oh_cutoff_A,
        "ch_cutoff_A": args.ch_cutoff_A,
        "surface_depth_A": args.surface_depth_A,
        "tpcl_half_width_A": args.tpcl_half_width_A,
        "min_persistence_frames": args.min_persistence_frames,
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
        "drop_first_frame": args.drop_first_frame,
        "surface_event_definition": "consecutive sampled frames with the same non-nominal surface-site identity and state",
        "solution_ion_event_definition": "consecutive sampled frames with the same oxygen atom ID and H3O-like or OH-like state",
        "system_pair_occupancy_definition": "consecutive sampled frames containing at least one H3O-like and one OH-like candidate; this does not track an ion-pair identity",
        "chemical_identity_limit": "coordination candidates require finer-cadence and bond-order validation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(frames, output / "surface_proton_transfer.png", args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-xyz", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--contact-line", type=Path, required=True)
    parser.add_argument("--contact-line-points", type=Path, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--surface-depth-A", type=float, default=3.0)
    parser.add_argument("--tpcl-half-width-A", type=float, default=4.0)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.25)
    parser.add_argument("--ch-cutoff-A", type=float, default=1.30)
    parser.add_argument("--min-persistence-frames", type=int, default=2)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument(
        "--drop-first-frame", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.surface_depth_A,
        args.tpcl_half_width_A,
        args.oh_cutoff_A,
        args.ch_cutoff_A,
        args.min_persistence_frames,
    ) <= 0:
        raise ValueError("Cutoffs, depth, width, and persistence must be positive")
    print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
