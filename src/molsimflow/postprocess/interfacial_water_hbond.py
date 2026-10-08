"""Snapshot hydrogen bonds in first-layer footprint, TPCL, and far-field water."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.interfacial_water_density import REGIONS, classify_regions
from molsimflow.postprocess.interfacial_water_orientation import (
    assign_hydrogen_neighbors,
    iter_orientation_frames,
)
from molsimflow.postprocess.nanodroplet_spreading import parse_range
from molsimflow.postprocess.surface_reference import load_surface_reference
from molsimflow.postprocess.surface_site_enrichment import (
    identify_surface_sites,
    read_contact_lines,
)


def donor_points_to(
    oh_vectors: np.ndarray, donor_acceptor_vector: np.ndarray, angle_cutoff_deg: float
) -> bool:
    distance = np.linalg.norm(donor_acceptor_vector)
    if distance <= 0 or len(oh_vectors) == 0:
        return False
    oh_unit = oh_vectors / np.linalg.norm(oh_vectors, axis=1)[:, None]
    direction = donor_acceptor_vector / distance
    return bool(np.any(oh_unit @ direction >= math.cos(math.radians(angle_cutoff_deg))))


def water_water_hbond_count(
    oxygen: np.ndarray,
    oh_vectors: Sequence[np.ndarray],
    bounds: np.ndarray,
    *,
    oo_cutoff: float,
    angle_cutoff_deg: float,
) -> int:
    if len(oxygen) < 2:
        return 0
    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    shifted = (oxygen - bounds[:, 0]) % lengths
    edges = 0
    for first, second in cKDTree(shifted, boxsize=lengths).query_pairs(oo_cutoff):
        vector = minimum_image_vectors(oxygen[second] - oxygen[first], lengths)
        if donor_points_to(oh_vectors[first], vector, angle_cutoff_deg) or donor_points_to(
            oh_vectors[second], -vector, angle_cutoff_deg
        ):
            edges += 1
    return edges


def water_surface_hbond_counts(
    water: np.ndarray,
    water_oh: Sequence[np.ndarray],
    surface: np.ndarray,
    surface_oh: Sequence[np.ndarray],
    bounds: np.ndarray,
    *,
    oo_cutoff: float,
    angle_cutoff_deg: float,
) -> tuple[int, int]:
    if len(water) == 0 or len(surface) == 0:
        return 0, 0
    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    shifted_surface = (surface - bounds[:, 0]) % lengths
    tree = cKDTree(shifted_surface, boxsize=lengths)
    water_donor = surface_donor = 0
    for water_index, neighbors in enumerate(
        tree.query_ball_point((water - bounds[:, 0]) % lengths, oo_cutoff)
    ):
        for surface_index in neighbors:
            vector = minimum_image_vectors(surface[surface_index] - water[water_index], lengths)
            water_donor += donor_points_to(
                water_oh[water_index], vector, angle_cutoff_deg
            )
            surface_donor += donor_points_to(
                surface_oh[surface_index], -vector, angle_cutoff_deg
            )
    return int(water_donor), int(surface_donor)


def analyze_frame(
    frame,
    boundary_xy: np.ndarray,
    center_xy: np.ndarray,
    surface_sioh_ids: set[int],
    *,
    surface_z: float,
    tpcl_half_width: float,
    oh_cutoff: float,
    oo_cutoff: float,
    angle_cutoff_deg: float,
    z_min: float,
    z_max: float,
) -> dict:
    lengths = box_lengths(frame.bounds)
    z = minimum_image_vectors(
        np.column_stack(
            (np.zeros(len(frame.oxygen)), np.zeros(len(frame.oxygen)), frame.oxygen[:, 2] - surface_z)
        ),
        lengths,
    )[:, 2]
    layer = (z >= z_min) & (z < z_max)
    regions = classify_regions(
        frame.oxygen[:, :2], boundary_xy, center_xy, lengths[:2], tpcl_half_width=tpcl_half_width
    )
    assigned = assign_hydrogen_neighbors(
        frame.candidate_oxygen, frame.hydrogen, frame.bounds, oh_cutoff
    )
    water_assigned = assigned[: len(frame.oxygen)]
    candidate_index = {int(atom_id): index for index, atom_id in enumerate(frame.candidate_oxygen_ids)}
    surface_indices = [
        candidate_index[atom_id]
        for atom_id in surface_sioh_ids
        if atom_id in candidate_index and len(assigned[candidate_index[atom_id]]) == 1
    ]
    surface_coords = frame.candidate_oxygen[surface_indices]
    surface_vectors = [
        minimum_image_vectors(
            frame.hydrogen[assigned[index]] - frame.candidate_oxygen[index], lengths
        )
        for index in surface_indices
    ]
    row = {
        "step": frame.step,
        "surface_reference_z_A": surface_z,
        "protonated_top_sioh_count": len(surface_indices),
    }
    for region in REGIONS:
        selected = np.flatnonzero(
            layer & regions[region] & np.asarray([len(items) == 2 for items in water_assigned])
        )
        water = frame.oxygen[selected]
        water_vectors = [
            minimum_image_vectors(frame.hydrogen[water_assigned[index]] - frame.oxygen[index], lengths)
            for index in selected
        ]
        water_water = water_water_hbond_count(
            water,
            water_vectors,
            frame.bounds,
            oo_cutoff=oo_cutoff,
            angle_cutoff_deg=angle_cutoff_deg,
        )
        water_donor, surface_donor = water_surface_hbond_counts(
            water,
            water_vectors,
            surface_coords,
            surface_vectors,
            frame.bounds,
            oo_cutoff=oo_cutoff,
            angle_cutoff_deg=angle_cutoff_deg,
        )
        count = len(water)
        row[f"{region}_h2o_count"] = count
        row[f"{region}_water_water_hbond_count"] = water_water
        row[f"{region}_water_water_hbond_degree"] = (
            2.0 * water_water / count if count else math.nan
        )
        row[f"{region}_water_donor_sioh_count"] = water_donor
        row[f"{region}_sioh_donor_water_count"] = surface_donor
        row[f"{region}_surface_water_hbond_per_h2o"] = (
            (water_donor + surface_donor) / count if count else math.nan
        )
    return row


def write_plot(rows: Sequence[dict], output: Path, timestep_fs: float, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    time = np.asarray([row["step"] for row in rows]) * timestep_fs / 1.0e6
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    for region in REGIONS:
        axes[0].plot(
            time,
            [row[f"{region}_water_water_hbond_degree"] for row in rows],
            label=region.replace("_", " "),
        )
        axes[1].plot(
            time,
            [row[f"{region}_surface_water_hbond_per_h2o"] for row in rows],
            label=region.replace("_", " "),
        )
    axes[0].set_ylabel("Water H-bond degree")
    axes[1].set_ylabel("Surface-water H-bonds / H2O")
    axes[1].set_xlabel("Time (ns)")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def sampling_summary(rows: Sequence[dict]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for region in REGIONS:
        counts = np.asarray([row[f"{region}_h2o_count"] for row in rows], dtype=int)
        summary[region] = {
            "mean_h2o_count_per_frame": float(np.mean(counts)),
            "nonempty_frame_count": int(np.count_nonzero(counts)),
            "total_h2o_frame_samples": int(np.sum(counts)),
        }
    return summary


def run_analysis(args: argparse.Namespace) -> dict:
    elements, coordinates, lengths = read_extxyz_positions(args.reference_structure)
    sites = identify_surface_sites(
        elements,
        coordinates,
        lengths,
        slab_range=args.surface_range,
        surface_z=args.surface_z_A,
        surface_depth=args.surface_depth_A,
        bond_cutoff=args.oh_cutoff_A,
    )
    surface_sioh_ids = {row["atom_id"] for row in sites if row["site_type"] == "SiOH"}
    contact_rows, boundaries = read_contact_lines(args.contact_line, args.contact_line_points)
    contacts = {int(row["step"]): row for row in contact_rows}
    surface_reference = load_surface_reference(
        args.reference_structure, args.surface_range, args.surface_z_A
    )
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
            contact, boundary = contacts.get(frame.step), boundaries.get(frame.step)
            if contact is None or boundary is None:
                continue
            center = np.asarray(
                [float(contact["contact_line_center_x_A"]), float(contact["contact_line_center_y_A"])]
            )
            surface_z = surface_reference.plane_z(frame.surface, frame.bounds)
            records[frame.step] = analyze_frame(
                frame,
                boundary,
                center,
                surface_sioh_ids,
                surface_z=surface_z,
                tpcl_half_width=args.tpcl_half_width_A,
                oh_cutoff=args.oh_cutoff_A,
                oo_cutoff=args.oo_cutoff_A,
                angle_cutoff_deg=args.angle_cutoff_deg,
                z_min=args.z_min_A,
                z_max=args.z_max_A,
            )
    missing = sorted(set(contacts).difference(records))
    if missing:
        raise ValueError(f"Missing trajectory data for {len(missing)} contact-line steps")
    rows = [records[step] for step in sorted(records)]
    if not rows:
        raise ValueError("No matched H-bond frames")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with (output / "hbond_by_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "matched_frames": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "top_surface_sioh_site_count": len(surface_sioh_ids),
        "mean_metrics": {
            f"{region}_{metric}": float(np.nanmean([row[f"{region}_{metric}"] for row in rows]))
            for region in REGIONS
            for metric in ("water_water_hbond_degree", "surface_water_hbond_per_h2o")
        },
        "mean_metric_policy": "unweighted mean over frames with a nonempty region",
        "sampling": sampling_summary(rows),
        "frame_interval_ps": (
            (rows[1]["step"] - rows[0]["step"]) * args.timestep_fs / 1000.0
            if len(rows) > 1 else None
        ),
        "hbond_lifetime_not_resolved_at_this_dump_interval": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "surface_atom_range": list(args.surface_range),
        "water_atom_range": list(args.water_range),
        "surface_reference_mode": "dynamic_slab_translation",
        "reference_structure": str(args.reference_structure.resolve()),
        "contact_line": str(args.contact_line.resolve()),
        "contact_line_points": str(args.contact_line_points.resolve()),
        "oh_cutoff_A": args.oh_cutoff_A,
        "oo_cutoff_A": args.oo_cutoff_A,
        "angle_cutoff_deg": args.angle_cutoff_deg,
        "tpcl_half_width_A": args.tpcl_half_width_A,
        "first_layer_z_range_A": [args.z_min_A, args.z_max_A],
        "regional_water_water_edges_require_both_oxygens_in_same_region": True,
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(rows, output / "interfacial_water_hbond.png", args.timestep_fs, args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--contact-line", type=Path, required=True)
    parser.add_argument("--contact-line-points", type=Path, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--reference-structure", type=Path, required=True)
    parser.add_argument("--surface-depth-A", type=float, default=3.0)
    parser.add_argument("--tpcl-half-width-A", type=float, default=4.0)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.25)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    parser.add_argument("--angle-cutoff-deg", type=float, default=30.0)
    parser.add_argument("--z-min-A", type=float, default=0.0)
    parser.add_argument("--z-max-A", type=float, default=6.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.surface_depth_A,
        args.tpcl_half_width_A,
        args.oh_cutoff_A,
        args.oo_cutoff_A,
        args.angle_cutoff_deg,
    ) <= 0:
        raise ValueError("Cutoffs, depth, and angle must be positive")
    if args.z_max_A <= args.z_min_A:
        raise ValueError("First-layer z range is invalid")
    print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
