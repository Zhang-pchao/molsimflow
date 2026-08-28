"""Pre-contact N2 z distributions split into main-bubble and disconnected molecules."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors, periodic_center
from molsimflow.postprocess.nanobubble_attachment import (
    iter_selected_frames,
    largest_cluster,
    molecule_centers,
    parse_range,
)
from molsimflow.postprocess.surface_reference import load_surface_reference

CATEGORIES = ("all", "main_bubble", "disconnected")


def frame_metrics(
    surface_relative_z: np.ndarray,
    main_members: np.ndarray,
    *,
    near_z_min: float,
    near_z_max: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    main = np.zeros(len(surface_relative_z), dtype=bool)
    main[main_members] = True
    near = (surface_relative_z >= near_z_min) & (surface_relative_z < near_z_max)
    masks = {"all": np.ones(len(main), dtype=bool), "main_bubble": main, "disconnected": ~main}
    values = {category: surface_relative_z[mask] for category, mask in masks.items()}
    row = {
        "n2_molecule_count": len(main),
        "main_bubble_n2_count": int(np.count_nonzero(main)),
        "disconnected_n2_count": int(np.count_nonzero(~main)),
    }
    for category, mask in masks.items():
        row[f"near_surface_{category}_count"] = int(np.count_nonzero(near & mask))
    return row, values


def projection_counts(
    centers: np.ndarray,
    surface_relative_z: np.ndarray,
    main_members: np.ndarray,
    bounds: np.ndarray,
    *,
    near_z_min: float,
    near_z_max: float,
    margin: float,
) -> dict:
    main = np.zeros(len(centers), dtype=bool)
    main[main_members] = True
    center = periodic_center(centers[main], bounds)
    radial = np.linalg.norm(
        minimum_image_vectors(centers - center, box_lengths(bounds))[:, :2], axis=1
    )
    projection_radius = float(np.quantile(radial[main], 0.90))
    near_disconnected = (
        (~main)
        & (surface_relative_z >= near_z_min)
        & (surface_relative_z < near_z_max)
    )
    outside = near_disconnected & (radial > projection_radius + margin)
    return {
        "main_bubble_lateral_radius_p90_A": projection_radius,
        "near_surface_disconnected_inside_projection_count": int(
            np.count_nonzero(near_disconnected & ~outside)
        ),
        "near_surface_disconnected_outside_projection_count": int(np.count_nonzero(outside)),
    }


def write_plot(
    profiles: Sequence[dict], frames: Sequence[dict], output: Path, font_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.5))
    for category in CATEGORIES:
        rows = [row for row in profiles if row["category"] == category]
        axes[0].plot(
            [row["z_A"] for row in rows],
            [row["number_density_A-3"] for row in rows],
            label=category.replace("_", " "),
        )
    axes[0].set_xlabel("Height above surface (Å)")
    axes[0].set_ylabel(r"N$_2$ density (Å$^{-3}$)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        [row["time_ns"] for row in frames],
        [row["near_surface_main_bubble_count"] for row in frames],
        label="main bubble",
    )
    axes[1].plot(
        [row["time_ns"] for row in frames],
        [row["near_surface_disconnected_count"] for row in frames],
        label="disconnected",
    )
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel(r"Near-surface N$_2$ count")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    surface_reference = load_surface_reference(
        args.reference_structure, args.surface_range, args.surface_z_A
    )
    end_step = round(args.end_ns * 1.0e6 / args.timestep_fs)
    records = {}
    raw_frames = 0
    for trajectory in args.trajectory:
        for frame in iter_selected_frames(trajectory, args.surface_range, args.nitrogen_range):
            raw_frames += 1
            if frame.step > end_step:
                continue
            centers = molecule_centers(frame.nitrogen, frame.bounds)
            members = largest_cluster(centers, frame.bounds, args.cluster_cutoff_A)
            surface_z = surface_reference.plane_z(frame.surface, frame.bounds)
            lengths = box_lengths(frame.bounds)
            z = minimum_image_vectors(
                np.column_stack(
                    (np.zeros(len(centers)), np.zeros(len(centers)), centers[:, 2] - surface_z)
                ),
                lengths,
            )[:, 2]
            row, values = frame_metrics(
                z,
                members,
                near_z_min=args.near_z_min_A,
                near_z_max=args.near_z_max_A,
            )
            row.update(
                projection_counts(
                    centers,
                    z,
                    members,
                    frame.bounds,
                    near_z_min=args.near_z_min_A,
                    near_z_max=args.near_z_max_A,
                    margin=args.projection_margin_A,
                )
            )
            row.update(
                {
                    "step": frame.step,
                    "time_ns": frame.step * args.timestep_fs / 1.0e6,
                    "surface_reference_z_A": surface_z,
                    "main_bubble_min_surface_dz_A": float(np.min(values["main_bubble"])),
                }
            )
            records[frame.step] = row, values, frame.bounds
    steps = sorted(records)
    if args.drop_first_frame and steps:
        steps = steps[1:]
    if not steps:
        raise ValueError("No pre-contact frames remain")
    reference_bounds = records[steps[0]][2]
    if any(not np.allclose(records[step][2], reference_bounds) for step in steps):
        raise ValueError("N2 density analysis requires a constant orthorhombic box")
    frames = [records[step][0] for step in steps]
    edges = np.arange(args.z_min_A, args.z_max_A + args.dz_A, args.dz_A)
    volume = np.prod(box_lengths(reference_bounds)[:2]) * np.diff(edges) * len(steps)
    profiles = []
    for category in CATEGORIES:
        counts = np.sum(
            [np.histogram(records[step][1][category], bins=edges)[0] for step in steps], axis=0
        )
        for index, count in enumerate(counts):
            profiles.append(
                {
                    "category": category,
                    "z_A": 0.5 * (edges[index] + edges[index + 1]),
                    "count": int(count),
                    "sampled_volume_A3": float(volume[index]),
                    "number_density_A-3": float(count / volume[index]),
                }
            )
    block_rows = []
    for block_index, start in enumerate(range(0, len(steps), args.block_frames)):
        block_steps = steps[start : start + args.block_frames]
        for category in CATEGORIES:
            counts = np.sum(
                [np.histogram(records[step][1][category], bins=edges)[0] for step in block_steps],
                axis=0,
            )
            block_volume = np.prod(box_lengths(reference_bounds)[:2]) * np.diff(edges) * len(block_steps)
            for index, count in enumerate(counts):
                block_rows.append(
                    {
                        "block_index": block_index,
                        "start_time_ns": records[block_steps[0]][0]["time_ns"],
                        "end_time_ns": records[block_steps[-1]][0]["time_ns"],
                        "category": category,
                        "z_A": 0.5 * (edges[index] + edges[index + 1]),
                        "number_density_A-3": float(count / block_volume[index]),
                    }
                )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("precontact_n2_by_frame.csv", frames),
        ("n2_z_profiles.csv", profiles),
        ("n2_time_z.csv", block_rows),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    disconnected_near = np.asarray(
        [row["near_surface_disconnected_count"] for row in frames], dtype=int
    )
    outside_projection = np.asarray(
        [row["near_surface_disconnected_outside_projection_count"] for row in frames],
        dtype=int,
    )
    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "analyzed_frames": len(frames),
        "first_step": frames[0]["step"],
        "last_step": frames[-1]["step"],
        "precontact_end_ns": args.end_ns,
        "near_surface_z_range_A": [args.near_z_min_A, args.near_z_max_A],
        "frames_with_near_surface_disconnected_n2": int(np.count_nonzero(disconnected_near)),
        "maximum_near_surface_disconnected_n2": int(np.max(disconnected_near)),
        "maximum_total_disconnected_n2": max(row["disconnected_n2_count"] for row in frames),
        "independent_pre_enrichment_observed": bool(np.any(disconnected_near > 0)),
        "frames_with_disconnected_n2_outside_bubble_projection": int(
            np.count_nonzero(outside_projection)
        ),
        "maximum_disconnected_n2_outside_bubble_projection": int(
            np.max(outside_projection)
        ),
        "outside_projection_is_stricter_pre_enrichment_candidate": True,
        "surface_attraction_requires_cross_surface_comparison": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "surface_atom_range": list(args.surface_range),
        "nitrogen_atom_range": list(args.nitrogen_range),
        "surface_reference_mode": "dynamic_slab_translation",
        "reference_structure": str(args.reference_structure.resolve()),
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
        "drop_first_frame": args.drop_first_frame,
        "cluster_cutoff_A": args.cluster_cutoff_A,
        "projection_margin_A": args.projection_margin_A,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(profiles, frames, output / "precontact_n2_enrichment.png", args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--nitrogen-range", type=parse_range, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--reference-structure", type=Path, required=True)
    parser.add_argument("--end-ns", type=float, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--cluster-cutoff-A", type=float, default=5.5)
    parser.add_argument("--near-z-min-A", type=float, default=0.0)
    parser.add_argument("--near-z-max-A", type=float, default=10.0)
    parser.add_argument("--projection-margin-A", type=float, default=5.0)
    parser.add_argument("--z-min-A", type=float, default=0.0)
    parser.add_argument("--z-max-A", type=float, default=80.0)
    parser.add_argument("--dz-A", type=float, default=1.0)
    parser.add_argument("--block-frames", type=int, default=50)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--drop-first-frame", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.end_ns,
        args.cluster_cutoff_A,
        args.projection_margin_A,
        args.dz_A,
        args.block_frames,
    ) <= 0:
        raise ValueError("Time, cutoffs, bin width, and block size must be positive")
    if args.near_z_max_A <= args.near_z_min_A or args.z_max_A <= args.z_min_A:
        raise ValueError("Invalid z range")
    print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
