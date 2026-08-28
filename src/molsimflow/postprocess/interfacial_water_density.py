"""Water-oxygen density in footprint, TPCL, and far-field surface regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.nanodroplet_spreading import iter_selected_frames, parse_range
from molsimflow.postprocess.surface_reference import load_surface_reference
from molsimflow.postprocess.surface_site_enrichment import (
    point_segment_distances,
    read_contact_lines,
)

REGIONS = ("footprint", "tpcl", "far_field")


def classify_regions(
    points_xy: np.ndarray,
    boundary_xy: np.ndarray,
    center_xy: np.ndarray,
    lengths_xy: np.ndarray,
    *,
    tpcl_half_width: float,
) -> dict[str, np.ndarray]:
    from matplotlib.path import Path as PolygonPath

    local_points = minimum_image_vectors(points_xy - center_xy, lengths_xy)
    local_boundary = minimum_image_vectors(boundary_xy - center_xy, lengths_xy)
    inside = PolygonPath(local_boundary).contains_points(local_points, radius=1e-10)
    near = point_segment_distances(local_points, local_boundary) <= tpcl_half_width
    return {
        "footprint": inside & ~near,
        "tpcl": near,
        "far_field": ~inside & ~near,
    }


def area_grid(lengths_xy: np.ndarray, spacing: float) -> tuple[np.ndarray, float]:
    counts = np.ceil(lengths_xy / spacing).astype(int)
    widths = lengths_xy / counts
    axes = [(-0.5 * length + (np.arange(count) + 0.5) * width) for length, count, width in zip(lengths_xy, counts, widths)]
    x, y = np.meshgrid(*axes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel())), float(np.prod(widths))


def analyze_frame(
    coordinates: np.ndarray,
    bounds: np.ndarray,
    boundary_xy: np.ndarray,
    center_xy: np.ndarray,
    *,
    surface_z: float,
    tpcl_half_width: float,
    grid_spacing: float,
    z_edges: np.ndarray,
    hydration_z_max: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict]:
    lengths = box_lengths(bounds)
    water_masks = classify_regions(
        coordinates[:, :2],
        boundary_xy,
        center_xy,
        lengths[:2],
        tpcl_half_width=tpcl_half_width,
    )
    local_grid, cell_area = area_grid(lengths[:2], grid_spacing)
    local_boundary = minimum_image_vectors(boundary_xy - center_xy, lengths[:2])
    grid_masks = classify_regions(
        local_grid,
        local_boundary,
        np.zeros(2),
        lengths[:2],
        tpcl_half_width=tpcl_half_width,
    )
    areas = {region: float(np.count_nonzero(grid_masks[region]) * cell_area) for region in REGIONS}
    z = minimum_image_vectors(
        np.column_stack(
            (np.zeros(len(coordinates)), np.zeros(len(coordinates)), coordinates[:, 2] - surface_z)
        ),
        lengths,
    )[:, 2]
    counts = {
        region: np.histogram(z[water_masks[region]], bins=z_edges)[0] for region in REGIONS
    }
    frame_row = {"footprint_area_A2": areas["footprint"], "tpcl_area_A2": areas["tpcl"], "far_field_area_A2": areas["far_field"]}
    for region in REGIONS:
        hydration_count = int(np.count_nonzero(water_masks[region] & (z >= 0.0) & (z < hydration_z_max)))
        frame_row[f"{region}_hydration_oxygen_count"] = hydration_count
        frame_row[f"{region}_hydration_areal_density_A-2"] = (
            hydration_count / areas[region] if areas[region] > 0 else math.nan
        )
    return counts, areas, frame_row


def write_plot(
    profiles: Sequence[dict], frames: Sequence[dict], output: Path, timestep_fs: float, font_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.5))
    for region in REGIONS:
        selected = [row for row in profiles if row["region"] == region]
        axes[0].plot(
            [row["z_A"] for row in selected],
            [row["oxygen_number_density_A-3"] for row in selected],
            label=region.replace("_", " "),
        )
    axes[0].set_xlabel("Height above nominal surface (Å)")
    axes[0].set_ylabel(r"O number density (Å$^{-3}$)")
    axes[0].legend(frameon=False)
    time = np.asarray([row["step"] for row in frames]) * timestep_fs / 1.0e6
    for region in REGIONS:
        axes[1].plot(
            time,
            [row[f"{region}_hydration_areal_density_A-2"] for row in frames],
            label=region.replace("_", " "),
        )
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel(r"Hydration-layer O (Å$^{-2}$)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    contact_rows, boundaries = read_contact_lines(args.contact_line, args.contact_line_points)
    contacts = {int(row["step"]): row for row in contact_rows}
    z_edges = np.arange(args.z_min_A, args.z_max_A + args.dz_A, args.dz_A)
    records: dict[int, tuple[dict[str, np.ndarray], dict[str, float], dict]] = {}
    raw_frames = 0
    reference_bounds = None
    surface_reference = (
        load_surface_reference(args.reference_structure, args.surface_range, args.surface_z_A)
        if args.reference_structure is not None else None
    )
    for trajectory in args.trajectory:
        for frame in iter_selected_frames(trajectory, args.surface_range, args.water_range):
            raw_frames += 1
            if reference_bounds is None:
                reference_bounds = frame.bounds
            elif not np.allclose(frame.bounds, reference_bounds):
                raise ValueError("Water-density analysis requires a constant orthorhombic box")
            contact = contacts.get(frame.step)
            boundary = boundaries.get(frame.step)
            if contact is None or boundary is None:
                continue
            coordinates = frame.water[frame.water_types == args.oxygen_type]
            surface_z = (
                surface_reference.plane_z(frame.surface, frame.bounds)
                if surface_reference is not None else args.surface_z_A
            )
            center = np.asarray(
                [float(contact["contact_line_center_x_A"]), float(contact["contact_line_center_y_A"])]
            )
            counts, areas, row = analyze_frame(
                coordinates,
                frame.bounds,
                boundary,
                center,
                surface_z=surface_z,
                tpcl_half_width=args.tpcl_half_width_A,
                grid_spacing=args.area_grid_A,
                z_edges=z_edges,
                hydration_z_max=args.hydration_z_max_A,
            )
            row["step"] = frame.step
            row["surface_reference_z_A"] = surface_z
            records[frame.step] = counts, areas, row
    missing = sorted(set(contacts).difference(records))
    if missing:
        raise ValueError(f"Missing trajectory data for {len(missing)} contact-line steps")
    frames = [records[step][2] for step in sorted(records)]
    if not frames:
        raise ValueError("No matched water-density frames")
    bin_widths = np.diff(z_edges)
    profiles = []
    for region in REGIONS:
        total_counts = np.sum([records[step][0][region] for step in sorted(records)], axis=0)
        volumes = np.sum(
            [records[step][1][region] * bin_widths for step in sorted(records)], axis=0
        )
        if np.any(volumes <= 0):
            raise ValueError(f"Region {region} has zero accumulated volume")
        for index, count in enumerate(total_counts):
            profiles.append(
                {
                    "region": region,
                    "z_A": 0.5 * (z_edges[index] + z_edges[index + 1]),
                    "oxygen_count": int(count),
                    "sampled_volume_A3": float(volumes[index]),
                    "oxygen_number_density_A-3": float(count / volumes[index]),
                }
            )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with (output / "water_density_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0])); writer.writeheader(); writer.writerows(profiles)
    with (output / "hydration_by_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frames[0])); writer.writeheader(); writer.writerows(frames)
    summary = {
        "status": "PASS", "raw_frames": raw_frames, "matched_frames": len(frames),
        "first_step": frames[0]["step"], "last_step": frames[-1]["step"],
        "region_partition": "footprint core, two-sided TPCL band, and remaining far field",
        "hydration_layer_z_range_A": [0.0, args.hydration_z_max_A],
        "mean_hydration_areal_density_A-2": {
            region: float(np.nanmean([row[f"{region}_hydration_areal_density_A-2"] for row in frames]))
            for region in REGIONS
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "water_atom_range": list(args.water_range), "oxygen_type": args.oxygen_type,
        "contact_line": str(Path(args.contact_line).resolve()),
        "contact_line_points": str(Path(args.contact_line_points).resolve()),
        "surface_z_A": args.surface_z_A,
        "surface_reference_mode": (
            "dynamic_slab_translation" if surface_reference is not None else "fixed_nominal_plane"
        ),
        "reference_structure": (
            None if args.reference_structure is None else str(args.reference_structure.resolve())
        ),
        "surface_atom_range": list(args.surface_range), "tpcl_half_width_A": args.tpcl_half_width_A,
        "area_grid_A": args.area_grid_A, "z_range_A": [args.z_min_A, args.z_max_A],
        "dz_A": args.dz_A, "hydration_z_max_A": args.hydration_z_max_A,
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(profiles, frames, output / "interfacial_water_density.png", args.timestep_fs, args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--contact-line", type=Path, required=True)
    parser.add_argument("--contact-line-points", type=Path, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--reference-structure", type=Path)
    parser.add_argument("--tpcl-half-width-A", type=float, default=4.0)
    parser.add_argument("--area-grid-A", type=float, default=1.0)
    parser.add_argument("--z-min-A", type=float, default=0.0)
    parser.add_argument("--z-max-A", type=float, default=15.0)
    parser.add_argument("--dz-A", type=float, default=0.5)
    parser.add_argument("--hydration-z-max-A", type=float, default=6.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.tpcl_half_width_A, args.area_grid_A, args.dz_A, args.hydration_z_max_A) <= 0:
        raise ValueError("Grid and region widths must be positive")
    if args.z_max_A <= args.z_min_A or args.hydration_z_max_A > args.z_max_A:
        raise ValueError("Invalid z range or hydration-layer upper bound")
    print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
