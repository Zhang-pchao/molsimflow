"""Restart-aware PBC analysis of nanodroplet position and spreading."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors, periodic_center
from molsimflow.postprocess.surface_reference import load_surface_reference


@dataclass(frozen=True)
class DropletFrame:
    source: Path
    source_frame: int
    step: int
    bounds: np.ndarray
    surface: np.ndarray
    water_types: np.ndarray
    water: np.ndarray


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Atom range must be START:END")
    start, end = map(int, parts)
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Atom range must satisfy 1 <= START <= END")
    return start, end


def iter_selected_frames(
    path: Path, surface_range: tuple[int, int], water_range: tuple[int, int]
) -> Iterator[DropletFrame]:
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
                raise ValueError(f"Missing atom-count header at step {step}")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing orthorhombic box at step {step}")
            bounds = np.array([list(map(float, handle.readline().split()[:2])) for _ in range(3)])
            fields = handle.readline().split()[2:]
            index = {name: i for i, name in enumerate(fields)}
            missing = {"id", "type", "x", "y", "z"}.difference(index)
            if missing:
                raise ValueError(f"Missing dump columns at step {step}: {sorted(missing)}")
            surface, water = [], []
            for _ in range(atom_count):
                values = handle.readline().split()
                atom_id = int(values[index["id"]])
                atom_type = int(values[index["type"]])
                xyz = [float(values[index[key]]) for key in ("x", "y", "z")]
                if surface_range[0] <= atom_id <= surface_range[1]:
                    surface.append((atom_id, *xyz))
                elif water_range[0] <= atom_id <= water_range[1]:
                    water.append((atom_id, atom_type, *xyz))
            surface_array = np.asarray(sorted(surface), dtype=float)[:, 1:]
            water_array = np.asarray(sorted(water), dtype=float)
            if len(water_array) != water_range[1] - water_range[0] + 1:
                raise ValueError(f"Incomplete water selection at step {step}")
            yield DropletFrame(
                Path(path), frame_index, step, bounds, surface_array,
                water_array[:, 1].astype(int), water_array[:, 2:],
            )
            frame_index += 1


def largest_cluster(coords: np.ndarray, bounds: np.ndarray, cutoff: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    shifted = (coords - bounds[:, 0]) % lengths
    parent = np.arange(len(coords))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for left, right in cKDTree(shifted, boxsize=lengths).query_pairs(cutoff):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    roots = np.array([find(i) for i in range(len(coords))])
    labels, counts = np.unique(roots, return_counts=True)
    return np.flatnonzero(roots == labels[np.argmax(counts)])


def periodic_weighted_center(coords: np.ndarray, weights: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lengths = box_lengths(bounds)
    center = np.empty(3)
    for dim in range(3):
        angles = 2.0 * np.pi * (coords[:, dim] - bounds[dim, 0]) / lengths[dim]
        value = np.sum(weights * np.exp(1j * angles)) / np.sum(weights)
        center[dim] = bounds[dim, 0] + lengths[dim] * (np.angle(value) % (2.0 * np.pi)) / (2.0 * np.pi)
    return center


def footprint_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError

    try:
        return float(ConvexHull(points[:, :2]).volume)
    except QhullError:
        return 0.0


def analyze_frame(
    frame: DropletFrame,
    *,
    oxygen_type: int,
    surface_z: float,
    cluster_cutoff: float,
    contact_cutoff: float,
) -> dict:
    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    oxygen = frame.water[frame.water_types == oxygen_type]
    members = largest_cluster(oxygen, frame.bounds, cluster_cutoff)
    droplet = oxygen[members]
    center = periodic_center(droplet, frame.bounds)
    vectors = minimum_image_vectors(droplet - center, lengths)
    radial = np.linalg.norm(vectors[:, :2], axis=1)
    dz_values = minimum_image_vectors(
        np.column_stack((np.zeros(len(droplet)), np.zeros(len(droplet)), droplet[:, 2] - surface_z)),
        lengths,
    )[:, 2]
    shifted_surface = (frame.surface - frame.bounds[:, 0]) % lengths
    shifted_oxygen = (oxygen - frame.bounds[:, 0]) % lengths
    distances = cKDTree(shifted_surface, boxsize=lengths).query(shifted_oxygen, k=1)[0]
    member_distances = distances[members]
    contact_vectors = vectors[member_distances <= contact_cutoff]
    masses = np.where(frame.water_types == oxygen_type, 15.999, 1.008)
    mass_center = periodic_weighted_center(frame.water, masses, frame.bounds)
    q05, q95 = np.quantile(dz_values, [0.05, 0.95])
    height = float(q95 - q05)
    radius = float(np.quantile(radial, 0.90))
    return {
        "source_file": str(frame.source), "source_frame": frame.source_frame, "step": frame.step,
        "droplet_oxygen_center_x_A": float(center[0]),
        "droplet_oxygen_center_y_A": float(center[1]),
        "droplet_oxygen_center_z_A": float(center[2]),
        "droplet_center_surface_dz_A": float(minimum_image_vectors(
            np.array([[0.0, 0.0, center[2] - surface_z]]), lengths
        )[0, 2]),
        "inserted_water_mass_com_x_A": float(mass_center[0]),
        "inserted_water_mass_com_y_A": float(mass_center[1]),
        "inserted_water_mass_com_z_A": float(mass_center[2]),
        "largest_cluster_water_count": len(members),
        "droplet_height_q05_q95_A": height,
        "droplet_lateral_radius_p90_A": radius,
        "spreading_aspect_ratio": radius / height if height > 0 else float("nan"),
        "contact_water_count": int(np.count_nonzero(member_distances <= contact_cutoff)),
        "footprint_convex_hull_area_A2": footprint_area(contact_vectors),
        "min_droplet_surface_distance_A": float(member_distances.min()),
    }


def add_unwrapped_centers(rows: list[dict], bounds: np.ndarray) -> None:
    lengths = box_lengths(bounds)
    for prefix in ("droplet_oxygen_center", "inserted_water_mass_com"):
        previous = None
        unwrapped = None
        for row in rows:
            current = np.array([row[f"{prefix}_{dim}_A"] for dim in "xyz"])
            if previous is None:
                unwrapped = current.copy()
            else:
                unwrapped += minimum_image_vectors(current - previous, lengths)
            for dim, value in zip("xyz", unwrapped):
                row[f"{prefix}_{dim}_unwrapped_A"] = float(value)
            previous = current


def write_plot(rows: Sequence[dict], output: Path, timestep_fs: float, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = properties.get_name()
    time = np.array([row["step"] for row in rows]) * timestep_fs / 1.0e6
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    axes[0].plot(time, [row["droplet_center_surface_dz_A"] for row in rows])
    axes[0].set_ylabel(r"Droplet center $z$ (Å)")
    axes[1].plot(time, [row["droplet_height_q05_q95_A"] for row in rows], label="Height")
    axes[1].plot(time, [row["droplet_lateral_radius_p90_A"] for row in rows], label="Lateral radius")
    axes[1].set_ylabel("Length (Å)"); axes[1].legend(frameon=False)
    axes[2].plot(time, [row["footprint_convex_hull_area_A2"] for row in rows])
    axes[2].set_ylabel(r"Footprint (Å$^2$)"); axes[2].set_xlabel("Time (ns)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    rows_by_step: dict[int, dict] = {}; raw_frames = 0; last_bounds = None; stop = False
    surface_reference = (
        load_surface_reference(args.reference_structure, args.surface_range, args.surface_z_A)
        if args.reference_structure is not None else None
    )
    for trajectory in args.trajectory:
        for frame in iter_selected_frames(trajectory, args.surface_range, args.water_range):
            raw_frames += 1; last_bounds = frame.bounds
            surface_z = (
                surface_reference.plane_z(frame.surface, frame.bounds)
                if surface_reference is not None else args.surface_z_A
            )
            row = analyze_frame(
                frame, oxygen_type=args.oxygen_type, surface_z=surface_z,
                cluster_cutoff=args.cluster_cutoff_A, contact_cutoff=args.contact_cutoff_A,
            )
            row["surface_reference_z_A"] = surface_z
            rows_by_step[frame.step] = row
            if args.max_frames is not None and raw_frames >= args.max_frames:
                stop = True; break
        if stop: break
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    if args.drop_first_frame: rows = rows[1:]
    if not rows or last_bounds is None: raise ValueError("No analyzed frames remain")
    add_unwrapped_centers(rows, last_bounds)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with (output / "droplet_spreading.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "PASS", "raw_frames": raw_frames,
        "unique_frames_before_drop": len(rows_by_step), "analyzed_frames": len(rows),
        "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
        "timestep_fs": args.timestep_fs,
        "frame_interval_ps": (rows[1]["step"] - rows[0]["step"]) * args.timestep_fs / 1000.0,
        "geometry_definition": "largest oxygen connectivity cluster; hydrogens are not assigned to fixed molecules",
        "surface_reference_mode": (
            "dynamic_slab_translation" if surface_reference is not None else "fixed_nominal_plane"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "surface_atom_range": list(args.surface_range), "water_atom_range": list(args.water_range),
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
        "drop_first_frame": args.drop_first_frame, "oxygen_type": args.oxygen_type,
        "cluster_cutoff_A": args.cluster_cutoff_A, "contact_cutoff_A": args.contact_cutoff_A,
        "reference_structure": (
            None if args.reference_structure is None else str(args.reference_structure.resolve())
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(rows, output / "droplet_spreading.png", args.timestep_fs, args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--reference-structure", type=Path)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--cluster-cutoff-A", type=float, default=3.5)
    parser.add_argument("--contact-cutoff-A", type=float, default=3.5)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--drop-first-frame", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run_analysis(build_parser().parse_args(argv)), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
