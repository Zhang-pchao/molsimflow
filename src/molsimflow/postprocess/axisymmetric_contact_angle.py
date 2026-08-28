"""Estimate axisymmetric density contours and spherical-cap contact angles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors, periodic_center
from molsimflow.postprocess.surface_reference import SurfaceReference, load_surface_reference


@dataclass(frozen=True)
class CoordinateFrame:
    step: int
    bounds: np.ndarray
    coordinates: np.ndarray
    surface: np.ndarray


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Atom range must be START:END")
    start, end = map(int, parts)
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Atom range must satisfy 1 <= START <= END")
    return start, end


def iter_selected_frames(
    path: Path,
    atom_range: tuple[int, int],
    *,
    mode: str,
    atom_type: int | None,
    surface_range: tuple[int, int] | None = None,
) -> Iterator[CoordinateFrame]:
    with Path(path).open(encoding="utf-8") as handle:
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
            selected, surface = [], []
            for _ in range(atom_count):
                values = handle.readline().split()
                atom_id = int(values[index["id"]])
                xyz = [float(values[index[key]]) for key in ("x", "y", "z")]
                if surface_range is not None and surface_range[0] <= atom_id <= surface_range[1]:
                    surface.append((atom_id, *xyz))
                if not atom_range[0] <= atom_id <= atom_range[1]:
                    continue
                current_type = int(values[index["type"]])
                if mode == "atom-type" and current_type != atom_type:
                    continue
                selected.append((atom_id, *xyz))
            array = np.asarray(sorted(selected), dtype=float)
            if array.size == 0:
                raise ValueError(f"No selected coordinates at step {step}")
            coordinates = array[:, 1:]
            if mode == "paired-centers":
                if len(coordinates) % 2:
                    raise ValueError("paired-centers mode requires an even atom count")
                lengths = box_lengths(bounds)
                delta = minimum_image_vectors(coordinates[1::2] - coordinates[0::2], lengths)
                coordinates = coordinates[0::2] + 0.5 * delta
                coordinates = (coordinates - bounds[:, 0]) % lengths + bounds[:, 0]
            surface_coordinates = (
                np.asarray(sorted(surface), dtype=float)[:, 1:]
                if surface else np.empty((0, 3), dtype=float)
            )
            yield CoordinateFrame(
                step=step, bounds=bounds, coordinates=coordinates, surface=surface_coordinates
            )


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


def build_density(
    frames: Sequence[CoordinateFrame],
    *,
    surface_z: float,
    cluster_cutoff: float,
    r_edges: np.ndarray,
    z_edges: np.ndarray,
    smoothing_sigma: float,
    surface_reference: SurfaceReference | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    counts = np.zeros((len(z_edges) - 1, len(r_edges) - 1), dtype=float)
    cluster_sizes = []
    for frame in frames:
        members = largest_cluster(frame.coordinates, frame.bounds, cluster_cutoff)
        cluster = frame.coordinates[members]
        center = periodic_center(cluster, frame.bounds)
        lengths = box_lengths(frame.bounds)
        vectors = minimum_image_vectors(cluster - center, lengths)
        radius = np.linalg.norm(vectors[:, :2], axis=1)
        frame_surface_z = (
            surface_reference.plane_z(frame.surface, frame.bounds)
            if surface_reference is not None else surface_z
        )
        z = minimum_image_vectors(
            np.column_stack(
                (np.zeros(len(cluster)), np.zeros(len(cluster)), cluster[:, 2] - frame_surface_z)
            ),
            lengths,
        )[:, 2]
        counts += np.histogram2d(z, radius, bins=(z_edges, r_edges))[0]
        cluster_sizes.append(len(cluster))
    annulus_area = math.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    volumes = (z_edges[1:] - z_edges[:-1])[:, None] * annulus_area[None, :]
    density = counts / (len(frames) * volumes)
    return gaussian_filter(density, smoothing_sigma), np.asarray(cluster_sizes)


def contour_points(
    density: np.ndarray,
    r_centers: np.ndarray,
    z_centers: np.ndarray,
    threshold: float,
    *,
    fit_z_min: float,
    fit_z_max: float,
) -> np.ndarray:
    points = []
    for z, profile in zip(z_centers, density):
        if not fit_z_min <= z <= fit_z_max:
            continue
        above = np.flatnonzero(profile >= threshold)
        if len(above) == 0 or above[-1] >= len(profile) - 1:
            continue
        index = int(above[-1])
        left_density, right_density = profile[index], profile[index + 1]
        fraction = (
            (threshold - left_density) / (right_density - left_density)
            if not np.isclose(right_density, left_density)
            else 0.0
        )
        radius = r_centers[index] + fraction * (r_centers[index + 1] - r_centers[index])
        if radius > 0:
            points.append((radius, z))
    return np.asarray(points, dtype=float)


def fit_axisymmetric_circle(points: np.ndarray) -> dict:
    if len(points) < 5:
        raise ValueError("At least five contour points are required for a circle fit")
    radius_coord, z = points[:, 0], points[:, 1]
    target = radius_coord**2 + z**2
    design = np.column_stack((2.0 * z, np.ones(len(z))))
    center_z, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    circle_radius_squared = float(intercept + center_z**2)
    if circle_radius_squared <= 0:
        raise ValueError("Circle fit produced a non-positive radius")
    circle_radius = math.sqrt(circle_radius_squared)
    residual = np.sqrt(radius_coord**2 + (z - center_z) ** 2) - circle_radius
    intersects = abs(center_z) < circle_radius
    cosine = float(np.clip(-center_z / circle_radius, -1.0, 1.0))
    dense_angle = math.degrees(math.acos(cosine)) if intersects else float("nan")
    contact_radius = math.sqrt(max(0.0, circle_radius**2 - center_z**2)) if intersects else float("nan")
    return {
        "center_z_A": float(center_z),
        "circle_radius_A": circle_radius,
        "contact_radius_A": contact_radius,
        "dense_phase_contact_angle_deg": dense_angle,
        "complementary_phase_contact_angle_deg": 180.0 - dense_angle if intersects else float("nan"),
        "fit_rmse_A": float(np.sqrt(np.mean(residual**2))),
        "fit_point_count": len(points),
        "circle_intersects_surface": bool(intersects),
    }


def write_outputs(
    output: Path,
    density: np.ndarray,
    r_centers: np.ndarray,
    z_centers: np.ndarray,
    fits: list[dict],
    contours: list[tuple[float, np.ndarray]],
    *,
    phase_label: str,
    font_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    np.savez_compressed(output / "density_grid.npz", density=density, r_A=r_centers, z_A=z_centers)
    with (output / "density_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_A", "r_A", "number_density_A-3"])
        for z_index, z in enumerate(z_centers):
            for r_index, radius in enumerate(r_centers):
                writer.writerow([z, radius, density[z_index, r_index]])
    with (output / "contour_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold_fraction", "r_A", "z_A"])
        for fraction, points in contours:
            for radius, z in points:
                writer.writerow([fraction, radius, z])
    with (output / "contact_angle_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fits[0]))
        writer.writeheader()
        writer.writerows(fits)
    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axis = plt.subplots(figsize=(7.2, 5.5))
    image = axis.pcolormesh(r_centers, z_centers, density, shading="auto")
    for fraction, points in contours:
        axis.plot(
            points[:, 0],
            points[:, 1],
            ".",
            ms=3,
            label=rf"${fraction:.1f}\rho_{{\rm ref}}$",
        )
    axis.set_xlabel("Radial distance (Å)")
    axis.set_ylabel("Height above surface (Å)")
    axis.set_title(f"{phase_label.replace('_', ' ')} number density")
    axis.legend(frameon=False)
    figure.colorbar(image, ax=axis, label=r"Number density (Å$^{-3}$)")
    figure.tight_layout()
    figure.savefig(output / "density_contact_angle.png", dpi=300)
    plt.close(figure)


def block_contact_angles(
    frames: Sequence[CoordinateFrame],
    *,
    block_frames: int,
    surface_z: float,
    cluster_cutoff: float,
    r_edges: np.ndarray,
    z_edges: np.ndarray,
    smoothing_sigma: float,
    reference_density: float,
    fit_z_min: float,
    fit_z_max: float,
    timestep_fs: float,
    surface_reference: SurfaceReference | None,
) -> list[dict]:
    rows = []
    block_frames = min(block_frames, len(frames))
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    for start in range(0, len(frames) - block_frames + 1, block_frames):
        block = frames[start : start + block_frames]
        density, cluster_sizes = build_density(
            block,
            surface_z=surface_z,
            cluster_cutoff=cluster_cutoff,
            r_edges=r_edges,
            z_edges=z_edges,
            smoothing_sigma=smoothing_sigma,
            surface_reference=surface_reference,
        )
        points = contour_points(
            density,
            r_centers,
            z_centers,
            0.5 * reference_density,
            fit_z_min=fit_z_min,
            fit_z_max=fit_z_max,
        )
        row = {
            "block_index": len(rows),
            "first_step": block[0].step,
            "last_step": block[-1].step,
            "start_time_ns": block[0].step * timestep_fs / 1.0e6,
            "end_time_ns": block[-1].step * timestep_fs / 1.0e6,
            "frame_count": len(block),
            "mean_largest_cluster_size": float(cluster_sizes.mean()),
        }
        try:
            row.update(fit_axisymmetric_circle(points))
        except ValueError:
            row.update(
                {
                    "center_z_A": float("nan"),
                    "circle_radius_A": float("nan"),
                    "contact_radius_A": float("nan"),
                    "dense_phase_contact_angle_deg": float("nan"),
                    "complementary_phase_contact_angle_deg": float("nan"),
                    "fit_rmse_A": float("nan"),
                    "fit_point_count": len(points),
                    "circle_intersects_surface": False,
                }
            )
        rows.append(row)
    return rows


def run_analysis(args: argparse.Namespace) -> dict:
    start_step = round(args.start_ns * 1.0e6 / args.timestep_fs)
    end_step = round(args.end_ns * 1.0e6 / args.timestep_fs)
    frames_by_step: dict[int, CoordinateFrame] = {}
    raw_window_frames = 0
    surface_reference = (
        load_surface_reference(args.reference_structure, args.surface_range, args.surface_z_A)
        if args.reference_structure is not None and args.surface_range is not None else None
    )
    if args.reference_structure is not None and args.surface_range is None:
        raise ValueError("--surface-range is required with --reference-structure")
    for trajectory in args.trajectory:
        for frame in iter_selected_frames(
            trajectory,
            args.atom_range,
            mode=args.mode,
            atom_type=args.atom_type,
            surface_range=args.surface_range,
        ):
            if start_step <= frame.step <= end_step:
                raw_window_frames += 1
                frames_by_step[frame.step] = frame
    frames = [frames_by_step[step] for step in sorted(frames_by_step)]
    if len(frames) < args.minimum_frames:
        raise ValueError(f"Only {len(frames)} unique frames found; need {args.minimum_frames}")
    r_edges = np.arange(0.0, args.r_max_A + args.dr_A, args.dr_A)
    z_edges = np.arange(args.z_min_A, args.z_max_A + args.dz_A, args.dz_A)
    density, cluster_sizes = build_density(
        frames,
        surface_z=args.surface_z_A,
        cluster_cutoff=args.cluster_cutoff_A,
        r_edges=r_edges,
        z_edges=z_edges,
        smoothing_sigma=args.smoothing_sigma_bins,
        surface_reference=surface_reference,
    )
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    block_rows = block_contact_angles(
        frames,
        block_frames=args.block_frames,
        surface_z=args.surface_z_A,
        cluster_cutoff=args.cluster_cutoff_A,
        r_edges=r_edges,
        z_edges=z_edges,
        smoothing_sigma=args.smoothing_sigma_bins,
        reference_density=args.reference_density_A3,
        fit_z_min=args.fit_z_min_A,
        fit_z_max=args.fit_z_max_A,
        timestep_fs=args.timestep_fs,
        surface_reference=surface_reference,
    )
    fits, contours = [], []
    for fraction in args.threshold_fraction:
        points = contour_points(
            density,
            r_centers,
            z_centers,
            args.reference_density_A3 * fraction,
            fit_z_min=args.fit_z_min_A,
            fit_z_max=args.fit_z_max_A,
        )
        fit = fit_axisymmetric_circle(points)
        fit.update(
            {
                "threshold_fraction": fraction,
                "threshold_density_A-3": args.reference_density_A3 * fraction,
            }
        )
        fits.append(fit)
        contours.append((fraction, points))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_outputs(
        output, density, r_centers, z_centers, fits, contours,
        phase_label=args.phase_label, font_path=args.font_path,
    )
    with (output / "block_contact_angles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_rows[0]))
        writer.writeheader()
        writer.writerows(block_rows)
    primary = min(fits, key=lambda row: abs(row["threshold_fraction"] - 0.5))
    angles = [row["dense_phase_contact_angle_deg"] for row in fits]
    block_angles = np.asarray(
        [row["dense_phase_contact_angle_deg"] for row in block_rows], dtype=float
    )
    finite_block_angles = block_angles[np.isfinite(block_angles)]
    summary = {
        "status": "PASS",
        "phase_label": args.phase_label,
        "raw_window_frames": raw_window_frames,
        "unique_window_frames": len(frames),
        "first_step": frames[0].step,
        "last_step": frames[-1].step,
        "reference_density_A-3": args.reference_density_A3,
        "mean_largest_cluster_size": float(cluster_sizes.mean()),
        "primary_threshold_fraction": primary["threshold_fraction"],
        "primary_dense_phase_contact_angle_deg": primary["dense_phase_contact_angle_deg"],
        "primary_complementary_phase_contact_angle_deg": primary[
            "complementary_phase_contact_angle_deg"
        ],
        "threshold_angle_span_deg": float(max(angles) - min(angles)),
        "block_frames": args.block_frames,
        "valid_block_count": len(finite_block_angles),
        "block_dense_phase_angle_mean_deg": float(np.mean(finite_block_angles)),
        "block_dense_phase_angle_std_deg": (
            float(np.std(finite_block_angles, ddof=1))
            if len(finite_block_angles) > 1
            else None
        ),
        "contact_angle_is_density_fit_candidate": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest["trajectory"] = [str(Path(path).resolve()) for path in args.trajectory]
    manifest["atom_range"] = list(args.atom_range)
    manifest["font_path"] = str(args.font_path)
    manifest["surface_reference_mode"] = (
        "dynamic_slab_translation" if surface_reference is not None else "fixed_nominal_plane"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atom-range", type=parse_range, required=True)
    parser.add_argument("--mode", choices=("paired-centers", "atom-type"), required=True)
    parser.add_argument("--atom-type", type=int)
    parser.add_argument("--phase-label", required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--surface-range", type=parse_range)
    parser.add_argument("--reference-structure", type=Path)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--start-ns", type=float, required=True)
    parser.add_argument("--end-ns", type=float, required=True)
    parser.add_argument("--minimum-frames", type=int, default=100)
    parser.add_argument("--block-frames", type=int, default=20)
    parser.add_argument("--cluster-cutoff-A", type=float, required=True)
    parser.add_argument("--reference-density-A3", type=float, required=True)
    parser.add_argument("--threshold-fraction", type=float, action="append", default=[])
    parser.add_argument("--r-max-A", type=float, default=40.0)
    parser.add_argument("--z-min-A", type=float, default=0.0)
    parser.add_argument("--z-max-A", type=float, default=60.0)
    parser.add_argument("--dr-A", type=float, default=1.0)
    parser.add_argument("--dz-A", type=float, default=1.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--fit-z-min-A", type=float, default=2.0)
    parser.add_argument("--fit-z-max-A", type=float, default=55.0)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "atom-type" and args.atom_type is None:
        raise ValueError("--atom-type is required in atom-type mode")
    if not args.threshold_fraction:
        args.threshold_fraction = [0.4, 0.5, 0.6]
    print(json.dumps(run_analysis(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
