"""PBC-aware particle-level contact-line geometry and jump candidates."""

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
from molsimflow.postprocess.nanobubble_attachment import (
    iter_selected_frames as iter_bubble_frames,
)
from molsimflow.postprocess.nanobubble_attachment import largest_cluster, molecule_centers
from molsimflow.postprocess.nanodroplet_spreading import (
    iter_selected_frames as iter_droplet_frames,
)


@dataclass(frozen=True)
class ContactFrame:
    source: Path
    source_frame: int
    step: int
    bounds: np.ndarray
    surface: np.ndarray
    phase_points: np.ndarray
    contact_sites: np.ndarray


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Atom range must be START:END")
    start, end = map(int, parts)
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Atom range must satisfy 1 <= START <= END")
    return start, end


def iter_contact_frames(
    path: Path,
    surface_range: tuple[int, int],
    phase_range: tuple[int, int],
    *,
    mode: str,
    atom_type: int | None,
) -> Iterator[ContactFrame]:
    if mode == "paired-centers":
        for frame in iter_bubble_frames(path, surface_range, phase_range):
            yield ContactFrame(
                source=frame.source,
                source_frame=frame.source_frame,
                step=frame.step,
                bounds=frame.bounds,
                surface=frame.surface,
                phase_points=molecule_centers(frame.nitrogen, frame.bounds),
                contact_sites=frame.nitrogen.reshape(-1, 2, 3),
            )
        return
    if atom_type is None:
        raise ValueError("--atom-type is required in atom-type mode")
    for frame in iter_droplet_frames(path, surface_range, phase_range):
        points = frame.water[frame.water_types == atom_type]
        if len(points) == 0:
            raise ValueError(f"No atoms of type {atom_type} at step {frame.step}")
        yield ContactFrame(
            source=frame.source,
            source_frame=frame.source_frame,
            step=frame.step,
            bounds=frame.bounds,
            surface=frame.surface,
            phase_points=points,
            contact_sites=points[:, None, :],
        )


def polygon_metrics(points: np.ndarray) -> tuple[dict, np.ndarray]:
    """Return 2D convex-hull metrics for local minimum-image coordinates."""
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return {}, np.empty((0, 2))
    try:
        hull = ConvexHull(points)
    except QhullError:
        return {}, np.empty((0, 2))
    boundary = points[hull.vertices]
    following = np.roll(boundary, -1, axis=0)
    cross = boundary[:, 0] * following[:, 1] - following[:, 0] * boundary[:, 1]
    twice_signed_area = float(cross.sum())
    if np.isclose(twice_signed_area, 0.0):
        return {}, np.empty((0, 2))
    centroid = np.sum((boundary + following) * cross[:, None], axis=0) / (
        3.0 * twice_signed_area
    )
    radii = np.linalg.norm(boundary - centroid, axis=1)
    area = float(hull.volume)
    perimeter = float(hull.area)
    return {
        "contact_line_area_A2": area,
        "contact_line_perimeter_A": perimeter,
        "contact_line_equivalent_radius_A": math.sqrt(area / math.pi),
        "contact_line_vertex_count": len(boundary),
        "contact_line_mean_vertex_radius_A": float(radii.mean()),
        "contact_line_radius_std_A": float(radii.std(ddof=1)) if len(radii) > 1 else 0.0,
        "contact_line_radius_min_A": float(radii.min()),
        "contact_line_radius_max_A": float(radii.max()),
        "contact_line_circularity": 4.0 * math.pi * area / perimeter**2,
        "centroid_dx_A": float(centroid[0]),
        "centroid_dy_A": float(centroid[1]),
    }, boundary


def analyze_frame(
    frame: ContactFrame, *, cluster_cutoff: float, contact_cutoff: float
) -> tuple[dict, np.ndarray]:
    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    members = largest_cluster(frame.phase_points, frame.bounds, cluster_cutoff)
    cluster = frame.phase_points[members]
    center = periodic_center(cluster, frame.bounds)
    vectors = minimum_image_vectors(cluster - center, lengths)
    shifted_surface = (frame.surface - frame.bounds[:, 0]) % lengths
    shifted_sites = (frame.contact_sites - frame.bounds[:, 0]) % lengths
    distances = cKDTree(shifted_surface, boxsize=lengths).query(
        shifted_sites.reshape(-1, 3), k=1
    )[0].reshape(len(frame.phase_points), -1).min(axis=1)
    contact_vectors = vectors[distances[members] <= contact_cutoff, :2]
    geometry, boundary = polygon_metrics(contact_vectors)
    row = {
        "source_file": str(frame.source),
        "source_frame": frame.source_frame,
        "step": frame.step,
        "largest_cluster_size": len(members),
        "contact_point_count": len(contact_vectors),
        "phase_center_x_A": float(center[0]),
        "phase_center_y_A": float(center[1]),
    }
    if not geometry:
        row.update(
            {
                "contact_line_center_x_A": float("nan"),
                "contact_line_center_y_A": float("nan"),
                "contact_line_area_A2": float("nan"),
                "contact_line_perimeter_A": float("nan"),
                "contact_line_equivalent_radius_A": float("nan"),
                "contact_line_vertex_count": 0,
                "contact_line_mean_vertex_radius_A": float("nan"),
                "contact_line_radius_std_A": float("nan"),
                "contact_line_radius_min_A": float("nan"),
                "contact_line_radius_max_A": float("nan"),
                "contact_line_circularity": float("nan"),
            }
        )
        return row, boundary
    center_xy = center[:2] + np.array([geometry.pop("centroid_dx_A"), geometry.pop("centroid_dy_A")])
    center_xy = (center_xy - frame.bounds[:2, 0]) % lengths[:2] + frame.bounds[:2, 0]
    row.update(
        {
            "contact_line_center_x_A": float(center_xy[0]),
            "contact_line_center_y_A": float(center_xy[1]),
            **geometry,
        }
    )
    absolute_boundary = boundary + center[:2]
    absolute_boundary = (
        (absolute_boundary - frame.bounds[:2, 0]) % lengths[:2] + frame.bounds[:2, 0]
    )
    return row, absolute_boundary


def add_unwrapped_centers(rows: list[dict], bounds: np.ndarray) -> None:
    lengths = box_lengths(bounds)[:2]
    previous = None
    unwrapped = None
    origin = None
    for row in rows:
        current = np.array(
            [row["contact_line_center_x_A"], row["contact_line_center_y_A"]], dtype=float
        )
        if not np.all(np.isfinite(current)):
            row["contact_line_center_x_unwrapped_A"] = float("nan")
            row["contact_line_center_y_unwrapped_A"] = float("nan")
            row["contact_line_center_displacement_A"] = float("nan")
            continue
        if previous is None:
            unwrapped = current.copy()
            origin = current.copy()
        else:
            unwrapped += minimum_image_vectors(current - previous, lengths)
        row["contact_line_center_x_unwrapped_A"] = float(unwrapped[0])
        row["contact_line_center_y_unwrapped_A"] = float(unwrapped[1])
        row["contact_line_center_displacement_A"] = float(np.linalg.norm(unwrapped - origin))
        previous = current


def block_rows(
    rows: Sequence[dict],
    *,
    block_frames: int,
    timestep_fs: float,
    jump_sigma: float,
    minimum_jump: float,
) -> tuple[list[dict], float]:
    blocks = []
    for start in range(0, len(rows) - block_frames + 1, block_frames):
        block = rows[start : start + block_frames]
        radii = np.asarray([row["contact_line_equivalent_radius_A"] for row in block])
        finite = radii[np.isfinite(radii)]
        if len(finite) == 0:
            continue
        blocks.append(
            {
                "block_index": len(blocks),
                "first_step": block[0]["step"],
                "last_step": block[-1]["step"],
                "start_time_ns": block[0]["step"] * timestep_fs / 1.0e6,
                "end_time_ns": block[-1]["step"] * timestep_fs / 1.0e6,
                "frame_count": len(block),
                "valid_frame_count": len(finite),
                "mean_equivalent_radius_A": float(finite.mean()),
                "std_equivalent_radius_A": (
                    float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
                ),
            }
        )
    if not blocks:
        return blocks, minimum_jump
    means = np.asarray([row["mean_equivalent_radius_A"] for row in blocks])
    deltas = np.diff(means)
    median_delta = float(np.median(deltas)) if len(deltas) else 0.0
    mad = float(np.median(np.abs(deltas - median_delta))) if len(deltas) else 0.0
    threshold = max(minimum_jump, jump_sigma * 1.4826 * mad)
    for index, row in enumerate(blocks):
        delta = float("nan") if index == 0 else float(means[index] - means[index - 1])
        row["delta_mean_radius_A"] = delta
        row["radius_jump_candidate"] = bool(
            index > 0 and abs(delta - median_delta) >= threshold
        )
    return blocks, threshold


def write_plot(
    rows: Sequence[dict], blocks: Sequence[dict], output: Path, timestep_fs: float, font_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = properties.get_name()
    time = np.asarray([row["step"] for row in rows]) * timestep_fs / 1.0e6
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    axes[0].plot(time, [row["contact_line_equivalent_radius_A"] for row in rows], lw=0.9)
    axes[0].set_ylabel(r"$R_{CL}$ (Å)")
    axes[1].plot(time, [row["contact_line_center_displacement_A"] for row in rows], lw=0.9)
    axes[1].set_ylabel("Center displacement (Å)")
    axes[2].plot(time, [row["contact_line_circularity"] for row in rows], lw=0.9)
    axes[2].set_ylabel("Circularity")
    axes[2].set_xlabel("Time (ns)")
    for block in blocks:
        if block["radius_jump_candidate"]:
            for axis in axes:
                axis.axvline(block["start_time_ns"], color="tab:red", alpha=0.35, lw=0.8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    start_step = None if args.start_ns is None else round(args.start_ns * 1.0e6 / args.timestep_fs)
    end_step = None if args.end_ns is None else round(args.end_ns * 1.0e6 / args.timestep_fs)
    records: dict[int, tuple[dict, np.ndarray]] = {}
    raw_frames = 0
    reference_bounds = None
    stop = False
    for trajectory in args.trajectory:
        for frame in iter_contact_frames(
            trajectory,
            args.surface_range,
            args.phase_range,
            mode=args.mode,
            atom_type=args.atom_type,
        ):
            raw_frames += 1
            if reference_bounds is None:
                reference_bounds = frame.bounds
            elif not np.allclose(frame.bounds, reference_bounds):
                raise ValueError("Contact-line analysis requires a constant orthorhombic box")
            if (start_step is None or frame.step >= start_step) and (
                end_step is None or frame.step <= end_step
            ):
                records[frame.step] = analyze_frame(
                    frame,
                    cluster_cutoff=args.cluster_cutoff_A,
                    contact_cutoff=args.contact_cutoff_A,
                )
            if args.max_frames is not None and raw_frames >= args.max_frames:
                stop = True
                break
        if stop:
            break
    ordered = [(step, *records[step]) for step in sorted(records)]
    if args.drop_first_frame and ordered and ordered[0][0] == 0:
        ordered = ordered[1:]
    if not ordered or reference_bounds is None:
        raise ValueError("No analyzed frames remain")
    rows = [item[1] for item in ordered]
    add_unwrapped_centers(rows, reference_bounds)
    if not any(math.isfinite(row["contact_line_equivalent_radius_A"]) for row in rows):
        raise ValueError("No frame contains at least three non-collinear contact points")
    blocks, jump_threshold = block_rows(
        rows,
        block_frames=args.block_frames,
        timestep_fs=args.timestep_fs,
        jump_sigma=args.jump_sigma,
        minimum_jump=args.minimum_jump_A,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "contact_line.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "contact_line_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "vertex_index", "x_A", "y_A"])
        for step, _, boundary in ordered:
            for index, (x, y) in enumerate(boundary):
                writer.writerow([step, index, x, y])
    with (output / "contact_line_blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
    finite = [row for row in rows if math.isfinite(row["contact_line_equivalent_radius_A"])]
    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "unique_window_frames_before_drop": len(records),
        "analyzed_frames": len(rows),
        "valid_contact_line_frames": len(finite),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "block_frames": args.block_frames,
        "valid_block_count": len(blocks),
        "jump_threshold_A": jump_threshold,
        "jump_candidate_count": sum(row["radius_jump_candidate"] for row in blocks),
        "jump_candidates_are_operational_not_pinning_proof": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "surface_atom_range": list(args.surface_range),
        "phase_atom_range": list(args.phase_range),
        "mode": args.mode,
        "atom_type": args.atom_type,
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
        "contact_definition": "convex hull of largest-cluster phase points whose contact sites are within the surface cutoff",
        "cluster_cutoff_A": args.cluster_cutoff_A,
        "contact_cutoff_A": args.contact_cutoff_A,
        "start_ns": args.start_ns,
        "end_ns": args.end_ns,
        "jump_sigma": args.jump_sigma,
        "minimum_jump_A": args.minimum_jump_A,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(rows, blocks, output / "contact_line.png", args.timestep_fs, args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--phase-range", type=parse_range, required=True)
    parser.add_argument("--mode", choices=("paired-centers", "atom-type"), required=True)
    parser.add_argument("--atom-type", type=int)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--start-ns", type=float)
    parser.add_argument("--end-ns", type=float)
    parser.add_argument("--cluster-cutoff-A", type=float, required=True)
    parser.add_argument("--contact-cutoff-A", type=float, required=True)
    parser.add_argument("--block-frames", type=int, default=20)
    parser.add_argument("--jump-sigma", type=float, default=4.0)
    parser.add_argument("--minimum-jump-A", type=float, default=2.0)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--drop-first-frame", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "atom-type" and args.atom_type is None:
        raise ValueError("--atom-type is required in atom-type mode")
    if args.block_frames < 1 or args.jump_sigma < 0 or args.minimum_jump_A < 0:
        raise ValueError("Block size must be positive and jump thresholds non-negative")
    print(json.dumps(run_analysis(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
