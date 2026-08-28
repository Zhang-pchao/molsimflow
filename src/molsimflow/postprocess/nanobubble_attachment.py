"""PBC-aware attachment kinetics for a single N2 nanobubble above a slab."""

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
class SelectedFrame:
    source: Path
    source_frame: int
    step: int
    bounds: np.ndarray
    surface: np.ndarray
    nitrogen: np.ndarray


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Atom range must be START:END")
    start, end = map(int, parts)
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Atom range must satisfy 1 <= START <= END")
    return start, end


def iter_selected_frames(
    path: Path, surface_range: tuple[int, int], nitrogen_range: tuple[int, int]
) -> Iterator[SelectedFrame]:
    """Stream only surface and nitrogen coordinates from a custom LAMMPS dump."""
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
            missing = {"id", "x", "y", "z"}.difference(index)
            if missing:
                raise ValueError(f"Missing dump columns at step {step}: {sorted(missing)}")
            surface, nitrogen = [], []
            for _ in range(atom_count):
                values = handle.readline().split()
                atom_id = int(values[index["id"]])
                xyz = [float(values[index[key]]) for key in ("x", "y", "z")]
                if surface_range[0] <= atom_id <= surface_range[1]:
                    surface.append((atom_id, *xyz))
                elif nitrogen_range[0] <= atom_id <= nitrogen_range[1]:
                    nitrogen.append((atom_id, *xyz))
            surface_array = np.asarray(sorted(surface), dtype=float)[:, 1:]
            nitrogen_array = np.asarray(sorted(nitrogen), dtype=float)[:, 1:]
            if len(nitrogen_array) != nitrogen_range[1] - nitrogen_range[0] + 1:
                raise ValueError(f"Incomplete nitrogen selection at step {step}")
            yield SelectedFrame(Path(path), frame_index, step, bounds, surface_array, nitrogen_array)
            frame_index += 1


def molecule_centers(nitrogen: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    if len(nitrogen) % 2:
        raise ValueError("The nitrogen atom range must contain complete consecutive N2 pairs")
    lengths = box_lengths(bounds)
    delta = minimum_image_vectors(nitrogen[1::2] - nitrogen[0::2], lengths)
    centers = nitrogen[0::2] + 0.5 * delta
    return (centers - bounds[:, 0]) % lengths + bounds[:, 0]


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


def footprint_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError

    try:
        return float(ConvexHull(points[:, :2]).volume)
    except QhullError:
        return 0.0


def analyze_frame(frame: SelectedFrame, *, surface_z: float, cluster_cutoff: float, contact_cutoff: float) -> dict:
    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    centers = molecule_centers(frame.nitrogen, frame.bounds)
    members = largest_cluster(centers, frame.bounds, cluster_cutoff)
    bubble_centers = centers[members]
    center = periodic_center(bubble_centers, frame.bounds)
    vectors = minimum_image_vectors(bubble_centers - center, lengths)
    radii = np.linalg.norm(vectors, axis=1)
    shifted_surface = (frame.surface - frame.bounds[:, 0]) % lengths
    shifted_n = (frame.nitrogen - frame.bounds[:, 0]) % lengths
    distances = cKDTree(shifted_surface, boxsize=lengths).query(shifted_n, k=1)[0].reshape(-1, 2)
    molecule_distance = distances.min(axis=1)
    cluster_distance = molecule_distance[members]
    contact_vectors = vectors[cluster_distance <= contact_cutoff]
    lateral_radii = np.linalg.norm(vectors[:, :2], axis=1)
    z_q05, z_q95 = np.quantile(vectors[:, 2], [0.05, 0.95])
    gyration = vectors.T @ vectors / len(vectors)
    eigenvalues = np.sort(np.linalg.eigvalsh(gyration))[::-1]
    eigenvalue_sum = float(eigenvalues.sum())
    relative_anisotropy = (
        1.5 * float(np.sum(eigenvalues**2)) / eigenvalue_sum**2 - 0.5
        if eigenvalue_sum > 0
        else float("nan")
    )
    dz = float(minimum_image_vectors(np.array([[0.0, 0.0, center[2] - surface_z]]), lengths)[0, 2])
    return {
        "source_file": str(frame.source),
        "source_frame": frame.source_frame,
        "step": frame.step,
        "bubble_center_x_A": float(center[0]),
        "bubble_center_y_A": float(center[1]),
        "bubble_center_z_A": float(center[2]),
        "bubble_center_surface_dz_A": dz,
        "bubble_radius_p90_A": float(np.quantile(radii, 0.9)),
        "bubble_lower_edge_gap_p90_A": dz - float(np.quantile(radii, 0.9)),
        "largest_cluster_n2_count": len(members),
        "dissolved_or_disconnected_n2_count": len(centers) - len(members),
        "bubble_height_q05_q95_A": float(z_q95 - z_q05),
        "bubble_lateral_radius_p90_A": float(np.quantile(lateral_radii, 0.9)),
        "footprint_convex_hull_area_A2": footprint_area(contact_vectors),
        "gyration_eigenvalue_1_A2": float(eigenvalues[0]),
        "gyration_eigenvalue_2_A2": float(eigenvalues[1]),
        "gyration_eigenvalue_3_A2": float(eigenvalues[2]),
        "relative_shape_anisotropy": relative_anisotropy,
        "bubble_contact_n2_count": int(np.count_nonzero(molecule_distance[members] <= contact_cutoff)),
        "all_n2_contact_count": int(np.count_nonzero(molecule_distance <= contact_cutoff)),
        "min_bubble_surface_distance_A": float(molecule_distance[members].min()),
    }


def first_persistent_contact(rows: Sequence[dict], minimum: int, persistence: int) -> dict | None:
    for start in range(len(rows) - persistence + 1):
        window = rows[start : start + persistence]
        if all(int(row["bubble_contact_n2_count"]) >= minimum for row in window):
            return rows[start]
    return None


def add_unwrapped_center(rows: list[dict], bounds: np.ndarray) -> None:
    lengths = box_lengths(bounds)
    previous = None
    unwrapped = None
    origin = None
    for row in rows:
        current = np.array([row[f"bubble_center_{dim}_A"] for dim in "xyz"])
        if previous is None:
            unwrapped = current.copy()
            origin = current.copy()
        else:
            unwrapped += minimum_image_vectors(current - previous, lengths)
        for dim, value in zip("xyz", unwrapped):
            row[f"bubble_center_{dim}_unwrapped_A"] = float(value)
        displacement = unwrapped[:2] - origin[:2]
        row["bubble_lateral_displacement_A"] = float(np.linalg.norm(displacement))
        row["bubble_lateral_displacement_squared_A2"] = float(displacement @ displacement)
        previous = current


def write_plot(
    rows: Sequence[dict], output: Path, font_family: str, timestep_fs: float, font_path: Path | None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    if font_path is not None:
        font_manager.fontManager.addfont(font_path)
        font_properties = font_manager.FontProperties(fname=font_path)
        resolved_family = font_properties.get_name()
        font_manager.findfont(font_properties, fallback_to_default=False)
    else:
        resolved_family = font_family
        font_manager.findfont(font_family, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = resolved_family
    time_ns = np.array([row["step"] for row in rows], dtype=float) * timestep_fs / 1.0e6
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    axes[0].plot(time_ns, [row["bubble_center_surface_dz_A"] for row in rows], lw=1.2)
    axes[0].set_ylabel(r"Bubble center $z$ (Å)")
    axes[1].plot(time_ns, [row["bubble_height_q05_q95_A"] for row in rows], label="Height")
    axes[1].plot(time_ns, [row["bubble_lateral_radius_p90_A"] for row in rows], label="Lateral radius")
    axes[1].set_ylabel("Length (Å)")
    axes[1].legend(frameon=False)
    axes[2].plot(time_ns, [row["footprint_convex_hull_area_A2"] for row in rows], lw=1.2)
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel(r"Footprint (Å$^2$)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    rows_by_step: dict[int, dict] = {}
    raw_frames = 0
    stop = False
    last_bounds = None
    surface_reference = (
        load_surface_reference(args.reference_structure, args.surface_range, args.surface_z_A)
        if args.reference_structure is not None
        else None
    )
    for trajectory in args.trajectory:
        for frame in iter_selected_frames(trajectory, args.surface_range, args.nitrogen_range):
            raw_frames += 1
            last_bounds = frame.bounds
            surface_z = (
                surface_reference.plane_z(frame.surface, frame.bounds)
                if surface_reference is not None
                else args.surface_z_A
            )
            row = analyze_frame(
                frame,
                surface_z=surface_z,
                cluster_cutoff=args.cluster_cutoff_A,
                contact_cutoff=args.contact_cutoff_A,
            )
            row["surface_reference_z_A"] = surface_z
            rows_by_step[frame.step] = row
            if args.max_frames is not None and raw_frames >= args.max_frames:
                stop = True
                break
        if stop:
            break
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    if args.drop_first_frame and rows:
        rows = rows[1:]
    if not rows or last_bounds is None:
        raise ValueError("No analyzed frames remain")
    add_unwrapped_center(rows, last_bounds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / f"{args.output_stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    event = first_persistent_contact(rows, args.minimum_contact_n2, args.persistence_frames)
    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "unique_frames_before_drop": len(rows_by_step),
        "analyzed_frames": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "time_step_fs": args.timestep_fs,
        "frame_interval_ps": (rows[1]["step"] - rows[0]["step"]) * args.timestep_fs / 1000.0,
        "surface_z_A": args.surface_z_A,
        "surface_reference_mode": (
            "dynamic_slab_translation" if surface_reference is not None else "fixed_nominal_plane"
        ),
        "contact_definition": {
            "distance_cutoff_A": args.contact_cutoff_A,
            "minimum_n2_molecules": args.minimum_contact_n2,
            "persistence_frames": args.persistence_frames,
        },
        "attachment_step": None if event is None else event["step"],
        "attachment_time_ns": None if event is None else event["step"] * args.timestep_fs / 1.0e6,
        "attachment_is_operational_not_thermodynamic": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "surface_atom_range": list(args.surface_range),
        "nitrogen_atom_range": list(args.nitrogen_range),
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
        "drop_first_frame": args.drop_first_frame,
        "cluster_cutoff_A": args.cluster_cutoff_A,
        "reference_structure": (
            None if args.reference_structure is None else str(args.reference_structure.resolve())
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(
        rows,
        output / f"{args.output_stem}.png",
        args.font_family,
        args.timestep_fs,
        args.font_path,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="attachment_kinetics")
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--nitrogen-range", type=parse_range, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--reference-structure", type=Path)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--cluster-cutoff-A", type=float, default=5.5)
    parser.add_argument("--contact-cutoff-A", type=float, default=4.0)
    parser.add_argument("--minimum-contact-n2", type=int, default=3)
    parser.add_argument("--persistence-frames", type=int, default=3)
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--drop-first-frame", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_analysis(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
