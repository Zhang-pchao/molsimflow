"""First-layer water orientation in footprint, TPCL, and far-field regions."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.interfacial_water_density import REGIONS, classify_regions
from molsimflow.postprocess.nanodroplet_spreading import parse_range
from molsimflow.postprocess.surface_reference import load_surface_reference
from molsimflow.postprocess.surface_site_enrichment import read_contact_lines


@dataclass(frozen=True)
class OrientationFrame:
    step: int
    bounds: np.ndarray
    surface: np.ndarray
    oxygen_ids: np.ndarray
    oxygen: np.ndarray
    candidate_oxygen_ids: np.ndarray
    candidate_oxygen: np.ndarray
    hydrogen_ids: np.ndarray
    hydrogen: np.ndarray


def iter_orientation_frames(
    path: Path,
    surface_range: tuple[int, int],
    water_range: tuple[int, int],
    *,
    oxygen_type: int,
    hydrogen_type: int,
) -> Iterator[OrientationFrame]:
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
            surface, oxygen, other_oxygen, hydrogen = [], [], [], []
            for _ in range(atom_count):
                values = handle.readline().split()
                atom_id = int(values[index["id"]])
                atom_type = int(values[index["type"]])
                xyz = [float(values[index[key]]) for key in ("x", "y", "z")]
                if surface_range[0] <= atom_id <= surface_range[1]:
                    surface.append((atom_id, *xyz))
                if water_range[0] <= atom_id <= water_range[1] and atom_type == oxygen_type:
                    oxygen.append((atom_id, *xyz))
                elif atom_type == oxygen_type:
                    other_oxygen.append((atom_id, *xyz))
                if atom_type == hydrogen_type:
                    hydrogen.append((atom_id, *xyz))
            if len(surface) != surface_range[1] - surface_range[0] + 1:
                raise ValueError(f"Incomplete surface selection at step {step}")
            water_array = np.asarray(sorted(oxygen), dtype=float)
            candidate_array = np.asarray(sorted(oxygen) + sorted(other_oxygen), dtype=float)
            hydrogen_array = np.asarray(sorted(hydrogen), dtype=float)
            yield OrientationFrame(
                step,
                bounds,
                np.asarray(sorted(surface), dtype=float)[:, 1:],
                water_array[:, 0].astype(int),
                water_array[:, 1:],
                candidate_array[:, 0].astype(int),
                candidate_array[:, 1:],
                hydrogen_array[:, 0].astype(int),
                hydrogen_array[:, 1:],
            )


def assign_hydrogen_neighbors(
    candidate_oxygen: np.ndarray,
    hydrogen: np.ndarray,
    bounds: np.ndarray,
    bond_cutoff: float,
) -> list[list[int]]:
    """Assign every H to its nearest O within the cutoff under periodic boundaries."""
    from scipy.spatial import cKDTree

    lengths = box_lengths(bounds)
    shifted_h = (hydrogen - bounds[:, 0]) % lengths
    shifted_o = (candidate_oxygen - bounds[:, 0]) % lengths
    distances, nearest = cKDTree(shifted_o, boxsize=lengths).query(
        shifted_h, distance_upper_bound=bond_cutoff
    )
    neighbors = [[] for _ in range(len(candidate_oxygen))]
    for hydrogen_index, (distance, oxygen_index) in enumerate(zip(distances, nearest)):
        if np.isfinite(distance) and oxygen_index < len(candidate_oxygen):
            neighbors[int(oxygen_index)].append(hydrogen_index)
    return neighbors


def molecular_orientations(
    oxygen: np.ndarray,
    candidate_oxygen: np.ndarray,
    hydrogen: np.ndarray,
    bounds: np.ndarray,
    bond_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign every H to its nearest O and orient target waters with exactly two H."""
    lengths = box_lengths(bounds)
    neighbors = assign_hydrogen_neighbors(
        candidate_oxygen, hydrogen, bounds, bond_cutoff
    )[: len(oxygen)]
    coordination = np.asarray([len(items) for items in neighbors], dtype=int)
    dipole_cos = np.full(len(oxygen), np.nan)
    oh_cos = np.full((len(oxygen), 2), np.nan)
    for index, items in enumerate(neighbors):
        if len(items) != 2:
            continue
        vectors = minimum_image_vectors(hydrogen[items] - oxygen[index], lengths)
        unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        bisector = unit.sum(axis=0)
        norm = np.linalg.norm(bisector)
        if norm > 0:
            dipole_cos[index] = bisector[2] / norm
            oh_cos[index] = unit[:, 2]
    return coordination, dipole_cos, oh_cos


def analyze_frame(
    frame: OrientationFrame,
    boundary_xy: np.ndarray,
    center_xy: np.ndarray,
    *,
    surface_z: float,
    tpcl_half_width: float,
    bond_cutoff: float,
    z_min: float,
    z_max: float,
) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
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
    coordination, dipole_cos, oh_cos = molecular_orientations(
        frame.oxygen, frame.candidate_oxygen, frame.hydrogen, frame.bounds, bond_cutoff
    )
    row = {"step": frame.step, "surface_reference_z_A": surface_z}
    values = {}
    for region in REGIONS:
        selected = layer & regions[region]
        valid = selected & (coordination == 2) & np.isfinite(dipole_cos)
        row[f"{region}_oxygen_count"] = int(np.count_nonzero(selected))
        row[f"{region}_zero_h_count"] = int(np.count_nonzero(selected & (coordination == 0)))
        row[f"{region}_one_h_count"] = int(np.count_nonzero(selected & (coordination == 1)))
        row[f"{region}_two_h_count"] = int(np.count_nonzero(valid))
        row[f"{region}_three_plus_h_count"] = int(np.count_nonzero(selected & (coordination >= 3)))
        row[f"{region}_mean_dipole_cos"] = (
            float(np.mean(dipole_cos[valid])) if np.any(valid) else float("nan")
        )
        region_oh = oh_cos[valid].ravel()
        row[f"{region}_mean_oh_cos"] = (
            float(np.mean(region_oh)) if len(region_oh) else float("nan")
        )
        values[region] = dipole_cos[valid], region_oh
    return row, values


def write_plot(rows: Sequence[dict], output: Path, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    for region in REGIONS:
        selected = [row for row in rows if row["region"] == region]
        for axis, observable in zip(axes, ("dipole", "oh")):
            data = [row for row in selected if row["observable"] == observable]
            sample_count = sum(row["count"] for row in data)
            axis.plot(
                [row["cos_theta"] for row in data],
                [row["probability_density"] for row in data],
                label=f"{region.replace('_', ' ')} (n={sample_count})",
            )
    axes[0].set_ylabel("Dipole probability density")
    axes[1].set_ylabel("O-H probability density")
    axes[1].set_xlabel(r"$\cos\theta$ relative to +z")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    contact_rows, boundaries = read_contact_lines(args.contact_line, args.contact_line_points)
    contacts = {int(row["step"]): row for row in contact_rows}
    surface_reference = load_surface_reference(
        args.reference_structure, args.surface_range, args.surface_z_A
    )
    records, sample_records = {}, {}
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
            surface_z = surface_reference.plane_z(frame.surface, frame.bounds)
            center = np.asarray(
                [float(contact["contact_line_center_x_A"]), float(contact["contact_line_center_y_A"])]
            )
            row, values = analyze_frame(
                frame,
                boundary,
                center,
                surface_z=surface_z,
                tpcl_half_width=args.tpcl_half_width_A,
                bond_cutoff=args.bond_cutoff_A,
                z_min=args.z_min_A,
                z_max=args.z_max_A,
            )
            row["time_ns"] = frame.step * args.timestep_fs / 1.0e6
            records[frame.step] = row
            sample_records[frame.step] = values
    missing = sorted(set(contacts).difference(records))
    if missing:
        raise ValueError(f"Missing trajectory data for {len(missing)} contact-line steps")
    frames = [records[step] for step in sorted(records)]
    if not frames:
        raise ValueError("No matched orientation frames")
    samples = {region: [[], []] for region in REGIONS}
    for step in sorted(sample_records):
        for region in REGIONS:
            samples[region][0].append(sample_records[step][region][0])
            samples[region][1].append(sample_records[step][region][1])
    edges = np.linspace(-1.0, 1.0, args.cosine_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profiles = []
    summary_means, summary_counts = {}, {}
    for region in REGIONS:
        for observable, index in (("dipole", 0), ("oh", 1)):
            arrays = [values for values in samples[region][index] if len(values)]
            values = np.concatenate(arrays) if arrays else np.empty(0)
            counts = np.histogram(values, bins=edges)[0]
            density = counts / (len(values) * np.diff(edges)) if len(values) else np.zeros_like(centers)
            summary_means[f"{region}_{observable}"] = float(np.mean(values)) if len(values) else None
            summary_counts[f"{region}_{observable}"] = len(values)
            for center, count, probability in zip(centers, counts, density):
                profiles.append({"region": region, "observable": observable, "cos_theta": center, "count": int(count), "probability_density": float(probability)})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with (output / "orientation_by_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frames[0])); writer.writeheader(); writer.writerows(frames)
    with (output / "orientation_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0])); writer.writeheader(); writer.writerows(profiles)
    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "matched_frames": len(frames),
        "first_step": frames[0]["step"],
        "last_step": frames[-1]["step"],
        "first_layer_z_range_A": [args.z_min_A, args.z_max_A],
        "mean_cos_theta": summary_means,
        "orientation_sample_counts": summary_counts,
        "coordination_counts": {
            region: {
                label: sum(row[f"{region}_{label}_count"] for row in frames)
                for label in ("zero_h", "one_h", "two_h", "three_plus_h")
            }
            for region in REGIONS
        },
        "coordination_policy": "orientation only for water-range O with exactly two H-type neighbors",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "surface_atom_range": list(args.surface_range),
        "water_atom_range": list(args.water_range),
        "oxygen_type": args.oxygen_type,
        "hydrogen_type": args.hydrogen_type,
        "bond_cutoff_A": args.bond_cutoff_A,
        "surface_reference_mode": "dynamic_slab_translation",
        "reference_structure": str(args.reference_structure.resolve()),
        "contact_line": str(args.contact_line.resolve()),
        "contact_line_points": str(args.contact_line_points.resolve()),
        "tpcl_half_width_A": args.tpcl_half_width_A,
        "timestep_fs": args.timestep_fs,
        "restart_policy": "later segment replaces earlier frame at duplicate timestep",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(profiles, output / "interfacial_water_orientation.png", args.font_path)
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
    parser.add_argument("--tpcl-half-width-A", type=float, default=4.0)
    parser.add_argument("--bond-cutoff-A", type=float, default=1.25)
    parser.add_argument("--z-min-A", type=float, default=0.0)
    parser.add_argument("--z-max-A", type=float, default=6.0)
    parser.add_argument("--cosine-bins", type=int, default=40)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.tpcl_half_width_A, args.bond_cutoff_A, args.cosine_bins) <= 0:
        raise ValueError("Region width, bond cutoff, and bin count must be positive")
    if args.z_max_A <= args.z_min_A:
        raise ValueError("First-layer z range is invalid")
    print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
