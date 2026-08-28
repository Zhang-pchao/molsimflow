"""Map particle contact lines onto initial CH3 and SiOH surface sites."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import minimum_image_vectors


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Atom range must be START:END")
    start, end = map(int, parts)
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Atom range must satisfy 1 <= START <= END")
    return start, end


def identify_surface_sites(
    elements: np.ndarray,
    coordinates: np.ndarray,
    lengths: np.ndarray,
    *,
    slab_range: tuple[int, int],
    surface_z: float,
    surface_depth: float,
    bond_cutoff: float,
) -> list[dict]:
    from scipy.spatial import cKDTree

    start, end = slab_range[0] - 1, slab_range[1]
    if end > len(elements):
        raise ValueError("Slab range exceeds the XYZ atom count")
    slab_elements = elements[start:end]
    slab_coordinates = coordinates[start:end]
    hydrogen = np.flatnonzero(slab_elements == "H")
    if len(hydrogen) == 0:
        raise ValueError("The slab selection contains no hydrogen atoms")
    tree = cKDTree(slab_coordinates[hydrogen] % lengths, boxsize=lengths)
    sites = []
    for local_index, (element, point) in enumerate(zip(slab_elements, slab_coordinates)):
        if point[2] < surface_z - surface_depth:
            continue
        neighbors = tree.query_ball_point(point % lengths, bond_cutoff)
        if element == "C" and len(neighbors) >= 3:
            site_type = "CH3"
        elif element == "O" and len(neighbors) >= 1:
            site_type = "SiOH"
        else:
            continue
        sites.append(
            {
                "atom_id": start + local_index + 1,
                "site_type": site_type,
                "x_A": float(point[0]),
                "y_A": float(point[1]),
                "z_A": float(point[2]),
                "bonded_H_count": len(neighbors),
            }
        )
    if not sites:
        raise ValueError("No CH3 or SiOH sites were identified on the selected surface")
    return sites


def point_segment_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    start = polygon
    segment = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - start[None, :, :]
    denominator = np.sum(segment**2, axis=1)
    projection = np.divide(
        np.sum(relative * segment[None, :, :], axis=2),
        denominator[None, :],
        out=np.zeros((len(points), len(segment))),
        where=denominator[None, :] > 0,
    )
    closest = start[None, :, :] + np.clip(projection, 0.0, 1.0)[:, :, None] * segment
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def fraction(mask: np.ndarray, is_ch3: np.ndarray) -> float:
    return float(np.count_nonzero(mask & is_ch3) / np.count_nonzero(mask)) if np.any(mask) else math.nan


def region_metrics(
    site_xy: np.ndarray,
    site_types: np.ndarray,
    boundary_xy: np.ndarray,
    center_xy: np.ndarray,
    lengths_xy: np.ndarray,
    *,
    tpcl_half_width: float,
    boundary_proximity: float,
) -> dict:
    from matplotlib.path import Path as PolygonPath

    local_sites = minimum_image_vectors(site_xy - center_xy, lengths_xy)
    local_boundary = minimum_image_vectors(boundary_xy - center_xy, lengths_xy)
    inside = PolygonPath(local_boundary).contains_points(local_sites, radius=1e-10)
    distance_to_line = point_segment_distances(local_sites, local_boundary)
    tpcl = distance_to_line <= tpcl_half_width
    is_ch3 = site_types == "CH3"
    surface_fraction = float(np.mean(is_ch3))
    ch3_sites = local_sites[is_ch3]
    sioh_sites = local_sites[~is_ch3]
    if len(ch3_sites) and len(sioh_sites):
        d_ch3 = np.linalg.norm(
            local_boundary[:, None, :] - ch3_sites[None, :, :], axis=2
        ).min(axis=1)
        d_sioh = np.linalg.norm(
            local_boundary[:, None, :] - sioh_sites[None, :, :], axis=2
        ).min(axis=1)
        boundary_proxy = 0.5 * np.abs(d_ch3 - d_sioh)
        boundary_proxy_mean = float(boundary_proxy.mean())
        boundary_near_fraction = float(np.mean(boundary_proxy <= boundary_proximity))
    else:
        boundary_proxy_mean = math.nan
        boundary_near_fraction = math.nan
    footprint_fraction = fraction(inside, is_ch3)
    tpcl_fraction = fraction(tpcl, is_ch3)
    return {
        "surface_site_count": len(site_xy),
        "surface_ch3_fraction": surface_fraction,
        "footprint_site_count": int(np.count_nonzero(inside)),
        "footprint_ch3_fraction": footprint_fraction,
        "footprint_ch3_enrichment": footprint_fraction - surface_fraction,
        "tpcl_site_count": int(np.count_nonzero(tpcl)),
        "tpcl_ch3_fraction": tpcl_fraction,
        "tpcl_ch3_enrichment": tpcl_fraction - surface_fraction,
        "mean_contact_line_boundary_proxy_A": boundary_proxy_mean,
        "contact_line_vertex_boundary_near_fraction": boundary_near_fraction,
    }


def read_contact_lines(table: Path, points_table: Path) -> tuple[list[dict], dict[int, np.ndarray]]:
    with Path(table).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points: dict[int, list[tuple[float, float]]] = {}
    with Path(points_table).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.setdefault(int(row["step"]), []).append((float(row["x_A"]), float(row["y_A"])))
    return rows, {step: np.asarray(values) for step, values in points.items()}


def write_plot(rows: Sequence[dict], output: Path, timestep_fs: float, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = properties.get_name()
    time = np.asarray([row["step"] for row in rows]) * timestep_fs / 1.0e6
    baseline = rows[0]["surface_ch3_fraction"]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].axhline(baseline, color="black", ls="--", lw=0.8, label="Surface")
    axes[0].plot(time, [row["footprint_ch3_fraction"] for row in rows], label="Footprint")
    axes[0].plot(time, [row["tpcl_ch3_fraction"] for row in rows], label="TPCL")
    axes[0].set_ylabel(r"CH$_3$ site fraction")
    axes[0].legend(frameon=False)
    axes[1].plot(time, [row["tpcl_ch3_enrichment"] for row in rows])
    axes[1].axhline(0.0, color="black", ls="--", lw=0.8)
    axes[1].set_ylabel(r"$E_{TPCL}$")
    axes[1].set_xlabel("Time (ns)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    elements, coordinates, lengths = read_extxyz_positions(args.initial_xyz)
    sites = identify_surface_sites(
        elements,
        coordinates,
        lengths,
        slab_range=args.slab_range,
        surface_z=args.surface_z_A,
        surface_depth=args.surface_depth_A,
        bond_cutoff=args.bond_cutoff_A,
    )
    contact_rows, boundaries = read_contact_lines(args.contact_line, args.contact_line_points)
    site_xy = np.asarray([[row["x_A"], row["y_A"]] for row in sites])
    site_types = np.asarray([row["site_type"] for row in sites])
    rows = []
    for source in contact_rows:
        step = int(source["step"])
        boundary = boundaries.get(step)
        if boundary is None or len(boundary) < 3:
            continue
        center = np.asarray(
            [float(source["contact_line_center_x_A"]), float(source["contact_line_center_y_A"])]
        )
        row = {"step": step}
        row.update(
            region_metrics(
                site_xy,
                site_types,
                boundary,
                center,
                lengths[:2],
                tpcl_half_width=args.tpcl_half_width_A,
                boundary_proximity=args.boundary_proximity_A,
            )
        )
        rows.append(row)
    if not rows:
        raise ValueError("No contact-line frame could be mapped to surface sites")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "surface_sites.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sites[0]))
        writer.writeheader(); writer.writerows(sites)
    with (output / "surface_site_enrichment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    ch3_count = sum(row["site_type"] == "CH3" for row in sites)
    sioh_count = len(sites) - ch3_count
    summary = {
        "status": "PASS",
        "mapped_frames": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "top_surface_ch3_sites": ch3_count,
        "top_surface_sioh_sites": sioh_count,
        "surface_ch3_fraction": ch3_count / len(sites),
        "mean_footprint_ch3_enrichment": float(np.nanmean([row["footprint_ch3_enrichment"] for row in rows])),
        "mean_tpcl_ch3_enrichment": float(np.nanmean([row["tpcl_ch3_enrichment"] for row in rows])),
        "chemical_boundary_metric_is_nearest_site_bisector_proxy": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "initial_xyz": str(Path(args.initial_xyz).resolve()),
        "slab_atom_range": list(args.slab_range),
        "surface_z_A": args.surface_z_A,
        "surface_depth_A": args.surface_depth_A,
        "bond_cutoff_A": args.bond_cutoff_A,
        "contact_line": str(Path(args.contact_line).resolve()),
        "contact_line_points": str(Path(args.contact_line_points).resolve()),
        "tpcl_half_width_A": args.tpcl_half_width_A,
        "boundary_proximity_A": args.boundary_proximity_A,
        "site_identity": "initial CH3 carbon and SiOH oxygen sites from C-H/O-H bonding",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(rows, output / "surface_site_enrichment.png", args.timestep_fs, args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-xyz", type=Path, required=True)
    parser.add_argument("--slab-range", type=parse_range, required=True)
    parser.add_argument("--contact-line", type=Path, required=True)
    parser.add_argument("--contact-line-points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--surface-depth-A", type=float, default=3.0)
    parser.add_argument("--bond-cutoff-A", type=float, default=1.25)
    parser.add_argument("--tpcl-half-width-A", type=float, default=4.0)
    parser.add_argument("--boundary-proximity-A", type=float, default=2.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.surface_depth_A,
        args.bond_cutoff_A,
        args.tpcl_half_width_A,
        args.boundary_proximity_A,
    ) <= 0:
        raise ValueError("All geometric cutoffs must be positive")
    print(json.dumps(run_analysis(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
