"""Track CH3 and SiOH orientations relative to global and local surface normals."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.interfacial_water_orientation import iter_orientation_frames
from molsimflow.postprocess.surface_proton_transfer import (
    assign_hydrogens_to_oxygen_or_carbon,
    hydrogen_ids_by_owner,
)
from molsimflow.postprocess.surface_reference import load_surface_reference
from molsimflow.postprocess.surface_site_enrichment import (
    identify_surface_sites,
    parse_range,
)

FRAME_METRICS = (
    "mean_site_axis_cos_global",
    "mean_site_axis_cos_local",
    "mean_site_axis_tilt_global_deg",
    "mean_site_axis_tilt_local_deg",
    "mean_terminal_height_A",
    "mean_local_normal_tilt_deg",
    "mean_local_plane_rms_A",
    "group_integrity_fraction",
    "mean_oh_axis_cos_global",
    "mean_oh_axis_cos_local",
    "mean_oh_axis_tilt_global_deg",
    "mean_oh_axis_tilt_local_deg",
    "mean_si_o_h_angle_deg",
)


@dataclass(frozen=True)
class FunctionalSite:
    terminal_atom_id: int
    site_type: str
    anchor_si_id: int
    initial_hydrogen_ids: tuple[int, ...]
    local_normal_si_ids: tuple[int, ...]


def _finite_mean(values: Sequence[float]) -> float:
    clean = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    return float(clean.mean()) if len(clean) else math.nan


def _angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if norm <= 0:
        return math.nan
    cosine = float(np.clip(np.dot(vector_a, vector_b) / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _tilt_deg(vector: np.ndarray, normal: np.ndarray) -> float:
    return _angle_deg(vector, normal)


def _azimuth_deg(vector: np.ndarray) -> float:
    if float(np.linalg.norm(vector[:2])) <= 1.0e-12:
        return math.nan
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0)


def _circular_stats_deg(values: Sequence[float]) -> tuple[float, float]:
    clean = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if not len(clean):
        return math.nan, math.nan
    unit = np.exp(1j * np.radians(clean))
    mean = unit.mean()
    return float(np.degrees(np.angle(mean)) % 360.0), float(abs(mean))


def identify_functional_sites(
    elements: np.ndarray,
    coordinates: np.ndarray,
    lengths: np.ndarray,
    *,
    surface_range: tuple[int, int],
    surface_z_A: float,
    surface_depth_A: float,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
    si_terminal_cutoff_A: float,
    local_normal_neighbors: int,
) -> list[FunctionalSite]:
    """Identify top terminations, their Si anchors, and frozen normal neighborhoods."""

    from scipy.spatial import cKDTree

    if local_normal_neighbors < 3:
        raise ValueError("At least three local-normal Si neighbors are required")
    start, end = surface_range[0] - 1, surface_range[1]
    if end > len(elements):
        raise ValueError("Surface range exceeds the initial structure atom count")
    raw_sites = identify_surface_sites(
        elements,
        coordinates,
        lengths,
        slab_range=surface_range,
        surface_z=surface_z_A,
        surface_depth=surface_depth_A,
        bond_cutoff=min(oh_cutoff_A, ch_cutoff_A),
    )
    slab_ids = np.arange(start + 1, end + 1, dtype=int)
    slab_elements = elements[start:end]
    slab_coordinates = coordinates[start:end]
    silicon_mask = slab_elements == "Si"
    hydrogen_mask = slab_elements == "H"
    carbon_mask = slab_elements == "C"
    oxygen_mask = slab_elements == "O"
    silicon_ids = slab_ids[silicon_mask]
    if len(silicon_ids) < local_normal_neighbors:
        raise ValueError("Too few slab Si atoms for the requested local-normal fit")

    bounds = np.column_stack((np.zeros(3), lengths))
    assignment = assign_hydrogens_to_oxygen_or_carbon(
        slab_ids[oxygen_mask],
        slab_coordinates[oxygen_mask],
        slab_ids[carbon_mask],
        slab_coordinates[carbon_mask],
        slab_ids[hydrogen_mask],
        slab_coordinates[hydrogen_mask],
        bounds,
        oh_cutoff_A=oh_cutoff_A,
        ch_cutoff_A=ch_cutoff_A,
    )
    grouped_h = hydrogen_ids_by_owner(assignment)
    silicon_tree = cKDTree(slab_coordinates[silicon_mask] % lengths, boxsize=lengths)
    terminal_ids = np.asarray([int(site["atom_id"]) for site in raw_sites], dtype=int)
    terminal_coordinates = coordinates[terminal_ids - 1]
    anchor_distance, anchor_local = silicon_tree.query(terminal_coordinates % lengths)
    if np.any(anchor_distance > si_terminal_cutoff_A):
        bad = terminal_ids[anchor_distance > si_terminal_cutoff_A]
        raise ValueError(f"Terminal atoms without a Si anchor inside cutoff: {bad[:10].tolist()}")
    anchor_ids = silicon_ids[np.asarray(anchor_local, dtype=int)]
    unique_anchor_ids = np.asarray(sorted(set(map(int, anchor_ids))), dtype=int)
    if len(unique_anchor_ids) < local_normal_neighbors:
        raise ValueError("Too few unique terminal-group Si anchors for local-normal fitting")
    anchor_xy = coordinates[unique_anchor_ids - 1, :2]
    normal_tree = cKDTree(anchor_xy % lengths[:2], boxsize=lengths[:2])
    _, neighbor_local = normal_tree.query(
        anchor_xy % lengths[:2], k=local_normal_neighbors
    )
    normal_neighbors = {
        int(anchor_id): tuple(map(int, unique_anchor_ids[np.atleast_1d(local_indices)]))
        for anchor_id, local_indices in zip(unique_anchor_ids, neighbor_local)
    }

    sites: list[FunctionalSite] = []
    for raw, terminal_id, anchor_id in zip(raw_sites, terminal_ids, anchor_ids):
        site_type = str(raw["site_type"])
        hydrogen_ids = grouped_h.get(int(terminal_id), ())
        expected = 3 if site_type == "CH3" else 1
        if len(hydrogen_ids) != expected:
            raise ValueError(
                f"Initial {site_type} atom {terminal_id} has {len(hydrogen_ids)} assigned H; "
                f"expected {expected}"
            )
        sites.append(
            FunctionalSite(
                terminal_atom_id=int(terminal_id),
                site_type=site_type,
                anchor_si_id=int(anchor_id),
                initial_hydrogen_ids=tuple(map(int, hydrogen_ids)),
                local_normal_si_ids=normal_neighbors[int(anchor_id)],
            )
        )
    return sites


def fit_local_normal(
    surface: np.ndarray,
    surface_start_id: int,
    anchor_si_id: int,
    neighbor_si_ids: Sequence[int],
    bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit a signed local plane to frozen neighboring terminal-group Si anchors."""

    lengths = box_lengths(bounds)
    anchor = surface[anchor_si_id - surface_start_id]
    indices = np.asarray(neighbor_si_ids, dtype=int) - surface_start_id
    if np.any(indices < 0) or np.any(indices >= len(surface)):
        raise ValueError("A local-normal Si ID lies outside the surface range")
    points = anchor + minimum_image_vectors(surface[indices] - anchor, lengths)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= 1.0e-12:
        return np.full(3, math.nan), math.nan
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal *= -1.0
    residual = centered @ normal
    return normal, float(np.sqrt(np.mean(residual**2)))


def _position(surface: np.ndarray, surface_start_id: int, atom_id: int) -> np.ndarray:
    index = atom_id - surface_start_id
    if index < 0 or index >= len(surface):
        raise ValueError(f"Surface atom ID {atom_id} lies outside the configured range")
    return surface[index]


def analyze_geometry(
    *,
    step: int,
    segment_index: int,
    bounds: np.ndarray,
    surface: np.ndarray,
    candidate_oxygen_ids: np.ndarray,
    candidate_oxygen: np.ndarray,
    hydrogen_ids: np.ndarray,
    hydrogen: np.ndarray,
    sites: Sequence[FunctionalSite],
    surface_range: tuple[int, int],
    plane_z_A: float,
    timestep_fs: float,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
    si_terminal_cutoff_A: float,
) -> list[dict]:
    """Measure all top-surface functional groups in one geometry."""

    lengths = box_lengths(bounds)
    carbon_ids = np.asarray(
        [site.terminal_atom_id for site in sites if site.site_type == "CH3"], dtype=int
    )
    carbon = np.asarray(
        [_position(surface, surface_range[0], atom_id) for atom_id in carbon_ids],
        dtype=float,
    ).reshape((-1, 3))
    assignment = assign_hydrogens_to_oxygen_or_carbon(
        candidate_oxygen_ids,
        candidate_oxygen,
        carbon_ids,
        carbon,
        hydrogen_ids,
        hydrogen,
        bounds,
        oh_cutoff_A=oh_cutoff_A,
        ch_cutoff_A=ch_cutoff_A,
    )
    grouped_h = hydrogen_ids_by_owner(assignment)
    hydrogen_index = {int(atom_id): index for index, atom_id in enumerate(hydrogen_ids)}
    rows = []
    global_normal = np.asarray([0.0, 0.0, 1.0])
    for site in sites:
        terminal = _position(surface, surface_range[0], site.terminal_atom_id)
        anchor = _position(surface, surface_range[0], site.anchor_si_id)
        site_axis = minimum_image_vectors(
            np.asarray([terminal - anchor]), lengths
        )[0]
        site_axis_length = float(np.linalg.norm(site_axis))
        normal, plane_rms = fit_local_normal(
            surface,
            surface_range[0],
            site.anchor_si_id,
            site.local_normal_si_ids,
            bounds,
        )
        normal_valid = bool(np.isfinite(normal).all())
        axis_valid = bool(0 < site_axis_length <= si_terminal_cutoff_A and normal_valid)
        assigned_h = grouped_h.get(site.terminal_atom_id, ())
        expected_h = 3 if site.site_type == "CH3" else 1
        integrity = len(assigned_h) == expected_h
        relative_z = minimum_image_vectors(
            np.asarray([[0.0, 0.0, terminal[2] - plane_z_A]]), lengths
        )[0, 2]
        row = {
            "step": int(step),
            "time_ns": step * timestep_fs / 1.0e6,
            "segment_index": int(segment_index),
            "site_type": site.site_type,
            "terminal_atom_id": site.terminal_atom_id,
            "anchor_si_id": site.anchor_si_id,
            "initial_hydrogen_ids": ";".join(map(str, site.initial_hydrogen_ids)),
            "current_hydrogen_ids": ";".join(map(str, assigned_h)),
            "group_h_count": len(assigned_h),
            "group_integrity": integrity,
            "site_axis_valid": axis_valid,
            "site_axis_bond_length_A": site_axis_length,
            "site_axis_cos_global": (
                float(site_axis[2] / site_axis_length) if axis_valid else math.nan
            ),
            "site_axis_cos_local": (
                float(np.dot(site_axis, normal) / site_axis_length)
                if axis_valid else math.nan
            ),
            "site_axis_tilt_global_deg": (
                _tilt_deg(site_axis, global_normal) if axis_valid else math.nan
            ),
            "site_axis_tilt_local_deg": (
                _tilt_deg(site_axis, normal) if axis_valid else math.nan
            ),
            "site_axis_azimuth_deg": _azimuth_deg(site_axis) if axis_valid else math.nan,
            "terminal_height_A": float(relative_z),
            "local_normal_valid": normal_valid,
            "local_normal_tilt_deg": (
                _tilt_deg(normal, global_normal) if normal_valid else math.nan
            ),
            "local_plane_rms_A": plane_rms,
            "local_normal_neighbor_count": len(site.local_normal_si_ids),
            "oh_h_atom_id": "",
            "oh_h_same_as_initial": "",
            "oh_bond_length_A": math.nan,
            "oh_axis_cos_global": math.nan,
            "oh_axis_cos_local": math.nan,
            "oh_axis_tilt_global_deg": math.nan,
            "oh_axis_tilt_local_deg": math.nan,
            "oh_axis_azimuth_deg": math.nan,
            "si_o_h_angle_deg": math.nan,
        }
        if site.site_type == "SiOH" and integrity:
            hydrogen_id = int(assigned_h[0])
            h_point = hydrogen[hydrogen_index[hydrogen_id]]
            oh_axis = minimum_image_vectors(
                np.asarray([h_point - terminal]), lengths
            )[0]
            oh_length = float(np.linalg.norm(oh_axis))
            oh_valid = 0 < oh_length <= oh_cutoff_A and normal_valid
            row.update(
                {
                    "oh_h_atom_id": hydrogen_id,
                    "oh_h_same_as_initial": hydrogen_id in site.initial_hydrogen_ids,
                    "oh_bond_length_A": oh_length,
                    "oh_axis_cos_global": (
                        float(oh_axis[2] / oh_length) if oh_valid else math.nan
                    ),
                    "oh_axis_cos_local": (
                        float(np.dot(oh_axis, normal) / oh_length)
                        if oh_valid else math.nan
                    ),
                    "oh_axis_tilt_global_deg": (
                        _tilt_deg(oh_axis, global_normal) if oh_valid else math.nan
                    ),
                    "oh_axis_tilt_local_deg": (
                        _tilt_deg(oh_axis, normal) if oh_valid else math.nan
                    ),
                    "oh_axis_azimuth_deg": (
                        _azimuth_deg(oh_axis) if oh_valid else math.nan
                    ),
                    "si_o_h_angle_deg": (
                        _angle_deg(-site_axis, oh_axis) if oh_valid and axis_valid else math.nan
                    ),
                }
            )
        rows.append(row)
    return rows


def summarize_frames(site_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in site_rows:
        grouped[(int(row["step"]), str(row["site_type"]))].append(row)
    output = []
    for (step, site_type), rows in sorted(grouped.items()):
        axis = [row for row in rows if row["site_axis_valid"]]
        intact = [row for row in rows if row["group_integrity"]]
        oh = [row for row in rows if math.isfinite(float(row["oh_axis_cos_global"]))]
        output.append(
            {
                "step": step,
                "time_ns": rows[0]["time_ns"],
                "segment_index": rows[0]["segment_index"],
                "site_type": site_type,
                "site_count": len(rows),
                "site_axis_valid_count": len(axis),
                "group_integrity_count": len(intact),
                "group_integrity_fraction": len(intact) / len(rows),
                "mean_site_axis_cos_global": _finite_mean(
                    [row["site_axis_cos_global"] for row in axis]
                ),
                "mean_site_axis_cos_local": _finite_mean(
                    [row["site_axis_cos_local"] for row in axis]
                ),
                "mean_site_axis_tilt_global_deg": _finite_mean(
                    [row["site_axis_tilt_global_deg"] for row in axis]
                ),
                "mean_site_axis_tilt_local_deg": _finite_mean(
                    [row["site_axis_tilt_local_deg"] for row in axis]
                ),
                "mean_terminal_height_A": _finite_mean(
                    [row["terminal_height_A"] for row in rows]
                ),
                "mean_local_normal_tilt_deg": _finite_mean(
                    [row["local_normal_tilt_deg"] for row in rows]
                ),
                "mean_local_plane_rms_A": _finite_mean(
                    [row["local_plane_rms_A"] for row in rows]
                ),
                "mean_oh_axis_cos_global": _finite_mean(
                    [row["oh_axis_cos_global"] for row in oh]
                ),
                "mean_oh_axis_cos_local": _finite_mean(
                    [row["oh_axis_cos_local"] for row in oh]
                ),
                "mean_oh_axis_tilt_global_deg": _finite_mean(
                    [row["oh_axis_tilt_global_deg"] for row in oh]
                ),
                "mean_oh_axis_tilt_local_deg": _finite_mean(
                    [row["oh_axis_tilt_local_deg"] for row in oh]
                ),
                "mean_si_o_h_angle_deg": _finite_mean(
                    [row["si_o_h_angle_deg"] for row in oh]
                ),
            }
        )
    return output


def build_blocks(frame_rows: Sequence[dict], block_frames: int) -> tuple[list[dict], int | None]:
    steps = sorted({int(row["step"]) for row in frame_rows})
    intervals = np.diff(steps)
    nominal_interval = int(np.median(intervals[intervals > 0])) if np.any(intervals > 0) else None
    rows_by_type: dict[str, list[dict]] = defaultdict(list)
    for row in frame_rows:
        rows_by_type[str(row["site_type"])].append(row)
    blocks = []
    block_index = 0
    for site_type, values in sorted(rows_by_type.items()):
        values.sort(key=lambda row: int(row["step"]))
        runs: list[list[dict]] = []
        run: list[dict] = []
        for row in values:
            discontinuity = bool(
                run
                and (
                    int(row["segment_index"]) != int(run[-1]["segment_index"])
                    or (
                        nominal_interval is not None
                        and int(row["step"]) - int(run[-1]["step"]) != nominal_interval
                    )
                )
            )
            if discontinuity:
                runs.append(run)
                run = []
            run.append(row)
        if run:
            runs.append(run)
        for run_rows in runs:
            for start in range(0, len(run_rows), block_frames):
                part = run_rows[start : start + block_frames]
                row = {
                    "block_index": block_index,
                    "site_type": site_type,
                    "segment_index": part[0]["segment_index"],
                    "first_step": part[0]["step"],
                    "last_step": part[-1]["step"],
                    "first_time_ns": part[0]["time_ns"],
                    "last_time_ns": part[-1]["time_ns"],
                    "frame_count": len(part),
                    "complete_block": len(part) == block_frames,
                }
                for metric in FRAME_METRICS:
                    row[metric] = _finite_mean([item[metric] for item in part])
                blocks.append(row)
                block_index += 1
    return blocks, nominal_interval


def build_site_summary(site_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in site_rows:
        grouped[int(row["terminal_atom_id"])].append(row)
    output = []
    for atom_id, rows in sorted(grouped.items()):
        axis = [row for row in rows if row["site_axis_valid"]]
        intact = [row for row in rows if row["group_integrity"]]
        oh = [row for row in rows if math.isfinite(float(row["oh_axis_cos_global"]))]
        output.append(
            {
                "terminal_atom_id": atom_id,
                "anchor_si_id": rows[0]["anchor_si_id"],
                "site_type": rows[0]["site_type"],
                "sample_count": len(rows),
                "site_axis_valid_fraction": len(axis) / len(rows),
                "group_integrity_fraction": len(intact) / len(rows),
                "mean_site_axis_cos_global": _finite_mean(
                    [row["site_axis_cos_global"] for row in axis]
                ),
                "mean_site_axis_cos_local": _finite_mean(
                    [row["site_axis_cos_local"] for row in axis]
                ),
                "mean_site_axis_tilt_global_deg": _finite_mean(
                    [row["site_axis_tilt_global_deg"] for row in axis]
                ),
                "mean_site_axis_tilt_local_deg": _finite_mean(
                    [row["site_axis_tilt_local_deg"] for row in axis]
                ),
                "mean_terminal_height_A": _finite_mean(
                    [row["terminal_height_A"] for row in rows]
                ),
                "mean_oh_axis_cos_global": _finite_mean(
                    [row["oh_axis_cos_global"] for row in oh]
                ),
                "mean_oh_axis_cos_local": _finite_mean(
                    [row["oh_axis_cos_local"] for row in oh]
                ),
                "mean_si_o_h_angle_deg": _finite_mean(
                    [row["si_o_h_angle_deg"] for row in oh]
                ),
            }
        )
    return output


def build_histograms(site_rows: Sequence[dict], cosine_bins: int, azimuth_bins: int) -> list[dict]:
    specs = (
        ("site_axis", "global", "site_axis_cos_global", -1.0, 1.0, cosine_bins),
        ("site_axis", "local", "site_axis_cos_local", -1.0, 1.0, cosine_bins),
        ("oh_axis", "global", "oh_axis_cos_global", -1.0, 1.0, cosine_bins),
        ("oh_axis", "local", "oh_axis_cos_local", -1.0, 1.0, cosine_bins),
        ("site_axis_azimuth", "global", "site_axis_azimuth_deg", 0.0, 360.0, azimuth_bins),
        ("oh_axis_azimuth", "global", "oh_axis_azimuth_deg", 0.0, 360.0, azimuth_bins),
    )
    output = []
    for site_type in sorted({str(row["site_type"]) for row in site_rows}):
        subset = [row for row in site_rows if row["site_type"] == site_type]
        for observable, reference, column, lower, upper, bins in specs:
            values = np.asarray(
                [float(row[column]) for row in subset if math.isfinite(float(row[column]))],
                dtype=float,
            )
            if not len(values):
                continue
            counts, edges = np.histogram(values, bins=bins, range=(lower, upper))
            for index, count in enumerate(counts):
                output.append(
                    {
                        "site_type": site_type,
                        "observable": observable,
                        "reference": reference,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                        "probability": float(count / len(values)),
                        "sample_count": len(values),
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _configure_matplotlib(font_path: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    if not font_path.is_file():
        raise FileNotFoundError(f"Font file does not exist: {font_path}")
    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams.update({"font.family": properties.get_name(), "font.size": 8})
    return plt


def write_plots(
    site_rows: Sequence[dict], frame_rows: Sequence[dict], output: Path, font_path: Path
) -> None:
    plt = _configure_matplotlib(font_path)
    output.mkdir(parents=True, exist_ok=True)
    colors = {"CH3": "#4C78A8", "SiOH": "#F58518"}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for axis, reference, column in zip(
        axes,
        ("global +z", "local Si plane"),
        ("site_axis_cos_global", "site_axis_cos_local"),
    ):
        for site_type in sorted({row["site_type"] for row in site_rows}):
            values = [
                row[column]
                for row in site_rows
                if row["site_type"] == site_type and math.isfinite(float(row[column]))
            ]
            axis.hist(
                values,
                bins=np.linspace(-1, 1, 41),
                density=True,
                histtype="step",
                lw=1.1,
                color=colors[site_type],
                label=site_type,
            )
        axis.set_xlabel(r"$\cos\theta$ (Si$\rightarrow$terminal)")
        axis.set_title(reference)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Probability density")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "site_axis_cosine_distribution.png", dpi=300)
    plt.close(figure)

    oh_rows = [row for row in site_rows if math.isfinite(float(row["oh_axis_cos_global"]))]
    if oh_rows:
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for axis, reference, column in zip(
            axes,
            ("global +z", "local Si plane"),
            ("oh_axis_cos_global", "oh_axis_cos_local"),
        ):
            axis.hist(
                [row[column] for row in oh_rows],
                bins=np.linspace(-1, 1, 41),
                density=True,
                histtype="step",
                lw=1.1,
                color=colors["SiOH"],
            )
            axis.set_xlabel(r"$\cos\theta$ (O$\rightarrow$H)")
            axis.set_title(reference)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Probability density")
        figure.tight_layout()
        figure.savefig(output / "oh_axis_cosine_distribution.png", dpi=300)
        plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for site_type in sorted({row["site_type"] for row in frame_rows}):
        rows = [row for row in frame_rows if row["site_type"] == site_type]
        axes[0].plot(
            [row["time_ns"] for row in rows],
            [row["mean_site_axis_cos_global"] for row in rows],
            lw=0.8,
            color=colors[site_type],
            label=site_type,
        )
        axes[1].plot(
            [row["time_ns"] for row in rows],
            [row["group_integrity_fraction"] for row in rows],
            lw=0.8,
            color=colors[site_type],
        )
    axes[0].set_ylabel(r"Frame mean $\cos\theta$")
    axes[1].set_ylabel("Group integrity")
    axes[1].set_xlabel("Time (ns)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "orientation_time_series.png", dpi=300)
    plt.close(figure)


def _summary_by_type(
    frame_rows: Sequence[dict],
    block_rows: Sequence[dict],
    initial_rows: Sequence[dict],
) -> dict:
    output = {}
    initial_frames = summarize_frames(initial_rows)
    for site_type in sorted({str(row["site_type"]) for row in frame_rows}):
        frames = [row for row in frame_rows if row["site_type"] == site_type]
        complete = [
            row
            for row in block_rows
            if row["site_type"] == site_type and row["complete_block"]
        ]
        initial = next(row for row in initial_frames if row["site_type"] == site_type)
        payload = {
            "site_count": int(frames[0]["site_count"]),
            "frame_count": len(frames),
            "complete_block_count": len(complete),
            "minimum_group_integrity_fraction": float(
                min(row["group_integrity_fraction"] for row in frames)
            ),
        }
        for metric in FRAME_METRICS:
            dynamic = _finite_mean([row[metric] for row in frames])
            block_values = [
                float(row[metric])
                for row in complete
                if math.isfinite(float(row[metric]))
            ]
            initial_value = float(initial[metric])
            payload[metric] = {
                "initial": initial_value,
                "dynamic_frame_mean": dynamic,
                "dynamic_minus_initial": (
                    dynamic - initial_value
                    if math.isfinite(dynamic) and math.isfinite(initial_value)
                    else math.nan
                ),
                "complete_block_mean": _finite_mean(block_values),
                "complete_block_std": (
                    float(np.std(block_values, ddof=1)) if len(block_values) > 1 else math.nan
                ),
            }
        return_output = payload
        output[site_type] = return_output
    return output


def run_analysis(args: argparse.Namespace) -> dict:
    elements, initial_coordinates, initial_lengths = read_extxyz_positions(args.initial_xyz)
    sites = identify_functional_sites(
        elements,
        initial_coordinates,
        initial_lengths,
        surface_range=args.surface_range,
        surface_z_A=args.surface_z_A,
        surface_depth_A=args.surface_depth_A,
        oh_cutoff_A=args.oh_cutoff_A,
        ch_cutoff_A=args.ch_cutoff_A,
        si_terminal_cutoff_A=args.si_terminal_cutoff_A,
        local_normal_neighbors=args.local_normal_neighbors,
    )
    counts = {
        "CH3": sum(site.site_type == "CH3" for site in sites),
        "SiOH": sum(site.site_type == "SiOH" for site in sites),
    }
    if args.expected_ch3_sites is not None and counts["CH3"] != args.expected_ch3_sites:
        raise ValueError(f"Expected {args.expected_ch3_sites} CH3 sites, found {counts['CH3']}")
    if args.expected_sioh_sites is not None and counts["SiOH"] != args.expected_sioh_sites:
        raise ValueError(f"Expected {args.expected_sioh_sites} SiOH sites, found {counts['SiOH']}")

    initial_bounds = np.column_stack((np.zeros(3), initial_lengths))
    surface_start, surface_end = args.surface_range
    initial_surface = initial_coordinates[surface_start - 1 : surface_end]
    all_ids = np.arange(1, len(elements) + 1, dtype=int)
    initial_oxygen_mask = elements == "O"
    initial_hydrogen_mask = elements == "H"
    initial_rows = analyze_geometry(
        step=0,
        segment_index=-1,
        bounds=initial_bounds,
        surface=initial_surface,
        candidate_oxygen_ids=all_ids[initial_oxygen_mask],
        candidate_oxygen=initial_coordinates[initial_oxygen_mask],
        hydrogen_ids=all_ids[initial_hydrogen_mask],
        hydrogen=initial_coordinates[initial_hydrogen_mask],
        sites=sites,
        surface_range=args.surface_range,
        plane_z_A=args.surface_z_A,
        timestep_fs=args.timestep_fs,
        oh_cutoff_A=args.oh_cutoff_A,
        ch_cutoff_A=args.ch_cutoff_A,
        si_terminal_cutoff_A=args.si_terminal_cutoff_A,
    )

    surface_reference = load_surface_reference(
        args.initial_xyz, args.surface_range, args.surface_z_A
    )
    records = {}
    raw_frames = 0
    raw_steps = []
    stop_early = False
    for segment_index, trajectory in enumerate(args.trajectory):
        for frame in iter_orientation_frames(
            trajectory,
            args.surface_range,
            args.water_range,
            oxygen_type=args.oxygen_type,
            hydrogen_type=args.hydrogen_type,
        ):
            raw_frames += 1
            raw_steps.append(frame.step)
            plane_z = surface_reference.plane_z(frame.surface, frame.bounds)
            records[frame.step] = analyze_geometry(
                step=frame.step,
                segment_index=segment_index,
                bounds=frame.bounds,
                surface=frame.surface,
                candidate_oxygen_ids=frame.candidate_oxygen_ids,
                candidate_oxygen=frame.candidate_oxygen,
                hydrogen_ids=frame.hydrogen_ids,
                hydrogen=frame.hydrogen,
                sites=sites,
                surface_range=args.surface_range,
                plane_z_A=plane_z,
                timestep_fs=args.timestep_fs,
                oh_cutoff_A=args.oh_cutoff_A,
                ch_cutoff_A=args.ch_cutoff_A,
                si_terminal_cutoff_A=args.si_terminal_cutoff_A,
            )
            selected_count = len(records) - int(args.drop_first_frame and 0 in records)
            if args.max_frames is not None and selected_count >= args.max_frames:
                stop_early = True
                break
        if stop_early:
            break
    steps = sorted(records)
    if args.drop_first_frame and steps and steps[0] == 0:
        steps = steps[1:]
    if args.max_frames is not None:
        steps = steps[: args.max_frames]
    if not steps:
        raise ValueError("No trajectory frames remain after selection")
    site_rows = [row for step in steps for row in records[step]]
    frame_rows = summarize_frames(site_rows)
    block_rows, nominal_interval = build_blocks(frame_rows, args.block_frames)
    site_summary = build_site_summary(site_rows)
    histograms = build_histograms(site_rows, args.cosine_bins, args.azimuth_bins)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "initial_site_orientations.csv", initial_rows)
    _write_csv(output / "site_orientations.csv.gz", site_rows)
    _write_csv(output / "frame_summary.csv", frame_rows)
    _write_csv(output / "block_summary.csv", block_rows)
    _write_csv(output / "site_summary.csv", site_summary)
    _write_csv(output / "orientation_histograms.csv", histograms)

    by_type = _summary_by_type(frame_rows, block_rows, initial_rows)
    integrity_gate = all(
        payload["minimum_group_integrity_fraction"] >= args.minimum_group_integrity_fraction
        for payload in by_type.values()
    )
    axis_gate = all(
        all(row["site_axis_valid_count"] == row["site_count"] for row in frame_rows if row["site_type"] == site_type)
        for site_type in by_type
    )
    block_gate = all(payload["complete_block_count"] >= 2 for payload in by_type.values())
    status = "PASS" if integrity_gate and axis_gate and block_gate else "FAIL"
    azimuth = {}
    for site_type in by_type:
        rows = [row for row in site_rows if row["site_type"] == site_type]
        azimuth[site_type] = {}
        for name, column in (
            ("site_axis", "site_axis_azimuth_deg"),
            ("oh_axis", "oh_axis_azimuth_deg"),
        ):
            mean, resultant = _circular_stats_deg([row[column] for row in rows])
            azimuth[site_type][name] = {
                "circular_mean_deg": mean,
                "resultant_length": resultant,
            }
    summary = {
        "status": status,
        "raw_frame_count": raw_frames,
        "unique_frame_count_before_drop": len(records),
        "analyzed_frame_count": len(steps),
        "duplicate_raw_frame_count": raw_frames - len(set(raw_steps)),
        "first_step": steps[0],
        "last_step": steps[-1],
        "nominal_frame_interval_steps": nominal_interval,
        "nominal_frame_interval_ps": (
            nominal_interval * args.timestep_fs / 1000.0
            if nominal_interval is not None else None
        ),
        "site_counts": counts,
        "by_site_type": by_type,
        "global_azimuth_circular_statistics": azimuth,
        "gates": {
            "group_integrity": integrity_gate,
            "site_axis_bond_and_local_normal": axis_gate,
            "at_least_two_complete_blocks_per_present_site_type": block_gate,
        },
        "estimator_notes": {
            "site_axis": "Si-to-terminal C or O vector",
            "sioh_axis": "O-to-current assigned H when exactly one H is assigned",
            "ch3_hydrogens": "used only for three-H integrity gating",
            "local_normal": "PCA plane through frozen neighboring terminal-group Si anchors",
            "uncertainty": "standard deviation of complete block means; descriptive single-trajectory estimator",
            "histograms": "direct counts in cos(theta) or azimuth bins without smoothing or interpolation",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "initial_xyz": str(Path(args.initial_xyz).resolve()),
        "surface_range": list(args.surface_range),
        "water_range": list(args.water_range),
        "surface_z_A": args.surface_z_A,
        "surface_depth_A": args.surface_depth_A,
        "oh_cutoff_A": args.oh_cutoff_A,
        "ch_cutoff_A": args.ch_cutoff_A,
        "si_terminal_cutoff_A": args.si_terminal_cutoff_A,
        "local_normal_neighbors": args.local_normal_neighbors,
        "block_frames": args.block_frames,
        "timestep_fs": args.timestep_fs,
        "drop_first_frame": args.drop_first_frame,
        "restart_policy": "later trajectory replaces earlier duplicate timestep",
        "no_gap_interpolation": True,
        "expected_ch3_sites": args.expected_ch3_sites,
        "expected_sioh_sites": args.expected_sioh_sites,
        "minimum_group_integrity_fraction": args.minimum_group_integrity_fraction,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_plots:
        write_plots(site_rows, frame_rows, output / "figures", args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-xyz", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--surface-z-A", type=float, required=True)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--surface-depth-A", type=float, default=3.0)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.25)
    parser.add_argument("--ch-cutoff-A", type=float, default=1.30)
    parser.add_argument("--si-terminal-cutoff-A", type=float, default=2.20)
    parser.add_argument("--local-normal-neighbors", type=int, default=7)
    parser.add_argument("--block-frames", type=int, default=25)
    parser.add_argument("--cosine-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=36)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--expected-ch3-sites", type=int)
    parser.add_argument("--expected-sioh-sites", type=int)
    parser.add_argument("--minimum-group-integrity-fraction", type=float, default=0.99)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--drop-first-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = (
        args.surface_depth_A,
        args.oh_cutoff_A,
        args.ch_cutoff_A,
        args.si_terminal_cutoff_A,
        args.local_normal_neighbors,
        args.block_frames,
        args.cosine_bins,
        args.azimuth_bins,
        args.timestep_fs,
    )
    if min(positive) <= 0:
        raise ValueError("All cutoffs, counts, bin counts, and timestep must be positive")
    if not 0 < args.minimum_group_integrity_fraction <= 1:
        raise ValueError("minimum_group_integrity_fraction must lie in (0, 1]")
    if not args.no_plots and args.font_path is None:
        raise ValueError("--font-path is required unless --no-plots is used")
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
