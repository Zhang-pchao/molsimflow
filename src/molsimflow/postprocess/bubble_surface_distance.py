#!/usr/bin/env python3
"""Analyze two-bubble surface-to-surface distance in LAMMPS trajectories."""

from __future__ import annotations

import argparse
import bisect
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.centroids import BubbleCentroidCalculator


def _get_pyplot():
    """Import matplotlib lazily so non-plot helpers can be imported without it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class FrameDistanceResult:
    """Per-frame two-bubble distance metrics."""

    frame_index: int
    timestep: int
    time_fs: float
    time_ps: float
    b1_size: int
    b2_size: int
    b1_surface_size: int
    b2_surface_size: int
    centroid_distance: float
    min_surface_distance: float
    b1_radius_max: float
    b2_radius_max: float
    gap_rmax: float
    b1_radius_p90: float
    b2_radius_p90: float
    gap_r90: float


class BubbleSurfaceDistanceAnalyzer:
    """Analyze bubble centroid and surface distances for two-bubble coalescence."""

    def __init__(
        self,
        cutoff_distance: float = 5.5,
        surface_fraction: float = 0.8,
        nitrogen_type: Optional[int] = None,
        min_cluster_size: int = 1,
        fs_per_step: float = 1.0,
    ) -> None:
        if surface_fraction <= 0.0 or surface_fraction > 1.0:
            raise ValueError("surface_fraction must be in (0, 1]")
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1")
        if fs_per_step <= 0.0:
            raise ValueError("fs_per_step must be > 0")

        self.surface_fraction = float(surface_fraction)
        self.min_cluster_size = int(min_cluster_size)
        self.fs_per_step = float(fs_per_step)
        self.centroid_helper = BubbleCentroidCalculator(cutoff_distance=cutoff_distance)

        if nitrogen_type is not None:
            self.centroid_helper.nitrogen_type = int(nitrogen_type)

        if self.centroid_helper.nitrogen_type is None:
            raise ValueError("Nitrogen atom type is not configured")

        self.logger = logging.getLogger(__name__)

    def _radial_distances(self, coords: np.ndarray, center: np.ndarray, box_dims: np.ndarray) -> np.ndarray:
        """Return periodic radial distances from `center` to each coordinate in `coords`."""
        delta = coords - center
        delta = delta - box_dims * np.round(delta / box_dims)
        return np.linalg.norm(delta, axis=1)

    def _surface_coords(self, coords: np.ndarray, centroid: np.ndarray, box_dims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select surface atoms based on radial distance from centroid."""
        if len(coords) == 0:
            return coords, np.array([], dtype=float)

        radii = self._radial_distances(coords, centroid, box_dims)
        r_max = float(np.max(radii))
        threshold = self.surface_fraction * r_max
        mask = radii >= threshold

        if not np.any(mask):
            mask[np.argmax(radii)] = True

        return coords[mask], radii

    def _minimum_pair_distance_pbc(self, coords_a: np.ndarray, coords_b: np.ndarray, box_dims: np.ndarray) -> float:
        """Compute minimum pairwise distance between two coordinate sets under PBC."""
        if len(coords_a) == 0 or len(coords_b) == 0:
            return float("nan")

        delta = coords_a[:, None, :] - coords_b[None, :, :]
        delta = delta - box_dims[None, None, :] * np.round(delta / box_dims[None, None, :])
        dist_mat = np.linalg.norm(delta, axis=2)
        return float(np.min(dist_mat))

    def _format_float(self, value: float) -> str:
        if value is None or not np.isfinite(value):
            return "nan"
        return f"{value:.6f}"

    def _time_from_timestep(self, timestep: int) -> Tuple[float, float]:
        time_fs = float(timestep) * self.fs_per_step
        time_ps = time_fs / 1000.0
        return time_fs, time_ps

    def _read_dump_timesteps(self, traj_file: str, expected_frames: Optional[int] = None) -> List[int]:
        """Parse LAMMPS dump TIMESTEP values in frame order."""
        timesteps: List[int] = []

        with open(traj_file, "r") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break

                if line.startswith("ITEM: TIMESTEP"):
                    step_line = handle.readline()
                    if not step_line:
                        break
                    token = step_line.strip().split()[0]
                    timesteps.append(int(token))

                    if expected_frames is not None and len(timesteps) >= expected_frames:
                        break

        return timesteps

    def _resolve_timestep(
        self,
        frame_index: int,
        ts,
        parsed_timesteps: Sequence[int],
    ) -> int:
        if frame_index < len(parsed_timesteps):
            return int(parsed_timesteps[frame_index])

        if hasattr(ts, "data") and isinstance(ts.data, dict) and ("step" in ts.data):
            return int(ts.data["step"])

        return int(frame_index)

    def _build_empty_result(
        self,
        frame_index: int,
        timestep: int,
        b1_size: int = 0,
        b2_size: int = 0,
    ) -> FrameDistanceResult:
        time_fs, time_ps = self._time_from_timestep(timestep)
        return FrameDistanceResult(
            frame_index=frame_index,
            timestep=timestep,
            time_fs=time_fs,
            time_ps=time_ps,
            b1_size=b1_size,
            b2_size=b2_size,
            b1_surface_size=0,
            b2_surface_size=0,
            centroid_distance=float("nan"),
            min_surface_distance=float("nan"),
            b1_radius_max=float("nan"),
            b2_radius_max=float("nan"),
            gap_rmax=float("nan"),
            b1_radius_p90=float("nan"),
            b2_radius_p90=float("nan"),
            gap_r90=float("nan"),
        )

    def _analyze_frame(
        self,
        frame_index: int,
        timestep: int,
        n_coords: np.ndarray,
        box_dims: np.ndarray,
    ) -> FrameDistanceResult:
        clusters = self.centroid_helper.cluster_nitrogen_atoms(n_coords, box_dims)
        if not clusters:
            return self._build_empty_result(frame_index=frame_index, timestep=timestep)

        b1_size = len(clusters[0])
        b2_size = len(clusters[1]) if len(clusters) > 1 else 0
        if len(clusters) < 2 or b1_size < self.min_cluster_size or b2_size < self.min_cluster_size:
            return self._build_empty_result(
                frame_index=frame_index,
                timestep=timestep,
                b1_size=b1_size,
                b2_size=b2_size,
            )

        c1 = n_coords[clusters[0]]
        c2 = n_coords[clusters[1]]

        centroid_1 = self.centroid_helper.calculate_centroid_pbc(c1, box_dims)
        centroid_2 = self.centroid_helper.calculate_centroid_pbc(c2, box_dims)

        r1_surface_coords, r1_all = self._surface_coords(c1, centroid_1, box_dims)
        r2_surface_coords, r2_all = self._surface_coords(c2, centroid_2, box_dims)

        d_centroid = self.centroid_helper.periodic_distance(centroid_1, centroid_2, box_dims)
        d_surface_min = self._minimum_pair_distance_pbc(r1_surface_coords, r2_surface_coords, box_dims)

        r1_max = float(np.max(r1_all)) if len(r1_all) else float("nan")
        r2_max = float(np.max(r2_all)) if len(r2_all) else float("nan")
        r1_p90 = float(np.percentile(r1_all, 90.0)) if len(r1_all) else float("nan")
        r2_p90 = float(np.percentile(r2_all, 90.0)) if len(r2_all) else float("nan")

        gap_rmax = float(d_centroid - (r1_max + r2_max)) if np.isfinite(r1_max) and np.isfinite(r2_max) else float("nan")
        gap_r90 = float(d_centroid - (r1_p90 + r2_p90)) if np.isfinite(r1_p90) and np.isfinite(r2_p90) else float("nan")

        time_fs, time_ps = self._time_from_timestep(timestep)
        return FrameDistanceResult(
            frame_index=frame_index,
            timestep=timestep,
            time_fs=time_fs,
            time_ps=time_ps,
            b1_size=b1_size,
            b2_size=b2_size,
            b1_surface_size=len(r1_surface_coords),
            b2_surface_size=len(r2_surface_coords),
            centroid_distance=float(d_centroid),
            min_surface_distance=float(d_surface_min),
            b1_radius_max=r1_max,
            b2_radius_max=r2_max,
            gap_rmax=gap_rmax,
            b1_radius_p90=r1_p90,
            b2_radius_p90=r2_p90,
            gap_r90=gap_r90,
        )

    def _match_step_index(
        self,
        target_step: int,
        steps: np.ndarray,
        step_to_index: Dict[int, int],
        tolerance_steps: int,
    ) -> Optional[int]:
        if target_step in step_to_index:
            return step_to_index[target_step]

        if tolerance_steps <= 0 or steps.size == 0:
            return None

        pos = bisect.bisect_left(steps, target_step)
        candidate_positions = []
        if pos < len(steps):
            candidate_positions.append(pos)
        if pos > 0:
            candidate_positions.append(pos - 1)

        best_idx = None
        best_delta = None
        for cpos in candidate_positions:
            delta = abs(int(steps[cpos]) - int(target_step))
            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_idx = cpos

        if best_idx is None or best_delta is None:
            return None

        if best_delta <= tolerance_steps:
            return int(best_idx)

        return None

    def _write_colvar_surface_distance(
        self,
        colvar_file: str,
        colvar_output: str,
        results: List[FrameDistanceResult],
        match_tolerance_steps: int,
    ) -> None:
        if not os.path.exists(colvar_file):
            raise FileNotFoundError(f"COLVAR file not found: {colvar_file}")

        header_lines: List[str] = []
        data_rows: List[Tuple[int, List[str]]] = []

        with open(colvar_file, "r") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("#"):
                    header_lines.append(line)
                    continue

                cols = stripped.split()
                if not cols:
                    continue

                try:
                    time_ps = float(cols[0])
                except ValueError:
                    continue

                step = int(round(time_ps * 1000.0 / self.fs_per_step))
                data_rows.append((step, cols))

        if not data_rows:
            raise ValueError(f"No data rows parsed from COLVAR: {colvar_file}")

        colvar_steps = np.array([row[0] for row in data_rows], dtype=int)
        step_to_index = {}
        for idx, step in enumerate(colvar_steps):
            if int(step) not in step_to_index:
                step_to_index[int(step)] = idx

        os.makedirs(os.path.dirname(os.path.abspath(colvar_output)) or ".", exist_ok=True)

        written = 0
        missing = 0
        used_indices = set()

        with open(colvar_output, "w") as out:
            wrote_fields_header = False
            for hline in header_lines:
                if hline.startswith("#!") and ("FIELDS" in hline) and (not wrote_fields_header):
                    out.write(hline + " surface_min_dist\n")
                    wrote_fields_header = True
                else:
                    out.write(hline + "\n")

            if not wrote_fields_header:
                out.write("#! FIELDS time surface_min_dist\n")

            for row in results:
                idx = self._match_step_index(
                    target_step=row.timestep,
                    steps=colvar_steps,
                    step_to_index=step_to_index,
                    tolerance_steps=match_tolerance_steps,
                )
                if idx is None:
                    missing += 1
                    continue

                if idx in used_indices:
                    # Avoid writing duplicated COLVAR rows when two trajectory frames map to same step.
                    continue

                used_indices.add(idx)
                cols = list(data_rows[idx][1])
                cols.append(self._format_float(row.min_surface_distance))
                out.write(" ".join(cols) + "\n")
                written += 1

        self.logger.info(
            "Saved COLVAR surface distance file: %s (written=%d, unmatched=%d)",
            colvar_output,
            written,
            missing,
        )

    def analyze_trajectory(
        self,
        traj_file: str,
        output_file: str,
        data_file: Optional[str] = None,
        atom_style: Optional[str] = None,
        step_interval: int = 1,
        start_frame: int = 0,
        end_frame: int = -1,
        max_frames: Optional[int] = None,
        make_plot: bool = True,
        colvar_file: Optional[str] = None,
        colvar_output: Optional[str] = None,
        colvar_match_tolerance_steps: int = 0,
    ) -> List[FrameDistanceResult]:
        """Run two-bubble surface distance analysis on a trajectory."""
        universe = self.centroid_helper.read_lammps_with_mda(
            traj_file=traj_file,
            data_file=data_file,
            atom_style=atom_style,
        )

        total_frames = len(universe.trajectory)
        actual_end_frame = total_frames if end_frame == -1 else min(end_frame, total_frames)
        frame_indices = list(range(start_frame, actual_end_frame, step_interval))

        if max_frames is not None and max_frames > 0:
            frame_indices = frame_indices[:max_frames]

        if not frame_indices:
            raise ValueError("No frames selected for analysis")

        output_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        os.makedirs(output_dir, exist_ok=True)

        parsed_timesteps = self._read_dump_timesteps(traj_file=traj_file, expected_frames=total_frames)
        if len(parsed_timesteps) < total_frames:
            self.logger.warning(
                "Parsed dump timesteps (%d) fewer than trajectory frames (%d); fallback methods will be used",
                len(parsed_timesteps),
                total_frames,
            )

        self.logger.info("Starting bubble surface distance analysis")
        self.logger.info("Trajectory: %s", traj_file)
        self.logger.info("Selected frames: %d", len(frame_indices))
        self.logger.info("Surface fraction: %.3f", self.surface_fraction)
        self.logger.info("Time scale: %.6f fs per step", self.fs_per_step)

        results: List[FrameDistanceResult] = []

        for i, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            ts = universe.trajectory.ts
            box_dims = np.asarray(universe.dimensions[:3], dtype=float)

            timestep = self._resolve_timestep(frame_index=frame_index, ts=ts, parsed_timesteps=parsed_timesteps)

            n_atoms = universe.select_atoms(f"type {self.centroid_helper.nitrogen_type}")
            if len(n_atoms) == 0:
                result = self._build_empty_result(frame_index=frame_index, timestep=timestep)
                results.append(result)
                continue

            result = self._analyze_frame(
                frame_index=frame_index,
                timestep=timestep,
                n_coords=np.asarray(n_atoms.positions, dtype=float),
                box_dims=box_dims,
            )
            results.append(result)

            if i < 3 or (i + 1) % 20 == 0 or (i + 1) == len(frame_indices):
                self.logger.info(
                    "Processed %d/%d frames (frame=%d, step=%d, c2c=%s, s2s=%s)",
                    i + 1,
                    len(frame_indices),
                    frame_index,
                    timestep,
                    self._format_float(result.centroid_distance),
                    self._format_float(result.min_surface_distance),
                )

        self._save_table(output_file=output_file, results=results)
        self._save_stats(output_file=output_file, results=results, total_frames=total_frames, selected_frames=len(frame_indices))
        self._save_timeseries_data(output_file=output_file, results=results)

        if make_plot:
            self._plot_timeseries(output_file=output_file, results=results)

        if colvar_file:
            if colvar_output is None:
                colvar_output = os.path.join(output_dir, "COLVAR_surf_dis")
            self._write_colvar_surface_distance(
                colvar_file=colvar_file,
                colvar_output=colvar_output,
                results=results,
                match_tolerance_steps=int(colvar_match_tolerance_steps),
            )

        return results

    def _save_table(self, output_file: str, results: List[FrameDistanceResult]) -> None:
        with open(output_file, "w") as handle:
            handle.write(
                "# FrameIndex Step Time(fs) Time(ps) B1_Size B2_Size B1_Surface_Size B2_Surface_Size "
                "Centroid_Dist(A) Surface_Min_Dist(A) B1_Rmax(A) B2_Rmax(A) Gap_Rmax(A) "
                "B1_R90(A) B2_R90(A) Gap_R90(A)\n"
            )
            for row in results:
                handle.write(
                    "{} {} {:.6f} {:.6f} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                        row.frame_index,
                        row.timestep,
                        row.time_fs,
                        row.time_ps,
                        row.b1_size,
                        row.b2_size,
                        row.b1_surface_size,
                        row.b2_surface_size,
                        self._format_float(row.centroid_distance),
                        self._format_float(row.min_surface_distance),
                        self._format_float(row.b1_radius_max),
                        self._format_float(row.b2_radius_max),
                        self._format_float(row.gap_rmax),
                        self._format_float(row.b1_radius_p90),
                        self._format_float(row.b2_radius_p90),
                        self._format_float(row.gap_r90),
                    )
                )

        self.logger.info("Saved per-frame distance table: %s", output_file)

    def _save_timeseries_data(self, output_file: str, results: List[FrameDistanceResult]) -> None:
        csv_file = output_file.replace(".txt", "_timeseries.csv")
        with open(csv_file, "w") as handle:
            handle.write(
                "frame_index,step,time_fs,time_ps,b1_size,b2_size,b1_surface_size,b2_surface_size,"
                "centroid_distance,surface_min_distance,b1_radius_max,b2_radius_max,gap_rmax,"
                "b1_radius_p90,b2_radius_p90,gap_r90\n"
            )
            for row in results:
                handle.write(
                    "{},{},{:.6f},{:.6f},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        row.frame_index,
                        row.timestep,
                        row.time_fs,
                        row.time_ps,
                        row.b1_size,
                        row.b2_size,
                        row.b1_surface_size,
                        row.b2_surface_size,
                        self._format_float(row.centroid_distance),
                        self._format_float(row.min_surface_distance),
                        self._format_float(row.b1_radius_max),
                        self._format_float(row.b2_radius_max),
                        self._format_float(row.gap_rmax),
                        self._format_float(row.b1_radius_p90),
                        self._format_float(row.b2_radius_p90),
                        self._format_float(row.gap_r90),
                    )
                )

        self.logger.info("Saved time-series CSV: %s", csv_file)

    def _save_stats(
        self,
        output_file: str,
        results: List[FrameDistanceResult],
        total_frames: int,
        selected_frames: int,
    ) -> None:
        stats_file = output_file.replace(".txt", "_stats.txt")

        valid_two_bubble = [r for r in results if np.isfinite(r.min_surface_distance)]
        c2c_values = np.array([r.centroid_distance for r in valid_two_bubble], dtype=float)
        s2s_values = np.array([r.min_surface_distance for r in valid_two_bubble], dtype=float)
        gap90_values = np.array([r.gap_r90 for r in valid_two_bubble], dtype=float)

        def line_stats(values: np.ndarray, label: str) -> str:
            if values.size == 0:
                return f"{label}: not available\n"
            return (
                f"{label}: min={np.min(values):.6f} A, mean={np.mean(values):.6f} A, "
                f"max={np.max(values):.6f} A\n"
            )

        with open(stats_file, "w") as handle:
            handle.write("Bubble surface distance analysis summary\n")
            handle.write("======================================\n")
            handle.write(f"Total trajectory frames: {total_frames}\n")
            handle.write(f"Selected frames: {selected_frames}\n")
            handle.write(f"Frames with two valid bubbles: {len(valid_two_bubble)}\n")
            handle.write(
                "Two-bubble coverage: {:.2f}%\n".format(
                    100.0 * len(valid_two_bubble) / selected_frames if selected_frames else 0.0
                )
            )
            handle.write(f"Surface fraction (radial): {self.surface_fraction:.3f}\n")
            handle.write(f"Minimum cluster size: {self.min_cluster_size}\n")
            handle.write(f"Time scale: {self.fs_per_step:.6f} fs/step\n")
            handle.write("\n")
            handle.write(line_stats(c2c_values, "Centroid distance"))
            handle.write(line_stats(s2s_values, "Surface minimum distance"))
            handle.write(line_stats(gap90_values, "Gap using R90 radii"))

        self.logger.info("Saved summary stats: %s", stats_file)

    def _plot_timeseries(self, output_file: str, results: List[FrameDistanceResult]) -> None:
        valid = [r for r in results if np.isfinite(r.min_surface_distance)]
        if not valid:
            self.logger.warning("No valid two-bubble frames for plotting")
            return

        plt = _get_pyplot()

        t_fs = np.array([r.time_fs for r in valid], dtype=float)
        c2c = np.array([r.centroid_distance for r in valid], dtype=float)
        s2s = np.array([r.min_surface_distance for r in valid], dtype=float)
        gap90 = np.array([r.gap_r90 for r in valid], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

        axes[0].plot(t_fs, c2c, color="#1f77b4", lw=1.8)
        axes[0].set_ylabel("Centroid Dist (A)")
        axes[0].grid(alpha=0.3)

        axes[1].plot(t_fs, s2s, color="#d62728", lw=1.8)
        axes[1].set_ylabel("Surface Min Dist (A)")
        axes[1].grid(alpha=0.3)

        axes[2].plot(t_fs, gap90, color="#2ca02c", lw=1.8)
        axes[2].axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)
        axes[2].set_ylabel("Gap R90 (A)")
        axes[2].set_xlabel("Simulation time (fs)")
        axes[2].grid(alpha=0.3)

        fig.suptitle("Two-bubble centroid and surface distance analysis")
        fig.tight_layout()

        png_file = output_file.replace(".txt", "_timeseries.png")
        fig.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info("Saved time-series plot: %s", png_file)


def get_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for bubble surface distance analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze two-bubble surface-to-surface distance in LAMMPS trajectories"
    )
    parser.add_argument("--traj_file", required=True, help="LAMMPS trajectory file path")
    parser.add_argument("--data", default=None, help="Optional LAMMPS topology/data file path")
    parser.add_argument("--output", default="bubble_surface_distance.txt", help="Output table path")
    parser.add_argument("--cutoff", type=float, default=5.5, help="N-N clustering cutoff in angstrom")
    parser.add_argument("--atom_style", default="id type x y z", help="LAMMPS atom_style")
    parser.add_argument("--step_interval", type=int, default=1, help="Analyze every Nth frame")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index (exclusive), -1 means to the end")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on analyzed frames")
    parser.add_argument("--surface_fraction", type=float, default=0.8, help="Surface cutoff as fraction of cluster max radius")
    parser.add_argument("--min_cluster_size", type=int, default=1, help="Minimum atoms required for each of the two bubble clusters")
    parser.add_argument("--nitrogen_type", type=int, default=None, help="Optional override for nitrogen atom type")
    parser.add_argument("--fs_per_step", type=float, default=1.0, help="Physical timestep size in fs")
    parser.add_argument("--colvar_file", default=None, help="Optional COLVAR file for generating COLVAR_surf_dis")
    parser.add_argument("--colvar_output", default=None, help="Output path for COLVAR_surf_dis")
    parser.add_argument(
        "--colvar_match_tolerance_steps",
        type=int,
        default=0,
        help="Allowed step mismatch when matching trajectory frame to COLVAR row",
    )
    parser.add_argument("--disable_plot", action="store_true", help="Disable time-series plot generation")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point."""
    args = get_args(argv)

    if not os.path.exists(args.traj_file):
        print(f"Error: trajectory file not found: {args.traj_file}")
        return 1

    if args.data and (not os.path.exists(args.data)):
        print("Error: optional data file {} does not exist".format(args.data))
        print("Tip: omit --data to run trajectory-only fallback mode")
        return 1

    analyzer = BubbleSurfaceDistanceAnalyzer(
        cutoff_distance=args.cutoff,
        surface_fraction=args.surface_fraction,
        nitrogen_type=args.nitrogen_type,
        min_cluster_size=args.min_cluster_size,
        fs_per_step=args.fs_per_step,
    )

    try:
        results = analyzer.analyze_trajectory(
            traj_file=args.traj_file,
            output_file=args.output,
            data_file=args.data,
            atom_style=args.atom_style,
            step_interval=args.step_interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_frames=args.max_frames,
            make_plot=(not args.disable_plot),
            colvar_file=args.colvar_file,
            colvar_output=args.colvar_output,
            colvar_match_tolerance_steps=args.colvar_match_tolerance_steps,
        )

        valid = sum(1 for row in results if np.isfinite(row.min_surface_distance))
        print("\n" + "=" * 60)
        print("Bubble surface distance analysis completed")
        print(f"Output file: {args.output}")
        print(f"Frames processed: {len(results)}")
        print(f"Frames with two valid bubbles: {valid}")
        if args.colvar_file:
            print("COLVAR_surf_dis generation: enabled")
        print("=" * 60)
    except Exception as exc:
        print(f"Bubble surface distance analysis failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
