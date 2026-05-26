#!/usr/bin/env python3
"""Compatibility note retained from migrated legacy script"""

import os
import sys
import numpy as np
import argparse
import time
import re
from collections import defaultdict
import logging


def _get_pyplot():
    """Import matplotlib lazily so non-plot utilities stay lightweight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

class UnionFind:
    """Compatibility note retained from migrated legacy script"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Legacy comment retained for compatibility.
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Legacy comment retained for compatibility.
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

class BubbleCentroidCalculator:
    """Compatibility note retained from migrated legacy script"""
    
    def __init__(self, cutoff_distance=5.5):
        self.cutoff_distance = cutoff_distance
        
        # Legacy comment retained for compatibility.
        self.type_to_element = {
            '1': 'H',   
            '2': 'O',     
            '3': 'N',  
            '4': 'Na',   
            '5': 'Cl',
            '6': 'Ti',
        }
        
        # Legacy comment retained for compatibility.
        self.nitrogen_type = None
        for atom_type, element in self.type_to_element.items():
            if element == 'N':
                self.nitrogen_type = int(atom_type)
                break
        
        # Legacy comment retained for compatibility.
        self.ion_box_data = {}
        
        # Legacy comment retained for compatibility.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_progress(self, message, flush=True):
        """Compatibility note retained from migrated legacy script"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        if flush:
            sys.stdout.flush()
    
    def periodic_distance(self, coord1, coord2, box_dims):
        """Compatibility note retained from migrated legacy script"""
        diff = coord1 - coord2
        for i in range(3):
            box_length = box_dims[i]
            diff[i] = diff[i] - box_length * round(diff[i] / box_length)
        return np.linalg.norm(diff)
    
    def cluster_nitrogen_atoms(self, n_coords, box_dims):
        """Compatibility note retained from migrated legacy script"""
        if len(n_coords) == 0:
            return [], []
        
        n_points = len(n_coords)
        uf = UnionFind(n_points)
        
        self.logger.debug("Legacy compatibility informational message")
        
        # Legacy comment retained for compatibility.
        for i in range(n_points):
            for j in range(i+1, n_points):
                if self.periodic_distance(n_coords[i], n_coords[j], box_dims) <= self.cutoff_distance:
                    uf.union(i, j)
        
        # Legacy comment retained for compatibility.
        clusters_dict = defaultdict(list)
        for atom in range(n_points):
            clusters_dict[uf.find(atom)].append(atom)
        
        # Legacy comment retained for compatibility.
        clusters = sorted(clusters_dict.values(), key=len, reverse=True)
        
        self.logger.debug("Legacy compatibility informational message")
        if clusters:
            self.logger.debug("Legacy compatibility informational message")
        
        return clusters
    
    def calculate_centroid_pbc(self, coords, box_dims):
        """Compatibility note retained from migrated legacy script"""
        centroid = np.zeros(3)
        
        for dim in range(3):
            box_length = box_dims[dim]
            
            # Legacy comment retained for compatibility.
            angles = 2 * np.pi * coords[:, dim] / box_length
            
            # Legacy comment retained for compatibility.
            cos_mean = np.mean(np.cos(angles))
            sin_mean = np.mean(np.sin(angles))
            mean_angle = np.arctan2(sin_mean, cos_mean)
            
            # Legacy comment retained for compatibility.
            centroid[dim] = (mean_angle * box_length) / (2 * np.pi)
            if centroid[dim] < 0:
                centroid[dim] += box_length
        
        return centroid
    
    def read_lammps_with_mda(self, traj_file, data_file=None, atom_style=None):
        """Compatibility note retained from migrated legacy script"""
        try:
            try:
                import MDAnalysis as mda
            except ImportError as exc:
                raise ImportError(
                    "MDAnalysis is required for trajectory reading; install molsimflow[analysis]."
                ) from exc

            self.logger.info("Loading LAMMPS trajectory with MDAnalysis...")
            self.logger.info(f"Trajectory file: {traj_file}")
            if data_file:
                self.logger.info(f"Optional topology/data file: {data_file}")
            else:
                self.logger.info("No topology/data file provided; using trajectory-only fallback")

            style_candidates = [atom_style] if atom_style else ["id type x y z", "atomic", "full", None]

            universe_attempts = []
            if data_file:
                universe_attempts.append(("topology+trajectory", (data_file, traj_file)))
            universe_attempts.append(("trajectory-only", (traj_file,)))

            last_error = None
            for mode_label, universe_args in universe_attempts:
                for style in style_candidates:
                    kwargs = {"format": "LAMMPSDUMP"}
                    if style:
                        kwargs["atom_style"] = style
                    try:
                        self.logger.info(
                            "Trying mode={} atom_style={}".format(mode_label, style if style else "auto")
                        )
                        u = mda.Universe(*universe_args, **kwargs)
                        self.logger.info(f"Loaded {len(u.atoms)} atoms across {len(u.trajectory)} frames")
                        return u
                    except Exception as exc:
                        last_error = exc
                        self.logger.warning(
                            "Reader attempt failed (mode={}, atom_style={}): {}".format(
                                mode_label, style if style else "auto", exc
                            )
                        )

            raise RuntimeError("Failed to load trajectory with all reader fallbacks: {}".format(last_error))

        except Exception as e:
            self.logger.error("Failed to read LAMMPS files with MDAnalysis: {}".format(e))
            raise
    
    def process_trajectory(self, traj_file, data_file=None, atom_style=None, output_file="bubble_centroids.txt", 
                         step_interval=1, start_frame=0, end_frame=-1, ion_files=None, ions_analysis_output=None):
        """Compatibility note retained from migrated legacy script"""
        
        # Legacy comment retained for compatibility.
        if self.nitrogen_type is None:
            raise ValueError("Nitrogen atom type is not configured")

        self.logger.info("Starting bubble centroid analysis")

        # Legacy comment retained for compatibility.
        u = self.read_lammps_with_mda(traj_file=traj_file, data_file=data_file, atom_style=atom_style)

        # Legacy comment retained for compatibility.
        total_frames = len(u.trajectory)
        actual_end_frame = total_frames if end_frame == -1 else min(end_frame, total_frames)
        frames_to_analyze = list(range(start_frame, actual_end_frame, step_interval))

        self.logger.info("Trajectory frames available: %d", total_frames)
        self.logger.info("Requested frame range: [%d, %d)", start_frame, actual_end_frame)
        self.logger.info("Step interval: %d", step_interval)
        self.logger.info("Frames selected for analysis: %d", len(frames_to_analyze))

        if not frames_to_analyze:
            raise ValueError("No frames selected for centroid analysis")

        output_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        os.makedirs(output_dir, exist_ok=True)
        progress_file = os.path.splitext(output_file)[0] + "_progress.log"

        # Create early artifacts so running jobs do not appear as empty-output failures.
        with open(output_file, 'w') as f:
            f.write("# FrameIndex Time(ps) B1_X(Å) B1_Y(Å) B1_Z(Å) B1_Size B2_X(Å) B2_Y(Å) B2_Z(Å) B2_Size\n")

        with open(progress_file, 'w') as f:
            f.write("status=running\n")
            f.write(f"traj_file={traj_file}\n")
            f.write(f"total_frames={total_frames}\n")
            f.write(f"selected_frames={len(frames_to_analyze)}\n")
            f.write(f"step_interval={step_interval}\n")
        
        # Legacy comment retained for compatibility.
        ion_frames_data = {}
        if ion_files:
            self.logger.info("Loading ion coordinate files for distance analysis")
            ion_configs = {
                'H3O': {'file': ion_files.get('h3o'), 'atoms_per_molecule': 4},
                'bulk_OH': {'file': ion_files.get('bulk_oh'), 'atoms_per_molecule': 2},
                'surface_OH': {'file': ion_files.get('surface_oh'), 'atoms_per_molecule': 2},
                'surface_H': {'file': ion_files.get('surface_h'), 'atoms_per_molecule': 1},
                'Na': {'file': ion_files.get('na'), 'atoms_per_molecule': 1},
                'Cl': {'file': ion_files.get('cl'), 'atoms_per_molecule': 1}
            }
            
            for ion_name, config in ion_configs.items():
                if config['file']:
                    ion_frames_data[ion_name] = self.read_xyz_file_with_frame_filter(
                        config['file'], set(frames_to_analyze), config['atoms_per_molecule'], ion_name)
            
            # Legacy comment retained for compatibility.
            if hasattr(self, 'ion_box_data') and self.ion_box_data:
                self.logger.debug("Legacy compatibility informational message")
                sample_frame = next(iter(self.ion_box_data))
                sample_box = self.ion_box_data[sample_frame]
                self.logger.info("Loaded ion box dimensions from sample frame %s: %s", sample_frame, sample_box)
            else:
                self.logger.warning("No ion box dimensions found in ion XYZ files; trajectory box will be used")
        
        # Legacy comment retained for compatibility.
        centroids_data = []
        times = []
        bubble_sizes = []
        frame_numbers = []
        secondary_centroids_data = []
        secondary_bubble_sizes = []
        
        # Legacy comment retained for compatibility.
        ions_distance_data = {ion_name: [] for ion_name in ion_frames_data.keys()}
        
        self.logger.info("Processing %d selected frames", len(frames_to_analyze))
        
        # Legacy comment retained for compatibility.
        for i, frame_idx in enumerate(frames_to_analyze):
            # Legacy comment retained for compatibility.
            u.trajectory[frame_idx]
            ts = u.trajectory.ts
            
            if i < 3 or (i + 1) % 20 == 0:
                self.logger.info(
                    "Progress: processed %d/%d selected frames (frame index %d)",
                    i + 1,
                    len(frames_to_analyze),
                    frame_idx,
                )
            
            # Legacy comment retained for compatibility.
            box_dims = u.dimensions[:3]  # Legacy comment retained for compatibility.
            
            # Legacy comment retained for compatibility.
            nitrogen_atoms = u.select_atoms(f"type {self.nitrogen_type}")
            
            if len(nitrogen_atoms) == 0:
                self.logger.warning("Frame %d: no nitrogen atoms found for configured type %s", frame_idx, self.nitrogen_type)
                continue
            
            # Legacy comment retained for compatibility.
            n_coords = nitrogen_atoms.positions
            
            # Legacy comment retained for compatibility.
            clusters = self.cluster_nitrogen_atoms(n_coords, box_dims)
            
            if not clusters:
                self.logger.warning("Frame %d: no bubble clusters detected", frame_idx)
                continue
            
            bubble_clusters = clusters[:2] if len(clusters) >= 2 else clusters[:1]
            bubble_records = []

            for bubble_idx, cluster_indices in enumerate(bubble_clusters, start=1):
                cluster_coords = n_coords[cluster_indices]
                centroid = self.calculate_centroid_pbc(cluster_coords, box_dims)
                surface_coords = self.find_bubble_surface_n2_atoms(cluster_coords, box_dims)
                bubble_records.append(
                    {
                        "bubble_index": bubble_idx,
                        "cluster_indices": cluster_indices,
                        "cluster_coords": cluster_coords,
                        "centroid": centroid,
                        "size": len(cluster_indices),
                        "surface_coords": surface_coords,
                    }
                )

            primary = bubble_records[0]
            centroid = primary["centroid"]

            times.append(ts.time)
            centroids_data.append(primary["centroid"])
            bubble_sizes.append(primary["size"])
            frame_numbers.append(frame_idx)

            if len(bubble_records) > 1:
                secondary = bubble_records[1]
                secondary_centroids_data.append(secondary["centroid"])
                secondary_bubble_sizes.append(secondary["size"])
            else:
                secondary_centroids_data.append(np.array([np.nan, np.nan, np.nan]))
                secondary_bubble_sizes.append(0)

            self.logger.info(
                "Frame {}: detected {} bubble cluster(s); bubble_1 size={}, bubble_2 size={}".format(
                    frame_idx,
                    len(bubble_records),
                    primary["size"],
                    secondary_bubble_sizes[-1],
                )
            )

            if ion_frames_data:
                primary_surface_coords = primary["surface_coords"]
                for ion_name, ion_frames in ion_frames_data.items():
                    if frame_idx in ion_frames:
                        molecules = ion_frames[frame_idx]
                        if molecules:
                            distances_data = self.calculate_ion_bubble_distances(
                                molecules, centroid, primary_surface_coords, box_dims, ion_name, frame_idx)

                            ions_distance_data[ion_name].extend(distances_data)
                            self.logger.info(
                                "Frame {}: analyzed {} {} molecules/ions".format(
                                    frame_idx, len(molecules), ion_name
                                )
                            )
        
        if not centroids_data:
            self.logger.warning("No valid bubble centroids detected; writing header-only summary outputs")

        with open(progress_file, 'a') as f:
            f.write(f"processed_frames={len(frame_numbers)}\n")

        # Legacy comment retained for compatibility.
        self.save_results(
            times,
            centroids_data,
            bubble_sizes,
            frame_numbers,
            output_file,
            step_interval,
            start_frame,
            actual_end_frame,
            secondary_centroids=secondary_centroids_data,
            secondary_sizes=secondary_bubble_sizes,
        )
        
        # Legacy comment retained for compatibility.
        if any(ions_distance_data.values()) and ions_analysis_output:
            self.save_raw_ion_distances(ions_distance_data, ions_analysis_output)
        
        # Legacy comment retained for compatibility.
        if any(ions_distance_data.values()) and ions_analysis_output:
            total_ions = sum(len(distances) for distances in ions_distance_data.values())
            self.logger.info("Collected %d ion-to-bubble distance samples", total_ions)
            self.plot_all_ions_distance_distributions(ions_distance_data, ions_analysis_output)
        elif ion_files and not any(ions_distance_data.values()):
            self.logger.warning("Ion files were provided, but no matching ion-distance samples were found")

        with open(progress_file, 'a') as f:
            f.write("status=completed\n")

        return times, centroids_data, bubble_sizes
    
    def save_results(self, times, centroids_data, bubble_sizes, frame_numbers, output_file, 
                   step_interval, start_frame, end_frame, secondary_centroids=None, secondary_sizes=None):
        """Compatibility note retained from migrated legacy script"""
        
        centroids_file = output_file
        if secondary_centroids is None:
            secondary_centroids = [np.array([np.nan, np.nan, np.nan]) for _ in frame_numbers]
        if secondary_sizes is None:
            secondary_sizes = [0 for _ in frame_numbers]

        with open(centroids_file, 'w') as f:
            f.write(
                "# FrameIndex Time(ps) B1_X(Å) B1_Y(Å) B1_Z(Å) B1_Size B2_X(Å) B2_Y(Å) B2_Z(Å) B2_Size\n"
            )
            for frame_idx, time, centroid, size, centroid2, size2 in zip(
                frame_numbers,
                times,
                centroids_data,
                bubble_sizes,
                secondary_centroids,
                secondary_sizes,
            ):
                f.write(
                    f"{frame_idx} {time:.1f} "
                    f"{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} {size} "
                    f"{centroid2[0]:.6f} {centroid2[1]:.6f} {centroid2[2]:.6f} {size2}\n"
                )

        self.logger.info("Centroid coordinates saved to: {}".format(centroids_file))
        
        # Legacy comment retained for compatibility.
        stats_file = output_file.replace('.txt', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Bubble centroid analysis summary\n")
            f.write("================================\n")
            f.write(f"Requested frame range: {start_frame} to {end_frame}\n")
            f.write(f"Step interval: {step_interval}\n")
            f.write(f"Frames with centroid results: {len(frame_numbers)}\n")

            if frame_numbers:
                f.write(f"First processed frame index: {frame_numbers[0]}\n")
                f.write(f"Last processed frame index: {frame_numbers[-1]}\n")
                f.write(f"Time range (ps): {times[0]:.3f} to {times[-1]:.3f}\n")
                f.write(f"Average bubble 1 size: {np.mean(bubble_sizes):.2f}\n")

                valid_secondary = [size for size in secondary_sizes if size > 0]
                if valid_secondary:
                    f.write(f"Average bubble 2 size: {np.mean(valid_secondary):.2f}\n")
                    f.write(f"Frames with bubble 2 detected: {len(valid_secondary)}\n")
                else:
                    f.write("Bubble 2 not detected in analyzed frames\n")
            else:
                f.write("No valid bubble clusters were detected in selected frames\n")

        self.logger.info("Centroid summary statistics saved to: %s", stats_file)
    
    def parse_lattice_from_extended_xyz(self, header_line):
        """Compatibility note retained from migrated legacy script"""
        try:
            # Legacy comment retained for compatibility.
            lattice_match = re.search(r'lattice="([^"]+)"', header_line)
            if lattice_match:
                lattice_str = lattice_match.group(1)
                lattice_values = list(map(float, lattice_str.split()))
                
                # Legacy comment retained for compatibility.
                if len(lattice_values) == 9:
                    box_dims = np.array([lattice_values[0], lattice_values[4], lattice_values[8]])
                    self.logger.debug(f"Compatibility note retained from migrated legacy script")
                    return box_dims
                else:
                    self.logger.warning("Legacy compatibility warning: frame data failed a legacy validation check")
                    return None
            else:
                self.logger.warning("Legacy compatibility warning: frame data failed a legacy validation check")
                return None
                
        except Exception as e:
            self.logger.error("Legacy compatibility error encountered during analysis")
            return None
    
    def read_xyz_file_with_frame_filter(self, xyz_file, frames_to_analyze, molecules_per_group, molecule_name):
        """Compatibility note retained from migrated legacy script"""
        frames_data = {}  # {frame: [molecules]}
        frames_box_data = {}  # {frame: box_dims}
        
        if not os.path.exists(xyz_file):
            self.logger.warning("Legacy compatibility warning: frame data failed a legacy validation check")
            return frames_data
        
        self.logger.debug("Legacy compatibility informational message")
        
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # Legacy comment retained for compatibility.
                if lines[i].strip().isdigit():
                    n_atoms = int(lines[i].strip())
                    i += 1
                    
                    # Legacy comment retained for compatibility.
                    frame_line = lines[i].strip()
                    frame_match = re.search(r'Frame[=\s](\d+)', frame_line)
                    if frame_match:
                        frame = int(frame_match.group(1))
                        
                        # Legacy comment retained for compatibility.
                        box_dims = self.parse_lattice_from_extended_xyz(frame_line)
                        
                        i += 1
                        
                        # Legacy comment retained for compatibility.
                        if frame in frames_to_analyze:
                            # Legacy comment retained for compatibility.
                            atoms = []
                            for j in range(n_atoms):
                                if i + j < len(lines):
                                    parts = lines[i + j].strip().split()
                                    if len(parts) >= 4:
                                        element = parts[0]
                                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                        atoms.append((element, x, y, z))
                            
                            # Legacy comment retained for compatibility.
                            molecules = self.parse_molecules(atoms, molecules_per_group, molecule_name)
                            frames_data[frame] = molecules
                            
                            # Legacy comment retained for compatibility.
                            if box_dims is not None:
                                frames_box_data[frame] = box_dims
                            
                        i += n_atoms
                    else:
                        i += 1
                else:
                    i += 1
            
            self.logger.debug("Legacy compatibility informational message")
            total_molecules = sum(len(molecules) for molecules in frames_data.values())
            self.logger.debug("Legacy compatibility informational message")
            
            # Legacy comment retained for compatibility.
            # Legacy comment retained for compatibility.
            if not hasattr(self, 'ion_box_data') or not self.ion_box_data:
                self.ion_box_data = frames_box_data
            else:
                # Legacy comment retained for compatibility.
                for frame, box_dims in frames_box_data.items():
                    if frame not in self.ion_box_data:
                        self.ion_box_data[frame] = box_dims
            
        except Exception as e:
            self.logger.error("Legacy compatibility error encountered during analysis")
        
        return frames_data
    
    def parse_molecules(self, atoms, molecules_per_group, molecule_name):
        """Compatibility note retained from migrated legacy script"""
        molecules = []
        
        if molecule_name == "H3O":
            # Legacy comment retained for compatibility.
            for atom_idx in range(0, len(atoms), 4):
                if atom_idx + 3 < len(atoms):
                    o_atom = atoms[atom_idx]
                    h1_atom = atoms[atom_idx + 1]
                    h2_atom = atoms[atom_idx + 2] 
                    h3_atom = atoms[atom_idx + 3]
                    
                    if (o_atom[0] == 'O' and h1_atom[0] == 'H' and 
                        h2_atom[0] == 'H' and h3_atom[0] == 'H'):
                        o_coord = np.array([o_atom[1], o_atom[2], o_atom[3]])
                        h_coords = [
                            np.array([h1_atom[1], h1_atom[2], h1_atom[3]]),
                            np.array([h2_atom[1], h2_atom[2], h2_atom[3]]),
                            np.array([h3_atom[1], h3_atom[2], h3_atom[3]])
                        ]
                        molecules.append(('O', o_coord, h_coords))
        
        elif molecule_name in ["bulk_OH", "surface_OH"]:
            # Legacy comment retained for compatibility.
            for atom_idx in range(0, len(atoms), 2):
                if atom_idx + 1 < len(atoms):
                    o_atom = atoms[atom_idx]
                    h_atom = atoms[atom_idx + 1]
                    
                    if o_atom[0] == 'O' and h_atom[0] == 'H':
                        o_coord = np.array([o_atom[1], o_atom[2], o_atom[3]])
                        h_coord = np.array([h_atom[1], h_atom[2], h_atom[3]])
                        molecules.append(('O', o_coord, [h_coord]))
        
        elif molecule_name == "surface_H":
            # Legacy comment retained for compatibility.
            for atom in atoms:
                if atom[0] == 'H':
                    h_coord = np.array([atom[1], atom[2], atom[3]])
                    molecules.append(('H', h_coord, []))
        
        elif molecule_name in ["Na", "Cl"]:
            # Legacy comment retained for compatibility.
            expected_element = 'Na' if molecule_name == 'Na' else 'Cl'
            for atom in atoms:
                if atom[0] == expected_element:
                    coord = np.array([atom[1], atom[2], atom[3]])
                    molecules.append((expected_element, coord, []))
        
        return molecules
    
    def find_bubble_surface_n2_atoms(self, largest_cluster_coords, box_dims):
        """Compatibility note retained from migrated legacy script"""
        if len(largest_cluster_coords) < 2:
            return largest_cluster_coords
        
        # Legacy comment retained for compatibility.
        centroid = self.calculate_centroid_pbc(largest_cluster_coords, box_dims)
        
        # Legacy comment retained for compatibility.
        distances_to_center = []
        for coord in largest_cluster_coords:
            dist = self.periodic_distance(coord, centroid, box_dims)
            distances_to_center.append(dist)
        
        # Legacy comment retained for compatibility.
        max_dist = max(distances_to_center)
        surface_threshold = max_dist * 0.8
        
        # Legacy comment retained for compatibility.
        surface_indices = [i for i, dist in enumerate(distances_to_center) 
                          if dist >= surface_threshold]
        surface_coords = largest_cluster_coords[surface_indices]
        
        self.logger.debug(f"Compatibility note retained from migrated legacy script")
        
        return surface_coords
    
    def calculate_ion_bubble_distances(self, molecules, centroid, surface_coords, box_dims, molecule_name, frame_idx=None):
        """Compatibility note retained from migrated legacy script"""
        distances_data = []
        
        # Legacy comment retained for compatibility.
        ion_box_dims = box_dims  # Legacy comment retained for compatibility.
        if hasattr(self, 'ion_box_data') and frame_idx is not None and frame_idx in self.ion_box_data:
            ion_box_dims = self.ion_box_data[frame_idx]
            self.logger.debug(f"Compatibility note retained from migrated legacy script")
        else:
            self.logger.debug(f"Compatibility note retained from migrated legacy script")
        
        for molecule in molecules:
            element, center_coord, other_coords = molecule
            
            # Legacy comment retained for compatibility.
            d_centroid = self.periodic_distance(center_coord, centroid, ion_box_dims)
            
            # Legacy comment retained for compatibility.
            min_surface_dist = float('inf')
            for surface_coord in surface_coords:
                dist = self.periodic_distance(center_coord, surface_coord, ion_box_dims)
                if dist < min_surface_dist:
                    min_surface_dist = dist
            
            d_interface = min_surface_dist
            distances_data.append((d_centroid, d_interface))
        
        return distances_data
    
    def save_raw_ion_distances(self, ions_distance_data, output_dir):
        """Compatibility note retained from migrated legacy script"""
        # Legacy comment retained for compatibility.
        os.makedirs(output_dir, exist_ok=True)
        
        # Legacy comment retained for compatibility.
        total_count = 0
        for ion_name, distances in ions_distance_data.items():
            if distances:
                ion_file = os.path.join(output_dir, f"raw_{ion_name}_distances.txt")
                with open(ion_file, 'w') as f:
                    f.write(f"Compatibility note retained from migrated legacy script")
                    f.write("Compatibility note retained from migrated legacy script")
                    f.write("d_centroid\td_interface\n")
                    
                    for d_cent, d_int in distances:
                        f.write(f"{d_cent:.6f}\t{d_int:.6f}\n")
                
                total_count += len(distances)
                self.logger.debug("Legacy compatibility informational message")
        
        self.logger.debug("Legacy compatibility informational message")
    
    def setup_nature_style(self):
        """Compatibility note retained from migrated legacy script"""
        plt = _get_pyplot()
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'text.usetex': False,
            'mathtext.default': 'regular'
        })
        return plt
    
    def plot_all_ions_distance_distributions(self, ions_distance_data, output_dir):
        """Compatibility note retained from migrated legacy script"""
        if not ions_distance_data:
            self.logger.warning("Legacy compatibility warning: frame data failed a legacy validation check")
            return
        
        # Legacy comment retained for compatibility.
        plt = self.setup_nature_style()
        
        # Legacy comment retained for compatibility.
        os.makedirs(output_dir, exist_ok=True)
        
        # Legacy comment retained for compatibility.
        ion_styles = {
            'H3O': {'color': '#1f77b4', 'marker': 'o', 'label': r'$\mathrm{H_3O^+}$'},
            'bulk_OH': {'color': '#ff7f0e', 'marker': 's', 'label': r'$\mathrm{OH^-(bulk)}$'},
            'surface_OH': {'color': '#2ca02c', 'marker': '^', 'label': r'$\mathrm{OH^-(surf)}$'},
            'surface_H': {'color': '#d62728', 'marker': 'v', 'label': r'$\mathrm{H^+(surf)}$'},
            'Na': {'color': '#9467bd', 'marker': 'D', 'label': r'$\mathrm{Na^+}$'},
            'Cl': {'color': '#8c564b', 'marker': 'p', 'label': r'$\mathrm{Cl^-}$'}
        }
        
        bins = 50
        
        # Legacy comment retained for compatibility.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Legacy comment retained for compatibility.
        for ion_name, distances in ions_distance_data.items():
            if not distances:
                continue
                
            all_d_centroid = [d[0] for d in distances]
            all_d_interface = [d[1] for d in distances]
            
            style = ion_styles.get(ion_name, {'color': 'black', 'marker': 'o', 'label': ion_name})
            
            # Legacy comment retained for compatibility.
            hist_centroid, bin_edges_centroid = np.histogram(all_d_centroid, bins=bins, density=True)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            
            ax1.plot(bin_centers_centroid, hist_centroid, 
                    marker=style['marker'], color=style['color'], 
                    linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Legacy comment retained for compatibility.
            hist_interface, bin_edges_interface = np.histogram(all_d_interface, bins=bins, density=True)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            
            ax2.plot(bin_centers_interface, hist_interface,
                    marker=style['marker'], color=style['color'],
                    linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
        
        # Legacy comment retained for compatibility.
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.set_title('Ion Distance to Bubble Centroid', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=11)
        
        # Legacy comment retained for compatibility.
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.set_title('Ion Distance to Bubble Surface', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=11)
        
        plt.tight_layout()
        
        # Legacy comment retained for compatibility.
        plot_file = os.path.join(output_dir, "all_ions_bubble_distance_distributions.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Legacy comment retained for compatibility.
        data_file = os.path.join(output_dir, "all_ions_distance_distribution_data.txt")
        with open(data_file, 'w') as f:
            f.write("# All ions distance distribution data\n")
            f.write("# Format: ion_type d_centroid_center d_centroid_density d_interface_center d_interface_density\n")
            
            for ion_name, distances in ions_distance_data.items():
                if not distances:
                    continue
                    
                all_d_centroid = [d[0] for d in distances]
                all_d_interface = [d[1] for d in distances]
                
                hist_centroid, bin_edges_centroid = np.histogram(all_d_centroid, bins=bins, density=True)
                bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
                
                hist_interface, bin_edges_interface = np.histogram(all_d_interface, bins=bins, density=True)
                bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
                
                f.write(f"\n# {ion_name} data\n")
                max_len = max(len(bin_centers_centroid), len(bin_centers_interface))
                for i in range(max_len):
                    centroid_center = bin_centers_centroid[i] if i < len(bin_centers_centroid) else ""
                    centroid_density = hist_centroid[i] if i < len(hist_centroid) else ""
                    interface_center = bin_centers_interface[i] if i < len(bin_centers_interface) else ""
                    interface_density = hist_interface[i] if i < len(hist_interface) else ""
                    f.write(f"{ion_name}\t{centroid_center}\t{centroid_density}\t{interface_center}\t{interface_density}\n")
        

        
        self.logger.debug("Legacy compatibility informational message")
        self.logger.debug("Legacy compatibility informational message")
        
        # Legacy comment retained for compatibility.
        for ion_name, distances in ions_distance_data.items():
            if distances:
                all_d_centroid = [d[0] for d in distances]
                all_d_interface = [d[1] for d in distances]
                self.logger.debug("Legacy compatibility informational message")
                self.logger.debug("Legacy compatibility informational message")
                self.logger.debug("Legacy compatibility informational message")

def get_args(argv=None):
    """Parse command-line arguments for centroid analysis."""
    parser = argparse.ArgumentParser(description="Compute bubble centroids from LAMMPS trajectories")
    
    parser.add_argument('--traj_file', required=True, help='LAMMPS trajectory file path')
    
    # Legacy comment retained for compatibility.
    parser.add_argument('--data', default=None, help='Optional LAMMPS topology/data file path')
    parser.add_argument('--output', default='bubble_centroids.txt', help='Output centroid table file')
    parser.add_argument('--cutoff', type=float, default=5.5, help='N-N clustering cutoff in angstrom')
    parser.add_argument('--atom_style', default='id type x y z', help='LAMMPS atom_style for trajectory parsing')
    
    # Legacy comment retained for compatibility.
    parser.add_argument('--step_interval', type=int, default=1, help='Analyze every Nth frame')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=-1, help='End frame index (exclusive), -1 means to the end')
    
    parser.add_argument('--h3o_file', default=None, help='H3O coordinates file')
    parser.add_argument('--bulk_oh_file', default=None, help='Bulk OH coordinates file')
    parser.add_argument('--surface_oh_file', default=None, help='Surface OH coordinates file')
    parser.add_argument('--surface_h_file', default=None, help='Surface H coordinates file')
    parser.add_argument('--na_file', default=None, help='Na+ coordinates file')
    parser.add_argument('--cl_file', default=None, help='Cl- coordinates file')
    parser.add_argument('--ions_output', default='ions_analysis', help='Directory for ion-distance analysis outputs')
    parser.add_argument('--disable_ions', action='store_true', help='Disable ion-distance analysis even if ion files exist')
    
    return parser.parse_args(argv)

def main(argv=None):
    """CLI entry point for centroid analysis."""
    args = get_args(argv)
    
    # Legacy comment retained for compatibility.
    if not os.path.exists(args.traj_file):
        print(f"Error: trajectory file not found: {args.traj_file}")
        return 1
    
    if args.data and (not os.path.exists(args.data)):
        print("Error: optional data file {} does not exist".format(args.data))
        print("Tip: omit --data to run trajectory-only fallback mode")
        return 1
    
    # Legacy comment retained for compatibility.
    ion_files = {}
    if not args.disable_ions:
        ion_file_configs = {
            'h3o': args.h3o_file,
            'bulk_oh': args.bulk_oh_file,
            'surface_oh': args.surface_oh_file,
            'surface_h': args.surface_h_file,
            'na': args.na_file,
            'cl': args.cl_file
        }
        
        for ion_name, file_path in ion_file_configs.items():
            if file_path and os.path.exists(file_path):
                ion_files[ion_name] = file_path
                print(f"Loaded ion file for {ion_name}: {file_path}")
            elif file_path:
                # Legacy comment retained for compatibility.
                if ion_name in ['na', 'cl']:
                    print(f"Optional ion file missing for {ion_name}: {file_path}")
                else:
                    print(f"Ion file missing for {ion_name}: {file_path}")
        
        if not ion_files:
            print("No ion files loaded; ion-distance analysis will be skipped")
    else:
        print("Ion-distance analysis disabled by --disable_ions")
    
    # Legacy comment retained for compatibility.
    calculator = BubbleCentroidCalculator(
        cutoff_distance=args.cutoff
    )
    
    # Legacy comment retained for compatibility.
    data_file = args.data
    
    try:
        # Legacy comment retained for compatibility.
        times, centroids, sizes = calculator.process_trajectory(
            data_file=data_file,
            traj_file=args.traj_file,
            atom_style=args.atom_style,
            output_file=args.output,
            step_interval=args.step_interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            ion_files=ion_files if ion_files else None,
            ions_analysis_output=args.ions_output if ion_files else None
        )
        
        print("\n" + "="*50)
        print("Bubble centroid analysis completed")
        print(f"Output file: {args.output}")
        if times:
            print(f"Frames processed: {len(times)}")
            print(f"Time range: {times[0]:.3f} ps to {times[-1]:.3f} ps")
        else:
            print("Frames processed: 0")

        if ion_files:
            print(f"Ion files loaded: {len(ion_files)}")
            print("Ion-distance distributions were enabled")
            print(f"Ion analysis output directory: {args.ions_output}")

        print("="*50)
        
    except Exception as e:
        print(f"Centroid analysis failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main()) 
