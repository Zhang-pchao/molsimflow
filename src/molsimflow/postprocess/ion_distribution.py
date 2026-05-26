"""Ion distribution analysis from classified ion-species coordinate files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SPECIES_GROUP_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    "h3o": ("O", "H", "H", "H"),
    "oh_bulk": ("O", "H"),
    "oh_surface": ("O", "H"),
    "h_surface": ("H",),
    "na": ("Na",),
    "cl": ("Cl",),
}

SPECIES_CENTER_ELEMENT: Dict[str, str] = {
    "h3o": "O",
    "oh_bulk": "O",
    "oh_surface": "O",
    "h_surface": "H",
    "na": "Na",
    "cl": "Cl",
}

SURFACE_Z_COLUMNS = (
    "TiO2_Surface_O_Avg_Z",
    "tio2_surface_o_avg_z",
    "surface_o_avg_z",
    "surface_z",
)


@dataclass(frozen=True)
class AtomRecord:
    """One atom row from a multi-frame XYZ file."""

    element: str
    x: float
    y: float
    z: float
    atom_index: Optional[int] = None


@dataclass
class IonZDistribution:
    """Relative z-distribution statistics for one ion species."""

    species: str
    z_coords: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    density: np.ndarray
    counts: np.ndarray
    total_count: int
    filtered_count: int
    avg_per_frame: float
    frame_counts: List[int]


def parse_frame_index(header: str, fallback: int) -> int:
    """Extract `Frame=...` or `Frame ...` from an XYZ comment line."""

    match = re.search(r"Frame[=\s]+(-?\d+)", header)
    if not match:
        return fallback
    return int(match.group(1))


def read_multiframe_xyz(path: Path) -> Dict[int, List[AtomRecord]]:
    """Read a simple multi-frame XYZ file keyed by frame index."""

    path = Path(path)
    frames: Dict[int, List[AtomRecord]] = {}
    with path.open() as handle:
        fallback_frame = 0
        while True:
            count_line = handle.readline()
            if not count_line:
                break
            count_line = count_line.strip()
            if not count_line:
                continue
            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom count line in {path}: {count_line}") from exc

            header = handle.readline()
            if not header:
                raise ValueError(f"Missing XYZ header after atom count in {path}")
            frame_index = parse_frame_index(header, fallback=fallback_frame)
            fallback_frame += 1

            atoms: List[AtomRecord] = []
            for _ in range(atom_count):
                line = handle.readline()
                if not line:
                    raise ValueError(f"Unexpected end of XYZ file in {path}")
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid XYZ atom row in {path}: {line.rstrip()}")
                atom_index = int(parts[4]) if len(parts) >= 5 and parts[4].lstrip("-").isdigit() else None
                atoms.append(
                    AtomRecord(
                        element=parts[0],
                        x=float(parts[1]),
                        y=float(parts[2]),
                        z=float(parts[3]),
                        atom_index=atom_index,
                    )
                )
            frames[frame_index] = atoms
    return frames


def _validate_group_symbols(group: Sequence[AtomRecord], species: str) -> bool:
    expected = SPECIES_GROUP_SYMBOLS[species]
    observed = tuple(atom.element for atom in group)
    return observed == expected


def group_species_atoms(atoms: Sequence[AtomRecord], species: str) -> List[List[AtomRecord]]:
    """Group sequential atom rows into molecules/ions for a classified species."""

    if species not in SPECIES_GROUP_SYMBOLS:
        raise ValueError(f"Unsupported ion species: {species}")
    group_size = len(SPECIES_GROUP_SYMBOLS[species])
    groups: List[List[AtomRecord]] = []
    for start in range(0, len(atoms), group_size):
        group = list(atoms[start : start + group_size])
        if len(group) == group_size and _validate_group_symbols(group, species):
            groups.append(group)
    return groups


def load_species_xyz(path: Path, species: str) -> Dict[int, List[List[AtomRecord]]]:
    """Load a classified ion XYZ file into grouped molecules by frame."""

    raw_frames = read_multiframe_xyz(path)
    return {frame: group_species_atoms(atoms, species) for frame, atoms in raw_frames.items()}


def center_z(group: Sequence[AtomRecord], species: str) -> Optional[float]:
    """Return the z coordinate of the species-defining center atom."""

    center_element = SPECIES_CENTER_ELEMENT[species]
    for atom in group:
        if atom.element == center_element:
            return float(atom.z)
    return None


def read_surface_z_statistics(path: Path) -> Dict[int, float]:
    """Read frame-indexed surface-z values from `species_statistics.txt`."""

    path = Path(path)
    surface_z: Dict[int, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Statistics file has no header: {path}")

        z_column = next((name for name in SURFACE_Z_COLUMNS if name in reader.fieldnames), None)
        if z_column is None:
            raise ValueError(
                "Statistics file must contain one of these surface-z columns: "
                + ", ".join(SURFACE_Z_COLUMNS)
            )
        frame_column = "Frame" if "Frame" in reader.fieldnames else reader.fieldnames[0]

        for row in reader:
            if not row:
                continue
            try:
                value = float(row[z_column])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            surface_z[int(float(row[frame_column]))] = value
    return surface_z


def _histogram(values: Sequence[float], z_bins: int, z_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, bin_edges = np.histogram(values, bins=z_bins, range=z_range, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if len(values) == 0:
        density = np.zeros_like(bin_centers, dtype=float)
    else:
        density, _ = np.histogram(values, bins=z_bins, range=z_range, density=True)
        density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    return counts, bin_edges, bin_centers, density


def compute_ion_z_distributions(
    species_frames: Mapping[str, Mapping[int, Sequence[Sequence[AtomRecord]]]],
    surface_z_by_frame: Mapping[int, float],
    z_min_threshold: float = 15.0,
    z_bins: int = 100,
    z_range: Tuple[float, float] = (0.0, 30.0),
) -> Dict[str, IonZDistribution]:
    """Compute relative ion-z distributions for explicitly supplied species frames."""

    if z_bins < 1:
        raise ValueError("z_bins must be >= 1")
    if z_range[1] <= z_range[0]:
        raise ValueError("z_range upper bound must be greater than lower bound")

    distributions: Dict[str, IonZDistribution] = {}
    frame_indices = sorted(surface_z_by_frame)
    for species, frames in species_frames.items():
        z_coords: List[float] = []
        frame_counts: List[int] = []
        filtered_count = 0
        for frame_index in frame_indices:
            surface_z = float(surface_z_by_frame[frame_index])
            count = 0
            for group in frames.get(frame_index, []):
                z_value = center_z(group, species)
                if z_value is None:
                    continue
                if z_value >= z_min_threshold:
                    z_coords.append(z_value - surface_z)
                    count += 1
                else:
                    filtered_count += 1
            frame_counts.append(count)

        counts, bin_edges, bin_centers, density = _histogram(z_coords, z_bins=z_bins, z_range=z_range)
        distributions[species] = IonZDistribution(
            species=species,
            z_coords=np.asarray(z_coords, dtype=float),
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            density=density,
            counts=counts,
            total_count=len(z_coords),
            filtered_count=filtered_count,
            avg_per_frame=float(np.mean(frame_counts)) if frame_counts else 0.0,
            frame_counts=frame_counts,
        )
    return distributions


def write_distribution_summary(distributions: Mapping[str, IonZDistribution], output_path: Path) -> None:
    """Write one-row-per-species summary statistics."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("Species\tTotalCount\tFilteredCount\tAvgPerFrame\tMeanZ\tStdZ\tMinZ\tMaxZ\n")
        for species in sorted(distributions):
            dist = distributions[species]
            if len(dist.z_coords):
                mean_z = float(np.mean(dist.z_coords))
                std_z = float(np.std(dist.z_coords))
                min_z = float(np.min(dist.z_coords))
                max_z = float(np.max(dist.z_coords))
            else:
                mean_z = std_z = min_z = max_z = float("nan")
            handle.write(
                f"{species}\t{dist.total_count}\t{dist.filtered_count}\t"
                f"{dist.avg_per_frame:.6f}\t{mean_z:.6f}\t{std_z:.6f}\t{min_z:.6f}\t{max_z:.6f}\n"
            )


def write_density_table(distributions: Mapping[str, IonZDistribution], output_path: Path) -> None:
    """Write histogram density/count rows for all species."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("Species\tBinCenter\tDensity\tCount\n")
        for species in sorted(distributions):
            dist = distributions[species]
            for center, density, count in zip(dist.bin_centers, dist.density, dist.counts):
                handle.write(f"{species}\t{center:.6f}\t{float(density):.12g}\t{int(count)}\n")


def load_species_frames_from_files(species_files: Mapping[str, Optional[Path]]) -> Dict[str, Dict[int, List[List[AtomRecord]]]]:
    """Load all existing species XYZ files from an explicit path mapping."""

    loaded: Dict[str, Dict[int, List[List[AtomRecord]]]] = {}
    for species, path in species_files.items():
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        loaded[species] = load_species_xyz(path, species)
    return loaded


def analyze_ion_z_distribution(
    species_statistics: Path,
    species_files: Mapping[str, Optional[Path]],
    output_dir: Path,
    z_min_threshold: float = 15.0,
    z_bins: int = 100,
    z_range: Tuple[float, float] = (0.0, 30.0),
) -> Dict[str, IonZDistribution]:
    """Load classified species files, compute distributions, and write TSV outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_z_by_frame = read_surface_z_statistics(species_statistics)
    loaded_species = load_species_frames_from_files(species_files)
    distributions = compute_ion_z_distributions(
        loaded_species,
        surface_z_by_frame=surface_z_by_frame,
        z_min_threshold=z_min_threshold,
        z_bins=z_bins,
        z_range=z_range,
    )
    write_distribution_summary(distributions, output_dir / "ion_z_distribution_summary.tsv")
    write_density_table(distributions, output_dir / "ion_z_density.tsv")
    return distributions


def _species_file_args(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    return {
        "h3o": args.h3o_file,
        "oh_bulk": args.bulk_oh_file,
        "oh_surface": args.surface_oh_file,
        "h_surface": args.surface_h_file,
        "na": args.na_file,
        "cl": args.cl_file,
    }


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ion z-distributions from classified ion XYZ files")
    parser.add_argument("--species-statistics", type=Path, required=True, help="species_statistics.txt path")
    parser.add_argument("--output-dir", type=Path, default=Path("ion_z_distribution_results"))
    parser.add_argument("--h3o-file", type=Path, help="solution_bulk_h3o.xyz")
    parser.add_argument("--bulk-oh-file", type=Path, help="solution_bulk_oh.xyz")
    parser.add_argument("--surface-oh-file", type=Path, help="solution_surface_oh.xyz")
    parser.add_argument("--surface-h-file", type=Path, help="tio2_surface_h.xyz")
    parser.add_argument("--na-file", type=Path, help="na_ions.xyz")
    parser.add_argument("--cl-file", type=Path, help="cl_ions.xyz")
    parser.add_argument("--z-min", type=float, default=15.0, help="Absolute z cutoff before surface subtraction")
    parser.add_argument("--z-bins", type=int, default=100)
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.0, 30.0])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    if not args.species_statistics.exists():
        print(f"Error: species statistics file not found: {args.species_statistics}")
        return 1
    try:
        distributions = analyze_ion_z_distribution(
            species_statistics=args.species_statistics,
            species_files=_species_file_args(args),
            output_dir=args.output_dir,
            z_min_threshold=args.z_min,
            z_bins=args.z_bins,
            z_range=(float(args.z_range[0]), float(args.z_range[1])),
        )
    except Exception as exc:
        print(f"Ion z-distribution analysis failed: {exc}")
        return 1

    print(args.output_dir)
    print(f"species_processed={len(distributions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
