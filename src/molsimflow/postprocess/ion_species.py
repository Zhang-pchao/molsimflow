"""Ion species classification for MD trajectories.

This module contains the reusable core of the legacy ion-species workflow.  It
keeps trajectory paths, output directories, and atom-type mappings explicit so
the analysis can be reused outside one case directory layout.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.species_assignment import assign_hydrogen_to_nearest_oxygen


DEFAULT_TYPE_TO_ELEMENT: Dict[str, str] = {
    "1": "H",
    "2": "O",
    "3": "N",
    "4": "Na",
    "5": "Cl",
    "6": "Ti",
}

SPECIES_XYZ_FILES: Dict[str, str] = {
    "tio2_surface_h": "tio2_surface_h.xyz",
    "solution_surface_oh": "solution_surface_oh.xyz",
    "solution_surface_h2o": "solution_surface_h2o.xyz",
    "solution_bulk_oh": "solution_bulk_oh.xyz",
    "solution_bulk_h3o": "solution_bulk_h3o.xyz",
    "na_ions": "na_ions.xyz",
    "cl_ions": "cl_ions.xyz",
}

SPECIES_EXPECTED_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    "tio2_surface_h": ("H",),
    "solution_surface_oh": ("O", "H"),
    "solution_surface_h2o": ("O", "H", "H"),
    "solution_bulk_oh": ("O", "H"),
    "solution_bulk_h3o": ("O", "H", "H", "H"),
    "na_ions": ("Na",),
    "cl_ions": ("Cl",),
}


@dataclass(frozen=True)
class IonSpeciesConfig:
    """Parameters controlling ion-species assignment."""

    ti_o_cutoff: float = 3.5
    oh_cutoff: float = 1.35
    max_oh_distance: float = 1.8
    min_oh_distance: float = 0.5
    surface_ti_z_tolerance: float = 2.0
    type_to_element: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TYPE_TO_ELEMENT))


@dataclass
class MoleculeRecord:
    """A classified molecule or one-atom ion."""

    symbols: List[str]
    coords: np.ndarray
    indices: List[int]

    @property
    def atom_count(self) -> int:
        return len(self.symbols)


@dataclass
class IonSpeciesFrameResult:
    """Ion-species assignments for one frame."""

    frame_index: int
    species: MutableMapping[str, List[MoleculeRecord]]
    surface_o_z: List[float] = field(default_factory=list)

    def count(self, species_name: str) -> int:
        return len(self.species.get(species_name, []))

    @property
    def surface_o_avg_z(self) -> float:
        if not self.surface_o_z:
            return float("nan")
        return float(np.mean(self.surface_o_z))

    def counts(self) -> Dict[str, int]:
        values = {name: self.count(name) for name in SPECIES_XYZ_FILES}
        values["tio2_surface_o"] = len(self.surface_o_z)
        return values


class SimpleAtomsFrame:
    """Small adapter matching the ASE methods used by the classifier."""

    def __init__(self, symbols: Sequence[str], positions: Sequence[Sequence[float]]) -> None:
        self._symbols = list(symbols)
        self._positions = np.asarray(positions, dtype=float)

    def get_chemical_symbols(self) -> List[str]:
        return list(self._symbols)

    def get_positions(self) -> np.ndarray:
        return np.array(self._positions, copy=True)


def minimum_image_delta(delta: np.ndarray, box_dims: Sequence[float]) -> np.ndarray:
    """Apply an orthorhombic minimum-image transform to displacement vectors."""

    values = np.asarray(delta, dtype=float)
    box = np.asarray(box_dims, dtype=float)
    if box.shape[0] < 3:
        return values
    wrapped = np.array(values, copy=True)
    for dim in range(3):
        length = box[dim]
        if length > 0.0:
            wrapped[..., dim] = wrapped[..., dim] - length * np.round(wrapped[..., dim] / length)
    return wrapped


def periodic_distances(
    points: np.ndarray,
    reference: np.ndarray,
    box_dims: Sequence[float],
) -> np.ndarray:
    """Return minimum-image distances from `reference` to each point."""

    delta = np.asarray(points, dtype=float) - np.asarray(reference, dtype=float)
    return np.linalg.norm(minimum_image_delta(delta, box_dims), axis=-1)


def find_nearest_oxygen_hydrogens(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    box_dims: Sequence[float],
    oh_cutoff: float,
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Assign each hydrogen to the nearest oxygen within `oh_cutoff`.

    Returns a mapping from local oxygen index to hydrogen count and a mapping
    from local oxygen index to local hydrogen indices.
    """

    if len(h_positions) == 0 or len(o_positions) == 0:
        return {}, {}

    assignment = assign_hydrogen_to_nearest_oxygen(
        oxygen_coords=np.asarray(o_positions, dtype=float),
        hydrogen_coords=np.asarray(h_positions, dtype=float),
        bounds_or_lengths=box_dims,
        oh_cutoff=oh_cutoff,
    )
    o_h_bonds = assignment.hydrogen_indices_by_oxygen
    return {o_index: int(count) for o_index, count in enumerate(assignment.h_count_per_oxygen) if int(count) > 0}, o_h_bonds


def find_surface_oxygen_mask(
    o_positions: np.ndarray,
    ti_positions: np.ndarray,
    box_dims: Sequence[float],
    ti_o_cutoff: float,
    top_ti_z_tolerance: float,
) -> np.ndarray:
    """Identify oxygen atoms coordinated to the top Ti layer."""

    mask = np.zeros(len(o_positions), dtype=bool)
    if len(o_positions) == 0 or len(ti_positions) == 0:
        return mask

    max_ti_z = float(np.max(ti_positions[:, 2]))
    top_ti_positions = ti_positions[np.abs(ti_positions[:, 2] - max_ti_z) <= top_ti_z_tolerance]
    if len(top_ti_positions) == 0:
        return mask

    for index, o_pos in enumerate(o_positions):
        distances = periodic_distances(top_ti_positions, o_pos, box_dims)
        if len(distances) and float(np.min(distances)) <= ti_o_cutoff:
            mask[index] = True
    return mask


def _valid_oh_geometry(
    o_coord: np.ndarray,
    h_coords: Sequence[np.ndarray],
    config: IonSpeciesConfig,
) -> bool:
    if not h_coords:
        return False
    for h_coord in h_coords:
        distance = float(np.linalg.norm(np.asarray(o_coord, dtype=float) - np.asarray(h_coord, dtype=float)))
        if distance > config.max_oh_distance or distance < config.min_oh_distance:
            return False
    return True


def _make_molecule(
    o_coord: np.ndarray,
    h_coords: Sequence[np.ndarray],
    o_index: int,
    h_indices: Sequence[int],
    expected_h_count: int,
    config: IonSpeciesConfig,
) -> Optional[MoleculeRecord]:
    if len(h_coords) != expected_h_count or len(h_indices) != expected_h_count:
        return None
    if not _valid_oh_geometry(o_coord, h_coords, config):
        return None
    coords = np.vstack([np.asarray(o_coord, dtype=float), np.asarray(h_coords, dtype=float)])
    return MoleculeRecord(
        symbols=["O"] + ["H"] * expected_h_count,
        coords=coords,
        indices=[int(o_index)] + [int(index) for index in h_indices],
    )


def _empty_species() -> Dict[str, List[MoleculeRecord]]:
    return {name: [] for name in SPECIES_XYZ_FILES}


def classify_ion_species(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    box_dims: Sequence[float],
    frame_index: int = 0,
    config: Optional[IonSpeciesConfig] = None,
    atom_indices: Optional[Sequence[int]] = None,
) -> IonSpeciesFrameResult:
    """Classify ion species for one atomistic frame.

    The TiO2/solution split follows the legacy atom-order convention: TiO2
    oxygen atoms appear before the Ti atom block, while solution oxygen atoms
    appear after the Ti block.  The thresholds remain configurable so other
    systems can adapt the same workflow without editing code.
    """

    config = config or IonSpeciesConfig()
    coords = np.asarray(positions, dtype=float)
    labels = list(symbols)
    if len(labels) != len(coords):
        raise ValueError("symbols and positions must have the same length")

    source_indices = list(atom_indices) if atom_indices is not None else list(range(len(labels)))
    species = _empty_species()

    o_indices = [i for i, symbol in enumerate(labels) if symbol == "O"]
    h_indices = [i for i, symbol in enumerate(labels) if symbol == "H"]
    ti_indices = [i for i, symbol in enumerate(labels) if symbol == "Ti"]
    na_indices = [i for i, symbol in enumerate(labels) if symbol == "Na"]
    cl_indices = [i for i, symbol in enumerate(labels) if symbol == "Cl"]

    for atom_index in na_indices:
        species["na_ions"].append(
            MoleculeRecord(["Na"], np.asarray([coords[atom_index]], dtype=float), [source_indices[atom_index]])
        )
    for atom_index in cl_indices:
        species["cl_ions"].append(
            MoleculeRecord(["Cl"], np.asarray([coords[atom_index]], dtype=float), [source_indices[atom_index]])
        )

    result = IonSpeciesFrameResult(frame_index=frame_index, species=species)
    if not o_indices or not h_indices or not ti_indices:
        return result

    o_positions = coords[o_indices]
    h_positions = coords[h_indices]
    ti_positions = coords[ti_indices]
    surface_mask = find_surface_oxygen_mask(
        o_positions=o_positions,
        ti_positions=ti_positions,
        box_dims=box_dims,
        ti_o_cutoff=config.ti_o_cutoff,
        top_ti_z_tolerance=config.surface_ti_z_tolerance,
    )
    h_counts, o_h_bonds = find_nearest_oxygen_hydrogens(
        h_positions=h_positions,
        o_positions=o_positions,
        box_dims=box_dims,
        oh_cutoff=config.oh_cutoff,
    )

    min_ti_index = min(ti_indices)
    max_ti_index = max(ti_indices)

    for local_o_index, atom_index in enumerate(o_indices):
        h_count = h_counts.get(local_o_index, 0)
        bonded_local_h = o_h_bonds.get(local_o_index, [])
        bonded_h_indices = [h_indices[h_local] for h_local in bonded_local_h]
        bonded_h_source_indices = [source_indices[h_index] for h_index in bonded_h_indices]
        bonded_h_coords = [coords[h_index] for h_index in bonded_h_indices]
        is_top_surface = bool(surface_mask[local_o_index])
        o_coord = coords[atom_index]

        if atom_index < min_ti_index:
            if is_top_surface and h_count == 0:
                result.surface_o_z.append(float(o_coord[2]))
            elif is_top_surface and h_count == 1 and bonded_h_indices:
                h_index = bonded_h_indices[0]
                species["tio2_surface_h"].append(
                    MoleculeRecord(["H"], np.asarray([coords[h_index]], dtype=float), [source_indices[h_index]])
                )
            continue

        if atom_index <= max_ti_index:
            continue

        if is_top_surface and h_count == 1:
            molecule = _make_molecule(
                o_coord=o_coord,
                h_coords=bonded_h_coords,
                o_index=source_indices[atom_index],
                h_indices=bonded_h_source_indices,
                expected_h_count=1,
                config=config,
            )
            if molecule is not None:
                species["solution_surface_oh"].append(molecule)
        elif is_top_surface and h_count == 2:
            molecule = _make_molecule(
                o_coord=o_coord,
                h_coords=bonded_h_coords,
                o_index=source_indices[atom_index],
                h_indices=bonded_h_source_indices,
                expected_h_count=2,
                config=config,
            )
            if molecule is not None:
                species["solution_surface_h2o"].append(molecule)
        elif not is_top_surface and h_count == 1:
            molecule = _make_molecule(
                o_coord=o_coord,
                h_coords=bonded_h_coords,
                o_index=source_indices[atom_index],
                h_indices=bonded_h_source_indices,
                expected_h_count=1,
                config=config,
            )
            if molecule is not None:
                species["solution_bulk_oh"].append(molecule)
        elif not is_top_surface and h_count == 3:
            molecule = _make_molecule(
                o_coord=o_coord,
                h_coords=bonded_h_coords,
                o_index=source_indices[atom_index],
                h_indices=bonded_h_source_indices,
                expected_h_count=3,
                config=config,
            )
            if molecule is not None:
                species["solution_bulk_h3o"].append(molecule)

    return result


def classify_atoms_frame(
    atoms,
    box_dims: Sequence[float],
    frame_index: int = 0,
    config: Optional[IonSpeciesConfig] = None,
) -> IonSpeciesFrameResult:
    """Classify a frame object exposing ASE-like `get_*` methods."""

    return classify_ion_species(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        box_dims=box_dims,
        frame_index=frame_index,
        config=config,
    )


def load_lammps_universe(
    topology_file: Optional[Path],
    traj_file: Path,
    atom_style: Optional[str] = None,
):
    """Load a LAMMPS dump with MDAnalysis using explicit fallback styles."""

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError("MDAnalysis is required for LAMMPS trajectory analysis.") from exc

    style_candidates = [atom_style] if atom_style else ["id type x y z", "atomic", "full", None]
    attempts: List[Tuple[str, Tuple[Path, ...]]] = []
    if topology_file:
        attempts.append(("topology+trajectory", (Path(topology_file), Path(traj_file))))
    attempts.append(("trajectory-only", (Path(traj_file),)))

    last_error: Optional[Exception] = None
    for mode_label, universe_args in attempts:
        for style in style_candidates:
            kwargs = {"format": "LAMMPSDUMP"}
            if style:
                kwargs["atom_style"] = style
            try:
                logging.info("Trying MDAnalysis mode=%s atom_style=%s", mode_label, style or "auto")
                return mda.Universe(*(str(arg) for arg in universe_args), **kwargs)
            except Exception as exc:  # pragma: no cover - depends on MDAnalysis reader details
                last_error = exc
                logging.warning("MDAnalysis reader failed mode=%s atom_style=%s: %s", mode_label, style, exc)

    raise RuntimeError("Unable to load trajectory with provided fallbacks: {}".format(last_error))


def read_lammps_frame(
    universe,
    frame_index: int,
    type_to_element: Optional[Mapping[str, str]] = None,
) -> Tuple[SimpleAtomsFrame, np.ndarray, np.ndarray]:
    """Read one MDAnalysis LAMMPS frame into a small atoms adapter."""

    type_to_element = type_to_element or DEFAULT_TYPE_TO_ELEMENT
    universe.trajectory[frame_index]
    positions = np.asarray(universe.atoms.positions, dtype=float)
    symbols = [type_to_element.get(str(atom_type), "X") for atom_type in universe.atoms.types]
    box_info = np.asarray(universe.dimensions, dtype=float)
    box_dims = box_info[:3]
    return SimpleAtomsFrame(symbols=symbols, positions=positions), box_dims, box_info


def _lattice_header(box_info: Optional[Sequence[float]]) -> str:
    if box_info is None or len(box_info) < 3:
        return ""
    a, b, c = [float(value) for value in box_info[:3]]
    return f'Lattice="{a} 0.0 0.0 0.0 {b} 0.0 0.0 0.0 {c}" '


def write_xyz_frame(
    path: Path,
    records: Iterable[MoleculeRecord],
    frame_index: int,
    box_info: Optional[Sequence[float]] = None,
    append: bool = False,
) -> None:
    """Write one multi-frame XYZ block for classified records."""

    atoms: List[Tuple[str, np.ndarray, int]] = []
    for record in records:
        for symbol, coord, atom_index in zip(record.symbols, record.coords, record.indices):
            atoms.append((symbol, np.asarray(coord, dtype=float), int(atom_index)))

    mode = "a" if append else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(
            f"Frame={frame_index} pbc=\"T T T\" "
            f"{_lattice_header(box_info)}Properties=species:S:1:pos:R:3:atom_index:I:1\n"
        )
        for symbol, coord, atom_index in atoms:
            handle.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {atom_index}\n")


def write_species_xyz(
    result: IonSpeciesFrameResult,
    output_dir: Path,
    box_info: Optional[Sequence[float]] = None,
    append: bool = False,
) -> None:
    """Write classified species coordinate files for one frame."""

    output_dir = Path(output_dir)
    for species_name, filename in SPECIES_XYZ_FILES.items():
        records = result.species.get(species_name, [])
        if records:
            write_xyz_frame(output_dir / filename, records, result.frame_index, box_info=box_info, append=append)


def write_species_statistics(results: Sequence[IonSpeciesFrameResult], output_path: Path) -> None:
    """Write a compact per-frame ion-species statistics table."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Frame",
        "Solution_Surface_OH",
        "Solution_Surface_H2O",
        "Solution_Bulk_OH",
        "Solution_Bulk_H3O",
        "TiO2_Surface_H",
        "Na_Ions",
        "Cl_Ions",
        "TiO2_Surface_O_Count",
        "TiO2_Surface_O_Avg_Z",
    ]
    with output_path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for result in results:
            row = [
                result.frame_index,
                result.count("solution_surface_oh"),
                result.count("solution_surface_h2o"),
                result.count("solution_bulk_oh"),
                result.count("solution_bulk_h3o"),
                result.count("tio2_surface_h"),
                result.count("na_ions"),
                result.count("cl_ions"),
                len(result.surface_o_z),
                result.surface_o_avg_z,
            ]
            handle.write("\t".join(str(value) for value in row) + "\n")


def _clear_species_outputs(output_dir: Path) -> None:
    for filename in list(SPECIES_XYZ_FILES.values()) + ["species_statistics.txt", "run_summary.txt"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def analyze_lammps_ion_species(
    traj_file: Path,
    output_dir: Path,
    topology_file: Optional[Path] = None,
    atom_style: Optional[str] = None,
    start_frame: int = 0,
    end_frame: int = -1,
    step_interval: int = 100,
    config: Optional[IonSpeciesConfig] = None,
) -> List[IonSpeciesFrameResult]:
    """Run ion-species analysis over a LAMMPS trajectory."""

    if step_interval < 1:
        raise ValueError("step_interval must be >= 1")

    config = config or IonSpeciesConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_species_outputs(output_dir)

    universe = load_lammps_universe(topology_file=topology_file, traj_file=traj_file, atom_style=atom_style)
    total_frames = len(universe.trajectory)
    actual_end = total_frames if end_frame == -1 else min(end_frame, total_frames)
    frame_indices = list(range(start_frame, actual_end, step_interval))

    with (output_dir / "run_summary.txt").open("w") as handle:
        handle.write("status=running\n")
        handle.write(f"trajectory={traj_file}\n")
        handle.write(f"topology={topology_file if topology_file else 'None'}\n")
        handle.write(f"total_frames={total_frames}\n")
        handle.write(f"selected_frames={len(frame_indices)}\n")
        handle.write(f"step_interval={step_interval}\n")

    results: List[IonSpeciesFrameResult] = []
    for write_index, frame_index in enumerate(frame_indices):
        atoms, box_dims, box_info = read_lammps_frame(
            universe,
            frame_index=frame_index,
            type_to_element=config.type_to_element,
        )
        result = classify_atoms_frame(atoms, box_dims=box_dims, frame_index=frame_index, config=config)
        results.append(result)
        write_species_xyz(result, output_dir=output_dir, box_info=box_info, append=(write_index > 0))

    write_species_statistics(results, output_dir / "species_statistics.txt")
    with (output_dir / "run_summary.txt").open("a") as handle:
        handle.write("status=complete\n")
    return results


def _parse_type_map(values: Optional[Sequence[str]]) -> Dict[str, str]:
    type_map = dict(DEFAULT_TYPE_TO_ELEMENT)
    if not values:
        return type_map
    for value in values:
        if "=" not in value:
            raise ValueError("--type-map entries must use TYPE=ELEMENT syntax")
        atom_type, element = value.split("=", 1)
        type_map[atom_type.strip()] = element.strip()
    return type_map


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify ion species from a LAMMPS trajectory")
    parser.add_argument("--traj", type=Path, required=True, help="LAMMPS trajectory file")
    parser.add_argument("--data", type=Path, help="Optional LAMMPS topology/data file")
    parser.add_argument("--output-dir", type=Path, default=Path("ion_analysis_results"))
    parser.add_argument("--atom-style", default=None, help="Optional MDAnalysis LAMMPS atom_style")
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--ti-o-cutoff", type=float, default=3.5)
    parser.add_argument("--oh-cutoff", type=float, default=1.35)
    parser.add_argument("--max-oh-distance", type=float, default=1.8)
    parser.add_argument("--surface-ti-z-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--type-map",
        nargs="*",
        help="Override atom type mapping entries, for example 1=H 2=O 6=Ti",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    if not args.traj.exists():
        print(f"Error: trajectory file not found: {args.traj}")
        return 1
    if args.data and not args.data.exists():
        print(f"Error: data file not found: {args.data}")
        return 1

    try:
        config = IonSpeciesConfig(
            ti_o_cutoff=args.ti_o_cutoff,
            oh_cutoff=args.oh_cutoff,
            max_oh_distance=args.max_oh_distance,
            surface_ti_z_tolerance=args.surface_ti_z_tolerance,
            type_to_element=_parse_type_map(args.type_map),
        )
        results = analyze_lammps_ion_species(
            traj_file=args.traj,
            topology_file=args.data,
            output_dir=args.output_dir,
            atom_style=args.atom_style,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step_interval=args.step_interval,
            config=config,
        )
    except Exception as exc:
        print(f"Ion species analysis failed: {exc}")
        return 1

    print(args.output_dir)
    print(f"frames_processed={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
