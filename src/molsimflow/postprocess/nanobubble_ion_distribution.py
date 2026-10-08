"""Sample reactive ions relative to a silica surface and an N2 nanobubble."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors, periodic_center
from molsimflow.postprocess.nanobubble_attachment import (
    largest_cluster,
    molecule_centers,
    parse_range,
)
from molsimflow.postprocess.surface_proton_transfer import (
    assign_hydrogens_to_oxygen_or_carbon,
    hydrogen_ids_by_owner,
    identify_initial_donor_sites,
)
from molsimflow.postprocess.surface_reference import load_surface_reference

SPECIES = (
    "Na_plus",
    "Cl_minus",
    "H3O_plus_candidate",
    "OH_minus_candidate",
)
SPECIES_CHARGE = {
    "Na_plus": 1,
    "Cl_minus": -1,
    "H3O_plus_candidate": 1,
    "OH_minus_candidate": -1,
}
SAMPLE_FIELDS = (
    "stage",
    "step",
    "time_ns",
    "species",
    "formal_charge_e",
    "atom_id",
    "hydrogen_ids",
    "surface_origin_hydrogen_ids",
    "surface_origin_donor_ids",
    "x_A",
    "y_A",
    "z_A",
    "z_from_top_si_A",
    "z_from_terminal_plane_A",
    "rho_xy_from_bubble_center_A",
    "r_from_bubble_center_A",
    "r_minus_bubble_R90_A",
    "nearest_main_n2_center_A",
)


@dataclass(frozen=True)
class Stage:
    """Inclusive time window used as an explicit physical-stage label."""

    name: str
    start_ns: float
    end_ns: float

    def contains(self, time_ns: float) -> bool:
        return self.start_ns <= time_ns <= self.end_ns


@dataclass(frozen=True)
class IonFrame:
    """Coordinates selected from one custom LAMMPS dump frame."""

    source: Path
    source_frame: int
    step: int
    bounds: np.ndarray
    surface: np.ndarray
    nitrogen: np.ndarray
    oxygen_ids: np.ndarray
    oxygen: np.ndarray
    solution_oxygen_ids: np.ndarray
    solution_oxygen: np.ndarray
    hydrogen_ids: np.ndarray
    hydrogen: np.ndarray
    sodium_ids: np.ndarray
    sodium: np.ndarray
    chloride_ids: np.ndarray
    chloride: np.ndarray


def parse_stage(text: str) -> Stage:
    """Parse ``NAME:START_NS:END_NS`` into an inclusive stage window."""

    parts = text.split(":")
    if len(parts) != 3 or not parts[0].strip():
        raise argparse.ArgumentTypeError("Stage must be NAME:START_NS:END_NS")
    try:
        start, end = map(float, parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Stage bounds must be numbers") from exc
    if not np.isfinite([start, end]).all() or end < start or start < 0:
        raise argparse.ArgumentTypeError("Stage must satisfy 0 <= START_NS <= END_NS")
    return Stage(parts[0].strip(), start, end)


def stage_names(stages: Sequence[Stage], time_ns: float) -> tuple[str, ...]:
    """Return every explicit stage containing ``time_ns``."""

    return tuple(stage.name for stage in stages if stage.contains(time_ns))


def top_surface_si_ids(
    elements: np.ndarray,
    coordinates: np.ndarray,
    surface_range: tuple[int, int],
    window_A: float,
) -> np.ndarray:
    """Select reference Si atoms within ``window_A`` of the highest Si atom."""

    if window_A <= 0:
        raise ValueError("top-Si selection window must be positive")
    start, end = surface_range
    slab_elements = np.asarray(elements[start - 1 : end])
    slab_coordinates = np.asarray(coordinates[start - 1 : end], dtype=float)
    si = np.flatnonzero(slab_elements == "Si")
    if not len(si):
        raise ValueError("Surface range contains no Si atoms")
    selected = si[slab_coordinates[si, 2] >= float(np.max(slab_coordinates[si, 2])) - window_A]
    if not len(selected):
        raise ValueError("Top-Si selection is empty")
    return selected + start


def _rows_to_arrays(rows: list[tuple[int, float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)
    array = np.asarray(sorted(rows), dtype=float)
    return array[:, 0].astype(int), array[:, 1:]


@contextmanager
def open_dump_text(path: Path) -> Iterator[TextIO]:
    """Open a text dump directly or stream a Zstandard-compressed dump."""

    resolved = Path(path).resolve()
    if resolved.suffix != ".zst":
        with resolved.open(encoding="utf-8") as handle:
            yield handle
        return
    try:
        process = subprocess.Popen(
            ["zstd", "-dc", "--", str(resolved)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("zstd is required to stream a .zst trajectory") from exc
    if process.stdout is None:
        process.kill()
        process.wait()
        raise RuntimeError("zstd did not provide a stdout stream")
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError(f"zstd failed while streaming {resolved}")


def iter_ion_frames(
    path: Path,
    surface_range: tuple[int, int],
    nitrogen_range: tuple[int, int],
    solution_range: tuple[int, int],
    *,
    hydrogen_type: int,
    oxygen_type: int,
    sodium_type: int,
    chloride_type: int,
    stop_after_step: int | None = None,
) -> Iterator[IonFrame]:
    """Stream only coordinates needed for nanobubble ion analysis.

    ``stop_after_step`` is an explicit trajectory-manifest boundary.  It keeps
    readers from interpreting a preallocated NUL tail after a valid restart
    segment as a malformed LAMMPS frame.
    """

    with open_dump_text(path) as handle:
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
            missing = {"id", "type", "x", "y", "z"}.difference(index)
            if missing:
                raise ValueError(f"Missing dump columns at step {step}: {sorted(missing)}")
            surface: list[tuple[int, float, float, float]] = []
            nitrogen: list[tuple[int, float, float, float]] = []
            oxygen: list[tuple[int, float, float, float]] = []
            solution_oxygen: list[tuple[int, float, float, float]] = []
            hydrogen: list[tuple[int, float, float, float]] = []
            sodium: list[tuple[int, float, float, float]] = []
            chloride: list[tuple[int, float, float, float]] = []
            for _ in range(atom_count):
                values = handle.readline().split()
                if not values:
                    raise ValueError(f"Unexpected end of atom rows at step {step}")
                atom_id = int(values[index["id"]])
                atom_type = int(values[index["type"]])
                xyz = tuple(float(values[index[key]]) for key in ("x", "y", "z"))
                row = (atom_id, *xyz)
                if surface_range[0] <= atom_id <= surface_range[1]:
                    surface.append(row)
                if nitrogen_range[0] <= atom_id <= nitrogen_range[1]:
                    nitrogen.append(row)
                if atom_type == oxygen_type:
                    oxygen.append(row)
                    if solution_range[0] <= atom_id <= solution_range[1]:
                        solution_oxygen.append(row)
                elif atom_type == hydrogen_type:
                    hydrogen.append(row)
                elif atom_type == sodium_type:
                    sodium.append(row)
                elif atom_type == chloride_type:
                    chloride.append(row)
            surface_ids, surface_xyz = _rows_to_arrays(surface)
            nitrogen_ids, nitrogen_xyz = _rows_to_arrays(nitrogen)
            oxygen_ids, oxygen_xyz = _rows_to_arrays(oxygen)
            solution_ids, solution_xyz = _rows_to_arrays(solution_oxygen)
            hydrogen_ids, hydrogen_xyz = _rows_to_arrays(hydrogen)
            sodium_ids, sodium_xyz = _rows_to_arrays(sodium)
            chloride_ids, chloride_xyz = _rows_to_arrays(chloride)
            if len(surface_ids) != surface_range[1] - surface_range[0] + 1:
                raise ValueError(f"Incomplete surface selection at step {step}")
            if len(nitrogen_ids) != nitrogen_range[1] - nitrogen_range[0] + 1:
                raise ValueError(f"Incomplete nitrogen selection at step {step}")
            yield IonFrame(
                Path(path),
                frame_index,
                step,
                bounds,
                surface_xyz,
                nitrogen_xyz,
                oxygen_ids,
                oxygen_xyz,
                solution_ids,
                solution_xyz,
                hydrogen_ids,
                hydrogen_xyz,
                sodium_ids,
                sodium_xyz,
                chloride_ids,
                chloride_xyz,
            )
            frame_index += 1
            if stop_after_step is not None and step >= stop_after_step:
                return


def classify_mobile_species(
    frame: IonFrame,
    carbon_ids: np.ndarray,
    carbon_indices: np.ndarray,
    *,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[int, tuple[int, ...]], dict[str, int]]:
    """Classify fixed ions and reactive H3O/OH candidates in one frame."""

    assignment = assign_hydrogens_to_oxygen_or_carbon(
        frame.oxygen_ids,
        frame.oxygen,
        carbon_ids,
        frame.surface[carbon_indices],
        frame.hydrogen_ids,
        frame.hydrogen,
        frame.bounds,
        oh_cutoff_A=oh_cutoff_A,
        ch_cutoff_A=ch_cutoff_A,
    )
    grouped = hydrogen_ids_by_owner(assignment)
    h_counts = np.asarray(
        [len(grouped.get(int(atom_id), ())) for atom_id in frame.solution_oxygen_ids],
        dtype=int,
    )
    h3o = h_counts == 3
    hydroxide = h_counts == 1
    species = {
        "Na_plus": (frame.sodium_ids, frame.sodium),
        "Cl_minus": (frame.chloride_ids, frame.chloride),
        "H3O_plus_candidate": (
            frame.solution_oxygen_ids[h3o],
            frame.solution_oxygen[h3o],
        ),
        "OH_minus_candidate": (
            frame.solution_oxygen_ids[hydroxide],
            frame.solution_oxygen[hydroxide],
        ),
    }
    diagnostics = {
        "solution_h2o_candidate_count": int(np.count_nonzero(h_counts == 2)),
        "solution_zero_h_oxygen_count": int(np.count_nonzero(h_counts == 0)),
        "solution_four_plus_h_oxygen_count": int(np.count_nonzero(h_counts >= 4)),
        "solution_excess_proton_candidate_count": int(np.sum(h_counts - 2)),
        "unassigned_hydrogen_count": int(np.count_nonzero(assignment.owner_ids < 0)),
    }
    return species, grouped, diagnostics


def build_ion_samples(
    species: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    hydrogen_ids_by_oxygen: dict[int, tuple[int, ...]],
    surface_hydrogen_donors: dict[int, int],
    frame: IonFrame,
    time_ns: float,
    stages: Sequence[str],
    top_si_z_A: float,
    terminal_plane_z_A: float,
    bubble_center: np.ndarray,
    bubble_R90_A: float,
    main_n2_centers: np.ndarray,
) -> list[dict]:
    """Build per-ion surface and bubble coordinates for one frame."""

    from scipy.spatial import cKDTree

    lengths = box_lengths(frame.bounds)
    origin = frame.bounds[:, 0]
    tree = cKDTree((main_n2_centers - origin) % lengths, boxsize=lengths)
    rows: list[dict] = []
    for name in SPECIES:
        atom_ids, coordinates = species[name]
        if not len(coordinates):
            continue
        center_vectors = minimum_image_vectors(coordinates - bubble_center, lengths)
        radial = np.linalg.norm(center_vectors, axis=1)
        rho = np.linalg.norm(center_vectors[:, :2], axis=1)
        shifted = (coordinates - origin) % lengths
        nearest = np.asarray(tree.query(shifted, k=1)[0], dtype=float)
        z_top = (coordinates[:, 2] - top_si_z_A) % lengths[2]
        z_terminal = (coordinates[:, 2] - terminal_plane_z_A) % lengths[2]
        for stage in stages:
            for atom_id, point, z_si, z_terminal_i, rho_i, radial_i, nearest_i in zip(
                atom_ids,
                coordinates,
                z_top,
                z_terminal,
                rho,
                radial,
                nearest,
            ):
                hydrogen_ids = hydrogen_ids_by_oxygen.get(int(atom_id), ())
                surface_hydrogen_ids = tuple(
                    hydrogen_id
                    for hydrogen_id in hydrogen_ids
                    if hydrogen_id in surface_hydrogen_donors
                )
                donor_ids = tuple(
                    sorted({surface_hydrogen_donors[hydrogen_id] for hydrogen_id in surface_hydrogen_ids})
                )
                rows.append(
                    {
                        "stage": stage,
                        "step": frame.step,
                        "time_ns": time_ns,
                        "species": name,
                        "formal_charge_e": SPECIES_CHARGE[name],
                        "atom_id": int(atom_id),
                        "hydrogen_ids": ";".join(map(str, hydrogen_ids)),
                        "surface_origin_hydrogen_ids": ";".join(
                            map(str, surface_hydrogen_ids)
                        ),
                        "surface_origin_donor_ids": ";".join(map(str, donor_ids)),
                        "x_A": float(point[0]),
                        "y_A": float(point[1]),
                        "z_A": float(point[2]),
                        "z_from_top_si_A": float(z_si),
                        "z_from_terminal_plane_A": float(z_terminal_i),
                        "rho_xy_from_bubble_center_A": float(rho_i),
                        "r_from_bubble_center_A": float(radial_i),
                        "r_minus_bubble_R90_A": float(radial_i - bubble_R90_A),
                        "nearest_main_n2_center_A": float(nearest_i),
                    }
                )
    return rows


def analyze_frame(
    frame: IonFrame,
    *,
    stages: Sequence[str],
    time_ns: float,
    top_si_ids: np.ndarray,
    carbon_ids: np.ndarray,
    carbon_indices: np.ndarray,
    initial_sioh_ids: np.ndarray,
    surface_hydrogen_donors: dict[int, int],
    surface_start: int,
    surface_reference,
    cluster_cutoff_A: float,
    oh_cutoff_A: float,
    ch_cutoff_A: float,
) -> tuple[dict, list[dict]]:
    """Analyze one staged frame and return its summary and per-ion samples."""

    lengths = box_lengths(frame.bounds)
    top_indices = top_si_ids - surface_start
    top_si_z = float(periodic_center(frame.surface[top_indices], frame.bounds)[2])
    terminal_plane_z = float(surface_reference.plane_z(frame.surface, frame.bounds))
    n2_centers = molecule_centers(frame.nitrogen, frame.bounds)
    members = largest_cluster(n2_centers, frame.bounds, cluster_cutoff_A)
    main_centers = n2_centers[members]
    bubble_center = periodic_center(main_centers, frame.bounds)
    bubble_vectors = minimum_image_vectors(main_centers - bubble_center, lengths)
    bubble_r90 = float(np.quantile(np.linalg.norm(bubble_vectors, axis=1), 0.90))
    species, grouped, diagnostics = classify_mobile_species(
        frame,
        carbon_ids,
        carbon_indices,
        oh_cutoff_A=oh_cutoff_A,
        ch_cutoff_A=ch_cutoff_A,
    )
    sioh_h_counts = np.asarray(
        [len(grouped.get(int(atom_id), ())) for atom_id in initial_sioh_ids], dtype=int
    )
    species_counts = {f"{name}_count": len(species[name][0]) for name in SPECIES}
    mobile_charge = int(
        sum(SPECIES_CHARGE[name] * len(species[name][0]) for name in SPECIES)
    )
    deprotonated = int(np.count_nonzero(sioh_h_counts == 0))
    hyperprotonated = int(np.count_nonzero(sioh_h_counts >= 2))
    terminal_offset = float((terminal_plane_z - top_si_z) % lengths[2])
    frame_row = {
        "source_file": str(frame.source),
        "source_frame": frame.source_frame,
        "step": frame.step,
        "time_ns": time_ns,
        "stages": ";".join(stages),
        "box_x_A": float(lengths[0]),
        "box_y_A": float(lengths[1]),
        "box_z_A": float(lengths[2]),
        "top_si_plane_z_A": top_si_z,
        "terminal_plane_z_A": terminal_plane_z,
        "terminal_minus_top_si_A": terminal_offset,
        "bubble_center_x_A": float(bubble_center[0]),
        "bubble_center_y_A": float(bubble_center[1]),
        "bubble_center_z_A": float(bubble_center[2]),
        "bubble_center_from_top_si_A": float(
            (bubble_center[2] - top_si_z) % lengths[2]
        ),
        "bubble_R90_A": bubble_r90,
        "largest_cluster_n2_count": len(members),
        "disconnected_n2_count": len(n2_centers) - len(members),
        **species_counts,
        **diagnostics,
        "initial_top_sioh_count": len(initial_sioh_ids),
        "top_sioh_nominal_candidate_count": int(np.count_nonzero(sioh_h_counts == 1)),
        "top_sioh_deprotonated_candidate_count": deprotonated,
        "top_sioh_hyperprotonated_candidate_count": hyperprotonated,
        "mobile_formal_charge_candidate_e": mobile_charge,
        "top_sioh_formal_charge_candidate_e": hyperprotonated - deprotonated,
        "mobile_plus_top_sioh_formal_charge_candidate_e": (
            mobile_charge + hyperprotonated - deprotonated
        ),
    }
    samples = build_ion_samples(
        species,
        hydrogen_ids_by_oxygen=grouped,
        surface_hydrogen_donors=surface_hydrogen_donors,
        frame=frame,
        time_ns=time_ns,
        stages=stages,
        top_si_z_A=top_si_z,
        terminal_plane_z_A=terminal_plane_z,
        bubble_center=bubble_center,
        bubble_R90_A=bubble_r90,
        main_n2_centers=main_centers,
    )
    return frame_row, samples


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_samples(path: Path, rows: Sequence[dict]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _selected_steps(
    records: dict[int, tuple[dict, list[dict]]],
    raw_steps: set[int],
    drop_first_frame: bool,
) -> list[int]:
    """Drop only the earliest raw frame, not the first frame in a stage window."""

    steps = sorted(records)
    if drop_first_frame and raw_steps and min(raw_steps) in records:
        steps.remove(min(raw_steps))
    return steps


def run_analysis(args: argparse.Namespace) -> dict:
    elements, reference_coordinates, reference_lengths = read_extxyz_positions(
        args.reference_structure
    )
    top_si = top_surface_si_ids(
        elements,
        reference_coordinates,
        args.surface_range,
        args.top_si_window_A,
    )
    surface_elements = elements[args.surface_range[0] - 1 : args.surface_range[1]]
    carbon_indices = np.flatnonzero(surface_elements == "C")
    carbon_ids = carbon_indices + args.surface_range[0]
    sites = identify_initial_donor_sites(
        elements,
        reference_coordinates,
        reference_lengths,
        slab_range=args.surface_range,
        surface_z_A=args.terminal_surface_z_A,
        surface_depth_A=args.surface_depth_A,
        oh_cutoff_A=args.oh_cutoff_A,
        ch_cutoff_A=args.ch_cutoff_A,
    )
    initial_sioh_ids = np.asarray(
        [int(site["atom_id"]) for site in sites if site["site_type"] == "SiOH"],
        dtype=int,
    )
    surface_hydrogen_donors = {
        int(hydrogen_id): int(site["atom_id"])
        for site in sites
        if site["site_type"] == "SiOH"
        for hydrogen_id in site["_initial_hydrogen_ids"]
    }
    surface_reference = load_surface_reference(
        args.reference_structure,
        args.surface_range,
        args.terminal_surface_z_A,
    )
    records: dict[int, tuple[dict, list[dict]]] = {}
    raw_frames = 0
    raw_steps: set[int] = set()
    trajectory_max_steps = (
        args.trajectory_max_step
        if args.trajectory_max_step
        else [None] * len(args.trajectory)
    )
    for trajectory, max_step in zip(args.trajectory, trajectory_max_steps, strict=True):
        for frame in iter_ion_frames(
            trajectory,
            args.surface_range,
            args.nitrogen_range,
            args.solution_range,
            hydrogen_type=args.hydrogen_type,
            oxygen_type=args.oxygen_type,
            sodium_type=args.sodium_type,
            chloride_type=args.chloride_type,
            stop_after_step=max_step,
        ):
            raw_frames += 1
            raw_steps.add(frame.step)
            if not np.allclose(box_lengths(frame.bounds), reference_lengths):
                raise ValueError("Reference structure and trajectory cell lengths differ")
            time_ns = frame.step * args.timestep_fs / 1.0e6
            labels = stage_names(args.stage, time_ns)
            if not labels:
                continue
            records[frame.step] = analyze_frame(
                frame,
                stages=labels,
                time_ns=time_ns,
                top_si_ids=top_si,
                carbon_ids=carbon_ids,
                carbon_indices=carbon_indices,
                initial_sioh_ids=initial_sioh_ids,
                surface_hydrogen_donors=surface_hydrogen_donors,
                surface_start=args.surface_range[0],
                surface_reference=surface_reference,
                cluster_cutoff_A=args.cluster_cutoff_A,
                oh_cutoff_A=args.oh_cutoff_A,
                ch_cutoff_A=args.ch_cutoff_A,
            )
    steps = _selected_steps(records, raw_steps, args.drop_first_frame)
    if not steps:
        raise ValueError("No staged trajectory frames remain")
    frames = [records[step][0] for step in steps]
    samples = [row for step in steps for row in records[step][1]]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "frame_summary.csv", frames, frames[0].keys())
    _write_samples(output / "ion_samples.csv.gz", samples)
    frames_per_stage = Counter(
        stage for row in frames for stage in str(row["stages"]).split(";") if stage
    )
    species_ranges = {}
    for name in SPECIES:
        values = [int(row[f"{name}_count"]) for row in frames]
        species_ranges[name] = {"minimum": min(values), "maximum": max(values)}
    summary = {
        "status": "PASS",
        "raw_frames": raw_frames,
        "unique_raw_frames": len(raw_steps),
        "unique_staged_frames_before_drop": len(records),
        "analyzed_frames": len(frames),
        "first_step": steps[0],
        "last_step": steps[-1],
        "frames_per_stage": dict(sorted(frames_per_stage.items())),
        "top_si_atom_count": len(top_si),
        "initial_top_sioh_count": len(initial_sioh_ids),
        "species_count_ranges": species_ranges,
        "sample_rows": len(samples),
        "surface_origin_h3o_sample_rows": sum(
            row["species"] == "H3O_plus_candidate"
            and bool(row["surface_origin_hydrogen_ids"])
            for row in samples
        ),
        "reactive_species_are_geometric_candidates": True,
        "formal_charges_are_species_labels_not_atomic_partial_charges": True,
        "gas_interface_is_main_n2_cluster_geometry_not_a_thermodynamic_dividing_surface": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "trajectory_max_steps": trajectory_max_steps,
        "reference_structure": str(args.reference_structure.resolve()),
        "surface_atom_range": list(args.surface_range),
        "nitrogen_atom_range": list(args.nitrogen_range),
        "solution_atom_range": list(args.solution_range),
        "stages": [stage.__dict__ for stage in args.stage],
        "restart_policy": "later trajectory replaces earlier frame at duplicate timestep",
        "drop_first_frame_policy": (
            "drop the earliest raw timestep only when it is inside a requested stage"
        ),
        "top_si_definition": (
            "reference Si atoms within top_si_window_A of the highest reference Si; "
            "per-frame plane is their periodic mean z and ion distance is directed "
            "along +z modulo the box length"
        ),
        "terminal_plane_definition": (
            "dynamic translated-slab reference at terminal_surface_z_A with ion "
            "distance directed along +z modulo the box length"
        ),
        "gas_cluster_definition": "largest PBC-connected cluster of consecutive N2 molecular centers",
        "bubble_R90_definition": "90th percentile radius of main-cluster N2 molecular centers",
        "nearest_gas_interface_proxy": "minimum distance to a main-cluster N2 molecular center",
        "surface_origin_definition": (
            "a current solution-oxygen H atom has the same atom ID as an H initially "
            "assigned to a top-surface SiOH oxygen"
        ),
        "top_si_window_A": args.top_si_window_A,
        "terminal_surface_z_A": args.terminal_surface_z_A,
        "surface_depth_A": args.surface_depth_A,
        "cluster_cutoff_A": args.cluster_cutoff_A,
        "oh_cutoff_A": args.oh_cutoff_A,
        "ch_cutoff_A": args.ch_cutoff_A,
        "atom_types": {
            "H": args.hydrogen_type,
            "O": args.oxygen_type,
            "Na": args.sodium_type,
            "Cl": args.chloride_type,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-max-step", type=int, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-structure", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--nitrogen-range", type=parse_range, required=True)
    parser.add_argument("--solution-range", type=parse_range, required=True)
    parser.add_argument("--stage", type=parse_stage, action="append", required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--sodium-type", type=int, default=4)
    parser.add_argument("--chloride-type", type=int, default=5)
    parser.add_argument("--cluster-cutoff-A", type=float, default=5.5)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.25)
    parser.add_argument("--ch-cutoff-A", type=float, default=1.30)
    parser.add_argument("--top-si-window-A", type=float, default=1.0)
    parser.add_argument("--terminal-surface-z-A", type=float, required=True)
    parser.add_argument("--surface-depth-A", type=float, default=3.0)
    parser.add_argument(
        "--drop-first-frame", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.timestep_fs,
        args.cluster_cutoff_A,
        args.oh_cutoff_A,
        args.ch_cutoff_A,
        args.top_si_window_A,
        args.surface_depth_A,
    ) <= 0:
        raise ValueError("Time step, cutoffs, surface window, and depth must be positive")
    if args.trajectory_max_step and len(args.trajectory_max_step) != len(args.trajectory):
        raise ValueError("--trajectory-max-step must be supplied once per --trajectory")
    if args.trajectory_max_step and min(args.trajectory_max_step) < 0:
        raise ValueError("Trajectory maximum steps must be non-negative")
    names = [stage.name for stage in args.stage]
    if len(names) != len(set(names)):
        raise ValueError("Stage names must be unique")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
