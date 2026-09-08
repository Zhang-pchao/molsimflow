"""Evaluate frozen TPCL snapshots with a Deep Potential model at zero time."""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

import numpy as np

from molsimflow.io.lammps_dump import open_lammps_dump_text

SCIENTIFIC_STATUS = (
    "ZERO_TIME_DPA_SNAPSHOT_ENERGY_FORCE_VIRIAL_AND_COARSE_TRACTION_"
    "NOT_DYNAMICS_BARRIER_DISSIPATION_OR_CAUSAL_EVIDENCE"
)
GLOBAL_FORCE_SCIENTIFIC_STATUS = (
    "ZERO_TIME_DPA_SNAPSHOT_ENERGY_ATOM_FORCE_AND_GLOBAL_VIRIAL_"
    "NOT_ATOMIC_ENERGY_LOCAL_STRESS_DYNAMICS_BARRIER_DISSIPATION_OR_CAUSAL_EVIDENCE"
)
VIRIAL_NAMES = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")


@dataclass(frozen=True)
class SnapshotSpec:
    snapshot_id: str
    pair_id: str
    case_id: str
    sample_kind: str
    phase: str
    step: int
    dump_path: Path
    contact_x_A: float
    contact_y_A: float
    radial_x: float
    radial_y: float
    patch_radius_A: float
    source_time_block_200ps: int
    response_stratum: int
    response_affected_arc_fraction: float


@dataclass(frozen=True)
class DumpFrame:
    step: int
    atom_ids: np.ndarray
    atom_types: np.ndarray
    coordinates: np.ndarray
    cell: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _read_specs(path: Path) -> list[SnapshotSpec]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty snapshot manifest: {path}")
    output = []
    seen = set()
    for row in rows:
        snapshot_id = f"{row['pair_id']}__{row['sample_kind']}__{row['phase']}"
        if snapshot_id in seen:
            raise ValueError(f"duplicate snapshot id: {snapshot_id}")
        seen.add(snapshot_id)
        output.append(
            SnapshotSpec(
                snapshot_id=snapshot_id,
                pair_id=row["pair_id"],
                case_id=row["case_id"],
                sample_kind=row["sample_kind"],
                phase=row["phase"],
                step=int(row["step"]),
                dump_path=Path(row["dump_path"]),
                contact_x_A=float(row["contact_x_A"]),
                contact_y_A=float(row["contact_y_A"]),
                radial_x=float(row["radial_x"]),
                radial_y=float(row["radial_y"]),
                patch_radius_A=float(row["patch_radius_A"]),
                source_time_block_200ps=int(row["source_time_block_200ps"]),
                response_stratum=int(row["response_stratum"]),
                response_affected_arc_fraction=float(row["response_affected_arc_fraction"]),
            )
        )
    return output


def _next_line(handle: IO[str], context: str) -> str:
    line = handle.readline()
    if not line:
        raise ValueError(f"unexpected EOF while reading {context}")
    return line.rstrip("\n")


def _iter_dump(path: Path) -> Iterator[DumpFrame]:
    with open_lammps_dump_text(path) as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.rstrip("\n") != "ITEM: TIMESTEP":
                raise ValueError(f"{path}: expected TIMESTEP marker")
            step = int(_next_line(handle, "step"))
            if _next_line(handle, "atom-count marker") != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"{path}/{step}: missing atom-count marker")
            atom_count = int(_next_line(handle, "atom count"))
            bounds_header = _next_line(handle, "box header")
            if bounds_header != "ITEM: BOX BOUNDS pp pp pp":
                raise ValueError(f"{path}/{step}: only orthogonal periodic boxes are supported")
            bounds = []
            for axis in range(3):
                tokens = _next_line(handle, f"box bound {axis}").split()
                if len(tokens) != 2:
                    raise ValueError(f"{path}/{step}: invalid orthogonal box bound")
                bounds.append((float(tokens[0]), float(tokens[1])))
            atom_header = _next_line(handle, "atom header").split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError(f"{path}/{step}: invalid atom header")
            columns = atom_header[2:]
            required = {"id", "type", "x", "y", "z"}
            if not required.issubset(columns):
                raise ValueError(f"{path}/{step}: required atom fields are absent")
            positions = {field: columns.index(field) for field in required}
            atom_ids = np.empty(atom_count, dtype=np.int64)
            atom_types = np.empty(atom_count, dtype=np.int32)
            coordinates = np.empty((atom_count, 3), dtype=np.float64)
            for index in range(atom_count):
                tokens = _next_line(handle, f"atom row {index}").split()
                if len(tokens) != len(columns):
                    raise ValueError(
                        f"{path}/{step}: atom row {index} has {len(tokens)} fields; "
                        f"expected {len(columns)}"
                    )
                atom_ids[index] = int(tokens[positions["id"]])
                atom_types[index] = int(tokens[positions["type"]])
                coordinates[index] = [float(tokens[positions[field]]) for field in ("x", "y", "z")]
            order = np.argsort(atom_ids)
            atom_ids = atom_ids[order]
            atom_types = atom_types[order]
            coordinates = coordinates[order]
            origin = np.asarray([item[0] for item in bounds], dtype=float)
            coordinates -= origin
            lengths = np.asarray([item[1] - item[0] for item in bounds], dtype=float)
            yield DumpFrame(
                step=step,
                atom_ids=atom_ids,
                atom_types=atom_types,
                coordinates=coordinates,
                cell=np.diag(lengths),
            )


def read_selected_frames(specs: Sequence[SnapshotSpec]) -> dict[str, DumpFrame]:
    """Read each compressed dump once and retain only requested timesteps."""

    grouped = defaultdict(list)
    for spec in specs:
        grouped[spec.dump_path].append(spec)
    output = {}
    for path, path_specs in grouped.items():
        requested = {spec.step for spec in path_specs}
        remaining = set(requested)
        by_step = {}
        for frame in _iter_dump(path):
            if frame.step in remaining:
                by_step[frame.step] = frame
                remaining.remove(frame.step)
                if not remaining:
                    break
        if remaining:
            raise ValueError(f"{path}: missing selected steps {sorted(remaining)}")
        for spec in path_specs:
            output[spec.snapshot_id] = by_step[spec.step]
    return output


def _model_type_indices(
    model_type_map: Sequence[str], simulation_type_map: Sequence[str]
) -> np.ndarray:
    if not simulation_type_map or len(simulation_type_map) != len(set(simulation_type_map)):
        raise ValueError("simulation type map must be non-empty and unique")
    missing = [element for element in simulation_type_map if element not in model_type_map]
    if missing:
        raise ValueError(f"simulation elements are absent from model type map: {missing}")
    return np.asarray([model_type_map.index(element) for element in simulation_type_map])


def _evaluate(
    model: object, frame: DumpFrame, model_type_indices: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.any(frame.atom_types < 1) or np.any(frame.atom_types > len(model_type_indices)):
        raise ValueError("atom type lies outside the declared simulation type map")
    mapped_types = model_type_indices[frame.atom_types.astype(int) - 1]
    result = model.eval(
        frame.coordinates.reshape(1, -1),
        frame.cell.reshape(1, 9),
        mapped_types,
        atomic=True,
    )
    if len(result) != 5:
        raise ValueError(f"DeepPot atomic evaluation returned {len(result)} arrays, expected 5")
    energy, forces, virial, atomic_energy, atomic_virial = result
    return (
        float(np.asarray(energy).reshape(-1)[0]),
        np.asarray(forces, dtype=float).reshape(-1, 3),
        np.asarray(virial, dtype=float).reshape(-1, 9)[0],
        np.asarray(atomic_energy, dtype=float).reshape(-1),
        np.asarray(atomic_virial, dtype=float).reshape(-1, 9),
    )


def _evaluate_global(
    model: object, frame: DumpFrame, model_type_indices: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    if np.any(frame.atom_types < 1) or np.any(frame.atom_types > len(model_type_indices)):
        raise ValueError("atom type lies outside the declared simulation type map")
    mapped_types = model_type_indices[frame.atom_types.astype(int) - 1]
    result = model.eval(
        frame.coordinates.reshape(1, -1),
        frame.cell.reshape(1, 9),
        mapped_types,
        atomic=False,
    )
    if len(result) != 3:
        raise ValueError(f"DeepPot global evaluation returned {len(result)} arrays, expected 3")
    energy, forces, virial = result
    return (
        float(np.asarray(energy).reshape(-1)[0]),
        np.asarray(forces, dtype=float).reshape(-1, 3),
        np.asarray(virial, dtype=float).reshape(-1, 9)[0],
    )


def _minimum_image(delta: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return delta - lengths * np.round(delta / lengths)


def _minimum_pair_distance(coordinates: np.ndarray, lengths: np.ndarray) -> float:
    """Return the closest periodic pair distance for an orthogonal cell."""

    from scipy.spatial import cKDTree

    wrapped = np.mod(coordinates, lengths)
    distances, _ = cKDTree(wrapped, boxsize=lengths).query(wrapped, k=2, workers=1)
    return float(np.min(distances[:, 1]))


def _project_tensor(tensor: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ tensor.reshape(3, 3) @ right)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_predictions(
    snapshot_manifest: Path,
    model_path: Path,
    type_map: Sequence[str],
    surface_elements: Sequence[str],
    output_dir: Path,
    *,
    atomic_outputs: bool = True,
    surface_quantile: float = 0.995,
    patch_below_surface_A: float = 2.0,
    patch_above_surface_A: float = 20.0,
    patch_radii_A: Optional[Sequence[float]] = None,
    minimum_pair_distance_floor_A: float = 0.45,
    maximum_force_ceiling_eV_A: float = 100.0,
    virial_reconstruction_atol_eV: float = 1.0e-6,
    virial_reconstruction_rtol: float = 1.0e-8,
) -> dict[str, object]:
    """Evaluate immutable coordinates and write coarse mechanical observables."""

    if not 0.5 < surface_quantile <= 1.0 or patch_below_surface_A <= 0.0 or patch_above_surface_A <= 0.0:
        raise ValueError("invalid patch geometry")
    if minimum_pair_distance_floor_A <= 0.0 or maximum_force_ceiling_eV_A <= 0.0:
        raise ValueError("invalid anomaly guard")
    if virial_reconstruction_atol_eV < 0.0 or virial_reconstruction_rtol < 0.0:
        raise ValueError("invalid virial reconstruction tolerance")
    specs = _read_specs(snapshot_manifest)
    if patch_radii_A is not None:
        patch_radii_A = tuple(float(radius) for radius in patch_radii_A)
        if not patch_radii_A or any(radius <= 0.0 for radius in patch_radii_A):
            raise ValueError("patch radii must be positive")
        if len(set(patch_radii_A)) != len(patch_radii_A):
            raise ValueError("patch radii must be unique")
    frames = read_selected_frames(specs)
    from deepmd.infer import DeepPot  # imported only in the declared DeepMD runtime

    model = DeepPot(str(model_path))
    model_type_map = tuple(model.get_type_map())
    model_type_indices = _model_type_indices(model_type_map, type_map)
    surface_type_ids = {
        index + 1 for index, element in enumerate(type_map) if element in set(surface_elements)
    }
    if not surface_type_ids:
        raise ValueError("surface element selection is empty")

    summary_rows = []
    type_rows = []
    case_identity = {}
    maximum_virial_difference = 0.0
    maximum_virial_relative_difference = 0.0
    minimum_observed_pair_distance = math.inf
    maximum_observed_force = 0.0
    scientific_status = SCIENTIFIC_STATUS if atomic_outputs else GLOBAL_FORCE_SCIENTIFIC_STATUS
    for spec in specs:
        frame = frames[spec.snapshot_id]
        before_hash = _array_sha256(frame.atom_ids, frame.atom_types, frame.coordinates, frame.cell)
        identity_hash = _array_sha256(frame.atom_ids, frame.atom_types)
        if spec.case_id in case_identity and case_identity[spec.case_id] != identity_hash:
            raise ValueError(f"{spec.case_id}: atom id/type order changed across selected snapshots")
        case_identity[spec.case_id] = identity_hash
        if np.any(frame.atom_types < 1) or np.any(frame.atom_types > len(type_map)):
            raise ValueError(f"{spec.snapshot_id}: atom type lies outside the declared type map")
        minimum_distance = _minimum_pair_distance(frame.coordinates, np.diag(frame.cell))
        minimum_observed_pair_distance = min(minimum_observed_pair_distance, minimum_distance)
        if minimum_distance < minimum_pair_distance_floor_A:
            raise ValueError(
                f"{spec.snapshot_id}: minimum pair distance {minimum_distance:.6g} A is below "
                f"the frozen {minimum_pair_distance_floor_A:.6g} A guard"
            )
        if atomic_outputs:
            energy, forces, virial, atomic_energy, atomic_virial = _evaluate(
                model, frame, model_type_indices
            )
        else:
            energy, forces, virial = _evaluate_global(model, frame, model_type_indices)
            atomic_energy = None
            atomic_virial = None
        after_hash = _array_sha256(frame.atom_ids, frame.atom_types, frame.coordinates, frame.cell)
        if before_hash != after_hash:
            raise ValueError(f"{spec.snapshot_id}: coordinates or identities changed during prediction")
        if forces.shape != frame.coordinates.shape:
            raise ValueError(f"{spec.snapshot_id}: unexpected atomic output shape")
        if atomic_outputs and (
            atomic_energy is None
            or atomic_virial is None
            or len(atomic_energy) != len(frame.atom_ids)
            or atomic_virial.shape != (len(frame.atom_ids), 9)
        ):
            raise ValueError(f"{spec.snapshot_id}: unexpected atomic output shape")
        predicted_arrays = [forces, virial]
        if atomic_outputs:
            predicted_arrays.extend([atomic_energy, atomic_virial])
        if not all(np.all(np.isfinite(values)) for values in predicted_arrays) or not math.isfinite(
            energy
        ):
            raise ValueError(f"{spec.snapshot_id}: non-finite model output")
        force_norm = np.linalg.norm(forces, axis=1)
        maximum_force = float(np.max(force_norm))
        maximum_observed_force = max(maximum_observed_force, maximum_force)
        if maximum_force > maximum_force_ceiling_eV_A:
            raise ValueError(
                f"{spec.snapshot_id}: maximum force {maximum_force:.6g} eV/A exceeds "
                f"the frozen {maximum_force_ceiling_eV_A:.6g} eV/A guard"
            )
        virial_difference = None
        virial_relative_difference = None
        if atomic_outputs:
            reconstructed = np.sum(atomic_virial, axis=0)
            virial_difference = float(np.max(np.abs(reconstructed - virial)))
            virial_scale = max(1.0, float(np.max(np.abs(virial))))
            virial_relative_difference = virial_difference / virial_scale
            maximum_virial_difference = max(maximum_virial_difference, virial_difference)
            maximum_virial_relative_difference = max(
                maximum_virial_relative_difference, virial_relative_difference
            )
            if not np.allclose(
                reconstructed,
                virial,
                atol=virial_reconstruction_atol_eV,
                rtol=virial_reconstruction_rtol,
            ):
                raise ValueError(
                    f"{spec.snapshot_id}: atomic virial sum does not reconstruct global virial"
                )
        lengths = np.diag(frame.cell)
        lateral = frame.coordinates[:, :2] - np.asarray([spec.contact_x_A, spec.contact_y_A])
        lateral = _minimum_image(lateral, lengths[:2])
        lateral_distance = np.linalg.norm(lateral, axis=1)
        surface_mask = np.isin(frame.atom_types, list(surface_type_ids))
        if not np.any(surface_mask):
            raise ValueError(f"{spec.snapshot_id}: no declared surface atoms")
        surface_top = float(np.quantile(frame.coordinates[surface_mask, 2], surface_quantile))
        radial = np.asarray([spec.radial_x, spec.radial_y, 0.0])
        radial /= np.linalg.norm(radial)
        tangent = np.asarray([-radial[1], radial[0], 0.0])
        normal = np.asarray([0.0, 0.0, 1.0])
        radii = patch_radii_A if patch_radii_A is not None else (spec.patch_radius_A,)
        for patch_radius_A in radii:
            patch = (
                (lateral_distance <= patch_radius_A)
                & (frame.coordinates[:, 2] >= surface_top - patch_below_surface_A)
                & (frame.coordinates[:, 2] <= surface_top + patch_above_surface_A)
            )
            if np.count_nonzero(patch) < 10:
                raise ValueError(f"{spec.snapshot_id}: TPCL patch is unexpectedly empty")
            patch_force = np.sum(forces[patch], axis=0)
            patch_virial = np.sum(atomic_virial[patch], axis=0) if atomic_outputs else None
            patch_virial_tensor = patch_virial.reshape(3, 3) if patch_virial is not None else None
            patch_area = math.pi * patch_radius_A**2
            row: dict[str, object] = {
                "snapshot_id": spec.snapshot_id,
                "pair_id": spec.pair_id,
                "case_id": spec.case_id,
                "sample_kind": spec.sample_kind,
                "phase": spec.phase,
                "step": spec.step,
                "source_time_block_200ps": spec.source_time_block_200ps,
                "response_stratum": spec.response_stratum,
                "response_affected_arc_fraction": spec.response_affected_arc_fraction,
                "atom_count": len(frame.atom_ids),
                "coordinate_identity_sha256": before_hash,
                "atom_id_type_sha256": identity_hash,
                "atomic_outputs_requested": int(atomic_outputs),
                "energy_eV": energy,
                "atomic_energy_sum_eV": float(np.sum(atomic_energy)) if atomic_outputs else "",
                "atomic_energy_sum_difference_eV": (
                    float(np.sum(atomic_energy) - energy) if atomic_outputs else ""
                ),
                "minimum_pair_distance_A": minimum_distance,
                "force_rms_eV_A": float(np.sqrt(np.mean(forces**2))),
                "force_norm_p99_eV_A": float(np.quantile(force_norm, 0.99)),
                "force_norm_max_eV_A": maximum_force,
                "net_system_force_norm_eV_A": float(np.linalg.norm(np.sum(forces, axis=0))),
                "atomic_virial_sum_max_abs_difference_eV": virial_difference,
                "atomic_virial_sum_max_relative_difference": virial_relative_difference,
                "surface_top_z_A": surface_top,
                "patch_radius_A": patch_radius_A,
                "is_primary_patch_radius": int(
                    math.isclose(patch_radius_A, spec.patch_radius_A, abs_tol=1.0e-12)
                ),
                "patch_atom_count": int(np.count_nonzero(patch)),
                "patch_area_A2": patch_area,
                "patch_force_x_eV_A": patch_force[0],
                "patch_force_y_eV_A": patch_force[1],
                "patch_force_z_eV_A": patch_force[2],
                "patch_generalized_radial_force_eV_A": float(np.dot(patch_force, radial)),
                "patch_generalized_tangential_force_eV_A": float(np.dot(patch_force, tangent)),
                "patch_generalized_normal_force_eV_A": patch_force[2],
                "patch_radial_force_per_area_eV_A3": float(np.dot(patch_force, radial))
                / patch_area,
                "patch_tangential_force_per_area_eV_A3": float(np.dot(patch_force, tangent))
                / patch_area,
                "patch_normal_force_per_area_eV_A3": patch_force[2] / patch_area,
                "patch_atomic_virial_rr_eV": (
                    _project_tensor(patch_virial_tensor, radial, radial)
                    if atomic_outputs
                    else ""
                ),
                "patch_atomic_virial_tt_eV": (
                    _project_tensor(patch_virial_tensor, tangent, tangent)
                    if atomic_outputs
                    else ""
                ),
                "patch_atomic_virial_nn_eV": (
                    _project_tensor(patch_virial_tensor, normal, normal)
                    if atomic_outputs
                    else ""
                ),
                "patch_atomic_virial_rn_sym_eV": (
                    0.5
                    * (
                        _project_tensor(patch_virial_tensor, radial, normal)
                        + _project_tensor(patch_virial_tensor, normal, radial)
                    )
                    if atomic_outputs
                    else ""
                ),
            }
            for index, name in enumerate(VIRIAL_NAMES):
                row[f"global_virial_{name}_eV"] = virial[index]
                row[f"patch_atomic_virial_{name}_eV"] = (
                    patch_virial[index] if atomic_outputs else ""
                )
            row["scientific_status"] = scientific_status
            summary_rows.append(row)
            for type_id, element in enumerate(type_map, start=1):
                selected = patch & (frame.atom_types == type_id)
                selected_force = (
                    np.sum(forces[selected], axis=0) if np.any(selected) else np.zeros(3)
                )
                selected_virial = (
                    np.sum(atomic_virial[selected], axis=0)
                    if np.any(selected)
                    else np.zeros(9)
                ).reshape(3, 3) if atomic_outputs else None
                type_rows.append(
                    {
                        "snapshot_id": spec.snapshot_id,
                        "pair_id": spec.pair_id,
                        "case_id": spec.case_id,
                        "sample_kind": spec.sample_kind,
                        "phase": spec.phase,
                        "source_time_block_200ps": spec.source_time_block_200ps,
                        "response_stratum": spec.response_stratum,
                        "response_affected_arc_fraction": spec.response_affected_arc_fraction,
                        "patch_radius_A": patch_radius_A,
                        "type_id": type_id,
                        "element": element,
                        "patch_atom_count": int(np.count_nonzero(selected)),
                        "patch_force_radial_eV_A": float(np.dot(selected_force, radial)),
                        "patch_force_tangential_eV_A": float(np.dot(selected_force, tangent)),
                        "patch_force_normal_eV_A": selected_force[2],
                        "patch_atomic_virial_rr_eV": (
                            _project_tensor(selected_virial, radial, radial)
                            if atomic_outputs
                            else ""
                        ),
                        "patch_atomic_virial_tt_eV": (
                            _project_tensor(selected_virial, tangent, tangent)
                            if atomic_outputs
                            else ""
                        ),
                        "patch_atomic_virial_nn_eV": (
                            _project_tensor(selected_virial, normal, normal)
                            if atomic_outputs
                            else ""
                        ),
                        "scientific_status": scientific_status,
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "snapshot_mechanics.csv", summary_rows)
    _write_csv(output / "snapshot_patch_by_type.csv", type_rows)
    summary = {
        "status": "PASS",
        "snapshot_count": len(specs),
        "case_count": len({spec.case_id for spec in specs}),
        "atomic_outputs_requested": atomic_outputs,
        "model_type_map": model_type_map,
        "simulation_type_map": list(type_map),
        "simulation_to_model_type_index": model_type_indices.tolist(),
        "surface_elements": list(surface_elements),
        "maximum_atomic_global_virial_difference_eV": (
            maximum_virial_difference if atomic_outputs else None
        ),
        "maximum_atomic_global_virial_relative_difference": (
            maximum_virial_relative_difference if atomic_outputs else None
        ),
        "minimum_observed_pair_distance_A": minimum_observed_pair_distance,
        "maximum_observed_force_eV_A": maximum_observed_force,
        "anomaly_guard_status": "PASS",
        "coordinate_update_count": 0,
        "timestep_advancement": 0,
        "scientific_status": scientific_status,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "snapshot_manifest": {
                    "path": str(snapshot_manifest),
                    "sha256": _sha256(snapshot_manifest),
                },
                "model": {"path": str(model_path), "sha256": _sha256(model_path)},
                "type_map": list(type_map),
                "surface_quantile": surface_quantile,
                "patch_below_surface_A": patch_below_surface_A,
                "patch_above_surface_A": patch_above_surface_A,
                "atomic_outputs_requested": atomic_outputs,
                "patch_radii_A": list(patch_radii_A) if patch_radii_A is not None else None,
                "minimum_pair_distance_floor_A": minimum_pair_distance_floor_A,
                "maximum_force_ceiling_eV_A": maximum_force_ceiling_eV_A,
                "virial_reconstruction_atol_eV": virial_reconstruction_atol_eV,
                "virial_reconstruction_rtol": virial_reconstruction_rtol,
                "dump_files": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in sorted({spec.dump_path for spec in specs})
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if atomic_outputs:
        report = (
            "Atomic virials are used only after their sum reconstructs the model global "
            "virial and are reported as coarse patch sums, not unique local stress."
        )
    else:
        report = (
            "The supported global-output path returns atom-resolved forces but no atomic "
            "energy or atomic virial. Patch-force sums are instantaneous mechanical "
            "imbalance proxies, not local stress."
        )
    (output / "REPORT.md").write_text(
        "# Zero-time DPA snapshot mechanics\n\n"
        "The original coordinates are evaluated without a timestep, velocity update, "
        f"thermostat, minimization, or trajectory continuation. {report} Generalized "
        "patch forces are not barriers, dissipated work, entropy production, or causal "
        "propagation.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type-map", required=True, help="comma-separated DeepMD type map")
    parser.add_argument("--surface-elements", required=True, help="comma-separated elements")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--global-force-only",
        action="store_true",
        help="request energy, atom-resolved force, and global virial without atomic outputs",
    )
    parser.add_argument("--patch-radii-A", help="optional comma-separated patch radii")
    parser.add_argument("--surface-quantile", type=float, default=0.995)
    parser.add_argument("--patch-below-surface-A", type=float, default=2.0)
    parser.add_argument("--patch-above-surface-A", type=float, default=20.0)
    parser.add_argument("--minimum-pair-distance-floor-A", type=float, default=0.45)
    parser.add_argument("--maximum-force-ceiling-eV-A", type=float, default=100.0)
    parser.add_argument("--virial-reconstruction-atol-eV", type=float, default=1.0e-6)
    parser.add_argument("--virial-reconstruction-rtol", type=float, default=1.0e-8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_predictions(
        args.snapshot_manifest,
        args.model,
        tuple(item.strip() for item in args.type_map.split(",") if item.strip()),
        tuple(item.strip() for item in args.surface_elements.split(",") if item.strip()),
        args.output_dir,
        atomic_outputs=not args.global_force_only,
        patch_radii_A=(
            tuple(float(item) for item in args.patch_radii_A.split(",") if item.strip())
            if args.patch_radii_A
            else None
        ),
        surface_quantile=args.surface_quantile,
        patch_below_surface_A=args.patch_below_surface_A,
        patch_above_surface_A=args.patch_above_surface_A,
        minimum_pair_distance_floor_A=args.minimum_pair_distance_floor_A,
        maximum_force_ceiling_eV_A=args.maximum_force_ceiling_eV_A,
        virial_reconstruction_atol_eV=args.virial_reconstruction_atol_eV,
        virial_reconstruction_rtol=args.virial_reconstruction_rtol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
