"""Evaluate symmetric zero-time response kernels on frozen TPCL snapshots."""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from molsimflow.postprocess.tpcl_snapshot_mechanics import (
    _array_sha256,
    _evaluate_global,
    _minimum_image,
    _minimum_pair_distance,
    _model_type_indices,
    _read_specs,
    _sha256,
    _write_csv,
    read_selected_frames,
)

SCIENTIFIC_STATUS = (
    "SYMMETRIC_FROZEN_COORDINATE_STATIC_RESPONSE_KERNEL_"
    "NOT_DYNAMICS_PROPAGATION_BARRIER_RATE_OR_DISSIPATION_EVIDENCE"
)
ALLOWED_MODES = ("radial", "tangential", "normal")


@dataclass(frozen=True)
class MobileRange:
    case_id: str
    first_atom_id: int
    last_atom_id: int
    label: str


def _read_mobile_ranges(path: Path) -> dict[str, MobileRange]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty mobile-range table: {path}")
    output = {}
    for row in rows:
        item = MobileRange(
            case_id=row["case_id"],
            first_atom_id=int(row["first_atom_id"]),
            last_atom_id=int(row["last_atom_id"]),
            label=row["label"],
        )
        if item.case_id in output or item.first_atom_id < 1 or item.last_atom_id < item.first_atom_id:
            raise ValueError(f"invalid mobile range for {item.case_id}")
        output[item.case_id] = item
    return output


def _mode_direction(mode: str, radial: np.ndarray) -> np.ndarray:
    if mode == "radial":
        return radial
    if mode == "tangential":
        return np.asarray([-radial[1], radial[0], 0.0])
    if mode == "normal":
        return np.asarray([0.0, 0.0, 1.0])
    raise ValueError(f"unsupported response mode: {mode}")


def _localized_mode_vector(
    atom_ids: np.ndarray,
    coordinates: np.ndarray,
    lengths: np.ndarray,
    contact_xy: np.ndarray,
    radial: np.ndarray,
    mobile_range: MobileRange,
    mode: str,
    patch_radius_A: float,
    z_min_A: float,
    z_max_A: float,
) -> tuple[np.ndarray, int, float]:
    lateral = _minimum_image(coordinates[:, :2] - contact_xy, lengths[:2])
    distance = np.linalg.norm(lateral, axis=1)
    selected = (
        (atom_ids >= mobile_range.first_atom_id)
        & (atom_ids <= mobile_range.last_atom_id)
        & (distance < patch_radius_A)
        & (coordinates[:, 2] >= z_min_A)
        & (coordinates[:, 2] <= z_max_A)
    )
    if np.count_nonzero(selected) < 10:
        raise ValueError("localized mobile response field contains fewer than 10 atoms")
    weights = np.zeros(len(atom_ids), dtype=float)
    scaled = distance[selected] / patch_radius_A
    weights[selected] = 0.5 * (1.0 + np.cos(np.pi * scaled))
    maximum = float(np.max(weights))
    if maximum <= 0.0:
        raise ValueError("localized response field has zero weight")
    weights /= maximum
    vector = weights[:, None] * _mode_direction(mode, radial)[None, :]
    return vector, int(np.count_nonzero(selected)), float(np.sum(weights**2))


def _generalized_force(forces: np.ndarray, mode_vector: np.ndarray) -> float:
    return float(np.sum(forces * mode_vector))


def _relative_difference(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def run_virtual_response(
    snapshot_manifest: Path,
    mobile_ranges_path: Path,
    model_path: Path,
    type_map: Sequence[str],
    surface_elements: Sequence[str],
    output_dir: Path,
    *,
    modes: Sequence[str] = ALLOWED_MODES,
    displacement_magnitudes_A: Sequence[float] = (0.005, 0.010, 0.020),
    surface_quantile: float = 0.995,
    patch_below_surface_A: float = 2.0,
    patch_above_surface_A: float = 20.0,
    minimum_pair_distance_floor_A: float = 0.45,
    maximum_force_ceiling_eV_A: float = 100.0,
    stiffness_floor_eV_A2: float = 0.1,
    maximum_stiffness_relative_deviation: float = 0.20,
    maximum_force_even_ratio: float = 0.25,
    maximum_energy_gradient_relative_difference: float = 0.10,
    maximum_curvature_relative_difference: float = 0.25,
) -> dict[str, object]:
    modes = tuple(modes)
    displacements = tuple(sorted(float(value) for value in displacement_magnitudes_A))
    if not modes or len(modes) != len(set(modes)) or not set(modes).issubset(ALLOWED_MODES):
        raise ValueError("response modes are invalid or duplicated")
    if (
        len(displacements) != 3
        or len(displacements) != len(set(displacements))
        or displacements[0] <= 0.0
    ):
        raise ValueError("exactly three unique positive displacement magnitudes are required")
    if not 0.5 < surface_quantile <= 1.0 or patch_below_surface_A <= 0.0 or patch_above_surface_A <= 0.0:
        raise ValueError("invalid response-patch geometry")
    specs = _read_specs(snapshot_manifest)
    mobile_ranges = _read_mobile_ranges(mobile_ranges_path)
    if set(mobile_ranges) != {spec.case_id for spec in specs}:
        raise ValueError("mobile-range and snapshot case identities differ")
    frames = read_selected_frames(specs)
    from deepmd.infer import DeepPot

    model = DeepPot(str(model_path))
    model_type_map = tuple(model.get_type_map())
    model_type_indices = _model_type_indices(model_type_map, type_map)
    surface_type_ids = {
        index + 1 for index, element in enumerate(type_map) if element in set(surface_elements)
    }
    if not surface_type_ids:
        raise ValueError("surface element selection is empty")
    detail_rows = []
    summary_rows = []
    maximum_observed_force = 0.0
    minimum_observed_pair_distance = math.inf
    coordinate_update_count = 0
    for spec in specs:
        frame = frames[spec.snapshot_id]
        original_hash = _array_sha256(
            frame.atom_ids, frame.atom_types, frame.coordinates, frame.cell
        )
        lengths = np.diag(frame.cell)
        minimum_distance = _minimum_pair_distance(frame.coordinates, lengths)
        minimum_observed_pair_distance = min(minimum_observed_pair_distance, minimum_distance)
        if minimum_distance - 2.0 * displacements[-1] < minimum_pair_distance_floor_A:
            raise ValueError(
                f"{spec.snapshot_id}: baseline pair-distance margin cannot protect perturbations"
            )
        energy_zero, forces_zero, _ = _evaluate_global(model, frame, model_type_indices)
        if not math.isfinite(energy_zero) or not np.all(np.isfinite(forces_zero)):
            raise ValueError(f"{spec.snapshot_id}: non-finite baseline output")
        maximum_zero_force = float(np.max(np.linalg.norm(forces_zero, axis=1)))
        maximum_observed_force = max(maximum_observed_force, maximum_zero_force)
        if maximum_zero_force > maximum_force_ceiling_eV_A:
            raise ValueError(f"{spec.snapshot_id}: baseline force guard failed")
        surface_mask = np.isin(frame.atom_types, list(surface_type_ids))
        if not np.any(surface_mask):
            raise ValueError(f"{spec.snapshot_id}: no declared surface atoms")
        surface_top = float(np.quantile(frame.coordinates[surface_mask, 2], surface_quantile))
        radial = np.asarray([spec.radial_x, spec.radial_y, 0.0])
        radial /= np.linalg.norm(radial)
        contact_xy = np.asarray([spec.contact_x_A, spec.contact_y_A])
        for mode in modes:
            mode_vector, selected_count, effective_count = _localized_mode_vector(
                frame.atom_ids,
                frame.coordinates,
                lengths,
                contact_xy,
                radial,
                mobile_ranges[spec.case_id],
                mode,
                spec.patch_radius_A,
                surface_top - patch_below_surface_A,
                surface_top + patch_above_surface_A,
            )
            generalized_force_zero = _generalized_force(forces_zero, mode_vector)
            mode_details = []
            for displacement in displacements:
                plus_coordinates = np.mod(
                    frame.coordinates + displacement * mode_vector, lengths
                )
                minus_coordinates = np.mod(
                    frame.coordinates - displacement * mode_vector, lengths
                )
                plus_frame = type(frame)(
                    frame.step, frame.atom_ids, frame.atom_types, plus_coordinates, frame.cell
                )
                minus_frame = type(frame)(
                    frame.step, frame.atom_ids, frame.atom_types, minus_coordinates, frame.cell
                )
                energy_plus, forces_plus, _ = _evaluate_global(
                    model, plus_frame, model_type_indices
                )
                energy_minus, forces_minus, _ = _evaluate_global(
                    model, minus_frame, model_type_indices
                )
                if not all(
                    (
                        math.isfinite(energy_plus),
                        math.isfinite(energy_minus),
                        np.all(np.isfinite(forces_plus)),
                        np.all(np.isfinite(forces_minus)),
                    )
                ):
                    raise ValueError(f"{spec.snapshot_id}/{mode}: non-finite perturbed output")
                maximum_force = max(
                    float(np.max(np.linalg.norm(forces_plus, axis=1))),
                    float(np.max(np.linalg.norm(forces_minus, axis=1))),
                )
                maximum_observed_force = max(maximum_observed_force, maximum_force)
                if maximum_force > maximum_force_ceiling_eV_A:
                    raise ValueError(f"{spec.snapshot_id}/{mode}: perturbed force guard failed")
                force_plus = _generalized_force(forces_plus, mode_vector)
                force_minus = _generalized_force(forces_minus, mode_vector)
                stiffness_force = -(force_plus - force_minus) / (2.0 * displacement)
                energy_gradient = (energy_plus - energy_minus) / (2.0 * displacement)
                stiffness_energy = (
                    energy_plus + energy_minus - 2.0 * energy_zero
                ) / displacement**2
                even_ratio = abs(
                    force_plus + force_minus - 2.0 * generalized_force_zero
                ) / max(abs(force_plus - force_minus), 1.0e-12)
                gradient_difference = _relative_difference(
                    energy_gradient, -generalized_force_zero, 1.0e-8
                )
                curvature_difference = _relative_difference(
                    stiffness_force, stiffness_energy, stiffness_floor_eV_A2
                )
                item = {
                    "snapshot_id": spec.snapshot_id,
                    "pair_id": spec.pair_id,
                    "case_id": spec.case_id,
                    "sample_kind": spec.sample_kind,
                    "phase": spec.phase,
                    "source_time_block_200ps": spec.source_time_block_200ps,
                    "response_stratum": spec.response_stratum,
                    "response_affected_arc_fraction": spec.response_affected_arc_fraction,
                    "patch_radius_A": spec.patch_radius_A,
                    "mode": mode,
                    "mobile_label": mobile_ranges[spec.case_id].label,
                    "selected_atom_count": selected_count,
                    "effective_weighted_atom_count": effective_count,
                    "displacement_A": displacement,
                    "energy_zero_eV": energy_zero,
                    "energy_plus_eV": energy_plus,
                    "energy_minus_eV": energy_minus,
                    "generalized_force_zero_eV_A": generalized_force_zero,
                    "generalized_force_plus_eV_A": force_plus,
                    "generalized_force_minus_eV_A": force_minus,
                    "stiffness_from_force_eV_A2": stiffness_force,
                    "stiffness_from_energy_eV_A2": stiffness_energy,
                    "force_even_ratio": even_ratio,
                    "energy_gradient_relative_difference": gradient_difference,
                    "curvature_relative_difference": curvature_difference,
                    "maximum_force_eV_A": maximum_force,
                    "coordinate_identity_sha256": original_hash,
                    "scientific_status": SCIENTIFIC_STATUS,
                }
                detail_rows.append(item)
                mode_details.append(item)
            reference_stiffness = float(mode_details[0]["stiffness_from_force_eV_A2"])
            maximum_stiffness_deviation = max(
                abs(float(item["stiffness_from_force_eV_A2"]) - reference_stiffness)
                for item in mode_details
            ) / max(abs(reference_stiffness), stiffness_floor_eV_A2)
            maximum_even = max(float(item["force_even_ratio"]) for item in mode_details)
            maximum_gradient_difference = max(
                float(item["energy_gradient_relative_difference"]) for item in mode_details
            )
            maximum_curvature_difference = max(
                float(item["curvature_relative_difference"]) for item in mode_details
            )
            linear_pass = (
                maximum_stiffness_deviation <= maximum_stiffness_relative_deviation
                and maximum_even <= maximum_force_even_ratio
                and maximum_gradient_difference
                <= maximum_energy_gradient_relative_difference
                and maximum_curvature_difference <= maximum_curvature_relative_difference
            )
            summary_rows.append(
                {
                    "snapshot_id": spec.snapshot_id,
                    "pair_id": spec.pair_id,
                    "case_id": spec.case_id,
                    "sample_kind": spec.sample_kind,
                    "phase": spec.phase,
                    "source_time_block_200ps": spec.source_time_block_200ps,
                    "response_stratum": spec.response_stratum,
                    "response_affected_arc_fraction": spec.response_affected_arc_fraction,
                    "patch_radius_A": spec.patch_radius_A,
                    "mode": mode,
                    "mobile_label": mobile_ranges[spec.case_id].label,
                    "selected_atom_count": selected_count,
                    "effective_weighted_atom_count": effective_count,
                    "reference_displacement_A": displacements[0],
                    "static_stiffness_eV_A2": reference_stiffness,
                    "median_static_stiffness_eV_A2": float(
                        np.median(
                            [item["stiffness_from_force_eV_A2"] for item in mode_details]
                        )
                    ),
                    "maximum_stiffness_relative_deviation": maximum_stiffness_deviation,
                    "maximum_force_even_ratio": maximum_even,
                    "maximum_energy_gradient_relative_difference": maximum_gradient_difference,
                    "maximum_curvature_relative_difference": maximum_curvature_difference,
                    "linear_response_pass": int(linear_pass),
                    "coordinate_identity_sha256": original_hash,
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
        after_hash = _array_sha256(
            frame.atom_ids, frame.atom_types, frame.coordinates, frame.cell
        )
        coordinate_update_count += int(after_hash != original_hash)
        if after_hash != original_hash:
            raise ValueError(f"{spec.snapshot_id}: original coordinates changed")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "virtual_response_by_displacement.csv", detail_rows)
    _write_csv(output / "snapshot_virtual_response.csv", summary_rows)
    pair_modes = defaultdict(list)
    for row in summary_rows:
        pair_modes[(row["pair_id"], row["mode"])].append(row)
    complete_linear_pair_modes = [
        key
        for key, rows in pair_modes.items()
        if len(rows) == 6 and all(int(row["linear_response_pass"]) for row in rows)
    ]
    summary = {
        "status": "PASS",
        "snapshot_count": len(specs),
        "case_count": len({spec.case_id for spec in specs}),
        "mode_count": len(modes),
        "response_row_count": len(summary_rows),
        "displacement_row_count": len(detail_rows),
        "modes": list(modes),
        "model_type_map": list(model_type_map),
        "simulation_type_map": list(type_map),
        "simulation_to_model_type_index": model_type_indices.tolist(),
        "displacement_magnitudes_A": list(displacements),
        "linear_response_pass_count": sum(
            int(row["linear_response_pass"]) for row in summary_rows
        ),
        "complete_linear_pair_mode_count": len(complete_linear_pair_modes),
        "coordinate_update_count": coordinate_update_count,
        "timestep_advancement": 0,
        "maximum_observed_force_eV_A": maximum_observed_force,
        "minimum_observed_pair_distance_A": minimum_observed_pair_distance,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "snapshot_manifest": {
                    "path": str(snapshot_manifest),
                    "sha256": _sha256(snapshot_manifest),
                },
                "mobile_ranges": {
                    "path": str(mobile_ranges_path),
                    "sha256": _sha256(mobile_ranges_path),
                },
                "model": {"path": str(model_path), "sha256": _sha256(model_path)},
                "type_map": list(type_map),
                "surface_elements": list(surface_elements),
                "surface_quantile": surface_quantile,
                "patch_below_surface_A": patch_below_surface_A,
                "patch_above_surface_A": patch_above_surface_A,
                "minimum_pair_distance_floor_A": minimum_pair_distance_floor_A,
                "maximum_force_ceiling_eV_A": maximum_force_ceiling_eV_A,
                "stiffness_floor_eV_A2": stiffness_floor_eV_A2,
                "maximum_stiffness_relative_deviation": maximum_stiffness_relative_deviation,
                "maximum_force_even_ratio": maximum_force_even_ratio,
                "maximum_energy_gradient_relative_difference": (
                    maximum_energy_gradient_relative_difference
                ),
                "maximum_curvature_relative_difference": (
                    maximum_curvature_relative_difference
                ),
                "dump_files": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in sorted({spec.dump_path for spec in specs})
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "# Symmetric frozen-coordinate TPCL response kernel\n\n"
        "A cosine-tapered local displacement field is applied to the declared mobile "
        "atom-ID range at plus/minus three amplitudes along radial, tangential, and "
        "normal directions. Every energy and force is a zero-time prediction; the "
        "original coordinates are restored exactly. Linear-range, force symmetry, "
        "and energy-force consistency gates are recorded per snapshot and mode. "
        "Accepted quantities are static stiffness/response kernels, not dynamics, "
        "propagation, barriers, rates, dissipated work, or entropy production.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--mobile-ranges", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type-map", required=True)
    parser.add_argument("--surface-elements", required=True)
    parser.add_argument("--modes", default=",".join(ALLOWED_MODES))
    parser.add_argument("--displacement-magnitudes-A", default="0.005,0.010,0.020")
    parser.add_argument("--surface-quantile", type=float, default=0.995)
    parser.add_argument("--patch-below-surface-A", type=float, default=2.0)
    parser.add_argument("--patch-above-surface-A", type=float, default=20.0)
    parser.add_argument("--minimum-pair-distance-floor-A", type=float, default=0.45)
    parser.add_argument("--maximum-force-ceiling-eV-A", type=float, default=100.0)
    parser.add_argument("--stiffness-floor-eV-A2", type=float, default=0.1)
    parser.add_argument("--maximum-stiffness-relative-deviation", type=float, default=0.20)
    parser.add_argument("--maximum-force-even-ratio", type=float, default=0.25)
    parser.add_argument(
        "--maximum-energy-gradient-relative-difference", type=float, default=0.10
    )
    parser.add_argument("--maximum-curvature-relative-difference", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_virtual_response(
        args.snapshot_manifest,
        args.mobile_ranges,
        args.model,
        tuple(item.strip() for item in args.type_map.split(",") if item.strip()),
        tuple(item.strip() for item in args.surface_elements.split(",") if item.strip()),
        args.output_dir,
        modes=tuple(item.strip() for item in args.modes.split(",") if item.strip()),
        displacement_magnitudes_A=tuple(
            float(item)
            for item in args.displacement_magnitudes_A.split(",")
            if item.strip()
        ),
        surface_quantile=args.surface_quantile,
        patch_below_surface_A=args.patch_below_surface_A,
        patch_above_surface_A=args.patch_above_surface_A,
        minimum_pair_distance_floor_A=args.minimum_pair_distance_floor_A,
        maximum_force_ceiling_eV_A=args.maximum_force_ceiling_eV_A,
        stiffness_floor_eV_A2=args.stiffness_floor_eV_A2,
        maximum_stiffness_relative_deviation=args.maximum_stiffness_relative_deviation,
        maximum_force_even_ratio=args.maximum_force_even_ratio,
        maximum_energy_gradient_relative_difference=(
            args.maximum_energy_gradient_relative_difference
        ),
        maximum_curvature_relative_difference=(
            args.maximum_curvature_relative_difference
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
