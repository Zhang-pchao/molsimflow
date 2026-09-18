"""Periodic phase coordinate utilities for laterally localized droplets.

The primary coordinate is the phase of the first Fourier mode of a fixed atom
selection.  Constant positive weights are supported.  Coordinate-dependent
weights are intentionally excluded because their derivatives are part of the
biased Hamiltonian and cannot be treated as detached membership labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records


@dataclass(frozen=True)
class PeriodicPhase:
    """Value, resultant amplitude, and analytic Cartesian derivatives."""

    value: float
    resultant: float
    derivatives: np.ndarray


def minimum_image(value: float, period: float) -> float:
    """Return the scalar shortest displacement in ``[-period/2, period/2)``."""

    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")
    return float(value - period * math.floor(value / period + 0.5))


def periodic_phase(
    values: np.ndarray,
    lower: float,
    upper: float,
    *,
    weights: np.ndarray | None = None,
    minimum_resultant: float = 1.0e-12,
) -> PeriodicPhase:
    """Evaluate a weighted periodic phase and ``dq/dx_i``.

    ``weights`` must be constant with respect to the coordinates.  The
    derivatives therefore describe the complete gradient only for fixed
    weights.  The derivatives sum to one but individual values can be
    negative for atoms far from the localized density maximum.
    """

    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("values contain non-finite entries")
    period = float(upper) - float(lower)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("upper must be greater than lower")
    if weights is None:
        fixed_weights = np.ones(coordinates.size, dtype=float)
    else:
        fixed_weights = np.asarray(weights, dtype=float)
        if fixed_weights.shape != coordinates.shape:
            raise ValueError("weights must match values")
        if not np.all(np.isfinite(fixed_weights)) or np.any(fixed_weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
    weight_sum = float(np.sum(fixed_weights))
    if weight_sum <= 0.0:
        raise ValueError("at least one weight must be positive")

    angles = 2.0 * np.pi * (coordinates - float(lower)) / period
    complex_mean = np.sum(fixed_weights * np.exp(1j * angles)) / weight_sum
    resultant = float(abs(complex_mean))
    if resultant < minimum_resultant:
        raise ValueError(
            f"periodic phase is ill-conditioned: resultant={resultant:.6g} "
            f"< {minimum_resultant:.6g}"
        )
    angle = float(np.angle(complex_mean) % (2.0 * np.pi))
    phase = float(lower) + period * angle / (2.0 * np.pi)
    derivatives = fixed_weights * np.cos(angles - angle) / (weight_sum * resultant)
    return PeriodicPhase(phase, resultant, derivatives)


def harmonic_periodic_restraint(
    phase: float, target: float, kappa: float, period: float
) -> tuple[float, float, float]:
    """Return target-minus-phase displacement, energy, and force on the CV."""

    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    displacement = minimum_image(float(target) - float(phase), float(period))
    energy = 0.5 * float(kappa) * displacement * displacement
    force = float(kappa) * displacement
    return displacement, energy, force


def render_plumed_phase_restraint(
    atom_ids: Sequence[int],
    *,
    lower: float,
    upper: float,
    target: float,
    kappa: float,
    stride: int = 10,
    output: str = "COLVAR",
) -> str:
    """Render a PLUMED fixed-center restraint for an equal-weight phase CV."""

    ids = tuple(int(atom_id) for atom_id in atom_ids)
    if not ids or len(ids) != len(set(ids)) or min(ids) < 1:
        raise ValueError("atom_ids must be unique positive integers")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not math.isfinite(upper - lower) or upper <= lower:
        raise ValueError("upper must be greater than lower")
    if not math.isfinite(target) or not lower <= target < upper:
        raise ValueError("target must lie in [lower, upper)")
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    return "\n".join(
        (
            "UNITS LENGTH=A ENERGY=eV TIME=ps",
            f"waterO: GROUP ATOMS={','.join(map(str, ids))}",
            "center: CENTER ATOMS=waterO PHASES",
            "pos: POSITION ATOM=center NOPBC",
            f"qx: COMBINE ARG=pos.x COEFFICIENTS=1 PERIODIC={lower:.16g},{upper:.16g}",
            f"rest: RESTRAINT ARG=qx AT={target:.16g} KAPPA={kappa:.16g}",
            f"PRINT STRIDE={stride} ARG=qx,rest.bias,rest.force2 FILE={output} FMT=%20.12g",
            "FLUSH STRIDE=100",
            "",
        )
    )


def _frame_arrays(frame):
    index = {name: position for position, name in enumerate(frame.atom_fields)}
    missing = {"id", "type", "x", "y", "z"}.difference(index)
    if missing:
        raise ValueError(f"timestep {frame.timestep} missing columns: {sorted(missing)}")
    ids = np.asarray([int(row[index["id"]]) for row in frame.atom_rows], dtype=np.int64)
    types = np.asarray([int(row[index["type"]]) for row in frame.atom_rows], dtype=np.int64)
    coordinates = np.asarray(
        [[float(row[index[name]]) for name in ("x", "y", "z")] for row in frame.atom_rows],
        dtype=float,
    )
    return ids, types, coordinates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_trajectory(args: argparse.Namespace) -> dict[str, object]:
    """Audit phase conditioning and predicted force safety over a trajectory."""

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    identity = None
    period_x = None
    for frame in iter_lammps_dump_records(args.trajectory):
        ids, types, coordinates = _frame_arrays(frame)
        mask = (ids > args.nsub) & (types == args.oxygen_type)
        water_ids = ids[mask]
        water = coordinates[mask]
        if len(water_ids) != args.expected_count:
            raise ValueError(
                f"timestep {frame.timestep}: selected {len(water_ids)} water O, "
                f"expected {args.expected_count}"
            )
        current_identity = tuple(map(int, water_ids))
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"timestep {frame.timestep}: water-O identity/order changed")
        lower, upper = map(float, frame.bounds[0])
        current_period_x = upper - lower
        if period_x is None:
            period_x = current_period_x
        elif not math.isclose(current_period_x, period_x, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError(f"timestep {frame.timestep}: x box length changed")
        phase_x = periodic_phase(
            water[:, 0], lower, upper, minimum_resultant=args.minimum_resultant
        )
        phase_y = periodic_phase(water[:, 1], *map(float, frame.bounds[1]))
        order = np.argsort(water[:, 2])
        trim_count = max(1, math.ceil(0.01 * len(water)))
        retained = order[:-trim_count]
        trimmed = periodic_phase(water[retained, 0], lower, upper)
        period = upper - lower
        predicted = {}
        for offset in args.force_offset_A:
            force_q = args.kappa_eV_A2 * float(offset)
            predicted[f"offset_{offset:g}_A"] = float(
                abs(force_q) * np.max(np.abs(phase_x.derivatives))
            )
        rows.append(
            {
                "frame_index": frame.frame_index,
                "step": frame.timestep,
                "phase_x_A": phase_x.value,
                "phase_y_A": phase_y.value,
                "R1_x": phase_x.resultant,
                "R1_y": phase_y.resultant,
                "max_abs_dqx_dxi": float(np.max(np.abs(phase_x.derivatives))),
                "min_dqx_dxi": float(np.min(phase_x.derivatives)),
                "negative_dqx_fraction": float(np.mean(phase_x.derivatives < 0.0)),
                "sum_dqx_dxi": float(np.sum(phase_x.derivatives)),
                "remove_top_1pct_phase_shift_A": minimum_image(
                    trimmed.value - phase_x.value, period
                ),
                **{
                    f"predicted_max_atom_force_offset_{offset:g}_A_eV_A": predicted[
                        f"offset_{offset:g}_A"
                    ]
                    for offset in args.force_offset_A
                },
            }
        )
    if not rows or identity is None or period_x is None:
        raise ValueError("trajectory contains no frames")

    phase_unwrapped = [float(rows[0]["phase_x_A"])]
    for row in rows[1:]:
        phase_unwrapped.append(
            phase_unwrapped[-1]
            + minimum_image(float(row["phase_x_A"]) - phase_unwrapped[-1], period_x)
        )
    for row, value in zip(rows, phase_unwrapped):
        row["phase_x_unwrapped_A"] = value

    fields = list(rows[0])
    with (output / "phase_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    predicted_columns = [
        f"predicted_max_atom_force_offset_{offset:g}_A_eV_A"
        for offset in args.force_offset_A
    ]
    summary = {
        "status": "PASS",
        "trajectory": str(Path(args.trajectory).resolve()),
        "trajectory_size_bytes": Path(args.trajectory).stat().st_size,
        "trajectory_sha256": _sha256(Path(args.trajectory)),
        "frames": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "water_oxygen_count": len(identity),
        "selection": f"id > {args.nsub} and type == {args.oxygen_type}",
        "coordinate": "equal-weight first Fourier-mode phase of fixed water-O IDs",
        "constant_weights_only": True,
        "coordinate_dependent_membership_weights_are_not_detached": True,
        "R1_x_min": min(float(row["R1_x"]) for row in rows),
        "R1_x_max": max(float(row["R1_x"]) for row in rows),
        "negative_dqx_fraction_max": max(float(row["negative_dqx_fraction"]) for row in rows),
        "max_abs_dqx_dxi": max(float(row["max_abs_dqx_dxi"]) for row in rows),
        "max_abs_derivative_sum_error": max(
            abs(float(row["sum_dqx_dxi"]) - 1.0) for row in rows
        ),
        "max_abs_remove_top_1pct_phase_shift_A": max(
            abs(float(row["remove_top_1pct_phase_shift_A"])) for row in rows
        ),
        "phase_x_unwrapped_range_A": [min(phase_unwrapped), max(phase_unwrapped)],
        "kappa_eV_A2": args.kappa_eV_A2,
        "predicted_max_atom_force_eV_A": {
            column: max(float(row[column]) for row in rows) for column in predicted_columns
        },
        "force_ceiling_eV_A": args.force_ceiling_eV_A,
        "gates": {
            "phase_conditioning_R1_pass": min(float(row["R1_x"]) for row in rows)
            >= args.minimum_resultant,
            "derivative_sum_pass": max(
                abs(float(row["sum_dqx_dxi"]) - 1.0) for row in rows
            )
            <= 1.0e-10,
            "predicted_force_pass": max(
                float(row[column]) for row in rows for column in predicted_columns
            )
            <= args.force_ceiling_eV_A,
        },
        "scientific_scope": (
            "CV conditioning and analytic force-safety audit only; not an umbrella, PMF, "
            "friction, kinetics, or sampling-convergence result."
        ),
    }
    summary["status"] = "PASS" if all(summary["gates"].values()) else "FAIL"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "waterO.ids").write_text("\n".join(map(str, identity)) + "\n", encoding="utf-8")
    return summary


def _read_ids(path: Path) -> tuple[int, ...]:
    ids = tuple(int(value) for value in Path(path).read_text(encoding="utf-8").split())
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("ID file must contain unique integer atom IDs")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Audit a fixed water-O phase CV")
    audit.add_argument("--trajectory", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--nsub", type=int, required=True)
    audit.add_argument("--oxygen-type", type=int, default=2)
    audit.add_argument("--expected-count", type=int, required=True)
    audit.add_argument("--minimum-resultant", type=float, default=0.30)
    audit.add_argument("--kappa-eV-A2", type=float, default=0.02)
    audit.add_argument("--force-offset-A", type=float, nargs="+", default=(0.5, 1.0))
    audit.add_argument("--force-ceiling-eV-A", type=float, default=1.0e-4)
    render = subparsers.add_parser("render", help="Render a fixed-center PLUMED restraint")
    render.add_argument("--ids", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--lower", type=float, required=True)
    render.add_argument("--upper", type=float, required=True)
    render.add_argument("--target", type=float, required=True)
    render.add_argument("--kappa", type=float, required=True)
    render.add_argument("--stride", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit_trajectory(args), indent=2))
        return 0
    text = render_plumed_phase_restraint(
        _read_ids(args.ids),
        lower=args.lower,
        upper=args.upper,
        target=args.target,
        kappa=args.kappa,
        stride=args.stride,
    )
    Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
