"""Contract-driven offline path descriptors and grouped conditional diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from molsimflow.io import lammps_dump
from molsimflow.io.hashes import _sha256
from molsimflow.postprocess import (
    pimd_fes, pimd_path_io, pimd_reweight, quantum_path, quantum_path_contract, quantum_path_stats,
)


def native_logdistance(distance: np.ndarray | float) -> np.ndarray:
    """Native log(D+.03)*step(1-D)+(D-.9704412)*step(D-1), step(0)=1.

    Lengths must use the units of this specific Reactive Voronoi transform.
    No unit conversion or clipping of its logarithm domain is performed.
    """
    values = np.asarray(distance, dtype=float)
    result = pimd_reweight.piecewise_logdistance(values)
    return np.where(values == 1.0, np.log(1.03) + 1.0 - 0.9704412, result)


def _weights(config: dict, steps: Sequence[int]) -> np.ndarray:
    if config["kind"] == "uniform_sampler":
        return np.zeros(len(steps), dtype=float)
    with config["path"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["step", "log_weight"]:
            raise ValueError("weight CSV fields must be exactly step,log_weight")
        rows = list(reader)
    if any(set(row) != {"step", "log_weight"} for row in rows):
        raise ValueError("malformed weight CSV row")
    observed = [int(row["step"]) for row in rows]
    values = np.asarray([float(row["log_weight"]) for row in rows])
    if observed != list(steps) or not np.isfinite(values).all():
        raise ValueError("weight rows must be finite and exactly match selected physical steps")
    return values


def _region_fractions(values: dict[str, np.ndarray], regions: list[dict]) -> np.ndarray:
    """Each target region uses paired components of the same bead."""
    fractions = []
    for region in regions:
        inside = np.ones_like(values["q"], dtype=bool)
        for coordinate, (lower, upper) in region["bounds"].items():
            array = values[coordinate]
            inside &= (array >= lower) & (array < upper)
        fractions.append(float(np.mean(inside)))
    return np.asarray(fractions)


def _sources() -> dict[str, dict[str, str]]:
    paths = {"workflow": Path(__file__), "cli": Path(__file__).parents[1] / "cli.py", **{
        module.__name__: Path(module.__file__) for module in (
            lammps_dump, pimd_fes, pimd_path_io, pimd_reweight, quantum_path,
            quantum_path_contract, quantum_path_stats,
        )
    }}
    return {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()}


def analyze(contract_path: Path, output: Path) -> dict:
    """Run one explicitly identified trajectory into an exclusively new directory.

    Uniform weights describe the sampled distribution. External log weights
    remain a caller declaration; this command cannot establish equilibrium,
    time-dependent reweighting validity, physical kinetics or a GO decision.
    """
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    summary = {
        "status": "FAIL", "scientific_acceptance": "NOT_ASSESSED",
        "contract_path": str(Path(contract_path).resolve()), "source_before": {},
        "outputs": {},
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__},
    }
    contract = None
    try:
        summary["source_before"] = _sources()
        contract = quantum_path_contract.load_contract(contract_path)
        summary["input_hashes_before"] = dict(contract["_input_hashes"])
        if contract["geometry"]["length_unit"] != "angstrom":
            raise ValueError("native logdistance output requires length_unit=angstrom")
        summary["run"] = contract["run"]
        summary["weights"] = {key: str(value) if isinstance(value, Path) else value
                              for key, value in contract["weights"].items()}
        summary["distribution"] = (
            "SAMPLED_DISTRIBUTION_ONLY" if contract["weights"]["kind"] == "uniform_sampler"
            else "DECLARED_REWEIGHTED_TARGET_NOT_INDEPENDENTLY_ADMITTED"
        )
        grid = contract["steps"]
        steps = list(range(grid["first"], grid["last"] + 1, grid["stride"]))
        log_weights = _weights(contract["weights"], steps)
        beads = contract["beads"]
        paths = {bead["bead_id"]: bead["path"] for bead in beads}
        order = [bead["bead_id"] for bead in beads]
        identity = {atom["id"]: atom["type"] for atom in contract["atom_identity"]}
        geometry = {key: value for key, value in contract["geometry"].items()
                    if key != "length_unit"}
        analysis = contract.get("analysis")
        selected_rows, fractions = [], []
        observed = []
        with (output / "frames.csv").open("x", newline="") as frame_handle, (
            output / "beads.csv"
        ).open("x", newline="") as bead_handle:
            frame_writer = None
            bead_writer = csv.DictWriter(bead_handle, fieldnames=[
                "run_id", "step", "time_fs", "bead_id", "q", "distance", "logdistance",
            ])
            bead_writer.writeheader()
            for frame in pimd_path_io.iter_pimd_path_frames(
                paths, bead_order=order, expected_identity=identity, selected_steps=steps,
            ):
                index = len(observed)
                if index >= len(steps) or frame.step != steps[index]:
                    raise ValueError("observed path sequence differs from selected steps")
                observed.append(frame.step)
                descriptor = quantum_path.describe_quantum_path(
                    frame.positions, frame.atom_types, frame.bounds[:, 1] - frame.bounds[:, 0],
                    **geometry,
                )
                if abs(descriptor["variance_identity_residual"]) > 1e-12:
                    raise ValueError("occupation variance identity exceeds 1e-12")
                log_beads = native_logdistance(descriptor["distance_beads"])
                row = {**contract["run"], "step": frame.step,
                       "time_fs": frame.step * grid["timestep_fs"],
                       "log_weight": float(log_weights[index])}
                row.update({key: float(value) for key, value in descriptor.items()
                            if np.isscalar(value)})
                row.update(logdistance_centroid=float(native_logdistance(
                    descriptor["distance_centroid"])),
                    logdistance_bead_mean=float(np.mean(log_beads)),
                    logdistance_bead_variance=float(np.var(log_beads)))
                if not all(np.isfinite(value) for value in row.values()
                           if isinstance(value, (float, int))):
                    raise ValueError("nonfinite frame output")
                if frame_writer is None:
                    frame_writer = csv.DictWriter(frame_handle, fieldnames=list(row))
                    frame_writer.writeheader()
                frame_writer.writerow(row)
                for i, bead_id in enumerate(order):
                    bead_writer.writerow({"run_id": contract["run"]["run_id"],
                                          "step": frame.step, "time_fs": row["time_fs"],
                                          "bead_id": bead_id, "q": descriptor["q_beads"][i],
                                          "distance": descriptor["distance_beads"][i],
                                          "logdistance": log_beads[i]})
                if analysis is not None:
                    selected_rows.append(row)
                    fractions.append(_region_fractions({"q": descriptor["q_beads"],
                                                        "distance": descriptor["distance_beads"],
                                                        "logdistance": log_beads},
                                                       analysis["regions"]))
        if observed != steps:
            raise ValueError("input trajectories do not contain all selected steps")
        if analysis is not None:
            stats = quantum_path_stats.conditional_region_statistics(
                np.asarray([[row[key] for key in analysis["conditioning_fields"]]
                            for row in selected_rows]),
                np.asarray([row["v_occ"] for row in selected_rows]),
                np.asarray(fractions), log_weights, bin_edges=analysis["bin_edges"],
                block_ids=np.arange(len(steps), dtype=int) // analysis["block_frames"],
            )
            stats.update(conditioning_fields=analysis["conditioning_fields"],
                         path_field="v_occ", regions=analysis["regions"],
                         regions_may_overlap=True, target_region_intervals="[lower,upper)",
                         distribution=summary["distribution"],
                         scientific_acceptance="NOT_ASSESSED",
                         uncertainty_scope="Caller-declared contiguous blocks within one run")
            (output / "conditional.json").write_text(json.dumps(stats, indent=2, allow_nan=False)
                                                     + "\n")
        summary.update(status="PASS", physical_frames=len(observed), beads=len(order),
                       length_unit=contract["geometry"]["length_unit"],
                       frame_grouping="One physical frame; beads are not independent samples",
                       region_statistics="NOT_REQUESTED" if analysis is None else "WRITTEN")
    except Exception as exc:
        summary.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
    finally:
        try:
            if contract is not None:
                quantum_path_contract.verify_inputs_unchanged(contract)
                summary["input_hashes_unchanged"] = True
            summary["source_after"] = _sources()
            summary["source_unchanged"] = summary["source_before"] == summary["source_after"]
            if not summary["source_unchanged"]:
                raise ValueError("imported source changed during analysis")
        except Exception as exc:
            summary.update(status="FAIL", preservation_error=str(exc))
        summary["outputs"] = {path.name: _sha256(path) for path in output.iterdir()
                              if path.is_file() and path.name != "result.json"}
        (output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.contract, args.output)
    except (OSError, ValueError) as exc:
        print(f"QUANTUM_PATH_FAIL: {exc}")
        return 1
    print(f"QUANTUM_PATH_{summary['status']}: {args.output / 'result.json'}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
