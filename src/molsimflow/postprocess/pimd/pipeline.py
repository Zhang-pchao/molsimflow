"""Execute the legacy contract through explicit PIMD analysis stages."""

from __future__ import annotations

import json
from .plotting import configure_plotter
from dataclasses import replace
from .kde import resolve_bandwidths
from .reporting import write_kde_uncertainty, write_unsmoothed_histogram
from pathlib import Path
from typing import Dict

from . import plotting as _default_plotting
from .alignment import align_inputs
from .contract import analysis_profile
from .core_reporting import build_joint_outputs, finalize_core_1d, finalize_core_2d
from .derived_reporting import write_derived_coordinate
from .io import prepare_inputs
from .reporting import write_block_diagnostics, write_frame_tables
from .water_ionization import finalize_water
from .weights import prepare_weights


def analyze(contract_path: Path, output: Path, *, _plotter=None) -> Dict[str, object]:
    """Read, align, weight, estimate and report without changing legacy defaults."""
    if _plotter is None:
        _plotter = _default_plotting
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _plotter = configure_plotter(_plotter, contract.get("plots", {}))
    profile = analysis_profile(contract)
    output = Path(output)
    require(not output.exists(), f"output exists: {output}")
    for name in ("inputs", "tables", "fes1d", "fes2d", "blocks", "figures", "qc", "provenance"):
        (output / name).mkdir(parents=True, exist_ok=False)
    inputs = prepare_inputs(contract, output, profile)
    aligned = align_inputs(contract, output, profile, inputs)
    weighted = prepare_weights(contract, output, profile, inputs, aligned)
    if "bandwidth_rule" in inputs.reweight:
        explicit = bool(inputs.reweight.get("bandwidth_variants"))
        inputs = replace(inputs, reweight=resolve_bandwidths(inputs.reweight, aligned.beads, weighted.weights))
        (output / "qc" / "bandwidth-selection.json").write_text(json.dumps({
            "rule": "weighted_scott", "selection": "explicit" if explicit else "computed",
            "bandwidth_variants": inputs.reweight["bandwidth_variants"],
            "scope": "Global frame-Kish rule; not correlation-adjusted or locally adaptive",
        }, indent=2) + "\n", encoding="utf-8")
    write_kde_uncertainty(output, inputs.reweight, aligned.cv_names, aligned.beads,
                          weighted.raw_log_weights, weighted.kbt_ev)
    if "histogram" in inputs.reweight:
        write_unsmoothed_histogram(output, inputs.reweight, aligned.cv_names, aligned.beads,
                                  weighted.raw_log_weights, weighted.kbt_ev)


    if len(aligned.cv_names) == 1:
        return finalize_core_1d(
            contract=contract,
            output=output,
            source=inputs.source,
            reweight=inputs.reweight,
            cv_name=str(aligned.cv_names[0]),
            cv_label=str(aligned.cv_labels[0]),
            sampling_label=aligned.sampling_label,
            sampling_slug=aligned.sampling_slug,
            bias_mode=inputs.bias_mode,
            weight_kind=weighted.weight_kind,
            selected_time_ps=aligned.selected_time_ps,
            selected_steps=aligned.selected_steps,
            sampling=aligned.centroid,
            beads=aligned.beads,
            raw_log_weights=weighted.raw_log_weights,
            bias_ev=weighted.bias_ev,
            kbt_ev=weighted.kbt_ev,
            sampling_restart_duplicates=aligned.centroid_restart_duplicates,
            bead_restart_duplicates=aligned.bead_restart_duplicates,
            _plotter=_plotter,
        )
    joint = build_joint_outputs(contract, output, profile, inputs, aligned, weighted, _plotter)
    derived_coordinate_summary = write_derived_coordinate(
        contract, output, profile, inputs, aligned, weighted, joint, _plotter
    )
    write_block_diagnostics(contract, output, profile, inputs, aligned, weighted, joint)
    filtered_colvar = write_frame_tables(contract, output, profile, inputs, aligned, weighted)
    if profile == "core":
        return finalize_core_2d(
            contract=contract,
            output=output,
            source=inputs.source,
            reweight=inputs.reweight,
            cv_names=aligned.cv_names,
            cv_labels=aligned.cv_labels,
            sampling_label=aligned.sampling_label,
            sampling_slug=aligned.sampling_slug,
            bias_mode=inputs.bias_mode,
            weight_kind=weighted.weight_kind,
            selected_time_ps=aligned.selected_time_ps,
            sampling=aligned.centroid,
            beads=aligned.beads,
            raw_log_weights=weighted.raw_log_weights,
            bias_ev=weighted.bias_ev,
            kbt_ev=weighted.kbt_ev,
            primary_name=joint.primary_name,
            primary=joint.primary,
            primary_kcal=joint.primary_kcal,
            primary_supports=joint.primary_supports,
            variants=joint.variants,
            filtered_colvar=filtered_colvar,
            sampling_restart_duplicates=aligned.centroid_restart_duplicates,
            bead_restart_duplicates=aligned.bead_restart_duplicates,
            _plotter=_plotter,
        )
    return finalize_water(
        contract,
        output,
        profile,
        inputs,
        aligned,
        weighted,
        joint,
        derived_coordinate_summary,
        filtered_colvar,
        _plotter,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require
