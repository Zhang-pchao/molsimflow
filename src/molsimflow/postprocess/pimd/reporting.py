"""PIMD table, label, manifest and uncertainty output compatibility."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
import csv
import json
from molsimflow.io.hashes import sha256_file as sha256
from .types import validate_bias_mode, validate_primary_estimator
from .units import KCAL_TO_KJ_MOL
from .uncertainty import quantum_kde_block_jackknife
FES_TABLE_FIELDS = (
    "sampling_support",
    "probability_mean_support",
    "free_energy_mean_support",
    "common_support",
    "estimator_common_support",
    "primary_support",
    "F_sampling_kcal_mol",
    "F_quantum_probability_mean_kcal_mol",
    "F_quantum_free_energy_mean_kcal_mol",
    "F_primary_kcal_mol",
    # Historical public name retained as a compatibility alias.
    "F_bead_free_energy_mean_diagnostic_kcal_mol",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def fes_table_columns(
    surfaces_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    primary_estimator: str,
    index: int | Tuple[int, ...],
) -> Dict[str, float | int]:
    """Return schema-3 FES columns plus the historical diagnostic alias."""

    primary = validate_primary_estimator(primary_estimator)
    return {
        "sampling_support": int(supports["centroid"][index]),
        "probability_mean_support": int(supports["probability_mean"][index]),
        "free_energy_mean_support": int(supports["free_energy_mean"][index]),
        "common_support": int(supports["estimator_common"][index]),
        "estimator_common_support": int(supports["estimator_common"][index]),
        "primary_support": int(supports[primary][index]),
        "F_sampling_kcal_mol": float(surfaces_kcal["centroid"][index]),
        "F_quantum_probability_mean_kcal_mol": float(
            surfaces_kcal["probability_mean"][index]
        ),
        "F_quantum_free_energy_mean_kcal_mol": float(
            surfaces_kcal["free_energy_mean"][index]
        ),
        "F_primary_kcal_mol": float(surfaces_kcal[primary][index]),
        "F_bead_free_energy_mean_diagnostic_kcal_mol": float(
            surfaces_kcal["free_energy_mean"][index]
        ),
    }



def estimator_plot_labels(bias_mode: str) -> Dict[str, str]:
    """Name both documented bead estimators without assigning policy priority."""
    validate_bias_mode(bias_mode)
    return {
        "probability_mean": "Probability-mean bead FES",
        "free_energy_mean": "Free-energy-mean bead FES",
    }



def default_cv_label(name: str) -> str:
    """Return publication-readable labels for the maintained water-ionization CVs."""
    return {
        "ionization": r"Ionization state, $s_a$ (dimensionless)",
        "iondistance": r"Ion separation, $s_t$ ($\AA$)",
        "logdistance": r"Log ion-separation coordinate, $s'_t$ (dimensionless)",
    }.get(str(name), str(name))



def sampling_protocol_label(reweight: Mapping[str, object]) -> str:
    """Return an explicit plot label for the declared weighting protocol."""
    if "protocol_label" in reweight:
        label = str(reweight["protocol_label"])
        require(bool(label), "protocol_label must not be empty")
        return label
    return {
        "fixed_bias": "fixed bias",
        "quasi_static_opes": "quasi-static OPES",
        "precomputed": "precomputed weights",
    }[str(reweight.get("weight_kind", "quasi_static_opes"))]



def portable_artifact_path(path: Path | str, output: Path) -> str:
    """Represent paths inside a movable output tree without staging prefixes."""
    value = Path(path)
    if not value.is_absolute():
        return str(path)
    try:
        relative = value.relative_to(output)
    except ValueError:
        return str(path)
    return "{output}/" + relative.as_posix()



def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write CSV while adding kJ/mol peers for legacy kcal/mol energy columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expanded = []
    kj_columns = {}
    for name in fieldnames:
        expanded.append(name)
        if name.endswith("_kcal_mol"):
            target = name.replace("_kcal_mol", "_kJ_mol")
            expanded.append(target)
            kj_columns[name] = target
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expanded)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for legacy, target in kj_columns.items():
                row[target] = float(row[legacy]) * KCAL_TO_KJ_MOL
            writer.writerow(row)



def write_filtered_colvar(
    path: Path, fields: Sequence[str], data: np.ndarray, selected: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for row in data[selected]:
            handle.write(" ".join(f"{float(value):.16e}" for value in row) + "\n")



def write_kde_uncertainty(
    output: Path,
    reweight: Mapping[str, object],
    cv_names: Sequence[str],
    beads: np.ndarray,
    raw_log_weights: np.ndarray,
    kbt: float,
) -> None:
    """Write optional whole-frame sampling errors without changing block diagnostics."""
    if "uncertainty" not in reweight:
        return
    config = reweight["uncertainty"]
    require(
        isinstance(config, dict) and set(config) == {"block_frames", "reference_grid_index"},
        "invalid uncertainty configuration",
    )
    grids = [np.linspace(*reweight["grid"][name]) for name in cv_names]
    bandwidth = reweight["bandwidth_variants"][reweight["primary_bandwidth"]]
    result = quantum_kde_block_jackknife(
        beads,
        raw_log_weights,
        grids,
        bandwidth,
        kbt=kbt,
        block_frames=config["block_frames"],
        reference_grid_index=config["reference_grid_index"],
        relative_density_support=float(reweight["relative_density_support"]),
        primary_estimator=str(reweight.get("primary_estimator", "probability_mean")),
    )
    points = np.column_stack([axis.ravel() for axis in np.meshgrid(*grids)])
    fields = list(cv_names) + ["delta_F_eV", "standard_error_eV", "support"]
    rows = []
    for n, point in enumerate(points):
        row = dict(zip(cv_names, point))
        row.update(
            delta_F_eV=result["free_energy_difference"].ravel()[n],
            standard_error_eV=result["standard_error"].ravel()[n],
            support=int(result["support"].ravel()[n]),
        )
        rows.append(row)
    write_csv(output / "blocks" / "quantum-fes-uncertainty.csv", rows, fields)
    metadata = {
        "estimator": str(reweight.get("primary_estimator", "probability_mean")),
        "method": "delete_one_frame_block_jackknife",
        "blocks": result["blocks"],
        "block_frames": result["block_frames"],
        "reference_grid_index": config["reference_grid_index"],
        "reference_coordinates": [
            float(axis[i]) for axis, i in zip(grids, config["reference_grid_index"])
        ],
        "bandwidth": list(bandwidth),
        "relative_density_support": float(reweight["relative_density_support"]),
        "units": "eV",
        "scope": "Fixed-bandwidth sampling standard error; not smoothing bias or a confidence interval.",
    }
    (output / "blocks" / "quantum-fes-uncertainty.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )



def write_manifest(output: Path) -> None:
    records = []
    for path in sorted((item for item in output.rglob("*") if item.is_file())):
        if path.name == "OUTPUT-SHA256SUMS":
            continue
        records.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "provenance" / "OUTPUT-SHA256SUMS").write_text("\n".join(records) + "\n", encoding="utf-8")
