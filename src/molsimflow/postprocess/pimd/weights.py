"""One reweighting weight per complete ring-polymer path."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .io import field
from .types import (
    WEIGHT_KINDS,
    AlignedPaths,
    PreparedInputs,
    WeightedPaths,
    WeightProvider,
    validate_bias_mode,
)
from .units import KB_EV_PER_K


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


require = _require


def normalized_log_weights(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize finite log weights without exponentiating their absolute scale."""

    raw = np.asarray(values, dtype=float)
    _require(raw.ndim == 1 and raw.size > 0, "log weights must be a nonempty vector")
    _require(np.isfinite(raw).all(), "log weights must be finite")
    maximum = float(np.max(raw))
    shifted = raw - maximum
    return shifted - math.log(float(np.sum(np.exp(shifted))))


def total_bias_energy(
    bias_mode: str,
    *,
    sampling_bias_energy: Sequence[float] | np.ndarray | None = None,
    bead_bias_energies: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    """Return one complete-path bias energy per ring-polymer frame."""

    mode = validate_bias_mode(bias_mode)
    if mode == "bead_density_shared":
        _require(sampling_bias_energy is None, "bead-density bias uses bead-local energies")
        _require(bead_bias_energies is not None, "bead-density bias energies are required")
        values = np.asarray(bead_bias_energies, dtype=float)
        _require(values.ndim == 2 and min(values.shape) > 0, "invalid bead-density bias table")
        _require(np.isfinite(values).all(), "bead-density bias energies must be finite")
        return np.mean(values, axis=1)

    _require(bead_bias_energies is None, "this bias mode uses one complete-path sampling energy")
    _require(sampling_bias_energy is not None, "sampling bias energy is required")
    values = np.asarray(sampling_bias_energy, dtype=float)
    _require(values.ndim == 1 and values.size > 0, "sampling bias energy must be a vector")
    _require(np.isfinite(values).all(), "sampling bias energy must be finite")
    return values


def frame_log_weights(
    weight_kind: str,
    *,
    bias_energy: Sequence[float] | np.ndarray | None = None,
    kbt: float | None = None,
    precomputed: Sequence[float] | np.ndarray | None = None,
    quasi_static: bool | None = None,
) -> np.ndarray:
    """Build one log weight per complete ring-polymer frame.

    ``fixed_bias`` and ``quasi_static_opes`` both use ``+U_bias / kBT``.  The
    distinct names keep their sampling assumptions explicit.  In particular,
    this function never subtracts an OPES ``rct`` column and does not infer that
    an adaptive trajectory has reached the quasi-static regime.
    """

    kind = str(weight_kind)
    _require(kind in WEIGHT_KINDS, f"unsupported PIMD weight kind: {kind}")
    if kind == "precomputed":
        _require(precomputed is not None, "precomputed log weights are required")
        _require(bias_energy is None, "precomputed weights cannot also use bias energy")
        raw = np.asarray(precomputed, dtype=float)
    else:
        _require(bias_energy is not None, f"bias energy is required for {kind}")
        _require(precomputed is None, "bias-energy weights cannot also be precomputed")
        _require(kbt is not None and np.isfinite(kbt) and kbt > 0.0, "kBT must be positive")
        if kind == "quasi_static_opes":
            _require(quasi_static is True, "quasi-static OPES must be declared explicitly")
        raw = np.asarray(bias_energy, dtype=float) / float(kbt)
    _require(raw.ndim == 1 and raw.size > 0, "frame weights must be a nonempty vector")
    _require(np.isfinite(raw).all(), "frame weights must be finite")
    return raw


def cumulative_weight_diagnostics(
    weights: Sequence[float] | np.ndarray,
    *,
    log_weights: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return prefix Kish ESS fraction and maximum share without overflow.

    Supply log weights before exponentiation to preserve prefixes whose linear
    weights would underflow to zero after full-trajectory normalization.
    """
    values = np.asarray(weights, dtype=float)
    require(values.ndim == 1 and values.size > 0, "empty weights")
    require(np.isfinite(values).all(), "invalid weights")
    if not log_weights:
        require(np.all(values > 0.0), "invalid weights")
        values = np.log(values)
    values = values - np.max(values)
    total = np.logaddexp.accumulate(values)
    squared_total = np.logaddexp.accumulate(2.0 * values)
    count = np.arange(1, values.size + 1, dtype=float)
    ess_fraction = np.exp(2.0 * total - squared_total) / count
    maximum_share = np.exp(np.maximum.accumulate(values) - total)
    return ess_fraction, maximum_share


@dataclass(frozen=True)
class PrecomputedWeights:
    """A supplied complete-frame log-weight vector in aligned frame order."""

    values: np.ndarray

    def raw_log_weights(self) -> np.ndarray:
        return frame_log_weights("precomputed", precomputed=self.values)


@dataclass(frozen=True)
class BiasEnergyWeights:
    """Declared fixed or quasi-static complete-path energy weights."""

    kind: str
    energy: np.ndarray
    kbt: float
    quasi_static: bool

    def raw_log_weights(self) -> np.ndarray:
        return frame_log_weights(
            self.kind, bias_energy=self.energy, kbt=self.kbt, quasi_static=self.quasi_static
        )


def prepare_weights(
    contract: Mapping[str, object],
    output: Path,
    profile: str,
    inputs: PreparedInputs,
    aligned: AlignedPaths,
) -> WeightedPaths:

    reweight = inputs.reweight
    bias_mode = inputs.bias_mode
    conditional_spec = inputs.conditional_spec
    probability_spec = inputs.probability_spec
    conditional_model = inputs.conditional_model
    probability_model = inputs.probability_model
    fields = inputs.fields
    centroid_data = aligned.centroid_data
    selected = aligned.selected
    cv_names = aligned.cv_names
    centroid = aligned.centroid
    beads = aligned.beads
    selected_bead_field = aligned.selected_bead_field
    weight_kind = str(reweight.get("weight_kind", "quasi_static_opes"))
    bias_column_value = reweight.get("bias_column")
    bias_column = None if bias_column_value is None else str(bias_column_value)
    bias_ev = None
    shared_diagnostic_max_deltas: Dict[str, float] = {}
    if bias_column is not None:
        if bias_mode == "bead_density_shared":
            bias_ev = total_bias_energy(
                bias_mode,
                bead_bias_energies=selected_bead_field(bias_column),
            )
        else:
            bias_ev = total_bias_energy(
                bias_mode,
                sampling_bias_energy=field(centroid_data, fields, bias_column)[selected],
            )
    opes_bias_ev = bias_ev
    extra_bias_columns = reweight.get("extra_bias_columns", [])
    require(isinstance(extra_bias_columns, list), "extra_bias_columns must be a list")
    require(
        all(isinstance(name, str) and name for name in extra_bias_columns),
        "invalid extra bias column",
    )
    require(len(set(extra_bias_columns)) == len(extra_bias_columns), "duplicate extra bias column")
    require(bias_column not in extra_bias_columns, "primary bias repeated in extra_bias_columns")
    require(not extra_bias_columns or bias_ev is not None, "extra biases require bias_column")
    for name in extra_bias_columns:
        if bias_mode == "bead_density_shared":
            extra = total_bias_energy(bias_mode, bead_bias_energies=selected_bead_field(name))
        else:
            extra = total_bias_energy(
                bias_mode, sampling_bias_energy=field(centroid_data, fields, name)[selected]
            )
        bias_ev = bias_ev + extra
    require(
        profile == "core" or bias_ev is not None,
        "water-ionization diagnostics require bias_column",
    )

    rct_ev = zed = neff = nker = None
    if profile == "water_ionization_opes":
        diagnostic_columns = {
            "rct": str(reweight["rct_column"]),
            "zed": str(reweight.get("zed_column", "opes.zed")),
            "neff": str(reweight.get("neff_column", "opes.neff")),
            "nker": str(reweight.get("nker_column", "opes.nker")),
        }
        if bias_mode == "bead_density_shared":
            diagnostics = {}
            tolerance = float(reweight.get("shared_diagnostic_tolerance", 1e-9))
            require(tolerance >= 0.0, "shared diagnostic tolerance must be nonnegative")
            for name, column in diagnostic_columns.items():
                values = selected_bead_field(column)
                delta = float(np.max(np.max(values, axis=1) - np.min(values, axis=1)))
                require(delta <= tolerance, f"shared OPES diagnostic mismatch: {column}")
                shared_diagnostic_max_deltas[column] = delta
                diagnostics[name] = np.mean(values, axis=1)
            rct_ev, zed, neff, nker = (
                diagnostics["rct"],
                diagnostics["zed"],
                diagnostics["neff"],
                diagnostics["nker"],
            )
        else:
            rct_ev = field(centroid_data, fields, diagnostic_columns["rct"])[selected]
            zed = field(centroid_data, fields, diagnostic_columns["zed"])[selected]
            neff = field(centroid_data, fields, diagnostic_columns["neff"])[selected]
            nker = field(centroid_data, fields, diagnostic_columns["nker"])[selected]
    kbt_ev = float(reweight["kbt_eV"])
    temperature = float(reweight["temperature_K"])
    require(
        np.isfinite(temperature) and temperature > 0.0, "temperature_K must be finite and positive"
    )
    require(np.isfinite(kbt_ev) and kbt_ev > 0.0, "kbt_eV must be finite and positive")
    require(
        reweight.get("energy_unit", "eV") == "eV",
        "energy_unit must be eV; convert energy columns before analysis",
    )
    expected_kbt = KB_EV_PER_K * temperature
    require(abs(kbt_ev - expected_kbt) <= 1e-12, "kBT/temperature mismatch")
    if conditional_spec is not None:
        from molsimflow.postprocess.conditional_path import audit_record

        audit = audit_record(
            conditional_model,
            centroid[:, 0],
            selected_bead_field(conditional_spec["bead_region_column"]),
            field(centroid_data, fields, conditional_spec["log_normalizer_column"])[selected],
            field(centroid_data, fields, conditional_spec["centroid_bias_column"])[selected],
            bias_ev,
            coupling=conditional_spec["coupling"],
            kbt=kbt_ev,
            energy_atol=conditional_spec["energy_atol_eV"],
            normalizer_atol=conditional_spec["log_normalizer_atol"],
        )
        audit["model_sha256"] = conditional_spec["model_sha256"]
        (output / "qc" / "conditional-path.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if probability_spec is not None:
        from molsimflow.postprocess.probability_mixture import audit_record

        require(abs(probability_model["kbt"] - kbt_ev) <= 1e-12, "probability mixture kBT mismatch")
        centroid_bias = None
        if probability_model["mode"] == "centroid_probability_mixture":
            centroid_bias = field(centroid_data, fields, probability_spec["centroid_bias_column"])[
                selected
            ]
        audit = audit_record(
            probability_model,
            None
            if probability_model["mode"] == "centroid_probability_mixture"
            and probability_model["coupling"] == 0
            else selected_bead_field(probability_spec["bead_bias_column"]),
            bias_ev,
            field_sha256=probability_model["field_sha256"],
            centroid_bias=centroid_bias,
            energy_atol=probability_spec["energy_atol_eV"],
        )
        audit["manifest_sha256"] = probability_spec["manifest_sha256"]
        (output / "qc" / "probability-mixture.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    quasi_static_declared = reweight.get("quasi_static") is True
    if weight_kind == "precomputed":
        require(not extra_bias_columns, "precomputed weights cannot also use extra bias columns")
        log_weight_column = str(reweight["log_weight_column"])
        provider: WeightProvider = PrecomputedWeights(
            field(centroid_data, fields, log_weight_column)[selected]
        )
    else:
        require(bias_ev is not None, f"{weight_kind} requires bias_column")
        provider = BiasEnergyWeights(weight_kind, bias_ev, kbt_ev, quasi_static_declared)
    raw_log_weights = provider.raw_log_weights()
    log_weights = normalized_log_weights(raw_log_weights)
    weights = np.exp(log_weights)
    require(abs(float(np.sum(weights)) - 1.0) <= 1e-12, "weight normalization failed")

    return WeightedPaths(
        weight_kind=weight_kind,
        bias_column=bias_column,
        bias_ev=bias_ev,
        opes_bias_ev=opes_bias_ev,
        extra_bias_columns=extra_bias_columns,
        rct_ev=rct_ev,
        zed=zed,
        neff=neff,
        nker=nker,
        shared_diagnostic_max_deltas=shared_diagnostic_max_deltas,
        kbt_ev=kbt_ev,
        raw_log_weights=raw_log_weights,
        weights=weights,
        quasi_static_declared=quasi_static_declared,
    )
