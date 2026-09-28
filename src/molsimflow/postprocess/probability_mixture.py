"""Frozen probability-ratio mixtures of complete ring-polymer paths.

The last array dimension always denotes beads of one complete path. Softmax
coefficients distribute forces; observables retain uniform bead weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

MODES = {"bead_probability_mixture", "centroid_probability_mixture"}


def _finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _kbt(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("kBT must be finite and positive")
    return value


def log_mean_exp(log_ratios: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return log arithmetic mean and its bead-local derivatives stably."""
    values = _finite(log_ratios, "log ratios")
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("log ratios require a bead dimension")
    maximum = np.max(values, axis=-1, keepdims=True)
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.exp(values - maximum)
    total = np.sum(scaled, axis=-1, keepdims=True)
    result = maximum[..., 0] + np.log(total[..., 0] / values.shape[-1])
    return result, scaled / total


def arithmetic_bias(bead_bias: Any, *, kbt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return V_A and dV_A/dv_b; no extra 1/P follows this derivative."""
    thermal = _kbt(kbt)
    values = _finite(bead_bias, "bead bias")
    with np.errstate(over="ignore"):
        log_mean, alpha = log_mean_exp(-values / thermal)
        energy = -thermal * log_mean
    return _finite(energy, "path bias"), alpha


def centroid_mixture(
    centroid_bias: Any, arithmetic_energy: Any, *, kbt: float,
    coupling: float, log_normalizer: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return V_eta and chi=dV_eta/dV_A with a frozen global log C.

    C=Z_A/Z_c. It is not a coordinate-dependent conditional normalizer.
    At eta=0 the inactive arithmetic field is deliberately not evaluated.
    """
    thermal = _kbt(kbt)
    eta, log_c = float(coupling), float(log_normalizer)
    if not math.isfinite(eta) or not 0 <= eta < 1:
        raise ValueError("coupling must be finite and in [0,1)")
    if not math.isfinite(log_c):
        raise ValueError("log normalizer must be finite")
    vc = _finite(centroid_bias, "centroid bias")
    if eta == 0:
        return vc.copy(), np.zeros_like(vc)
    va = _finite(arithmetic_energy, "arithmetic bias")
    if vc.shape != va.shape:
        raise ValueError("centroid and arithmetic bias shapes differ")
    # Use log component energies to avoid cancellation of Vc in the result.
    with np.errstate(over="ignore", invalid="ignore"):
        lhs = math.log1p(-eta) - vc / thermal
        rhs = math.log(eta) - va / thermal - log_c
        log_density = np.logaddexp(lhs, rhs)
        result = -thermal * log_density
        chi = np.exp(rhs - log_density)
    return _finite(result, "mixed bias"), _finite(chi, "mixture derivative")


def validate_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the portable frozen-field identity and thermodynamic contract."""
    if spec.get("schema") != "probability-mixture-v1" or spec.get("frozen") is not True:
        raise ValueError("a probability-mixture-v1 frozen manifest is required")
    if spec.get("mode") not in MODES:
        raise ValueError("invalid probability mixture mode")
    beads = spec.get("expected_beads")
    if not isinstance(beads, int) or isinstance(beads, bool) or beads < 1:
        raise ValueError("expected_beads must be a positive integer")
    _kbt(spec["kbt"])
    if not isinstance(spec.get("energy_unit"), str) or not spec["energy_unit"].strip():
        raise ValueError("an explicit energy unit is required")
    for name in ("field_sha256",):
        if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get(name, ""))):
            raise ValueError(f"invalid {name}")
    if spec["mode"] == "centroid_probability_mixture":
        centroid_mixture(0.0, 0.0, kbt=spec["kbt"], coupling=spec["coupling"],
                         log_normalizer=spec["log_normalizer"])
        if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("centroid_field_sha256", ""))):
            raise ValueError("invalid centroid_field_sha256")
    return dict(spec)


def load_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Read exactly the frozen manifest admitted by the analysis contract."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("probability mixture manifest hash mismatch")
    return validate_manifest(json.loads(data))


def audit_record(
    spec: dict[str, Any], bead_bias: Any, total_bias: Any, *,
    field_sha256: str, centroid_bias: Any = None, energy_atol: float = 1e-10,
) -> dict[str, Any]:
    """Check whole-frame energies against a declared common frozen field.

    File hashes are checked by the caller's input manifest. This function
    validates the recorded field identity, shape and numerical energy chain.
    """
    spec = validate_manifest(spec)
    if field_sha256 != spec["field_sha256"]:
        raise ValueError("frozen field identity mismatch")
    atol = float(energy_atol)
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("energy_atol must be finite and nonnegative")
    total = _finite(total_bias, "total bias")
    if spec["mode"] == "centroid_probability_mixture" and spec["coupling"] == 0:
        expected = _finite(centroid_bias, "centroid bias")
        if total.ndim != 1 or total.shape != expected.shape:
            raise ValueError("one centroid and total bias is required per frame")
        error = float(np.max(np.abs(expected - total)))
        if error > atol:
            raise ValueError("zero-coupling total bias differs from centroid bias")
        return {"status": "PASS", "mode": spec["mode"], "frames": len(total),
                "expected_beads": spec["expected_beads"], "max_energy_error": error,
                "max_alpha": None, "field_sha256": field_sha256,
                "weight_kind": "fixed_bias", "observable_bead_weight": "uniform",
                "inactive_arithmetic_field": True}
    values = _finite(bead_bias, "bead bias")
    if values.ndim != 2 or values.shape[1] != spec["expected_beads"]:
        raise ValueError("bead bias must contain every bead of each frame")
    if total.shape != (values.shape[0],):
        raise ValueError("one total bias is required per complete path")
    expected, alpha = arithmetic_bias(values, kbt=spec["kbt"])
    if spec["mode"] == "centroid_probability_mixture":
        expected, _ = centroid_mixture(centroid_bias, expected, kbt=spec["kbt"],
                                      coupling=spec["coupling"],
                                      log_normalizer=spec["log_normalizer"])
    if expected.shape != total.shape:
        raise ValueError("centroid bias must contain one value per frame")
    error = float(np.max(np.abs(expected - total)))
    if error > atol:
        raise ValueError("recorded total bias differs from frozen probability mixture")
    return {"status": "PASS", "mode": spec["mode"], "frames": len(total),
            "expected_beads": values.shape[1], "max_energy_error": error,
            "max_alpha": float(np.max(alpha)), "field_sha256": field_sha256,
            "weight_kind": "fixed_bias", "observable_bead_weight": "uniform"}


def export_plumed(
    spec: dict[str, Any], *, field_arg: str, centroid_arg: str | None = None,
    prefix: str = "pm", active_field_bias: bool = False,
) -> str:
    """Compose already defined frozen scalar fields into one path bias.

    A true active_field_bias subtracts the local field's direct bias (e.g.
    frozen OPES .bias). centroid_arg MUST be a non-bias scalar. Input field
    definitions, immutable state loading and update suppression remain the
    caller's responsibility. At eta=0 construct only the centroid graph and
    omit any inactive field definitions from the caller's complete input.
    """
    spec = validate_manifest(spec)
    for label in (field_arg, centroid_arg, prefix):
        if label is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", label):
            raise ValueError("invalid PLUMED scalar label")
    mixed = spec["mode"] == "centroid_probability_mixture"
    if mixed and centroid_arg is None:
        raise ValueError("mixed mode requires a non-bias centroid scalar")
    if mixed and spec["coupling"] == 0:
        if active_field_bias:
            raise ValueError("eta=0 requires omission of the inactive active-bias field")
        return f"{prefix}_bias: BIASVALUE ARG={centroid_arg}\n"
    thermal = format(float(spec["kbt"]), ".17g")
    lines = [f"{prefix}_ell: CUSTOM ARG={field_arg} FUNC=-x/({thermal}) PERIODIC=NO",
             f"{prefix}_logmean: PATH_LOGMEANEXP ARG={prefix}_ell EXPECTED_REPLICAS={spec['expected_beads']}",
             f"{prefix}_va: CUSTOM ARG={prefix}_logmean FUNC=-({thermal})*x PERIODIC=NO"]
    energy = f"{prefix}_va"
    if mixed:
        lines.append(f"{prefix}_total: PROBABILITY_MIX ARG={centroid_arg},{energy} "
                     f"KBT={thermal} COUPLING={spec['coupling']:.17g} "
                     f"LOG_NORMALIZER={spec['log_normalizer']:.17g}")
        energy = f"{prefix}_total"
    if active_field_bias:
        lines.append(f"{prefix}_correction: CUSTOM ARG={energy},{field_arg} FUNC=x-y PERIODIC=NO")
        energy = f"{prefix}_correction"
    lines.append(f"{prefix}_bias: BIASVALUE ARG={energy}")
    return "\n".join(lines) + "\n"
