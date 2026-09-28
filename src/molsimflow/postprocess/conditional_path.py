"""Frozen one-dimensional conditional path normalizers and exact bias algebra.

A fitted normalizer defines a potential, not a certified conditional reference.
Inputs and weights describe whole paths. No independent-bead assumption is made.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial

from molsimflow.postprocess.quantum_path_stats import conditional_region_statistics

SCHEMA = "conditional-path-log-polynomial-v1"


def _array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real numeric data")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def log_mixture(log_score: object, log_normalizer: object, coupling: float) -> tuple:
    """Return log R and chi; derivatives in log-score/log-normalizer are +/-chi."""
    if isinstance(coupling, bool) or not np.isfinite(coupling) or not 0 <= coupling < 1:
        raise ValueError("coupling must be finite and in [0,1)")
    loga = _array(log_score, "log_score")
    logm = _array(log_normalizer, "log_normalizer")
    if loga.shape != logm.shape:
        raise ValueError("score and normalizer must have the same shape")
    if coupling == 0:
        return np.zeros_like(loga), np.zeros_like(loga)
    with np.errstate(over="raise", invalid="raise"):
        try:
            tilted = np.log(coupling) + loga - logm
            ratio = np.logaddexp(np.log1p(-coupling), tilted)
            return ratio, np.exp(tilted - ratio)
        except FloatingPointError as exc:
            raise ValueError("conditional log ratio exceeds floating-point range") from exc


def validate_model(model: dict) -> None:
    """Validate an explicit frozen log-polynomial representation, without fitting."""
    if not isinstance(model, dict) or set(model) != {
        "schema",
        "domain",
        "coefficients",
        "epsilon",
        "fit_diagnostics",
        "provenance",
    }:
        raise ValueError("invalid conditional model fields")
    if model["schema"] != SCHEMA:
        raise ValueError("unsupported conditional model schema")
    domain = _array(model["domain"], "domain")
    coeff = _array(model["coefficients"], "coefficients")
    if domain.shape != (2,) or not domain[0] < domain[1]:
        raise ValueError("domain must have two increasing finite endpoints")
    if coeff.ndim != 1 or not 1 <= len(coeff) <= 6:
        raise ValueError("one to six polynomial coefficients are required")
    epsilon = model["epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a positive finite scalar")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite scalar")
    if not isinstance(model["fit_diagnostics"], dict) or not isinstance(model["provenance"], dict):
        raise TypeError("diagnostics and provenance must be objects")
    width = domain[1] - domain[0]
    if not np.isfinite(width) or width / 2 == 0:
        raise ValueError("domain width is outside the supported floating-point range")


def evaluate_log_normalizer(model: dict, conditioning: object) -> tuple:
    """Return log(m) and its derivative on the model's closed finite domain."""
    validate_model(model)
    c = _array(conditioning, "conditioning")
    lo, hi = model["domain"]
    if np.any((c < lo) | (c > hi)):
        raise ValueError("conditioning coordinate is outside the frozen model domain")
    scale = (hi - lo) / 2
    t = (c - lo) / scale - 1
    coeff = np.asarray(model["coefficients"], dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            value = polynomial.polyval(t, coeff)
            derivative = polynomial.polyval(t, polynomial.polyder(coeff)) / scale
        except FloatingPointError as exc:
            raise ValueError("normalizer polynomial exceeds floating-point range") from exc
    return value, derivative


def fit_log_normalizer(
    conditioning: object,
    region_fraction: object,
    log_weights: object,
    *,
    bin_edges: object,
    block_ids: object,
    epsilon: float,
    degree: int,
    min_frame_ess: float,
    stationary_pilot: bool,
    provenance: dict,
) -> dict:
    """Fit log conditional bin means with a low-degree scaled polynomial.

    Explicit log weights must target the desired conditional law. Uniform
    weights may be used for a stationary centroid-only biased pilot; finite
    conditioning bins still introduce approximation error. Polynomial fitting
    uses cell Kish ESS as least-squares weights, not an independence count.
    Reuse whole-block diagnostic estimates; no held-out validation is implied.
    """
    if stationary_pilot is not True:
        raise ValueError("a stationary pilot must be explicitly declared")
    if type(degree) is not int or not 0 <= degree <= 5:
        raise ValueError("degree must be an integer between zero and five")
    if isinstance(min_frame_ess, bool) or not np.isfinite(min_frame_ess) or min_frame_ess < 1:
        raise ValueError("min_frame_ess must be finite and at least one")
    c = _array(conditioning, "conditioning")
    fraction = _array(region_fraction, "region_fraction")
    edges = _array(bin_edges, "bin_edges")
    if c.ndim != 1 or c.size == 0 or fraction.shape != c.shape:
        raise ValueError("conditioning and region_fraction must have shape (N,)")
    if edges.ndim != 1 or len(edges) < degree + 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("too few or nonincreasing conditioning bin edges")
    model = {
        "schema": SCHEMA,
        "domain": [float(edges[0]), float(edges[-1])],
        "coefficients": [0.0],
        "epsilon": epsilon,
        "fit_diagnostics": {},
        "provenance": provenance,
    }
    validate_model(model)
    stats = conditional_region_statistics(
        c[:, None],
        epsilon + fraction,
        fraction[:, None],
        log_weights,
        bin_edges=[edges],
        block_ids=block_ids,
    )
    if stats["out_of_range_frames"]:
        raise ValueError("pilot frames outside the fitted domain are not silently discarded")
    for cell in stats["cells"]:
        if cell["jackknife_status"] != "SUPPORTED" or cell["frame_ess"] < min_frame_ess:
            raise ValueError("every fitted cell must retain whole-block support and minimum ESS")
    centers = (edges[:-1] + edges[1:]) / 2
    lo, hi = model["domain"]
    scaled = 2 * (centers - lo) / (hi - lo) - 1
    means = np.array([v["moments"]["path_mean"] for v in stats["cells"]])
    ess = np.array([v["frame_ess"] for v in stats["cells"]])
    coeff = polynomial.polyfit(scaled, np.log(means), degree, w=np.sqrt(ess))
    model["coefficients"] = coeff.tolist()
    model["fit_diagnostics"] = {
        "method": "ESS-weighted log conditional bin means",
        "cell_centers": centers.tolist(),
        "cell_mean_scores": means.tolist(),
        "log_fit_residuals": (polynomial.polyval(scaled, coeff) - np.log(means)).tolist(),
        "minimum_frame_ess": float(min_frame_ess),
        "conditional_statistics": stats,
        "held_out_validation": "NOT_ASSESSED",
        "scientific_acceptance": "NOT_ASSESSED",
    }
    validate_model(model)
    return model


def load_model(path: Path, *, expected_sha256: str) -> dict:
    """Bind offline/export/restart preparation to an explicit immutable digest."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA256 digest")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("frozen normalizer hash mismatch")
    model = json.loads(raw)
    validate_model(model)
    return model


def save_model(model: dict, path: Path) -> str:
    """Write a new frozen artifact, refusing to overwrite an existing model."""
    validate_model(model)
    raw = (json.dumps(model, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def plumed_correction(
    model: dict,
    *,
    centroid_label: str,
    score_label: str,
    prefix: str,
    coupling: float,
    kbt: float,
) -> str:
    """Return a graph fragment; caller supplies c, positive a and fixed B_c.

    The fragment includes only the correction, not the baseline centroid bias.
    It uses existing CUSTOM autodifferentiation and the optional pathbias action.
    Labels are validated to avoid treating arbitrary text as PLUMED input.
    """
    validate_model(model)
    for label in (centroid_label, score_label, prefix):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", label) is None:
            raise ValueError("labels must be simple PLUMED identifiers")
    if "." in prefix:
        raise ValueError("prefix must be a simple action label")
    generated = {f"{prefix}_{name}" for name in ("loga", "logm", "ratio", "energy", "bias")}
    if centroid_label == score_label or any(
        label.split(".")[0] in generated for label in (centroid_label, score_label)
    ):
        raise ValueError("input labels must be distinct from each other and generated actions")
    log_mixture(0.0, 0.0, coupling)
    if isinstance(kbt, bool) or not np.isfinite(kbt) or kbt <= 0:
        raise ValueError("kbt must be finite and positive")
    lo, hi = model["domain"]
    scaled = f"((x-({lo:.17g}))/({(hi - lo) / 2:.17g})-1)"
    expression = f"({model['coefficients'][-1]:.17g})"
    for coefficient in reversed(model["coefficients"][:-1]):
        expression = f"(({coefficient:.17g})+{scaled}*{expression})"
    return (
        f"{prefix}_loga: CUSTOM ARG={score_label} FUNC=log(x) PERIODIC=NO\n"
        f"{prefix}_logm: CUSTOM ARG={centroid_label} FUNC={expression} PERIODIC=NO\n"
        f"{prefix}_ratio: CONDITIONAL_PATH ARG={centroid_label},{prefix}_loga,{prefix}_logm "
        f"COUPLING={coupling:.17g} LOWER={lo:.17g} UPPER={hi:.17g}\n"
        f"{prefix}_energy: CUSTOM ARG={prefix}_ratio FUNC=-({kbt:.17g})*x PERIODIC=NO\n"
        f"{prefix}_bias: BIASVALUE ARG={prefix}_energy\n"
    )


def audit_total_bias(
    total_bias: object,
    centroid_bias: object,
    log_score: object,
    log_normalizer: object,
    *,
    coupling: float,
    kbt: float,
    atol: float,
) -> float:
    """Check one recorded complete-path energy per frame against its components."""
    total = _array(total_bias, "total_bias")
    baseline = _array(centroid_bias, "centroid_bias")
    ratio, _ = log_mixture(log_score, log_normalizer, coupling)
    if (
        total.ndim != 1
        or total.size == 0
        or total.shape != baseline.shape
        or total.shape != ratio.shape
    ):
        raise ValueError("all complete-path fields must have shape (N,)")
    if not np.isfinite(kbt) or kbt <= 0 or not np.isfinite(atol) or atol < 0:
        raise ValueError("kbt must be positive and atol nonnegative, both finite")
    residual = np.max(np.abs(total - (baseline - kbt * ratio)))
    if not np.isfinite(residual) or residual > atol:
        raise ValueError("recorded total bias does not match the conditional path potential")
    return float(residual)


def audit_record(
    model: dict,
    conditioning: object,
    bead_region_scores: object,
    recorded_log_normalizer: object,
    centroid_bias: object,
    total_bias: object,
    *,
    coupling: float,
    kbt: float,
    energy_atol: float,
    normalizer_atol: float,
) -> dict:
    """Reconstruct the correction from a frozen model and synchronized bead fields."""
    c = _array(conditioning, "conditioning")
    beads = _array(bead_region_scores, "bead_region_scores")
    recorded = _array(recorded_log_normalizer, "recorded_log_normalizer")
    if c.ndim != 1 or beads.ndim != 2 or beads.shape[0] != len(c) or beads.shape[1] == 0:
        raise ValueError("require one conditioning value and a complete bead row per frame")
    if recorded.shape != c.shape or np.any((beads < 0) | (beads > 1)):
        raise ValueError("normalizer shape or bead region scores are invalid")
    logm, _ = evaluate_log_normalizer(model, c)
    if not np.isfinite(normalizer_atol) or normalizer_atol < 0:
        raise ValueError("normalizer_atol must be finite and nonnegative")
    residual = float(np.max(np.abs(recorded - logm)))
    if residual > normalizer_atol:
        raise ValueError("recorded normalizer differs from the frozen model")
    loga = np.log(model["epsilon"] + beads.mean(axis=1))
    energy_residual = audit_total_bias(
        total_bias, centroid_bias, loga, logm, coupling=coupling, kbt=kbt, atol=energy_atol
    )
    return {
        "frame_count": len(c),
        "bead_count": beads.shape[1],
        "maximum_log_normalizer_error": residual,
        "maximum_total_bias_error": energy_residual,
        "weight_unit": "complete_path_frame",
        "scientific_acceptance": "NOT_ASSESSED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fit = actions.add_parser(
        "fit", help="fit explicitly declared stationary whole-frame pilot data"
    )
    fit.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ: conditioning, region_fraction, log_weights, block_ids, bin_edges",
    )
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epsilon", type=float, required=True)
    fit.add_argument("--degree", type=int, default=2)
    fit.add_argument("--min-frame-ess", type=float, required=True)
    fit.add_argument("--stationary-pilot", action="store_true")
    export = actions.add_parser(
        "export", help="verify a frozen model and emit its PLUMED correction"
    )
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--model-sha256", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--centroid-label", required=True)
    export.add_argument("--score-label", required=True)
    export.add_argument("--prefix", default="conditional")
    export.add_argument("--coupling", type=float, required=True)
    export.add_argument("--kbt", type=float, required=True)
    args = parser.parse_args(argv)
    if args.action == "fit":
        raw = args.input.read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            model = fit_log_normalizer(
                data["conditioning"],
                data["region_fraction"],
                data["log_weights"],
                bin_edges=data["bin_edges"],
                block_ids=data["block_ids"],
                epsilon=args.epsilon,
                degree=args.degree,
                min_frame_ess=args.min_frame_ess,
                stationary_pilot=args.stationary_pilot,
                provenance={"pilot_sha256": hashlib.sha256(raw).hexdigest()},
            )
        print(save_model(model, args.output))
    else:
        model = load_model(args.model, expected_sha256=args.model_sha256)
        graph = plumed_correction(
            model,
            centroid_label=args.centroid_label,
            score_label=args.score_label,
            prefix=args.prefix,
            coupling=args.coupling,
            kbt=args.kbt,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(f"# frozen_normalizer_sha256={args.model_sha256}\n" + graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
