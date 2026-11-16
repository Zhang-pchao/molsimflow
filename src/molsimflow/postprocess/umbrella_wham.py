"""Binless WHAM for fixed umbrella potentials, with explicit sample counts.

Energy inputs must share one unit; harmonic coordinates and force constants must
be consistent. The result is an equilibrium estimator, not an equilibrium test.
No radial Jacobian, smoothing, missing-support filling, or iid error bars are
applied. Numerics are equivalent to MBAR when only window biases differ.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from molsimflow.postprocess.fes_analysis import read_plumed_numeric_table
from molsimflow.io.lammps_table import read_fix_ave_time_table


@dataclass(frozen=True)
class WhamResult:
    """Dimensionless offsets and normalized unbiased frame weights."""

    free_energies: np.ndarray
    log_weights: np.ndarray
    overlap_matrix: np.ndarray
    counts: np.ndarray
    iterations: int
    score_residual: float
    converged: bool


@dataclass(frozen=True)
class HistogramPmf:
    centers: np.ndarray
    free_energy: np.ndarray
    density: np.ndarray
    raw_counts: np.ndarray
    weight_ess_iid: np.ndarray


def _lse(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    return result.squeeze() if axis is None else np.squeeze(result, axis=axis)


def harmonic_bias_matrix(
    coordinates: np.ndarray, centers: Sequence[float], kappas: Sequence[float]
) -> np.ndarray:
    """Return U_kn = kappa_k * (q_n - center_k)**2 / 2."""

    q = np.asarray(coordinates, dtype=float)
    c = np.asarray(centers, dtype=float)
    k = np.asarray(kappas, dtype=float)
    if q.ndim != 1 or c.ndim != 1 or k.shape != c.shape or not len(c):
        raise ValueError("Expected 1D coordinates and equal nonempty centers/kappas")
    if not np.isfinite(q).all() or not np.isfinite(c).all() or not np.isfinite(k).all():
        raise ValueError("Nonfinite coordinates or umbrella parameters")
    if np.any(k <= 0):
        raise ValueError("Umbrella spring constants must be positive")
    return 0.5 * k[:, None] * (q[None, :] - c[:, None]) ** 2


def solve_wham(
    reduced_biases: np.ndarray,
    sample_counts: Sequence[int],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> WhamResult:
    """Solve the convex unbinned WHAM likelihood, fixing f[0] = 0.

    ``reduced_biases`` has shape (windows, pooled samples), with U/kBT.
    Counts describe the actual retained sample number from each window; all
    counts must be positive and sum to the pooled sample count. Samples may be
    correlated; this solver does not turn them into independent observations.
    A normalized likelihood-score residual controls numerical convergence.
    """

    u = np.asarray(reduced_biases, dtype=float)
    raw = np.asarray(sample_counts, dtype=float)
    if u.ndim != 2 or raw.shape != (u.shape[0],) or not u.shape[1] or not len(raw):
        raise ValueError("Bias matrix/count dimensions do not match")
    if not np.isfinite(u).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite biases or sample counts")
    if np.any(raw <= 0) or np.any(raw != np.floor(raw)) or raw.sum() != u.shape[1]:
        raise ValueError("Positive integer counts must sum to pooled sample count")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid convergence settings")
    counts = raw.astype(int)
    total = u.shape[1]
    logn = np.log(raw)
    f = np.zeros(len(counts))

    def evaluate(offsets: np.ndarray):
        logits = logn[:, None] + offsets[:, None] - u
        logd = _lse(logits, axis=0)
        responsibilities = np.exp(logits - logd)
        objective = float((logd.sum() - np.dot(raw, offsets)) / total)
        gradient = (responsibilities.sum(axis=1) - raw) / total
        return objective, gradient, responsibilities, logd

    # Stable self-consistent warmup avoids an extreme initial Newton step when
    # adjacent umbrella offsets differ substantially.
    for _ in range(30):
        _, _, _, logd = evaluate(f)
        new = -_lse(-u - logd[None, :], axis=1)
        f = new - new[0]
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        objective, gradient, resp, logd = evaluate(f)
        residual = float(np.max(np.abs(gradient)))
        if residual <= tolerance:
            converged = True
            break
        hessian = (np.diag(resp.sum(axis=1)) - resp @ resp.T) / total
        direction = np.zeros(len(f))
        direction[1:] = -np.linalg.lstsq(
            hessian[1:, 1:], gradient[1:], rcond=1e-12
        )[0]
        scale = float(np.max(np.abs(direction)))
        if scale > 10:
            direction *= 10 / scale
        descent = float(np.dot(gradient, direction))
        if not np.isfinite(direction).all() or descent >= 0:
            direction = -gradient
            direction[0] = 0
            descent = float(np.dot(gradient, direction))
        accepted = False
        for power in range(40):
            alpha = 0.5**power
            candidate = f + alpha * direction
            value, score, _, _ = evaluate(candidate)
            improved = value <= objective + 1e-4 * alpha * descent
            near_minimum = (
                abs(value - objective) < 1e-13
                and np.max(np.abs(score)) < residual
            )
            if improved or near_minimum:
                f = candidate
                accepted = True
                break
        if not accepted:
            break
    _, gradient, _, logd = evaluate(f)
    residual = float(np.max(np.abs(gradient)))
    converged = converged or residual <= tolerance
    log_weights = -logd
    log_weights -= float(_lse(log_weights))
    biased_weights = np.exp(f[:, None] - u - logd[None, :])
    overlap = (biased_weights @ biased_weights.T) * raw[None, :]
    return WhamResult(f, log_weights, overlap, counts, iteration, residual, converged)


def fit_harmonic_umbrellas(
    samples: Sequence[np.ndarray],
    centers: Sequence[float],
    kappas: Sequence[float],
    kbt: float,
    **solver_options,
) -> WhamResult:
    """Fit same-temperature windows whose only differing bias is harmonic."""

    arrays = [np.asarray(q, dtype=float) for q in samples]
    if len(arrays) != len(centers) or any(q.ndim != 1 or not len(q) for q in arrays):
        raise ValueError("Each umbrella requires a nonempty 1D sample array")
    if not np.isfinite(kbt) or kbt <= 0:
        raise ValueError("kBT must be positive")
    u = harmonic_bias_matrix(np.concatenate(arrays), centers, kappas) / kbt
    return solve_wham(u, [len(q) for q in arrays], **solver_options)


def histogram_pmf(
    coordinates: np.ndarray,
    log_weights: np.ndarray,
    edges: np.ndarray,
    kbt: float,
    *,
    reference_coordinate: Optional[float] = None,
) -> HistogramPmf:
    """Histogram in log space; unsupported bins remain +inf, without errors.

    ``weight_ess_iid`` is a weight-concentration diagnostic, not a count of
    temporally independent samples or a PMF uncertainty estimate.
    """

    q = np.asarray(coordinates, dtype=float)
    lw = np.asarray(log_weights, dtype=float)
    bins = np.asarray(edges, dtype=float)
    if q.ndim != 1 or lw.shape != q.shape or not len(q):
        raise ValueError("Coordinate/weight dimensions do not match")
    if bins.ndim != 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("Bin edges must be strictly increasing")
    if not all(np.isfinite(v).all() for v in (q, lw, bins)) or not np.isfinite(kbt) or kbt <= 0:
        raise ValueError("Nonfinite histogram input or invalid kBT")
    if q.min() < bins[0] or q.max() > bins[-1]:
        raise ValueError("Histogram range does not cover all samples")
    lw = lw - float(_lse(lw))
    index = np.searchsorted(bins, q, side="right") - 1
    index[q == bins[-1]] = len(bins) - 2
    n = len(bins) - 1
    logmass = np.full(n, -np.inf)
    logsquared = np.full(n, -np.inf)
    np.logaddexp.at(logmass, index, lw)
    np.logaddexp.at(logsquared, index, 2 * lw)
    raw_counts = np.bincount(index, minlength=n)
    density = np.exp(logmass) / np.diff(bins)
    free = -kbt * (logmass - np.log(np.diff(bins)))
    centers = 0.5 * (bins[:-1] + bins[1:])
    finite = raw_counts > 0
    if reference_coordinate is None:
        free[finite] -= free[finite].min()
    else:
        if not np.isfinite(reference_coordinate):
            raise ValueError("Reference coordinate must be finite")
        j = int(np.argmin(abs(centers - reference_coordinate)))
        if not finite[j]:
            raise ValueError("Reference bin has no support")
        free[finite] -= free[j]
    ess = np.zeros(n)
    ess[finite] = np.exp(2 * logmass[finite] - logsquared[finite])
    return HistogramPmf(centers, free, density, raw_counts, ess)


def histogram_overlap(left: np.ndarray, right: np.ndarray, edges: np.ndarray) -> float:
    """Observed probability overlap on common caller-supplied bins.

    Every observation must lie inside the grid. This empirical overlap differs
    from the fitted WHAM state-overlap matrix and uses no numerical threshold.
    """
    a, b, bins = (np.asarray(value, dtype=float) for value in (left, right, edges))
    if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all() for x in (a, b, bins)):
        raise ValueError("Finite nonempty vectors are required")
    if len(bins) < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("Common histogram edges must increase")
    if min(a.min(), b.min()) < bins[0] or max(a.max(), b.max()) > bins[-1]:
        raise ValueError("Histogram edges must cover both observation sets")
    pa = np.histogram(a, bins)[0] / len(a)
    pb = np.histogram(b, bins)[0] / len(b)
    return float(np.minimum(pa, pb).sum())


def load_window(
    path: Path,
    center: float,
    kappa: float,
    *,
    discard_time: float = 10,
    end_time: Optional[float] = None,
    cv_field: str = "dz.z",
    bias_field: str = "umb.bias",
    bias_tolerance: float = 1e-4,
    input_format: str = "plumed",
    time_field: Optional[str] = None,
    time_scale: float = 1.0,
    bias_energy_scale: float = 1.0,
    expected_samples: Optional[int] = None,
    expected_time_spacing: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain discard_time < scaled_time <= end_time and audit harmonic energy.

    PLUMED input remains the default. LAMMPS scalar fix-ave/time tables use
    named columns and default to TimeStep. time_scale converts the raw clock;
    bias_energy_scale converts recorded energies to the unit of kappa and kBT.
    """

    if input_format == "plumed":
        table = read_plumed_numeric_table(path, deduplicate_time=False)
        fields, data = table.fields, table.data
        clock = time_field or "time"
    elif input_format == "lammps_fix_ave_time":
        fields, data = read_fix_ave_time_table(path)
        clock = time_field or "TimeStep"
    else:
        raise ValueError(f"Unsupported umbrella input format: {input_format}")
    if len(set(fields)) != len(fields):
        raise ValueError("Duplicate umbrella table columns")
    for name in (clock, cv_field, bias_field):
        if name not in fields:
            raise ValueError(f"Missing umbrella table column: {name}")
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite umbrella table values")
    for value in (time_scale, bias_energy_scale):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Time and bias energy scales must be finite and positive")
    if not np.isfinite(discard_time) or (
        end_time is not None and (not np.isfinite(end_time) or end_time <= discard_time)
    ):
        raise ValueError("Invalid umbrella time selection")
    if not np.isfinite(bias_tolerance) or bias_tolerance < 0:
        raise ValueError("Bias tolerance must be finite and nonnegative")
    t = data[:, fields.index(clock)] * time_scale
    q = data[:, fields.index(cv_field)]
    bias = data[:, fields.index(bias_field)] * bias_energy_scale
    if not np.isfinite(t).all() or not np.isfinite(bias).all():
        raise ValueError("Nonfinite converted umbrella values")
    keep = t > discard_time
    if end_time is not None:
        keep &= t <= end_time
    t, q, bias = t[keep], q[keep], bias[keep]
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("Empty window or non-increasing production time")
    if expected_samples is not None:
        if (
            isinstance(expected_samples, bool)
            or not isinstance(expected_samples, (int, np.integer))
            or expected_samples < 1
        ):
            raise ValueError("Expected sample count must be a positive integer")
        if len(t) != expected_samples:
            raise ValueError("Window sample count disagrees with expected_samples")
    if expected_time_spacing is not None:
        if not np.isfinite(expected_time_spacing) or expected_time_spacing <= 0:
            raise ValueError("Expected time spacing must be finite and positive")
        if not np.allclose(np.diff(t), expected_time_spacing, rtol=1e-9, atol=1e-12):
            raise ValueError("Window timestamps disagree with expected_time_spacing")
    expected = harmonic_bias_matrix(q, [center], [kappa])[0]
    if not np.allclose(bias, expected, atol=bias_tolerance, rtol=0):
        raise ValueError("Recorded bias disagrees with supplied harmonic potential")
    return t, q


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Analyze an explicit, portable JSON manifest; no scheduler discovery."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.2)
    parser.add_argument("--reference", type=float)
    args = parser.parse_args(argv)
    config = json.loads(args.manifest.read_text())
    kbt = float(config["kbt"])
    windows = config["windows"]
    samples, centers, kappas = [], [], []
    options = (
        "discard_time", "end_time", "cv_field", "bias_field", "bias_tolerance",
        "input_format", "time_field", "time_scale", "bias_energy_scale",
        "expected_samples", "expected_time_spacing",
    )
    identities, source_paths = set(), set()
    inputs = []
    for index, row in enumerate(windows):
        path = Path(row["colvar"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        identity = str(row.get("id", index))
        source_identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in identities or source_identity in source_paths:
            raise ValueError("Duplicate umbrella window identity or input file")
        identities.add(identity)
        source_paths.add(source_identity)
        selection = {
            key: row.get(key, config.get(key))
            for key in options
            if key in row or key in config
        }
        t, q = load_window(path, row["center"], row["kappa"], **selection)
        samples.append(q)
        centers.append(row["center"])
        kappas.append(row["kappa"])
        inputs.append({
            "id": identity, "colvar": str(path), "center": row["center"],
            "kappa": row["kappa"], "samples": len(q),
            "first_time": float(t[0]), "last_time": float(t[-1]),
            "input_format": selection.get("input_format", "plumed"),
            "selection": selection,
        })
    result = fit_harmonic_umbrellas(samples, centers, kappas, kbt)
    if not result.converged:
        raise RuntimeError(f"WHAM failed numerical convergence: {result.score_residual}")
    q = np.concatenate(samples)
    width = args.bin_width
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Bin width must be positive")
    edges = np.arange(np.floor(q.min() / width) * width, q.max() + 1.01 * width, width)
    profile = histogram_pmf(q, result.log_weights, edges, kbt, reference_coordinate=args.reference)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.output / "pmf.csv", np.column_stack((profile.centers, profile.free_energy, profile.density, profile.raw_counts, profile.weight_ess_iid)), delimiter=",", header="q,pmf,density,raw_count,weight_ess_iid", comments="")
    np.savetxt(args.output / "overlap.csv", result.overlap_matrix, delimiter=",")
    metadata = {
        "method": "binless WHAM", "counts": result.counts.tolist(),
        "converged": result.converged, "iterations": result.iterations,
        "score_residual": result.score_residual, "kbt": kbt,
        "equilibrium_qualified": False, "uncertainty": "not estimated",
        "jacobian_applied": False, "windows": inputs,
    }
    (args.output / "solver.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
