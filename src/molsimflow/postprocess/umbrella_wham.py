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
        j = int(np.argmin(abs(centers - reference_coordinate)))
        if not finite[j]:
            raise ValueError("Reference bin has no support")
        free[finite] -= free[j]
    ess = np.zeros(n)
    ess[finite] = np.exp(2 * logmass[finite] - logsquared[finite])
    return HistogramPmf(centers, free, density, raw_counts, ess)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse the strict PLUMED reader, retaining discard_time < t <= end_time.

    Time and coordinates retain input units. Bias tolerance is in the energy
    unit of kappa. No duplicate or non-monotonic production times are accepted.
    """

    table = read_plumed_numeric_table(path, deduplicate_time=False)
    t = table.data[:, table.fields.index("time")]
    q = table.data[:, table.fields.index(cv_field)]
    bias = table.data[:, table.fields.index(bias_field)]
    keep = t > discard_time
    if end_time is not None:
        keep &= t <= end_time
    t, q, bias = t[keep], q[keep], bias[keep]
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("Empty window or non-increasing production time")
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
    for row in windows:
        path = Path(row["colvar"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        _, q = load_window(
            path, row["center"], row["kappa"],
            discard_time=config.get("discard_time", 10),
            end_time=config.get("end_time"),
            cv_field=config.get("cv_field", "dz.z"),
            bias_field=config.get("bias_field", "umb.bias"),
        )
        samples.append(q)
        centers.append(row["center"])
        kappas.append(row["kappa"])
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
    metadata = {"method": "binless WHAM", "counts": result.counts.tolist(), "converged": result.converged, "iterations": result.iterations, "score_residual": result.score_residual, "kbt": kbt, "equilibrium_qualified": False, "uncertainty": "not estimated", "jacobian_applied": False}
    (args.output / "solver.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
