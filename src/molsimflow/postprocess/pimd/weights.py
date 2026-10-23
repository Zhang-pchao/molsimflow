"""One reweighting weight per complete ring-polymer path."""
from __future__ import annotations
import math
from typing import Sequence, Tuple
import numpy as np
from .types import WEIGHT_KINDS, validate_bias_mode

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
