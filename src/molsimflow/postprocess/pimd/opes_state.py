"""Read OPES_METAD STATE snapshots and evaluate a nonperiodic frozen CV field.

The returned force is in CV coordinates. Cartesian chain rules and ring-polymer
force scaling must be checked separately against the simulation runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OpesState:
    time_fs: float
    cv_names: tuple[str, ...]
    centers: np.ndarray
    sigmas: np.ndarray
    heights: np.ndarray
    biasfactor: float
    epsilon: float
    kernel_cutoff: float
    zed: float
    sum_weights: float
    counter: int


def _build_state(fields: tuple[str, ...], metadata: dict[str, str], rows: list[list[float]], time_scale_to_fs: float) -> OpesState:
    if len(fields) < 4 or fields[0] != "time" or fields[-1] != "height":
        raise ValueError("unsupported OPES STATE fields")
    if (len(fields) - 2) % 2:
        raise ValueError("CV and sigma fields must pair")
    ncv = (len(fields) - 2) // 2
    names = fields[1 : 1 + ncv]
    if fields[1 + ncv : -1] != tuple(f"sigma_{name}" for name in names):
        raise ValueError("CV and sigma fields do not match")
    if metadata.get("action") != "OPES_METAD_state" or not rows:
        raise ValueError("expected a nonempty OPES_METAD STATE block")
    if len(set(names)) != len(names):
        raise ValueError("duplicate OPES STATE CV names")
    if any(metadata.get(prefix + name, "none") != "none" for name in names for prefix in ("min_", "max_")):
        raise ValueError("periodic OPES STATE fields are unsupported")
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(fields) or not np.isfinite(values).all():
        raise ValueError("invalid OPES STATE rows")
    time = values[0, 0]
    if not np.all(values[:, 0] == time) or time < 0:
        raise ValueError("STATE block must have one finite nonnegative time")
    try:
        biasfactor = float(metadata["biasfactor"])
        epsilon = float(metadata["epsilon"])
        cutoff = float(metadata["kernel_cutoff"])
        zed = float(metadata["zed"])
        total = float(metadata["sum_weights"])
        counter = int(metadata["counter"])
    except (KeyError, ValueError) as exc:
        raise ValueError("missing or invalid OPES STATE metadata") from exc
    parameters = np.array([biasfactor, epsilon, cutoff, zed, total], dtype=float)
    centers = values[:, 1 : 1 + ncv].copy()
    sigmas = values[:, 1 + ncv : -1].copy()
    heights = values[:, -1].copy()
    if (not np.isfinite(parameters).all() or biasfactor <= 1 or epsilon <= 0
            or cutoff <= 0 or zed <= 0 or total <= 0 or counter <= 0
            or np.any(sigmas <= 0) or np.any(heights < 0)):
        raise ValueError("invalid OPES STATE parameter or kernel")
    return OpesState(
        float(time) * time_scale_to_fs, names, centers, sigmas, heights, biasfactor,
        epsilon, cutoff, zed, total, counter,
    )


def read_opes_states(path: Path, *, time_scale_to_fs: float = 1.0) -> tuple[OpesState, ...]:
    """Read every appended STATE block; each is a compressed field snapshot."""
    if not np.isfinite(time_scale_to_fs) or time_scale_to_fs <= 0:
        raise ValueError("time_scale_to_fs must be finite and positive")
    states: list[OpesState] = []
    fields: tuple[str, ...] | None = None
    metadata: dict[str, str] = {}
    rows: list[list[float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#! FIELDS "):
                if fields is not None:
                    states.append(_build_state(fields, metadata, rows, time_scale_to_fs))
                fields = tuple(line.split()[2:])
                metadata, rows = {}, []
            elif line.startswith("#! SET "):
                if fields is None:
                    raise ValueError("OPES STATE metadata precedes fields")
                parts = line.split(maxsplit=3)
                if len(parts) != 4:
                    raise ValueError("invalid OPES STATE metadata line")
                if parts[2] in metadata:
                    raise ValueError("duplicate OPES STATE metadata")
                metadata[parts[2]] = parts[3].strip()
            elif line.startswith("#") or not line.strip():
                continue
            else:
                if fields is None:
                    raise ValueError("OPES STATE data precedes fields")
                try:
                    rows.append([float(value) for value in line.split()])
                except ValueError as exc:
                    raise ValueError("invalid OPES STATE numeric row") from exc
    if fields is not None:
        states.append(_build_state(fields, metadata, rows, time_scale_to_fs))
    if not states or any(next_state.time_fs <= state.time_fs for state, next_state in zip(states, states[1:])):
        raise ValueError("OPES STATE blocks must have increasing times")
    if any(state.cv_names != states[0].cv_names for state in states[1:]):
        raise ValueError("OPES STATE CV fields changed")
    return tuple(states)


def evaluate_nonperiodic_opes_state(
    state: OpesState, points: np.ndarray, *, kbt_eV: float, batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen OPES energy and negative CV gradient in eV and eV/CV.

    This implements the truncated, shifted Gaussian and normalization used by
    OPES_METAD. Periodic CVs are intentionally unsupported.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if (values.ndim != 2 or values.shape[1] != len(state.cv_names)
            or not values.size or not np.isfinite(values).all()
            or not np.isfinite(kbt_eV) or kbt_eV <= 0
            or not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or batch_size <= 0):
        raise ValueError("invalid CV points, kBT, or batch size")
    energy = np.empty(len(values), dtype=float)
    cv_force = np.empty_like(values)
    cutoff2 = state.kernel_cutoff**2
    cutoff_value = np.exp(-0.5 * cutoff2)
    prefactor = kbt_eV * (1.0 - 1.0 / state.biasfactor)
    for first in range(0, len(values), batch_size):
        last = min(first + batch_size, len(values))
        delta = values[first:last, None, :] - state.centers[None, :, :]
        scaled = delta / state.sigmas[None, :, :]
        radius2 = np.sum(scaled * scaled, axis=2)
        inside = radius2 < cutoff2
        gaussian = np.exp(-0.5 * radius2)
        kernel = state.heights[None, :] * np.where(inside, gaussian - cutoff_value, 0.0)
        probability = np.sum(kernel, axis=1) / state.sum_weights
        deriv_kernel = np.where(inside, state.heights[None, :] * gaussian, 0.0)
        probability_gradient = np.sum(
            -deriv_kernel[:, :, None] * delta / (state.sigmas[None, :, :] ** 2),
            axis=1,
        ) / state.sum_weights
        energy[first:last] = prefactor * np.log(probability / state.zed + state.epsilon)
        cv_force[first:last] = -prefactor * probability_gradient / (
            probability[:, None] + state.zed * state.epsilon
        )
    return energy, cv_force
