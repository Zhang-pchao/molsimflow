"""Weighted whole-frame conditional diagnostics; no sampling or science admission."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from molsimflow.postprocess.pimd_fes import normalized_log_weights


def _finite_array(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real numeric array")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def conditional_region_statistics(
    conditioning: np.ndarray,
    path_values: np.ndarray,
    region_fraction: np.ndarray,
    log_weights: np.ndarray,
    *,
    bin_edges: Sequence[np.ndarray],
    block_ids: np.ndarray,
) -> dict[str, object]:
    """Describe fixed conditioning cells with one explicit weight per frame.

    Inputs have shapes (N, K), (N,), (N, R), and (N,). Region fractions are
    precomputed whole-frame bead averages in [0, 1]; overlapping regions are
    allowed. This function neither validates a weight provider nor infers an
    equilibrium measure. Equal-size contiguous blocks must cover one run in
    temporal order; the caller establishes cadence and adequate block length.

    Bins include their left edge and exclude the right, except that each
    final right edge is included. Outside frames are reported, not silently
    discarded or used to renormalize conditioning-cell masses. Each cell's
    conditional moments normalize weights only within that cell.

    Leave-one-whole-block-out estimates normalize raw log weights afresh.
    Uncertainty is null when a cell loses support, fewer than two blocks
    exist, or a region/complement is unobserved after a deletion. Support
    flags and Kish ESS are diagnostics, not evidence of independent samples,
    equilibrium, sufficient statistical power, a slow mode, or CV utility.
    All values are JSON serializable; no NaN is used for missing estimates.
    """
    coordinates = _finite_array(conditioning, "conditioning")
    path = _finite_array(path_values, "path_values")
    regions = _finite_array(region_fraction, "region_fraction")
    logs = _finite_array(log_weights, "log_weights")
    if coordinates.ndim != 2 or min(coordinates.shape) == 0:
        raise ValueError("conditioning must have nonempty shape (N, K)")
    frames, dimensions = coordinates.shape
    if path.shape != (frames,) or logs.shape != (frames,):
        raise ValueError("path_values and log_weights must have shape (N,)")
    if regions.ndim != 2 or regions.shape[0] != frames or regions.shape[1] == 0:
        raise ValueError("region_fraction must have nonempty shape (N, R)")
    if np.any((regions < 0) | (regions > 1)):
        raise ValueError("region_fraction must lie in [0, 1]")
    edges = [_finite_array(edge, "bin_edges") for edge in bin_edges]
    if len(edges) != dimensions or any(
        edge.ndim != 1 or len(edge) < 2 or np.any(edge[1:] <= edge[:-1])
        for edge in edges
    ):
        raise ValueError("one strictly increasing edge vector is required per dimension")
    blocks = np.asarray(block_ids)
    if blocks.shape != (frames,) or not np.issubdtype(blocks.dtype, np.integer):
        raise ValueError("block_ids must be an integer vector of length N")
    if np.any(blocks < 0):
        raise ValueError("block_ids must be nonnegative")
    starts = np.r_[0, np.flatnonzero(blocks[1:] != blocks[:-1]) + 1]
    labels = blocks[starts]
    sizes = np.diff(np.r_[starts, frames])
    if len(set(labels.tolist())) != len(labels) or np.any(sizes != sizes[0]):
        raise ValueError("blocks must be contiguous, disjoint and equal size, covering all frames")
    block_count = len(labels)
    shape = tuple(len(edge) - 1 for edge in edges)
    indices = np.empty_like(coordinates, dtype=int)
    inside = np.ones(frames, dtype=bool)
    for dim, edge in enumerate(edges):
        value = coordinates[:, dim]
        indices[:, dim] = np.searchsorted(edge, value, side="right") - 1
        indices[value == edge[-1], dim] = len(edge) - 2
        inside &= (value >= edge[0]) & (value <= edge[-1])

    def moments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if not np.any(mask):
            return None
        weight = np.exp(normalized_log_weights(logs[mask]))
        mean = np.dot(weight, path[mask])
        variance = np.dot(weight, (path[mask] - mean) ** 2)
        probabilities = weight @ regions[mask]
        return np.r_[mean, variance, probabilities], weight

    def estimate_record(estimate: np.ndarray | None) -> dict[str, object] | None:
        if estimate is None:
            return None
        return {"path_mean": float(estimate[0]), "path_variance": float(estimate[1]),
                "region_probabilities": estimate[2:].tolist()}

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            global_weights = np.exp(normalized_log_weights(logs))
            cells = []
            for cell_index in np.ndindex(shape):
                mask = inside & np.all(indices == cell_index, axis=1)
                full = moments(mask)
                leave_out = []
                region_support = np.ones(regions.shape[1], dtype=bool)
                for label in labels:
                    keep = mask & (blocks != label)
                    partial = moments(keep)
                    leave_out.append(None if partial is None else partial[0])
                    if partial is None:
                        region_support[:] = False
                    else:
                        # Observed events and complements must retain nonzero weight.
                        weight = partial[1]
                        region_support &= np.all(np.vstack([
                            weight @ regions[keep] > 0,
                            weight @ (1 - regions[keep]) > 0,
                        ]), axis=0)
                supported = full is not None and block_count >= 2 and all(
                    estimate is not None for estimate in leave_out
                )
                region_support &= supported
                standard_error = None
                if supported:
                    estimates = np.asarray(leave_out)
                    standard_error = np.sqrt((block_count - 1) / block_count * np.sum(
                        (estimates - estimates.mean(axis=0)) ** 2, axis=0
                    ))
                status = ("EMPTY_CELL" if full is None else "TOO_FEW_BLOCKS"
                          if block_count < 2 else "CELL_LOSES_SUPPORT"
                          if not supported else "SUPPORTED")
                entry = {
                    "index": list(cell_index), "frame_count": int(mask.sum()),
                    "weight_mass": float(global_weights[mask].sum()),
                    "moments": estimate_record(None if full is None else full[0]),
                    "frame_ess": None if full is None else float(1 / np.dot(full[1], full[1])),
                    "max_frame_weight": None if full is None else float(full[1].max()),
                    "jackknife_status": status,
                    "region_jackknife_support": region_support.tolist(),
                    "path_mean_standard_error": None if standard_error is None
                    else float(standard_error[0]),
                    "path_variance_standard_error": None if standard_error is None
                    else float(standard_error[1]),
                    "region_standard_errors": [float(standard_error[2 + r])
                    if region_support[r] else None for r in range(regions.shape[1])],
                    "leave_one_block_out": [estimate_record(item) for item in leave_out],
                }
                cells.append(entry)
            return {
                "scope": "weighted whole-frame descriptive statistics only",
                "frame_count": frames, "region_count": regions.shape[1],
                "conditioning_dimensions": dimensions, "bin_shape": list(shape),
                "bin_edges": [edge.tolist() for edge in edges],
                "block_labels": labels.tolist(), "block_size": int(sizes[0]),
                "block_count": block_count, "blocks_independent": "NOT_ASSESSED",
                "in_range_frames": int(inside.sum()),
                "out_of_range_frames": int((~inside).sum()),
                "in_range_weight_mass": float(global_weights[inside].sum()),
                "out_of_range_weight_mass": float(global_weights[~inside].sum()),
                "frame_ess": float(1 / np.dot(global_weights, global_weights)),
                "max_frame_weight": float(global_weights.max()), "cells": cells,
            }
    except FloatingPointError as exc:
        raise ValueError("conditional statistics exceed floating-point range") from exc
