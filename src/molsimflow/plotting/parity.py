"""Reusable parity-panel helpers for model-validation figures."""

from __future__ import annotations

import math
from typing import Mapping, Union

import numpy as np


def adaptive_histogram_edges(
    values: np.ndarray,
    *,
    quantile: float = 0.995,
    min_bins: int = 55,
    max_bins: int = 160,
) -> np.ndarray:
    """Return uniform histogram edges for the central error distribution.

    The visible range excludes only the upper tail selected by ``quantile``.
    The number of bins grows with the square root of the displayed sample
    count, subject to bounds that remain legible in a compact inset.
    """

    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Histogram values must contain at least one finite value")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if min_bins < 1 or max_bins < min_bins:
        raise ValueError("Require 1 <= min_bins <= max_bins")

    upper = float(np.quantile(data, quantile))
    if upper <= 0.0:
        upper = float(np.max(data))
    if upper <= 0.0:
        upper = 1.0
    displayed_count = int(np.count_nonzero(data <= upper))
    bins = min(max(int(math.ceil(math.sqrt(displayed_count))), min_bins), max_bins)
    return np.linspace(0.0, 1.05 * upper, bins + 1)


def _add_stats(ax, result: Mapping[str, Union[float, int]], kind: str) -> None:
    if kind == "energy":
        unit = r"meV atom$^{-1}$"
        text = (
            f"MAE {result['energy_mae_meV_per_atom']:.2f} {unit}\n"
            f"RMSE {result['energy_rmse_meV_per_atom']:.2f} {unit}"
        )
    else:
        unit = r"meV $\mathrm{\AA}^{-1}$"
        text = (
            f"MAE {result['force_mae_meV_per_A']:.2f} {unit}\n"
            f"RMSE {result['force_rmse_meV_per_A']:.2f} {unit}"
        )
    ax.text(
        0.97,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.92,
        },
    )


def plot_parity_panel(
    ax,
    reference: np.ndarray,
    model: np.ndarray,
    result: Mapping[str, Union[float, int]],
    *,
    kind: str,
    title: str,
    reference_symbol: str,
    model_symbol: str,
    dense: bool = False,
) -> None:
    """Plot one energy or force parity panel with an adaptive error inset."""

    import matplotlib.pyplot as plt

    x = np.asarray(reference, dtype=float).reshape(-1)
    y = np.asarray(model, dtype=float).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("reference and model must be non-empty arrays with matching shapes")
    if kind not in {"energy", "force"}:
        raise ValueError("kind must be 'energy' or 'force'")

    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    margin = max(0.04 * (high - low), 1.0e-6)
    low, high = low - margin, high + margin
    if kind == "energy" and not dense:
        ax.scatter(x, y, s=18, color="#f28e2b", alpha=0.68, edgecolors="none", rasterized=True)
        color = "#f28e2b"
    else:
        image = ax.hexbin(
            x,
            y,
            gridsize=125,
            bins="log",
            mincnt=1,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        colorbar = plt.colorbar(image, ax=ax, fraction=0.042, pad=0.02)
        colorbar.set_label("Point density", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        color = "#4e79a7"

    if kind == "energy":
        xlabel = rf"$E_{{{reference_symbol}}}-E_{{{reference_symbol},\min}}$ (meV atom$^{{-1}}$)"
        ylabel = rf"$E_{{{model_symbol}}}-E_{{{reference_symbol},\min}}$ (meV atom$^{{-1}}$)"
        error_label = r"Absolute error (meV atom$^{-1}$)"
    else:
        xlabel = rf"$F_{{{reference_symbol}}}$ (eV $\mathrm{{\AA}}^{{-1}}$)"
        ylabel = rf"$F_{{{model_symbol}}}$ (eV $\mathrm{{\AA}}^{{-1}}$)"
        error_label = r"Absolute error (eV $\mathrm{\AA}^{-1}$)"

    ax.plot([low, high], [low, high], "--", color="0.25", lw=1.7, zorder=5)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12, pad=8)
    ax.minorticks_on()

    error = np.abs(y - x)
    edges = adaptive_histogram_edges(error)
    displayed = error[error <= edges[-1]]
    inset = ax.inset_axes([0.08, 0.67, 0.36, 0.27])
    inset.hist(displayed, bins=edges, density=True, color=color, alpha=0.82, edgecolor="none")
    inset.set_xlim(edges[0], edges[-1])
    inset.set_yticks([])
    inset.tick_params(labelsize=7)
    inset.set_xlabel(error_label, fontsize=8)
    _add_stats(ax, result, kind)
