"""Lazy plotting with injectable figure writers for compatibility facades."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .reporting import estimator_plot_labels
from .units import EV_TO_KCAL_MOL, KCAL_TO_KJ_MOL
from .weights import cumulative_weight_diagnostics

FES_COLORS = (
    (34, 75, 121),
    (40, 108, 133),
    (57, 156, 146),
    (70, 168, 143),
    (85, 180, 138),
    (118, 194, 116),
    (184, 216, 81),
    (216, 220, 72),
    (250, 223, 63),
    (255, 207, 69),
    (255, 186, 78),
    (252, 164, 83),
    (237, 133, 74),
    (214, 87, 59),
    (255, 255, 255),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    return plt, TwoSlopeNorm


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")


def scatter_norm(values: np.ndarray):
    _, TwoSlopeNorm = _matplotlib()
    low = float(np.min(values))
    high = float(np.max(values))
    if low < 0.0 < high:
        return TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
    return None


def plot_fes2d(
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    max_kcal: float,
    cv_labels: Sequence[str],
    *,
    zoom: Sequence[Sequence[float]] | None = None,
    suffix: str = "",
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
    protocol_label: str = "bias",
    bead_count: int | None = None,
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    max_kj = float(max_kcal) * KCAL_TO_KJ_MOL
    levels = np.linspace(0.0, max_kj, len(FES_COLORS) + 1)
    colors = [tuple(channel / 255.0 for channel in color) for color in FES_COLORS]
    cmap = LinearSegmentedColormap.from_list("molsimflow_fes", colors, N=len(FES_COLORS))
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    estimator_labels = estimator_plot_labels(bias_mode)
    labels = (
        ("centroid", sampling_label),
        ("probability_mean", estimator_labels["probability_mean"]),
        ("free_energy_mean", estimator_labels["free_energy_mean"]),
    )
    image = None
    for axis, (name, title) in zip(axes, labels):
        values = np.ma.masked_where(
            ~np.asarray(supports[name], dtype=bool),
            np.asarray(surfaces_kcal[name]) * KCAL_TO_KJ_MOL,
        )
        if np.any(supports[name]):
            image = axis.contourf(x_grid, y_grid, values, levels=levels, cmap=cmap, extend="max")
            axis.contour(
                x_grid,
                y_grid,
                values,
                levels=levels[1:-1],
                colors="black",
                linewidths=0.4,
                alpha=0.55,
            )
        else:
            axis.text(
                0.5, 0.5, "Insufficient support", transform=axis.transAxes, ha="center", va="center"
            )
        axis.set_title(title)
        axis.set_xlabel(cv_labels[0])
    axes[0].set_ylabel(cv_labels[1])
    if zoom is not None:
        for axis in axes:
            axis.set_xlim(*zoom[0])
            axis.set_ylim(*zoom[1])
    fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        label="Free energy (kJ/mol)",
        pad=0.13,
        shrink=0.72,
        ticks=levels[::2],
    )
    bead_prefix = f"P={bead_count} " if bead_count is not None else ""
    fig.suptitle(f"{bead_prefix}reweighted free-energy comparison")
    _save_figure(fig, output / "figures" / f"fes2d-comparison{suffix}")
    plt.close(fig)


def plot_fes_differences(
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    max_abs_kcal: float,
    cv_labels: Sequence[str],
    *,
    zoom: Sequence[Sequence[float]] | None = None,
    suffix: str = "",
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, TwoSlopeNorm = _matplotlib()
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    estimator_labels = estimator_plot_labels(bias_mode)
    probability_label = estimator_labels["probability_mean"]
    diagnostic_label = estimator_labels["free_energy_mean"]
    panels = (
        (
            surfaces_kcal["probability_mean"] - surfaces_kcal["centroid"],
            supports["sampling_probability_common"],
            f"{probability_label} - {sampling_label}",
        ),
        (
            surfaces_kcal["free_energy_mean"] - surfaces_kcal["centroid"],
            supports["sampling_free_energy_common"],
            f"{diagnostic_label} - {sampling_label}",
        ),
        (
            surfaces_kcal["free_energy_mean"] - surfaces_kcal["probability_mean"],
            supports["estimator_common"],
            f"{diagnostic_label} - {probability_label}",
        ),
    )
    image = None
    norm = TwoSlopeNorm(vmin=-max_abs_kcal, vcenter=0.0, vmax=max_abs_kcal)
    for axis, (values, support, title) in zip(axes, panels):
        image = axis.pcolormesh(
            x_grid,
            y_grid,
            np.where(support, values, np.nan),
            shading="auto",
            cmap="coolwarm",
            norm=norm,
        )
        axis.set_title(title)
        axis.set_xlabel(cv_labels[0])
    axes[0].set_ylabel(cv_labels[1])
    if zoom is not None:
        for axis in axes:
            axis.set_xlim(*zoom[0])
            axis.set_ylim(*zoom[1])
    fig.colorbar(image, ax=axes, label="Free-energy difference (kcal/mol)", pad=0.02, shrink=0.9)
    fig.suptitle("Reweighted 2D free-energy differences on common support")
    _save_figure(fig, output / "figures" / f"fes2d-differences{suffix}")
    plt.close(fig)


def plot_fes1d(
    output: Path,
    name: str,
    grid: np.ndarray,
    curves_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    max_kcal: float,
    cv_label: str,
    *,
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    estimator_labels = estimator_plot_labels(bias_mode)
    styles = {
        "centroid": (f"{sampling_label} FES", "#d97706", "-"),
        "probability_mean": (estimator_labels["probability_mean"], "#2563eb", "-"),
        "free_energy_mean": (estimator_labels["free_energy_mean"], "#b91c1c", "-"),
    }
    for key in ("centroid", "probability_mean", "free_energy_mean"):
        label, color, linestyle = styles[key]
        axis.plot(
            grid,
            np.where(supports[key], np.asarray(curves_kcal[key]) * KCAL_TO_KJ_MOL, np.nan),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
        )
    visible = np.any(
        np.stack(
            [
                np.where(supports[key], curves_kcal[key], np.inf)
                for key in ("centroid", "probability_mean", "free_energy_mean")
            ]
        )
        <= float(max_kcal),
        axis=0,
    )
    if np.any(visible):
        selected = grid[visible]
        padding = 0.03 * max(float(selected[-1] - selected[0]), float(grid[1] - grid[0]))
        axis.set_xlim(
            max(float(grid[0]), float(selected[0] - padding)),
            min(float(grid[-1]), float(selected[-1] + padding)),
        )
    axis.set_xlabel(cv_label)
    axis.set_ylabel("Free energy (kJ/mol)")
    axis.set_ylim(0.0, float(max_kcal) * KCAL_TO_KJ_MOL)
    axis.set_title("Reweighted 1D free-energy comparison")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    _save_figure(fig, output / "figures" / f"fes1d-{name}")
    plt.close(fig)


def plot_bead_cv_bias(
    output: Path,
    time_ps: np.ndarray,
    centroid: np.ndarray,
    beads: np.ndarray,
    bias_kcal: np.ndarray,
    cv_labels: Sequence[str],
    plot_stride: int,
    *,
    sampling_label: str = "Centroid",
    sampling_slug: str = "centroid",
    protocol_label: str = "bias",
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    stride = max(1, int(plot_stride))
    norm = scatter_norm(bias_kcal)
    fig, axes = plt.subplots(
        beads.shape[1],
        2,
        figsize=(12.5, 2.6 * beads.shape[1]),
        sharex=True,
        sharey="col",
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for bead in range(beads.shape[1]):
        for component, name in enumerate(cv_labels):
            axis = axes[bead, component]
            image = axis.scatter(
                time_ps[::stride],
                beads[::stride, bead, component],
                c=bias_kcal[::stride],
                s=5,
                cmap="coolwarm",
                norm=norm,
                rasterized=True,
            )
            axis.set_ylabel(f"Bead {bead + 1}\n{name}")
            axis.grid(color="#e5e7eb", linewidth=0.4)
    for axis in axes[-1]:
        axis.set_xlabel("Time (ps)")
    fig.colorbar(
        image, ax=axes, label=f"{sampling_label} {protocol_label} (kcal/mol)", pad=0.01, shrink=0.9
    )
    fig.suptitle(
        f"{beads.shape[1]}-bead CV trajectories colored by the "
        f"{sampling_label.lower()} {protocol_label}"
    )
    _save_figure(fig, output / "figures" / "bead-cv-time-bias")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True, constrained_layout=True)
    image = None
    for component, (axis, name) in enumerate(zip(axes, cv_labels)):
        image = axis.scatter(
            time_ps[::stride],
            centroid[::stride, component],
            c=bias_kcal[::stride],
            s=7,
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        axis.set_ylabel(name)
        axis.grid(color="#e5e7eb", linewidth=0.4)
    axes[-1].set_xlabel("Time (ps)")
    fig.colorbar(
        image,
        ax=axes,
        label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.01,
        shrink=0.9,
    )
    fig.suptitle(f"{sampling_label} CV trajectories colored by the applied {protocol_label}")
    _save_figure(fig, output / "figures" / f"{sampling_slug}-cv-time-bias")
    plt.close(fig)

    columns = min(2, beads.shape[1])
    rows = math.ceil(beads.shape[1] / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.25 * columns, 4.25 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for bead, axis in enumerate(axes.ravel()[: beads.shape[1]]):
        image = axis.scatter(
            beads[::stride, bead, 0],
            beads[::stride, bead, 1],
            c=bias_kcal[::stride],
            s=6,
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        axis.set_title(f"Bead {bead + 1}")
        axis.set_xlabel(cv_labels[0])
        axis.set_ylabel(cv_labels[1])
    for axis in axes.ravel()[beads.shape[1] :]:
        axis.remove()
    fig.colorbar(
        image, ax=axes, label=f"{sampling_label} {protocol_label} (kcal/mol)", pad=0.01, shrink=0.9
    )
    fig.suptitle(
        f"{beads.shape[1]}-bead CV sampling colored by the shared "
        f"{sampling_label.lower()} {protocol_label}"
    )
    _save_figure(fig, output / "figures" / "bead-cv2d-bias")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    image = axis.scatter(
        centroid[::stride, 0],
        centroid[::stride, 1],
        c=bias_kcal[::stride],
        s=8,
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )
    axis.set_xlabel(cv_labels[0])
    axis.set_ylabel(cv_labels[1])
    axis.set_title(f"{sampling_label} CV sampling colored by the applied {protocol_label}")
    axis.grid(color="#e5e7eb", linewidth=0.4)
    fig.colorbar(
        image,
        ax=axis,
        label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.02,
    )
    _save_figure(fig, output / "figures" / f"{sampling_slug}-cv2d-bias")
    plt.close(fig)


def plot_cv_spread(
    output: Path, time_ps: np.ndarray, beads: np.ndarray, names: Sequence[str], *, _save_figure=None
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(
        len(names), 1, figsize=(9.5, 3.4 * len(names)), sharex=True, squeeze=False
    )
    for component, (axis, name) in enumerate(zip(axes[:, 0], names)):
        axis.plot(time_ps, np.std(beads[:, :, component], axis=1), color="#2563eb", linewidth=0.8)
        axis.set_ylabel(f"Std. across beads\n{name}")
        axis.grid(color="#e5e7eb", linewidth=0.5)
    axes[-1, 0].set_xlabel("Time (ps)")
    fig.suptitle(f"Instantaneous spread of the {beads.shape[1]} bead CVs")
    _save_figure(fig, output / "figures" / "bead-cv-spread")
    plt.close(fig)


def plot_cv_time_series(
    output: Path,
    time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    cv_labels: Sequence[str],
    sampling_label: str,
    *,
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    """Plot arbitrary one- or two-dimensional CV trajectories without case assumptions."""
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(
        len(cv_labels),
        1,
        figsize=(10.0, 3.6 * len(cv_labels)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for component, (axis, label) in enumerate(zip(axes[:, 0], cv_labels)):
        for bead in range(beads.shape[1]):
            axis.plot(
                time_ps,
                beads[:, bead, component],
                linewidth=0.55,
                alpha=0.6,
                label=f"Bead {bead + 1}",
            )
        axis.plot(
            time_ps,
            sampling[:, component],
            color="#111827",
            linewidth=1.2,
            label=sampling_label,
        )
        axis.set_ylabel(label)
        axis.grid(color="#e5e7eb", linewidth=0.5)
    axes[0, 0].legend(frameon=False, ncol=min(beads.shape[1] + 1, 5), fontsize=8)
    axes[-1, 0].set_xlabel("Time (ps)")
    fig.suptitle("Sampling and bead-local CV trajectories")
    _save_figure(fig, output / "figures" / "cv-time-series")
    plt.close(fig)


def plot_diagnostic_cv_bias(
    output: Path,
    time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    bias_kcal: np.ndarray,
    label: str,
    sampling_label: str,
    plot_stride: int,
    protocol_label: str = "bias",
    *,
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    stride = max(1, int(plot_stride))
    norm = scatter_norm(bias_kcal)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, constrained_layout=True)
    image = axes[0].scatter(
        time_ps[::stride],
        sampling[::stride],
        c=bias_kcal[::stride],
        s=7,
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )
    axes[0].set_ylabel(label)
    axes[0].set_title(f"{sampling_label} diagnostic CV")
    for bead in range(beads.shape[1]):
        axes[1].scatter(
            time_ps[::stride],
            beads[::stride, bead],
            c=bias_kcal[::stride],
            s=4,
            cmap="coolwarm",
            norm=norm,
            alpha=0.45,
            rasterized=True,
        )
    axes[1].set_ylabel(f"Bead-local {label}")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_title(f"{beads.shape[1]} bead-local diagnostic CVs")
    for axis in axes:
        axis.grid(color="#e5e7eb", linewidth=0.4)
    fig.colorbar(
        image,
        ax=axes,
        label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.01,
        shrink=0.9,
    )
    fig.suptitle("Directly printed diagnostic CV; no independent FES bandwidth assigned")
    _save_figure(fig, output / "figures" / "iondistance-time-bias")
    plt.close(fig)


def plot_opes(
    output: Path,
    time_ps: np.ndarray,
    bias_kcal: np.ndarray,
    rct_kcal: np.ndarray,
    zed: np.ndarray,
    neff: np.ndarray,
    nker: np.ndarray,
    weights: np.ndarray,
    kernel_time_ps: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    *,
    weights_are_log: bool = False,
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=False)
    stride = max(1, int(math.ceil(len(time_ps) / 4000)))
    axes[0, 0].scatter(
        time_ps[::stride],
        bias_kcal[::stride],
        label="bias samples",
        color="#d97706",
        s=4,
        alpha=0.25,
        edgecolors="none",
        rasterized=True,
    )
    axes[0, 0].plot(
        time_ps, rct_kcal, label="rct (diagnostic only)", color="#4b5563", linewidth=0.9
    )
    axes[0, 0].set_ylabel("Energy (kcal/mol)")
    axes[0, 0].set_xlabel("Time (ps)")
    axes[0, 0].set_title("Bias samples and reweighting offset")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(
        time_ps,
        zed,
        label="opes.zed",
        color="#059669",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0, 1].step(
        time_ps,
        nker,
        label="opes.nker",
        color="#d97706",
        where="post",
        linewidth=0.9,
    )
    axes[0, 1].set_yscale("log")
    twin = axes[0, 1].twinx()
    twin.plot(time_ps, neff, label="opes.neff", color="#2563eb", linewidth=0.9)
    axes[0, 1].set_ylabel("OPES zed / kernel count (log scale)")
    twin.set_ylabel("OPES neff")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[0, 1].legend(
        handles + twin_handles,
        labels + twin_labels,
        frameon=False,
        loc="best",
    )
    axes[0, 1].set_xlabel("Time (ps)")
    axes[0, 1].set_title("Adaptive OPES state")
    ess_fraction, maximum_share = cumulative_weight_diagnostics(
        weights, log_weights=weights_are_log
    )
    axes[1, 0].plot(
        time_ps,
        ess_fraction,
        label="cumulative Kish ESS fraction",
        color="#7c3aed",
        linewidth=1.0,
    )
    weight_axis = axes[1, 0].twinx()
    weight_axis.plot(
        time_ps,
        maximum_share,
        label="maximum cumulative weight share",
        color="#dc2626",
        linewidth=0.9,
    )
    weight_axis.set_yscale("log")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("Cumulative Kish ESS fraction")
    weight_axis.set_ylabel("Maximum weight share (log scale)")
    axes[1, 0].set_xlabel("Time (ps)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    weight_handles, weight_labels = weight_axis.get_legend_handles_labels()
    axes[1, 0].legend(
        handles + weight_handles,
        labels + weight_labels,
        frameon=False,
        loc="best",
    )
    axes[1, 0].set_title("Cumulative reweighting quality")
    axes[1, 1].plot(kernel_time_ps, sigma_x, label="sigma_logdistance", color="#2563eb")
    axes[1, 1].plot(kernel_time_ps, sigma_y, label="sigma_ionization", color="#d97706")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("OPES kernel sigma")
    axes[1, 1].set_xlabel("Time (ps)")
    axes[1, 1].set_title("Adaptive kernel widths")
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(color="#e5e7eb", linewidth=0.5)
    fig.suptitle("OPES bias, reweighting, and adaptive-kernel diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, output / "figures" / "opes-diagnostics")
    plt.close(fig)


def plot_thermo(
    output: Path,
    time_ps: np.ndarray,
    temperatures: np.ndarray,
    potential: np.ndarray,
    global_values: Mapping[str, np.ndarray],
    beads: int,
    target_temperature: float,
    *,
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), sharex=True)
    for bead in range(beads):
        axes[0, 0].plot(time_ps, temperatures[:, bead] / beads, linewidth=0.55, alpha=0.55)
    axes[0, 0].plot(
        time_ps,
        np.mean(temperatures, axis=1) / beads,
        color="#111827",
        linewidth=1.0,
        label="bead mean / P",
    )
    axes[0, 0].axhline(target_temperature, color="#b91c1c", linestyle="--", label="target")
    axes[0, 0].set_ylabel("Scaled kinetic temperature (K)")
    axes[0, 0].legend(frameon=False)
    mean_potential = np.mean(potential, axis=1) * EV_TO_KCAL_MOL
    axes[0, 1].plot(
        time_ps, mean_potential - np.mean(mean_potential), color="#2563eb", linewidth=0.8
    )
    axes[0, 1].set_ylabel("Mean bead potential - mean (kcal/mol)")
    axes[1, 0].plot(
        time_ps,
        global_values["f_fpimd[5]"] * EV_TO_KCAL_MOL,
        label="primitive",
        color="#d97706",
        linewidth=0.8,
    )
    axes[1, 0].plot(
        time_ps,
        global_values["f_fpimd[7]"] * EV_TO_KCAL_MOL,
        label="centroid virial",
        color="#2563eb",
        linewidth=0.8,
    )
    axes[1, 0].set_ylabel("Kinetic-energy estimator (kcal/mol)")
    axes[1, 0].legend(frameon=False)
    spring = np.sum(global_values["spring_per_bead"], axis=1) * EV_TO_KCAL_MOL
    axes[1, 1].plot(time_ps, spring, color="#7c3aed", linewidth=0.8)
    axes[1, 1].set_ylabel("Total ring-polymer spring energy (kcal/mol)")
    for axis in axes[-1]:
        axis.set_xlabel("Time (ps)")
    for axis in axes.ravel():
        axis.grid(color="#e5e7eb", linewidth=0.5)
    fig.suptitle("PIMD thermostat and energy-estimator diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, output / "figures" / "pimd-thermo-energy")
    plt.close(fig)


def plot_ring_spread(
    output: Path, rows: Sequence[Mapping[str, float]], timestep_fs: float, *, _save_figure=None
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    time_ps = np.asarray([row["step"] * timestep_fs / 1000.0 for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    for name, color in (("rg_H_A", "#d97706"), ("rg_O_A", "#2563eb"), ("rg_all_A", "#111827")):
        if name in rows[0]:
            axes[0].plot(
                time_ps,
                [row[name] for row in rows],
                label=name.replace("rg_", "").replace("_A", ""),
                color=color,
                linewidth=0.8,
            )
    axes[0].set_ylabel("Ring-polymer RMS spread (A)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        time_ps,
        [row["rg_p95_A"] for row in rows],
        label="95th percentile",
        color="#7c3aed",
        linewidth=0.8,
    )
    axes[1].plot(
        time_ps,
        [row["rg_max_A"] for row in rows],
        label="maximum atom",
        color="#b91c1c",
        linewidth=0.6,
        alpha=0.8,
    )
    axes[1].set_ylabel("Atomic bead spread (A)")
    axes[1].set_xlabel("Time (ps)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(color="#e5e7eb", linewidth=0.5)
    fig.suptitle("Ring-polymer spatial spread")
    _save_figure(fig, output / "figures" / "ring-polymer-spread")
    plt.close(fig)


def plot_ionization_events(
    output: Path,
    time_ps: np.ndarray,
    centroid: np.ndarray,
    beads: np.ndarray,
    bias_kcal: np.ndarray,
    thresholds: Sequence[float],
    ideal_pair_score: float,
    *,
    sampling_label: str = "Centroid",
    protocol_label: str = "bias",
    _save_figure=None,
) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    bead_colors = ("#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8")
    for bead, color in enumerate(bead_colors[:beads.shape[1]]):
        axes[0].plot(
            time_ps,
            beads[:, bead],
            color=color,
            linewidth=0.45,
            alpha=0.55,
            label=f"Bead {bead + 1}",
        )
    axes[0].plot(time_ps, centroid, color="#111827", linewidth=1.0, label=sampling_label)
    axes[0].axhline(
        float(ideal_pair_score),
        color="#b45309",
        linestyle="--",
        linewidth=1.4,
        label=f"Ideal localized ion-pair score ({ideal_pair_score:g})",
    )
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Ionization defect score")
    axes[0].set_title(f"{sampling_label} and bead ionization scores")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].grid(color="#e5e7eb", linewidth=0.5)

    stride = 5
    maximum_bead = np.max(beads, axis=1)
    image = axes[1].scatter(
        centroid[::stride],
        maximum_bead[::stride],
        c=bias_kcal[::stride],
        s=9,
        cmap="coolwarm",
        norm=scatter_norm(bias_kcal),
        rasterized=True,
    )
    upper = max(float(ideal_pair_score), float(np.max(maximum_bead))) * 1.04
    axes[1].plot([0.0, upper], [0.0, upper], color="#6b7280", linestyle=":", linewidth=1.0)
    for threshold in thresholds:
        axes[1].axvline(float(threshold), color="#9ca3af", linewidth=0.5, alpha=0.5)
        axes[1].axhline(float(threshold), color="#9ca3af", linewidth=0.5, alpha=0.5)
    axes[1].set_xlim(0.0, upper)
    axes[1].set_ylim(0.0, upper)
    axes[1].set_xlabel(f"{sampling_label} ionization score")
    axes[1].set_ylabel("Maximum bead ionization score")
    axes[1].set_title(f"Bead-local excursions versus {sampling_label.lower()} response")
    axes[1].grid(color="#e5e7eb", linewidth=0.5)
    fig.colorbar(
        image,
        ax=axes[1],
        label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.02,
    )
    fig.suptitle("Water ionization diagnostic; an ideal localized H3O+/OH- pair scores about 2")
    _save_figure(fig, output / "figures" / "ionization-event-diagnostics")
    plt.close(fig)


def fes_plot_limit(
    surfaces: Mapping[str, np.ndarray], supports: Mapping[str, np.ndarray],
    requested: float | None,
) -> float:
    """Use an explicit cap or a rounded supported-energy percentile for plots."""
    if requested is not None:
        limit = float(requested)
        require(np.isfinite(limit) and limit > 0, "plot maximum must be positive")
        return limit
    supported = []
    for name, surface in surfaces.items():
        if name not in supports:
            continue
        values = np.asarray(surface, dtype=float)
        mask = np.asarray(supports[name], dtype=bool) & np.isfinite(values)
        supported.extend(values[mask].tolist())
    require(bool(supported), "no finite supported FES values for automatic plot limit")
    high = max(float(np.percentile(supported, 98)), np.finfo(float).eps)
    order = 10.0 ** math.floor(math.log10(high))
    return float(next(value * order for value in (1, 2, 3, 5, 10) if value * order >= high))



def plot_fes2d_discrete(output: Path, x_grid: np.ndarray, y_grid: np.ndarray, surfaces_kcal: Mapping[str, np.ndarray], supports: Mapping[str, np.ndarray], max_kcal: float | None, cv_labels: Sequence[str], *, zoom: Sequence[Sequence[float]] | None=None, suffix: str='', sampling_label: str='Centroid', bias_mode: str='centroid_coord', protocol_label: str='bias', bead_count: int | None=None, energy_unit="kJ/mol", _save_figure=None) -> None:
    _save_figure = save_figure if _save_figure is None else _save_figure
    require(energy_unit in {"kJ/mol", "kcal/mol"}, "unsupported FES display energy unit")
    if energy_unit == "kJ/mol":
        surfaces_kcal = {name: np.asarray(value) * KCAL_TO_KJ_MOL for name, value in surfaces_kcal.items()}
        max_kcal = None if max_kcal is None else float(max_kcal) * KCAL_TO_KJ_MOL
    """Render supported KDE FES in the discrete comparison style.

    The palette and layout are defaults; the color scale follows the supported
    values unless a positive cap is supplied by the contract.
    """
    plt, _ = _matplotlib()
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    cap = fes_plot_limit(surfaces_kcal, supports, max_kcal)
    levels = np.linspace(0.0, cap, len(FES_COLORS) + 1)
    colors = [tuple(channel / 255.0 for channel in color) for color in FES_COLORS]
    cmap = LinearSegmentedColormap.from_list("molsimflow_fes", colors, N=len(FES_COLORS))
    cmap.set_bad("white")
    norm = BoundaryNorm(levels, cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True)
    estimator_labels = estimator_plot_labels(bias_mode)
    labels = (
        ("centroid", sampling_label),
        ("probability_mean", estimator_labels["probability_mean"]),
        ("free_energy_mean", estimator_labels["free_energy_mean"]),
    )
    image = ScalarMappable(norm=norm, cmap=cmap)
    for index, (axis, (name, title)) in enumerate(zip(axes, labels)):
        support = np.asarray(supports[name], dtype=bool)
        values = np.ma.masked_where(
            ~support | ~np.isfinite(surfaces_kcal[name]),
            np.minimum(np.asarray(surfaces_kcal[name], dtype=float), cap),
        )
        if np.any(support):
            image = axis.contourf(
                x_grid, y_grid, values, levels=levels, cmap=cmap, norm=norm
            )
            axis.contour(
                x_grid, y_grid, values, levels=levels, colors="black",
                linewidths=0.27, alpha=0.58,
            )
        else:
            axis.text(0.5, 0.5, "Insufficient support", transform=axis.transAxes,
                      ha="center", va="center")
        axis.set_title(title)
        axis.set_xlabel(cv_labels[0])
        axis.tick_params(direction="in", top=True, right=True)
        axis.text(0.02, 0.96, f"({chr(97 + index)})", transform=axis.transAxes,
                  ha="left", va="top", fontweight="bold", fontsize=12,
                  bbox={"facecolor": "white", "edgecolor": "none",
                        "alpha": 0.78, "pad": 1.5})
    axes[0].set_ylabel(cv_labels[1])
    if zoom is not None:
        for axis in axes:
            axis.set_xlim(*zoom[0])
            axis.set_ylim(*zoom[1])
    ticks = np.linspace(0.0, cap, 6)
    colorbar = fig.colorbar(
        image, ax=axes, orientation="vertical", label=f"Free energy ({energy_unit})",
        ticks=ticks, fraction=0.025, pad=0.025,
    )
    colorbar.ax.set_yticklabels([f"{tick:g}" for tick in ticks[:-1]] + [f">{cap:g}"])
    bead_prefix = f"P={bead_count} " if bead_count is not None else ""
    fig.suptitle(f"{bead_prefix}reweighted free-energy comparison", y=1.03)
    _save_figure(fig, output / "figures" / f"fes2d-comparison{suffix}")
    plt.close(fig)



class _ConfiguredPlotter:
    def __init__(self, base, energy_unit):
        self.base = base
        self.energy_unit = energy_unit

    def __getattr__(self, name):
        return getattr(self.base, name)

    def plot_fes2d(self, *args, **kwargs):
        return plot_fes2d_discrete(*args, **kwargs, energy_unit=self.energy_unit,
                                   _save_figure=self.base.save_figure)


def configure_plotter(base, plots):
    """Select an explicit 2D display style without altering numerical arrays."""
    style = plots.get("fes2d_style", "legacy")
    require(style in {"legacy", "discrete"}, "unsupported 2D FES plot style")
    unit = plots.get("fes2d_energy_unit", "kJ/mol")
    require(unit in {"kJ/mol", "kcal/mol"}, "unsupported FES display energy unit")
    if style == "legacy":
        require(unit == "kJ/mol", "unit selection requires the discrete 2D style")
        return base
    return _ConfiguredPlotter(base, unit)
