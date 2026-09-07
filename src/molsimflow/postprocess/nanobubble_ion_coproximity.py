"""Joint fixed-ion proximity distributions for accepted nanobubble case runs.

The two coordinates are an ion's distance from a dynamic terminal surface
plane and its nearest main-N2-center distance.  They are joint geometric
coordinates only; their overlap is not defined as a TPCL population or an
adsorption state.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.nanobubble_ion_partition import (
    CaseRun,
    load_cases,
    read_fixed_samples,
    write_csv,
)


def edges(maximum: float, width: float) -> np.ndarray:
    if not (np.isfinite([maximum, width]).all() and maximum > 0.0 and width > 0.0):
        raise ValueError("maximum and width must be finite and positive")
    return np.arange(0.0, maximum + 0.5 * width, width)


def thresholds(values: Sequence[float] | None, maximum: float, name: str) -> tuple[float, ...]:
    selected = tuple(sorted(set(values or (2.0, 4.0, 6.0, 8.0))))
    if not selected or any(not np.isfinite(value) or value <= 0.0 or value > maximum for value in selected):
        raise ValueError(f"{name} thresholds must be finite, positive, and no greater than the plot maximum")
    return selected


def collect_joint_profiles(
    cases: Sequence[CaseRun],
    surface_max_A: float,
    gas_max_A: float,
    bin_width_A: float,
    surface_thresholds_A: Sequence[float],
    gas_thresholds_A: Sequence[float],
) -> tuple[list[dict], list[dict], list[dict]]:
    surface_edges = edges(surface_max_A, bin_width_A)
    gas_edges = edges(gas_max_A, bin_width_A)
    joint_rows: list[dict] = []
    threshold_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for case in cases:
        samples = read_fixed_samples(case)
        by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in samples:
            by_species[row["species"]].append(row)
        validation = json.loads((case.run_dir / "VALIDATION.json").read_text(encoding="utf-8"))
        for species, records in sorted(by_species.items()):
            surface = np.asarray([float(row["z_from_terminal_plane_A"]) for row in records], dtype=float)
            gas = np.asarray([float(row["nearest_main_n2_center_A"]) for row in records], dtype=float)
            finite = np.isfinite(surface) & np.isfinite(gas)
            in_window = finite & (surface >= 0.0) & (surface < surface_max_A) & (gas >= 0.0) & (gas < gas_max_A)
            sample_count = len(records)
            if sample_count == 0:
                raise ValueError(f"no fixed-ion samples for {case.case_id}:{species}")
            counts, _, _ = np.histogram2d(surface[in_window], gas[in_window], bins=(surface_edges, gas_edges))
            for i in range(len(surface_edges) - 1):
                for j in range(len(gas_edges) - 1):
                    count = int(counts[i, j])
                    joint_rows.append(
                        {
                            "case_id": case.case_id,
                            "surface": case.surface,
                            "condition": case.condition,
                            "species": species,
                            "surface_bin_center_A": 0.5 * (surface_edges[i] + surface_edges[i + 1]),
                            "gas_bin_center_A": 0.5 * (gas_edges[j] + gas_edges[j + 1]),
                            "bin_count": count,
                            "probability_density_per_A2": count / (sample_count * bin_width_A**2),
                            "sample_count": sample_count,
                        }
                    )
            for surface_threshold_A in surface_thresholds_A:
                for gas_threshold_A in gas_thresholds_A:
                    threshold_rows.append(
                        {
                            "case_id": case.case_id,
                            "surface": case.surface,
                            "condition": case.condition,
                            "species": species,
                            "surface_threshold_A": surface_threshold_A,
                            "gas_threshold_A": gas_threshold_A,
                            "co_proximity_fraction": float(np.mean(finite & (surface <= surface_threshold_A) & (gas <= gas_threshold_A))),
                            "sample_count": sample_count,
                        }
                    )
            coverage_rows.append(
                {
                    "case_id": case.case_id,
                    "surface": case.surface,
                    "condition": case.condition,
                    "species": species,
                    "fixed_ion_sample_count": sample_count,
                    "finite_coordinate_fraction": float(np.mean(finite)),
                    "joint_plot_window_fraction": float(np.mean(in_window)),
                    "main_n2_integrity_fraction": float(validation["main_n2_integrity_fraction"]),
                    "min_main_n2_count": int(validation["min_main_n2_count"]),
                }
            )
    return joint_rows, threshold_rows, coverage_rows


def _display(value: str) -> str:
    return value.replace("_", " ")


def plot_joint_maps(rows: Sequence[dict], path: Path, surface_max_A: float, gas_max_A: float, bin_width_A: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["species"]))].append(row)
    ordered = sorted(grouped.values(), key=lambda group: (group[0]["surface"], group[0]["condition"], group[0]["species"]))
    maximum = max(float(row["probability_density_per_A2"]) for row in rows)
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15.2, 7.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for axis, group in zip(axes.flat, ordered):
        matrix = np.zeros((round(surface_max_A / bin_width_A), round(gas_max_A / bin_width_A)))
        for row in group:
            i = round((float(row["surface_bin_center_A"]) - 0.5 * bin_width_A) / bin_width_A)
            j = round((float(row["gas_bin_center_A"]) - 0.5 * bin_width_A) / bin_width_A)
            matrix[i, j] = float(row["probability_density_per_A2"])
        image = axis.imshow(matrix.T, origin="lower", aspect="auto", extent=(0.0, surface_max_A, 0.0, gas_max_A), vmin=0.0, vmax=maximum, cmap="magma")
        meta = group[0]
        axis.set_title(f"{_display(str(meta['surface']))}\n{_display(str(meta['condition']))}: {meta['species']}", fontsize=9)
        axis.set_xlabel("Dynamic terminal-plane distance (A)")
        axis.set_ylabel("Nearest main-N2 center distance (A)")
    for axis in axes.flat[len(ordered) :]:
        axis.set_visible(False)
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86)
        colorbar.set_label("Probability density (A-2)")
    figure.suptitle("Joint fixed-ion geometric coordinates", y=0.995)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_threshold_curves(rows: Sequence[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfaces = sorted({str(row["surface"]) for row in rows})
    figure, axes = plt.subplots(1, len(surfaces), figsize=(4.1 * len(surfaces), 3.8), squeeze=False, sharey=True)
    for axis, surface in zip(axes[0], surfaces):
        groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
        for row in rows:
            if row["surface"] == surface:
                groups[(str(row["condition"]), str(row["species"]), float(row["surface_threshold_A"]))].append(row)
        for (condition, species, surface_threshold_A), group in sorted(groups.items()):
            ordered = sorted(group, key=lambda row: float(row["gas_threshold_A"]))
            axis.plot(
                [float(row["gas_threshold_A"]) for row in ordered],
                [float(row["co_proximity_fraction"]) for row in ordered],
                marker="o",
                label=f"{_display(condition)}, {species}; d_s<={surface_threshold_A:g} A",
            )
        axis.set_title(_display(surface))
        axis.set_xlabel("Gas-proximity threshold (A)")
        axis.set_ylabel("Fraction within both thresholds")
        axis.set_ylim(0.0, 1.0)
        axis.legend(fontsize=6.5)
    figure.suptitle("Threshold sweep for fixed-ion co-proximity", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_coverage(rows: Sequence[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: (str(row["surface"]), str(row["condition"]), str(row["species"])))
    labels = [f"{_display(str(row['surface']))}\n{_display(str(row['condition']))}\n{row['species']}" for row in ordered]
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10.2, 4.0))
    width = 0.36
    axis.bar(positions - width / 2, [float(row["joint_plot_window_fraction"]) for row in ordered], width, label="Joint-coordinate window")
    axis.bar(positions + width / 2, [float(row["main_n2_integrity_fraction"]) for row in ordered], width, label="Main-N2 integrity")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Fraction of fixed-ion samples or frames")
    axis.set_title("Coordinate support and gas-integrity coverage")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    cases = load_cases(args.case_manifest)
    surface_thresholds_A = thresholds(args.surface_threshold_A, args.surface_max_A, "surface")
    gas_thresholds_A = thresholds(args.gas_threshold_A, args.gas_max_A, "gas")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    joint_rows, threshold_rows, coverage_rows = collect_joint_profiles(
        cases,
        args.surface_max_A,
        args.gas_max_A,
        args.bin_width_A,
        surface_thresholds_A,
        gas_thresholds_A,
    )
    write_csv(
        output / "joint_profiles.csv",
        joint_rows,
        (
            "case_id", "surface", "condition", "species", "surface_bin_center_A", "gas_bin_center_A",
            "bin_count", "probability_density_per_A2", "sample_count",
        ),
    )
    write_csv(
        output / "co_proximity_thresholds.csv",
        threshold_rows,
        (
            "case_id", "surface", "condition", "species", "surface_threshold_A", "gas_threshold_A",
            "co_proximity_fraction", "sample_count",
        ),
    )
    write_csv(
        output / "coordinate_coverage.csv",
        coverage_rows,
        (
            "case_id", "surface", "condition", "species", "fixed_ion_sample_count", "finite_coordinate_fraction",
            "joint_plot_window_fraction", "main_n2_integrity_fraction", "min_main_n2_count",
        ),
    )
    figures = output / "figures"
    figures.mkdir()
    plot_joint_maps(joint_rows, figures / "01_joint_fixed_ion_proximity_maps.png", args.surface_max_A, args.gas_max_A, args.bin_width_A)
    plot_threshold_curves(threshold_rows, figures / "02_fixed_ion_co_proximity_threshold_sweep.png")
    plot_coverage(coverage_rows, figures / "03_joint_coordinate_coverage.png")
    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "fixed_species": sorted({str(row["species"]) for row in coverage_rows}),
        "joint_profile_rows": len(joint_rows),
        "threshold_rows": len(threshold_rows),
        "surface_max_A": args.surface_max_A,
        "gas_max_A": args.gas_max_A,
        "bin_width_A": args.bin_width_A,
        "surface_thresholds_A": list(surface_thresholds_A),
        "gas_thresholds_A": list(gas_thresholds_A),
        "claim_boundary": [
            "Joint coordinates are descriptive spatial associations, not a TPCL definition or an adsorption state.",
            "Threshold sweeps do not establish ion adsorption free energies, pH causality, equilibrium double layers, transport, or friction.",
            "Only fixed Na_plus and Cl_minus samples are included; geometric H3O+/OH- candidates are excluded.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-max-A", type=float, default=40.0)
    parser.add_argument("--gas-max-A", type=float, default=40.0)
    parser.add_argument("--bin-width-A", type=float, default=1.0)
    parser.add_argument("--surface-threshold-A", action="append", type=float)
    parser.add_argument("--gas-threshold-A", action="append", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
