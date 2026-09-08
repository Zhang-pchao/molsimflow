"""Compare completed PIMD reweighting analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.pimd_reweight import _matplotlib, require, save_figure, sha256


def comparison_report_title(contract: Mapping[str, object]) -> str:
    """Return an explicit comparison title without assuming the bias representation."""
    return str(contract.get("report_title", "PIMD OPES comparison"))


def aligned_surface_difference(
    reference: np.ndarray, current: np.ndarray, support: np.ndarray
) -> Tuple[np.ndarray, float, float, float]:
    """Remove the arbitrary FES offset and compare only shared support."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    support = np.asarray(support, dtype=bool)
    require(reference.shape == current.shape == support.shape, "surface shape mismatch")
    require(int(np.count_nonzero(support)) >= 2, "insufficient shared support")
    offset = float(np.median((current - reference)[support]))
    difference = current - offset - reference
    selected = difference[support]
    return (
        difference,
        offset,
        float(np.sqrt(np.mean(selected**2))),
        float(np.max(np.abs(selected))),
    )


def read_numeric_csv(path: Path) -> Dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"empty table: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in rows[0]
    }


def load_run(config: Mapping[str, object]) -> Dict[str, object]:
    root = Path(str(config["analysis_root"]))
    require(root.is_dir(), f"analysis root missing: {root}")
    manifest = root / "provenance" / "OUTPUT-SHA256SUMS"
    expected = config.get("output_manifest_sha256")
    if expected is not None:
        require(sha256(manifest) == str(expected), f"analysis manifest mismatch: {root}")
    summary = json.loads((root / "qc" / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("status") == "PASS", f"analysis is not PASS: {root}")
    return {"config": config, "root": root, "summary": summary}


def _one_dimensional_figure(
    output: Path,
    runs: Sequence[Mapping[str, object]],
    cv: str,
    cv_label: str,
    max_kcal: float,
    window_label: str,
) -> Dict[str, object]:
    plt, _ = _matplotlib()
    tables = [read_numeric_csv(Path(run["root"]) / "fes1d" / f"{cv}.csv") for run in runs]
    grid = tables[0][cv]
    require(all(np.allclose(grid, table[cv]) for table in tables[1:]), f"{cv} grid mismatch")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for axis, run, table in zip(axes[:2], runs, tables):
        styles = (
            ("centroid", str(run["config"].get("sampling_label", "Centroid")), "#d97706", "-"),
            ("eq8", "Quantum Eq. 8", "#2563eb", "--"),
            ("eq10", "Quantum Eq. 10", "#b91c1c", "-"),
        )
        for key, label, color, linestyle in styles:
            support = table[f"{key}_support"].astype(bool)
            axis.plot(
                grid,
                np.where(support, table[f"F_{key}_kcal_mol"], np.nan),
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )
        axis.set_title(str(run["config"]["label"]))
        axis.set_xlabel(cv_label)
        axis.grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Free energy (kcal/mol)")
    axes[0].legend(frameon=False, fontsize=8)

    overlay_metrics = {}
    for key, estimator, linestyle in (
        ("centroid", "Centroid", "-"),
        ("eq10", "Quantum Eq. 10", "--"),
    ):
        support = np.logical_and.reduce(
            [table[f"{key}_support"].astype(bool) for table in tables]
        )
        reference = tables[0][f"F_{key}_kcal_mol"]
        current = tables[1][f"F_{key}_kcal_mol"]
        _, offset, rmse, maximum = aligned_surface_difference(
            reference, current, support
        )
        for run, values, color in zip(
            runs, (reference, current - offset), ("#2563eb", "#d97706")
        ):
            axes[2].plot(
                grid,
                np.where(support, values, np.nan),
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=f"{run['config']['short_label']} | {estimator}",
            )
        overlay_metrics[key] = {
            "shared_support_points": int(np.count_nonzero(support)),
            "alignment_offset_kcal_mol": offset,
            "shape_rmse_kcal_mol": rmse,
            "shape_max_abs_kcal_mol": maximum,
        }
    axes[2].set_title("Cross-run centroid and Eq. 10 overlays")
    axes[2].set_xlabel(cv_label)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    for axis in axes:
        axis.set_ylim(0.0, max_kcal)
    fig.suptitle(f"Reweighted 1D FES: {cv} | {window_label}")
    save_figure(fig, output / "figures" / f"fes1d-{cv}-comparison")
    plt.close(fig)
    eq10 = overlay_metrics["eq10"]
    return {
        "cv": cv,
        "centroid": overlay_metrics["centroid"],
        "eq10": eq10,
        "shared_eq10_support_points": eq10["shared_support_points"],
        "eq10_alignment_offset_kcal_mol": eq10["alignment_offset_kcal_mol"],
        "eq10_shape_rmse_kcal_mol": eq10["shape_rmse_kcal_mol"],
        "eq10_shape_max_abs_kcal_mol": eq10["shape_max_abs_kcal_mol"],
    }


def _two_dimensional_figures(
    output: Path,
    runs: Sequence[Mapping[str, object]],
    max_kcal: float,
    difference_max_kcal: float,
    window_label: str,
    cv_labels: Sequence[str],
    zoom: Sequence[Sequence[float]] | None = None,
) -> Dict[str, object]:
    plt, TwoSlopeNorm = _matplotlib()
    tables = [read_numeric_csv(Path(run["root"]) / "fes2d" / "primary.csv") for run in runs]
    x = np.unique(tables[0]["logdistance"])
    y = np.unique(tables[0]["ionization"])
    require(all(np.allclose(x, np.unique(table["logdistance"])) and np.allclose(y, np.unique(table["ionization"])) for table in tables[1:]), "2D grid mismatch")
    shape = (len(y), len(x))

    views = [(None, "")]
    if zoom is not None:
        views.append((zoom, "-sampled-region"))
    for view, suffix in views:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True, constrained_layout=True)
        image = None
        for row, (run, table) in enumerate(zip(runs, tables)):
            for column, key in enumerate(("centroid", "eq10")):
                values = table[f"F_{key}_kcal_mol"].reshape(shape)
                support = table[f"{key}_support"].reshape(shape).astype(bool)
                image = axes[row, column].pcolormesh(x, y, np.where(support, values, np.nan), shading="auto", cmap="viridis", vmin=0.0, vmax=max_kcal)
                sampling_label = str(run["config"].get("sampling_label", "Centroid"))
                axes[row, column].set_title(f"{run['config']['short_label']} | {sampling_label if key == 'centroid' else 'Quantum Eq. 10'}")
                axes[row, column].set_xlabel(cv_labels[0])
                axes[row, column].set_ylabel(cv_labels[1])
                if view is not None:
                    axes[row, column].set_xlim(*view[0])
                    axes[row, column].set_ylim(*view[1])
        fig.colorbar(image, ax=axes, label="Free energy (kcal/mol)", pad=0.02, shrink=0.9)
        fig.suptitle(f"Reweighted 2D FES comparison | {window_label}")
        save_figure(fig, output / "figures" / f"fes2d-cross-run-comparison{suffix}")
        plt.close(fig)

    norm = TwoSlopeNorm(vmin=-difference_max_kcal, vcenter=0.0, vmax=difference_max_kcal)
    metrics: Dict[str, object] = {}
    for view, suffix in views:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True, constrained_layout=True)
        image = None
        for axis, key in zip(axes, ("centroid", "eq10")):
            reference = tables[0][f"F_{key}_kcal_mol"].reshape(shape)
            current = tables[1][f"F_{key}_kcal_mol"].reshape(shape)
            support = np.logical_and.reduce([table[f"{key}_support"].reshape(shape).astype(bool) for table in tables])
            difference, offset, rmse, maximum = aligned_surface_difference(reference, current, support)
            image = axis.pcolormesh(x, y, np.where(support, difference, np.nan), shading="auto", cmap="coolwarm", norm=norm)
            sampling_label = str(runs[0]["config"].get("sampling_label", "Centroid"))
            axis.set_title(f"{sampling_label if key == 'centroid' else 'Quantum Eq. 10'}: run 2 - run 1")
            axis.set_xlabel(cv_labels[0])
            axis.set_ylabel(cv_labels[1])
            if view is not None:
                axis.set_xlim(*view[0])
                axis.set_ylim(*view[1])
            metrics[key] = {
                "shared_support_points": int(np.count_nonzero(support)),
                "alignment_offset_kcal_mol": offset,
                "shape_rmse_kcal_mol": rmse,
                "shape_max_abs_kcal_mol": maximum,
            }
        fig.colorbar(image, ax=axes, label="Aligned FES difference (kcal/mol)", pad=0.02, shrink=0.9)
        fig.suptitle(f"Cross-run 2D FES differences on shared support | {window_label}")
        save_figure(fig, output / "figures" / f"fes2d-cross-run-differences{suffix}")
        plt.close(fig)
    return metrics


def compare(contract_path: Path, output: Path) -> Dict[str, object]:
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output = Path(output)
    require(not output.exists(), f"output exists: {output}")
    (output / "figures").mkdir(parents=True)
    (output / "provenance").mkdir()
    runs = [load_run(config) for config in contract["runs"]]
    require(len(runs) == 2, "comparison requires exactly two analyses")
    window_label = str(contract["window_label"])
    cv_labels = contract.get("cv_labels", {})
    one_dimensional = [
        _one_dimensional_figure(
            output, runs, cv, str(cv_labels.get(cv, cv)),
            float(contract["plot_max_kcal_mol"]), window_label,
        )
        for cv in contract["cvs"]
    ]
    two_dimensional = _two_dimensional_figures(
        output,
        runs,
        float(contract["plot_max_kcal_mol"]),
        float(contract["difference_max_kcal_mol"]),
        window_label,
        tuple(str(cv_labels.get(cv, cv)) for cv in contract["cvs"]),
        contract.get("fes_zoom"),
    )

    metric_rows = []
    baseline_walltime = float(runs[0]["config"]["walltime_seconds"])
    baseline_engine_time = float(runs[0]["config"]["engine_loop_seconds"])
    for run in runs:
        summary = run["summary"]
        config = run["config"]
        walltime = float(config["walltime_seconds"])
        engine_time = float(config["engine_loop_seconds"])
        metric_rows.append(
            {
                "label": config["label"],
                "configuration": config["configuration"],
                "source_job": summary["source_job"],
                "walltime_seconds": walltime,
                "speed_relative_to_run1": baseline_walltime / walltime,
                "engine_loop_seconds": engine_time,
                "engine_speed_relative_to_run1": baseline_engine_time / engine_time,
                "frames": summary["selection"]["frames"],
                "ess": summary["reweighting"]["ess"],
                "ess_fraction": summary["reweighting"]["ess_fraction"],
                "maximum_weight": summary["reweighting"]["maximum_normalized_weight"],
                "eq8_eq10_rmse_kcal_mol": summary["fes"]["eq8_eq10_rmse_common_support_kcal_mol"],
                "sampling_max_ionization": summary["ionization_diagnostic"].get(
                    "sampling_maximum_score",
                    summary["ionization_diagnostic"]["centroid_maximum_score"],
                ),
                "maximum_bead_ionization": summary["ionization_diagnostic"]["maximum_bead_score"],
                "maximum_exact_ionization_reconstruction_error": summary["ionization_diagnostic"]["maximum_reconstruction_error"],
                "ionization_classification": summary["ionization_diagnostic"]["classification"],
                "nlist_numerical": summary["gates"].get("nlist_numerical", "NOT_APPLICABLE"),
                "mean_temperature_K": summary["pimd"]["mean_scaled_temperature_K"],
                "mean_H_spread_A": summary["pimd"]["mean_ring_spread_H_A"],
                "mean_O_spread_A": summary["pimd"]["mean_ring_spread_O_A"],
            }
        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    result = {
        "schema_version": 1,
        "status": "PASS",
        "runs": metric_rows,
        "one_dimensional": one_dimensional,
        "two_dimensional": two_dimensional,
        "comparison_boundary": contract["comparison_boundary"],
        "gates": {
            "artifact_output": "PASS",
            "postprocessing_plumbing": "PASS",
            "physical": "NOT_ASSESSED",
            "scientific_fes_convergence": "NOT_ASSESSED",
        },
    }
    (output / "comparison-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "provenance" / "comparison-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {comparison_report_title(contract)}",
        "",
        "Status: `PASS`",
        f"Window: `{window_label}`",
        f"Run-2 walltime speed relative to run 1: `{metric_rows[1]['speed_relative_to_run1']:.3f}x`",
        f"Run-2 LAMMPS OPES-loop speed relative to run 1: `{metric_rows[1]['engine_speed_relative_to_run1']:.3f}x`",
        "",
    ]
    for row in metric_rows:
        lines.extend(
            [
                f"## {row['label']}",
                "",
                f"- source job: `{row['source_job']}`",
                f"- configuration: `{row['configuration']}`",
                f"- ESS: `{row['ess']:.2f}` (`{100.0 * row['ess_fraction']:.1f}%`)",
                f"- bead mean / maximum-bead ionization score: `{row['sampling_max_ionization']:.4f}` / `{row['maximum_bead_ionization']:.4f}`",
                f"- ionization diagnostic: `{row['ionization_classification']}`",
                "",
            ]
        )
    lines.extend(
        [
            "Cross-run FES differences are descriptive shape comparisons after removal of an arbitrary free-energy offset.",
            "",
            str(contract["comparison_boundary"]),
        ]
    )
    (output / "analysis-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "OUTPUT-SHA256SUMS":
            continue
        manifest_rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "provenance" / "OUTPUT-SHA256SUMS").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return result


def run(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare two completed PIMD reweighting analyses")
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare(args.contract, args.output)
    except Exception as exc:
        print(f"PIMD reweight comparison failed: {exc}")
        return 1
    print(f"PIMD_REWEIGHT_COMPARE_{result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
