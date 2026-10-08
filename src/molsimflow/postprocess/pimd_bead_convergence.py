"""Compare one or more PIMD thermo estimators across bead counts."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.pimd_reweight import read_thermo

ESTIMATOR_SPECS = {
    "f_pi[5]": ("Primitive kinetic-energy estimator", "eV"),
    "f_pi[6]": ("Virial energy estimator", "eV"),
    "f_pi[7]": ("Centroid-virial energy estimator", "eV"),
    "f_pi[10]": ("Centroid-virial pressure estimator", "bar"),
}


def _normalise_fields(field: str | Sequence[str]) -> list[str]:
    fields = [field] if isinstance(field, str) else list(field)
    if not fields:
        raise ValueError("at least one estimator field is required")
    if len(set(fields)) != len(fields):
        raise ValueError("estimator fields must be unique")
    return fields


def _estimator_spec(field: str) -> tuple[str, str]:
    return ESTIMATOR_SPECS.get(field, (field, "unknown"))


def _read_cases(
    manifest: Path, fields: Sequence[str], burn_in_ps: float
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            beads = int(row["beads"])
            log = Path(row["log"])
            if not log.is_absolute():
                log = manifest.parent / log
            thermo = read_thermo(log)
            series: dict[str, dict[str, np.ndarray]] = {}
            for field in fields:
                selected = sorted(
                    (values["Time"], values[field])
                    for values in thermo.values()
                    if values["Time"] >= burn_in_ps and field in values
                )
                if len(selected) < 2:
                    raise ValueError(f"{log}: fewer than two {field} samples after burn-in")
                series[field] = {
                    "time": np.asarray([item[0] for item in selected]),
                    "values": np.asarray([item[1] for item in selected]),
                }
            cases.append(
                {
                    "label": row.get("label") or f"P={beads}",
                    "beads": beads,
                    "log": log,
                    "series": series,
                }
            )
    if len(cases) < 2 or len({case["beads"] for case in cases}) != len(cases):
        raise ValueError("manifest requires at least two unique bead counts")
    return sorted(cases, key=lambda case: int(case["beads"]))


def _block_stats(values: np.ndarray, blocks: int) -> tuple[float, float, int]:
    count = min(blocks, len(values))
    if count < 2:
        raise ValueError("at least two non-empty blocks are required")
    means = np.asarray([chunk.mean() for chunk in np.array_split(values, count)])
    return float(values.mean()), float(means.std(ddof=1) / math.sqrt(count)), count


def analyze(
    manifest: Path,
    output: Path,
    *,
    field: str | Sequence[str] = "f_pi[7]",
    burn_in_ps: float = 0.0,
    blocks: int = 5,
    sigma: float = 2.0,
    write_plot: bool = False,
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    fields = _normalise_fields(field)
    cases = _read_cases(manifest, fields, burn_in_ps)
    output.mkdir(parents=True)

    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for estimator in fields:
        physical_quantity, unit = _estimator_spec(estimator)
        estimator_summaries: list[dict[str, object]] = []
        for case in cases:
            series = case["series"][estimator]
            mean, sem, used_blocks = _block_stats(series["values"], blocks)
            row = {
                "field": estimator,
                "physical_quantity": physical_quantity,
                "unit": unit,
                "label": case["label"],
                "beads": case["beads"],
                "samples": len(series["values"]),
                "start_time_ps": float(series["time"][0]),
                "end_time_ps": float(series["time"][-1]),
                "mean": mean,
                "block_sem": sem,
                "blocks": used_blocks,
            }
            summaries.append(row)
            estimator_summaries.append(row)

        reference = estimator_summaries[-1]
        for row in estimator_summaries[:-1]:
            difference = float(row["mean"]) - float(reference["mean"])
            combined = math.hypot(float(row["block_sem"]), float(reference["block_sem"]))
            z_score = abs(difference) / combined if combined else math.inf
            comparisons.append(
                {
                    "field": estimator,
                    "physical_quantity": physical_quantity,
                    "unit": unit,
                    "beads": row["beads"],
                    "reference_beads": reference["beads"],
                    "difference": difference,
                    "combined_block_sem": combined,
                    "z_score": z_score,
                    "within_sigma": z_score <= sigma,
                }
            )

    _write_csv(output / "estimator_summary.csv", summaries)
    _write_csv(output / "estimator_reference_comparison.csv", comparisons)
    if len(fields) == 1:
        _write_legacy_csvs(output, summaries, comparisons)
    if write_plot:
        plot_name = "bead_convergence.png" if len(fields) == 1 else "estimator_convergence.png"
        _plot(cases, fields, output / plot_name)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_legacy_csvs(
    output: Path,
    summaries: list[dict[str, object]],
    comparisons: list[dict[str, object]],
) -> None:
    unit = str(summaries[0]["unit"])
    suffix = unit if unit != "unknown" else "value"
    legacy_summaries = [
        {
            "label": row["label"],
            "beads": row["beads"],
            "samples": row["samples"],
            "start_time_ps": row["start_time_ps"],
            "end_time_ps": row["end_time_ps"],
            f"mean_{suffix}": row["mean"],
            f"block_sem_{suffix}": row["block_sem"],
            "blocks": row["blocks"],
        }
        for row in summaries
    ]
    legacy_comparisons = [
        {
            "beads": row["beads"],
            "reference_beads": row["reference_beads"],
            f"difference_{suffix}": row["difference"],
            f"combined_block_sem_{suffix}": row["combined_block_sem"],
            "z_score": row["z_score"],
            "within_sigma": row["within_sigma"],
        }
        for row in comparisons
    ]
    _write_csv(output / "bead_summary.csv", legacy_summaries)
    _write_csv(output / "reference_comparison.csv", legacy_comparisons)


def _plot(cases: list[dict[str, object]], fields: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    columns = min(2, len(fields))
    rows = math.ceil(len(fields) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 3.8 * rows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = list(axes.flat)
    for axis, estimator in zip(flat_axes, fields):
        physical_quantity, unit = _estimator_spec(estimator)
        for case in cases:
            series = case["series"][estimator]
            axis.plot(
                series["time"],
                series["values"],
                lw=1.0,
                label=f"{case['beads']} beads",
            )
        axis.set_title(physical_quantity)
        axis.set_xlabel("Time (ps)")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.2)
    for axis in flat_axes[len(fields) :]:
        axis.remove()
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(cases), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=240)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="LAMMPS thermo estimator; repeat to compare several estimators",
    )
    parser.add_argument("--burn-in-ps", type=float, default=0.0)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--write-plot", action="store_true")
    args = parser.parse_args(argv)
    analyze(
        args.manifest,
        args.output,
        field=args.fields or "f_pi[7]",
        burn_in_ps=args.burn_in_ps,
        blocks=args.blocks,
        sigma=args.sigma,
        write_plot=args.write_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
