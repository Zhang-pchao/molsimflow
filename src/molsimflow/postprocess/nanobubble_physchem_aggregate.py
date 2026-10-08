"""Aggregate validated nanobubble physicochemical case summaries.

The module is deliberately manifest driven: every accepted case must point to
an immutable run directory whose validator and output checksum manifest pass.
Unavailable cases remain visible in the coverage ledger instead of being
silently omitted.  The plots describe individual trajectories; temporal
variation is never presented as replicate uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANIFEST_REQUIRED = {"case_id", "surface", "condition", "status", "run_dir"}
GEOMETRY_REQUIRED = {
    "time_ns",
    "footprint_equivalent_radius_A",
    "bubble_height_q05_q95_A",
    "gas_side_angle_candidate_deg",
    "relative_shape_anisotropy",
    "largest_cluster_n2_count",
}
THERMO_REQUIRED = {"time_ns", "Temp", "Press", "normal_minus_tangential_bar"}
PLOT_METRICS = (
    ("footprint_equivalent_radius_A", "Footprint radius (A)", "geometry"),
    ("bubble_height_q05_q95_A", "Bubble height (A)", "geometry"),
    ("gas_side_angle_candidate_deg", "Gas-side angle proxy (degree)", "geometry"),
    ("relative_shape_anisotropy", "Relative shape anisotropy", "geometry"),
    ("Temp", "Whole-box temperature (K)", "thermo"),
    ("Press", "Whole-box pressure (bar)", "thermo"),
    ("normal_minus_tangential_bar", "Pnormal - Ptangential (bar)", "thermo"),
)
CONDITION_COLORS = {
    "pure_water": "#4C78A8",
    "hcl_63pairs": "#E45756",
    "hcl_63pairs_v2": "#E45756",
    "naoh_ph13p4": "#54A24B",
}


@dataclass(frozen=True)
class CaseData:
    case_id: str
    surface: str
    condition: str
    geometry: list[dict[str, str]]
    thermo: list[dict[str, str]]
    late: dict[str, dict[str, float]]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or MANIFEST_REQUIRED.difference(reader.fieldnames):
            raise ValueError(f"{path}: missing manifest columns {sorted(MANIFEST_REQUIRED)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty case manifest")
    seen = set()
    for row in rows:
        if row["case_id"] in seen:
            raise ValueError(f"{path}: duplicate case_id {row['case_id']}")
        seen.add(row["case_id"])
        if row["status"] not in {"ACCEPTED", "DEFERRED"}:
            raise ValueError(f"{path}: unsupported status {row['status']}")
        if row["status"] == "ACCEPTED" and not row["run_dir"].strip():
            raise ValueError(f"{path}: accepted case lacks run_dir")
    return rows


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(required)}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two rows")
    return rows


def _float(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite {field}")
    return value


def _strict_time(rows: list[dict[str, str]], path: Path) -> tuple[float, float]:
    times = [_float(row, "time_ns", path) for row in rows]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{path}: time_ns is not strictly increasing")
    return times[0], times[-1]


def _verify_hashes(run_dir: Path) -> None:
    manifest = run_dir / "OUTPUT-SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    root = run_dir.resolve()
    with manifest.open(encoding="utf-8") as handle:
        records = [line.split(maxsplit=1) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"{manifest}: empty checksum manifest")
    for parts in records:
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{manifest}: malformed checksum record")
        candidate = Path(parts[1].lstrip(" *").strip()).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"{manifest}: checksum target escapes run directory")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != parts[0]:
            raise ValueError(f"{manifest}: checksum mismatch for {candidate}")


def _late_summary(path: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(path, {"metric", "mean", "std", "sample_count"})
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        metric = row["metric"]
        try:
            mean = float(row["mean"])
            std = float(row["std"])
            count = int(float(row["sample_count"]))
        except ValueError as error:
            raise ValueError(f"{path}: invalid late-window value") from error
        if count < 0 or (count > 0 and not math.isfinite(mean)):
            raise ValueError(f"{path}: invalid late-window summary for {metric}")
        result[metric] = {
            "mean": mean if count > 0 else math.nan,
            "std": std if count > 1 and math.isfinite(std) else 0.0,
            "count": count,
        }
    missing = {metric for metric, _, _ in PLOT_METRICS}.difference(result)
    if missing:
        raise ValueError(f"{path}: missing plotted metrics {sorted(missing)}")
    return result


def _load_case(row: dict[str, str], max_initial_time_ns: float, required_end_ns: float) -> CaseData:
    run_dir = Path(row["run_dir"]).resolve()
    validation_path = run_dir / "VALIDATION.json"
    result_path = run_dir / "ANALYSIS-RESULT.txt"
    if not validation_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"{run_dir}: missing case terminal records")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError(f"{validation_path}: not PASS")
    if "status=PASS" not in result_path.read_text(encoding="utf-8"):
        raise ValueError(f"{result_path}: not PASS")
    _verify_hashes(run_dir)
    physchem = run_dir / "physchem"
    geometry_path = physchem / "geometry_timeseries.csv"
    thermo_path = physchem / "thermo_timeseries.csv"
    geometry = _read_csv(geometry_path, GEOMETRY_REQUIRED)
    thermo = _read_csv(thermo_path, THERMO_REQUIRED)
    first, last = _strict_time(geometry, geometry_path)
    _strict_time(thermo, thermo_path)
    if first > max_initial_time_ns or last < required_end_ns - 1.0e-9:
        raise ValueError(
            f"{geometry_path}: coverage [{first}, {last}] ns does not satisfy "
            f"initial <= {max_initial_time_ns} ns and final >= {required_end_ns} ns"
        )
    return CaseData(
        case_id=row["case_id"],
        surface=row["surface"],
        condition=row["condition"],
        geometry=geometry,
        thermo=thermo,
        late=_late_summary(physchem / "late_window_summary.csv"),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _configure_matplotlib(font_path: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        from matplotlib import font_manager

        font_manager.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()


def _plot_time_series(cases: list[CaseData], output: Path, surfaces: list[str]) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(len(surfaces), 2, figsize=(9.0, 2.35 * len(surfaces)), sharex=True, squeeze=False)
    for index, surface in enumerate(surfaces):
        selected = [case for case in cases if case.surface == surface]
        for case in selected:
            color = CONDITION_COLORS.get(case.condition, "#777777")
            time = [float(row["time_ns"]) for row in case.geometry]
            axes[index, 0].plot(time, [float(row["footprint_equivalent_radius_A"]) for row in case.geometry], color=color, lw=0.7, label=case.condition)
            axes[index, 1].plot(time, [float(row["bubble_height_q05_q95_A"]) for row in case.geometry], color=color, lw=0.7, label=case.condition)
        axes[index, 0].set_ylabel(f"{surface}\nradius (A)")
        axes[index, 1].set_ylabel("height (A)")
        axes[index, 0].legend(frameon=False, fontsize=7, ncol=3, loc="best")
    axes[0, 0].set_title("Equivalent footprint radius")
    axes[0, 1].set_title("N2-cluster height")
    for axis in axes[-1]:
        axis.set_xlabel("Time (ns)")
        axis.set_xlim(0.0, 10.0)
    figure.suptitle("0--10 ns geometry: one trajectory per condition", y=0.997, fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "01_geometry_0_10ns.png", dpi=300)
    plt.close(figure)


def _plot_late_metrics(cases: list[CaseData], output: Path, surfaces: list[str], metrics: Sequence[tuple[str, str, str]], name: str, title: str) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.7), squeeze=False)
    conditions = sorted({case.condition for case in cases})
    shifts = np.linspace(-0.22, 0.22, num=max(len(conditions), 1))
    position = {surface: index for index, surface in enumerate(surfaces)}
    for axis, (metric, label, _) in zip(axes[0], metrics):
        for condition, shift in zip(conditions, shifts):
            selected = [
                case
                for case in cases
                if case.condition == condition
                and case.late[metric]["count"] > 0
                and math.isfinite(case.late[metric]["mean"])
            ]
            if not selected:
                continue
            x = [position[case.surface] + shift for case in selected]
            y = [case.late[metric]["mean"] for case in selected]
            error = [case.late[metric]["std"] for case in selected]
            axis.errorbar(x, y, yerr=error, color=CONDITION_COLORS.get(condition, "#777777"), fmt="o", capsize=2.5, ms=5, label=condition)
        axis.set_xticks(range(len(surfaces)), surfaces, rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(title + "\nBars: within-trajectory temporal SD, not replicate confidence intervals", y=1.03, fontsize=10)
    figure.tight_layout()
    figure.savefig(output / name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_coverage(coverage: list[dict[str, object]], output: Path) -> None:
    from matplotlib import pyplot as plt

    labels = ["Surface", "Condition", "Status", "Geometry rows", "Last time (ns)", "Thermo rows", "Late geometry", "Late thermo"]
    values = [
        [
            str(row["surface"]), str(row["condition"]), str(row["status"]),
            str(row["geometry_rows"]), str(row["geometry_last_time_ns"]), str(row["thermo_rows"]),
            str(row["late_geometry_metrics_available"]), str(row["late_thermo_metrics_available"]),
        ]
        for row in coverage
    ]
    figure, axis = plt.subplots(figsize=(12.0, max(3.1, 0.38 * len(values) + 1.1)))
    axis.axis("off")
    table = axis.table(cellText=values, colLabels=labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for row_index, row in enumerate(coverage, start=1):
        if row["status"] == "DEFERRED":
            for column in range(len(labels)):
                table[(row_index, column)].set_facecolor("#f1f1f1")
    axis.set_title("Coverage ledger: only ACCEPTED rows enter comparisons", pad=16)
    figure.tight_layout()
    figure.savefig(output / "04_coverage_ledger.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if args.max_initial_time_ns < 0.0 or args.required_end_ns <= 0.0:
        raise ValueError("time coverage bounds must be non-negative and have a positive required end")
    rows = _read_tsv(Path(args.case_manifest))
    accepted_rows = [row for row in rows if row["status"] == "ACCEPTED"]
    deferred_rows = [row for row in rows if row["status"] == "DEFERRED"]
    if deferred_rows and not args.allow_incomplete:
        raise ValueError("manifest contains deferred cases; rerun explicitly with --allow-incomplete")
    cases = [_load_case(row, args.max_initial_time_ns, args.required_end_ns) for row in accepted_rows]
    if not cases:
        raise ValueError("no accepted cases to aggregate")
    surfaces = list(dict.fromkeys(row["surface"] for row in rows))
    geometry_metrics = [metric for metric, _, source in PLOT_METRICS if source == "geometry"]
    thermo_metrics = [metric for metric, _, source in PLOT_METRICS if source == "thermo"]
    coverage: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    for row in rows:
        matching = next((case for case in cases if case.case_id == row["case_id"]), None)
        if matching is None:
            coverage.append({"case_id": row["case_id"], "surface": row["surface"], "condition": row["condition"], "status": row["status"], "run_dir": row["run_dir"], "geometry_rows": "", "geometry_last_time_ns": "", "thermo_rows": "", "late_geometry_metrics_available": "", "late_thermo_metrics_available": ""})
            continue
        geometry_available = sum(matching.late[metric]["count"] > 0 for metric in geometry_metrics)
        thermo_available = sum(matching.late[metric]["count"] > 0 for metric in thermo_metrics)
        coverage.append({"case_id": matching.case_id, "surface": matching.surface, "condition": matching.condition, "status": "ACCEPTED", "run_dir": row["run_dir"], "geometry_rows": len(matching.geometry), "geometry_last_time_ns": f"{float(matching.geometry[-1]['time_ns']):.6g}", "thermo_rows": len(matching.thermo), "late_geometry_metrics_available": f"{geometry_available}/{len(geometry_metrics)}", "late_thermo_metrics_available": f"{thermo_available}/{len(thermo_metrics)}"})
        for metric, summary in matching.late.items():
            late_rows.append({"case_id": matching.case_id, "surface": matching.surface, "condition": matching.condition, "metric": metric, "mean": summary["mean"], "temporal_std": summary["std"], "sample_count": summary["count"], "available": summary["count"] > 0})
    _write_csv(output / "coverage_ledger.csv", coverage)
    _write_csv(output / "late_window_comparison.csv", late_rows)
    if not args.no_plots:
        _configure_matplotlib(args.font_path)
        figures = output / "figures"
        figures.mkdir()
        _plot_time_series(cases, figures, surfaces)
        _plot_late_metrics(cases, figures, surfaces, PLOT_METRICS[:4], "02_late_window_geometry_shape.png", "8--10 ns geometry and shape proxies")
        _plot_late_metrics(cases, figures, surfaces, PLOT_METRICS[4:], "03_late_window_whole_box_thermo.png", "8--10 ns whole-box thermodynamic diagnostics")
        _plot_coverage(coverage, figures)
    validation = {
        "status": "PASS",
        "coverage_complete": not deferred_rows,
        "accepted_case_count": len(cases),
        "deferred_case_count": len(deferred_rows),
        "max_initial_time_ns": args.max_initial_time_ns,
        "required_end_ns": args.required_end_ns,
        "late_metric_case_counts": {metric: sum(case.late[metric]["count"] > 0 for case in cases) for metric, _, _ in PLOT_METRICS},
        "surfaces": surfaces,
        "claim_boundary": [
            "Each condition is represented by one trajectory; figures are descriptive comparisons only.",
            "Error bars are within-trajectory temporal standard deviations, not replicate uncertainty or confidence intervals.",
            "Spherical-cap geometry and gas-side angle are molecular-center proxies, not density-dividing-surface contact angles.",
            "Unavailable proxy values remain unavailable in the ledger and plots; no interpolation or replacement is applied.",
            "Whole-box pressure/stress are diagnostics, not local surface tension or bubble pressure.",
            "No ion causal law, free energy, friction, dissipation, or universal mechanism is inferred.",
        ],
    }
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-initial-time-ns", type=float, default=0.02)
    parser.add_argument("--required-end-ns", type=float, default=10.0)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
