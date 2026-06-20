"""Compare printed PLUMED CV traces across multiple simulation cases."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from molsimflow.postprocess.plumed_cv_diagnostics import (
    PlumedTable,
    drop_duplicate_time_rows,
    read_plumed_table,
)

DEFAULT_CVS = (
    "n2_num",
    "cgs",
    "foot_total",
    "nfilm",
    "n2_com_h",
    "n2_com_pos.z",
    "sum_cn.sum",
    "surf_ref_pos.z",
)
DEFAULT_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2")


@dataclass(frozen=True)
class CaseSpec:
    label: str
    run_dir: Path


@dataclass(frozen=True)
class SegmentTable:
    segment_label: str
    segment_dir: Path
    table: PlumedTable
    raw_time_first: float
    raw_time_last: float
    time_source: str
    first_step: float
    last_step: float
    step_stride: float
    timestep_ps: float


@dataclass(frozen=True)
class CaseTable:
    spec: CaseSpec
    table: PlumedTable
    relative_time: np.ndarray
    segments: tuple[SegmentTable, ...]


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def parse_case(value: str) -> CaseSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Case must be LABEL=RUN_DIR")
    label, run_dir = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Case label must not be empty")
    return CaseSpec(label=label, run_dir=Path(run_dir).expanduser())


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{digits}g}"
    return str(value)


def segment_sort_key(path: Path) -> tuple[int, float, str]:
    try:
        return (0, float(path.name), path.name)
    except ValueError:
        match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", path.name)
        if match:
            return (1, float(match.group(1)), path.name)
        return (2, math.inf, path.name)


def discover_colvar_segments(run_dir: Path, colvar_name: str) -> list[Path]:
    if (run_dir / colvar_name).is_file():
        return [run_dir]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing case directory: {run_dir}")
    segments = [child for child in run_dir.iterdir() if child.is_dir() and (child / colvar_name).is_file()]
    segments.sort(key=segment_sort_key)
    if not segments:
        raise FileNotFoundError(f"No {colvar_name} files found under {run_dir}")
    return segments


def read_print_stride(plumed_path: Path) -> Optional[int]:
    if not plumed_path.is_file():
        return None
    pattern = re.compile(r"\bPRINT\b.*\bSTRIDE\s*=\s*([0-9]+)", re.IGNORECASE)
    for line in plumed_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def read_lammps_timestep_ps(log_path: Path) -> Optional[float]:
    if not log_path.is_file():
        return None
    pattern = re.compile(r"Time step\s*:\s*([0-9.eE+-]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def read_first_dump_step(dump_path: Path) -> Optional[float]:
    if not dump_path.is_file():
        return None
    with dump_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ITEM: TIMESTEP"):
                step_line = next(handle, "").strip()
                return float(step_line) if step_line else None
    return None


def retime_segment_table(
    table: PlumedTable,
    segment_dir: Path,
) -> tuple[PlumedTable, dict[str, Union[float, str]]]:
    if not table.has("time") or table.row_count == 0:
        return table, {
            "raw_time_first": math.nan,
            "raw_time_last": math.nan,
            "time_source": "missing_time_column",
            "first_step": math.nan,
            "last_step": math.nan,
            "step_stride": math.nan,
            "timestep_ps": math.nan,
        }

    raw_time = table.column("time")
    raw_time_first = float(raw_time[0])
    raw_time_last = float(raw_time[-1])
    print_stride = read_print_stride(segment_dir / "in.plumed")
    timestep_ps = read_lammps_timestep_ps(segment_dir / "lmp.out")
    first_step = read_first_dump_step(segment_dir / "bubble_1k.lammpstrj")

    if print_stride is None or timestep_ps is None or first_step is None:
        return table, {
            "raw_time_first": raw_time_first,
            "raw_time_last": raw_time_last,
            "time_source": "colvar_time",
            "first_step": first_step if first_step is not None else math.nan,
            "last_step": math.nan,
            "step_stride": print_stride if print_stride is not None else math.nan,
            "timestep_ps": timestep_ps if timestep_ps is not None else math.nan,
        }

    steps = first_step + np.arange(table.row_count, dtype=float) * float(print_stride)
    physical_time = steps * float(timestep_ps)
    time_index = table.columns.index("time")
    data = table.data.copy()
    data[:, time_index] = physical_time
    if "colvar_time" in table.columns:
        colvar_index = table.columns.index("colvar_time")
        data[:, colvar_index] = raw_time
        columns = table.columns
    else:
        columns = tuple(list(table.columns) + ["colvar_time"])
        data = np.column_stack([data, raw_time])
    return (
        PlumedTable(
            path=table.path,
            columns=columns,
            data=data,
            header_count=table.header_count,
            skipped_lines=table.skipped_lines,
        ),
        {
            "raw_time_first": raw_time_first,
            "raw_time_last": raw_time_last,
            "time_source": "lammps_step_reconstructed",
            "first_step": float(first_step),
            "last_step": float(steps[-1]),
            "step_stride": float(print_stride),
            "timestep_ps": float(timestep_ps),
        },
    )


def merge_segment_tables(case: CaseSpec, segments: Sequence[SegmentTable]) -> PlumedTable:
    if not segments:
        raise ValueError(f"No COLVAR segments for {case.label}")
    columns: list[str] = []
    for segment in segments:
        for column in segment.table.columns:
            if column not in columns:
                columns.append(column)
    arrays: list[np.ndarray] = []
    for segment in segments:
        data = np.full((segment.table.row_count, len(columns)), np.nan, dtype=float)
        for source_index, column in enumerate(segment.table.columns):
            data[:, columns.index(column)] = segment.table.data[:, source_index]
        arrays.append(data)
    merged = np.vstack(arrays) if arrays else np.empty((0, len(columns)), dtype=float)
    if "time" in columns and merged.shape[0] > 1:
        order = np.argsort(merged[:, columns.index("time")], kind="mergesort")
        merged = merged[order, :]
    return PlumedTable(
        path=case.run_dir,
        columns=tuple(columns),
        data=merged,
        header_count=sum(segment.table.header_count for segment in segments),
        skipped_lines=sum(segment.table.skipped_lines for segment in segments),
    )


def read_case(case: CaseSpec, colvar_name: str, skip_last_data_line: bool) -> CaseTable:
    segment_dirs = discover_colvar_segments(case.run_dir, colvar_name)
    segments = []
    for segment_dir in segment_dirs:
        table = read_plumed_table(segment_dir / colvar_name, skip_last_data_line=skip_last_data_line)
        table, time_info = retime_segment_table(table, segment_dir)
        segments.append(
            SegmentTable(
                segment_label=segment_dir.name,
                segment_dir=segment_dir,
                table=table,
                raw_time_first=float(time_info["raw_time_first"]),
                raw_time_last=float(time_info["raw_time_last"]),
                time_source=str(time_info["time_source"]),
                first_step=float(time_info["first_step"]),
                last_step=float(time_info["last_step"]),
                step_stride=float(time_info["step_stride"]),
                timestep_ps=float(time_info["timestep_ps"]),
            )
        )
    table = merge_segment_tables(case, segments)
    table, _ = drop_duplicate_time_rows(table)
    if not table.has("time"):
        raise ValueError(f"Missing time column for {case.label}: {case.run_dir}")
    time = table.column("time")
    relative_time = time - float(time[0]) if time.size else np.asarray([], dtype=float)
    return CaseTable(spec=case, table=table, relative_time=relative_time, segments=tuple(segments))


def choose_cv_columns(cases: Sequence[CaseTable], requested: Sequence[str]) -> list[str]:
    if requested:
        seen = set()
        selected = []
        for column in requested:
            if column in seen:
                continue
            if any(case.table.has(column) for case in cases):
                selected.append(column)
                seen.add(column)
        return selected
    common = set(cases[0].table.columns)
    for case in cases[1:]:
        common &= set(case.table.columns)
    common.discard("time")
    selected = [column for column in DEFAULT_CVS if column in common]
    extra = sorted(column for column in common if column not in selected)
    return selected + extra


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(cases: Sequence[CaseTable], cv_columns: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        time = case.table.column("time")
        for column in cv_columns:
            if not case.table.has(column):
                rows.append(
                    {
                        "case": case.spec.label,
                        "run_dir": str(case.spec.run_dir),
                        "cv": column,
                        "present": 0,
                        "count": 0,
                        "time_first_ps": math.nan,
                        "time_last_ps": math.nan,
                        "rel_time_last_ps": math.nan,
                        "first": math.nan,
                        "last": math.nan,
                        "min": math.nan,
                        "max": math.nan,
                        "mean": math.nan,
                        "median": math.nan,
                        "std": math.nan,
                        "span": math.nan,
                        "delta": math.nan,
                    }
                )
                continue
            values = case.table.column(column)
            finite = finite_values(values)
            first = float(values[0]) if values.size else math.nan
            last = float(values[-1]) if values.size else math.nan
            rows.append(
                {
                    "case": case.spec.label,
                    "run_dir": str(case.spec.run_dir),
                    "cv": column,
                    "present": 1,
                    "count": int(values.size),
                    "time_first_ps": float(time[0]) if time.size else math.nan,
                    "time_last_ps": float(time[-1]) if time.size else math.nan,
                    "rel_time_last_ps": float(case.relative_time[-1]) if case.relative_time.size else math.nan,
                    "first": first,
                    "last": last,
                    "min": float(np.min(finite)) if finite.size else math.nan,
                    "max": float(np.max(finite)) if finite.size else math.nan,
                    "mean": float(np.mean(finite)) if finite.size else math.nan,
                    "median": float(np.median(finite)) if finite.size else math.nan,
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "span": float(np.max(finite) - np.min(finite)) if finite.size else math.nan,
                    "delta": last - first if math.isfinite(first) and math.isfinite(last) else math.nan,
                }
            )
    return rows


def long_rows(cases: Sequence[CaseTable], cv_columns: Sequence[str]) -> Iterable[dict[str, object]]:
    for case in cases:
        time = case.table.column("time")
        for column in cv_columns:
            if not case.table.has(column):
                continue
            values = case.table.column(column)
            for t_abs, t_rel, value in zip(time, case.relative_time, values):
                yield {
                    "case": case.spec.label,
                    "run_dir": str(case.spec.run_dir),
                    "time_ps": float(t_abs),
                    "time_rel_ps": float(t_rel),
                    "cv": column,
                    "value": float(value),
                }


def write_common_columns(path: Path, cases: Sequence[CaseTable]) -> None:
    all_columns = sorted(set().union(*(set(case.table.columns) for case in cases)))
    rows = []
    for column in all_columns:
        rows.append(
            {
                "column": column,
                "present_count": sum(1 for case in cases if case.table.has(column)),
                "cases": ";".join(case.spec.label for case in cases if case.table.has(column)),
            }
        )
    write_csv(path, rows, ["column", "present_count", "cases"])


def apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot_cv_overlay(
    cases: Sequence[CaseTable],
    column: str,
    output: Path,
    *,
    dpi: int,
    normalize_delta: bool = False,
) -> bool:
    present = [case for case in cases if case.table.has(column)]
    if not present:
        return False
    plt = _load_pyplot()
    apply_style(plt)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for index, case in enumerate(present):
        y = case.table.column(column).astype(float)
        if normalize_delta:
            finite = finite_values(y)
            denom = float(np.max(finite) - np.min(finite)) if finite.size else math.nan
            if not math.isfinite(denom) or denom == 0.0:
                continue
            y = (y - float(y[0])) / denom
        ax.plot(
            case.relative_time,
            y,
            lw=1.35,
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
            label=case.spec.label,
        )
    ax.set_xlabel("Relative time / ps")
    ax.set_ylabel(f"Delta-normalized {column}" if normalize_delta else column)
    title = f"{column} comparison"
    if normalize_delta:
        title += " (delta / span)"
    ax.set_title(title)
    ax.legend(ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return True


def plot_panel(cases: Sequence[CaseTable], columns: Sequence[str], output: Path, *, dpi: int) -> bool:
    present_columns = [column for column in columns if any(case.table.has(column) for case in cases)]
    if not present_columns:
        return False
    plt = _load_pyplot()
    apply_style(plt)
    ncols = 2
    nrows = int(math.ceil(len(present_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, max(2.2 * nrows, 2.8)), squeeze=False)
    for ax, column in zip(axes.ravel(), present_columns):
        for index, case in enumerate(cases):
            if not case.table.has(column):
                continue
            ax.plot(
                case.relative_time,
                case.table.column(column),
                lw=1.05,
                color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
                label=case.spec.label,
            )
        ax.set_title(column)
        ax.set_xlabel("Relative time / ps")
    for ax in axes.ravel()[len(present_columns) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.subplots_adjust(top=0.91)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return True


def plot_last_value_bars(summary: Sequence[dict[str, object]], cv_columns: Sequence[str], output: Path, *, dpi: int) -> bool:
    plt = _load_pyplot()
    apply_style(plt)
    rows = [row for row in summary if int(row["present"]) == 1 and row["cv"] in cv_columns]
    if not rows:
        return False
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    ncols = 2
    nrows = int(math.ceil(len(cv_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, max(2.2 * nrows, 2.8)), squeeze=False)
    by_cv_case = {(str(row["cv"]), str(row["case"])): row for row in rows}
    for ax, column in zip(axes.ravel(), cv_columns):
        values = []
        labels = []
        for case in cases:
            row = by_cv_case.get((column, case))
            if row is None:
                continue
            value = float(row["last"])
            if math.isfinite(value):
                values.append(value)
                labels.append(case)
        if not values:
            ax.axis("off")
            continue
        x = np.arange(len(values))
        ax.bar(x, values, color=[DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(values))])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"Last {column}")
    for ax in axes.ravel()[len(cv_columns) :]:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return True


def write_report(
    path: Path,
    cases: Sequence[CaseTable],
    cv_columns: Sequence[str],
    summary: Sequence[dict[str, object]],
    figures: Sequence[Path],
) -> None:
    summary_by_case_cv = {(str(row["case"]), str(row["cv"])): row for row in summary}
    lines = [
        "# Printed-CV comparison",
        "",
        f"This report compares the same printed PLUMED CV columns across {len(cases)} simulation cases.",
        "All traces are aligned by relative time after reconstructing segment time from LAMMPS steps when possible.",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        time = case.table.column("time")
        segment_text = ", ".join(
            f"{segment.segment_label}({segment.table.row_count}, {segment.time_source})" for segment in case.segments
        )
        lines.append(
            f"- `{case.spec.label}`: `{case.spec.run_dir}`; rows={case.table.row_count}; "
            f"time={fmt(float(time[0]))}-{fmt(float(time[-1]))} ps; segments={segment_text}"
        )
    lines.extend(["", "## Compared CVs", "", ", ".join(f"`{column}`" for column in cv_columns), ""])
    lines.extend(["## End-point summary", ""])
    header = "| case | cv | first | last | span | delta |"
    lines.append(header)
    lines.append("|---|---|---:|---:|---:|---:|")
    for case in cases:
        for column in cv_columns:
            row = summary_by_case_cv.get((case.spec.label, column))
            if row is None or int(row["present"]) != 1:
                continue
            lines.append(
                f"| {case.spec.label} | `{column}` | {fmt(row['first'])} | {fmt(row['last'])} | "
                f"{fmt(row['span'])} | {fmt(row['delta'])} |"
            )
    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"- `{figure}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colvar-name", default="COLVAR")
    parser.add_argument("--cv", action="append", default=[], help="CV column to plot; repeatable")
    parser.add_argument("--skip-last-data-line", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [read_case(case, args.colvar_name, args.skip_last_data_line) for case in args.case]
    cv_columns = choose_cv_columns(cases, args.cv)
    if not cv_columns:
        raise ValueError("No CV columns selected")

    summary = summary_rows(cases, cv_columns)
    long = list(long_rows(cases, cv_columns))
    summary_path = output_dir / "sphere_cv_summary.csv"
    long_path = output_dir / "sphere_cv_long.csv"
    common_path = output_dir / "sphere_cv_column_presence.csv"
    report_path = output_dir / "sphere_cv_comparison_report.md"
    segment_path = output_dir / "sphere_cv_segments.csv"
    segment_rows = [
        {
            "case": case.spec.label,
            "run_dir": str(case.spec.run_dir),
            "segment": segment.segment_label,
            "segment_dir": str(segment.segment_dir),
            "rows": segment.table.row_count,
            "time_first_ps": float(segment.table.column("time")[0]) if segment.table.has("time") and segment.table.row_count else math.nan,
            "time_last_ps": float(segment.table.column("time")[-1]) if segment.table.has("time") and segment.table.row_count else math.nan,
            "raw_colvar_time_first": segment.raw_time_first,
            "raw_colvar_time_last": segment.raw_time_last,
            "time_source": segment.time_source,
            "first_lammps_step": segment.first_step,
            "last_lammps_step": segment.last_step,
            "print_step_stride": segment.step_stride,
            "lammps_timestep_ps": segment.timestep_ps,
            "columns": ";".join(segment.table.columns),
        }
        for case in cases
        for segment in case.segments
    ]
    write_csv(
        segment_path,
        segment_rows,
        [
            "case",
            "run_dir",
            "segment",
            "segment_dir",
            "rows",
            "time_first_ps",
            "time_last_ps",
            "raw_colvar_time_first",
            "raw_colvar_time_last",
            "time_source",
            "first_lammps_step",
            "last_lammps_step",
            "print_step_stride",
            "lammps_timestep_ps",
            "columns",
        ],
    )
    write_csv(
        summary_path,
        summary,
        [
            "case",
            "run_dir",
            "cv",
            "present",
            "count",
            "time_first_ps",
            "time_last_ps",
            "rel_time_last_ps",
            "first",
            "last",
            "min",
            "max",
            "mean",
            "median",
            "std",
            "span",
            "delta",
        ],
    )
    write_csv(long_path, long, ["case", "run_dir", "time_ps", "time_rel_ps", "cv", "value"])
    write_common_columns(common_path, cases)

    figure_dir = output_dir / "figures"
    figures: list[Path] = []
    for column in cv_columns:
        stem = sanitize_name(column)
        overlay = figure_dir / f"{stem}_overlay.png"
        delta = figure_dir / f"{stem}_delta_overlay.png"
        if plot_cv_overlay(cases, column, overlay, dpi=args.dpi):
            figures.append(overlay)
        if plot_cv_overlay(cases, column, delta, dpi=args.dpi, normalize_delta=True):
            figures.append(delta)
    panel = figure_dir / "sphere_cv_overlay_panel.png"
    if plot_panel(cases, cv_columns, panel, dpi=args.dpi):
        figures.append(panel)
    bars = figure_dir / "sphere_cv_last_values.png"
    if plot_last_value_bars(summary, cv_columns, bars, dpi=args.dpi):
        figures.append(bars)

    write_report(report_path, cases, cv_columns, summary, figures)
    return {
        "output_dir": output_dir,
        "summary": summary_path,
        "long": long_path,
        "columns": common_path,
        "segments": segment_path,
        "report": report_path,
        "figure_dir": figure_dir,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = run(args)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
