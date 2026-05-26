"""CSV-driven plotting utilities.

The plotting layer is intentionally table-oriented.  It can consume outputs
from migrated post-processing commands without knowing any project-specific
case directory layout or publication panel design.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#6E6E6E",
    "#EECA3B",
)


@dataclass(frozen=True)
class HeatmapGrid:
    """Dense heatmap representation built from a long-form table."""

    row_labels: Tuple[str, ...]
    column_labels: Tuple[str, ...]
    values: np.ndarray


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a CSV file as dictionaries and return rows plus fieldnames."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader], [str(field) for field in reader.fieldnames]


def _as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _require_columns(fieldnames: Sequence[str], columns: Sequence[str], path: Path) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def ordered_unique(values: Iterable[object]) -> Tuple[str, ...]:
    """Return unique non-empty values in first-seen order."""

    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def group_rows(rows: Sequence[Mapping[str, object]], group_column: Optional[str] = None) -> Dict[str, List[Mapping[str, object]]]:
    """Group rows by a column, or return a single default group."""

    if group_column is None:
        return {"data": list(rows)}
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        key = str(row.get(group_column, "")).strip() or "unlabeled"
        grouped.setdefault(key, []).append(row)
    return grouped


def output_paths(output: Path, formats: Optional[Sequence[str]] = None) -> List[Path]:
    """Resolve output files from a target path and optional format list."""

    output = Path(output)
    if not formats:
        if output.suffix:
            return [output]
        return [output.with_suffix(".png")]
    paths: List[Path] = []
    for item in formats:
        suffix = str(item).strip().lower().lstrip(".")
        if not suffix:
            continue
        paths.append(output.with_suffix(f".{suffix}"))
    if not paths:
        raise ValueError("At least one non-empty output format is required")
    return paths


def build_heatmap_grid(
    rows: Sequence[Mapping[str, object]],
    row_column: str,
    column_column: str,
    value_column: str,
) -> HeatmapGrid:
    """Build a dense matrix from long-form rows."""

    row_labels = ordered_unique(row.get(row_column, "") for row in rows)
    column_labels = ordered_unique(row.get(column_column, "") for row in rows)
    matrix = np.full((len(row_labels), len(column_labels)), np.nan, dtype=float)
    row_index = {label: index for index, label in enumerate(row_labels)}
    column_index = {label: index for index, label in enumerate(column_labels)}
    for row in rows:
        row_label = str(row.get(row_column, "")).strip()
        column_label = str(row.get(column_column, "")).strip()
        if row_label not in row_index or column_label not in column_index:
            continue
        value = _as_float(row.get(value_column))
        if math.isfinite(value):
            matrix[row_index[row_label], column_index[column_label]] = value
    return HeatmapGrid(row_labels=row_labels, column_labels=column_labels, values=matrix)


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("matplotlib is required for plotting commands") from exc
    return plt


def _apply_style(plt, base_font: float = 7.2) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": base_font,
            "axes.labelsize": base_font,
            "axes.titlesize": base_font + 0.4,
            "xtick.labelsize": base_font - 0.5,
            "ytick.labelsize": base_font - 0.5,
            "legend.fontsize": base_font - 0.6,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, paths: Sequence[Path], dpi: int) -> List[Path]:
    saved: List[Path] = []
    for path in paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".pdf", ".svg"}:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(path)
    return saved


def plot_line_table(
    input_csv: Path,
    output: Path,
    x_column: str,
    y_column: str,
    group_column: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    formats: Optional[Sequence[str]] = None,
    width: float = 3.4,
    height: float = 2.4,
    dpi: int = 300,
) -> List[Path]:
    """Plot one or more line series from a CSV table."""

    rows, fieldnames = read_csv_rows(input_csv)
    required = [x_column, y_column] + ([group_column] if group_column is not None else [])
    _require_columns(fieldnames, required, Path(input_csv))
    plt = _load_pyplot()
    _apply_style(plt)
    fig, ax = plt.subplots(figsize=(width, height))
    grouped = group_rows(rows, group_column)
    for index, (label, group) in enumerate(grouped.items()):
        points = [(_as_float(row.get(x_column)), _as_float(row.get(y_column))) for row in group]
        points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        points.sort(key=lambda item: item[0])
        if not points:
            continue
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            ms=3.0,
            lw=1.2,
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
            label=label if group_column is not None else None,
        )
    if group_column is not None and grouped:
        ax.legend(frameon=False)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or y_column)
    if title:
        ax.set_title(title)
    _despine(ax)
    fig.tight_layout()
    saved = _save_figure(fig, output_paths(output, formats), dpi=dpi)
    plt.close(fig)
    return saved


def plot_scatter_table(
    input_csv: Path,
    output: Path,
    x_column: str,
    y_column: str,
    group_column: Optional[str] = None,
    label_column: Optional[str] = None,
    fit_line: bool = False,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    formats: Optional[Sequence[str]] = None,
    width: float = 3.2,
    height: float = 2.6,
    dpi: int = 300,
) -> List[Path]:
    """Plot a scatter table, optionally grouped and linearly fitted."""

    rows, fieldnames = read_csv_rows(input_csv)
    required = [x_column, y_column]
    for optional in (group_column, label_column):
        if optional is not None:
            required.append(optional)
    _require_columns(fieldnames, required, Path(input_csv))
    plt = _load_pyplot()
    _apply_style(plt)
    fig, ax = plt.subplots(figsize=(width, height))
    grouped = group_rows(rows, group_column)
    all_points: List[Tuple[float, float]] = []
    for index, (label, group) in enumerate(grouped.items()):
        x_values: List[float] = []
        y_values: List[float] = []
        point_labels: List[str] = []
        for row in group:
            x = _as_float(row.get(x_column))
            y = _as_float(row.get(y_column))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            x_values.append(x)
            y_values.append(y)
            point_labels.append(str(row.get(label_column, "")).strip() if label_column is not None else "")
            all_points.append((x, y))
        if not x_values:
            continue
        ax.scatter(
            x_values,
            y_values,
            s=28,
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
            edgecolor="white",
            linewidth=0.45,
            label=label if group_column is not None else None,
            zorder=3,
        )
        if label_column is not None:
            for x, y, text in zip(x_values, y_values, point_labels):
                if text:
                    ax.text(x, y, " " + text, fontsize=6.2, va="center")
    if fit_line and len(all_points) >= 2:
        x = np.asarray([point[0] for point in all_points], dtype=float)
        y = np.asarray([point[1] for point in all_points], dtype=float)
        if float(np.std(x)) > 0.0:
            slope, intercept = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, slope * xs + intercept, color="#222222", lw=0.9, alpha=0.75, zorder=2)
    if group_column is not None and grouped:
        ax.legend(frameon=False)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or y_column)
    if title:
        ax.set_title(title)
    _despine(ax)
    fig.tight_layout()
    saved = _save_figure(fig, output_paths(output, formats), dpi=dpi)
    plt.close(fig)
    return saved


def plot_heatmap_table(
    input_csv: Path,
    output: Path,
    row_column: str,
    column_column: str,
    value_column: str,
    title: str = "",
    colorbar_label: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    formats: Optional[Sequence[str]] = None,
    width: float = 4.0,
    height: float = 3.0,
    dpi: int = 300,
) -> List[Path]:
    """Plot a heatmap from long-form CSV rows."""

    rows, fieldnames = read_csv_rows(input_csv)
    _require_columns(fieldnames, [row_column, column_column, value_column], Path(input_csv))
    grid = build_heatmap_grid(rows, row_column=row_column, column_column=column_column, value_column=value_column)
    plt = _load_pyplot()
    _apply_style(plt)
    fig, ax = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(grid.values)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(grid.column_labels)))
    ax.set_yticks(np.arange(len(grid.row_labels)))
    ax.set_xticklabels(grid.column_labels, rotation=45, ha="right")
    ax.set_yticklabels(grid.row_labels)
    ax.set_xlabel(column_column)
    ax.set_ylabel(row_column)
    if title:
        ax.set_title(title)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    fig.tight_layout()
    saved = _save_figure(fig, output_paths(output, formats), dpi=dpi)
    plt.close(fig)
    return saved


def _add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input CSV table")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path or stem")
    parser.add_argument("--format", dest="formats", action="append", help="Output format; may be repeated")
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSV-driven plotting helpers")
    subparsers = parser.add_subparsers(dest="plot_kind", required=True)

    line = subparsers.add_parser("line", help="Plot line series from a CSV table")
    _add_common_output_args(line)
    line.add_argument("--x-column", required=True)
    line.add_argument("--y-column", required=True)
    line.add_argument("--group-column")
    line.add_argument("--x-label", default="")
    line.add_argument("--y-label", default="")

    scatter = subparsers.add_parser("scatter", help="Plot a scatter table")
    _add_common_output_args(scatter)
    scatter.add_argument("--x-column", required=True)
    scatter.add_argument("--y-column", required=True)
    scatter.add_argument("--group-column")
    scatter.add_argument("--label-column")
    scatter.add_argument("--fit-line", action="store_true")
    scatter.add_argument("--x-label", default="")
    scatter.add_argument("--y-label", default="")

    heatmap = subparsers.add_parser("heatmap", help="Plot a heatmap from long-form CSV rows")
    _add_common_output_args(heatmap)
    heatmap.add_argument("--row-column", required=True)
    heatmap.add_argument("--column-column", required=True)
    heatmap.add_argument("--value-column", required=True)
    heatmap.add_argument("--colorbar-label", default="")
    heatmap.add_argument("--cmap", default="RdBu_r")
    heatmap.add_argument("--vmin", type=float)
    heatmap.add_argument("--vmax", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.plot_kind == "line":
            outputs = plot_line_table(
                input_csv=args.input,
                output=args.output,
                x_column=args.x_column,
                y_column=args.y_column,
                group_column=args.group_column,
                title=args.title,
                x_label=args.x_label,
                y_label=args.y_label,
                formats=args.formats,
                width=args.width or 3.4,
                height=args.height or 2.4,
                dpi=args.dpi,
            )
        elif args.plot_kind == "scatter":
            outputs = plot_scatter_table(
                input_csv=args.input,
                output=args.output,
                x_column=args.x_column,
                y_column=args.y_column,
                group_column=args.group_column,
                label_column=args.label_column,
                fit_line=args.fit_line,
                title=args.title,
                x_label=args.x_label,
                y_label=args.y_label,
                formats=args.formats,
                width=args.width or 3.2,
                height=args.height or 2.6,
                dpi=args.dpi,
            )
        elif args.plot_kind == "heatmap":
            outputs = plot_heatmap_table(
                input_csv=args.input,
                output=args.output,
                row_column=args.row_column,
                column_column=args.column_column,
                value_column=args.value_column,
                title=args.title,
                colorbar_label=args.colorbar_label,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                formats=args.formats,
                width=args.width or 4.0,
                height=args.height or 3.0,
                dpi=args.dpi,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown plot kind: {args.plot_kind}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")
        return 1

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
