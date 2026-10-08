"""Summarize accepted TPCL detector sensitivity results from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.postprocess.tpcl_pinning_slip import _configure_matplotlib, _save_figure


@dataclass(frozen=True)
class SensitivitySource:
    parameter_id: str
    case_id: str
    contact_cutoff_A: float
    arc_bins: int
    source_stage: str


def read_sources(path: Path) -> list[SensitivitySource]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "parameter_id",
        "case_id",
        "contact_cutoff_A",
        "arc_bins",
        "source_stage",
    }
    if not rows or required.difference(rows[0]):
        raise ValueError(f"{path}: missing sensitivity source rows or columns")
    sources = [
        SensitivitySource(
            parameter_id=row["parameter_id"],
            case_id=row["case_id"],
            contact_cutoff_A=float(row["contact_cutoff_A"]),
            arc_bins=int(row["arc_bins"]),
            source_stage=row["source_stage"],
        )
        for row in rows
    ]
    if any(source.contact_cutoff_A <= 0 or source.arc_bins < 8 for source in sources):
        raise ValueError(f"{path}: invalid cutoff or arc count")
    keys = {(source.parameter_id, source.case_id) for source in sources}
    if len(keys) != len(sources):
        raise ValueError(f"{path}: parameter/case pairs must be unique")
    return sources


def load_sources(sources: Sequence[SensitivitySource], analysis_root: Path) -> list[dict]:
    rows = []
    for source in sources:
        latest = Path(analysis_root) / "latest" / source.source_stage / source.case_id
        if not latest.is_symlink():
            raise ValueError(f"{source.parameter_id}/{source.case_id}: missing accepted latest link")
        run_root = latest.resolve(strict=True)
        if run_root.parent.name != "run":
            raise ValueError(f"{source.parameter_id}/{source.case_id}: latest target is not run/<jobid>")
        result = run_root / "ANALYSIS-RESULT.txt"
        if not result.is_file() or "status=PASS" not in result.read_text(encoding="utf-8"):
            raise ValueError(f"{source.parameter_id}/{source.case_id}: upstream analysis is not PASS")
        summary = json.loads((run_root / "results" / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_root / "inputs" / "config.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise ValueError(f"{source.parameter_id}/{source.case_id}: upstream summary is not PASS")
        if not math.isclose(float(config["contact_cutoff_A"]), source.contact_cutoff_A):
            raise ValueError(f"{source.parameter_id}/{source.case_id}: cutoff mismatch")
        if int(config["arc_bins"]) != source.arc_bins:
            raise ValueError(f"{source.parameter_id}/{source.case_id}: arc-bin mismatch")
        rows.append(
            {
                "parameter_id": source.parameter_id,
                "case_id": source.case_id,
                "source_stage": source.source_stage,
                "job_id": run_root.name,
                "contact_cutoff_A": source.contact_cutoff_A,
                "arc_bins": source.arc_bins,
                "raw_frames": int(summary["raw_frames"]),
                "valid_contour_frames": int(summary["valid_contour_frames"]),
                "contour_valid_fraction": float(summary["contour_valid_fraction"]),
                "candidate_event_count": int(summary["candidate_event_count"]),
                "candidate_arc_record_count": int(summary["candidate_arc_record_count"]),
                "scientific_classification": summary["scientific_classification"],
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _matrix(rows: Sequence[dict], field: str, parameters: Sequence[str], cases: Sequence[str]) -> np.ndarray:
    lookup = {(row["parameter_id"], row["case_id"]): float(row[field]) for row in rows}
    return np.asarray([[lookup.get((parameter, case), math.nan) for case in cases] for parameter in parameters])


def write_figures(rows: Sequence[dict], output: Path, font_path: Path) -> None:
    _configure_matplotlib(font_path)
    from matplotlib import pyplot as plt

    figures = Path(output) / "figures"
    figures.mkdir()
    parameters = sorted({row["parameter_id"] for row in rows})
    cases = sorted({row["case_id"] for row in rows})
    labels = {
        parameter: next(
            f"{row['contact_cutoff_A']:g} A; {row['arc_bins']} arcs"
            for row in rows
            if row["parameter_id"] == parameter
        )
        for parameter in parameters
    }
    for filename, field, title in (
        ("01_candidate_event_count", "candidate_event_count", "Repeated candidate dwell--jump clusters"),
        ("02_contour_valid_fraction", "contour_valid_fraction", "Valid dynamic-contour fraction"),
    ):
        matrix = _matrix(rows, field, parameters, cases)
        figure, axis = plt.subplots(figsize=(max(6.5, 1.35 * len(cases)), max(3.5, 0.7 * len(parameters))))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(cases)), cases, rotation=25, ha="right")
        axis.set_yticks(range(len(parameters)), [labels[parameter] for parameter in parameters])
        axis.set_xlabel("Case")
        axis.set_ylabel("Pre-registered detector setting")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text = "NA" if not math.isfinite(value) else f"{value:g}"
                axis.text(column_index, row_index, text, ha="center", va="center", color="white" if value < np.nanmean(matrix) else "black")
        figure.colorbar(image, ax=axis, shrink=0.85)
        figure.tight_layout()
        _save_figure(figure, figures / filename)
        plt.close(figure)


def write_report(path: Path, rows: Sequence[dict]) -> None:
    parameters = sorted({row["parameter_id"] for row in rows})
    cases = sorted({row["case_id"] for row in rows})
    events = sum(int(row["candidate_event_count"]) for row in rows)
    all_valid = all(math.isclose(float(row["contour_valid_fraction"]), 1.0) for row in rows)
    lines = [
        "# TPCL sensitivity-matrix report",
        "",
        f"Accepted cells: {len(rows)} ({len(parameters)} settings x {len(cases)} cases).",
        "",
        f"Repeated candidate dwell--jump clusters across the matrix: {events}.",
        f"All dynamic contours valid: {all_valid}.",
        "",
        "This is a robustness report for the pre-registered detector grid. It does not establish a free-energy barrier, causal mechanism, or the absence of sub-resolution transitions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summary(
    sources_path: Path, analysis_root: Path, output_dir: Path, *, font_path: Path
) -> dict:
    sources = read_sources(sources_path)
    rows = load_sources(sources, analysis_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "sensitivity_summary.csv", rows)
    write_figures(rows, output, font_path)
    write_report(output / "report.md", rows)
    summary = {
        "status": "PASS",
        "accepted_cell_count": len(rows),
        "parameter_count": len({row["parameter_id"] for row in rows}),
        "case_count": len({row["case_id"] for row in rows}),
        "candidate_event_count": sum(int(row["candidate_event_count"]) for row in rows),
        "scientific_boundary": "detector robustness only; no causal or free-energy claim",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_summary(args.sources, args.analysis_root, args.output_dir, font_path=args.font_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
