"""Render a review-only summary of blocked water-turnover prediction outputs.

The renderer is deliberately descriptive: it visualizes held-out score changes,
their specified resampling intervals, and history coverage.  It does not infer
causality, physical rates, or replicate-level uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

PRIMARY_COMPARISON = "M1_occupancy_history_to_M2_turnover_history"
REQUIRED_SCORE_COLUMNS = {"evaluation", "held_case", "model", "weighted_log_loss", "weighted_roc_auc", "weighted_brier"}
REQUIRED_EVIDENCE_COLUMNS = {"evaluation", "held_case", "comparison", "delta_weighted_log_loss", "bootstrap_ci025", "bootstrap_ci975", "bh_q", "qualified_incremental_turnover_information"}
REQUIRED_COVERAGE_COLUMNS = {"case_id", "risk_anchor_count", "complete_history_anchor_count", "incomplete_history_anchor_count"}


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty table")
    return rows


def _number(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite {key}")
    return value


def _flag(value: str) -> bool:
    if value in {"True", "1"}:
        return True
    if value in {"False", "0"}:
        return False
    raise ValueError(f"invalid boolean flag: {value}")


def _configure_matplotlib(font_path: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        from matplotlib import font_manager

        font_manager.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()


def _plot_incremental_evidence(rows: list[dict[str, str]], output: Path) -> None:
    from matplotlib import pyplot as plt

    selected = [row for row in rows if row["comparison"] == PRIMARY_COMPARISON]
    if len(selected) != 8:
        raise ValueError("expected exactly eight primary incremental-turnover comparisons")
    selected.sort(key=lambda row: (row["evaluation"], row["held_case"]))
    labels = [f"{row['evaluation']}\n{row['held_case']}" for row in selected]
    values = [_number(row, "delta_weighted_log_loss", output) for row in selected]
    low = [_number(row, "bootstrap_ci025", output) for row in selected]
    high = [_number(row, "bootstrap_ci975", output) for row in selected]
    color = ["#4C78A8" if row["evaluation"] == "within_case" else "#F58518" for row in selected]
    x = list(range(len(selected)))
    figure, axis = plt.subplots(figsize=(10.0, 4.4))
    axis.errorbar(x, values, yerr=[[value - lower for value, lower in zip(values, low)], [upper - value for value, upper in zip(values, high)]], fmt="none", color="#555555", capsize=3, zorder=1)
    axis.scatter(x, values, s=48, c=color, zorder=2)
    for index, row in enumerate(selected):
        if _flag(row["qualified_incremental_turnover_information"]):
            axis.annotate("qualified", (index, values[index]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set(xticks=x, xticklabels=labels, ylabel="M2 minus M1 improvement in weighted log loss", title="Incremental retrospective information from turnover history")
    axis.tick_params(axis="x", labelrotation=28)
    axis.text(0.01, -0.28, "Intervals are specified blocked-bootstrap intervals; they are not replicate confidence intervals.", transform=axis.transAxes, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "01_incremental_turnover_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_scores(rows: list[dict[str, str]], output: Path) -> None:
    from matplotlib import pyplot as plt

    models = ["M0_static", "M1_occupancy_history", "M2_turnover_history"]
    evaluations = sorted({row["evaluation"] for row in rows})
    cases = sorted({row["held_case"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    for evaluation, color in zip(evaluations, ["#4C78A8", "#F58518"]):
        selected = [row for row in rows if row["evaluation"] == evaluation]
        for model, marker in zip(models, ["o", "s", "^"]):
            values = []
            for case in cases:
                match = [row for row in selected if row["held_case"] == case and row["model"] == model]
                if len(match) != 1:
                    raise ValueError(f"missing score for {evaluation}, {case}, {model}")
                values.append(_number(match[0], "weighted_log_loss", output))
            axes[0].plot(cases, values, marker=marker, color=color, alpha=0.78, label=f"{evaluation}: {model}")
            values = [_number(next(row for row in selected if row["held_case"] == case and row["model"] == model), "weighted_roc_auc", output) for case in cases]
            axes[1].plot(cases, values, marker=marker, color=color, alpha=0.78, label=f"{evaluation}: {model}")
    axes[0].set_ylabel("Weighted log loss (lower is better)")
    axes[1].set_ylabel("Weighted ROC AUC")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=28)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=6.4, ncol=2)
    figure.suptitle("Blocked within-case and leave-one-case-out prediction scores", y=1.02, fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "02_prediction_scores.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_coverage(rows: list[dict[str, str]], output: Path) -> None:
    from matplotlib import pyplot as plt

    rows.sort(key=lambda row: row["case_id"])
    cases = [row["case_id"] for row in rows]
    complete = [int(row["complete_history_anchor_count"]) for row in rows]
    incomplete = [int(row["incomplete_history_anchor_count"]) for row in rows]
    total = [int(row["risk_anchor_count"]) for row in rows]
    if any(left + right != whole for left, right, whole in zip(complete, incomplete, total)):
        raise ValueError("membership history coverage does not close")
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.bar(cases, complete, label="complete 5 ps history", color="#54A24B")
    axis.bar(cases, incomplete, bottom=complete, label="incomplete history; excluded", color="#BAB0AC")
    axis.set(ylabel="Risk anchors", title="Water-turnover history coverage")
    axis.tick_params(axis="x", labelrotation=28)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "03_membership_history_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scores_path, evidence_path, coverage_path = Path(args.scores), Path(args.evidence), Path(args.coverage)
    scores = _read_csv(scores_path, REQUIRED_SCORE_COLUMNS)
    evidence = _read_csv(evidence_path, REQUIRED_EVIDENCE_COLUMNS)
    coverage = _read_csv(coverage_path, REQUIRED_COVERAGE_COLUMNS)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError("prediction summary is not PASS")
    if len(scores) != 24 or len(evidence) != 16 or len(coverage) != 4:
        raise ValueError("unexpected frozen WP16 result dimensions")
    primary = [row for row in evidence if row["comparison"] == PRIMARY_COMPARISON]
    qualification_count = sum(_flag(row["qualified_incremental_turnover_information"]) for row in primary)
    if qualification_count != int(summary["qualified_primary_count"]):
        raise ValueError("qualification count differs from WP16 summary")
    if not args.no_plots:
        _configure_matplotlib(args.font_path)
        figures = output / "figures"
        figures.mkdir()
        _plot_incremental_evidence(evidence, figures)
        _plot_scores(scores, figures)
        _plot_coverage(coverage, figures)
    validation = {
        "status": "PASS",
        "case_count": 4,
        "score_rows": len(scores),
        "evidence_rows": len(evidence),
        "qualified_primary_count": qualification_count,
        "claim_boundary": [
            "This is a retrospective blocked-prediction visualization, not causal evidence.",
            "Intervals describe the specified time-block bootstrap and are not replicate uncertainty.",
            "A qualified result would be incremental predictive information, not a physical rate, free energy, friction, or universal mechanism.",
        ],
    }
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
