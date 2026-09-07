from __future__ import annotations

import csv
import json
from pathlib import Path

from molsimflow.postprocess import tpcl_turnover_prediction_plot as plot


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_review_plot_validates_frozen_wp16_dimensions(tmp_path: Path) -> None:
    cases = ["oh_only", "ch3_only", "mixed_ch3_oh_natoms291", "mixed_ch3_oh_natoms275"]
    scores = []
    for evaluation in ("within_case", "leave_one_case_out"):
        for case in cases:
            for model in ("M0_static", "M1_occupancy_history", "M2_turnover_history"):
                scores.append({"evaluation": evaluation, "held_case": case, "model": model, "weighted_log_loss": 0.4, "weighted_roc_auc": 0.6, "weighted_brier": 0.2})
    evidence = []
    for evaluation in ("within_case", "leave_one_case_out"):
        for case in cases:
            for comparison in ("M0_static_to_M1_occupancy_history", plot.PRIMARY_COMPARISON):
                evidence.append({"evaluation": evaluation, "held_case": case, "comparison": comparison, "delta_weighted_log_loss": 0.01, "bootstrap_ci025": -0.01, "bootstrap_ci975": 0.02, "bh_q": 0.5, "qualified_incremental_turnover_information": "0"})
    coverage = [{"case_id": case, "risk_anchor_count": 100, "complete_history_anchor_count": 90, "incomplete_history_anchor_count": 10} for case in cases]
    scores_path, evidence_path, coverage_path = tmp_path / "scores.csv", tmp_path / "evidence.csv", tmp_path / "coverage.csv"
    _write(scores_path, scores); _write(evidence_path, evidence); _write(coverage_path, coverage)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status": "PASS", "qualified_primary_count": 0}), encoding="utf-8")
    output = tmp_path / "review"
    result = plot.run(
        plot.build_parser().parse_args(
            [
                "--scores",
                str(scores_path),
                "--evidence",
                str(evidence_path),
                "--coverage",
                str(coverage_path),
                "--summary",
                str(summary),
                "--output-dir",
                str(output),
            ]
        )
    )
    assert result["status"] == "PASS"
    assert result["qualified_primary_count"] == 0
    assert {path.name for path in (output / "figures").glob("*.png")} == {
        "01_incremental_turnover_evidence.png",
        "02_prediction_scores.png",
        "03_membership_history_coverage.png",
    }


def test_review_plot_rejects_mismatched_qualification_count(tmp_path: Path) -> None:
    test_review_plot_validates_frozen_wp16_dimensions(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status": "PASS", "qualified_primary_count": 1}), encoding="utf-8")
    try:
        plot.run(plot.build_parser().parse_args(["--scores", str(tmp_path / "scores.csv"), "--evidence", str(tmp_path / "evidence.csv"), "--coverage", str(tmp_path / "coverage.csv"), "--summary", str(summary), "--output-dir", str(tmp_path / "second"), "--no-plots"]))
    except ValueError as error:
        assert "qualification count" in str(error)
    else:
        raise AssertionError("mismatched qualification count was accepted")


def test_numeric_qualification_flag_is_accepted() -> None:
    assert plot._flag("1") is True
    assert plot._flag("0") is False
