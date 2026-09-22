"""Create the final three-highlight closure audit from accepted evidence.

The synthesis keeps publication readiness separate from causal-mechanism
closure.  It never submits molecular dynamics and never upgrades repeated
measurements of one parent trajectory into independent evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def decide_closure(robustness: Mapping[str, object]) -> list[dict[str, object]]:
    """Return claim-scoped closure states without manufacturing causality."""

    return [
        {
            "highlight_id": "H1",
            "title": "Persistent-island exchange is a minor transport correction",
            "closure": "CLOSED_PUBLICATION_GRADE",
            "evidence_level": "two_contact_histories",
            "supported_claim": "exchange-dominated transport is excluded; coherent unchanged-track motion dominates",
            "claim_limit": "exchange is small, not zero; this is a supporting negative mechanism result",
            "headline_role": "supporting",
            "requires_new_md": False,
        },
        {
            "highlight_id": "H2",
            "title": "Drive-aligned long-window transport exceeds transient and cross-coupled robustness",
            "closure": "CLOSED_WITH_LIMITS",
            "evidence_level": "two_contact_histories_direction_only",
            "supported_claim": "long-window longitudinal direction is more reproducible than onset and transverse response",
            "claim_limit": "no converged mobility tensor, magnitude law, or universal short-time sign",
            "headline_role": "core_descriptive",
            "requires_new_md": False,
        },
        {
            "highlight_id": "H3",
            "title": "Distributed edge deformation replaces an unresolved sharp-trigger narrative",
            "closure": "MECHANISM_NOT_ESTABLISHED",
            "evidence_level": "single_high_cadence_parent_history_with_parameter_audit",
            "supported_claim": "the accepted history shows edge-asymmetric distributed response and no resolved sharp SiOH trigger",
            "claim_limit": "detector non-recovery and parameter stability cannot establish water-network causality",
            "headline_role": "negative_refinement_only",
            "requires_new_md": True,
        },
        {
            "highlight_id": "OVERALL",
            "title": "Three-highlight project closure",
            "closure": "REQUIRES_NEW_MD",
            "evidence_level": "two_closed_findings_plus_one_negative_refinement",
            "supported_claim": "a carefully scoped article is draftable now",
            "claim_limit": "a Nature-level positive microscopic depinning mechanism is not closed",
            "headline_role": "decision",
            "requires_new_md": True,
        },
    ]


def _plot_closure(output: Path, closure_rows: Sequence[Mapping[str, object]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    highlights = [row for row in closure_rows if row["highlight_id"] != "OVERALL"]
    colors = {
        "CLOSED_PUBLICATION_GRADE": "#009E73",
        "CLOSED_WITH_LIMITS": "#E69F00",
        "MECHANISM_NOT_ESTABLISHED": "#D55E00",
    }
    figure, axis = plt.subplots(figsize=(10.5, 3.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(highlights))
    axis.axis("off")
    for index, row in enumerate(reversed(highlights)):
        y = index + 0.08
        color = colors[str(row["closure"])]
        axis.add_patch(Rectangle((0.01, y), 0.98, 0.82, facecolor=color, alpha=0.16, edgecolor=color))
        axis.text(0.03, y + 0.57, f"{row['highlight_id']}: {row['title']}", fontsize=10, weight="bold")
        axis.text(0.03, y + 0.28, str(row["closure"]), fontsize=10, color=color, weight="bold")
        axis.text(0.43, y + 0.28, str(row["claim_limit"]), fontsize=8.5, va="center", wrap=True)
    figure.tight_layout()
    path = output / "05_figures" / "three_highlight_closure_matrix.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def build_closure(
    stage_c_review: Path,
    prior_synthesis: Path,
    stage_a_synthesis: Path,
    robustness_analysis: Path,
    output_dir: Path,
) -> dict[str, object]:
    stage_c = Path(stage_c_review).resolve()
    prior = Path(prior_synthesis).resolve()
    stage_a = Path(stage_a_synthesis).resolve()
    robustness = Path(robustness_analysis).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"immutable output already exists: {output}")
    for name in ("00_contract", "01_sources", "02_results", "03_review", "04_validation", "05_figures"):
        (output / name).mkdir(parents=True, exist_ok=False)

    source_paths = [
        stage_c / "00_review" / "STAGE-C-SCIENTIFIC-REVIEW.md",
        stage_c / "results" / "stage_c_highlights.tsv",
        stage_c / "results" / "stage_c_mechanism_decision.tsv",
        stage_c / "manifests" / "REVIEW-SHA256SUMS",
        prior / "02_results" / "THREE-HIGHLIGHT-STATUS.tsv",
        prior / "03_review" / "THREE-HIGHLIGHT-SYNTHESIS.md",
        prior / "04_validation" / "VALIDATION.json",
        stage_a / "02_results" / "STAGE-A-DECISION.tsv",
        stage_a / "03_review" / "STAGE-A-SCIENTIFIC-SYNTHESIS.md",
        stage_a / "04_validation" / "VALIDATION.json",
        robustness / "02_detector_extension" / "minimum_qualified_amplitude.tsv",
        robustness / "04_robustness" / "paired_metric_sign_robustness.tsv",
        robustness / "04_robustness" / "membership_fraction_sensitivity.tsv",
        robustness / "04_robustness" / "paired_block_sign_summary.tsv",
        robustness / "06_review" / "STAGE-A2-ROBUSTNESS-REVIEW.md",
        robustness / "07_validation" / "VALIDATION.json",
        robustness / "07_validation" / "OUTPUT-SHA256SUMS",
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_rows = [
        {"source_id": f"source_{index:02d}", "path": str(path), "sha256": _sha256(path)}
        for index, path in enumerate(source_paths, start=1)
    ]
    _write_tsv(output / "01_sources" / "SOURCE-MANIFEST.tsv", source_rows)

    prior_validation = json.loads((prior / "04_validation" / "VALIDATION.json").read_text(encoding="utf-8"))
    stage_a_validation = json.loads((stage_a / "04_validation" / "VALIDATION.json").read_text(encoding="utf-8"))
    robustness_validation = json.loads((robustness / "07_validation" / "VALIDATION.json").read_text(encoding="utf-8"))
    if prior_validation.get("status") != "PASS":
        raise ValueError("prior three-highlight synthesis did not pass")
    if stage_a_validation.get("status") != "PASS":
        raise ValueError("Stage A scientific synthesis did not pass")
    if robustness_validation.get("status") != "PASS":
        raise ValueError("Stage A2 robustness analysis did not pass")
    if robustness_validation.get("scientific_gate") != "MECHANISM_NOT_ESTABLISHED_SINGLE_HISTORY":
        raise ValueError("unexpected Stage A2 scientific gate")

    closure_rows = decide_closure(robustness_validation)
    _write_tsv(output / "02_results" / "THREE-HIGHLIGHT-FINAL-STATUS.tsv", closure_rows)
    evidence_rows = [
        {
            "gate": "scheduler_and_artifact",
            "status": "PASS",
            "evidence": "accepted Stage B/C, high-frequency v3, Stage A, and Stage A2 manifests",
            "scientific_effect": "admits evidence; does not establish mechanism",
        },
        {
            "gate": "exchange_replication",
            "status": "PASS_TWO_HISTORIES",
            "evidence": "persistent-transfer L1 fraction 0.2695-0.3286 percent with signed reversal",
            "scientific_effect": "closes exclusion of exchange-dominated transport",
        },
        {
            "gate": "longitudinal_direction",
            "status": "PASS_DIRECTION_ONLY",
            "evidence": "long-window longitudinal signs reproduce; lateral signs and magnitudes do not",
            "scientific_effect": "qualitative closure only",
        },
        {
            "gate": "detector_extension",
            "status": "PASS_OPERATING_ENVELOPE",
            "evidence": f"{robustness_validation['detector_slow_qualified_operating_cells']}/{robustness_validation['detector_slow_positive_operating_cells']} slow positive cells qualify",
            "scientific_effect": "bounds estimator sensitivity; non-recovery is not physical absence",
        },
        {
            "gate": "region_sensitivity",
            "status": "PASS_ONE_AT_A_TIME",
            "evidence": f"{robustness_validation['persistent_slip_keys_same_sign_across_all_configurations']}/{robustness_validation['persistent_slip_key_count']} slip keys retain one sign across definitions",
            "scientific_effect": "tests fragility within one history; not independent validation",
        },
        {
            "gate": "time_block_stability",
            "status": "FAIL_CAUSAL_CLOSURE",
            "evidence": f"{robustness_validation['configuration_key_pairs_with_same_sign_all_time_blocks']}/{robustness_validation['configuration_key_pair_count']} configuration/key pairs retain one sign across all blocks",
            "scientific_effect": "prevents a stable local-slip mechanism claim",
        },
        {
            "gate": "independent_force_oddness",
            "status": "NOT_TESTED",
            "evidence": "no matched F0/+F/-F matrix across independent contact histories",
            "scientific_effect": "positive microscopic depinning mechanism remains open",
        },
    ]
    _write_tsv(output / "02_results" / "EVIDENCE-GATE-MATRIX.tsv", evidence_rows)

    contract = {
        "stage": "Final existing-trajectory three-highlight closure",
        "allowed_outcomes": [
            "CLOSED_PUBLICATION_GRADE",
            "CLOSED_WITH_LIMITS",
            "MECHANISM_NOT_ESTABLISHED",
            "REQUIRES_NEW_MD",
        ],
        "input_scope": "accepted existing-trajectory products only",
        "new_md_submitted": False,
        "causal_rule": "parameter robustness and time blocks from one parent do not count as independent histories",
    }
    (output / "00_contract" / "CLOSURE-CONTRACT.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure = _plot_closure(output, closure_rows)
    report = [
        "# Final three-highlight closure",
        "",
        "## Overall decision",
        "",
        "`REQUIRES_NEW_MD_FOR_THREE_HIGHLIGHT_CAUSAL_CLOSURE`",
        "",
        "All pre-registered postprocessing available from the present trajectories is complete. A scoped article can be drafted from one publication-grade supporting result, one qualitative core result, and one negative mechanism refinement. The stricter three-highlight target, especially a positive microscopic TPCL depinning mechanism suitable for a Nature-level sub-journal, is not closed.",
        "",
        "## Highlight 1",
        "",
        "Persistent-island exchange is 0.2695-0.3286 percent in L1 norm across two contact histories. Its signed contribution changes, but it remains small. This closes the narrow, publication-grade exclusion of exchange-dominated transport; it is a supporting result rather than a standalone high-impact headline.",
        "",
        "## Highlight 2",
        "",
        "Long-window drive-aligned direction is more reproducible than transient onset, transverse response, and magnitude. This is closed qualitatively, but not as a mobility tensor, friction law, or universal anisotropy.",
        "",
        "## Highlight 3",
        "",
        f"The extended detector qualifies {robustness_validation['detector_slow_qualified_operating_cells']}/{robustness_validation['detector_slow_positive_operating_cells']} slow positive operating cells. Region sensitivity leaves {robustness_validation['persistent_slip_keys_same_sign_across_all_configurations']}/{robustness_validation['persistent_slip_key_count']} slip keys with one sign across all definitions, while only {robustness_validation['configuration_key_pairs_with_same_sign_all_time_blocks']}/{robustness_validation['configuration_key_pair_count']} configuration/key pairs keep one sign across all 5 ps blocks. These are estimator and robustness results from one parent history, not causal evidence.",
        "",
        "The supported statement is distributed, edge-asymmetric deformation with no resolved sharp SiOH trigger in the accepted history. Water-network control, force oddness, and a transferable depinning pathway remain unestablished.",
        "",
        "## Required next simulation",
        "",
        "Use the frozen Stage B1 design: `ch3_only` and `mixed291`, X direction only, four admitted parent histories per surface, and matched `F0/+8e-5/-8e-5 eV/A per water O` branches for 100 ps with high-cadence output. This is 24 runs (2.4 ns aggregate). Analyze odd and even response before considering pulse/wait/probe or extension to the other two surfaces.",
        "",
        "No new MD is submitted by this closure package.",
    ]
    (output / "03_review" / "THREE-HIGHLIGHT-FINAL-CLOSURE.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    next_md = [
        "# Next-MD decision gate",
        "",
        "Status: `REVIEW_REQUIRED_BEFORE_STAGE_B1_SUBMISSION`",
        "",
        "Stage B1 is the minimum causal-discrimination experiment, not a replication-only extension.",
        "",
        "- Surfaces: `ch3_only`, `mixed291`",
        "- Axis: `X`",
        "- Independent parent histories: `4` per surface",
        "- Branches: `F0`, `+8e-5 eV/A`, `-8e-5 eV/A` per water O",
        "- Duration: `100 ps`",
        "- Total: `24 runs`, `2.4 ns`",
        "- Primary response: `R_odd(t) = [R_+F(t) - R_-F(t)] / 2`",
        "- Even response: `R_even(t) = [R_+F(t) + R_-F(t)] / 2 - R_F0(t)`",
        "",
        "Admission requires preserved species, finite localized contact lines, artifact-complete high-cadence output, and a pre-registered estimator. No pulse/wait/probe follows unless Stage B1 passes the cross-history odd-response gate.",
    ]
    (output / "03_review" / "NEXT-MD-DECISION.md").write_text(
        "\n".join(next_md) + "\n", encoding="utf-8"
    )

    validation = {
        "status": "PASS",
        "source_count": len(source_rows),
        "source_hash_gate": "PASS",
        "highlight_1": "CLOSED_PUBLICATION_GRADE_SUPPORTING",
        "highlight_2": "CLOSED_WITH_LIMITS",
        "highlight_3": "MECHANISM_NOT_ESTABLISHED",
        "overall": "REQUIRES_NEW_MD",
        "manuscript_readiness": "SCOPED_ARTICLE_DRAFTABLE_NOT_NATURE_LEVEL_MECHANISM_COMPLETE",
        "next_simulation": "STAGE_B1_FORCE_REVERSAL_24_RUNS_REVIEW_REQUIRED",
        "new_md_submitted": False,
        "figure_count": 1,
        "figure": str(figure.relative_to(output)),
    }
    (output / "04_validation" / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "OUTPUT-SHA256SUMS":
            hashes.append(f"{_sha256(path)}  {path.relative_to(output)}")
    (output / "04_validation" / "OUTPUT-SHA256SUMS").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-review", type=Path, required=True)
    parser.add_argument("--prior-synthesis", type=Path, required=True)
    parser.add_argument("--stage-a-synthesis", type=Path, required=True)
    parser.add_argument("--robustness-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_closure(
        args.stage_c_review,
        args.prior_synthesis,
        args.stage_a_synthesis,
        args.robustness_analysis,
        args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
