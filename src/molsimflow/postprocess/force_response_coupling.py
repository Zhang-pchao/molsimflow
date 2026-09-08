"""Join paired snapshot-force contrasts to continuous event-response outcomes."""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import rankdata

from molsimflow.postprocess.tpcl_snapshot_contrasts import (
    _sha256,
    _stable_seed,
    _write_csv,
)

SCIENTIFIC_STATUS = (
    "RESPONSE_STRATIFIED_SINGLE_TRAJECTORY_FORCE_RESPONSE_ASSOCIATION_"
    "NOT_CAUSAL_LOCAL_STRESS_POPULATION_EFFECT_OR_REPLICATE_EVIDENCE"
)
RESPONSE_COORDINATES = (
    "far_mobilized_fraction",
    "far_mean_mobilization_ratio",
    "far_fast_conversion_fraction",
)
SUPPORT_EXCLUSION_FIELDS = (
    "pair_id",
    "case_id",
    "primary_event_id",
    "source_time_block_200ps",
    "response_affected_arc_fraction",
    "exclusion_reason",
    "scientific_status",
)


def _read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_support_exclusions(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    """Write a header-only ledger when every selected pair has outcome support."""

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUPPORT_EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _boolean(raw: object) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value {raw!r}")


def parse_mechanical_coordinates(raw: Sequence[str]) -> tuple[tuple[str, str, str], ...]:
    """Parse repeated ``LABEL=METRIC,CONTRAST`` specifications."""

    output = []
    for item in raw:
        label, separator, remainder = item.partition("=")
        fields = remainder.split(",")
        if not separator or not label or len(fields) != 2 or not all(fields):
            raise ValueError(f"invalid mechanical coordinate {item!r}")
        output.append((label, fields[0], fields[1]))
    labels = [item[0] for item in output]
    if not output or len(labels) != len(set(labels)):
        raise ValueError("mechanical coordinates must be nonempty and uniquely labeled")
    return tuple(output)


def _selection_index(path: Path) -> dict[str, dict[str, object]]:
    rows = _read_csv(path, delimiter="\t")
    required = {
        "pair_id",
        "case_id",
        "primary_event_id",
        "source_time_block_200ps",
        "response_affected_arc_fraction",
    }
    if missing := required.difference(rows[0]):
        raise ValueError(f"selection table is missing columns: {sorted(missing)}")
    grouped: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    for row in rows:
        grouped[row["pair_id"]].add(
            (
                row["case_id"],
                int(row["primary_event_id"]),
                row["source_time_block_200ps"],
                float(row["response_affected_arc_fraction"]),
            )
        )
    output = {}
    for pair_id, identities in grouped.items():
        if len(identities) != 1:
            raise ValueError(f"selection metadata differ within pair {pair_id}")
        case_id, event_id, block, affected = identities.pop()
        output[pair_id] = {
            "pair_id": pair_id,
            "case_id": case_id,
            "primary_event_id": event_id,
            "source_time_block_200ps": block,
            "response_affected_arc_fraction": affected,
        }
    case_pairs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in output.values():
        case_pairs[str(row["case_id"])].append(row)
    for case_id, pairs in case_pairs.items():
        blocks = [str(row["source_time_block_200ps"]) for row in pairs]
        events = [int(row["primary_event_id"]) for row in pairs]
        if len(blocks) != len(set(blocks)) or len(events) != len(set(events)):
            raise ValueError(f"selection reuses a block or event in {case_id}")
    return output


def _mechanics_index(
    path: Path,
    coordinates: Sequence[tuple[str, str, str]],
    patch_radius_A: float,
    expected_cases: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    required = {"pair_id", "case_id", "patch_radius_A", "metric", "contrast", "value"}
    if missing := required.difference(rows[0]):
        raise ValueError(f"mechanics table is missing columns: {sorted(missing)}")
    wanted = {(metric, contrast): label for label, metric, contrast in coordinates}
    output: dict[str, dict[str, float]] = defaultdict(dict)
    pair_cases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not math.isclose(
            float(row["patch_radius_A"]), patch_radius_A, rel_tol=0.0, abs_tol=1.0e-12
        ):
            continue
        key = (row["metric"], row["contrast"])
        if key not in wanted:
            continue
        pair_id, label = row["pair_id"], wanted[key]
        if label in output[pair_id]:
            raise ValueError(f"duplicate mechanics coordinate {(pair_id, label)}")
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite mechanics coordinate {(pair_id, label)}")
        output[pair_id][label] = value
        pair_cases[pair_id].add(row["case_id"])
    labels = {item[0] for item in coordinates}
    for pair_id, values in output.items():
        if set(values) != labels or len(pair_cases[pair_id]) != 1:
            raise ValueError(f"incomplete or inconsistent mechanics pair {pair_id}")
        if (
            pair_id in expected_cases
            and next(iter(pair_cases[pair_id])) != expected_cases[pair_id]
        ):
            raise ValueError(f"mechanics/selection case mismatch for pair {pair_id}")
    if not output:
        raise ValueError("no mechanics rows survive the frozen selection")
    return dict(output)


def _outcome_index(
    path: Path, distance_label: str, window_label: str
) -> dict[tuple[str, int], dict[str, object]]:
    rows = _read_csv(path)
    required = {
        "case_id",
        "primary_event_id",
        "time_block_200ps",
        "target_arc_index",
        "distance_bin",
        "window",
        "mobilization_ratio",
        "mobilized",
        "secondary_event_detected",
    }
    if missing := required.difference(rows[0]):
        raise ValueError(f"outcome table is missing columns: {sorted(missing)}")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["distance_bin"] == distance_label and row["window"] == window_label:
            grouped[(row["case_id"], int(row["primary_event_id"]))].append(row)
    if not grouped:
        raise ValueError("no event outcomes survive the frozen distance/window selection")
    output = {}
    for key, group in grouped.items():
        targets = [int(row["target_arc_index"]) for row in group]
        blocks = {row["time_block_200ps"] for row in group}
        if len(targets) != len(set(targets)) or len(blocks) != 1:
            raise ValueError(f"duplicate target or inconsistent block for event {key}")
        ratios = np.asarray([float(row["mobilization_ratio"]) for row in group], dtype=float)
        if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
            raise ValueError(f"invalid mobilization ratios for event {key}")
        mobilized = np.asarray([_boolean(row["mobilized"]) for row in group], dtype=bool)
        secondary = np.asarray(
            [_boolean(row["secondary_event_detected"]) for row in group], dtype=bool
        )
        converted = int(np.count_nonzero(mobilized & secondary))
        mobilized_count = int(np.count_nonzero(mobilized))
        output[key] = {
            "time_block_200ps": blocks.pop(),
            "eligible_far_target_count": len(group),
            "mobilized_far_target_count": mobilized_count,
            "converted_mobilized_far_target_count": converted,
            "far_mobilized_fraction": mobilized_count / len(group),
            "far_mean_mobilization_ratio": float(np.mean(ratios)),
            "far_fast_conversion_fraction": converted / mobilized_count
            if mobilized_count
            else math.nan,
            "far_fast_any_conversion": int(converted > 0),
        }
    return output


def _join(
    selection: Mapping[str, Mapping[str, object]],
    mechanics: Mapping[str, Mapping[str, float]],
    outcomes: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if set(mechanics) != set(selection):
        missing = sorted(set(selection).difference(mechanics))
        extra = sorted(set(mechanics).difference(selection))
        raise ValueError(f"mechanics/selection pair support differs: missing={missing}, extra={extra}")
    admitted, excluded = [], []
    for pair_id, selected in sorted(selection.items()):
        event_key = (str(selected["case_id"]), int(selected["primary_event_id"]))
        if event_key not in outcomes:
            excluded.append(
                {
                    **selected,
                    "exclusion_reason": "absent_from_accepted_event_outcome_support",
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
            continue
        outcome = outcomes[event_key]
        if str(outcome["time_block_200ps"]) != str(selected["source_time_block_200ps"]):
            raise ValueError(f"selection/outcome block mismatch for pair {pair_id}")
        admitted.append(
            {
                **selected,
                **mechanics[pair_id],
                **outcome,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    if not admitted:
        raise ValueError("no force-response pairs survive the joint-support gate")
    return admitted, excluded


def _rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return math.nan
    left, right = rankdata(x).astype(float), rankdata(y).astype(float)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else math.nan


def _stratified_rank_correlation(
    rows: Sequence[Mapping[str, object]], mechanical: str, response: str
) -> float:
    centered_left, centered_right = [], []
    by_case: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(float(row[mechanical])) and math.isfinite(float(row[response])):
            by_case[str(row["case_id"])].append(row)
    for group in by_case.values():
        left = rankdata([float(row[mechanical]) for row in group]).astype(float)
        right = rankdata([float(row[response]) for row in group]).astype(float)
        centered_left.extend(left - np.mean(left))
        centered_right.extend(right - np.mean(right))
    denominator = float(
        np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
    )
    if denominator <= 0.0:
        return math.nan
    return float(np.dot(centered_left, centered_right) / denominator)


def _primary_inference(
    rows: Sequence[Mapping[str, object]],
    mechanical: str,
    response: str,
    permutation_draws: int,
    bootstrap_draws: int,
    seed: int,
) -> tuple[float, float, float, int]:
    observed = _stratified_rank_correlation(rows, mechanical, response)
    if not math.isfinite(observed):
        return math.nan, math.nan, math.nan, 0
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(dict(row))
    generator = np.random.default_rng(_stable_seed(seed, mechanical, response, "permutation"))
    exceed = 0
    for _ in range(permutation_draws):
        permuted = []
        for group in by_case.values():
            values = generator.permutation([float(row[response]) for row in group])
            permuted.extend({**row, response: value} for row, value in zip(group, values))
        statistic = _stratified_rank_correlation(permuted, mechanical, response)
        exceed += int(math.isfinite(statistic) and abs(statistic) >= abs(observed) - 1.0e-15)
    p_value = (exceed + 1.0) / (permutation_draws + 1.0)
    bootstrap_generator = np.random.default_rng(
        _stable_seed(seed, mechanical, response, "bootstrap")
    )
    estimates = []
    for _ in range(bootstrap_draws):
        sample = []
        for group in by_case.values():
            indices = bootstrap_generator.integers(0, len(group), size=len(group))
            sample.extend(group[index] for index in indices)
        statistic = _stratified_rank_correlation(sample, mechanical, response)
        if math.isfinite(statistic):
            estimates.append(statistic)
    if not estimates:
        return p_value, math.nan, math.nan, 0
    low, high = np.quantile(estimates, [0.025, 0.975])
    return p_value, float(low), float(high), len(estimates)


def _associations(
    rows: Sequence[Mapping[str, object]],
    coordinates: Sequence[tuple[str, str, str]],
    primary_label: str,
    *,
    permutation_draws: int,
    bootstrap_draws: int,
    seed: int,
) -> list[dict[str, object]]:
    labels = [item[0] for item in coordinates]
    if primary_label not in labels:
        raise ValueError("primary mechanical label is not configured")
    output = []
    for mechanical in labels:
        for response in RESPONSE_COORDINATES:
            finite = [row for row in rows if math.isfinite(float(row[response]))]
            is_primary = mechanical == primary_label and response == "far_mobilized_fraction"
            p_value, low, high, finite_bootstrap = ("", "", "", "")
            qualified = 0
            if is_primary:
                p_value, low, high, finite_bootstrap = _primary_inference(
                    finite,
                    mechanical,
                    response,
                    permutation_draws,
                    bootstrap_draws,
                    seed,
                )
                qualified = int(
                    math.isfinite(float(p_value))
                    and float(p_value) <= 0.05
                    and math.isfinite(float(low))
                    and (float(low) > 0.0 or float(high) < 0.0)
                )
            output.append(
                {
                    "scope": "pooled_within_surface_ranks",
                    "case_id": "ALL",
                    "mechanical_coordinate": mechanical,
                    "response_coordinate": response,
                    "pair_count": len(finite),
                    "rank_correlation": _stratified_rank_correlation(
                        finite, mechanical, response
                    ),
                    "permutation_p": p_value,
                    "bootstrap_ci025": low,
                    "bootstrap_ci975": high,
                    "finite_bootstrap_count": finite_bootstrap,
                    "is_primary": int(is_primary),
                    "within_selected_sample_qualified": qualified,
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
            for case_id in sorted({str(row["case_id"]) for row in finite}):
                group = [row for row in finite if str(row["case_id"]) == case_id]
                output.append(
                    {
                        "scope": "within_surface_descriptive",
                        "case_id": case_id,
                        "mechanical_coordinate": mechanical,
                        "response_coordinate": response,
                        "pair_count": len(group),
                        "rank_correlation": _rank_correlation(
                            [float(row[mechanical]) for row in group],
                            [float(row[response]) for row in group],
                        ),
                        "permutation_p": "",
                        "bootstrap_ci025": "",
                        "bootstrap_ci975": "",
                        "finite_bootstrap_count": "",
                        "is_primary": 0,
                        "within_selected_sample_qualified": 0,
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )
    return output


def run_analysis(
    mechanics_table: Path,
    selection_table: Path,
    event_outcomes: Path,
    output_dir: Path,
    *,
    mechanical_coordinates: Sequence[tuple[str, str, str]],
    primary_mechanical_label: str,
    patch_radius_A: float = 6.0,
    distance_label: str = "far",
    window_label: str = "fast",
    permutation_draws: int = 10000,
    bootstrap_draws: int = 2000,
    seed: int = 20260905,
) -> dict[str, object]:
    if patch_radius_A <= 0.0 or min(permutation_draws, bootstrap_draws) < 100:
        raise ValueError("invalid patch radius or resampling count")
    selection = _selection_index(selection_table)
    mechanics = _mechanics_index(
        mechanics_table,
        mechanical_coordinates,
        patch_radius_A,
        {pair_id: str(row["case_id"]) for pair_id, row in selection.items()},
    )
    outcomes = _outcome_index(event_outcomes, distance_label, window_label)
    joined, excluded = _join(selection, mechanics, outcomes)
    associations = _associations(
        joined,
        mechanical_coordinates,
        primary_mechanical_label,
        permutation_draws=permutation_draws,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )
    coverage = []
    for case_id in sorted({str(row["case_id"]) for row in selection.values()}):
        case_selected = [row for row in selection.values() if str(row["case_id"]) == case_id]
        case_joined = [row for row in joined if str(row["case_id"]) == case_id]
        coverage.append(
            {
                "case_id": case_id,
                "selected_mechanical_pair_count": len(case_selected),
                "joint_support_pair_count": len(case_joined),
                "excluded_no_outcome_count": len(case_selected) - len(case_joined),
                "finite_conversion_pair_count": sum(
                    math.isfinite(float(row["far_fast_conversion_fraction"]))
                    for row in case_joined
                ),
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "pair_force_response_table.csv", joined)
    _write_csv(output / "force_response_associations.csv", associations)
    _write_csv(output / "conversion_coverage_summary.csv", coverage)
    _write_support_exclusions(output / "support_exclusions.csv", excluded)
    primary = [row for row in associations if int(row["is_primary"])]
    if len(primary) != 1:
        raise ValueError("primary association accounting failed")
    summary = {
        "status": "PASS",
        "case_count": len(coverage),
        "selected_mechanical_pair_count": len(selection),
        "joint_support_pair_count": len(joined),
        "excluded_no_outcome_count": len(excluded),
        "mechanical_coordinates": [list(item) for item in mechanical_coordinates],
        "primary_mechanical_label": primary_mechanical_label,
        "response_coordinates": list(RESPONSE_COORDINATES),
        "primary_rank_correlation": primary[0]["rank_correlation"],
        "primary_permutation_p": primary[0]["permutation_p"],
        "primary_bootstrap_ci025": primary[0]["bootstrap_ci025"],
        "primary_bootstrap_ci975": primary[0]["bootstrap_ci975"],
        "primary_qualified": primary[0]["within_selected_sample_qualified"],
        "patch_radius_A": patch_radius_A,
        "distance_label": distance_label,
        "window_label": window_label,
        "permutation_draws": permutation_draws,
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "mechanics_table": {"path": str(mechanics_table), "sha256": _sha256(mechanics_table)},
                "selection_table": {"path": str(selection_table), "sha256": _sha256(selection_table)},
                "event_outcomes": {"path": str(event_outcomes), "sha256": _sha256(event_outcomes)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "# Frozen-snapshot force-response coupling\n\n"
        "The primary result is a surface-stratified rank association between the "
        "frozen radial force DID and far continuous mobilization. Unsupported "
        "snapshot pairs are excluded without imputation. This response-stratified "
        "single-trajectory analysis is not causal, local stress, a population "
        "effect, work, dissipation, or replicate-level evidence.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanics-table", type=Path, required=True)
    parser.add_argument("--selection-table", type=Path, required=True)
    parser.add_argument("--event-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mechanical-coordinate", action="append", required=True)
    parser.add_argument("--primary-mechanical-label", required=True)
    parser.add_argument("--patch-radius-A", type=float, default=6.0)
    parser.add_argument("--distance-label", default="far")
    parser.add_argument("--window-label", default="fast")
    parser.add_argument("--permutation-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            run_analysis(
                args.mechanics_table,
                args.selection_table,
                args.event_outcomes,
                args.output_dir,
                mechanical_coordinates=parse_mechanical_coordinates(
                    args.mechanical_coordinate
                ),
                primary_mechanical_label=args.primary_mechanical_label,
                patch_radius_A=args.patch_radius_A,
                distance_label=args.distance_label,
                window_label=args.window_label,
                permutation_draws=args.permutation_draws,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
