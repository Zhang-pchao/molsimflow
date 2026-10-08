"""Summarize paired event/control contrasts from frozen snapshot predictions."""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import rankdata

SCIENTIFIC_STATUS = (
    "SINGLE_TRAJECTORY_MATCHED_BLOCK_CONTRASTS_"
    "NOT_REPLICATE_CAUSAL_RATE_BARRIER_OR_DISSIPATION_EVIDENCE"
)
PHASES = ("pre", "transition", "post")
SAMPLE_KINDS = ("event", "circular_shift_control")
PRIMARY_CONTRASTS = ("did_transition_minus_pre", "did_post_minus_pre")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(values[original_index]) * len(values) / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def _sign_flip_p(values: np.ndarray) -> float:
    """Exact two-sided sign-flip test for a paired mean."""

    values = np.asarray(values, dtype=float)
    if not len(values) or len(values) > 20:
        raise ValueError("exact sign-flip test requires 1..20 values")
    observed = abs(float(np.mean(values)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1.0e-15)
        total += 1
    return exceed / total


def _bootstrap_interval(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def _permutation_spearman_p(
    predictor: np.ndarray, response: np.ndarray, draws: int, seed: int
) -> tuple[float, float]:
    predictor_rank = rankdata(predictor).astype(float)
    response_rank = rankdata(response).astype(float)
    predictor_rank -= np.mean(predictor_rank)
    response_rank -= np.mean(response_rank)
    denominator = float(np.linalg.norm(predictor_rank) * np.linalg.norm(response_rank))
    if denominator <= 0.0:
        return math.nan, math.nan
    observed = float(np.dot(predictor_rank, response_rank) / denominator)
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        statistic = float(
            np.dot(predictor_rank, generator.permutation(response_rank)) / denominator
        )
        exceed += int(abs(statistic) >= abs(observed) - 1.0e-15)
    return observed, (exceed + 1.0) / (draws + 1.0)


def _pair_contrast_rows(
    rows: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
    group_fields: Sequence[str] = (),
    *,
    drop_incomplete_groups: bool = False,
) -> list[dict[str, object]]:
    keys = set(rows[0])
    required = {
        "pair_id",
        "case_id",
        "sample_kind",
        "phase",
        "patch_radius_A",
        "source_time_block_200ps",
        "response_stratum",
        "response_affected_arc_fraction",
        *metrics,
        *group_fields,
    }
    missing = required.difference(keys)
    if missing:
        raise ValueError(f"mechanics table is missing columns: {sorted(missing)}")
    grouped: dict[tuple[object, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["pair_id"],
                float(row["patch_radius_A"]),
                *(row[field] for field in group_fields),
            )
        ].append(row)
    output = []
    for group_key, group in sorted(grouped.items()):
        pair_id, radius, *group_values = group_key
        group_metadata = dict(zip(group_fields, group_values))
        index = {}
        for row in group:
            state = (row["sample_kind"], row["phase"])
            if state in index:
                raise ValueError(f"{pair_id}/{radius}: duplicate state {state}")
            index[state] = row
        expected = set(itertools.product(SAMPLE_KINDS, PHASES))
        if set(index) != expected:
            if drop_incomplete_groups:
                continue
            raise ValueError(f"{pair_id}/{radius}: incomplete event/control phase grid")
        identities = {
            (
                row["case_id"],
                int(row["source_time_block_200ps"]),
                int(row["response_stratum"]),
                float(row["response_affected_arc_fraction"]),
            )
            for row in group
        }
        if len(identities) != 1:
            raise ValueError(f"{pair_id}/{radius}: inconsistent pair metadata")
        case_id, block, stratum, response = identities.pop()
        for metric in metrics:
            values = {
                state: float(row[metric])
                for state, row in index.items()
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{pair_id}/{radius}/{metric}: non-finite value")
            contrasts = {}
            for phase in PHASES:
                contrasts[f"event_minus_control_{phase}"] = (
                    values[("event", phase)]
                    - values[("circular_shift_control", phase)]
                )
            for phase in ("transition", "post"):
                event_change = values[("event", phase)] - values[("event", "pre")]
                control_change = (
                    values[("circular_shift_control", phase)]
                    - values[("circular_shift_control", "pre")]
                )
                contrasts[f"event_{phase}_minus_pre"] = event_change
                contrasts[f"control_{phase}_minus_pre"] = control_change
                contrasts[f"did_{phase}_minus_pre"] = event_change - control_change
            for contrast, value in contrasts.items():
                output.append(
                    {
                        "pair_id": pair_id,
                        "case_id": case_id,
                        "source_time_block_200ps": block,
                        "response_stratum": stratum,
                        "response_affected_arc_fraction": response,
                        "patch_radius_A": radius,
                        **group_metadata,
                        "metric": metric,
                        "contrast": contrast,
                        "value": value,
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )
    return output


def _summaries(
    contrasts: Sequence[Mapping[str, object]],
    *,
    primary_patch_radius_A: float,
    bootstrap_draws: int,
    seed: int,
    group_fields: Sequence[str] = (),
) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in contrasts:
        key = (
            row["case_id"],
            row["patch_radius_A"],
            *(row[field] for field in group_fields),
            row["metric"],
            row["contrast"],
        )
        grouped[key].append(row)
    output = []
    primary_indices = []
    primary_p_values = []
    for key, group in sorted(grouped.items()):
        case_id, radius, *middle, metric, contrast = key
        group_metadata = dict(zip(group_fields, middle))
        blocks = [int(row["source_time_block_200ps"]) for row in group]
        if len(blocks) != len(set(blocks)):
            raise ValueError(f"{case_id}/{radius}: a 200 ps block was reused")
        values = np.asarray([float(row["value"]) for row in group], dtype=float)
        low, high = _bootstrap_interval(
            values,
            bootstrap_draws,
            _stable_seed(
                seed,
                case_id,
                radius,
                *middle,
                metric,
                contrast,
                "bootstrap",
            ),
        )
        is_primary = math.isclose(
            float(radius), primary_patch_radius_A, rel_tol=0.0, abs_tol=1.0e-12
        ) and contrast in PRIMARY_CONTRASTS
        row = {
            "case_id": case_id,
            "patch_radius_A": radius,
            **group_metadata,
            "metric": metric,
            "contrast": contrast,
            "pair_count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "bootstrap_ci_2p5": low,
            "bootstrap_ci_97p5": high,
            "is_primary_family": int(is_primary),
            "sign_flip_p": _sign_flip_p(values) if is_primary else "",
            "bh_q": "",
            "within_trajectory_qualified": 0,
            "scientific_status": SCIENTIFIC_STATUS,
        }
        output.append(row)
        if is_primary:
            primary_indices.append(len(output) - 1)
            primary_p_values.append(float(row["sign_flip_p"]))
    for index, q_value in zip(primary_indices, _bh_adjust(primary_p_values)):
        output[index]["bh_q"] = q_value
        low = float(output[index]["bootstrap_ci_2p5"])
        high = float(output[index]["bootstrap_ci_97p5"])
        output[index]["within_trajectory_qualified"] = int(
            q_value < 0.05 and (low > 0.0 or high < 0.0)
        )
    return output


def _response_associations(
    contrasts: Sequence[Mapping[str, object]],
    *,
    primary_patch_radius_A: float,
    permutation_draws: int,
    seed: int,
    group_fields: Sequence[str] = (),
) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in contrasts:
        if row["contrast"] not in PRIMARY_CONTRASTS or not math.isclose(
            float(row["patch_radius_A"]),
            primary_patch_radius_A,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            continue
        grouped[
            (
                row["case_id"],
                *(row[field] for field in group_fields),
                row["metric"],
                row["contrast"],
            )
        ].append(row)
    output = []
    p_values = []
    for group_key, group in sorted(grouped.items()):
        case_id, *middle, metric, contrast = group_key
        group_metadata = dict(zip(group_fields, middle))
        predictor = np.asarray(
            [float(row["response_affected_arc_fraction"]) for row in group]
        )
        response = np.asarray([float(row["value"]) for row in group])
        rho, p_value = _permutation_spearman_p(
            predictor,
            response,
            permutation_draws,
            _stable_seed(seed, case_id, *middle, metric, contrast, "association"),
        )
        output.append(
            {
                "case_id": case_id,
                "patch_radius_A": primary_patch_radius_A,
                **group_metadata,
                "metric": metric,
                "contrast": contrast,
                "pair_count": len(group),
                "spearman_rho": rho,
                "permutation_p": p_value,
                "bh_q": "",
                "within_selected_sample_qualified": 0,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
        p_values.append(p_value)
    finite_indices = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    finite_q = _bh_adjust([p_values[index] for index in finite_indices])
    for index, q_value in zip(finite_indices, finite_q):
        output[index]["bh_q"] = q_value
        output[index]["within_selected_sample_qualified"] = int(q_value < 0.05)
    return output


def run_contrasts(
    mechanics_table: Path,
    metrics: Sequence[str],
    output_dir: Path,
    *,
    primary_patch_radius_A: float,
    bootstrap_draws: int = 2000,
    permutation_draws: int = 10000,
    seed: int = 20260904,
    group_fields: Sequence[str] = (),
    required_equal: Sequence[tuple[str, str]] = (),
    drop_incomplete_groups: bool = False,
) -> dict[str, object]:
    if not metrics or len(metrics) != len(set(metrics)):
        raise ValueError("metrics must be a non-empty unique sequence")
    if primary_patch_radius_A <= 0.0 or bootstrap_draws < 100 or permutation_draws < 100:
        raise ValueError("invalid contrast configuration")
    if len(group_fields) != len(set(group_fields)):
        raise ValueError("group fields must be unique")
    raw_rows = _read_csv(mechanics_table)
    for field, _ in required_equal:
        if field not in raw_rows[0]:
            raise ValueError(f"required filter field is absent: {field}")
    filtered_rows = [
        row
        for row in raw_rows
        if all(row[field] == value for field, value in required_equal)
    ]
    if not filtered_rows:
        raise ValueError("no rows remain after required-value filters")
    contrasts = _pair_contrast_rows(
        filtered_rows,
        metrics,
        group_fields,
        drop_incomplete_groups=drop_incomplete_groups,
    )
    if not contrasts:
        raise ValueError("no complete event/control phase groups remain")
    summaries = _summaries(
        contrasts,
        primary_patch_radius_A=primary_patch_radius_A,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
        group_fields=group_fields,
    )
    associations = _response_associations(
        contrasts,
        primary_patch_radius_A=primary_patch_radius_A,
        permutation_draws=permutation_draws,
        seed=seed,
        group_fields=group_fields,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "pair_contrasts.csv", contrasts)
    _write_csv(output / "contrast_summary.csv", summaries)
    _write_csv(output / "response_associations.csv", associations)
    primary_qualified = sum(
        int(row["within_trajectory_qualified"])
        for row in summaries
        if int(row["is_primary_family"])
    )
    association_qualified = sum(
        int(row["within_selected_sample_qualified"]) for row in associations
    )
    summary = {
        "status": "PASS",
        "case_count": len({str(row["case_id"]) for row in contrasts}),
        "pair_count": len({str(row["pair_id"]) for row in contrasts}),
        "metrics": list(metrics),
        "group_fields": list(group_fields),
        "required_equal": [list(item) for item in required_equal],
        "drop_incomplete_groups": drop_incomplete_groups,
        "input_row_count": len(raw_rows),
        "filtered_row_count": len(filtered_rows),
        "patch_radii_A": sorted({float(row["patch_radius_A"]) for row in contrasts}),
        "primary_patch_radius_A": primary_patch_radius_A,
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "bootstrap_draws": bootstrap_draws,
        "permutation_draws": permutation_draws,
        "seed": seed,
        "primary_family_test_count": sum(
            int(row["is_primary_family"]) for row in summaries
        ),
        "within_trajectory_qualified_count": primary_qualified,
        "selected_sample_association_qualified_count": association_qualified,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "mechanics_table": {
                    "path": str(mechanics_table),
                    "sha256": _sha256(mechanics_table),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "# Paired frozen-snapshot mechanics contrasts\n\n"
        "Event snapshots are compared with geometry-matched circular-shift controls. "
        "The primary quantities are transition-minus-pre and post-minus-pre "
        "difference-in-differences at the frozen primary patch radius. Whole 200 ps "
        "blocks are the resampling unit; exact paired sign-flip tests are BH-adjusted. "
        "Other radii are sensitivity analyses. Associations with affected-arc fraction "
        "are conditional on response-stratified selection. These are single-trajectory "
        "static mechanical contrasts, not replicate uncertainty, causality, rates, "
        "barriers, dissipated work, entropy production, or propagation evidence.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanics-table", type=Path, required=True)
    parser.add_argument("--metrics", required=True, help="comma-separated numeric columns")
    parser.add_argument("--primary-patch-radius-A", type=float, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--permutation-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--group-fields", default="", help="comma-separated stratification columns")
    parser.add_argument(
        "--require-equal",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="retain rows whose field exactly equals value",
    )
    parser.add_argument("--drop-incomplete-groups", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    required_equal = []
    for item in args.require_equal:
        if "=" not in item:
            raise ValueError("--require-equal must use FIELD=VALUE")
        required_equal.append(tuple(item.split("=", 1)))
    run_contrasts(
        args.mechanics_table,
        tuple(item.strip() for item in args.metrics.split(",") if item.strip()),
        args.output_dir,
        primary_patch_radius_A=args.primary_patch_radius_A,
        bootstrap_draws=args.bootstrap_draws,
        permutation_draws=args.permutation_draws,
        seed=args.seed,
        group_fields=tuple(
            item.strip() for item in args.group_fields.split(",") if item.strip()
        ),
        required_equal=required_equal,
        drop_incomplete_groups=args.drop_incomplete_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
