"""Attribute an event-aligned periodic-field response to additive components.

The input is an accepted event-level response table plus its cellwise null map.
Whole-block resampling quantifies within-trajectory uncertainty.  Component
fractions are conditional diagnostics, not causal or replicate-level effects.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_ADDITIVE_MODE_ATTRIBUTION_NOT_CAUSAL_PROPAGATION_"
    "ENERGY_DISSIPATION_OR_REPLICATE_LEVEL_EVIDENCE"
)


@dataclass(frozen=True)
class Bin:
    name: str
    start: float
    end: float


def parse_bins(raw: str, *, integer: bool) -> tuple[Bin, ...]:
    bins = []
    for item in raw.split(","):
        fields = item.split(":")
        if len(fields) != 3:
            raise ValueError(f"invalid bin specification: {item!r}")
        name, start_raw, end_raw = fields
        start = int(start_raw) if integer else float(start_raw)
        end = int(end_raw) if integer else float(end_raw)
        if not name or end <= start:
            raise ValueError(f"invalid bin specification: {item!r}")
        bins.append(Bin(name, float(start), float(end)))
    if not bins or len({item.name for item in bins}) != len(bins):
        raise ValueError("bins must be nonempty with unique names")
    ordered = sorted(bins, key=lambda item: item.start)
    if any(right.start < left.end for left, right in zip(ordered, ordered[1:])):
        raise ValueError("bins must not overlap")
    return tuple(bins)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_region(value: float, bins: Sequence[Bin]) -> Optional[str]:
    selected = [item.name for item in bins if item.start <= value < item.end]
    if len(selected) > 1:
        raise ValueError(f"distance {value} belongs to overlapping regions")
    return selected[0] if selected else None


def _select_window(value: float, bins: Sequence[Bin]) -> Optional[str]:
    selected = [item.name for item in bins if item.start < value <= item.end]
    if len(selected) > 1:
        raise ValueError(f"lag {value} belongs to overlapping windows")
    return selected[0] if selected else None


def _stream_accumulate(
    path: Path,
    *,
    fields: set[str],
    regions: Sequence[Bin],
    windows: Sequence[Bin],
    event_level: bool,
) -> tuple[dict[tuple[object, ...], list[float]], tuple[str, ...]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        value_column = "aligned_change" if event_level else "null_mean"
        required = {
            "case_id",
            "field",
            "lag_ps",
            "arc_distance",
            value_column,
        }
        if event_level:
            required.update(("event_id", "time_block_200ps"))
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        output: dict[tuple[object, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in reader:
            field = row["field"]
            if field not in fields:
                continue
            region = _select_region(float(row["arc_distance"]), regions)
            window = _select_window(float(row["lag_ps"]), windows)
            if region is None or window is None:
                continue
            value = float(row[value_column])
            if not math.isfinite(value):
                raise ValueError(f"{path}: nonfinite {value_column}")
            if event_level:
                key = (
                    row["case_id"],
                    int(row["event_id"]),
                    int(row["time_block_200ps"]),
                    region,
                    window,
                    field,
                )
            else:
                key = (row["case_id"], region, window, field)
            output[key][0] += value
            output[key][1] += 1.0
    if not output:
        raise ValueError(f"{path}: no rows survive the configured field/bin selection")
    return dict(output), fieldnames


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if len(finite) else math.nan


def run_analysis(
    event_response_table: Path,
    map_table: Path,
    output_dir: Path,
    *,
    total_field: str,
    component_fields: Sequence[str],
    regions: Sequence[Bin],
    windows: Sequence[Bin],
    primary_region: str,
    primary_window: str,
    bootstrap_samples: int,
    random_seed: int,
    closure_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Calculate additive response attribution with whole-block resampling."""

    components = tuple(component_fields)
    if not components or total_field in components or len(set(components)) != len(components):
        raise ValueError("component fields must be nonempty, unique, and exclude total_field")
    if primary_region not in {item.name for item in regions}:
        raise ValueError("primary_region is absent from regions")
    if primary_window not in {item.name for item in windows}:
        raise ValueError("primary_window is absent from windows")
    if bootstrap_samples < 20 or closure_tolerance <= 0.0:
        raise ValueError("bootstrap_samples or closure_tolerance is invalid")
    all_fields = (total_field, *components)
    event_acc, _ = _stream_accumulate(
        event_response_table,
        fields=set(all_fields),
        regions=regions,
        windows=windows,
        event_level=True,
    )
    null_acc, _ = _stream_accumulate(
        map_table,
        fields=set(all_fields),
        regions=regions,
        windows=windows,
        event_level=False,
    )

    by_event: dict[tuple[str, int, int, str, str], dict[str, float]] = defaultdict(dict)
    for key, (total, count) in event_acc.items():
        case_id, event_id, block, region, window, field = key
        by_event[(str(case_id), int(event_id), int(block), str(region), str(window))][
            str(field)
        ] = total / count
    event_rows = []
    for key, values in sorted(by_event.items()):
        missing = sorted(set(all_fields).difference(values))
        if missing:
            raise ValueError(f"event aggregation {key} is missing fields {missing}")
        closure = values[total_field] - sum(values[field] for field in components)
        if abs(closure) > closure_tolerance:
            raise ValueError(f"event aggregation closure failed for {key}: {closure}")
        event_rows.append(
            {
                "case_id": key[0],
                "event_id": key[1],
                "time_block_200ps": key[2],
                "region": key[3],
                "window": key[4],
                **{field: values[field] for field in all_fields},
                "additive_closure_error_A": closure,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )

    null_means = {key: total / count for key, (total, count) in null_acc.items()}
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        grouped[(str(row["case_id"]), str(row["region"]), str(row["window"]))].append(row)
    combinations = sorted(grouped)
    seed_sequences = np.random.SeedSequence(random_seed).spawn(len(combinations))
    summary_rows = []
    maximum_closure_error = 0.0
    for combination, seed_sequence in zip(combinations, seed_sequences):
        case_id, region, window = combination
        rows = grouped[combination]
        blocks = sorted({int(row["time_block_200ps"]) for row in rows})
        if len(blocks) < 2:
            raise ValueError(f"{combination}: fewer than two populated blocks")
        block_sums = {
            field: np.asarray(
                [
                    sum(float(row[field]) for row in rows if int(row["time_block_200ps"]) == block)
                    for block in blocks
                ],
                dtype=float,
            )
            for field in all_fields
        }
        block_counts = np.asarray(
            [sum(int(row["time_block_200ps"]) == block for row in rows) for block in blocks],
            dtype=float,
        )
        null = {}
        for field in all_fields:
            key = (case_id, region, window, field)
            if key not in null_means:
                raise ValueError(f"null map is missing {key}")
            null[field] = null_means[key]
        observed = {
            field: sum(float(row[field]) for row in rows) / len(rows) for field in all_fields
        }
        effects = {field: observed[field] - null[field] for field in all_fields}
        observed_closure = observed[total_field] - sum(observed[field] for field in components)
        null_closure = null[total_field] - sum(null[field] for field in components)
        effect_closure = effects[total_field] - sum(effects[field] for field in components)
        maximum_closure_error = max(
            maximum_closure_error,
            abs(observed_closure),
            abs(null_closure),
            abs(effect_closure),
        )
        if maximum_closure_error > closure_tolerance:
            raise ValueError(f"additive closure failed for {combination}")

        rng = np.random.default_rng(seed_sequence)
        selected = rng.integers(0, len(blocks), size=(bootstrap_samples, len(blocks)))
        selected_counts = np.sum(block_counts[selected], axis=1)
        bootstrap = {
            field: np.sum(block_sums[field][selected], axis=1) / selected_counts - null[field]
            for field in all_fields
        }
        total_bootstrap = bootstrap[total_field]
        for field in all_fields:
            fraction = (
                effects[field] / effects[total_field]
                if abs(effects[total_field]) > closure_tolerance
                else math.nan
            )
            fraction_samples = np.divide(
                bootstrap[field],
                total_bootstrap,
                out=np.full(bootstrap_samples, np.nan),
                where=np.abs(total_bootstrap) > closure_tolerance,
            )
            summary_rows.append(
                {
                    "case_id": case_id,
                    "region": region,
                    "window": window,
                    "field": field,
                    "field_role": "total" if field == total_field else "component",
                    "event_count": len(rows),
                    "block_count": len(blocks),
                    "observed_mean_aligned_change_A": observed[field],
                    "null_mean_aligned_change_A": null[field],
                    "event_minus_null_A": effects[field],
                    "effect_block_bootstrap_ci025_A": _quantile(bootstrap[field], 0.025),
                    "effect_block_bootstrap_ci975_A": _quantile(bootstrap[field], 0.975),
                    "signed_fraction_of_total_effect": fraction,
                    "fraction_block_bootstrap_ci025": _quantile(fraction_samples, 0.025),
                    "fraction_block_bootstrap_ci975": _quantile(fraction_samples, 0.975),
                    "primary_conditional_attribution": (
                        region == primary_region and window == primary_window and field != total_field
                    ),
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "event_mode_attribution.csv", event_rows)
    _write_csv(output_dir / "mode_attribution_summary.csv", summary_rows)
    primary_rows = [row for row in summary_rows if row["primary_conditional_attribution"]]
    _write_csv(output_dir / "primary_mode_attribution.csv", primary_rows)
    case_ids = sorted({str(row["case_id"]) for row in event_rows})
    summary = {
        "status": "PASS",
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "event_count": len({(row["case_id"], row["event_id"]) for row in event_rows}),
        "event_region_window_row_count": len(event_rows),
        "summary_row_count": len(summary_rows),
        "primary_row_count": len(primary_rows),
        "fields": list(all_fields),
        "regions": [item.__dict__ for item in regions],
        "windows": [item.__dict__ for item in windows],
        "primary_region": primary_region,
        "primary_window": primary_window,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "maximum_additive_closure_error_A": maximum_closure_error,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "event_response_table": {
            "path": str(Path(event_response_table).resolve()),
            "sha256": _sha256(event_response_table),
        },
        "map_table": {
            "path": str(Path(map_table).resolve()),
            "sha256": _sha256(map_table),
        },
        "total_field": total_field,
        "component_fields": list(components),
        "regions": [item.__dict__ for item in regions],
        "windows": [item.__dict__ for item in windows],
        "primary_region": primary_region,
        "primary_window": primary_window,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "closure_tolerance": closure_tolerance,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            (
                "# Event-aligned additive mode attribution",
                "",
                f"- Cases: {len(case_ids)}",
                f"- Accepted event anchors: {summary['event_count']}",
                f"- Primary conditional rows: {len(primary_rows)}",
                f"- Maximum additive closure error: {maximum_closure_error:.3e} A",
                "",
                (
                    "Fractions partition the accepted continuous response into additive "
                    "components. They do not establish causality, propagation, energy "
                    "transfer, dissipation, or replicate-level uncertainty."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-response-table", type=Path, required=True)
    parser.add_argument("--map-table", type=Path, required=True)
    parser.add_argument("--total-field", required=True)
    parser.add_argument("--component-fields", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--primary-region", required=True)
    parser.add_argument("--primary-window", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument("--closure-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        args.event_response_table,
        args.map_table,
        args.output_dir,
        total_field=args.total_field,
        component_fields=tuple(item for item in args.component_fields.split(",") if item),
        regions=parse_bins(args.regions, integer=True),
        windows=parse_bins(args.windows, integer=False),
        primary_region=args.primary_region,
        primary_window=args.primary_window,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        closure_tolerance=args.closure_tolerance,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
