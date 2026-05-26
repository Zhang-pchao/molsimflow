"""Case-level descriptor scorecards, deltas, and correlations.

This module migrates the reusable table layer from legacy case-comparison
scripts.  It intentionally avoids project-specific case roots, plotting, and
report templates.  Inputs are explicit CSV manifests so the same code can be
used for different systems and descriptor families.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_METADATA_COLUMNS = {
    "case_label",
    "case_id",
    "case_name",
    "case_path",
    "source_path",
    "source_file",
    "input_path",
    "output_dir",
    "matrix_output_dir",
    "dataset_key",
    "dataset_group",
    "label",
    "display_label",
}


@dataclass(frozen=True)
class DescriptorTableSpec:
    """Configuration for one case-level descriptor table."""

    name: str
    path: Path
    case_column: str = "case_label"
    columns: Tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        """Return a stable output prefix for columns from this table."""

        prefix = re.sub(r"[^0-9A-Za-z_]+", "_", self.name.strip()).strip("_").lower()
        if not prefix:
            raise ValueError("Descriptor table name cannot be empty")
        return prefix


@dataclass(frozen=True)
class CasePairSpec:
    """Reference/target case pair for target-minus-reference deltas."""

    reference_case: str
    target_case: str
    label: str = ""

    @property
    def pair_label(self) -> str:
        return self.label or f"{self.target_case}_minus_{self.reference_case}"


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader], [str(field) for field in reader.fieldnames]


def _write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _split_columns(raw: object) -> Tuple[str, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(item.strip() for item in re.split(r"[;,|]", text) if item.strip())


def _as_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "required"}


def _as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _clean_case_key(value: object) -> str:
    return str(value).strip()


def _resolve_manifest_path(raw_path: object, manifest_path: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def load_case_manifest(path: Path, case_column: str = "case_label") -> List[Dict[str, object]]:
    """Load a case manifest and normalize the selected case key to `case_label`.

    The preferred key is `case_column`.  If the requested default `case_label`
    is absent, `case_id` is accepted as a fallback.
    """

    rows, fieldnames = _read_csv_rows(path)
    selected_column = case_column
    if selected_column not in fieldnames:
        if case_column == "case_label" and "case_id" in fieldnames:
            selected_column = "case_id"
        else:
            raise ValueError(f"Case manifest is missing case column {case_column}: {path}")

    out: List[Dict[str, object]] = []
    seen = set()
    for index, row in enumerate(rows):
        case_label = _clean_case_key(row.get(selected_column, ""))
        if not case_label:
            continue
        if case_label in seen:
            raise ValueError(f"Duplicate case label in manifest: {case_label}")
        seen.add(case_label)
        item: Dict[str, object] = {"case_label": case_label, "case_manifest_row": index}
        for key, value in row.items():
            if key == "case_label":
                item["case_label_raw"] = value
            elif key != selected_column:
                item[key] = value
        if selected_column != "case_label":
            item[selected_column] = row.get(selected_column, "")
        out.append(item)
    if not out:
        raise ValueError(f"No cases found in manifest: {path}")
    return out


def load_descriptor_manifest(path: Path) -> List[DescriptorTableSpec]:
    """Load descriptor table specs from a CSV manifest.

    Required columns are `name` and `path`.  Optional columns are `case_column`,
    `columns`, and `required`.  Relative descriptor paths are resolved relative
    to the descriptor manifest.
    """

    rows, fieldnames = _read_csv_rows(path)
    missing = [column for column in ("name", "path") if column not in fieldnames]
    if missing:
        raise ValueError("Descriptor manifest missing required columns: " + ", ".join(missing))
    specs: List[DescriptorTableSpec] = []
    for row in rows:
        name = str(row.get("name", "")).strip()
        raw_path = str(row.get("path", "")).strip()
        if not name or not raw_path:
            continue
        descriptor_path = _resolve_manifest_path(raw_path, Path(path))
        if not descriptor_path.exists() and not _as_bool(row.get("required", True)):
            continue
        specs.append(
            DescriptorTableSpec(
                name=name,
                path=descriptor_path,
                case_column=str(row.get("case_column") or "case_label").strip(),
                columns=_split_columns(row.get("columns")),
            )
        )
    if not specs:
        raise ValueError(f"No descriptor tables found in manifest: {path}")
    return specs


def parse_descriptor_table_spec(values: Sequence[str]) -> DescriptorTableSpec:
    """Parse a CLI descriptor table tuple: NAME PATH CASE_COLUMN COLUMNS."""

    if len(values) != 4:
        raise ValueError("--descriptor-table requires NAME PATH CASE_COLUMN COLUMNS")
    name, path, case_column, columns = values
    return DescriptorTableSpec(name=name, path=Path(path), case_column=case_column, columns=_split_columns(columns))


def parse_case_pair(raw: str) -> CasePairSpec:
    """Parse `REFERENCE:TARGET[:LABEL]` case-pair syntax."""

    parts = str(raw).split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("Case pairs must use REFERENCE:TARGET[:LABEL] syntax")
    reference_case = parts[0].strip()
    target_case = parts[1].strip()
    label = parts[2].strip() if len(parts) == 3 else ""
    if not reference_case or not target_case:
        raise ValueError("Case pair reference and target cannot be empty")
    return CasePairSpec(reference_case=reference_case, target_case=target_case, label=label)


def infer_numeric_columns(
    rows: Sequence[Mapping[str, object]],
    case_column: str,
    metadata_columns: Iterable[str] = DEFAULT_METADATA_COLUMNS,
) -> Tuple[str, ...]:
    """Infer numeric descriptor columns from a case-level table."""

    if not rows:
        return ()
    excluded = set(metadata_columns)
    excluded.add(case_column)
    columns = rows[0].keys()
    out: List[str] = []
    for column in columns:
        name = str(column)
        low = name.lower()
        if name in excluded or low in excluded:
            continue
        if low.endswith("_rank_high_to_low") or low.startswith("rank_") or "_rank_" in low:
            continue
        values = [_as_float(row.get(name)) for row in rows]
        if any(math.isfinite(value) for value in values):
            out.append(name)
    return tuple(out)


def _index_rows_by_case(
    rows: Sequence[Mapping[str, object]],
    case_column: str,
) -> Dict[str, List[Mapping[str, object]]]:
    out: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        case_label = _clean_case_key(row.get(case_column, ""))
        if not case_label:
            continue
        out.setdefault(case_label, []).append(row)
    return out


def build_case_scorecard(
    case_rows: Sequence[Mapping[str, object]],
    descriptor_specs: Sequence[DescriptorTableSpec],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    """Join case metadata with one or more case-level descriptor tables."""

    if not case_rows:
        raise ValueError("At least one case row is required")
    scorecard: List[Dict[str, object]] = [dict(row) for row in case_rows]
    descriptor_columns: List[str] = []
    input_manifest_rows: List[Dict[str, object]] = []

    for spec in descriptor_specs:
        table_rows, fieldnames = _read_csv_rows(spec.path)
        if spec.case_column not in fieldnames:
            raise ValueError(f"{spec.path} is missing case column {spec.case_column}")
        selected_columns = spec.columns or infer_numeric_columns(table_rows, spec.case_column)
        missing = [column for column in selected_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"{spec.path} is missing descriptor columns: {', '.join(missing)}")
        by_case = _index_rows_by_case(table_rows, spec.case_column)
        prefixed_columns = [f"{spec.prefix}__{column}" for column in selected_columns]
        descriptor_columns.extend(prefixed_columns)
        input_manifest_rows.append(
            {
                "source_kind": "descriptor_table",
                "name": spec.name,
                "path": str(spec.path),
                "case_column": spec.case_column,
                "selected_columns": "|".join(selected_columns),
                "output_columns": "|".join(prefixed_columns),
                "row_count": len(table_rows),
                "matched_case_count": sum(1 for row in scorecard if row.get("case_label") in by_case),
            }
        )

        for row in scorecard:
            case_label = _clean_case_key(row.get("case_label", ""))
            matches = by_case.get(case_label, [])
            row[f"{spec.prefix}__source_path"] = str(spec.path) if matches else ""
            row[f"{spec.prefix}__source_row_count"] = len(matches)
            source_row = matches[0] if matches else {}
            for source_column, output_column in zip(selected_columns, prefixed_columns):
                row[output_column] = _as_float(source_row.get(source_column)) if source_row else math.nan

    case_manifest_row = {
        "source_kind": "case_manifest",
        "name": "cases",
        "path": "",
        "case_column": "case_label",
        "selected_columns": "",
        "output_columns": "case_label",
        "row_count": len(case_rows),
        "matched_case_count": len(case_rows),
    }
    return scorecard, [case_manifest_row] + input_manifest_rows, descriptor_columns


def _percent_change(reference_value: float, target_value: float) -> float:
    if not math.isfinite(reference_value) or not math.isfinite(target_value):
        return math.nan
    if abs(reference_value) < 1.0e-12:
        return math.nan
    return 100.0 * (target_value - reference_value) / abs(reference_value)


def compute_case_deltas(
    scorecard_rows: Sequence[Mapping[str, object]],
    pairs: Sequence[CasePairSpec],
    value_columns: Sequence[str],
) -> List[Dict[str, object]]:
    """Compute target-minus-reference deltas for selected scorecard columns."""

    by_case = {_clean_case_key(row.get("case_label", "")): row for row in scorecard_rows}
    rows: List[Dict[str, object]] = []
    for pair in pairs:
        reference = by_case.get(pair.reference_case)
        target = by_case.get(pair.target_case)
        if reference is None or target is None:
            rows.append(
                {
                    "case_pair_label": pair.pair_label,
                    "reference_case": pair.reference_case,
                    "target_case": pair.target_case,
                    "descriptor": "",
                    "reference_value": math.nan,
                    "target_value": math.nan,
                    "delta_target_minus_reference": math.nan,
                    "percent_change_target_vs_reference": math.nan,
                    "status": "MISSING_CASE",
                }
            )
            continue
        for column in value_columns:
            reference_value = _as_float(reference.get(column))
            target_value = _as_float(target.get(column))
            delta = target_value - reference_value if math.isfinite(reference_value) and math.isfinite(target_value) else math.nan
            rows.append(
                {
                    "case_pair_label": pair.pair_label,
                    "reference_case": pair.reference_case,
                    "target_case": pair.target_case,
                    "descriptor": column,
                    "reference_value": reference_value,
                    "target_value": target_value,
                    "delta_target_minus_reference": delta,
                    "percent_change_target_vs_reference": _percent_change(reference_value, target_value),
                    "status": "OK" if math.isfinite(delta) else "NONFINITE_VALUE",
                }
            )
    return rows


def _paired_numeric_arrays(
    rows: Sequence[Mapping[str, object]],
    x_column: str,
    y_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_values: List[float] = []
    y_values: List[float] = []
    for row in rows:
        x = _as_float(row.get(x_column))
        y = _as_float(row.get(y_column))
        if math.isfinite(x) and math.isfinite(y):
            x_values.append(x)
            y_values.append(y)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return math.nan
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return math.nan
    return _pearson(_average_ranks(x), _average_ranks(y))


def compute_correlations(
    scorecard_rows: Sequence[Mapping[str, object]],
    target_column: str,
    descriptor_columns: Sequence[str],
) -> List[Dict[str, object]]:
    """Compute Pearson and Spearman correlations against a target column."""

    rows: List[Dict[str, object]] = []
    for column in descriptor_columns:
        if column == target_column:
            continue
        descriptor_values, target_values = _paired_numeric_arrays(scorecard_rows, column, target_column)
        pearson = _pearson(descriptor_values, target_values)
        spearman = _spearman(descriptor_values, target_values)
        rows.append(
            {
                "descriptor": column,
                "target_column": target_column,
                "n_pairs": int(descriptor_values.size),
                "descriptor_mean": float(np.mean(descriptor_values)) if descriptor_values.size else math.nan,
                "target_mean": float(np.mean(target_values)) if target_values.size else math.nan,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "abs_pearson_r": abs(pearson) if math.isfinite(pearson) else math.nan,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -_as_float(row.get("abs_pearson_r")),
            -int(row.get("n_pairs", 0)),
            str(row.get("descriptor", "")),
        ),
    )


def _scorecard_fieldnames(
    scorecard_rows: Sequence[Mapping[str, object]],
    descriptor_columns: Sequence[str],
) -> List[str]:
    keys = {key for row in scorecard_rows for key in row.keys()}
    preferred = ["case_label", "case_id", "case_name", "case_group", "case_manifest_row"]
    ordered = [key for key in preferred if key in keys]
    for column in descriptor_columns:
        if column in keys and column not in ordered:
            ordered.append(column)
    ordered.extend(sorted(key for key in keys if key not in ordered))
    return ordered


def analyze_case_scorecard(
    case_manifest: Path,
    descriptor_specs: Sequence[DescriptorTableSpec],
    output_dir: Path,
    case_column: str = "case_label",
    pair_specs: Sequence[CasePairSpec] = (),
    target_column: Optional[str] = None,
    correlate_columns: Sequence[str] = (),
) -> Dict[str, Path]:
    """Build scorecard, optional case deltas, and optional correlations."""

    if not descriptor_specs:
        raise ValueError("At least one descriptor table is required")
    output_dir = Path(output_dir)
    case_rows = load_case_manifest(case_manifest, case_column=case_column)
    scorecard_rows, input_manifest_rows, descriptor_columns = build_case_scorecard(case_rows, descriptor_specs)

    selected_for_delta = list(correlate_columns) if correlate_columns else descriptor_columns
    delta_rows = compute_case_deltas(scorecard_rows, pair_specs, selected_for_delta) if pair_specs else []

    if target_column is not None:
        if target_column not in scorecard_rows[0]:
            raise ValueError(f"Target column is not present in scorecard: {target_column}")
        selected_for_correlation = list(correlate_columns) if correlate_columns else descriptor_columns
        correlation_rows = compute_correlations(scorecard_rows, target_column, selected_for_correlation)
    else:
        correlation_rows = []

    outputs = {
        "scorecard": output_dir / "case_scorecard.csv",
        "delta": output_dir / "case_descriptor_delta.csv",
        "correlation": output_dir / "case_descriptor_correlation.csv",
        "manifest": output_dir / "case_comparison_input_manifest.csv",
    }
    _write_csv_rows(outputs["scorecard"], scorecard_rows, _scorecard_fieldnames(scorecard_rows, descriptor_columns))
    _write_csv_rows(
        outputs["delta"],
        delta_rows,
        [
            "case_pair_label",
            "reference_case",
            "target_case",
            "descriptor",
            "reference_value",
            "target_value",
            "delta_target_minus_reference",
            "percent_change_target_vs_reference",
            "status",
        ],
    )
    _write_csv_rows(
        outputs["correlation"],
        correlation_rows,
        [
            "descriptor",
            "target_column",
            "n_pairs",
            "descriptor_mean",
            "target_mean",
            "pearson_r",
            "spearman_r",
            "abs_pearson_r",
        ],
    )
    _write_csv_rows(
        outputs["manifest"],
        input_manifest_rows,
        [
            "source_kind",
            "name",
            "path",
            "case_column",
            "selected_columns",
            "output_columns",
            "row_count",
            "matched_case_count",
        ],
    )
    return outputs


def _descriptor_specs_from_args(args: argparse.Namespace) -> List[DescriptorTableSpec]:
    specs: List[DescriptorTableSpec] = []
    if args.descriptor_manifest is not None:
        specs.extend(load_descriptor_manifest(args.descriptor_manifest))
    for values in args.descriptor_table or []:
        specs.append(parse_descriptor_table_spec(values))
    return specs


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build case-level descriptor scorecards, deltas, and correlations")
    parser.add_argument("--cases", type=Path, required=True, help="Case manifest CSV with case_label or selected case column")
    parser.add_argument("--case-column", default="case_label", help="Case key column in --cases")
    parser.add_argument("--descriptor-manifest", type=Path, help="CSV with name,path,case_column,columns columns")
    parser.add_argument(
        "--descriptor-table",
        nargs=4,
        action="append",
        metavar=("NAME", "PATH", "CASE_COLUMN", "COLUMNS"),
        help="Explicit descriptor table; COLUMNS may be comma, semicolon, or pipe separated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", action="append", help="Case pair as REFERENCE:TARGET[:LABEL]; may be repeated")
    parser.add_argument("--target-column", help="Scorecard column used as the correlation target")
    parser.add_argument("--correlate", help="Comma, semicolon, or pipe separated scorecard columns for deltas/correlations")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        specs = _descriptor_specs_from_args(args)
        pairs = [parse_case_pair(item) for item in args.pair or []]
        outputs = analyze_case_scorecard(
            case_manifest=args.cases,
            descriptor_specs=specs,
            output_dir=args.output_dir,
            case_column=args.case_column,
            pair_specs=pairs,
            target_column=args.target_column,
            correlate_columns=_split_columns(args.correlate),
        )
    except Exception as exc:
        print(f"Case comparison failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
