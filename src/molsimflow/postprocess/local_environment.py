"""Local-environment class summaries and transitions from explicit tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.transitions import (
    SpeciesStateRow,
    build_species_transition_matrix,
    transition_matrix_to_long_rows,
)


SAMPLE_COLUMNS = ("frame", "time", "entity_id", "environment_class", "source_row_index")


@dataclass(frozen=True)
class LocalEnvironmentConfig:
    """Settings for local-environment table summaries."""

    class_order: Tuple[str, ...] = ()


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
        fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "frame",
        "time",
        "environment_class",
        "entity_id",
        "count",
        "fraction",
        "n_samples",
        "n_entities",
        "from_species",
        "to_species",
        "probability",
        "metric",
        "value",
    ]
    ordered = [key for key in preferred if key in keys]
    ordered.extend(key for key in keys if key not in ordered)
    return ordered


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: object, default: int = -1) -> int:
    number = _as_float(value)
    return int(round(number)) if math.isfinite(number) else default


def _format_float(value: float) -> object:
    return value if math.isfinite(value) else ""


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[Optional[str]], required: bool = True) -> Optional[str]:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    if required:
        requested = ", ".join(str(candidate) for candidate in candidates if candidate)
        raise ValueError(f"Could not find required column among [{requested}]; available columns: {', '.join(fieldnames)}")
    return None


def parse_class_order(raw: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Parse repeated or delimited environment-class order arguments."""

    if not raw:
        return ()
    values: Iterable[str]
    if isinstance(raw, str):
        values = [raw]
    else:
        values = raw
    out: List[str] = []
    for value in values:
        for item in re.split(r"[,;|]", str(value)):
            item = item.strip()
            if item:
                out.append(item)
    return tuple(dict.fromkeys(out))


def _infer_feature_columns(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    excluded: Sequence[str],
    explicit: Sequence[str] = (),
) -> Tuple[str, ...]:
    if explicit:
        return tuple(dict.fromkeys(explicit))
    excluded_set = set(excluded)
    out: List[str] = []
    for column in fieldnames:
        if column in excluded_set:
            continue
        values = [_as_float(row.get(column)) for row in rows]
        if any(math.isfinite(value) for value in values):
            out.append(column)
    return tuple(out)


def _mean(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _median(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def _std(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return math.nan
    return float(np.std(arr, ddof=1))


def load_local_environment_rows(
    path: Path,
    frame_column: str = "frame",
    entity_column: str = "entity_id",
    class_column: str = "environment_class",
    time_column: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], List[str], Tuple[str, str, str, Optional[str]]]:
    """Load a local-environment sample table."""

    raw_rows, fieldnames = _read_csv_rows(path)
    frame_src = _pick_column(fieldnames, [frame_column, "frame", "Frame", "frame_index", "timestep", "step"])
    entity_src = _pick_column(fieldnames, [entity_column, "entity_id", "atom_id", "oxygen_id", "molecule_id"])
    class_src = _pick_column(
        fieldnames,
        [class_column, "environment_class", "local_environment", "class", "state"],
    )
    time_src = _pick_column(fieldnames, [time_column, "time", "time_ns", "Time", "Time(ns)", "t"], required=False)
    rows: List[Dict[str, object]] = []
    for index, raw in enumerate(raw_rows):
        entity_id = str(raw.get(entity_src, "")).strip()
        env_class = str(raw.get(class_src, "")).strip()
        if not entity_id:
            raise ValueError(f"Missing entity id at source row {index}")
        if not env_class:
            raise ValueError(f"Missing environment class at source row {index}")
        item: Dict[str, object] = dict(raw)
        item["frame"] = _as_int(raw.get(frame_src))
        item["entity_id"] = entity_id
        item["environment_class"] = env_class
        item["time"] = _as_float(raw.get(time_src)) if time_src is not None else math.nan
        item["source_row_index"] = index
        rows.append(item)
    rows.sort(key=lambda row: (_as_int(row.get("frame")), str(row.get("entity_id"))))
    return rows, fieldnames, (frame_src, entity_src, class_src, time_src)


def infer_class_order(rows: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for row in rows:
        label = str(row.get("environment_class", ""))
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return tuple(out)


def _class_counts(rows: Sequence[Mapping[str, object]], class_order: Sequence[str]) -> Dict[str, int]:
    counts = {label: 0 for label in class_order}
    for row in rows:
        label = str(row.get("environment_class", ""))
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def build_frame_environment_summary(
    rows: Sequence[Mapping[str, object]],
    class_order: Sequence[str],
    feature_columns: Sequence[str],
) -> List[Dict[str, object]]:
    """Summarize environment classes and numeric features per frame."""

    grouped: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_as_int(row.get("frame"))].append(row)
    out: List[Dict[str, object]] = []
    for frame in sorted(grouped):
        chunk = grouped[frame]
        total = max(len(chunk), 1)
        time_values = [_as_float(row.get("time")) for row in chunk if math.isfinite(_as_float(row.get("time")))]
        item: Dict[str, object] = {
            "frame": frame,
            "time": _format_float(time_values[0] if time_values else math.nan),
            "n_samples": len(chunk),
            "n_entities": len({str(row.get("entity_id")) for row in chunk}),
        }
        for label, count in _class_counts(chunk, class_order).items():
            item[f"class_count__{label}"] = count
            item[f"class_fraction__{label}"] = count / total
        for column in feature_columns:
            values = [row.get(column) for row in chunk]
            item[f"{column}_mean"] = _format_float(_mean(values))
            item[f"{column}_median"] = _format_float(_median(values))
            item[f"{column}_std"] = _format_float(_std(values))
        out.append(item)
    return out


def build_class_environment_summary(
    rows: Sequence[Mapping[str, object]],
    class_order: Sequence[str],
    feature_columns: Sequence[str],
) -> List[Dict[str, object]]:
    """Summarize numeric features grouped by environment class."""

    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("environment_class", ""))].append(row)
    labels = list(dict.fromkeys(list(class_order) + sorted(grouped)))
    total = max(len(rows), 1)
    out: List[Dict[str, object]] = []
    for label in labels:
        chunk = grouped.get(label, [])
        item: Dict[str, object] = {
            "environment_class": label,
            "count": len(chunk),
            "fraction": len(chunk) / total,
            "n_entities": len({str(row.get("entity_id")) for row in chunk}),
        }
        for column in feature_columns:
            values = [row.get(column) for row in chunk]
            item[f"{column}_mean"] = _format_float(_mean(values))
            item[f"{column}_median"] = _format_float(_median(values))
            item[f"{column}_std"] = _format_float(_std(values))
        out.append(item)
    return out


def _standardized_sample_rows(rows: Sequence[Mapping[str, object]], feature_columns: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        item: Dict[str, object] = {
            "frame": row.get("frame"),
            "time": _format_float(_as_float(row.get("time"))),
            "entity_id": row.get("entity_id"),
            "environment_class": row.get("environment_class"),
            "source_row_index": row.get("source_row_index"),
        }
        for column in feature_columns:
            item[column] = _format_float(_as_float(row.get(column)))
        out.append(item)
    return out


def _transition_rows(rows: Sequence[Mapping[str, object]], class_order: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    state_rows = [
        SpeciesStateRow(
            frame=_as_int(row.get("frame")),
            entity_id=str(row.get("entity_id")),
            species=str(row.get("environment_class")),
            time=_as_float(row.get("time")),
            source_row_index=_as_int(row.get("source_row_index")),
        )
        for row in rows
    ]
    result = build_species_transition_matrix(state_rows, species_order=class_order)
    counts = transition_matrix_to_long_rows(result.counts, result.species_order, "count")
    probabilities = transition_matrix_to_long_rows(result.probabilities, result.species_order, "probability")
    return counts, probabilities


def analyze_local_environment(
    input_csv: Path,
    output_dir: Path,
    config: LocalEnvironmentConfig = LocalEnvironmentConfig(),
    frame_column: str = "frame",
    entity_column: str = "entity_id",
    class_column: str = "environment_class",
    time_column: Optional[str] = None,
    feature_columns: Sequence[str] = (),
) -> Dict[str, Path]:
    """Run local-environment summaries and transition matrices."""

    rows, fieldnames, selected = load_local_environment_rows(
        input_csv,
        frame_column=frame_column,
        entity_column=entity_column,
        class_column=class_column,
        time_column=time_column,
    )
    frame_src, entity_src, class_src, time_src = selected
    inferred_features = _infer_feature_columns(
        rows,
        fieldnames,
        excluded=[frame_src, entity_src, class_src, time_src or "", "source_row_index"],
        explicit=feature_columns,
    )
    class_order = config.class_order or infer_class_order(rows)
    counts, probabilities = _transition_rows(rows, class_order)

    output_dir = Path(output_dir)
    outputs = {
        "sample_table": output_dir / "local_environment_samples.csv",
        "frame_summary": output_dir / "local_environment_frame_summary.csv",
        "class_summary": output_dir / "local_environment_class_summary.csv",
        "transition_counts": output_dir / "local_environment_transition_counts.csv",
        "transition_probabilities": output_dir / "local_environment_transition_probabilities.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    sample_fieldnames = list(SAMPLE_COLUMNS) + list(inferred_features)
    _write_csv_rows(outputs["sample_table"], _standardized_sample_rows(rows, inferred_features), fieldnames=sample_fieldnames)
    _write_csv_rows(outputs["frame_summary"], build_frame_environment_summary(rows, class_order, inferred_features))
    _write_csv_rows(outputs["class_summary"], build_class_environment_summary(rows, class_order, inferred_features))
    _write_csv_rows(outputs["transition_counts"], counts, fieldnames=["from_species", "to_species", "count"])
    _write_csv_rows(outputs["transition_probabilities"], probabilities, fieldnames=["from_species", "to_species", "probability"])
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_csv", "value": str(input_csv)},
            {"metric": "n_samples", "value": len(rows)},
            {"metric": "n_entities", "value": len({str(row.get("entity_id")) for row in rows})},
            {"metric": "n_frames", "value": len({_as_int(row.get("frame")) for row in rows})},
            {"metric": "class_order", "value": ",".join(class_order)},
            {"metric": "feature_columns", "value": ",".join(inferred_features)},
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local-environment classes from explicit sample tables")
    parser.add_argument("--input", type=Path, required=True, help="Input local-environment sample CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--entity-column", default="entity_id")
    parser.add_argument("--class-column", default="environment_class")
    parser.add_argument("--time-column")
    parser.add_argument("--feature-column", action="append", help="Feature column; may be repeated or comma separated")
    parser.add_argument("--class-order", action="append", help="Class order; may be repeated or comma separated")
    return parser


def _parse_columns(raw: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not raw:
        return ()
    out: List[str] = []
    for value in raw:
        for item in re.split(r"[,;|]", str(value)):
            item = item.strip()
            if item:
                out.append(item)
    return tuple(dict.fromkeys(out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_local_environment(
            input_csv=args.input,
            output_dir=args.output_dir,
            config=LocalEnvironmentConfig(class_order=parse_class_order(args.class_order)),
            frame_column=args.frame_column,
            entity_column=args.entity_column,
            class_column=args.class_column,
            time_column=args.time_column,
            feature_columns=_parse_columns(args.feature_column),
        )
    except Exception as exc:
        print(f"Local-environment analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
