"""Species-state transition matrices from long-form state tables."""

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


STATE_OUTPUT_COLUMNS = ("frame", "time", "entity_id", "species", "source_row_index")
DETAIL_OUTPUT_COLUMNS = (
    "from_frame",
    "to_frame",
    "frame_gap",
    "from_time",
    "to_time",
    "time_delta",
    "entity_id",
    "from_species",
    "to_species",
    "transition_changed",
    "from_source_row_index",
    "to_source_row_index",
)


@dataclass(frozen=True)
class SpeciesStateRow:
    """A single entity species assignment at one frame."""

    frame: int
    entity_id: str
    species: str
    time: float = math.nan
    source_row_index: int = -1


@dataclass(frozen=True)
class SpeciesTransitionResult:
    """Transition matrix and matched-frame transition details."""

    species_order: Tuple[str, ...]
    counts: np.ndarray
    probabilities: np.ndarray
    details: Tuple[Dict[str, object], ...]
    n_state_rows: int
    n_frames: int
    n_entities: int


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
        "from_species",
        "to_species",
        "count",
        "probability",
        "species",
        "state_rows",
        "state_fraction",
        "entity_count",
        "frame_count",
        "transition_out_count",
        "transition_in_count",
        "changed_out_count",
        "changed_in_count",
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


def _as_frame(value: object) -> int:
    number = _as_float(value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid frame value: {value!r}")
    return int(round(number))


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[Optional[str]], required: bool = True) -> Optional[str]:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    if required:
        requested = ", ".join(str(candidate) for candidate in candidates if candidate)
        raise ValueError(f"Could not find required column among [{requested}]; available columns: {', '.join(fieldnames)}")
    return None


def _format_float(value: float) -> object:
    return value if math.isfinite(value) else ""


def parse_species_order(raw: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Parse repeated or delimited species-order arguments."""

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


def load_species_state_rows(
    path: Path,
    frame_column: str = "frame",
    entity_column: str = "entity_id",
    species_column: str = "species",
    time_column: Optional[str] = None,
) -> List[SpeciesStateRow]:
    """Load a long-form species-state CSV.

    The table must identify a frame, a persistent entity id, and a species label.
    The entity is usually an oxygen atom index, but can be any stable id.
    """

    raw_rows, fieldnames = _read_csv_rows(path)
    frame_src = _pick_column(fieldnames, [frame_column, "frame", "Frame", "frame_index", "timestep", "step"])
    entity_src = _pick_column(
        fieldnames,
        [entity_column, "entity_id", "oxygen_index", "oxygen_id", "atom_id", "o_index", "molecule_id"],
    )
    species_src = _pick_column(fieldnames, [species_column, "species", "state", "species_name", "label"])
    time_src = _pick_column(
        fieldnames,
        [time_column, "time", "time_ns", "Time", "Time(ns)", "t"],
        required=False,
    )

    rows: List[SpeciesStateRow] = []
    for index, raw in enumerate(raw_rows):
        entity_id = str(raw.get(entity_src, "")).strip()
        species = str(raw.get(species_src, "")).strip()
        if not entity_id:
            raise ValueError(f"Missing entity id at source row {index}")
        if not species:
            raise ValueError(f"Missing species label at source row {index}")
        rows.append(
            SpeciesStateRow(
                frame=_as_frame(raw.get(frame_src)),
                entity_id=entity_id,
                species=species,
                time=_as_float(raw.get(time_src)) if time_src is not None else math.nan,
                source_row_index=index,
            )
        )
    rows.sort(key=lambda row: (row.frame, row.entity_id))
    return rows


def infer_species_order(rows: Sequence[SpeciesStateRow]) -> Tuple[str, ...]:
    """Infer species order from first appearance in the state table."""

    out: List[str] = []
    seen = set()
    for row in rows:
        if row.species not in seen:
            seen.add(row.species)
            out.append(row.species)
    return tuple(out)


def _frame_maps(rows: Sequence[SpeciesStateRow]) -> Dict[int, Dict[str, SpeciesStateRow]]:
    frames: Dict[int, Dict[str, SpeciesStateRow]] = defaultdict(dict)
    for row in rows:
        frame = frames[row.frame]
        if row.entity_id in frame:
            raise ValueError(f"Duplicate entity {row.entity_id!r} in frame {row.frame}")
        frame[row.entity_id] = row
    return dict(frames)


def transition_probabilities(counts: np.ndarray) -> np.ndarray:
    """Normalize transition counts by source-species row totals."""

    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0.0)
    return probabilities


def build_species_transition_matrix(
    rows: Sequence[SpeciesStateRow],
    species_order: Sequence[str] = (),
    include_self_transitions: bool = True,
) -> SpeciesTransitionResult:
    """Count species transitions for matched entities in adjacent frames."""

    order = tuple(species_order) if species_order else infer_species_order(rows)
    if not order:
        counts = np.zeros((0, 0), dtype=int)
        return SpeciesTransitionResult(
            species_order=order,
            counts=counts,
            probabilities=transition_probabilities(counts),
            details=(),
            n_state_rows=0,
            n_frames=0,
            n_entities=0,
        )
    index_by_species = {species: index for index, species in enumerate(order)}
    unknown = sorted({row.species for row in rows if row.species not in index_by_species})
    if unknown:
        raise ValueError(f"Species missing from species_order: {', '.join(unknown)}")

    frames = _frame_maps(rows)
    frame_ids = sorted(frames)
    counts = np.zeros((len(order), len(order)), dtype=int)
    details: List[Dict[str, object]] = []
    entities = {row.entity_id for row in rows}

    for from_frame, to_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current = frames[from_frame]
        next_frame = frames[to_frame]
        for entity_id in sorted(set(current) & set(next_frame)):
            source = current[entity_id]
            target = next_frame[entity_id]
            changed = source.species != target.species
            if changed or include_self_transitions:
                counts[index_by_species[source.species], index_by_species[target.species]] += 1
            time_delta = target.time - source.time if math.isfinite(source.time) and math.isfinite(target.time) else math.nan
            details.append(
                {
                    "from_frame": source.frame,
                    "to_frame": target.frame,
                    "frame_gap": target.frame - source.frame,
                    "from_time": _format_float(source.time),
                    "to_time": _format_float(target.time),
                    "time_delta": _format_float(time_delta),
                    "entity_id": entity_id,
                    "from_species": source.species,
                    "to_species": target.species,
                    "transition_changed": int(changed),
                    "from_source_row_index": source.source_row_index,
                    "to_source_row_index": target.source_row_index,
                }
            )

    return SpeciesTransitionResult(
        species_order=order,
        counts=counts,
        probabilities=transition_probabilities(counts),
        details=tuple(details),
        n_state_rows=len(rows),
        n_frames=len(frame_ids),
        n_entities=len(entities),
    )


def transition_matrix_to_long_rows(
    matrix: np.ndarray,
    species_order: Sequence[str],
    value_name: str,
) -> List[Dict[str, object]]:
    """Convert a square species matrix to long-form rows."""

    out: List[Dict[str, object]] = []
    for from_index, from_species in enumerate(species_order):
        for to_index, to_species in enumerate(species_order):
            value = matrix[from_index, to_index]
            if value_name == "count":
                value = int(value)
            else:
                value = float(value)
            out.append({"from_species": from_species, "to_species": to_species, value_name: value})
    return out


def summarize_species_states(
    rows: Sequence[SpeciesStateRow],
    result: SpeciesTransitionResult,
) -> List[Dict[str, object]]:
    """Summarize state occupancy and transition flow by species."""

    state_rows_by_species: Dict[str, int] = {species: 0 for species in result.species_order}
    entities_by_species: Dict[str, set] = {species: set() for species in result.species_order}
    frames_by_species: Dict[str, set] = {species: set() for species in result.species_order}
    for row in rows:
        state_rows_by_species[row.species] += 1
        entities_by_species[row.species].add(row.entity_id)
        frames_by_species[row.species].add(row.frame)

    changed_out: Dict[str, int] = {species: 0 for species in result.species_order}
    changed_in: Dict[str, int] = {species: 0 for species in result.species_order}
    for detail in result.details:
        if int(detail["transition_changed"]) == 0:
            continue
        changed_out[str(detail["from_species"])] += 1
        changed_in[str(detail["to_species"])] += 1

    total_state_rows = max(len(rows), 1)
    out: List[Dict[str, object]] = []
    for index, species in enumerate(result.species_order):
        out.append(
            {
                "species": species,
                "state_rows": state_rows_by_species[species],
                "state_fraction": state_rows_by_species[species] / total_state_rows,
                "entity_count": len(entities_by_species[species]),
                "frame_count": len(frames_by_species[species]),
                "transition_out_count": int(result.counts[index, :].sum()),
                "transition_in_count": int(result.counts[:, index].sum()),
                "changed_out_count": changed_out[species],
                "changed_in_count": changed_in[species],
            }
        )
    return out


def _state_rows_to_csv_rows(rows: Sequence[SpeciesStateRow]) -> List[Dict[str, object]]:
    return [
        {
            "frame": row.frame,
            "time": _format_float(row.time),
            "entity_id": row.entity_id,
            "species": row.species,
            "source_row_index": row.source_row_index,
        }
        for row in rows
    ]


def analyze_species_transitions(
    input_csv: Path,
    output_dir: Path,
    frame_column: str = "frame",
    entity_column: str = "entity_id",
    species_column: str = "species",
    time_column: Optional[str] = None,
    species_order: Sequence[str] = (),
    include_self_transitions: bool = True,
) -> Dict[str, Path]:
    """Run species transition analysis and write standard CSV outputs."""

    rows = load_species_state_rows(
        input_csv,
        frame_column=frame_column,
        entity_column=entity_column,
        species_column=species_column,
        time_column=time_column,
    )
    result = build_species_transition_matrix(
        rows,
        species_order=species_order,
        include_self_transitions=include_self_transitions,
    )

    output_dir = Path(output_dir)
    outputs = {
        "state_table": output_dir / "species_state_table.csv",
        "transition_counts": output_dir / "species_transition_counts.csv",
        "transition_probabilities": output_dir / "species_transition_probabilities.csv",
        "transition_details": output_dir / "species_transition_details.csv",
        "species_state_summary": output_dir / "species_state_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["state_table"], _state_rows_to_csv_rows(rows), fieldnames=STATE_OUTPUT_COLUMNS)
    _write_csv_rows(
        outputs["transition_counts"],
        transition_matrix_to_long_rows(result.counts, result.species_order, "count"),
        fieldnames=["from_species", "to_species", "count"],
    )
    _write_csv_rows(
        outputs["transition_probabilities"],
        transition_matrix_to_long_rows(result.probabilities, result.species_order, "probability"),
        fieldnames=["from_species", "to_species", "probability"],
    )
    _write_csv_rows(outputs["transition_details"], result.details, fieldnames=DETAIL_OUTPUT_COLUMNS)
    _write_csv_rows(outputs["species_state_summary"], summarize_species_states(rows, result))
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_csv", "value": str(input_csv)},
            {"metric": "frame_column", "value": frame_column},
            {"metric": "entity_column", "value": entity_column},
            {"metric": "species_column", "value": species_column},
            {"metric": "time_column", "value": time_column or ""},
            {"metric": "species_order", "value": ",".join(result.species_order)},
            {"metric": "include_self_transitions", "value": int(include_self_transitions)},
            {"metric": "n_state_rows", "value": result.n_state_rows},
            {"metric": "n_frames", "value": result.n_frames},
            {"metric": "n_entities", "value": result.n_entities},
            {"metric": "n_matched_transitions", "value": len(result.details)},
            {
                "metric": "n_changed_transitions",
                "value": sum(int(detail["transition_changed"]) for detail in result.details),
            },
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build species transition matrices from a long-form state CSV")
    parser.add_argument("--input", type=Path, required=True, help="Input long-form species-state CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--entity-column", default="entity_id")
    parser.add_argument("--species-column", default="species")
    parser.add_argument("--time-column")
    parser.add_argument("--species-order", action="append", help="Species order; may be repeated or comma separated")
    parser.add_argument(
        "--no-self-transitions",
        dest="include_self_transitions",
        action="store_false",
        help="Do not count unchanged adjacent-frame species assignments in the matrix",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_species_transitions(
            input_csv=args.input,
            output_dir=args.output_dir,
            frame_column=args.frame_column,
            entity_column=args.entity_column,
            species_column=args.species_column,
            time_column=args.time_column,
            species_order=parse_species_order(args.species_order),
            include_self_transitions=args.include_self_transitions,
        )
    except Exception as exc:
        print(f"Species transition analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
