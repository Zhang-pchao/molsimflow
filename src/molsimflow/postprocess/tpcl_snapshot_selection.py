"""Select paired TPCL event/control snapshots without inspecting force predictions."""

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
    "FROZEN_PRE_PREDICTION_EVENT_CONTROL_SNAPSHOT_SELECTION_"
    "NOT_CAUSAL_OR_REPLICATE_LEVEL_EVIDENCE"
)

MATCH_FIELDS = (
    "global_mean_radius_A",
    "global_mode_2_amplitude_A",
    "global_mode_3_amplitude_A",
    "global_mode_4_amplitude_A",
    "global_unresolved_mode_rms_A",
    "global_footprint_area_A2",
    "global_footprint_circularity",
    "global_cap_angle_candidate_deg",
)


@dataclass(frozen=True)
class CaseSource:
    case_id: str
    event_environment: Path
    bubble_state: Path


@dataclass(frozen=True)
class SelectionConfig:
    pairs_per_surface: int = 12
    response_strata: int = 3
    local_null_arc_radius: int = 3
    local_null_half_window_ps: float = 5.0
    frame_step_stride: int = 1000
    patch_radius_A: float = 6.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], *, delimiter: str = ","
) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def read_case_sources(path: Path) -> list[CaseSource]:
    rows = _read_csv(path, delimiter="\t")
    sources = [
        CaseSource(
            case_id=row["case_id"],
            event_environment=Path(row["event_environment"]),
            bubble_state=Path(row["bubble_state"]),
        )
        for row in rows
    ]
    case_ids = [source.case_id for source in sources]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case sources are not unique")
    return sources


def _circular_distance(left: int, right: int, count: int) -> int:
    direct = abs(left - right)
    return min(direct, count - direct)


def _eligible_local_null(
    control: Mapping[str, str],
    events: Sequence[Mapping[str, str]],
    config: SelectionConfig,
    arc_count: int,
) -> bool:
    time_ps = float(control["sample_time_ns"]) * 1000.0
    arc = int(control["primary_arc_index"])
    for event in events:
        if abs(float(event["transition_time_ns"]) * 1000.0 - time_ps) > config.local_null_half_window_ps:
            continue
        if (
            _circular_distance(arc, int(event["primary_arc_index"]), arc_count)
            <= config.local_null_arc_radius
        ):
            return False
    return True


def _select_events(
    rows: Sequence[Mapping[str, str]], config: SelectionConfig
) -> list[tuple[dict[str, str], int]]:
    ordered = sorted(rows, key=lambda row: (float(row["response_affected_arc_fraction"]), int(row["cluster_id"])))
    strata = np.array_split(np.arange(len(ordered)), config.response_strata)
    base = config.pairs_per_surface // config.response_strata
    remainder = config.pairs_per_surface % config.response_strata
    selected: list[tuple[dict[str, str], int]] = []
    used_blocks = set()
    for stratum_index, indices in enumerate(strata):
        target = base + int(stratum_index < remainder)
        center = float(np.median(indices))
        candidates = sorted(indices, key=lambda index: (abs(float(index) - center), int(index)))
        accepted = 0
        for index in candidates:
            row = ordered[int(index)]
            block = int(row["time_block_200ps"])
            if block in used_blocks:
                continue
            selected.append((dict(row), stratum_index))
            used_blocks.add(block)
            accepted += 1
            if accepted == target:
                break
        if accepted != target:
            raise ValueError(f"could not select {target} unique blocks in stratum {stratum_index}")
    if len(selected) != config.pairs_per_surface:
        raise ValueError("unexpected selected event count")
    return selected


def _environment_index(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, str, int, int], Mapping[str, str]]:
    output = {}
    for row in rows:
        key = (
            int(row["event_id"]),
            row["sample_kind"],
            int(row["circular_shift_frames"]),
            int(row["relative_frame"]),
        )
        if key in output:
            raise ValueError(f"duplicate environment key {key}")
        output[key] = row
    return output


def _bubble_index(rows: Sequence[Mapping[str, str]]) -> dict[int, Mapping[str, str]]:
    output = {}
    for row in rows:
        step = int(row["step"])
        if step in output:
            raise ValueError(f"duplicate bubble-state step {step}")
        output[step] = row
    return output


def _match_scale(rows: Sequence[Mapping[str, str]]) -> dict[str, tuple[float, float]]:
    output = {}
    for field in MATCH_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            output[field] = (0.0, 1.0)
            continue
        median = float(np.median(finite))
        scale = float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
        output[field] = (median, scale if scale > 1.0e-12 else 1.0)
    return output


def _match_distance(
    event: Mapping[str, str], control: Mapping[str, str], scale: Mapping[str, tuple[float, float]]
) -> float:
    terms = []
    for field in MATCH_FIELDS:
        left, right = float(event[field]), float(control[field])
        if math.isfinite(left) and math.isfinite(right):
            terms.append(((left - right) / scale[field][1]) ** 2)
    if not terms:
        raise ValueError("event/control pair has no finite matching fields")
    return float(math.sqrt(np.mean(terms)))


def _radial_direction(contact: np.ndarray, center: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = contact - center
    delta -= box * np.round(delta / box)
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-12:
        raise ValueError("contact point and bubble center coincide")
    return delta / norm


def _snapshot_rows(
    pair_id: str,
    case_id: str,
    sample_kind: str,
    event_id: int,
    shift: int,
    anchor_step: int,
    environment: Mapping[tuple[int, str, int, int], Mapping[str, str]],
    bubble: Mapping[int, Mapping[str, str]],
    pair_metadata: Mapping[str, object],
    config: SelectionConfig,
) -> list[dict[str, object]]:
    output = []
    for phase, relative in (("pre", -1), ("transition", 0), ("post", 1)):
        step = anchor_step + relative * config.frame_step_stride
        key = (event_id, sample_kind, shift, relative)
        if key not in environment or step not in bubble:
            raise ValueError(f"{pair_id}/{sample_kind}/{phase}: missing source state")
        local = environment[key]
        state = bubble[step]
        box = np.asarray([float(state[field]) for field in ("box_x_A", "box_y_A")])
        contact = np.asarray([float(local[field]) for field in ("contact_x_A", "contact_y_A")])
        center = np.asarray(
            [float(state[field]) for field in ("bubble_center_x_A", "bubble_center_y_A")]
        )
        radial = _radial_direction(contact, center, box)
        output.append(
            {
                "pair_id": pair_id,
                "case_id": case_id,
                "sample_kind": sample_kind,
                "phase": phase,
                "step": step,
                "dump_path": state["source_file"],
                "source_frame": int(state["source_frame"]),
                "contact_x_A": contact[0],
                "contact_y_A": contact[1],
                "bubble_center_x_A": center[0],
                "bubble_center_y_A": center[1],
                "radial_x": radial[0],
                "radial_y": radial[1],
                "patch_radius_A": config.patch_radius_A,
                **pair_metadata,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    return output


def run_selection(
    event_state_table: Path,
    risk_table: Path,
    case_sources_path: Path,
    output_dir: Path,
    *,
    config: Optional[SelectionConfig] = None,
) -> dict[str, object]:
    config = config or SelectionConfig()
    if config.pairs_per_surface < config.response_strata or config.frame_step_stride <= 0:
        raise ValueError("invalid selection configuration")
    event_rows = _read_csv(event_state_table)
    risk_rows = _read_csv(risk_table)
    sources = read_case_sources(case_sources_path)
    if {source.case_id for source in sources} != {row["case_id"] for row in event_rows}:
        raise ValueError("case source identities differ from event table")

    pair_rows = []
    snapshots = []
    for source in sources:
        case_events = [row for row in event_rows if row["case_id"] == source.case_id]
        case_risk = [row for row in risk_rows if row["case_id"] == source.case_id]
        risk_by_cluster = defaultdict(list)
        for row in case_risk:
            risk_by_cluster[int(row["source_cluster_id"])].append(row)
        environment = _environment_index(_read_csv(source.event_environment))
        bubble = _bubble_index(_read_csv(source.bubble_state))
        scale = _match_scale(case_risk)
        selected = _select_events(case_events, config)
        arc_counts = {int(row["arc_count"]) for row in case_events}
        if len(arc_counts) != 1:
            raise ValueError(f"{source.case_id}: inconsistent arc count")
        arc_count = arc_counts.pop()
        for sequence, (event, response_stratum) in enumerate(selected, start=1):
            cluster_id = int(event["cluster_id"])
            risk_set = risk_by_cluster[cluster_id]
            event_candidates = [row for row in risk_set if int(row["is_event"]) == 1]
            controls = [
                row
                for row in risk_set
                if int(row["is_event"]) == 0
                and _eligible_local_null(row, case_events, config, arc_count)
                and int(row["sample_step"]) - config.frame_step_stride in bubble
                and int(row["sample_step"]) + config.frame_step_stride in bubble
            ]
            if len(event_candidates) != 1 or not controls:
                raise ValueError(f"{source.case_id}/cluster {cluster_id}: no valid matched risk set")
            event_risk = event_candidates[0]
            control = min(
                controls,
                key=lambda row: (
                    _match_distance(event_risk, row, scale),
                    abs(int(row["control_shift_frames"])),
                    int(row["sample_step"]),
                ),
            )
            distance = _match_distance(event_risk, control, scale)
            pair_id = f"{source.case_id}__{sequence:02d}"
            metadata = {
                "source_cluster_id": cluster_id,
                "primary_event_id": int(event["primary_event_id"]),
                "primary_arc_index": int(event["primary_arc_index"]),
                "source_time_block_200ps": int(event["time_block_200ps"]),
                "response_stratum": response_stratum,
                "response_affected_arc_fraction": float(
                    event["response_affected_arc_fraction"]
                ),
                "geometry_match_distance": distance,
            }
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "case_id": source.case_id,
                    **metadata,
                    "event_anchor_step": int(event_risk["sample_step"]),
                    "control_anchor_step": int(control["sample_step"]),
                    "control_shift_frames": int(control["control_shift_frames"]),
                    "local_null_arc_radius": config.local_null_arc_radius,
                    "local_null_half_window_ps": config.local_null_half_window_ps,
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
            snapshots.extend(
                _snapshot_rows(
                    pair_id,
                    source.case_id,
                    "event",
                    int(event["primary_event_id"]),
                    0,
                    int(event_risk["sample_step"]),
                    environment,
                    bubble,
                    metadata,
                    config,
                )
            )
            snapshots.extend(
                _snapshot_rows(
                    pair_id,
                    source.case_id,
                    "circular_shift_control",
                    int(event["primary_event_id"]),
                    int(control["control_shift_frames"]),
                    int(control["sample_step"]),
                    environment,
                    bubble,
                    metadata,
                    config,
                )
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "selected_pairs.csv", pair_rows)
    _write_csv(output / "snapshot_manifest.tsv", snapshots, delimiter="\t")
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "pair_count": len(pair_rows),
        "snapshot_count": len(snapshots),
        "pairs_per_surface": config.pairs_per_surface,
        "response_strata": config.response_strata,
        "phases": ["pre", "transition", "post"],
        "sample_kinds": ["event", "circular_shift_control"],
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "event_state_table": {
                    "path": str(event_state_table),
                    "sha256": _sha256(event_state_table),
                },
                "risk_table": {"path": str(risk_table), "sha256": _sha256(risk_table)},
                "case_sources": {"path": str(case_sources_path), "sha256": _sha256(case_sources_path)},
                "source_files": [
                    {
                        "case_id": source.case_id,
                        "event_environment": {
                            "path": str(source.event_environment),
                            "sha256": _sha256(source.event_environment),
                        },
                        "bubble_state": {
                            "path": str(source.bubble_state),
                            "sha256": _sha256(source.bubble_state),
                        },
                    }
                    for source in sources
                ],
                "config": config.__dict__,
                "matching_fields": MATCH_FIELDS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "REPORT.md").write_text(
        "# Frozen TPCL snapshot selection\n\n"
        "Twelve event clusters per surface are sampled across three continuous "
        "affected-arc strata with at most one event per 200 ps block. Each event is "
        "paired to a same-arc circular-shift control selected from geometry-only "
        "pre-state matching after excluding nearby local events. Selection is frozen "
        "before any force, energy, or virial prediction. The controls are retrospective "
        "matched states, not causal counterfactuals or independent trajectories.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-state-table", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--case-sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-surface", type=int, default=12)
    parser.add_argument("--response-strata", type=int, default=3)
    parser.add_argument("--local-null-arc-radius", type=int, default=3)
    parser.add_argument("--local-null-half-window-ps", type=float, default=5.0)
    parser.add_argument("--frame-step-stride", type=int, default=1000)
    parser.add_argument("--patch-radius-A", type=float, default=6.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_selection(
        args.event_state_table,
        args.risk_table,
        args.case_sources,
        args.output_dir,
        config=SelectionConfig(
            pairs_per_surface=args.pairs_per_surface,
            response_strata=args.response_strata,
            local_null_arc_radius=args.local_null_arc_radius,
            local_null_half_window_ps=args.local_null_half_window_ps,
            frame_step_stride=args.frame_step_stride,
            patch_radius_A=args.patch_radius_A,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
