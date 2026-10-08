"""Build matched event/control risk rows for TPCL yielding prediction."""

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
from pathlib import Path
from typing import Optional

import numpy as np

from molsimflow.postprocess.tpcl_event_state import (
    GLOBAL_FIELDS,
    EventStateSource,
    read_sources,
)

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_MATCHED_EVENT_CONTROL_RISK_SET_"
    "NOT_CAUSAL_OR_REPLICATE_LEVEL_EVIDENCE"
)

RISK_LOCAL_FIELDS = (
    "nearest_site_distance_A",
    "local_site_count",
    "local_ch3_fraction",
    "nearest_ch3_distance_A",
    "nearest_sioh_distance_A",
    "chemical_boundary_distance_proxy_A",
    "local_hydration_areal_density_A-2",
    "local_water_dipole_cos_z",
    "local_water_water_hbond_degree",
    "local_surface_water_hbond_per_h2o",
    "local_n2_min_distance_A",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
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


def _finite(value: object, *, allow_nan: bool = False) -> float:
    number = float(value)
    if math.isfinite(number) or (allow_nan and math.isnan(number)):
        return number
    raise ValueError(f"expected finite value, got {value!r}")


def _index(rows: Sequence[Mapping[str, str]], field: str, path: Path) -> dict[int, Mapping[str, str]]:
    output = {}
    for row in rows:
        key = round(_finite(row[field]))
        if key in output:
            raise ValueError(f"duplicate {field}={key} in {path}")
        output[key] = row
    return output


def _aggregate_local(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    pre = [row for row in rows if int(row["relative_frame"]) in {-4, -3, -2, -1}]
    if len(pre) != 4:
        raise ValueError("risk sample does not contain four pre-event frames")
    output: dict[str, object] = {"pre_local_sample_count": len(pre)}
    for field in RISK_LOCAL_FIELDS:
        values = np.asarray([_finite(row[field], allow_nan=True) for row in pre], dtype=float)
        finite = values[np.isfinite(values)]
        output[f"pre_local_{field}"] = float(np.mean(finite)) if finite.size else math.nan
        output[f"pre_local_{field}_finite_count"] = int(finite.size)
    return output


def build_case_rows(
    source: EventStateSource,
    *,
    block_ps: float = 200.0,
    time_ps_per_step: float,
) -> list[dict[str, object]]:
    """Build weighted event/control risk rows for primary event clusters."""

    if block_ps <= 0.0 or time_ps_per_step <= 0.0:
        raise ValueError("block_ps and time_ps_per_step must be positive")

    summary = json.loads(source.propagation_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("case_id") != source.case_id:
        raise ValueError(f"{source.case_id}: propagation summary is not a matching PASS")
    start_ps = _finite(summary["first_time_ns"]) * 1000.0
    stop_ps = _finite(summary["last_time_ns"]) * 1000.0
    clusters = _read_csv(source.event_size_metrics)
    primary = {int(row["primary_event_id"]): row for row in clusters}
    if len(primary) != int(summary["event_cluster_count"]):
        raise ValueError(f"{source.case_id}: primary event identity mismatch")
    modes = _index(_read_csv(source.frame_modes), "step", source.frame_modes)
    geometry = _index(_read_csv(source.geometry), "step", source.geometry)
    stress = _index(_read_csv(source.global_stress), "step", source.global_stress)
    environment = _read_csv(source.event_environment)
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in environment:
        event_id = int(row["event_id"])
        if event_id in primary:
            grouped[(event_id, row["sample_kind"], int(row["circular_shift_frames"]))].append(row)

    output = []
    for event_id, cluster in primary.items():
        keys = [key for key in grouped if key[0] == event_id]
        event_keys = [key for key in keys if key[1] == "event"]
        control_keys = [key for key in keys if key[1] == "circular_shift_control"]
        if len(event_keys) != 1 or not control_keys:
            raise ValueError(f"{source.case_id}/event {event_id}: invalid risk-set groups")
        for key in event_keys + sorted(control_keys):
            samples = grouped[key]
            transition_rows = [row for row in samples if int(row["relative_frame"]) == 0]
            pre_rows = [row for row in samples if int(row["relative_frame"]) == -1]
            if len(transition_rows) != 1 or len(pre_rows) != 1:
                raise ValueError(f"{source.case_id}/event {event_id}: incomplete anchor")
            sample_step = int(transition_rows[0]["step"])
            pre_step = int(pre_rows[0]["step"])
            if pre_step not in modes or pre_step not in geometry or pre_step not in stress:
                raise ValueError(f"{source.case_id}/event {event_id}: missing aligned global state")
            sample_time_ps = sample_step * time_ps_per_step
            if not start_ps <= sample_time_ps <= stop_ps:
                raise ValueError(f"{source.case_id}/event {event_id}: sample outside support")
            sample_block = min(
                math.floor((sample_time_ps - start_ps) / block_ps),
                math.ceil((stop_ps - start_ps) / block_ps) - 1,
            )
            source_time_ps = _finite(cluster["transition_time_ns"]) * 1000.0
            source_block = min(
                math.floor((source_time_ps - start_ps) / block_ps),
                math.ceil((stop_ps - start_ps) / block_ps) - 1,
            )
            rows_by_source = {
                "frame_modes": modes[pre_step],
                "geometry": geometry[pre_step],
                "global_stress": stress[pre_step],
            }
            is_event = key[1] == "event"
            row: dict[str, object] = {
                "case_id": source.case_id,
                "source_cluster_id": int(cluster["cluster_id"]),
                "primary_event_id": event_id,
                "primary_arc_index": int(cluster["primary_arc_index"]),
                "sample_kind": key[1],
                "is_event": int(is_event),
                "control_shift_frames": key[2],
                "sample_step": sample_step,
                "sample_time_ns": sample_time_ps / 1000.0,
                "sample_pre_step": pre_step,
                "sample_time_block_200ps": sample_block,
                "source_time_block_200ps": source_block,
                "risk_set_weight": 0.5 if is_event else 0.5 / len(control_keys),
            }
            for output_name, (source_name, input_name) in GLOBAL_FIELDS.items():
                row[output_name] = _finite(rows_by_source[source_name][input_name])
            row.update(_aggregate_local(samples))
            row["scientific_status"] = SCIENTIFIC_STATUS
            output.append(row)
    return output


def run_analysis(
    sources_path: Path,
    output_dir: Path,
    *,
    block_ps: float = 200.0,
    time_ps_per_step: float,
) -> dict[str, object]:
    """Build and write all matched risk sets."""

    if block_ps <= 0.0 or time_ps_per_step <= 0.0:
        raise ValueError("block_ps and time_ps_per_step must be positive")
    sources = read_sources(sources_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    case_counts = {}
    for source in sources:
        case_rows = build_case_rows(
            source,
            block_ps=block_ps,
            time_ps_per_step=time_ps_per_step,
        )
        rows.extend(case_rows)
        case_counts[source.case_id] = len(case_rows)
    _write_csv(output / "yielding_risk_sets.csv", rows)
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "row_count": len(rows),
        "event_row_count": sum(int(row["is_event"]) for row in rows),
        "control_row_count": sum(1 - int(row["is_event"]) for row in rows),
        "case_counts": case_counts,
        "block_ps": block_ps,
        "time_ps_per_step": time_ps_per_step,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        **summary,
        "sources_manifest": {"path": str(sources_path), "sha256": _sha256(sources_path)},
        "inputs": [
            {
                "case_id": source.case_id,
                **{
                    field: {"path": str(getattr(source, field)), "sha256": _sha256(getattr(source, field))}
                    for field in (
                        "propagation_summary",
                        "event_size_metrics",
                        "frame_modes",
                        "geometry",
                        "global_stress",
                        "event_environment",
                    )
                },
            }
            for source in sources
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "# TPCL yielding risk sets\n\n"
        "Each accepted primary event cluster contributes one event row and all available "
        "circular-shift controls. Event and aggregate control weight are each 0.5 per "
        "source cluster. Features use only four pre-anchor frames and aligned global "
        "state. These are retrospective matched controls, not independent trajectories "
        "or causal counterfactuals.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--time-ps-per-step", type=float, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        args.sources,
        args.output_dir,
        block_ps=args.block_ps,
        time_ps_per_step=args.time_ps_per_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
