"""Inventory existing contact-state tables without reconstructing coordinates.

Legacy time/epoch joins are explicitly weaker than raw-frame-hash joins. Grid
labels and molecular distance-shell contact remain separate observables.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any

from molsimflow.postprocess.surface_contact_process import sampled_episodes

GRID_STATES = (
    "distant", "unsupported_or_ambiguous", "water_separated",
    "direct_contact_candidate", "mixed",
)
MATCH_VALIDATION = "LEGACY_EXACT_TIME_EPOCH_INDEX_NOT_RAW_HASH_VALIDATED"


@dataclass(frozen=True)
class SourceEpoch:
    segment: str
    first_step: int
    last_step: int
    source_path: str


@dataclass(frozen=True)
class ContactStateCase:
    name: str
    case: str
    tier: str
    dt_ps: float
    origin_step: int
    origin_time_ps: float
    epochs: tuple[SourceEpoch, ...]
    periodic: tuple[bool, bool, bool]
    geometry_dt_ps: float = 1.0
    contact_dt_ps: float = 10.0
    distance_shells: tuple[float, ...] = (3.5, 4.0, 4.5)


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not Boolean")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    if not result.is_finite():
        raise ValueError(f"Nonfinite {label}")
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    numeric = _decimal(value, label)
    if numeric != numeric.to_integral_value() or numeric < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(numeric)


def _boolean(value: Any, label: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"{label} must be Boolean or exact True/False text")


def _required(row: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = sorted(set(fields) - row.keys())
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")


def _configuration(config: ContactStateCase) -> tuple[dict[str, SourceEpoch], set[Decimal]]:
    if not all(isinstance(v, str) and v for v in (config.name, config.case, config.tier)):
        raise ValueError("Nonempty name, case and tier are required")
    if len(config.periodic) != 3 or any(type(x) is not bool for x in config.periodic):
        raise ValueError("periodic must contain three explicit Boolean flags")
    for field in ("dt_ps", "geometry_dt_ps", "contact_dt_ps"):
        if _decimal(getattr(config, field), field) <= 0:
            raise ValueError(f"Positive {field} is required")
    _integer(config.origin_step, "origin_step")
    _decimal(config.origin_time_ps, "origin_time_ps")
    epochs = {}
    for epoch in config.epochs:
        first = _integer(epoch.first_step, "first_step")
        last = _integer(epoch.last_step, "last_step")
        if not epoch.segment or epoch.segment in epochs or not epoch.source_path or last < first:
            raise ValueError("Invalid or duplicate source epoch")
        epochs[epoch.segment] = epoch
    if not epochs:
        raise ValueError("At least one explicit source epoch is required")
    shells = {_decimal(value, "distance shell") for value in config.distance_shells}
    if (len(shells) != len(config.distance_shells) or len(shells) != 3
            or Decimal(4) not in shells or any(value <= 0 for value in shells)):
        raise ValueError("Three unique positive distance shells including 4 A are required")
    return epochs, shells


def inventory_contact_states(
    geometry_records: Iterable[Mapping[str, Any]],
    contact_records: Iterable[Mapping[str, Any]],
    *, config: ContactStateCase,
) -> dict[str, Any]:
    """Validate exact legacy joins and summarize sampled contact observations.

    Records may contain TSV strings. Geometry fields are retained verbatim;
    `time_ps`, `step`, `main_contact_4A` and new derived fields are normalized in
    joined rows. All geometry rows must satisfy the declared time/step mapping
    and partition closure. Contact rows for other names are ignored. Missing
    shells fail; missing whole contact times remain gaps and censor episodes.
    No nearest-time match, raw-frame revalidation or causal fit is performed.
    """
    epochs, shells = _configuration(config)
    origin = _decimal(config.origin_time_ps, "origin_time_ps")
    step_dt = _decimal(config.dt_ps, "dt_ps")
    geometry_dt = _decimal(config.geometry_dt_ps, "geometry_dt_ps")
    contact_dt = _decimal(config.contact_dt_ps, "contact_dt_ps")
    geometry = [dict(row) for row in geometry_records]
    if not geometry:
        raise ValueError("No geometry records")
    index = {}
    seen_steps = set()
    seen_times = set()
    for row in geometry:
        _required(row, ("tier", "case", "step", "time_ps", "segment", "source_path",
                       "source_frame_index", "canonical_frame_sha256", "source_frame_verified",
                       "cell_area_A2", "cells", "main_N2", "contact_water_O"), "Geometry")
        _required(row, (f"{state}_{suffix}" for state in GRID_STATES
                        for suffix in ("cells", "area_A2")), "Geometry partition")
        if row["case"] != config.case or row["tier"] != config.tier:
            raise ValueError("Geometry case/tier conflicts with explicit configuration")
        step = _integer(row["step"], "step")
        time = _decimal(row["time_ps"], "time_ps")
        segment = str(row["segment"])
        if step in seen_steps or time in seen_times:
            raise ValueError("Duplicate geometry step or physical time")
        seen_steps.add(step)
        seen_times.add(time)
        if time != origin + (step - config.origin_step) * step_dt:
            raise ValueError("Geometry time/step does not match the declared origin")
        if (time - origin) % geometry_dt:
            raise ValueError("Geometry time is off the declared sample grid")
        if segment not in epochs:
            raise ValueError("Geometry source epoch is undeclared")
        epoch = epochs[segment]
        if not epoch.first_step <= step <= epoch.last_step or row["source_path"] != epoch.source_path:
            raise ValueError("Geometry source path/step conflicts with source epoch")
        if not _boolean(row["source_frame_verified"], "source_frame_verified"):
            raise ValueError("Geometry source frame is not historically verified")
        if not re.fullmatch(r"[a-f0-9]{64}", str(row["canonical_frame_sha256"])):
            raise ValueError("Geometry canonical frame hash is malformed")
        _integer(row["source_frame_index"], "source_frame_index")
        _integer(row["main_N2"], "main_N2")
        _integer(row["contact_water_O"], "contact_water_O")
        cell_area = float(_decimal(row["cell_area_A2"], "cell_area_A2"))
        cells = _integer(row["cells"], "cells", 1)
        if cell_area <= 0:
            raise ValueError("Positive cell area is required")
        partition_cells = 0
        partition_area = 0.0
        for state in GRID_STATES:
            count = _integer(row[f"{state}_cells"], f"{state}_cells")
            area = float(_decimal(row[f"{state}_area_A2"], f"{state}_area_A2"))
            if area < 0 or not math.isclose(area, count * cell_area, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"Grid area/count partition mismatch: {state}")
            partition_cells += count
            partition_area += area
        if (partition_cells != cells
                or not math.isclose(partition_area, cells * cell_area, rel_tol=1e-10, abs_tol=1e-8)):
            raise ValueError("Grid partition does not close")
        index[(time, segment)] = row
    geometry.sort(key=lambda row: _decimal(row["time_ps"], "time_ps"))

    contact_index = defaultdict(dict)
    for record in contact_records:
        if record.get("name") != config.name:
            continue
        row = dict(record)
        _required(row, ("time_ps", "segment", "distance_A", "main_supported_patches",
                       "unique_main_contact_N2", "some_main_supported_contact"), "Contact")
        time = _decimal(row["time_ps"], "contact time")
        segment = str(row["segment"])
        distance = _decimal(row["distance_A"], "distance_A")
        if distance not in shells:
            raise ValueError("Contact distance shell is undeclared")
        if (time - origin) % contact_dt:
            raise ValueError("Contact time is off the declared sample grid")
        key = (time, segment)
        if key not in index:
            raise ValueError("Contact has no exact time+epoch geometry match")
        if distance in contact_index[key]:
            raise ValueError("Duplicate contact name/time/epoch/distance record")
        patches = _integer(row["main_supported_patches"], "main_supported_patches")
        count = _integer(row["unique_main_contact_N2"], "unique_main_contact_N2")
        present = _boolean(row["some_main_supported_contact"], "some_main_supported_contact")
        if present != (patches > 0) or (present and count == 0):
            raise ValueError("Contact support flag/counts are inconsistent")
        if count > _integer(index[key]["main_N2"], "main_N2"):
            raise ValueError("Contact molecule count exceeds main cluster inventory")
        contact_index[key][distance] = row
    if not contact_index:
        raise ValueError("No contact records for configured name")

    joined = []
    for key in sorted(contact_index):
        shell_rows = contact_index[key]
        if set(shell_rows) != shells:
            raise ValueError("Incomplete three-shell contact observation")
        time, segment = key
        row = dict(index[key])
        present = _boolean(shell_rows[Decimal(4)]["some_main_supported_contact"], "4A contact")
        wet = _integer(row["water_separated_cells"], "water_separated_cells") > 0
        state = ("contact_and_water_separated" if present and wet else
                 "contact_without_water_separated" if present else
                 "water_separated_without_contact" if wet else "neither")
        row.update(name=config.name, time_ps=float(time), step=_integer(row["step"], "step"),
                   main_contact_4A=present, grid_water_separated_present=wet,
                   joint_observation=state, match_validation=MATCH_VALIDATION,
                   contact_shells={format(shell.normalize(), "f"): shell_rows[shell]
                                   for shell in sorted(shells)})
        joined.append(row)

    episodes = []
    for present, label in ((True, "contact"), (False, "absence")):
        states = [int(row["main_contact_4A"] == present) for row in joined]
        for episode in sampled_episodes(states, [row["time_ps"] for row in joined],
                                        [row["segment"] for row in joined], config.contact_dt_ps):
            first, last = int(episode["first_index"]), int(episode["last_index"])
            episodes.append(dict(episode, name=config.name, case=config.case,
                                 observed_state=label, first_time_ps=joined[first]["time_ps"],
                                 last_time_ps=joined[last]["time_ps"], segment=joined[first]["segment"],
                                 first_step=joined[first]["step"], last_step=joined[last]["step"]))
    groups = defaultdict(list)
    for row in joined:
        groups[row["joint_observation"]].append(row)
    representatives = []
    for category, rows in sorted(groups.items()):
        selected = rows[len(rows) // 2]
        representatives.append(dict(name=config.name, case=config.case, category=category,
                                    selection_rule="upper_time_median_observed_joint_category",
                                    category_samples=len(rows), time_ps=selected["time_ps"],
                                    step=selected["step"], segment=selected["segment"],
                                    source_path=selected["source_path"],
                                    source_frame_index=selected["source_frame_index"],
                                    canonical_frame_sha256=selected["canonical_frame_sha256"],
                                    interpretation="Observed frame, not a qualified wet/dry basin",
                                    input_record=selected))
    summary = dict(name=config.name, case=config.case, tier=config.tier,
                   geometry_samples=len(geometry), contact_samples=len(joined),
                   geometry_water_separated_frames=sum(
                       _integer(row["water_separated_cells"], "water_separated_cells") > 0
                       for row in geometry),
                   contact_frames=sum(row["main_contact_4A"] for row in joined),
                   water_separated_frames=sum(row["grid_water_separated_present"] for row in joined),
                   coexistence_frames=sum(row["main_contact_4A"] and row["grid_water_separated_present"]
                                          for row in joined),
                   geometry_first_time_ps=float(_decimal(geometry[0]["time_ps"], "time_ps")),
                   geometry_last_time_ps=float(_decimal(geometry[-1]["time_ps"], "time_ps")),
                   match_validation=MATCH_VALIDATION, periodic=list(config.periodic),
                   statistical_scope="Sample counts and censored episodes; not independent rates or PMF",
                   water_inventory_scope="contact_water_O is full-surface 0-3 A inventory, not bubble-bottom water")
    return dict(geometry_records=geometry, joined=joined, summary=summary,
                episodes=episodes, representatives=representatives)
