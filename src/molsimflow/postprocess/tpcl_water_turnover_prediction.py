"""Predict TPCL events from pre-event water-membership turnover histories.

The input membership tables are long-form frame records with one row per
selected water oxygen.  They are streamed one frame at a time so multi-million
row tables do not need to be materialized in memory.  Event/control anchors
are evaluated with time-blocked and leave-one-case-out prediction.

All reported associations are retrospective.  They are not causal evidence,
physical rates, free energies, friction, or replicate-level uncertainty.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_PRE_EVENT_WATER_MEMBERSHIP_TURNOVER_PREDICTION_"
    "NOT_CAUSAL_PHYSICAL_RATE_FREE_ENERGY_FRICTION_OR_REPLICATE_EVIDENCE"
)


@dataclass(frozen=True)
class TurnoverConfig:
    """Frozen feature, prediction, and resampling settings."""

    history_lags_ps: tuple[float, ...] = (1.0, 2.0, 5.0)
    frame_interval_ps: float = 0.5
    arc_count: int = 36
    arc_half_width: int = 1
    fold_count: int = 5
    embargo_blocks: int = 1
    penalty: float = 0.1
    bootstrap_samples: int = 2000
    null_samples: int = 200
    random_seed: int = 20260905
    max_iterations: int = 1000

    def lag_frames(self) -> tuple[int, ...]:
        """Return configured lags as exact positive frame counts."""

        if not math.isfinite(self.frame_interval_ps) or self.frame_interval_ps <= 0.0:
            raise ValueError("frame_interval_ps must be finite and positive")
        if not self.history_lags_ps:
            raise ValueError("at least one history lag is required")
        frames = []
        for lag in self.history_lags_ps:
            raw = lag / self.frame_interval_ps
            rounded = round(raw)
            if not math.isfinite(lag) or lag <= 0.0 or rounded <= 0:
                raise ValueError("history lags must be finite and positive")
            if not math.isclose(raw, rounded, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"history lag {lag:g} ps is not an integer number of frames")
            frames.append(rounded)
        if len(set(frames)) != len(frames):
            raise ValueError("history lags map to duplicate frame counts")
        return tuple(frames)

    def validate(self) -> None:
        """Reject settings that cannot support the declared analysis."""

        self.lag_frames()
        if self.arc_count <= 0:
            raise ValueError("arc_count must be positive")
        if self.arc_half_width < 0 or 2 * self.arc_half_width + 1 > self.arc_count:
            raise ValueError("arc_half_width is incompatible with arc_count")
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if self.embargo_blocks < 0:
            raise ValueError("embargo_blocks cannot be negative")
        if not math.isfinite(self.penalty) or self.penalty <= 0.0:
            raise ValueError("penalty must be finite and positive")
        if self.bootstrap_samples < 10:
            raise ValueError("bootstrap_samples must be at least ten")
        if self.null_samples < 1:
            raise ValueError("null_samples must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class InputColumns:
    """Column mapping for generic risk and membership tables."""

    case: str = "case_id"
    arc: str = "primary_arc_index"
    sample_step: str = "sample_step"
    anchor_step: str = "sample_pre_step"
    block: str = "sample_time_block_200ps"
    label: str = "is_event"
    weight: str = "risk_set_weight"
    membership_step: str = "step"
    membership_id: str = "oxygen_id"
    membership_arc: str = "arc_index"


@dataclass(frozen=True)
class MembershipFrame:
    """Water membership split over periodic TPCL arcs for one frame."""

    step: int
    by_arc: Mapping[int, frozenset[int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _delimiter(path: Path) -> str:
    return "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","


def _open_csv_text(path: Path) -> AbstractContextManager[TextIO]:
    if Path(path).suffix.lower() == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return Path(path).open("r", newline="", encoding="utf-8")


def _read_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(path))
        if reader.fieldnames is None:
            raise ValueError(f"table has no header: {path}")
        rows = list(reader)
        fields = tuple(reader.fieldnames)
    if not rows:
        raise ValueError(f"table is empty: {path}")
    return rows, fields


def _require_fields(path: Path, fields: Sequence[str], required: set[str]) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lag_label(lag_ps: float) -> str:
    text = f"{lag_ps:g}".replace("-", "m").replace(".", "p")
    return f"{text}ps"


def load_case_sources(
    path: Path,
    *,
    case_column: str = "case_id",
    membership_column: str = "membership_table",
) -> dict[str, Path]:
    """Load case-to-membership-table mappings with relative-path support."""

    rows, fields = _read_rows(path)
    _require_fields(path, fields, {case_column, membership_column})
    output = {}
    for row in rows:
        case_id = row[case_column].strip()
        if not case_id or case_id in output:
            raise ValueError(f"duplicate or empty case ID in {path}: {case_id!r}")
        source = Path(row[membership_column]).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"membership table does not exist: {source}")
        output[case_id] = source
    return output


def _same_number(left: object, right: object) -> bool:
    first, second = float(left), float(right)
    if math.isnan(first) and math.isnan(second):
        return True
    return math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-12)


def deduplicate_risk_rows(
    rows: Sequence[Mapping[str, str]],
    columns: InputColumns,
    baseline_features: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Collapse repeated case/arc/sample anchors without crossing folds."""

    grouped: dict[tuple[str, int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = str(row[columns.case])
        if not case_id.strip():
            raise ValueError("risk case IDs cannot be empty")
        label = int(row[columns.label])
        if label not in {0, 1}:
            raise ValueError("risk labels must be zero or one")
        weight = float(row[columns.weight])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("risk weights must be finite and positive")
        key = (
            case_id,
            int(row[columns.arc]),
            int(row[columns.sample_step]),
        )
        grouped[key].append(row)

    output = []
    duplicate_keys = 0
    invariant_fields = (
        columns.anchor_step,
        columns.block,
        columns.label,
        *baseline_features,
    )
    for key, selected in grouped.items():
        reference = selected[0]
        if len(selected) > 1:
            duplicate_keys += 1
        for row in selected[1:]:
            for field in invariant_fields:
                if not _same_number(reference[field], row[field]):
                    raise ValueError(f"risk anchor {key} has inconsistent {field}")
        combined: dict[str, object] = dict(reference)
        combined[columns.weight] = sum(float(row[columns.weight]) for row in selected)
        combined["aggregated_source_row_count"] = len(selected)
        output.append(combined)
    output.sort(
        key=lambda row: (
            str(row[columns.case]),
            int(row[columns.sample_step]),
            int(row[columns.arc]),
        )
    )
    return output, {
        "input_row_count": len(rows),
        "unique_anchor_count": len(output),
        "duplicate_anchor_key_count": duplicate_keys,
        "anchor_key": [columns.case, columns.arc, columns.sample_step],
    }


def iter_membership_frames(
    path: Path,
    columns: InputColumns,
    arc_count: int,
) -> Iterator[MembershipFrame]:
    """Stream a step-sorted long membership table as frame objects."""

    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(path))
        if reader.fieldnames is None:
            raise ValueError(f"membership table has no header: {path}")
        _require_fields(
            path,
            reader.fieldnames,
            {
                columns.membership_step,
                columns.membership_id,
                columns.membership_arc,
            },
        )
        current_step: Optional[int] = None
        by_arc: dict[int, set[int]] = defaultdict(set)
        seen_ids: set[int] = set()
        for row in reader:
            step = int(row[columns.membership_step])
            member_id = int(row[columns.membership_id])
            arc = int(row[columns.membership_arc])
            if arc < 0 or arc >= arc_count:
                raise ValueError(f"{path}: arc index {arc} outside [0, {arc_count})")
            if current_step is None:
                current_step = step
            elif step < current_step:
                raise ValueError(f"{path}: membership steps are not sorted")
            elif step != current_step:
                yield MembershipFrame(
                    current_step,
                    {key: frozenset(value) for key, value in by_arc.items()},
                )
                current_step = step
                by_arc = defaultdict(set)
                seen_ids = set()
            if member_id in seen_ids:
                raise ValueError(f"{path}: duplicate member {member_id} at step {step}")
            seen_ids.add(member_id)
            by_arc[arc].add(member_id)
        if current_step is not None:
            yield MembershipFrame(
                current_step,
                {key: frozenset(value) for key, value in by_arc.items()},
            )


def patch_membership(
    frame: MembershipFrame,
    center_arc: int,
    arc_count: int,
    arc_half_width: int,
) -> frozenset[int]:
    """Combine a periodic neighborhood of TPCL arcs into one membership set."""

    if center_arc < 0 or center_arc >= arc_count:
        raise ValueError(f"center arc {center_arc} outside [0, {arc_count})")
    members: set[int] = set()
    for offset in range(-arc_half_width, arc_half_width + 1):
        members.update(frame.by_arc.get((center_arc + offset) % arc_count, ()))
    return frozenset(members)


def turnover_feature_groups(config: TurnoverConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return occupancy-nuisance and molecular-turnover feature names."""

    occupancy = ["membership_patch_current_count"]
    turnover = []
    for lag in config.history_lags_ps:
        label = _lag_label(lag)
        occupancy.extend(
            [
                f"membership_patch_mean_count_{label}",
                f"membership_patch_count_cv_{label}",
                f"membership_patch_net_fraction_{label}",
            ]
        )
        turnover.extend(
            [
                f"membership_node_survival_fraction_{label}",
                f"membership_gross_turnover_fraction_{label}",
                f"membership_cumulative_turnover_fraction_{label}",
            ]
        )
    return tuple(occupancy), tuple(turnover)


def _nan_features(config: TurnoverConfig) -> dict[str, object]:
    occupancy, turnover = turnover_feature_groups(config)
    output = {name: math.nan for name in occupancy + turnover}
    output["turnover_history_complete"] = 0
    return output


def _anchor_features(
    frames: Sequence[MembershipFrame],
    center_arc: int,
    config: TurnoverConfig,
) -> dict[str, object]:
    lag_frames = config.lag_frames()
    cache: dict[int, frozenset[int]] = {}

    def members(index: int) -> frozenset[int]:
        if index not in cache:
            cache[index] = patch_membership(
                frames[index], center_arc, config.arc_count, config.arc_half_width
            )
        return cache[index]

    current = members(len(frames) - 1)
    output: dict[str, object] = {
        "turnover_history_complete": 1,
        "membership_patch_current_count": len(current),
    }
    for lag_ps, count in zip(config.history_lags_ps, lag_frames):
        label = _lag_label(lag_ps)
        selected = [members(index) for index in range(len(frames) - count - 1, len(frames))]
        past = selected[0]
        intersection = len(past & current)
        endpoint_denominator = len(past) + len(current)
        survival = intersection / len(past) if past else math.nan
        gross = len(past ^ current) / endpoint_denominator if endpoint_denominator else math.nan
        net = (
            (len(current) - len(past)) / endpoint_denominator if endpoint_denominator else math.nan
        )
        cumulative_numerator = 0
        cumulative_denominator = 0
        for left, right in zip(selected[:-1], selected[1:]):
            cumulative_numerator += len(left ^ right)
            cumulative_denominator += len(left) + len(right)
        cumulative = (
            cumulative_numerator / cumulative_denominator if cumulative_denominator else math.nan
        )
        counts = np.asarray([len(item) for item in selected], dtype=float)
        mean_count = float(np.mean(counts))
        count_cv = float(np.std(counts) / mean_count) if mean_count > 0.0 else math.nan
        output.update(
            {
                f"membership_patch_mean_count_{label}": mean_count,
                f"membership_patch_count_cv_{label}": count_cv,
                f"membership_patch_net_fraction_{label}": net,
                f"membership_node_survival_fraction_{label}": survival,
                f"membership_gross_turnover_fraction_{label}": gross,
                f"membership_cumulative_turnover_fraction_{label}": cumulative,
            }
        )
    return output


def enrich_turnover_features(
    rows: Sequence[Mapping[str, object]],
    case_sources: Mapping[str, Path],
    columns: InputColumns,
    config: TurnoverConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join streamed pre-event membership histories to risk-set anchors."""

    rows_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        case_id = str(row[columns.case])
        if case_id not in case_sources:
            raise ValueError(f"risk table contains undeclared case {case_id!r}")
        arc = int(row[columns.arc])
        if arc < 0 or arc >= config.arc_count:
            raise ValueError(f"risk anchor arc {arc} outside [0, {config.arc_count})")
        rows_by_case[case_id].append(row)
    missing_cases = sorted(set(case_sources) - set(rows_by_case))
    if missing_cases:
        raise ValueError(f"case table contains cases absent from risk table: {missing_cases}")

    max_lag = max(config.lag_frames())
    summaries = []
    for case_id, source_path in case_sources.items():
        selected_rows = rows_by_case[case_id]
        by_anchor_step: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in selected_rows:
            by_anchor_step[int(row[columns.anchor_step])].append(row)
        unresolved = set(by_anchor_step)
        history: deque[MembershipFrame] = deque(maxlen=max_lag + 1)
        frame_count = 0
        first_step: Optional[int] = None
        last_step: Optional[int] = None
        step_stride: Optional[int] = None
        previous_step: Optional[int] = None
        complete_count = 0
        for frame in iter_membership_frames(source_path, columns, config.arc_count):
            frame_count += 1
            if first_step is None:
                first_step = frame.step
            if previous_step is not None:
                difference = frame.step - previous_step
                if difference <= 0:
                    raise ValueError(f"{source_path}: nonpositive membership step stride")
                if step_stride is None:
                    step_stride = difference
                elif difference != step_stride:
                    raise ValueError(f"{source_path}: nonuniform membership step stride")
            previous_step = frame.step
            last_step = frame.step
            history.append(frame)
            if frame.step not in by_anchor_step:
                continue
            unresolved.discard(frame.step)
            history_list = list(history)
            history_complete = len(history_list) == max_lag + 1 and step_stride is not None
            if history_complete:
                for lag in config.lag_frames():
                    expected = frame.step - lag * step_stride
                    if history_list[-lag - 1].step != expected:
                        history_complete = False
                        break
            feature_cache: dict[int, dict[str, object]] = {}
            for row in by_anchor_step[frame.step]:
                if history_complete:
                    arc = int(row[columns.arc])
                    if arc not in feature_cache:
                        feature_cache[arc] = _anchor_features(history_list, arc, config)
                    row.update(feature_cache[arc])
                    complete_count += 1
                else:
                    row.update(_nan_features(config))
                row["membership_frame_step_stride"] = (
                    step_stride if step_stride is not None else math.nan
                )
        for step in unresolved:
            for row in by_anchor_step[step]:
                row.update(_nan_features(config))
                row["membership_frame_step_stride"] = (
                    step_stride if step_stride is not None else math.nan
                )
        summaries.append(
            {
                "case_id": case_id,
                "membership_table": str(source_path),
                "membership_frame_count": frame_count,
                "membership_first_step": first_step,
                "membership_last_step": last_step,
                "membership_step_stride": step_stride,
                "risk_anchor_count": len(selected_rows),
                "complete_history_anchor_count": complete_count,
                "incomplete_history_anchor_count": len(selected_rows) - complete_count,
                "unresolved_anchor_step_count": len(unresolved),
            }
        )
    output = [row for case_id in sorted(rows_by_case) for row in rows_by_case[case_id]]
    output.sort(
        key=lambda row: (
            str(row[columns.case]),
            int(row[columns.sample_step]),
            int(row[columns.arc]),
        )
    )
    return output, summaries


def _matrices(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    train_columns = []
    test_columns = []
    names = []
    for field in features:
        train = np.asarray([float(row[field]) for row in train_rows], dtype=float)
        test = np.asarray([float(row[field]) for row in test_rows], dtype=float)
        finite = np.isfinite(train)
        if not np.any(finite):
            continue
        median = float(np.median(train[finite]))
        train_missing = ~finite
        test_missing = ~np.isfinite(test)
        train = np.where(train_missing, median, train)
        test = np.where(test_missing, median, test)
        mean = float(np.mean(train))
        scale = float(np.std(train))
        if scale > 1.0e-12:
            train_columns.append((train - mean) / scale)
            test_columns.append((test - mean) / scale)
            names.append(field)
        if np.any(train_missing) and np.any(~train_missing):
            train_columns.append(train_missing.astype(float))
            test_columns.append(test_missing.astype(float))
            names.append(f"{field}__missing")
    if not train_columns:
        return np.empty((len(train_rows), 0)), np.empty((len(test_rows), 0)), ()
    return np.column_stack(train_columns), np.column_stack(test_columns), tuple(names)


def _fit_logistic(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    columns: InputColumns,
    config: TurnoverConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    target = np.asarray([int(row[columns.label]) for row in train_rows], dtype=float)
    weight = np.asarray([float(row[columns.weight]) for row in train_rows], dtype=float)
    if set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("binary training data must contain labels zero and one")
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("risk weights must be finite and positive")
    train, test, names = _matrices(train_rows, test_rows, features)
    mean_target = float(np.average(target, weights=weight))
    mean_target = float(np.clip(mean_target, 1.0e-8, 1.0 - 1.0e-8))
    if not names:
        return np.full(len(test_rows), mean_target), names

    design = np.column_stack([np.ones(len(train)), train])
    test_design = np.column_stack([np.ones(len(test)), test])
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = math.log(mean_target / (1.0 - mean_target))
    weight_sum = float(np.sum(weight))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ coefficients
        probability = expit(linear)
        loss = float(np.sum(weight * (np.logaddexp(0.0, linear) - target * linear)))
        loss /= weight_sum
        loss += 0.5 * config.penalty * float(np.dot(coefficients[1:], coefficients[1:]))
        gradient = design.T @ (weight * (probability - target)) / weight_sum
        gradient[1:] += config.penalty * coefficients[1:]
        return loss, np.asarray(gradient, dtype=float)

    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": config.max_iterations},
    )
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError(f"penalized logistic fit failed: {fit.message}")
    return np.asarray(expit(test_design @ fit.x), dtype=float), names


def _fold_labels(
    rows: Sequence[Mapping[str, object]],
    block_column: str,
    fold_count: int,
) -> np.ndarray:
    blocks = np.asarray([int(row[block_column]) for row in rows], dtype=int)
    unique = np.unique(blocks)
    if len(unique) < fold_count:
        raise ValueError("fewer time blocks than prediction folds")
    mapping = {
        int(block): min(index * fold_count // len(unique), fold_count - 1)
        for index, block in enumerate(unique)
    }
    return np.asarray([mapping[int(block)] for block in blocks], dtype=int)


def _split_predict(
    rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    held_case: str,
    evaluation: str,
    columns: InputColumns,
    config: TurnoverConfig,
) -> tuple[list[Mapping[str, object]], np.ndarray, tuple[str, ...]]:
    if evaluation == "within_case":
        selected = [row for row in rows if row[columns.case] == held_case]
        folds = _fold_labels(selected, columns.block, config.fold_count)
        blocks = np.asarray([int(row[columns.block]) for row in selected], dtype=int)
        prediction = np.full(len(selected), np.nan)
        names = set()
        for fold in range(config.fold_count):
            test_indices = np.flatnonzero(folds == fold)
            test_blocks = set(blocks[test_indices])
            excluded = {
                candidate
                for block in test_blocks
                for candidate in range(
                    block - config.embargo_blocks,
                    block + config.embargo_blocks + 1,
                )
            }
            train_indices = np.asarray(
                [index for index, block in enumerate(blocks) if block not in excluded],
                dtype=int,
            )
            if not len(test_indices) or len(train_indices) < 20:
                raise ValueError(f"{held_case}/fold {fold}: insufficient prediction rows")
            train_rows = [selected[index] for index in train_indices]
            test_rows = [selected[index] for index in test_indices]
            fold_prediction, used = _fit_logistic(train_rows, test_rows, features, columns, config)
            prediction[test_indices] = fold_prediction
            names.update(used)
        test_rows = selected
    elif evaluation == "leave_one_case_out":
        train_rows = [row for row in rows if row[columns.case] != held_case]
        test_rows = [row for row in rows if row[columns.case] == held_case]
        prediction, used = _fit_logistic(train_rows, test_rows, features, columns, config)
        names = set(used)
    else:
        raise ValueError(f"unknown evaluation {evaluation!r}")
    if not test_rows or np.any(~np.isfinite(prediction)):
        raise ValueError(f"{held_case}/{evaluation}: incomplete prediction")
    return test_rows, prediction, tuple(sorted(names))


def _losses(
    rows: Sequence[Mapping[str, object]],
    prediction: np.ndarray,
    columns: InputColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray([int(row[columns.label]) for row in rows], dtype=int)
    weight = np.asarray([float(row[columns.weight]) for row in rows], dtype=float)
    clipped = np.clip(prediction, 1.0e-12, 1.0 - 1.0e-12)
    loss = -(target * np.log(clipped) + (1 - target) * np.log(1.0 - clipped))
    return loss, target, weight


def _weighted_auc(target: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    positive_total = float(np.sum(weight[target == 1]))
    negative_total = float(np.sum(weight[target == 0]))
    if positive_total <= 0.0 or negative_total <= 0.0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    target = target[order]
    score = score[order]
    weight = weight[order]
    numerator = 0.0
    cumulative_negative = 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        group_positive = float(np.sum(weight[start:end][target[start:end] == 1]))
        group_negative = float(np.sum(weight[start:end][target[start:end] == 0]))
        numerator += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        start = end
    return numerator / (positive_total * negative_total)


def _score(
    rows: Sequence[Mapping[str, object]],
    prediction: np.ndarray,
    columns: InputColumns,
) -> tuple[np.ndarray, dict[str, float]]:
    loss, target, weight = _losses(rows, prediction, columns)
    return loss, {
        "weighted_log_loss": float(np.average(loss, weights=weight)),
        "weighted_roc_auc": _weighted_auc(target, prediction, weight),
        "weighted_brier": float(np.average((target - prediction) ** 2, weights=weight)),
    }


def _bootstrap_delta(
    difference: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    columns: InputColumns,
    config: TurnoverConfig,
    seed_offset: int,
) -> dict[str, float]:
    blocks = np.asarray([int(row[columns.block]) for row in rows], dtype=int)
    weights = np.asarray([float(row[columns.weight]) for row in rows], dtype=float)
    unique = np.unique(blocks)
    rng = np.random.default_rng(config.random_seed + seed_offset)
    draws = []
    for _ in range(config.bootstrap_samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(blocks == block) for block in selected])
        draws.append(float(np.average(difference[indices], weights=weights[indices])))
    values = np.asarray(draws, dtype=float)
    return {
        "bootstrap_ci025": float(np.quantile(values, 0.025)),
        "bootstrap_ci975": float(np.quantile(values, 0.975)),
    }


def _row_key(row: Mapping[str, object], columns: InputColumns) -> tuple[str, int, int]:
    return (
        str(row[columns.case]),
        int(row[columns.arc]),
        int(row[columns.sample_step]),
    )


def _permuted_features(
    rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    columns: InputColumns,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row[columns.case]), int(row[columns.block]))].append(index)
    for indices in grouped.values():
        order = rng.permutation(indices)
        for target_index, source_index in zip(indices, order):
            for field in features:
                output[target_index][field] = rows[int(source_index)][field]
    return output


def _bh_adjust(values: Sequence[float]) -> list[float]:
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty(len(raw), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(raw) - reverse_rank + 1
        running = min(running, float(raw[index]) * len(raw) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def evaluate_prediction(
    rows: Sequence[Mapping[str, object]],
    case_ids: Sequence[str],
    baseline_features: Sequence[str],
    columns: InputColumns,
    config: TurnoverConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Fit nested models and table-level turnover-feature null controls."""

    occupancy, turnover = turnover_feature_groups(config)
    model_features = {
        "M0_static": tuple(baseline_features),
        "M1_occupancy_history": tuple(baseline_features) + occupancy,
        "M2_turnover_history": tuple(baseline_features) + occupancy + turnover,
    }
    result_cache = {}
    score_rows = []
    prediction_rows = []
    evaluations = ("within_case", "leave_one_case_out")
    for evaluation in evaluations:
        for case_index, held_case in enumerate(case_ids):
            expected_keys = None
            for model, features in model_features.items():
                test_rows, prediction, used = _split_predict(
                    rows, features, held_case, evaluation, columns, config
                )
                keys = [_row_key(row, columns) for row in test_rows]
                if expected_keys is None:
                    expected_keys = keys
                elif keys != expected_keys:
                    raise ValueError(f"{held_case}/{evaluation}: model test rows changed")
                loss, scores = _score(test_rows, prediction, columns)
                result_cache[(evaluation, held_case, model)] = (
                    test_rows,
                    prediction,
                    loss,
                )
                score_rows.append(
                    {
                        "evaluation": evaluation,
                        "held_case": held_case,
                        "model": model,
                        "row_count": len(test_rows),
                        **scores,
                        "used_feature_count": len(used),
                        "used_features": ";".join(used),
                    }
                )
                for row, value, row_loss in zip(test_rows, prediction, loss):
                    prediction_rows.append(
                        {
                            "evaluation": evaluation,
                            "held_case": held_case,
                            "model": model,
                            "case_id": row[columns.case],
                            "primary_arc_index": row[columns.arc],
                            "sample_step": row[columns.sample_step],
                            "sample_pre_step": row[columns.anchor_step],
                            "time_block": row[columns.block],
                            "is_event": row[columns.label],
                            "risk_set_weight": row[columns.weight],
                            "prediction": float(value),
                            "log_loss": float(row_loss),
                        }
                    )

    evidence_rows = []
    null_rows = []
    primary_indices = []
    for evaluation_index, evaluation in enumerate(evaluations):
        for case_index, held_case in enumerate(case_ids):
            for comparison_index, (left, right) in enumerate(
                (
                    ("M0_static", "M1_occupancy_history"),
                    ("M1_occupancy_history", "M2_turnover_history"),
                )
            ):
                test_rows, _, left_loss = result_cache[(evaluation, held_case, left)]
                right_rows, _, right_loss = result_cache[(evaluation, held_case, right)]
                if [_row_key(row, columns) for row in test_rows] != [
                    _row_key(row, columns) for row in right_rows
                ]:
                    raise ValueError(f"{held_case}/{evaluation}: comparison rows changed")
                weight = np.asarray([float(row[columns.weight]) for row in test_rows])
                difference = left_loss - right_loss
                observed = float(np.average(difference, weights=weight))
                bootstrap = _bootstrap_delta(
                    difference,
                    test_rows,
                    columns,
                    config,
                    1000 * evaluation_index + 100 * case_index + 10 * comparison_index,
                )
                evidence = {
                    "evaluation": evaluation,
                    "held_case": held_case,
                    "comparison": f"{left}_to_{right}",
                    "row_count": len(test_rows),
                    "delta_weighted_log_loss": observed,
                    **bootstrap,
                    "permutation_p": math.nan,
                    "bh_q": math.nan,
                    "qualified_incremental_turnover_information": 0,
                    "scientific_status": SCIENTIFIC_STATUS,
                }
                if right == "M2_turnover_history":
                    null_values = []
                    for null_index in range(config.null_samples):
                        rng = np.random.default_rng(
                            config.random_seed
                            + 100000 * evaluation_index
                            + 10000 * case_index
                            + null_index
                        )
                        permuted = _permuted_features(rows, turnover, columns, rng)
                        null_test_rows, null_prediction, _ = _split_predict(
                            permuted,
                            model_features[right],
                            held_case,
                            evaluation,
                            columns,
                            config,
                        )
                        if [_row_key(row, columns) for row in null_test_rows] != [
                            _row_key(row, columns) for row in test_rows
                        ]:
                            raise ValueError(f"{held_case}/{evaluation}: null-control rows changed")
                        null_loss, _, _ = _losses(null_test_rows, null_prediction, columns)
                        value = float(np.average(left_loss - null_loss, weights=weight))
                        null_values.append(value)
                        null_rows.append(
                            {
                                "evaluation": evaluation,
                                "held_case": held_case,
                                "null_index": null_index,
                                "delta_weighted_log_loss": value,
                                "null_kind": "turnover_vector_permutation_within_case_time_block",
                            }
                        )
                    evidence["permutation_p"] = float(
                        (1 + np.count_nonzero(np.asarray(null_values) >= observed))
                        / (config.null_samples + 1)
                    )
                    primary_indices.append(len(evidence_rows))
                evidence_rows.append(evidence)

    adjusted = _bh_adjust(
        [float(evidence_rows[index]["permutation_p"]) for index in primary_indices]
    )
    for index, q_value in zip(primary_indices, adjusted):
        evidence_rows[index]["bh_q"] = q_value
        evidence_rows[index]["qualified_incremental_turnover_information"] = int(
            float(evidence_rows[index]["delta_weighted_log_loss"]) > 0.0
            and float(evidence_rows[index]["bootstrap_ci025"]) > 0.0
            and q_value <= 0.05
        )
    return score_rows, evidence_rows, null_rows + prediction_rows


def run_analysis(
    cases_table: Path,
    risk_table: Path,
    output_dir: Path,
    *,
    baseline_features: Sequence[str] = (),
    columns: Optional[InputColumns] = None,
    config: Optional[TurnoverConfig] = None,
    case_table_id_column: str = "case_id",
    case_table_membership_column: str = "membership_table",
) -> dict[str, object]:
    """Build turnover histories, run prediction, and write audit artifacts."""

    if columns is None:
        columns = InputColumns()
    if config is None:
        config = TurnoverConfig()
    config.validate()
    cases_table = Path(cases_table).resolve()
    risk_table = Path(risk_table).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_sources = load_case_sources(
        cases_table,
        case_column=case_table_id_column,
        membership_column=case_table_membership_column,
    )
    raw_rows, fields = _read_rows(risk_table)
    required = {
        columns.case,
        columns.arc,
        columns.sample_step,
        columns.anchor_step,
        columns.block,
        columns.label,
        columns.weight,
        *baseline_features,
    }
    _require_fields(risk_table, fields, required)
    rows, deduplication = deduplicate_risk_rows(raw_rows, columns, baseline_features)
    enriched, coverage_rows = enrich_turnover_features(rows, case_sources, columns, config)
    model_rows = [row for row in enriched if int(row["turnover_history_complete"]) == 1]
    if len(model_rows) < 40:
        raise ValueError("fewer than 40 anchors have complete turnover histories")
    case_ids = tuple(case_sources)
    for case_id in case_ids:
        labels = {int(row[columns.label]) for row in model_rows if row[columns.case] == case_id}
        if labels != {0, 1}:
            raise ValueError(f"case {case_id} lacks both event and control anchors")

    score_rows, evidence_rows, combined_rows = evaluate_prediction(
        model_rows, case_ids, baseline_features, columns, config
    )
    null_rows = [row for row in combined_rows if "null_kind" in row]
    prediction_rows = [row for row in combined_rows if "model" in row]
    _write_csv(output_dir / "turnover_enriched_risk_sets.csv", enriched)
    _write_csv(output_dir / "membership_coverage.csv", coverage_rows)
    _write_csv(output_dir / "prediction_scores.csv", score_rows)
    _write_csv(output_dir / "prediction_evidence.csv", evidence_rows)
    _write_csv(output_dir / "turnover_null_controls.csv", null_rows)
    _write_csv(output_dir / "out_of_fold_predictions.csv", prediction_rows)

    primary_rows = [
        row
        for row in evidence_rows
        if row["comparison"] == "M1_occupancy_history_to_M2_turnover_history"
    ]
    qualified_by_evaluation = {
        evaluation: sum(
            int(row["qualified_incremental_turnover_information"])
            for row in primary_rows
            if row["evaluation"] == evaluation
        )
        for evaluation in ("within_case", "leave_one_case_out")
    }
    summary = {
        "status": "PASS",
        "scientific_status": SCIENTIFIC_STATUS,
        "case_count": len(case_ids),
        "case_ids": list(case_ids),
        "input_risk_row_count": len(raw_rows),
        "unique_anchor_count": len(enriched),
        "complete_history_anchor_count": len(model_rows),
        "incomplete_history_anchor_count": len(enriched) - len(model_rows),
        "history_lags_ps": list(config.history_lags_ps),
        "frame_interval_ps": config.frame_interval_ps,
        "arc_count": config.arc_count,
        "arc_half_width": config.arc_half_width,
        "baseline_features": list(baseline_features),
        "primary_test_count": len(primary_rows),
        "qualified_primary_count": sum(
            int(row["qualified_incremental_turnover_information"]) for row in primary_rows
        ),
        "qualified_within_case_count": qualified_by_evaluation["within_case"],
        "qualified_leave_one_case_out_count": qualified_by_evaluation["leave_one_case_out"],
        "bootstrap_samples": config.bootstrap_samples,
        "null_samples": config.null_samples,
        "random_seed": config.random_seed,
    }
    manifest = {
        "scientific_status": SCIENTIFIC_STATUS,
        "cases_table": {"path": str(cases_table), "sha256": _sha256(cases_table)},
        "risk_table": {"path": str(risk_table), "sha256": _sha256(risk_table)},
        "membership_tables": [
            {"case_id": case_id, "path": str(path), "sha256": _sha256(path)}
            for case_id, path in case_sources.items()
        ],
        "columns": columns.__dict__,
        "config": {
            **config.__dict__,
            "history_lags_ps": list(config.history_lags_ps),
        },
        "baseline_features": list(baseline_features),
        "deduplication": deduplication,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        "# TPCL water-membership turnover prediction\n\n"
        f"- Cases: {len(case_ids)}\n"
        f"- Complete pre-event anchors: {len(model_rows)}\n"
        f"- Primary turnover tests: {len(primary_rows)}\n"
        f"- Qualified incremental-turnover tests: {summary['qualified_primary_count']}\n"
        f"- Qualified time-blocked within-case tests: "
        f"{summary['qualified_within_case_count']}\n"
        f"- Qualified leave-one-case-out tests: "
        f"{summary['qualified_leave_one_case_out_count']}\n\n"
        "Models use only water membership at or before each declared pre-event "
        "anchor. Turnover-vector permutations are performed within case and time "
        "block. Results are retrospective predictive associations, not causal "
        "effects, physical rates, free energies, friction, or replicate-level "
        "uncertainty.\n",
        encoding="utf-8",
    )
    return summary


def _parse_lags(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one comma-separated lag is required")
    return values


def _baseline_features(raw: Sequence[str]) -> tuple[str, ...]:
    output = []
    for item in raw:
        output.extend(value.strip() for value in item.split(",") if value.strip())
    if len(set(output)) != len(output):
        raise ValueError("baseline features contain duplicates")
    return tuple(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-table", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-feature", action="append", default=[])
    parser.add_argument("--history-lags-ps", type=_parse_lags, default=(1.0, 2.0, 5.0))
    parser.add_argument("--frame-interval-ps", type=float, default=0.5)
    parser.add_argument("--arc-count", type=int, default=36)
    parser.add_argument("--arc-half-width", type=int, default=1)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--embargo-blocks", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260905)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--arc-column", default="primary_arc_index")
    parser.add_argument("--sample-step-column", default="sample_step")
    parser.add_argument("--anchor-step-column", default="sample_pre_step")
    parser.add_argument("--block-column", default="sample_time_block_200ps")
    parser.add_argument("--label-column", default="is_event")
    parser.add_argument("--weight-column", default="risk_set_weight")
    parser.add_argument("--membership-step-column", default="step")
    parser.add_argument("--membership-id-column", default="oxygen_id")
    parser.add_argument("--membership-arc-column", default="arc_index")
    parser.add_argument("--case-table-id-column", default="case_id")
    parser.add_argument("--case-table-membership-column", default="membership_table")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = TurnoverConfig(
        history_lags_ps=args.history_lags_ps,
        frame_interval_ps=args.frame_interval_ps,
        arc_count=args.arc_count,
        arc_half_width=args.arc_half_width,
        fold_count=args.fold_count,
        embargo_blocks=args.embargo_blocks,
        penalty=args.penalty,
        bootstrap_samples=args.bootstrap_samples,
        null_samples=args.null_samples,
        random_seed=args.random_seed,
        max_iterations=args.max_iterations,
    )
    columns = InputColumns(
        case=args.case_column,
        arc=args.arc_column,
        sample_step=args.sample_step_column,
        anchor_step=args.anchor_step_column,
        block=args.block_column,
        label=args.label_column,
        weight=args.weight_column,
        membership_step=args.membership_step_column,
        membership_id=args.membership_id_column,
        membership_arc=args.membership_arc_column,
    )
    run_analysis(
        args.cases_table,
        args.risk_table,
        args.output_dir,
        baseline_features=_baseline_features(args.baseline_feature),
        columns=columns,
        config=config,
        case_table_id_column=args.case_table_id_column,
        case_table_membership_column=args.case_table_membership_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
