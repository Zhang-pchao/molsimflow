"""Test incremental TPCL event-history information with blocked prediction.

The module fits nested ridge-Poisson models to future cross-arc cluster counts.
Out-of-fold prediction, whole-block bootstrap diagnostics, reversed time,
circular arc-label shifts, and event-history block permutations are reported.
They test predictive information, not causal triggering or physical rates.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_TIME_BLOCKED_INCREMENTAL_PREDICTION_"
    "NOT_CAUSAL_PROPAGATION_OR_REPLICATE_LEVEL_EVIDENCE"
)

GLOBAL_FEATURES = (
    "global_mean_radius_A",
    "global_mode_2_amplitude_A",
    "global_mode_3_amplitude_A",
    "global_mode_4_amplitude_A",
    "global_unresolved_mode_rms_A",
    "global_footprint_area_A2",
    "global_footprint_circularity",
    "global_cap_angle_candidate_deg",
    "global_pressure_trace_bar",
    "global_normal_minus_tangential_bar",
    "global_shear_norm_bar",
)

LOCAL_FEATURES = (
    "pre_local_nearest_site_distance_A",
    "pre_local_local_ch3_fraction",
    "pre_local_chemical_boundary_distance_proxy_A",
    "pre_local_local_hydration_areal_density_A-2",
    "pre_local_local_water_dipole_cos_z",
    "pre_local_local_water_water_hbond_degree",
    "pre_local_local_surface_water_hbond_per_h2o",
    "pre_local_local_n2_min_distance_A",
)

ARC_BINS = (
    ("near", 1, 4),
    ("mid", 4, 10),
    ("far", 10, 19),
)


@dataclass(frozen=True)
class PredictionConfig:
    """Frozen blocked-prediction and null-control settings."""

    block_ps: float = 200.0
    fold_count: int = 5
    embargo_blocks: int = 1
    penalty: float = 0.1
    bootstrap_samples: int = 2000
    null_samples: int = 200
    random_seed: int = 20260904
    max_iterations: int = 500


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _history_fields(direction: str) -> tuple[str, ...]:
    return tuple(
        f"{direction}_history_{window}_{bin_name}_count"
        for window in ("fast", "slow")
        for bin_name, _, _ in ARC_BINS
    )


def _float_column(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError(f"infinite feature {field}")
    return values


def _design_matrices(
    train_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    features: Sequence[str],
    n_arcs: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build fold-local standardized matrices with training-only imputation."""

    train_columns = [np.ones(len(train_rows))]
    test_columns = [np.ones(len(test_rows))]
    names = ["intercept"]
    train_arcs = np.asarray([int(row["primary_arc_index"]) for row in train_rows])
    test_arcs = np.asarray([int(row["primary_arc_index"]) for row in test_rows])
    for arc in range(1, n_arcs):
        train_columns.append((train_arcs == arc).astype(float))
        test_columns.append((test_arcs == arc).astype(float))
        names.append(f"arc_{arc}")
    for field in features:
        train = _float_column(train_rows, field)
        test = _float_column(test_rows, field)
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
    return np.column_stack(train_columns), np.column_stack(test_columns), tuple(names)


def _fit_poisson(
    design: np.ndarray,
    target: np.ndarray,
    *,
    penalty: float,
    max_iterations: int,
) -> tuple[np.ndarray, bool, int]:
    if penalty <= 0.0 or np.any(target < 0.0):
        raise ValueError("positive penalty and nonnegative targets are required")
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = math.log(max(float(np.mean(target)), 1.0e-4))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(design @ coefficients, -20.0, 20.0)
        mean = np.exp(eta)
        loss = float(np.mean(mean - target * eta))
        gradient = design.T @ (mean - target) / len(target)
        loss += 0.5 * penalty * float(np.dot(coefficients[1:], coefficients[1:]))
        gradient[1:] += penalty * coefficients[1:]
        return loss, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1.0e-10, "gtol": 1.0e-7},
    )
    return np.asarray(result.x), bool(result.success), int(result.nit)


def _poisson_deviance_rows(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    prediction = np.clip(prediction, 1.0e-9, 1.0e9)
    log_term = np.zeros_like(target, dtype=float)
    positive = target > 0.0
    log_term[positive] = target[positive] * np.log(target[positive] / prediction[positive])
    return 2.0 * (log_term - (target - prediction))


def _score(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    if np.std(target) > 0.0 and np.std(prediction) > 0.0:
        target_rank = rankdata(target)
        prediction_rank = rankdata(prediction)
        spearman = float(np.corrcoef(target_rank, prediction_rank)[0, 1])
    else:
        spearman = math.nan
    return {
        "mean_poisson_deviance": float(np.mean(_poisson_deviance_rows(target, prediction))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman_r": spearman,
    }


def _fold_labels(rows: Sequence[Mapping[str, str]], fold_count: int) -> np.ndarray:
    blocks = np.asarray([int(row["time_block_200ps"]) for row in rows], dtype=int)
    block_count = int(np.max(blocks)) + 1
    if fold_count < 2 or block_count < fold_count:
        raise ValueError("need at least two folds and at least one time block per fold")
    return np.minimum(blocks * fold_count // block_count, fold_count - 1)


def _oof_predict(
    rows: Sequence[Mapping[str, str]],
    target_field: str,
    features: Sequence[str],
    config: PredictionConfig,
) -> tuple[np.ndarray, np.ndarray, bool, int, tuple[str, ...]]:
    target = _float_column(rows, target_field)
    folds = _fold_labels(rows, config.fold_count)
    blocks = np.asarray([int(row["time_block_200ps"]) for row in rows], dtype=int)
    n_arcs = int(rows[0]["arc_count"])
    prediction = np.full(len(rows), np.nan)
    all_converged = True
    total_iterations = 0
    used_names = set()
    for fold in range(config.fold_count):
        test_index = np.flatnonzero(folds == fold)
        test_blocks = set(blocks[test_index])
        embargo = {
            candidate
            for block in test_blocks
            for candidate in range(block - config.embargo_blocks, block + config.embargo_blocks + 1)
        }
        train_index = np.asarray(
            [index for index, block in enumerate(blocks) if block not in embargo], dtype=int
        )
        if len(test_index) == 0 or len(train_index) < 20:
            raise ValueError(f"fold {fold}: insufficient train/test rows")
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        train_design, test_design, names = _design_matrices(
            train_rows, test_rows, features, n_arcs
        )
        coefficients, converged, iterations = _fit_poisson(
            train_design,
            target[train_index],
            penalty=config.penalty,
            max_iterations=config.max_iterations,
        )
        prediction[test_index] = np.exp(np.clip(test_design @ coefficients, -20.0, 20.0))
        all_converged = all_converged and converged
        total_iterations += iterations
        used_names.update(names)
    if np.any(~np.isfinite(prediction)):
        raise ValueError("OOF prediction is incomplete or non-finite")
    return target, prediction, all_converged, total_iterations, tuple(sorted(used_names))


def _model_feature_sets(history_direction: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    history = _history_fields(history_direction)
    return (
        ("M0_arc_baseline", ()),
        ("M1_global_state", GLOBAL_FEATURES),
        ("M2_event_history", GLOBAL_FEATURES + history),
        ("M3_local_water", GLOBAL_FEATURES + history + LOCAL_FEATURES),
    )


def _evaluate_nested(
    rows: Sequence[Mapping[str, str]],
    target_field: str,
    history_direction: str,
    config: PredictionConfig,
    model_names: Optional[Sequence[str]] = None,
) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    output = []
    predictions = {}
    for model_name, features in _model_feature_sets(history_direction):
        if model_names is not None and model_name not in model_names:
            continue
        target, prediction, converged, iterations, used_names = _oof_predict(
            rows, target_field, features, config
        )
        output.append(
            {
                "model": model_name,
                "target_field": target_field,
                "row_count": len(rows),
                "target_mean_count": float(np.mean(target)),
                **_score(target, prediction),
                "all_folds_converged": converged,
                "total_optimizer_iterations": iterations,
                "used_feature_count": len(used_names),
                "used_features": ";".join(used_names),
            }
        )
        predictions[model_name] = (target, prediction)
    return output, predictions


def _history_from_sources(
    rows: Sequence[Mapping[str, str]],
    source_times_ps: np.ndarray,
    source_arcs: np.ndarray,
) -> list[dict[str, str]]:
    target_times = np.asarray([float(row["transition_time_ns"]) * 1000.0 for row in rows])
    target_arcs = np.asarray([int(row["primary_arc_index"]) for row in rows], dtype=int)
    n_arcs = int(rows[0]["arc_count"])
    modified = [dict(row) for row in rows]
    for index, (target_time, target_arc) in enumerate(zip(target_times, target_arcs)):
        lag = target_time - source_times_ps
        direct = np.abs(target_arc - source_arcs)
        distances = np.minimum(direct, n_arcs - direct)
        for window_name, (left_ps, right_ps) in (
            ("fast", (0.0, 5.0)),
            ("slow", (5.0, 50.0)),
        ):
            time_selected = (lag > left_ps) & (lag <= right_ps)
            for bin_name, left_arc, right_arc in ARC_BINS:
                count = np.count_nonzero(
                    time_selected & (distances >= left_arc) & (distances < right_arc)
                )
                modified[index][f"past_history_{window_name}_{bin_name}_count"] = str(
                    int(count)
                )
    return modified


def _null_history_rows(
    rows: Sequence[Mapping[str, str]],
    kind: str,
    rng: np.random.Generator,
    block_ps: float,
) -> list[dict[str, str]]:
    times = np.asarray([float(row["transition_time_ns"]) * 1000.0 for row in rows])
    arcs = np.asarray([int(row["primary_arc_index"]) for row in rows], dtype=int)
    blocks = np.asarray([int(row["time_block_200ps"]) for row in rows], dtype=int)
    block_count = int(np.max(blocks)) + 1
    n_arcs = int(rows[0]["arc_count"])
    if kind == "circular_arc_shift":
        offsets = rng.integers(0, n_arcs, size=block_count)
        source_times = times
        source_arcs = np.mod(arcs + offsets[blocks], n_arcs)
    elif kind == "permuted_history_blocks":
        permutation = rng.permutation(block_count)
        origin = float(np.min(times - blocks * block_ps))
        offsets = times - (origin + blocks * block_ps)
        source_times = origin + permutation[blocks] * block_ps + offsets
        source_arcs = arcs
    else:
        raise ValueError(f"unknown null kind {kind}")
    return _history_from_sources(rows, source_times, source_arcs)


def _delta_deviance(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    left_model: str,
    right_model: str,
) -> tuple[float, np.ndarray]:
    target, left = predictions[left_model]
    right_target, right = predictions[right_model]
    if not np.array_equal(target, right_target):
        raise ValueError("nested prediction targets differ")
    per_row = _poisson_deviance_rows(target, left) - _poisson_deviance_rows(target, right)
    return float(np.mean(per_row)), per_row


def _block_bootstrap(
    values: np.ndarray,
    blocks: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    unique = np.unique(blocks)
    draws = []
    for _ in range(samples):
        selected_blocks = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(blocks == block) for block in selected_blocks])
        draws.append(float(np.mean(values[indices])))
    array = np.asarray(draws)
    lower_p = float((1 + np.count_nonzero(array <= 0.0)) / (samples + 1))
    upper_p = float((1 + np.count_nonzero(array >= 0.0)) / (samples + 1))
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        lower_p,
        upper_p,
        min(1.0, 2.0 * min(lower_p, upper_p)),
    )


def analyze_case_window(
    rows: Sequence[Mapping[str, str]],
    window: str,
    config: PredictionConfig,
    rng: np.random.Generator,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    target_field = f"future_cross_arc_{window}_count"
    scores, predictions = _evaluate_nested(rows, target_field, "past", config)
    increments = []
    for left, right in (
        ("M0_arc_baseline", "M1_global_state"),
        ("M1_global_state", "M2_event_history"),
        ("M2_event_history", "M3_local_water"),
    ):
        delta, _ = _delta_deviance(predictions, left, right)
        increments.append(
            {
                "window": window,
                "left_model": left,
                "right_model": right,
                "delta_mean_poisson_deviance": delta,
            }
        )

    observed_delta, per_row_delta = _delta_deviance(
        predictions, "M1_global_state", "M2_event_history"
    )
    blocks = np.asarray([int(row["time_block_200ps"]) for row in rows], dtype=int)
    ci025, ci975, lower_p, upper_p, two_sided_p = _block_bootstrap(
        per_row_delta, blocks, config.bootstrap_samples, rng
    )
    _, reverse_predictions = _evaluate_nested(
        rows,
        f"past_cross_arc_{window}_count",
        "future",
        config,
        model_names=("M1_global_state", "M2_event_history"),
    )
    reverse_delta, _ = _delta_deviance(
        reverse_predictions, "M1_global_state", "M2_event_history"
    )
    controls = [
        {
            "window": window,
            "control_kind": "reversed_time",
            "sample": 0,
            "delta_mean_poisson_deviance": reverse_delta,
        }
    ]
    null_values = {"circular_arc_shift": [], "permuted_history_blocks": []}
    for kind, values in null_values.items():
        for sample in range(config.null_samples):
            null_rows = _null_history_rows(rows, kind, rng, config.block_ps)
            _, null_predictions = _evaluate_nested(
                null_rows,
                target_field,
                "past",
                config,
                model_names=("M2_event_history",),
            )
            null_predictions["M1_global_state"] = predictions["M1_global_state"]
            delta, _ = _delta_deviance(
                null_predictions, "M1_global_state", "M2_event_history"
            )
            values.append(delta)
            controls.append(
                {
                    "window": window,
                    "control_kind": kind,
                    "sample": sample,
                    "delta_mean_poisson_deviance": delta,
                }
            )
    evidence = {
        "window": window,
        "observed_history_delta_mean_poisson_deviance": observed_delta,
        "block_bootstrap_ci025": ci025,
        "block_bootstrap_ci975": ci975,
        "block_bootstrap_lower_p": lower_p,
        "block_bootstrap_upper_p": upper_p,
        "block_bootstrap_two_sided_p": two_sided_p,
        "reversed_time_delta_mean_poisson_deviance": reverse_delta,
        "circular_shift_null_mean": float(np.mean(null_values["circular_arc_shift"])),
        "circular_shift_empirical_upper_p": float(
            (1 + np.count_nonzero(np.asarray(null_values["circular_arc_shift"]) >= observed_delta))
            / (config.null_samples + 1)
        ),
        "block_permutation_null_mean": float(np.mean(null_values["permuted_history_blocks"])),
        "block_permutation_empirical_upper_p": float(
            (
                1
                + np.count_nonzero(
                    np.asarray(null_values["permuted_history_blocks"]) >= observed_delta
                )
            )
            / (config.null_samples + 1)
        ),
    }
    return scores, increments, evidence, controls


def run_analysis(
    event_state_path: Path,
    output_dir: Path,
    *,
    config: Optional[PredictionConfig] = None,
) -> dict[str, object]:
    """Run nested blocked prediction and negative controls for every case."""

    config = config or PredictionConfig()
    if config.bootstrap_samples < 20 or config.null_samples < 2:
        raise ValueError("bootstrap_samples must be >=20 and null_samples >=2")
    rows = _read_csv(event_state_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cases = list(dict.fromkeys(row["case_id"] for row in rows))
    score_rows = []
    increment_rows = []
    evidence_rows = []
    control_rows = []
    for case_index, case_id in enumerate(cases):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        for window_index, window in enumerate(("fast", "slow")):
            rng = np.random.default_rng(config.random_seed + 1000 * case_index + window_index)
            scores, increments, evidence, controls = analyze_case_window(
                case_rows, window, config, rng
            )
            score_rows.extend({"case_id": case_id, "window": window, **row} for row in scores)
            increment_rows.extend({"case_id": case_id, **row} for row in increments)
            evidence_rows.append({"case_id": case_id, **evidence})
            control_rows.extend({"case_id": case_id, **row} for row in controls)
    q_values = _bh_adjust(
        [float(row["block_bootstrap_two_sided_p"]) for row in evidence_rows]
    )
    for row, q_value in zip(evidence_rows, q_values):
        row["block_bootstrap_bh_q_primary_family"] = q_value
        row["interpretation"] = (
            "incremental_history_information_candidate"
            if float(row["observed_history_delta_mean_poisson_deviance"]) > 0.0
            and q_value < 0.05
            and float(row["circular_shift_empirical_upper_p"]) < 0.05
            and float(row["block_permutation_empirical_upper_p"]) < 0.05
            and float(row["observed_history_delta_mean_poisson_deviance"])
            > float(row["reversed_time_delta_mean_poisson_deviance"])
            else "no_qualified_incremental_history_information"
        )
        row["scientific_status"] = SCIENTIFIC_STATUS
    _write_csv(output / "model_scores.csv", score_rows)
    _write_csv(output / "incremental_scores.csv", increment_rows)
    _write_csv(output / "history_evidence.csv", evidence_rows)
    _write_csv(output / "history_null_controls.csv", control_rows)
    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "row_count": len(rows),
        "fold_count": config.fold_count,
        "embargo_blocks": config.embargo_blocks,
        "penalty": config.penalty,
        "bootstrap_samples": config.bootstrap_samples,
        "null_samples_per_kind": config.null_samples,
        "random_seed": config.random_seed,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        **summary,
        "event_state_table": {"path": str(event_state_path), "sha256": _sha256(event_state_path)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "# TPCL event-history prediction\n\n"
        "Nested ridge-Poisson models predict future cross-arc cluster counts from "
        "pre-event information. Five contiguous time folds use a one-block embargo. "
        "A positive deviance reduction means incremental out-of-fold information; it "
        "is not a causal propagation effect or a physical rate law. Whole-block "
        "bootstrap intervals and all negative controls remain within one trajectory.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-state-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--null-samples", type=int, default=200)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        args.event_state_table,
        args.output_dir,
        config=PredictionConfig(
            bootstrap_samples=args.bootstrap_samples,
            null_samples=args.null_samples,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
