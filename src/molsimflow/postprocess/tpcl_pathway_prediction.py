"""Classify TPCL event responses and predict yielding from pre-event state.

The analysis is deliberately retrospective.  It uses surface-blind response
clustering, matched event/control rows, blocked out-of-fold prediction, and
leave-one-surface-out tests.  None of these operations establish causality or
replicate-level uncertainty.
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
    silhouette_score,
)

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_SURFACE_BLIND_PHENOTYPE_AND_BLOCKED_PREDICTION_"
    "NOT_CAUSAL_OR_REPLICATE_LEVEL_EVIDENCE"
)

CHEMISTRY_FEATURES = (
    "pre_local_nearest_site_distance_A",
    "pre_local_local_ch3_fraction",
    "pre_local_chemical_boundary_distance_proxy_A",
)

MEAN_HYDRATION_FEATURES = (
    "pre_local_local_hydration_areal_density_A-2",
    "pre_local_local_water_dipole_cos_z",
    "pre_local_local_n2_min_distance_A",
)

TOPOLOGY_GEOMETRY_FEATURES = (
    "pre_local_local_water_water_hbond_degree",
    "pre_local_local_surface_water_hbond_per_h2o",
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

MODEL_FEATURES = (
    ("M0_intercept", ()),
    ("M1_local_chemistry", CHEMISTRY_FEATURES),
    ("M2_mean_hydration", CHEMISTRY_FEATURES + MEAN_HYDRATION_FEATURES),
    (
        "M3_topology_geometry",
        CHEMISTRY_FEATURES + MEAN_HYDRATION_FEATURES + TOPOLOGY_GEOMETRY_FEATURES,
    ),
)

RESPONSE_FEATURES = (
    "response_affected_arc_fraction",
    "response_event_size_residual_A2",
    "response_event_size_radius_A2",
    "response_mean_radius_change_A",
    "response_primary_residual_change_A",
    "response_delta_mode_2_amplitude_A",
    "response_delta_mode_3_amplitude_A",
    "response_delta_mode_4_amplitude_A",
    "response_delta_mode_5_amplitude_A",
    "response_delta_mode_6_amplitude_A",
    "response_delta_contact_contour_area_A2",
    "response_delta_contact_contour_perimeter_A",
    "response_delta_contact_contour_circularity",
    "response_delta_molecular_center_cap_angle_candidate_deg",
)


@dataclass(frozen=True)
class PathwayConfig:
    """Frozen clustering, prediction, and uncertainty settings."""

    fold_count: int = 5
    embargo_blocks: int = 1
    penalty: float = 0.1
    bootstrap_samples: int = 2000
    random_seed: int = 20260904
    cluster_min: int = 2
    cluster_max: int = 6
    cluster_n_init: int = 50
    silhouette_gate: float = 0.20
    median_leave_one_feature_ari_gate: float = 0.70
    minimum_cluster_fraction_gate: float = 0.05


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


def _bh_adjust(values: Sequence[float]) -> list[float]:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(p_values[index]) * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _float(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if math.isinf(value):
        raise ValueError(f"infinite {field}")
    return value


def _response_matrix(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Robust-standardize response columns within each surface."""

    matrix = np.full((len(rows), len(RESPONSE_FEATURES)), np.nan)
    cases = sorted({row["case_id"] for row in rows})
    for case_id in cases:
        indices = [index for index, row in enumerate(rows) if row["case_id"] == case_id]
        raw = np.asarray(
            [[_float(rows[index], field) for field in RESPONSE_FEATURES] for index in indices],
            dtype=float,
        )
        if np.any(~np.isfinite(raw)):
            raise ValueError(f"{case_id}: response matrix is not finite")
        median = np.median(raw, axis=0)
        scale = np.quantile(raw, 0.75, axis=0) - np.quantile(raw, 0.25, axis=0)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        matrix[indices] = (raw - median) / scale
    if np.any(~np.isfinite(matrix)):
        raise ValueError("standardized response matrix is not finite")
    return matrix


def _canonical_labels(labels: np.ndarray, rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    ordering = []
    for label in np.unique(labels):
        selected = labels == label
        ordering.append(
            (
                float(
                    np.median(
                        [_float(row, "response_affected_arc_fraction") for row, keep in zip(rows, selected) if keep]
                    )
                ),
                float(
                    np.median(
                        [_float(row, "response_event_size_residual_A2") for row, keep in zip(rows, selected) if keep]
                    )
                ),
                int(label),
            )
        )
    mapping = {old: new for new, (_, _, old) in enumerate(sorted(ordering))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def cluster_responses(
    rows: Sequence[Mapping[str, str]], config: PathwayConfig
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Choose k by silhouette without exposing surface labels to clustering."""

    matrix = _response_matrix(rows)
    score_rows = []
    models = {}
    for cluster_count in range(config.cluster_min, config.cluster_max + 1):
        model = KMeans(
            n_clusters=cluster_count,
            n_init=config.cluster_n_init,
            random_state=config.random_seed,
        ).fit(matrix)
        counts = np.bincount(model.labels_, minlength=cluster_count)
        score_rows.append(
            {
                "cluster_count": cluster_count,
                "silhouette": float(silhouette_score(matrix, model.labels_)),
                "minimum_cluster_fraction": float(np.min(counts) / len(rows)),
                "inertia": float(model.inertia_),
            }
        )
        models[cluster_count] = model
    selected = max(score_rows, key=lambda row: (row["silhouette"], -row["cluster_count"]))
    selected_k = int(selected["cluster_count"])
    labels = _canonical_labels(models[selected_k].labels_, rows)
    stability_rows = []
    for feature_index, omitted in enumerate(RESPONSE_FEATURES):
        reduced = np.delete(matrix, feature_index, axis=1)
        reduced_labels = KMeans(
            n_clusters=selected_k,
            n_init=config.cluster_n_init,
            random_state=config.random_seed,
        ).fit_predict(reduced)
        stability_rows.append(
            {
                "omitted_feature": omitted,
                "adjusted_rand_index": float(adjusted_rand_score(labels, reduced_labels)),
            }
        )
    median_ari = float(np.median([row["adjusted_rand_index"] for row in stability_rows]))
    stable = (
        float(selected["silhouette"]) >= config.silhouette_gate
        and median_ari >= config.median_leave_one_feature_ari_gate
        and float(selected["minimum_cluster_fraction"])
        >= config.minimum_cluster_fraction_gate
    )
    summary = {
        "selected_cluster_count": selected_k,
        "selected_silhouette": float(selected["silhouette"]),
        "selected_minimum_cluster_fraction": float(selected["minimum_cluster_fraction"]),
        "median_leave_one_feature_ari": median_ari,
        "phenotype_stable": bool(stable),
        "phenotype_status": "STABLE" if stable else "UNSTABLE_CONTINUOUS_RESPONSE_ONLY",
    }
    return labels, score_rows, stability_rows, summary


def _same_number(left: object, right: object) -> bool:
    first, second = float(left), float(right)
    if math.isnan(first) and math.isnan(second):
        return True
    return math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-12)


def deduplicate_risk_rows(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Collapse identical snapshot anchors so no coordinate can cross a fold."""

    grouped: dict[tuple[str, int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], int(row["primary_arc_index"]), int(row["sample_step"]))].append(row)
    output = []
    duplicate_keys = 0
    for key, selected in grouped.items():
        reference = selected[0]
        if len(selected) > 1:
            duplicate_keys += 1
        labels = {int(row["is_event"]) for row in selected}
        if len(labels) != 1:
            raise ValueError(f"risk anchor {key} has conflicting labels")
        for row in selected[1:]:
            for field in dict(MODEL_FEATURES)["M3_topology_geometry"]:
                if not _same_number(reference[field], row[field]):
                    raise ValueError(f"risk anchor {key} has inconsistent {field}")
        combined: dict[str, object] = dict(reference)
        combined["risk_set_weight"] = sum(float(row["risk_set_weight"]) for row in selected)
        combined["aggregated_source_row_count"] = len(selected)
        output.append(combined)
    output.sort(key=lambda row: (str(row["case_id"]), int(row["sample_step"]), int(row["primary_arc_index"])))
    audit = {
        "input_row_count": len(rows),
        "unique_anchor_count": len(output),
        "duplicate_anchor_key_count": duplicate_keys,
        "conflicting_label_anchor_count": 0,
        "anchor_key": ["case_id", "primary_arc_index", "sample_step"],
    }
    return output, audit


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


def _fold_labels(rows: Sequence[Mapping[str, object]], block_field: str, fold_count: int) -> np.ndarray:
    blocks = np.asarray([int(row[block_field]) for row in rows], dtype=int)
    unique = np.unique(blocks)
    if len(unique) < fold_count:
        raise ValueError("fewer time blocks than folds")
    mapping = {
        int(block): min(index * fold_count // len(unique), fold_count - 1)
        for index, block in enumerate(unique)
    }
    return np.asarray([mapping[int(block)] for block in blocks], dtype=int)


def _binary_prediction(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    penalty: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    target = np.asarray([int(row["is_event"]) for row in train_rows], dtype=int)
    weight = np.asarray([float(row["risk_set_weight"]) for row in train_rows], dtype=float)
    if len(np.unique(target)) != 2:
        raise ValueError("binary training fold does not contain both classes")
    train, test, names = _matrices(train_rows, test_rows, features)
    if not names:
        probability = float(np.average(target, weights=weight))
        return np.full(len(test_rows), probability), names
    model = LogisticRegression(C=1.0 / penalty, max_iter=1000, solver="lbfgs")
    model.fit(train, target, sample_weight=weight)
    return model.predict_proba(test)[:, 1], names


def _multiclass_prediction(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    penalty: float,
    classes: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    target = np.asarray([int(row["phenotype_id"]) for row in train_rows], dtype=int)
    train, test, names = _matrices(train_rows, test_rows, features)
    if not names:
        counts = np.asarray([np.count_nonzero(target == item) for item in classes], dtype=float)
        probabilities = counts / np.sum(counts)
        return np.tile(probabilities, (len(test_rows), 1)), names
    model = LogisticRegression(C=1.0 / penalty, max_iter=1000, solver="lbfgs")
    model.fit(train, target)
    raw = model.predict_proba(test)
    output = np.full((len(test_rows), len(classes)), 1.0e-12)
    for source, label in enumerate(model.classes_):
        output[:, np.flatnonzero(classes == label)[0]] = raw[:, source]
    output /= np.sum(output, axis=1, keepdims=True)
    return output, names


def _ridge_prediction(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    penalty: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    target = np.asarray(
        [float(row["response_affected_arc_fraction"]) for row in train_rows], dtype=float
    )
    train, test, names = _matrices(train_rows, test_rows, features)
    if not names:
        return np.full(len(test_rows), float(np.mean(target))), names
    model = Ridge(alpha=penalty).fit(train, target)
    return np.asarray(model.predict(test), dtype=float), names


def _split_predict(
    rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
    config: PathwayConfig,
    *,
    task: str,
    evaluation: str,
    held_case: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if evaluation == "within_surface":
        selected = [row for row in rows if row["case_id"] == held_case]
        block_field = "sample_time_block_200ps" if task == "risk" else "time_block_200ps"
        folds = _fold_labels(selected, block_field, config.fold_count)
        blocks = np.asarray([int(row[block_field]) for row in selected], dtype=int)
        prediction = None
        names = set()
        for fold in range(config.fold_count):
            test_indices = np.flatnonzero(folds == fold)
            test_blocks = set(blocks[test_indices])
            excluded = {
                candidate
                for block in test_blocks
                for candidate in range(
                    block - config.embargo_blocks, block + config.embargo_blocks + 1
                )
            }
            train_indices = np.asarray(
                [index for index, block in enumerate(blocks) if block not in excluded], dtype=int
            )
            if not len(test_indices) or len(train_indices) < 20:
                raise ValueError(f"{held_case}/{task}/fold {fold}: insufficient rows")
            train_rows = [selected[index] for index in train_indices]
            test_rows = [selected[index] for index in test_indices]
            if task == "risk":
                fold_prediction, used = _binary_prediction(
                    train_rows, test_rows, features, config.penalty
                )
                if prediction is None:
                    prediction = np.full(len(selected), np.nan)
            elif task == "phenotype":
                classes = np.unique([int(row["phenotype_id"]) for row in rows])
                fold_prediction, used = _multiclass_prediction(
                    train_rows, test_rows, features, config.penalty, classes
                )
                if prediction is None:
                    prediction = np.full((len(selected), len(classes)), np.nan)
            elif task == "extent":
                fold_prediction, used = _ridge_prediction(
                    train_rows, test_rows, features, config.penalty
                )
                if prediction is None:
                    prediction = np.full(len(selected), np.nan)
            else:
                raise ValueError(f"unknown task {task}")
            prediction[test_indices] = fold_prediction
            names.update(used)
        test_rows = selected
    elif evaluation == "leave_one_surface_out":
        train_rows = [row for row in rows if row["case_id"] != held_case]
        test_rows = [row for row in rows if row["case_id"] == held_case]
        if task == "risk":
            prediction, used = _binary_prediction(train_rows, test_rows, features, config.penalty)
        elif task == "phenotype":
            classes = np.unique([int(row["phenotype_id"]) for row in rows])
            prediction, used = _multiclass_prediction(
                train_rows, test_rows, features, config.penalty, classes
            )
        elif task == "extent":
            prediction, used = _ridge_prediction(train_rows, test_rows, features, config.penalty)
        else:
            raise ValueError(f"unknown task {task}")
        names = set(used)
    else:
        raise ValueError(f"unknown evaluation {evaluation}")
    if prediction is None or np.any(~np.isfinite(prediction)):
        raise ValueError(f"{held_case}/{task}/{evaluation}: incomplete prediction")
    if task == "risk":
        target = np.asarray([int(row["is_event"]) for row in test_rows], dtype=int)
    elif task == "phenotype":
        target = np.asarray([int(row["phenotype_id"]) for row in test_rows], dtype=int)
    else:
        target = np.asarray(
            [float(row["response_affected_arc_fraction"]) for row in test_rows], dtype=float
        )
    return target, np.asarray(prediction), tuple(sorted(names))


def _loss_and_score(
    rows: Sequence[Mapping[str, object]], task: str, target: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    if task == "risk":
        weight = np.asarray([float(row["risk_set_weight"]) for row in rows], dtype=float)
        clipped = np.clip(prediction, 1.0e-12, 1.0 - 1.0e-12)
        losses = -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))
        scores = {
            "primary_loss": float(np.average(losses, weights=weight)),
            "roc_auc": float(roc_auc_score(target, prediction, sample_weight=weight)),
            "brier": float(np.average((target - prediction) ** 2, weights=weight)),
        }
    elif task == "phenotype":
        classes = np.arange(prediction.shape[1])
        losses = -np.log(np.clip(prediction[np.arange(len(target)), target], 1.0e-12, 1.0))
        scores = {
            "primary_loss": float(log_loss(target, prediction, labels=classes)),
            "balanced_accuracy": float(
                balanced_accuracy_score(target, np.argmax(prediction, axis=1))
            ),
        }
    else:
        losses = (target - prediction) ** 2
        scores = {
            "primary_loss": float(mean_squared_error(target, prediction)),
            "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
            "target_prediction_pearson_r": (
                float(np.corrcoef(target, prediction)[0, 1])
                if np.std(target) > 0.0 and np.std(prediction) > 0.0
                else math.nan
            ),
        }
    return losses, scores


def _bootstrap_delta(
    difference: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    task: str,
    config: PathwayConfig,
    seed_offset: int,
) -> dict[str, float]:
    block_field = "sample_time_block_200ps" if task == "risk" else "time_block_200ps"
    blocks = np.asarray([int(row[block_field]) for row in rows], dtype=int)
    weights = (
        np.asarray([float(row["risk_set_weight"]) for row in rows], dtype=float)
        if task == "risk"
        else np.ones(len(rows))
    )
    unique = np.unique(blocks)
    rng = np.random.default_rng(config.random_seed + seed_offset)
    draws = []
    for _ in range(config.bootstrap_samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(blocks == block) for block in selected])
        draws.append(float(np.average(difference[indices], weights=weights[indices])))
    values = np.asarray(draws)
    lower_p = float((1 + np.count_nonzero(values <= 0.0)) / (len(values) + 1))
    upper_p = float((1 + np.count_nonzero(values >= 0.0)) / (len(values) + 1))
    return {
        "bootstrap_ci025": float(np.quantile(values, 0.025)),
        "bootstrap_ci975": float(np.quantile(values, 0.975)),
        "bootstrap_lower_p": lower_p,
        "bootstrap_upper_p": upper_p,
        "bootstrap_two_sided_p": min(1.0, 2.0 * min(lower_p, upper_p)),
    }


def _evaluate_task(
    rows: Sequence[Mapping[str, object]],
    task: str,
    config: PathwayConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    score_rows = []
    evidence_rows = []
    cases = sorted({str(row["case_id"]) for row in rows})
    seed_offset = 0
    for evaluation in ("within_surface", "leave_one_surface_out"):
        for held_case in cases:
            selected_rows = [row for row in rows if row["case_id"] == held_case]
            model_losses = {}
            for model_name, features in MODEL_FEATURES:
                target, prediction, used = _split_predict(
                    rows,
                    features,
                    config,
                    task=task,
                    evaluation=evaluation,
                    held_case=held_case,
                )
                losses, scores = _loss_and_score(selected_rows, task, target, prediction)
                model_losses[model_name] = losses
                score_rows.append(
                    {
                        "task": task,
                        "evaluation": evaluation,
                        "held_surface": held_case,
                        "model": model_name,
                        "row_count": len(selected_rows),
                        **scores,
                        "used_feature_count": len(used),
                        "used_features": ";".join(used),
                    }
                )
            difference = model_losses["M0_intercept"] - model_losses["M3_topology_geometry"]
            weights = (
                np.asarray([float(row["risk_set_weight"]) for row in selected_rows])
                if task == "risk"
                else np.ones(len(selected_rows))
            )
            evidence_rows.append(
                {
                    "task": task,
                    "evaluation": evaluation,
                    "held_surface": held_case,
                    "delta_primary_loss_M0_minus_M3": float(
                        np.average(difference, weights=weights)
                    ),
                    **_bootstrap_delta(difference, selected_rows, task, config, seed_offset),
                }
            )
            seed_offset += 1
    q_values = _bh_adjust([row["bootstrap_two_sided_p"] for row in evidence_rows])
    for row, q_value in zip(evidence_rows, q_values):
        row["bootstrap_bh_q_task_family"] = q_value
        row["qualified_incremental_information"] = bool(
            row["delta_primary_loss_M0_minus_M3"] > 0.0
            and row["bootstrap_ci025"] > 0.0
            and q_value < 0.05
        )
    return score_rows, evidence_rows


def run_analysis(
    event_state_table: Path,
    risk_table: Path,
    output_dir: Path,
    *,
    config: Optional[PathwayConfig] = None,
) -> dict[str, object]:
    """Run the frozen P3 phenotype, yielding-risk, and response analysis."""

    config = config or PathwayConfig()
    if config.penalty <= 0.0 or config.bootstrap_samples < 20:
        raise ValueError("positive penalty and at least 20 bootstrap samples are required")
    event_rows: list[dict[str, object]] = [dict(row) for row in _read_csv(event_state_table)]
    raw_risk = _read_csv(risk_table)
    risk_rows, leakage_audit = deduplicate_risk_rows(raw_risk)
    labels, cluster_scores, stability_rows, cluster_summary = cluster_responses(
        event_rows, config
    )
    for row, label in zip(event_rows, labels):
        row["phenotype_id"] = int(label)

    assignments = [
        {
            "case_id": row["case_id"],
            "cluster_id": int(row["cluster_id"]),
            "primary_event_id": int(row["primary_event_id"]),
            "time_block_200ps": int(row["time_block_200ps"]),
            "phenotype_id": int(row["phenotype_id"]),
            **{field: float(row[field]) for field in RESPONSE_FEATURES},
        }
        for row in event_rows
    ]
    distribution = []
    for case_id in sorted({str(row["case_id"]) for row in event_rows}):
        case_labels = [int(row["phenotype_id"]) for row in event_rows if row["case_id"] == case_id]
        for label in range(int(cluster_summary["selected_cluster_count"])):
            count = case_labels.count(label)
            distribution.append(
                {
                    "case_id": case_id,
                    "phenotype_id": label,
                    "event_count": count,
                    "within_surface_fraction": count / len(case_labels),
                }
            )

    risk_scores, risk_evidence = _evaluate_task(risk_rows, "risk", config)
    extent_scores, extent_evidence = _evaluate_task(event_rows, "extent", config)
    phenotype_scores: list[dict[str, object]] = []
    phenotype_evidence: list[dict[str, object]] = []
    if cluster_summary["phenotype_stable"]:
        phenotype_scores, phenotype_evidence = _evaluate_task(
            event_rows, "phenotype", config
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "phenotype_assignments.csv", assignments)
    _write_csv(output / "clustering_scores.csv", cluster_scores)
    _write_csv(output / "clustering_feature_stability.csv", stability_rows)
    _write_csv(output / "phenotype_surface_distribution.csv", distribution)
    _write_csv(output / "risk_model_scores.csv", risk_scores)
    _write_csv(output / "extent_model_scores.csv", extent_scores)
    if phenotype_scores:
        _write_csv(output / "phenotype_model_scores.csv", phenotype_scores)
    primary_evidence = risk_evidence + extent_evidence + phenotype_evidence
    _write_csv(output / "primary_prediction_evidence.csv", primary_evidence)
    (output / "leakage_audit.json").write_text(
        json.dumps(
            {
                **leakage_audit,
                "prediction_anchor_time_field": "sample_time_block_200ps",
                "within_surface_fold_count": config.fold_count,
                "embargo_blocks": config.embargo_blocks,
                "leave_one_surface_out": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "status": "PASS",
        "event_row_count": len(event_rows),
        "risk_input_row_count": len(raw_risk),
        "risk_unique_anchor_count": len(risk_rows),
        "case_count": len({row["case_id"] for row in event_rows}),
        **cluster_summary,
        "risk_qualified_count": sum(
            bool(row["qualified_incremental_information"]) for row in risk_evidence
        ),
        "extent_qualified_count": sum(
            bool(row["qualified_incremental_information"]) for row in extent_evidence
        ),
        "phenotype_qualified_count": sum(
            bool(row["qualified_incremental_information"]) for row in phenotype_evidence
        ),
        "bootstrap_samples": config.bootstrap_samples,
        "random_seed": config.random_seed,
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
                "config": config.__dict__,
                "response_features": RESPONSE_FEATURES,
                "model_features": {name: fields for name, fields in MODEL_FEATURES},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "REPORT.md").write_text(
        "# TPCL pathway taxonomy and prediction\n\n"
        "Response vectors are robust-standardized within each surface and clustered "
        "without surface labels. Phenotype names are not assigned automatically. "
        "Matched risk anchors are deduplicated before actual-time blocked prediction. "
        "All predictive comparisons use pre-event fields, training-fold imputation, "
        "a one-block embargo, and leave-one-surface-out tests. A PASS is computational "
        "acceptance only; results remain retrospective single-trajectory evidence and "
        "do not establish chemistry-driven causality, propagation, physical rates, or "
        "replicate-level uncertainty.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-state-table", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = PathwayConfig(bootstrap_samples=args.bootstrap_samples)
    run_analysis(args.event_state_table, args.risk_table, args.output_dir, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
