"""Join patch edge events, surface motifs, and pre-event water renewal models."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
)
from molsimflow.postprocess.nanobubble_edge_events import (
    anchor_displacements,
    rolling_median,
)

M0_FEATURES = (
    "global_water_velocity_mps", "global_oxygen_density_A3", "bubble_height_A",
    "bubble_drive_width_A", "global_film_water_oxygen_count",
    "recent_edge_velocity_A_ps", "water_oxygen_count", "exposed_ch3_fraction",
    "mean_local_boundary_edge_fraction",
)
M1_FEATURES = M0_FEATURES + ("turnover_history_mean",)
HIGH_CADENCE_WATER_FEATURES = (
    "turnover_history_mean", "rewiring_loss_history_mean",
    "dipole_rotation_history_deg", "hbond_break_rate_history_per_ps",
    "hbond_formation_rate_history_per_ps", "raw_force_impulse_abs_history",
)


def _m1_features(raw: dict[str, object]) -> tuple[str, ...]:
    mode = str(raw.get("water_feature_mode", "turnover"))
    if mode == "turnover":
        return M1_FEATURES
    if mode == "high_cadence":
        return M0_FEATURES + HIGH_CADENCE_WATER_FEATURES
    raise ValueError("water_feature_mode must be turnover or high_cadence")


def _recent_mean(table: pd.DataFrame, column: str, *, absolute: bool = False) -> float:
    if column not in table:
        return math.nan
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
    if absolute:
        values = np.abs(values)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else math.nan


def _read(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if table.empty:
        raise ValueError(f"Empty table: {path}")
    return table


def _segments(table: pd.DataFrame, cadence_ps: float) -> list[pd.DataFrame]:
    selected = table[
        (table.observer_valid.astype(int) == 1)
        & np.isfinite(table.edge_coordinate_A.astype(float))
    ].sort_values("time_ps")
    if selected.empty:
        return []
    breaks = np.r_[True, np.diff(selected.time_ps.to_numpy(float)) > cadence_ps + 1.0e-8]
    group = np.cumsum(breaks)
    return [part.reset_index(drop=True) for _, part in selected.groupby(group)]


def _segment_candidates(
    segment: pd.DataFrame,
    threshold_A: float,
    *,
    cadence_ps: float,
    smoothing_ps: float,
    pre_window_ps: float,
    post_window_ps: float,
    persistence_ps: float,
    persistence_fraction: float,
    refractory_ps: float,
) -> list[dict[str, object]]:
    values = rolling_median(
        segment.edge_coordinate_A.to_numpy(float),
        max(1, round(smoothing_ps / cadence_ps)),
    )
    pre_frames = max(1, round(pre_window_ps / cadence_ps))
    post_frames = max(1, round(post_window_ps / cadence_ps))
    persist_frames = max(1, round(persistence_ps / cadence_ps))
    refractory_frames = max(1, round(refractory_ps / cadence_ps))
    indices, changes, pre_levels, post_levels = anchor_displacements(
        values, pre_frames, post_frames
    )
    candidates = []
    for index, change, pre, post in zip(indices, changes, pre_levels, post_levels):
        if abs(change) < threshold_A or index + persist_frames >= len(values):
            continue
        sign = 1.0 if change > 0.0 else -1.0
        midpoint = pre + 0.5 * change
        persistence = float(np.mean(
            sign * (values[index + 1:index + persist_frames + 1] - midpoint) >= 0.0
        ))
        if persistence >= persistence_fraction:
            candidates.append({
                "index": int(index), "displacement_A": float(change),
                "pre_position_A": float(pre), "post_position_A": float(post),
                "persistence_fraction": persistence,
                "start_index": int(index - pre_frames), "end_index": int(index + post_frames),
            })
    selected = []
    for candidate in sorted(candidates, key=lambda row: abs(float(row["displacement_A"])), reverse=True):
        if all(abs(int(candidate["index"]) - int(row["index"])) > refractory_frames for row in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda row: int(row["index"]))


def _edge_process(edge: str, displacement: float) -> str:
    gas_advances = displacement > 0.0 if edge == "front" else displacement < 0.0
    return "gas_advance_liquid_retreat" if gas_advances else "gas_retreat_liquid_advance"


def _null_changes(
    table: pd.DataFrame,
    *,
    cadence_ps: float,
    smoothing_ps: float,
    pre_window_ps: float,
    post_window_ps: float,
) -> np.ndarray:
    output = []
    for segment in _segments(table, cadence_ps):
        values = rolling_median(
            segment.edge_coordinate_A.to_numpy(float),
            max(1, round(smoothing_ps / cadence_ps)),
        )
        if len(values) <= round((pre_window_ps + post_window_ps) / cadence_ps) + 1:
            continue
        _, changes, _, _ = anchor_displacements(
            values, max(1, round(pre_window_ps / cadence_ps)),
            max(1, round(post_window_ps / cadence_ps)),
        )
        output.extend(changes.tolist())
    return np.asarray(output, dtype=float)


def _detect_patch_events(
    driven: pd.DataFrame,
    baseline: pd.DataFrame,
    geometry: pd.DataFrame,
    raw: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cadence = float(raw["sample_interval_ps"])
    events, calibrations, sensitivity = [], [], []
    geometry_by_step = geometry.set_index("step")
    total_baseline_frames = baseline.step.nunique()
    total_driven_frames = driven.step.nunique()
    keys = sorted(set(zip(driven.patch_id, driven.edge)))
    for patch_id, edge in keys:
        d = driven[(driven.patch_id == patch_id) & (driven.edge == edge)].copy()
        b = baseline[(baseline.patch_id == patch_id) & (baseline.edge == edge)].copy()
        d_coverage = float(np.mean(d.observer_valid.astype(int)))
        b_coverage = float(np.mean(b.observer_valid.astype(int)))
        null = _null_changes(
            b, cadence_ps=cadence, smoothing_ps=float(raw["smoothing_ps"]),
            pre_window_ps=float(raw["pre_window_ps"]),
            post_window_ps=float(raw["post_window_ps"]),
        )
        status = "PASS"
        if (
            d_coverage < float(raw["minimum_patch_coverage"])
            or b_coverage < float(raw["minimum_patch_coverage"])
            or len(null) < int(raw["minimum_null_anchors"])
        ):
            status = "HOLD_LOW_COVERAGE"
        threshold = (
            max(
                float(raw["minimum_displacement_A"]),
                float(np.quantile(np.abs(null), float(raw["f0_quantile"]))),
            )
            if len(null) else math.nan
        )
        calibrations.append({
            "case_id": d.case_id.iloc[0], "branch_id": d.branch_id.iloc[0],
            "direction": d.direction.iloc[0], "patch_id": patch_id, "edge": edge,
            "driven_valid_fraction": d_coverage,
            "baseline_valid_fraction": b_coverage,
            "driven_total_frames": total_driven_frames,
            "baseline_total_frames": total_baseline_frames,
            "null_anchor_count": len(null), "f0_quantile": raw["f0_quantile"],
            "threshold_A": threshold, "calibration_status": status,
        })
        if status != "PASS":
            continue
        for quantile in raw.get("sensitivity_quantiles", [0.99, 0.995, 0.999]):
            q_threshold = max(
                float(raw["minimum_displacement_A"]),
                float(np.quantile(np.abs(null), float(quantile))),
            )
            count = 0
            for segment in _segments(d, cadence):
                count += len(_segment_candidates(
                    segment, q_threshold, cadence_ps=cadence,
                    smoothing_ps=float(raw["smoothing_ps"]),
                    pre_window_ps=float(raw["pre_window_ps"]),
                    post_window_ps=float(raw["post_window_ps"]),
                    persistence_ps=float(raw["persistence_ps"]),
                    persistence_fraction=float(raw["persistence_fraction"]),
                    refractory_ps=float(raw["refractory_ps"]),
                ))
            sensitivity.append({
                "case_id": d.case_id.iloc[0], "direction": d.direction.iloc[0],
                "patch_id": patch_id, "edge": edge, "f0_quantile": quantile,
                "threshold_A": q_threshold, "event_count": count,
            })
        for segment in _segments(d, cadence):
            for detected in _segment_candidates(
                segment, threshold, cadence_ps=cadence,
                smoothing_ps=float(raw["smoothing_ps"]),
                pre_window_ps=float(raw["pre_window_ps"]),
                post_window_ps=float(raw["post_window_ps"]),
                persistence_ps=float(raw["persistence_ps"]),
                persistence_fraction=float(raw["persistence_fraction"]),
                refractory_ps=float(raw["refractory_ps"]),
            ):
                anchor = segment.iloc[int(detected["index"])]
                start = segment.iloc[int(detected["start_index"])]
                end = segment.iloc[int(detected["end_index"])]
                global_row = geometry_by_step.loc[int(anchor.step)]
                reasons = []
                if int(anchor.global_pbc_spanning_drive):
                    reasons.append("drive_spanning")
                if int(anchor.global_pbc_spanning_transverse):
                    reasons.append("transverse_spanning")
                if int(anchor.water_oxygen_count) < int(raw["minimum_water_oxygen"]):
                    reasons.append("local_water_underpopulated")
                displacement = float(detected["displacement_A"])
                events.append({
                    "case_id": anchor.case_id, "branch_id": anchor.branch_id,
                    "direction": anchor.direction, "patch_id": patch_id,
                    "patch_index": int(anchor.patch_index), "edge": edge,
                    "process": _edge_process(edge, displacement),
                    "start_step": int(start.step), "anchor_step": int(anchor.step),
                    "end_step": int(end.step), "start_time_ps": float(start.time_ps),
                    "anchor_time_ps": float(anchor.time_ps), "end_time_ps": float(end.time_ps),
                    "pre_position_A": detected["pre_position_A"],
                    "post_position_A": detected["post_position_A"],
                    "displacement_A": displacement, "threshold_A": threshold,
                    "score": abs(displacement) / threshold,
                    "persistence_fraction": detected["persistence_fraction"],
                    "validity_status": "VALID" if not reasons else "AMBIGUOUS",
                    "ambiguity_reason": ";".join(reasons),
                    "water_oxygen_count": int(anchor.water_oxygen_count),
                    "water_water_hbond_degree": float(anchor.water_water_hbond_degree),
                    "surface_water_hbond_per_intact_water": float(anchor.surface_water_hbond_per_intact_water),
                    "exposed_ch3_fraction": float(anchor.exposed_ch3_fraction),
                    "mean_local_boundary_edge_fraction": float(anchor.mean_local_boundary_edge_fraction),
                    "nearest_site_id": anchor.nearest_site_id,
                    "nearest_site_type": anchor.nearest_site_type,
                    "bubble_height_A": float(global_row.bubble_height_A),
                    "bubble_drive_width_A": float(global_row.bubble_drive_width_A),
                    "global_film_water_oxygen_count": int(global_row.film_water_oxygen_count),
                })
    return events, calibrations, sensitivity


def _qualified_patch_keys(
    calibrations: Sequence[dict[str, object]],
) -> set[tuple[str, str]]:
    """Return patch-edge observers admitted by the frozen coverage contract."""

    return {
        (str(row["patch_id"]), str(row["edge"]))
        for row in calibrations
        if str(row["calibration_status"]) == "PASS"
    }


def _filter_patch_edges(
    table: pd.DataFrame, qualified: set[tuple[str, str]]
) -> pd.DataFrame:
    """Keep only observer-qualified patch-edge rows without changing thresholds."""

    if table.empty:
        return table.copy()
    mask = [
        (str(patch_id), str(edge)) in qualified
        for patch_id, edge in zip(table.patch_id, table.edge)
    ]
    return table.loc[mask].copy()


def _risk_rows(
    frames: pd.DataFrame,
    intervals: pd.DataFrame,
    geometry: pd.DataFrame,
    events: Sequence[dict[str, object]],
    raw: dict[str, object],
    m1_features: Sequence[str],
) -> list[dict[str, object]]:
    geometry_by_step = geometry.set_index("step")
    valid_events = [row for row in events if row["validity_status"] == "VALID"]
    output = []
    history = float(raw["turnover_history_ps"])
    horizon = float(raw["prediction_horizon_ps"])
    stride = float(raw["risk_stride_ps"])
    for (patch_id, edge), group in intervals.groupby(["patch_id", "edge"]):
        group = group.sort_values("end_time_ps")
        frame_group = frames[(frames.patch_id == patch_id) & (frames.edge == edge)].set_index("step")
        for interval in group.itertuples(index=False):
            time = float(interval.end_time_ps)
            if not math.isclose(time / stride, round(time / stride), abs_tol=1.0e-8):
                continue
            recent = group[
                (group.end_time_ps.astype(float) > time - history - 1.0e-8)
                & (group.end_time_ps.astype(float) <= time + 1.0e-8)
            ]
            if recent.empty or float(recent.start_time_ps.min()) > time - history + 1.0e-8:
                continue
            if int(interval.end_step) not in frame_group.index or int(interval.end_step) not in geometry_by_step.index:
                continue
            frame = frame_group.loc[int(interval.end_step)]
            global_row = geometry_by_step.loc[int(interval.end_step)]
            prefix = str(edge)
            future = [
                event for event in valid_events
                if event["patch_id"] == patch_id and event["edge"] == edge
                and time < float(event["anchor_time_ps"]) <= time + horizon
            ]
            record = {
                "case_id": frame.case_id, "branch_id": frame.branch_id,
                "direction": frame.direction, "patch_id": patch_id,
                "patch_index": int(frame.patch_index), "edge": edge,
                "step": int(interval.end_step), "time_ps": time,
                "block_id": math.floor(time / float(raw["block_ps"])),
                "outcome_event": int(bool(future)),
                "outcome_event_id": future[0].get("event_id", "") if future else "",
                "global_water_velocity_mps": float(global_row[f"{prefix}_water_drive_velocity_mps"]),
                "global_oxygen_density_A3": float(global_row[f"{prefix}_oxygen_density_A3"]),
                "bubble_height_A": float(global_row.bubble_height_A),
                "bubble_drive_width_A": float(global_row.bubble_drive_width_A),
                "global_film_water_oxygen_count": float(global_row.film_water_oxygen_count),
                "recent_edge_velocity_A_ps": float(interval.edge_displacement_A) / float(interval.interval_ps),
                "water_oxygen_count": float(frame.water_oxygen_count),
                "exposed_ch3_fraction": float(frame.exposed_ch3_fraction),
                "mean_local_boundary_edge_fraction": float(frame.mean_local_boundary_edge_fraction),
                "turnover_history_mean": _recent_mean(recent, "gross_membership_turnover"),
                "rewiring_loss_history_mean": (
                    1.0 - _recent_mean(recent, "retained_node_edge_survival_fraction")
                ),
                "dipole_rotation_history_deg": _recent_mean(
                    recent, "retained_dipole_mean_rotation_deg"
                ),
                "hbond_break_rate_history_per_ps": _recent_mean(
                    recent, "hbond_break_rate_per_ps"
                ),
                "hbond_formation_rate_history_per_ps": _recent_mean(
                    recent, "hbond_formation_rate_per_ps"
                ),
                "raw_force_impulse_abs_history": _recent_mean(
                    recent, "local_waterO_raw_force_impulse_eV_ps_per_A", absolute=True
                ),
                "water_hbond_degree": float(frame.water_water_hbond_degree),
                "surface_water_hbond_per_water": float(frame.surface_water_hbond_per_intact_water),
            }
            if all(math.isfinite(float(record[field])) for field in M0_FEATURES) and any(
                math.isfinite(float(record[field])) for field in m1_features[len(M0_FEATURES):]
            ):
                output.append(record)
    return output


def _validation_groups(risk: pd.DataFrame) -> dict[str, pd.Series]:
    """Return case-aware groups that never pool same-named branches across surfaces."""

    return {
        "held_block": (
            risk.case_id.astype(str) + ":" + risk.branch_id.astype(str)
            + ":" + risk.block_id.astype(str)
        ),
        "held_branch": risk.case_id.astype(str) + ":" + risk.branch_id.astype(str),
        "held_patch": (
            risk.case_id.astype(str) + ":" + risk.branch_id.astype(str)
            + ":" + risk.patch_id.astype(str)
        ),
    }


def _evaluate_models(
    risk: pd.DataFrame, m1_features: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    schemes = _validation_groups(risk)
    prediction_rows, score_rows = [], []
    y = risk.outcome_event.to_numpy(int)
    for scheme, groups in schemes.items():
        for held_group in sorted(groups.unique()):
            train = groups != held_group; test = groups == held_group
            if len(np.unique(y[train])) < 2 or not np.any(test):
                continue
            for model_name, features in (("M0", M0_FEATURES), ("M1", m1_features)):
                model = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
                )
                model.fit(risk.loc[train, list(features)], y[train])
                probability = model.predict_proba(risk.loc[test, list(features)])[:, 1]
                for index, value in zip(risk.index[test], probability):
                    prediction_rows.append({
                        "scheme": scheme, "held_group": held_group,
                        "model": model_name, "risk_row": int(index),
                        "outcome_event": int(risk.loc[index, "outcome_event"]),
                        "probability": float(value),
                    })
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        return predictions, pd.DataFrame()
    for (scheme, model), group in predictions.groupby(["scheme", "model"]):
        outcome = group.outcome_event.to_numpy(int)
        probability = np.clip(group.probability.to_numpy(float), 1.0e-8, 1.0 - 1.0e-8)
        score_rows.append({
            "scheme": scheme, "model": model, "samples": len(group),
            "events": int(np.sum(outcome)),
            "log_loss": float(log_loss(outcome, probability, labels=[0, 1])),
            "brier_score": float(brier_score_loss(outcome, probability)),
            "average_precision": (
                float(average_precision_score(outcome, probability))
                if np.any(outcome) else math.nan
            ),
        })
    scores = pd.DataFrame(score_rows)
    comparisons = []
    for scheme in sorted(scores.scheme.unique()):
        selected = scores[scores.scheme == scheme].set_index("model")
        if {"M0", "M1"}.issubset(selected.index):
            comparisons.append({
                "scheme": scheme, "model": "M1_minus_M0",
                "samples": int(selected.loc["M1", "samples"]),
                "events": int(selected.loc["M1", "events"]),
                "log_loss": float(selected.loc["M1", "log_loss"] - selected.loc["M0", "log_loss"]),
                "brier_score": float(selected.loc["M1", "brier_score"] - selected.loc["M0", "brier_score"]),
                "average_precision": float(selected.loc["M1", "average_precision"] - selected.loc["M0", "average_precision"]),
            })
    return predictions, pd.concat([scores, pd.DataFrame(comparisons)], ignore_index=True)


def _event_aligned(intervals: pd.DataFrame, events: Sequence[dict[str, object]], window_ps: float) -> pd.DataFrame:
    rows = []
    for event in events:
        selected = intervals[
            (intervals.patch_id == event["patch_id"])
            & (intervals.edge == event["edge"])
            & (intervals.end_time_ps.astype(float) >= float(event["anchor_time_ps"]) - window_ps)
            & (intervals.end_time_ps.astype(float) <= float(event["anchor_time_ps"]) + window_ps)
        ]
        for interval in selected.itertuples(index=False):
            rows.append({
                "event_id": event["event_id"], "case_id": event["case_id"],
                "branch_id": event["branch_id"], "direction": event["direction"],
                "patch_id": event["patch_id"], "edge": event["edge"],
                "process": event["process"],
                "validity_status": event["validity_status"],
                "lag_ps": float(interval.end_time_ps) - float(event["anchor_time_ps"]),
                "gross_membership_turnover": interval.gross_membership_turnover,
                "node_loss_fraction": 1.0 - float(interval.node_survival_fraction),
                "retained_edge_loss_fraction": (
                    1.0 - float(interval.retained_node_edge_survival_fraction)
                    if math.isfinite(float(interval.retained_node_edge_survival_fraction)) else math.nan
                ),
                "surface_water_hbond_change": interval.surface_water_hbond_change,
                "hbond_break_rate_per_ps": getattr(
                    interval, "hbond_break_rate_per_ps", math.nan
                ),
                "hbond_formation_rate_per_ps": getattr(
                    interval, "hbond_formation_rate_per_ps", math.nan
                ),
                "retained_dipole_mean_rotation_deg": getattr(
                    interval, "retained_dipole_mean_rotation_deg", math.nan
                ),
                "local_waterO_raw_force_impulse_eV_ps_per_A": getattr(
                    interval, "local_waterO_raw_force_impulse_eV_ps_per_A", math.nan
                ),
            })
    return pd.DataFrame(rows)


def _plot(events: pd.DataFrame, scores: pd.DataFrame, aligned: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    if not events.empty:
        counts = events[events.validity_status == "VALID"].groupby(["case_id", "process"]).size().unstack(fill_value=0)
        counts.plot.bar(ax=axes[0], color=["#5B8FF9", "#E8684A"][: len(counts.columns)])
    axes[0].set_title("a  Persistent patch events"); axes[0].set_ylabel("Count")
    selected = scores[scores.model.isin(["M0", "M1"])] if not scores.empty else scores
    if not selected.empty:
        pivot = selected.pivot(index="scheme", columns="model", values="log_loss")
        pivot.plot.bar(ax=axes[1], color=["#8C8C8C", "#54A24B"])
    axes[1].set_title("b  Held-out prediction"); axes[1].set_ylabel("Log loss")
    if not aligned.empty:
        profile = aligned.groupby("lag_ps")[["node_loss_fraction", "retained_edge_loss_fraction"]].mean()
        axes[2].plot(profile.index, profile.node_loss_fraction, label="node renewal")
        axes[2].plot(profile.index, profile.retained_edge_loss_fraction, label="retained-edge rewiring")
    axes[2].axvline(0.0, color="black", lw=0.7); axes[2].legend(frameon=False)
    axes[2].set_title("c  Event-aligned water"); axes[2].set_xlabel("Lag (ps)")
    axes[2].set_ylabel("Loss fraction")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output / "mechanism_overview.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve(); output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    base = contract_path.parent
    m1_features = _m1_features(raw)
    all_events, all_calibrations, all_sensitivity = [], [], []
    all_risk, all_aligned, all_sites = [], [], []
    input_rows = [{"role": "contract", "path": str(contract_path), "sha256": sha256(contract_path)}]
    event_index = 1
    for case in cases:
        paths = {key: resolve_path(case[key], base) for key in (
            "driven_patch_frames", "driven_patch_intervals", "driven_site_exposure",
            "driven_geometry_frames", "baseline_patch_frames",
        )}
        for role, path in paths.items():
            input_rows.append({"role": role, "path": str(path), "sha256": sha256(path)})
        frames = _read(paths["driven_patch_frames"])
        intervals = _read(paths["driven_patch_intervals"])
        sites = _read(paths["driven_site_exposure"])
        geometry = _read(paths["driven_geometry_frames"])
        baseline = _read(paths["baseline_patch_frames"])
        events, calibrations, sensitivity = _detect_patch_events(frames, baseline, geometry, raw)
        qualified = _qualified_patch_keys(calibrations)
        for event in events:
            event["event_id"] = f"PE{event_index:05d}"; event_index += 1
        all_events.extend(events); all_calibrations.extend(calibrations); all_sensitivity.extend(sensitivity)
        qualified_frames = _filter_patch_edges(frames, qualified)
        qualified_intervals = _filter_patch_edges(intervals, qualified)
        qualified_sites = _filter_patch_edges(sites, qualified)
        risk = _risk_rows(
            qualified_frames, qualified_intervals, geometry, events, raw, m1_features
        )
        all_risk.extend(risk)
        aligned = _event_aligned(
            qualified_intervals, events, float(raw["event_alignment_window_ps"])
        )
        if not aligned.empty:
            all_aligned.append(aligned)
        qualified_sites["case_id"] = frames.case_id.iloc[0]
        qualified_sites["branch_id"] = frames.branch_id.iloc[0]
        qualified_sites["direction"] = frames.direction.iloc[0]
        all_sites.append(qualified_sites)
    event_table = pd.DataFrame(all_events)
    risk_table = pd.DataFrame(all_risk)
    aligned_table = pd.concat(all_aligned, ignore_index=True) if all_aligned else pd.DataFrame()
    site_table = pd.concat(all_sites, ignore_index=True)
    event_site_counts = (
        event_table[event_table.validity_status == "VALID"].groupby(
            ["case_id", "branch_id", "direction", "patch_id", "edge", "nearest_site_id"]
        ).size().rename("nearest_site_event_count").reset_index()
        if not event_table.empty else pd.DataFrame()
    )
    motif_response = site_table.copy()
    if not event_site_counts.empty:
        event_site_counts["nearest_site_id"] = pd.to_numeric(
            event_site_counts.nearest_site_id, errors="coerce"
        )
        event_site_counts = event_site_counts.dropna(subset=["nearest_site_id"])
        event_site_counts["nearest_site_id"] = event_site_counts.nearest_site_id.astype(int)
        motif_response = motif_response.merge(
            event_site_counts,
            left_on=["case_id", "branch_id", "direction", "patch_id", "edge", "site_id"],
            right_on=["case_id", "branch_id", "direction", "patch_id", "edge", "nearest_site_id"],
            how="left",
        )
    motif_response["nearest_site_event_count"] = motif_response.get(
        "nearest_site_event_count", pd.Series(0, index=motif_response.index)
    ).fillna(0).astype(int)
    motif_response["events_per_1000_exposure_frames"] = np.divide(
        1000.0 * motif_response.nearest_site_event_count,
        motif_response.exposure_frames,
        out=np.full(len(motif_response), np.nan),
        where=motif_response.exposure_frames.to_numpy(float) > 0,
    )
    predictions, scores = (
        _evaluate_models(risk_table, m1_features)
        if not risk_table.empty and risk_table.outcome_event.nunique() == 2
        else (pd.DataFrame(), pd.DataFrame())
    )
    output.mkdir(parents=True)
    pd.DataFrame(all_calibrations).to_csv(output / "PATCH-EVENT-CALIBRATION.tsv", sep="\t", index=False)
    event_table.to_csv(output / "PATCH-EDGE-EVENTS.tsv", sep="\t", index=False)
    pd.DataFrame(all_sensitivity).to_csv(output / "PATCH-EVENT-SENSITIVITY.tsv", sep="\t", index=False)
    risk_table.to_csv(output / "WATER-EDGE-JOINT.tsv", sep="\t", index=False)
    site_table.to_csv(output / "SITE-MOTIF-LEDGER.tsv", sep="\t", index=False)
    motif_response.to_csv(output / "EXPOSED-MOTIF-RESPONSE.tsv", sep="\t", index=False)
    aligned_table.to_csv(output / "EVENT-ALIGNED-WATER.tsv", sep="\t", index=False)
    predictions.to_csv(output / "HELDOUT-PREDICTIONS.tsv", sep="\t", index=False)
    scores.to_csv(output / "MODEL-SCORES.tsv", sep="\t", index=False)
    pd.DataFrame(input_rows).to_csv(output / "input_manifest.tsv", sep="\t", index=False)
    valid_events = int(np.sum(event_table.validity_status == "VALID")) if not event_table.empty else 0
    comparison = scores[scores.model == "M1_minus_M0"] if not scores.empty else scores
    qualifies = (
        valid_events >= int(raw["minimum_events_for_model_gate"])
        and len(comparison) == 3
        and bool(np.all(comparison.log_loss.to_numpy(float) < 0.0))
    )
    decision = "ADDITIONAL_HYDRATION_SIGNAL" if qualifies else "NO_ADDITIONAL_HYDRATION_SIGNAL"
    summary = {
        "status": "PASS", "branches": len(cases), "events": len(event_table),
        "valid_events": valid_events, "risk_rows": len(risk_table),
        "qualified_patch_edges": sum(
            str(row["calibration_status"]) == "PASS" for row in all_calibrations
        ),
        "held_patch_edges": sum(
            str(row["calibration_status"]) != "PASS" for row in all_calibrations
        ),
        "observer_qualification_policy": (
            "Event, risk, aligned-water, and site-exposure tables use only patch-edge "
            "observers that pass the frozen coverage and null-anchor contract."
        ),
        "prediction_decision": decision,
        "water_feature_mode": str(raw.get("water_feature_mode", "turnover")),
        "m1_features": list(m1_features),
        "model_scope": "provisional held-block, held-branch, and held-patch prediction",
        "claim_boundary": (
            "All uncertainty is within the available trajectories; temporal precedence and "
            "held-out prediction are not causal mediation or replicate uncertainty."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "HELDOUT-PREDICTION-REPORT.md").write_text(
        "# Held-out prediction report\n\n"
        f"Decision: `{decision}`. M0 contains local flow, geometry, density, and motif "
        "features. M1 adds the frozen pre-event water-reorganization feature set recorded "
        f"in `{raw.get('water_feature_mode', 'turnover')}` mode. "
        "Evaluation holds out complete time blocks, branches, or transverse patches; no "
        "random-frame split is used. This is single-trajectory exploratory evidence and "
        "does not establish causality or replicate uncertainty.\n",
        encoding="utf-8",
    )
    _plot(event_table, scores, aligned_table, output)
    write_output_hashes(output)
    return summary
