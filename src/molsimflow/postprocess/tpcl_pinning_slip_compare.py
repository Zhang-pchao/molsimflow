"""Cross-case comparison for TPCL dwell--jump candidate analyses."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.postprocess.tpcl_pinning_slip import (
    _configure_matplotlib,
    _save_figure,
)

LOCAL_FLOAT_FIELDS = (
    "local_ch3_fraction",
    "chemical_boundary_proxy_A",
    "local_hydration_areal_density_A-2",
    "local_water_water_hbond_degree",
    "local_surface_water_hbond_per_h2o",
    "local_n2_contact_count",
    "local_contact_angle_deg",
    "local_curvature_A-1",
    "local_residual_A",
    "local_normal_velocity_A_per_ps",
)
TRACE_METRICS = (
    "local_hydration_areal_density_A-2",
    "local_water_water_hbond_degree",
    "local_surface_water_hbond_per_h2o",
    "local_n2_contact_count",
    "local_contact_angle_deg",
    "local_curvature_A-1",
    "local_residual_A",
)
BLOCK_METRICS = (
    "local_hydration_areal_density_A-2",
    "local_water_water_hbond_degree",
    "local_surface_water_hbond_per_h2o",
    "local_n2_contact_count",
    "local_contact_angle_deg",
    "local_curvature_A-1",
    "local_residual_A",
    "local_normal_velocity_A_per_ps",
)
FAIR_WINDOW_MIN_CONTOUR_COVERAGE = 0.95


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kind: str
    output_root: Path
    ch3_sites: int
    oh_sites: int
    attachment_time_ns: float | None

    @property
    def ch3_fraction(self) -> float:
        return self.ch3_sites / (self.ch3_sites + self.oh_sites)


@dataclass
class LocalData:
    step: np.ndarray
    time_ns: np.ndarray
    arc: np.ndarray
    segment: np.ndarray
    nearest_ch3: np.ndarray
    values: dict[str, np.ndarray]


@dataclass
class CaseData:
    spec: CaseSpec
    job_id: str
    results_root: Path
    summary: dict
    frames: list[dict]
    local: LocalData
    events: list[dict]


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int(value: object) -> int:
    return int(float(value))


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _mode(values: Iterable[str]) -> str:
    counts = Counter(values)
    return counts.most_common(1)[0][0] if counts else ""


def read_case_manifest(path: Path) -> list[CaseSpec]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("Case manifest is empty")
    required = {"case_id", "kind", "output_root", "ch3_sites", "oh_sites"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Case manifest lacks columns: {sorted(missing)}")
    cases = []
    for row in rows:
        attachment = row.get("attachment_time_ns", "")
        cases.append(
            CaseSpec(
                case_id=row["case_id"],
                kind=row["kind"],
                output_root=Path(row["output_root"]),
                ch3_sites=int(row["ch3_sites"]),
                oh_sites=int(row["oh_sites"]),
                attachment_time_ns=None
                if attachment in {"", "NA", "nan"}
                else float(attachment),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Case IDs are not unique")
    return cases


def _read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_local(path: Path) -> LocalData:
    columns: dict[str, list] = {
        "step": [],
        "time_ns": [],
        "arc": [],
        "segment": [],
        "nearest_ch3": [],
        **{name: [] for name in LOCAL_FLOAT_FIELDS},
    }
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            columns["step"].append(_int(row["step"]))
            columns["time_ns"].append(_float(row["time_ns"]))
            columns["arc"].append(_int(row["arc_index"]))
            columns["segment"].append(_int(row["segment_index"]))
            columns["nearest_ch3"].append(1.0 if row["nearest_site_type"] == "CH3" else 0.0)
            for name in LOCAL_FLOAT_FIELDS:
                columns[name].append(_float(row[name]))
    return LocalData(
        step=np.asarray(columns["step"], dtype=np.int64),
        time_ns=np.asarray(columns["time_ns"], dtype=float),
        arc=np.asarray(columns["arc"], dtype=np.int16),
        segment=np.asarray(columns["segment"], dtype=np.int16),
        nearest_ch3=np.asarray(columns["nearest_ch3"], dtype=float),
        values={name: np.asarray(columns[name], dtype=float) for name in LOCAL_FLOAT_FIELDS},
    )


def load_case(spec: CaseSpec) -> CaseData:
    latest = spec.output_root / "latest"
    if not latest.is_symlink():
        raise ValueError(f"{spec.case_id}: latest is not a symlink")
    run_root = latest.resolve(strict=True)
    if run_root.parent.name != "run":
        raise ValueError(f"{spec.case_id}: latest does not resolve below run/")
    result_record = run_root / "RUN-RESULT.txt"
    if "status=PASS" not in result_record.read_text(encoding="utf-8"):
        raise ValueError(f"{spec.case_id}: upstream run is not PASS")
    results = run_root / "results"
    required = (
        "summary.json",
        "frame_metrics.csv",
        "local_arc_metrics.csv.gz",
        "candidate_events.csv",
    )
    for name in required:
        if not (results / name).is_file():
            raise ValueError(f"{spec.case_id}: missing {name}")
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError(f"{spec.case_id}: upstream summary is not PASS")
    return CaseData(
        spec=spec,
        job_id=run_root.name,
        results_root=results,
        summary=summary,
        frames=_read_csv(results / "frame_metrics.csv"),
        local=_load_local(results / "local_arc_metrics.csv.gz"),
        events=_read_csv(results / "candidate_events.csv"),
    )


def candidate_event_clusters(events: Sequence[Mapping[str, str]]) -> list[list[Mapping[str, str]]]:
    clusters: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for event in events:
        if event.get("quality_status") == "candidate_stick_slip":
            clusters[event["event_cluster_id"]].append(event)
    return [clusters[key] for key in sorted(clusters, key=int)]


def physically_admissible_event_clusters(
    events: Sequence[Mapping[str, str]],
) -> list[list[Mapping[str, str]]]:
    """Return motion-qualified clusters before the chemistry repetition label is applied."""
    clusters: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for event in events:
        if event.get("quality_status") in {"candidate_stick_slip", "insufficient_repetition"}:
            clusters[event["event_cluster_id"]].append(event)
    return [clusters[key] for key in sorted(clusters, key=int)]


def _circular_mean_degrees(values: Sequence[float]) -> float:
    radians = np.radians(values)
    angle = math.degrees(math.atan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return angle % 360.0


def cluster_summary(case: CaseData) -> list[dict]:
    rows = []
    delta_fields = (
        "local_hydration_areal_density_A-2",
        "local_water_water_hbond_degree",
        "local_surface_water_hbond_per_h2o",
        "local_n2_contact_count",
        "local_contact_angle_deg",
        "local_curvature_A-1",
    )
    for cluster in candidate_event_clusters(case.events):
        pre_site_type = _mode(item["pre_nearest_site_type"] for item in cluster)
        post_site_type = _mode(item["post_nearest_site_type"] for item in cluster)
        row = {
            "case_id": case.spec.case_id,
            "kind": case.spec.kind,
            "ch3_fraction": case.spec.ch3_fraction,
            "event_cluster_id": cluster[0]["event_cluster_id"],
            "event_arc_record_count": len(cluster),
            "transition_time_ns": float(np.median([_float(item["transition_time_ns"]) for item in cluster])),
            "theta_deg": _circular_mean_degrees([_float(item["theta_deg"]) for item in cluster]),
            "mechanism_class": _mode(item["cluster_mechanism_class"] for item in cluster),
            "pre_nearest_site_type": pre_site_type,
            "post_nearest_site_type": post_site_type,
            "site_type_transition": f"{pre_site_type}->{post_site_type}",
            "surface_site_id_change_arc_fraction": float(
                np.mean(
                    [
                        item["pre_nearest_site_id"] != item["post_nearest_site_id"]
                        for item in cluster
                    ]
                )
            ),
            "mean_dwell_site_stability_fraction": float(
                np.mean([_float(item["dwell_site_stability_fraction"]) for item in cluster])
            ),
            "mean_jump_distance_A": float(
                np.mean([_float(item["jump_distance_A"]) for item in cluster])
            ),
            "mean_signed_jump_A": float(
                np.mean([_float(item["jump_signed_A"]) for item in cluster])
            ),
            "mean_dwell_time_ps": float(
                np.mean([_float(item["dwell_time_ps"]) for item in cluster])
            ),
        }
        for field in delta_fields:
            values = [_float(item[f"delta_{field}"]) for item in cluster]
            row[f"mean_delta_{field}"] = _nanmean(values)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event_count_in_window(clusters: Sequence[Mapping[str, object]], start: float, end: float) -> int:
    return sum(start <= float(row["transition_time_ns"]) <= end for row in clusters)


def _valid_times(case: CaseData) -> np.ndarray:
    return np.unique(case.local.time_ns)


def contour_validity_intervals(case: CaseData) -> list[dict]:
    rows = sorted(case.frames, key=lambda row: _int(row["step"]))
    expected_step = int(case.summary["expected_step_interval"])
    interval_ps = float(case.summary["frame_interval_ps"])
    groups: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        valid = row["contour_valid"] == "True"
        if current:
            previous = current[-1]
            previous_valid = previous["contour_valid"] == "True"
            consecutive = _int(row["step"]) - _int(previous["step"]) == expected_step
            if valid != previous_valid or not consecutive:
                groups.append(current)
                current = []
        current.append(row)
    if current:
        groups.append(current)
    output = []
    for index, group in enumerate(groups, start=1):
        valid = group[0]["contour_valid"] == "True"
        reasons = [row["contour_invalid_reason"] for row in group if row["contour_invalid_reason"]]
        output.append(
            {
                "case_id": case.spec.case_id,
                "kind": case.spec.kind,
                "interval_id": index,
                "contour_valid": valid,
                "start_step": _int(group[0]["step"]),
                "end_step": _int(group[-1]["step"]),
                "start_time_ns": _float(group[0]["time_ns"]),
                "end_time_ns": _float(group[-1]["time_ns"]),
                "frames": len(group),
                "sampled_duration_ns": len(group) * interval_ps / 1000.0,
                "dominant_invalid_reason": "" if valid else _mode(reasons),
            }
        )
    return output


def build_case_summary(cases: Sequence[CaseData], cluster_rows: Sequence[dict]) -> list[dict]:
    clusters_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in cluster_rows:
        clusters_by_case[row["case_id"]].append(row)
    droplets = [case for case in cases if case.spec.kind == "nanodroplet"]
    droplet_start = max(float(np.min(_valid_times(case))) for case in droplets)
    droplet_end = min(float(np.max(_valid_times(case))) for case in droplets)
    rows = []
    for case in cases:
        valid = _valid_times(case)
        interval_ns = float(case.summary["frame_interval_ps"]) / 1000.0
        clusters = clusters_by_case[case.spec.case_id]
        if case.spec.kind == "nanobubble":
            if case.spec.attachment_time_ns is None:
                raise ValueError(f"{case.spec.case_id}: bubble attachment time is missing")
            fair_start = case.spec.attachment_time_ns
            fair_end = fair_start + 3.5
            fair_name = "attachment_relative_0_3.5_ns"
        else:
            fair_start, fair_end = droplet_start, droplet_end
            fair_name = "common_valid_window"
        all_times = np.asarray([_float(row["time_ns"]) for row in case.frames])
        fair_total_frames = int(np.sum((all_times >= fair_start) & (all_times <= fair_end)))
        fair_frames = int(np.sum((valid >= fair_start) & (valid <= fair_end)))
        fair_coverage = fair_frames / fair_total_frames if fair_total_frames else math.nan
        fair_duration = fair_frames * interval_ns
        fair_events = _event_count_in_window(clusters, fair_start, fair_end)
        status_counts = Counter(row.get("quality_status", "") for row in case.events)
        mechanism_counts = Counter(str(row["mechanism_class"]) for row in clusters)
        intervals = contour_validity_intervals(case)
        valid_intervals = [row for row in intervals if row["contour_valid"]]
        final_interval = intervals[-1]
        rows.append(
            {
                "case_id": case.spec.case_id,
                "kind": case.spec.kind,
                "job_id": case.job_id,
                "ch3_fraction": case.spec.ch3_fraction,
                "first_valid_time_ns": float(np.min(valid)),
                "last_valid_time_ns": float(np.max(valid)),
                "valid_contour_frames": int(case.summary["valid_contour_frames"]),
                "contour_valid_fraction": float(case.summary["contour_valid_fraction"]),
                "contour_valid_fraction_after_first_valid": float(
                    case.summary["contour_valid_fraction_after_first_valid"]
                ),
                "localization_noise_A": float(case.summary["localization_noise_A"]),
                "jump_threshold_A": float(case.summary["jump_threshold_A"]),
                "candidate_event_clusters": len(clusters),
                "candidate_arc_records": int(case.summary["candidate_arc_record_count"]),
                "insufficient_repetition_arc_records": status_counts["insufficient_repetition"],
                "rejected_arc_records": status_counts["rejected"],
                "boundary_event_clusters": mechanism_counts["boundary"],
                "ch3_event_clusters": mechanism_counts["CH3"],
                "sioh_event_clusters": mechanism_counts["SiOH"],
                "valid_contour_interval_count": len(valid_intervals),
                "longest_valid_contour_interval_ns": max(
                    row["sampled_duration_ns"] for row in valid_intervals
                ),
                "persistent_valid_contour_start_ns": final_interval["start_time_ns"]
                if final_interval["contour_valid"]
                else math.nan,
                "fair_window": fair_name,
                "fair_start_ns": fair_start,
                "fair_end_ns": fair_end,
                "fair_total_frames": fair_total_frames,
                "fair_valid_frames": fair_frames,
                "fair_contour_coverage_fraction": fair_coverage,
                "fair_window_comparison_status": "ADMITTED"
                if fair_coverage >= FAIR_WINDOW_MIN_CONTOUR_COVERAGE
                else "INSUFFICIENT_CONTOUR_COVERAGE",
                "fair_valid_duration_ns": fair_duration,
                "fair_event_clusters": fair_events,
                "fair_event_rate_per_valid_ns": fair_events / fair_duration
                if fair_duration
                else math.nan,
            }
        )
    return rows


def frame_local_means(local: LocalData, fields: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    steps, inverse = np.unique(local.step, return_inverse=True)
    means = {}
    for field in fields:
        values = local.values[field]
        finite = np.isfinite(values)
        sums = np.bincount(inverse[finite], weights=values[finite], minlength=len(steps))
        counts = np.bincount(inverse[finite], minlength=len(steps))
        means[field] = np.divide(
            sums,
            counts,
            out=np.full(len(steps), np.nan),
            where=counts > 0,
        )
    return steps, means


def build_block_rows(case: CaseData, block_ps: float) -> list[dict]:
    steps, means = frame_local_means(case.local, BLOCK_METRICS)
    step_time = {
        int(step): float(time)
        for step, time in zip(case.local.step, case.local.time_ns)
    }
    times = np.asarray([step_time[int(step)] for step in steps])
    block_ns = block_ps / 1000.0
    blocks = np.floor((times + 1.0e-12) / block_ns).astype(int)
    cluster_times = [row["transition_time_ns"] for row in cluster_summary(case)]
    rows = []
    for block in np.unique(blocks):
        mask = blocks == block
        start, end = block * block_ns, (block + 1) * block_ns
        row = {
            "case_id": case.spec.case_id,
            "kind": case.spec.kind,
            "ch3_fraction": case.spec.ch3_fraction,
            "block_start_ns": start,
            "block_end_ns": end,
            "valid_frames": int(np.sum(mask)),
            "candidate_event_clusters": sum(start <= value < end for value in cluster_times),
        }
        for field in BLOCK_METRICS:
            row[f"mean_{field}"] = _nanmean(means[field][mask])
        rows.append(row)
    return rows


def bootstrap_block_summary(
    case_rows: Sequence[dict],
    fair_start_ns: float,
    fair_end_ns: float,
    rng: np.random.Generator,
    replicates: int,
) -> list[dict]:
    selected = [
        row
        for row in case_rows
        if row["block_start_ns"] >= fair_start_ns - 1.0e-9
        and row["block_end_ns"] <= fair_end_ns + 1.0e-9
    ]
    if not selected:
        return []
    output = []
    for field in BLOCK_METRICS:
        name = f"mean_{field}"
        values = np.asarray([row[name] for row in selected], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
        output.append(
            {
                "metric": field,
                "blocks": len(values),
                "mean": float(np.mean(values)),
                "bootstrap_ci025": float(np.quantile(samples, 0.025)),
                "bootstrap_ci975": float(np.quantile(samples, 0.975)),
            }
        )
    return output


def _local_index(local: LocalData) -> dict[tuple[int, int], int]:
    return {
        (int(step), int(arc)): index
        for index, (step, arc) in enumerate(zip(local.step, local.arc))
    }


def build_event_aligned_rows(case: CaseData, half_window_ps: float) -> list[dict]:
    expected_step = int(case.summary["expected_step_interval"])
    interval_ps = float(case.summary["frame_interval_ps"])
    half_frames = round(half_window_ps / interval_ps)
    index = _local_index(case.local)
    output = []
    for cluster in candidate_event_clusters(case.events):
        cluster_id = cluster[0]["event_cluster_id"]
        for lag_index in range(-half_frames, half_frames + 1):
            lag_ps = lag_index * interval_ps
            metric_values: dict[str, list[float]] = {field: [] for field in TRACE_METRICS}
            used_records = 0
            for event in cluster:
                arc = _int(event["arc_index"])
                transition = _int(event["transition_step"])
                target = transition + lag_index * expected_step
                transition_index = index.get((transition, arc))
                target_index = index.get((target, arc))
                if transition_index is None or target_index is None:
                    continue
                lower, upper = sorted((transition, target))
                path = range(lower, upper + expected_step, expected_step)
                if any((step, arc) not in index for step in path):
                    continue
                segment = case.local.segment[transition_index]
                if any(case.local.segment[index[(step, arc)]] != segment for step in path):
                    continue
                used_records += 1
                for field in TRACE_METRICS:
                    metric_values[field].append(case.local.values[field][target_index])
            if not used_records:
                continue
            row = {
                "case_id": case.spec.case_id,
                "kind": case.spec.kind,
                "ch3_fraction": case.spec.ch3_fraction,
                "event_cluster_id": cluster_id,
                "lag_ps": lag_ps,
                "arc_records": used_records,
            }
            for field, values in metric_values.items():
                row[field] = _nanmean(values)
            output.append(row)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in output:
        grouped[row["event_cluster_id"]].append(row)
    baseline_fields: dict[str, dict[str, float]] = {}
    for cluster_id, rows in grouped.items():
        baseline = [row for row in rows if -40.0 <= row["lag_ps"] < 0.0]
        baseline_fields[cluster_id] = {
            field: _nanmean(row[field] for row in baseline) if baseline else math.nan
            for field in TRACE_METRICS
        }
    for row in output:
        for field in TRACE_METRICS:
            row[f"delta_{field}"] = row[field] - baseline_fields[row["event_cluster_id"]][field]
    return output


def summarize_event_aligned(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    for row in rows:
        for field in TRACE_METRICS:
            key = (row["case_id"], row["lag_ps"], field)
            groups[key].append(row[f"delta_{field}"])
            meta[key] = row
    output = []
    for key, values in groups.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            continue
        row = meta[key]
        output.append(
            {
                "case_id": row["case_id"],
                "kind": row["kind"],
                "ch3_fraction": row["ch3_fraction"],
                "lag_ps": row["lag_ps"],
                "metric": key[2],
                "event_clusters": len(finite),
                "mean_delta": float(np.mean(finite)),
                "median_delta": float(np.median(finite)),
                "minimum_delta": float(np.min(finite)),
                "maximum_delta": float(np.max(finite)),
            }
        )
    return output


def circular_shift_null(case: CaseData, boundary_A: float = 2.0) -> list[dict]:
    clusters = physically_admissible_event_clusters(case.events)
    if not clusters or not (case.spec.ch3_sites and case.spec.oh_sites):
        return []
    index = _local_index(case.local)
    expected_step = int(case.summary["expected_step_interval"])
    arc_bins = int(case.summary["arc_bins"])
    rows = []
    for shift in range(arc_bins):
        cluster_values = []
        for cluster in clusters:
            boundary_values, ch3_values, pre_nearest_values, post_nearest_values = [], [], [], []
            environment_values = {field: [] for field in TRACE_METRICS[:4]}
            for event in cluster:
                shifted_arc = (_int(event["arc_index"]) + shift) % arc_bins
                start, end = _int(event["start_step"]), _int(event["end_step"])
                for step in range(start, end + expected_step, expected_step):
                    local_index = index.get((step, shifted_arc))
                    if local_index is None:
                        continue
                    boundary_values.append(
                        case.local.values["chemical_boundary_proxy_A"][local_index]
                    )
                    ch3_values.append(case.local.values["local_ch3_fraction"][local_index])
                    pre_nearest_values.append(case.local.nearest_ch3[local_index])
                    for field, collected in environment_values.items():
                        collected.append(case.local.values[field][local_index])
                transition = _int(event["transition_step"])
                post_end = _int(event["post_end_step"])
                for step in range(transition, post_end + expected_step, expected_step):
                    local_index = index.get((step, shifted_arc))
                    if local_index is not None:
                        post_nearest_values.append(case.local.nearest_ch3[local_index])
            if boundary_values:
                pre_nearest = _nanmean(pre_nearest_values)
                post_nearest = _nanmean(post_nearest_values)
                cluster_values.append(
                    (
                        _nanmean(boundary_values),
                        _nanmean(ch3_values),
                        pre_nearest,
                        float((pre_nearest >= 0.5) != (post_nearest >= 0.5)),
                        *(_nanmean(environment_values[field]) for field in environment_values),
                    )
                )
        if not cluster_values:
            continue
        values = np.asarray(cluster_values)
        rows.append(
            {
                "case_id": case.spec.case_id,
                "kind": case.spec.kind,
                "shift_bins": shift,
                "shift_degrees": 360.0 * shift / arc_bins,
                "is_observed": shift == 0,
                "event_clusters": len(values),
                "boundary_event_fraction": float(np.mean(values[:, 0] <= boundary_A)),
                "mean_boundary_proxy_A": _nanmean(values[:, 0]),
                "mean_local_ch3_fraction": _nanmean(values[:, 1]),
                "nearest_ch3_fraction": _nanmean(values[:, 2]),
                "site_transition_event_fraction": _nanmean(values[:, 3]),
                "mean_pre_hydration_areal_density_A-2": _nanmean(values[:, 4]),
                "mean_pre_water_water_hbond_degree": _nanmean(values[:, 5]),
                "mean_pre_surface_water_hbond_per_h2o": _nanmean(values[:, 6]),
                "mean_pre_n2_contact_count": _nanmean(values[:, 7]),
            }
        )
    return rows


def summarize_null(rows: Sequence[dict]) -> list[dict]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    directions = {
        "boundary_event_fraction": "high",
        "mean_boundary_proxy_A": "low",
        "mean_local_ch3_fraction": "two_sided",
        "nearest_ch3_fraction": "two_sided",
        "site_transition_event_fraction": "high",
        "mean_pre_hydration_areal_density_A-2": "two_sided",
        "mean_pre_water_water_hbond_degree": "two_sided",
        "mean_pre_surface_water_hbond_per_h2o": "two_sided",
        "mean_pre_n2_contact_count": "two_sided",
    }
    output = []
    for case_id, values in by_case.items():
        observed = next(row for row in values if row["is_observed"])
        null = [row for row in values if not row["is_observed"]]
        row = {
            "case_id": case_id,
            "kind": observed["kind"],
            "event_clusters": observed["event_clusters"],
            "null_shifts": len(null),
        }
        for field, direction in directions.items():
            actual = float(observed[field])
            null_values = np.asarray([item[field] for item in null], dtype=float)
            null_values = null_values[np.isfinite(null_values)]
            if not np.isfinite(actual) or not len(null_values):
                row[f"observed_{field}"] = actual
                row[f"null_median_{field}"] = math.nan
                row[f"null_q025_{field}"] = math.nan
                row[f"null_q975_{field}"] = math.nan
                row[f"empirical_p_{field}"] = math.nan
                continue
            center = float(np.median(null_values))
            if direction == "high":
                extreme = int(np.sum(null_values >= actual))
            elif direction == "low":
                extreme = int(np.sum(null_values <= actual))
            else:
                extreme = int(np.sum(np.abs(null_values - center) >= abs(actual - center)))
            row[f"observed_{field}"] = actual
            row[f"null_median_{field}"] = center
            row[f"null_q025_{field}"] = float(np.quantile(null_values, 0.025))
            row[f"null_q975_{field}"] = float(np.quantile(null_values, 0.975))
            row[f"empirical_p_{field}"] = (extreme + 1) / (len(null_values) + 1)
        output.append(row)
    tests = []
    for row_index, row in enumerate(output):
        for field in directions:
            p_value = float(row[f"empirical_p_{field}"])
            if np.isfinite(p_value):
                tests.append((p_value, row_index, field))
    ranked = sorted(tests)
    adjusted = [math.nan] * len(ranked)
    running = 1.0
    for rank_index in range(len(ranked) - 1, -1, -1):
        p_value = ranked[rank_index][0]
        rank = rank_index + 1
        running = min(running, p_value * len(ranked) / rank)
        adjusted[rank_index] = running
    for (_, row_index, field), q_value in zip(ranked, adjusted):
        output[row_index][f"bh_q_{field}"] = q_value
    return output


def bootstrap_event_deltas(
    cluster_rows: Sequence[dict],
    block_ps: float,
    replicates: int,
    seed: int,
) -> list[dict]:
    fields = [name for name in cluster_rows[0] if name.startswith("mean_delta_")] if cluster_rows else []
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in cluster_rows:
        by_case[row["case_id"]].append(row)
    rng = np.random.default_rng(seed)
    output = []
    block_ns = block_ps / 1000.0
    for case_id, rows in by_case.items():
        for field in fields:
            block_values: dict[int, list[float]] = defaultdict(list)
            cluster_values = []
            for row in rows:
                value = float(row[field])
                if not np.isfinite(value):
                    continue
                cluster_values.append(value)
                block = math.floor(float(row["transition_time_ns"]) / block_ns)
                block_values[block].append(value)
            means = np.asarray([np.mean(values) for values in block_values.values()])
            if not len(means):
                continue
            samples = rng.choice(means, size=(replicates, len(means)), replace=True).mean(axis=1)
            first = rows[0]
            output.append(
                {
                    "case_id": case_id,
                    "kind": first["kind"],
                    "ch3_fraction": first["ch3_fraction"],
                    "metric": field.removeprefix("mean_delta_"),
                    "event_clusters": len(cluster_values),
                    "event_time_blocks": len(means),
                    "mean_block_delta": float(np.mean(means)),
                    "median_cluster_delta": float(np.median(cluster_values)),
                    "positive_cluster_fraction": float(np.mean(np.asarray(cluster_values) > 0.0)),
                    "bootstrap_ci025": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci975": float(np.quantile(samples, 0.975)),
                }
            )
    return output


def threshold_sensitivity(case: CaseData, multipliers: Sequence[float]) -> list[dict]:
    baseline = candidate_event_clusters(case.events)
    threshold = float(case.summary["jump_threshold_A"])
    output = []
    for multiplier in multipliers:
        retained_by_class: dict[str, list[str]] = defaultdict(list)
        arc_records = 0
        for cluster in baseline:
            retained = [
                row for row in cluster if _float(row["jump_distance_A"]) >= multiplier * threshold
            ]
            if retained:
                mechanism = _mode(row["cluster_mechanism_class"] for row in retained)
                retained_by_class[mechanism].append(cluster[0]["event_cluster_id"])
                arc_records += len(retained)
        repeated_clusters = sum(
            len(cluster_ids) for cluster_ids in retained_by_class.values() if len(cluster_ids) >= 2
        )
        output.append(
            {
                "case_id": case.spec.case_id,
                "kind": case.spec.kind,
                "jump_threshold_multiplier": multiplier,
                "jump_threshold_A": multiplier * threshold,
                "retained_arc_records": arc_records,
                "retained_repeated_event_clusters": repeated_clusters,
                "sensitivity_direction": "baseline_or_stricter_only",
            }
        )
    return output


def _case_color(fraction: float):
    from matplotlib import colormaps

    return colormaps["viridis"](fraction)


def write_figures(
    cases: Sequence[CaseData],
    case_rows: Sequence[dict],
    clusters: Sequence[dict],
    block_bootstrap_rows: Sequence[dict],
    event_delta_rows: Sequence[dict],
    aligned_summary: Sequence[dict],
    null_rows: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    output: Path,
    font_path: Path,
) -> None:
    _configure_matplotlib(font_path)
    from matplotlib import pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.5), sharex=True)
    for kind, marker in (("nanobubble", "o"), ("nanodroplet", "s")):
        rows = sorted((row for row in case_rows if row["kind"] == kind), key=lambda row: row["ch3_fraction"])
        x = [row["ch3_fraction"] for row in rows]
        axes[0].plot(
            x,
            [
                row["fair_event_rate_per_valid_ns"]
                if row["fair_window_comparison_status"] == "ADMITTED"
                else math.nan
                for row in rows
            ],
            marker=marker,
            fillstyle="none",
            label=kind,
        )
        axes[1].plot(
            x,
            [row["localization_noise_A"] for row in rows],
            marker=marker,
            fillstyle="none",
            label=kind,
        )
        axes[2].plot(
            x,
            [row["fair_contour_coverage_fraction"] for row in rows],
            marker=marker,
            fillstyle="none",
            label=kind,
        )
    axes[0].set_ylabel(r"Candidate clusters ns$^{-1}$")
    axes[1].set_ylabel(r"Localization noise (Å)")
    axes[2].set_ylabel("Fair-window contour coverage")
    axes[2].set_xlabel(r"CH$_3$ site fraction")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, figures / "01_eight_case_summary")
    plt.close(figure)

    plotted_metrics = TRACE_METRICS[:4]
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True)
    labels = {
        "local_hydration_areal_density_A-2": r"Hydration density change (Å$^{-2}$)",
        "local_water_water_hbond_degree": "Water H-bond degree change",
        "local_surface_water_hbond_per_h2o": r"Surface--water H-bond change per H$_2$O",
        "local_n2_contact_count": r"Local N$_2$ contact change",
    }
    for axis, metric in zip(axes.flat, plotted_metrics):
        lag_grid = sorted(
            {row["lag_ps"] for row in aligned_summary if row["metric"] == metric}
        )
        for case in cases:
            rows = sorted(
                (
                    row
                    for row in aligned_summary
                    if row["case_id"] == case.spec.case_id and row["metric"] == metric
                ),
                key=lambda row: row["lag_ps"],
            )
            if not rows:
                continue
            values_by_lag = {row["lag_ps"]: row["mean_delta"] for row in rows}
            axis.plot(
                lag_grid,
                [values_by_lag.get(lag, math.nan) for lag in lag_grid],
                color=_case_color(case.spec.ch3_fraction),
                linestyle="-" if case.spec.kind == "nanobubble" else "--",
                label=case.spec.case_id,
            )
        axis.axvline(0.0, color="0.5", lw=0.8)
        axis.axhline(0.0, color="0.75", lw=0.6)
        axis.set_ylabel(labels[metric])
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Time from candidate transition (ps)")
    axes[1, 1].set_xlabel("Time from candidate transition (ps)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend_labels, frameon=False, ncol=4, loc="upper center")
        figure.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        axes[0, 0].text(0.5, 0.5, "No repeated candidate clusters", ha="center", transform=axes[0, 0].transAxes)
        figure.tight_layout()
    _save_figure(figure, figures / "02_event_aligned_environment")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for index, case_id in enumerate(sorted({row["case_id"] for row in null_rows})):
        rows = [row for row in null_rows if row["case_id"] == case_id]
        null = [row["boundary_event_fraction"] for row in rows if not row["is_observed"]]
        observed = next(row["boundary_event_fraction"] for row in rows if row["is_observed"])
        axes[0].scatter(
            np.full(len(null), index),
            null,
            s=18,
            facecolors="none",
            edgecolors="0.55",
        )
        axes[0].scatter(index, observed, marker="D", color="black", s=35)
        null_distance = [row["mean_boundary_proxy_A"] for row in rows if not row["is_observed"]]
        observed_distance = next(row["mean_boundary_proxy_A"] for row in rows if row["is_observed"])
        axes[1].scatter(
            np.full(len(null_distance), index),
            null_distance,
            s=18,
            facecolors="none",
            edgecolors="0.55",
        )
        axes[1].scatter(index, observed_distance, marker="D", color="black", s=35)
    labels = sorted({row["case_id"] for row in null_rows})
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Boundary-associated event fraction")
    axes[1].set_ylabel(r"Mean boundary proxy (Å)")
    if not labels:
        for axis in axes:
            axis.text(0.5, 0.5, "No mixed-case candidate clusters", ha="center", transform=axis.transAxes)
    figure.tight_layout()
    _save_figure(figure, figures / "03_chemistry_circular_shift_null")
    plt.close(figure)

    bubbles = [case for case in cases if case.spec.kind == "nanobubble"]
    figure, axes = plt.subplots(len(bubbles), 2, figsize=(9.0, 2.2 * len(bubbles)), squeeze=False)
    clusters_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in clusters:
        clusters_by_case[row["case_id"]].append(row)
    for row_index, case in enumerate(sorted(bubbles, key=lambda item: -item.spec.ch3_fraction)):
        time = np.asarray([_float(row["time_ns"]) for row in case.frames])
        radius = np.asarray([_float(row["decomposed_mean_radius_A"]) for row in case.frames])
        valid = np.isfinite(radius)
        color = _case_color(case.spec.ch3_fraction)
        axes[row_index, 0].plot(time, radius, color=color, lw=0.8)
        relative = time - float(case.spec.attachment_time_ns)
        common = valid & (relative >= 0.0) & (relative <= 3.5)
        baseline = _nanmean(radius[common][:5]) if np.any(common) else math.nan
        common_radius = np.where(common, radius - baseline, math.nan)
        axes[row_index, 1].plot(relative, common_radius, color=color, lw=0.8)
        for event in clusters_by_case[case.spec.case_id]:
            axes[row_index, 0].axvline(event["transition_time_ns"], color="black", lw=0.4, alpha=0.45)
            event_relative = event["transition_time_ns"] - float(case.spec.attachment_time_ns)
            if 0.0 <= event_relative <= 3.5:
                axes[row_index, 1].axvline(event_relative, color="black", lw=0.4, alpha=0.45)
        axes[row_index, 0].set_ylabel(case.spec.case_id)
        axes[row_index, 0].spines[["top", "right"]].set_visible(False)
        axes[row_index, 1].spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Absolute time (ns)")
    axes[-1, 1].set_xlabel("Time after attachment (ns)")
    axes[0, 0].set_title(r"Mean $R_{CL}$ (Å)")
    axes[0, 1].set_title(r"Common-window $\Delta R_{CL}$ (Å)")
    figure.tight_layout()
    _save_figure(figure, figures / "04_bubble_absolute_attachment_time")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)
    for kind, marker in (("nanobubble", "o"), ("nanodroplet", "s")):
        for multiplier in sorted({row["jump_threshold_multiplier"] for row in sensitivity_rows}):
            rows = sorted(
                (
                    row
                    for row in sensitivity_rows
                    if row["kind"] == kind and row["jump_threshold_multiplier"] == multiplier
                ),
                key=lambda row: row["case_id"],
            )
            axis = axes[0] if kind == "nanobubble" else axes[1]
            axis.plot(
                range(len(rows)),
                [row["retained_repeated_event_clusters"] for row in rows],
                marker=marker,
                fillstyle="none",
                label=f"{multiplier:.2g}x",
            )
            axis.set_xticks(range(len(rows)), [row["case_id"] for row in rows], rotation=30, ha="right")
            axis.set_title(kind)
            axis.set_ylabel("Retained repeated clusters")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, title="Jump threshold")
    figure.tight_layout()
    _save_figure(figure, figures / "05_stricter_threshold_sensitivity")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=False)
    for axis, kind in zip(axes, ("nanobubble", "nanodroplet")):
        rows = [row for row in clusters if row["kind"] == kind]
        for row in rows:
            axis.scatter(
                row["transition_time_ns"],
                row["theta_deg"],
                s=30,
                facecolors="none",
                edgecolors=_case_color(float(row["ch3_fraction"])),
            )
        axis.set_title(kind)
        axis.set_ylabel(r"Candidate angle $\theta$ (deg)")
        axis.spines[["top", "right"]].set_visible(False)
        if not rows:
            axis.text(0.5, 0.5, "No repeated candidate clusters", ha="center", transform=axis.transAxes)
    axes[1].set_xlabel("Absolute time (ns)")
    figure.tight_layout()
    _save_figure(figure, figures / "06_candidate_event_map")
    plt.close(figure)

    metrics = (
        "local_hydration_areal_density_A-2",
        "local_water_water_hbond_degree",
        "local_surface_water_hbond_per_h2o",
    )
    metric_labels = (
        r"Hydration density (Å$^{-2}$)",
        "Water H-bond degree",
        r"Surface--water H-bonds per H$_2$O",
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=True)
    admitted_cases = {
        row["case_id"]
        for row in case_rows
        if row["fair_window_comparison_status"] == "ADMITTED"
    }
    for axis, metric, label in zip(axes, metrics, metric_labels):
        for kind, marker, linestyle in (
            ("nanobubble", "o", "-"),
            ("nanodroplet", "s", "--"),
        ):
            rows = sorted(
                (
                    row
                    for row in block_bootstrap_rows
                    if row["kind"] == kind
                    and row["metric"] == metric
                    and row["case_id"] in admitted_cases
                ),
                key=lambda row: row["ch3_fraction"],
            )
            if not rows:
                continue
            means = np.asarray([row["mean"] for row in rows])
            lower = means - np.asarray([row["bootstrap_ci025"] for row in rows])
            upper = np.asarray([row["bootstrap_ci975"] for row in rows]) - means
            axis.errorbar(
                [row["ch3_fraction"] for row in rows],
                means,
                yerr=np.vstack((lower, upper)),
                marker=marker,
                fillstyle="none",
                linestyle=linestyle,
                capsize=2,
                label=kind,
            )
        axis.set_xlabel(r"CH$_3$ site fraction")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, figures / "07_fair_window_block_environment")
    plt.close(figure)

    delta_metrics = (
        "local_hydration_areal_density_A-2",
        "local_water_water_hbond_degree",
        "local_surface_water_hbond_per_h2o",
        "local_n2_contact_count",
    )
    delta_labels = (
        r"Event $\Delta$ hydration (Å$^{-2}$)",
        r"Event $\Delta$ water H-bond degree",
        r"Event $\Delta$ surface H-bonds per H$_2$O",
        r"Event $\Delta$ local N$_2$ contacts",
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True)
    for axis, metric, label in zip(axes.flat, delta_metrics, delta_labels):
        for kind, marker, linestyle in (
            ("nanobubble", "o", "-"),
            ("nanodroplet", "s", "--"),
        ):
            rows = sorted(
                (
                    row
                    for row in event_delta_rows
                    if row["kind"] == kind and row["metric"] == metric
                ),
                key=lambda row: row["ch3_fraction"],
            )
            if not rows:
                continue
            means = np.asarray([row["mean_block_delta"] for row in rows])
            lower = means - np.asarray([row["bootstrap_ci025"] for row in rows])
            upper = np.asarray([row["bootstrap_ci975"] for row in rows]) - means
            axis.errorbar(
                [row["ch3_fraction"] for row in rows],
                means,
                yerr=np.vstack((lower, upper)),
                marker=marker,
                fillstyle="none",
                linestyle=linestyle,
                capsize=2,
                label=kind,
            )
        axis.axhline(0.0, color="0.65", lw=0.7)
        axis.set_xlabel(r"CH$_3$ site fraction")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, figures / "08_event_delta_block_bootstrap")
    plt.close(figure)


def _write_report(
    path: Path,
    cases: Sequence[CaseData],
    case_rows: Sequence[dict],
    null_summary: Sequence[dict],
    event_delta_rows: Sequence[dict],
    seed: int,
    block_ps: float,
) -> None:
    lines = [
        "# TPCL pinning--slip cross-case P0 comparison",
        "",
        "Status: `PASS`",
        "",
        (
            "This report describes repeated candidate dwell--jump clusters. It does not identify "
            "a free-energy barrier or prove causality from one trajectory per condition."
        ),
        "",
        "## Inputs and time resolution",
        "",
    ]
    for case in cases:
        lines.append(
            f"- `{case.spec.case_id}`: upstream job {case.job_id}; "
            f"{case.summary['frame_interval_ps']:g} ps frames; "
            f"{case.summary['valid_contour_frames']} valid contours; "
            f"{case.summary['candidate_event_count']} candidate clusters."
        )
    lines.extend(
        [
            "",
            "## Fair-window descriptive comparison",
            "",
            f"Nonoverlapping {block_ps:g} ps blocks were used; bootstrap seed: `{seed}`.",
            "",
        ]
    )
    for row in case_rows:
        lines.append(
            f"- `{row['case_id']}`: {row['fair_event_clusters']} clusters in "
            f"{row['fair_valid_duration_ns']:.3f} valid ns "
            f"({row['fair_event_rate_per_valid_ns']:.3g} ns^-1), "
            f"{row['fair_contour_coverage_fraction']:.1%} contour coverage, "
            f"window `{row['fair_window']}`; "
            f"comparison `{row['fair_window_comparison_status']}`."
        )
    lines.extend(
        [
            "",
            "## Circular angular-shift chemistry null",
            "",
            (
                "The null uses every motion-qualified event cluster before the chemistry-based "
                "repetition label, avoiding selection on the same chemistry association being "
                "tested."
            ),
            "",
        ]
    )
    if null_summary:
        for row in null_summary:
            lines.append(
                f"- `{row['case_id']}`: boundary-fraction empirical p = "
                f"{row['empirical_p_boundary_event_fraction']:.3g} "
                f"(BH q = {row['bh_q_boundary_event_fraction']:.3g}); "
                f"nearest-site empirical p = {row['empirical_p_nearest_ch3_fraction']:.3g} "
                f"(BH q = {row['bh_q_nearest_ch3_fraction']:.3g})."
            )
    else:
        lines.append("- No mixed-surface case had motion-qualified event clusters; chemistry null is NA.")
    lines.extend(["", "## Event-associated environment changes", ""])
    for case in cases:
        rows = [row for row in event_delta_rows if row["case_id"] == case.spec.case_id]
        hydration = next(
            (row for row in rows if row["metric"] == "local_hydration_areal_density_A-2"),
            None,
        )
        if hydration is None:
            lines.append(f"- `{case.spec.case_id}`: hydration delta is NA.")
        else:
            lines.append(
                f"- `{case.spec.case_id}`: block-mean hydration delta "
                f"{hydration['mean_block_delta']:.4g} A^-2; 95% time-block bootstrap interval "
                f"[{hydration['bootstrap_ci025']:.4g}, {hydration['bootstrap_ci975']:.4g}]."
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "Chemistry association is reported only when the observed mapping differs from "
                "the composition-preserving circular-shift null. Hydration, H-bond, and N2 "
                "changes are event-aligned correlations, not proof that they caused the "
                "transition. Sub-frame motion remains unresolved, and no threshold was relaxed "
                "after observing results."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_compare(
    manifest: Path,
    output_dir: Path,
    *,
    font_path: Path,
    block_ps: float = 200.0,
    event_half_window_ps: float = 100.0,
    bootstrap_replicates: int = 2000,
    seed: int = 20260830,
) -> dict:
    if block_ps <= 0 or event_half_window_ps <= 0 or bootstrap_replicates < 100:
        raise ValueError("Comparison settings are outside their valid range")
    cases = [load_case(spec) for spec in read_case_manifest(manifest)]
    if {case.spec.kind for case in cases} != {"nanobubble", "nanodroplet"}:
        raise ValueError("Comparison requires both nanobubble and nanodroplet cases")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    clusters = [row for case in cases for row in cluster_summary(case)]
    validity_intervals = [row for case in cases for row in contour_validity_intervals(case)]
    case_rows = build_case_summary(cases, clusters)
    block_rows = [row for case in cases for row in build_block_rows(case, block_ps)]
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    summary_by_case = {row["case_id"]: row for row in case_rows}
    for case in cases:
        summary = summary_by_case[case.spec.case_id]
        rows = [row for row in block_rows if row["case_id"] == case.spec.case_id]
        for row in bootstrap_block_summary(
            rows,
            float(summary["fair_start_ns"]),
            float(summary["fair_end_ns"]),
            rng,
            bootstrap_replicates,
        ):
            bootstrap_rows.append(
                {
                    "case_id": case.spec.case_id,
                    "kind": case.spec.kind,
                    "ch3_fraction": case.spec.ch3_fraction,
                    **row,
                }
            )
    aligned = [row for case in cases for row in build_event_aligned_rows(case, event_half_window_ps)]
    aligned_summary = summarize_event_aligned(aligned)
    event_delta_rows = bootstrap_event_deltas(
        clusters,
        block_ps,
        bootstrap_replicates,
        seed + 1,
    )
    null_rows = [row for case in cases for row in circular_shift_null(case)]
    null_summary = summarize_null(null_rows)
    sensitivity = [
        row for case in cases for row in threshold_sensitivity(case, (1.0, 1.25, 1.5))
    ]
    tables = {
        "case_summary.csv": (case_rows, list(case_rows[0])),
        "contour_validity_intervals.csv": (
            validity_intervals,
            list(validity_intervals[0]),
        ),
        "event_cluster_summary.csv": (
            clusters,
            list(clusters[0])
            if clusters
            else [
                "case_id",
                "kind",
                "ch3_fraction",
                "event_cluster_id",
                "transition_time_ns",
                "theta_deg",
                "mechanism_class",
            ],
        ),
        "block_summary_200ps.csv": (block_rows, list(block_rows[0])),
        "block_bootstrap_summary.csv": (
            bootstrap_rows,
            list(bootstrap_rows[0]) if bootstrap_rows else ["case_id", "metric"],
        ),
        "event_aligned_cluster_metrics.csv": (
            aligned,
            list(aligned[0]) if aligned else ["case_id", "event_cluster_id", "lag_ps"],
        ),
        "event_aligned_summary.csv": (
            aligned_summary,
            list(aligned_summary[0]) if aligned_summary else ["case_id", "lag_ps", "metric"],
        ),
        "event_delta_block_bootstrap.csv": (
            event_delta_rows,
            list(event_delta_rows[0]) if event_delta_rows else ["case_id", "metric"],
        ),
        "circular_shift_null.csv": (
            null_rows,
            list(null_rows[0]) if null_rows else ["case_id", "shift_bins", "is_observed"],
        ),
        "circular_shift_null_summary.csv": (
            null_summary,
            list(null_summary[0]) if null_summary else ["case_id", "null_shifts"],
        ),
        "stricter_threshold_sensitivity.csv": (sensitivity, list(sensitivity[0])),
    }
    for name, (rows, fields) in tables.items():
        _write_csv(output / name, rows, fields)
    write_figures(
        cases,
        case_rows,
        clusters,
        bootstrap_rows,
        event_delta_rows,
        aligned_summary,
        null_rows,
        sensitivity,
        output,
        font_path,
    )
    _write_report(
        output / "report.md",
        cases,
        case_rows,
        null_summary,
        event_delta_rows,
        seed,
        block_ps,
    )
    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "upstream_jobs": {case.spec.case_id: case.job_id for case in cases},
        "candidate_event_clusters": len(clusters),
        "candidate_clusters_by_case": {
            row["case_id"]: row["candidate_event_clusters"] for row in case_rows
        },
        "block_ps": block_ps,
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": seed,
        "event_half_window_ps": event_half_window_ps,
        "circular_null": "all nonzero angular-bin shifts within each mixed case",
        "fair_window_min_contour_coverage": FAIR_WINDOW_MIN_CONTOUR_COVERAGE,
        "threshold_sensitivity": "baseline and stricter only; no relaxed threshold",
        "scientific_classification_ceiling": "WORTH_INDEPENDENT_TRAJECTORY_VALIDATION",
        "causal_or_free_energy_claim": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--event-half-window-ps", type=float, default=100.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_compare(
        args.manifest,
        args.output_dir,
        font_path=args.font_path,
        block_ps=args.block_ps,
        event_half_window_ps=args.event_half_window_ps,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
