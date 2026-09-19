"""Synthesize finite-droplet TPCL anchoring from accepted constant-force outputs."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

import numpy as np

from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)
from molsimflow.postprocess.interfacial_water_density import classify_regions

EVENT_LABELS = {
    "slow": "DWELL_CANDIDATE",
    "acceleration": "STARTUP_CANDIDATE",
    "fast": "ADVANCE_CANDIDATE",
}
REGIONS = ("leading_edge", "trailing_edge", "footprint_interior")
REGION_METRICS = (
    "surface_hbond_per_contact_water",
    "surface_hbond_per_contact_line_length_A-1",
    "surface_hbond_per_accessible_site",
    "ch3_fraction",
    "water_water_hbond_degree",
)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with _open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _float(raw: object) -> float:
    return float(str(raw))


def _mean(values: Iterable[object]) -> float:
    array = np.asarray([_float(value) for value in values], dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def _angular_distance(left_deg: float, right_deg: float) -> float:
    return abs((left_deg - right_deg + 180.0) % 360.0 - 180.0)


def _verify_hashes(root: Path) -> None:
    """Verify a result tree whose manifest may retain its original absolute root."""

    manifest = root / "OUTPUT-SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    marker = f"/{root.name}/"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(maxsplit=1)
        raw_path = raw_path.strip().removeprefix("./")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        elif not path.exists() and marker in raw_path:
            path = root / raw_path.split(marker, 1)[1]
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Source result hash mismatch: {path}")


def morphology_applicability(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Classify where a unique finite-droplet TPCL interpretation is valid."""

    output = []
    for row in rows:
        morphology = str(row["morphology_class"])
        direction = str(row["direction"]).lower()
        if morphology == "finite_droplet" and direction in {"x", "y"}:
            applicability = "DIRECTED_TPCL_APPLICABLE"
            reason = "unique_finite_droplet_leading_and_trailing_edges"
        elif morphology == "finite_droplet":
            applicability = "UNDIRECTED_REFERENCE_ONLY"
            reason = "finite_droplet_without_a_driven_direction"
        elif morphology == "water_islands":
            applicability = "NOT_APPLICABLE_MULTIPLE_ISLANDS"
            reason = "no_unique_single_droplet_contact_line"
        elif morphology == "spread_film":
            applicability = "NOT_APPLICABLE_PERIODIC_FILM"
            reason = "no_unique_finite_droplet_contact_line"
        else:
            applicability = "NOT_APPLICABLE_UNRESOLVED_MORPHOLOGY"
            reason = "morphology_not_supported_by_this_analysis"
        output.append(
            {
                "case_id": row["case_id"],
                "branch_id": row["branch_id"],
                "direction": direction,
                "morphology_class": morphology,
                "morphology_gate": row["morphology_gate"],
                "applicability": applicability,
                "reason": reason,
            }
        )
    return output


def _surface_site_regions(case_dir: Path) -> dict[int, dict[str, float]]:
    sites = _read_table(case_dir / "surface_site" / "surface_sites.csv")
    site_xy = np.asarray([[_float(row["x_A"]), _float(row["y_A"])] for row in sites], dtype=float)
    is_ch3 = np.asarray([row["site_type"] == "CH3" for row in sites], dtype=bool)
    contour_rows = _read_table(case_dir / "tpcl" / "contour_points.csv.gz")
    contours: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in contour_rows:
        contours[int(row["step"])].append((_float(row["contour_x_A"]), _float(row["contour_y_A"])))
    frame_rows = _read_table(case_dir / "tpcl" / "frame_metrics.csv")
    manifest = json.loads((case_dir / "tpcl" / "manifest.json").read_text(encoding="utf-8"))
    half_width = _float(manifest["config"]["tpcl_half_width_A"])
    output: dict[int, dict[str, float]] = {}
    for row in frame_rows:
        step = int(row["step"])
        boundary = np.asarray(contours[step], dtype=float)
        center = np.asarray(
            [_float(row["phase_center_x_A"]), _float(row["phase_center_y_A"])],
            dtype=float,
        )
        lengths = np.asarray([_float(row["box_x_A"]), _float(row["box_y_A"])])
        masks = classify_regions(
            site_xy,
            boundary,
            center,
            lengths,
            tpcl_half_width=half_width,
        )
        interior = masks["footprint"]
        count = int(np.count_nonzero(interior))
        output[step] = {
            "accessible_site_count": count,
            "ch3_fraction": (
                float(np.count_nonzero(interior & is_ch3) / count) if count else math.nan
            ),
            "contact_line_perimeter_A": _float(row["contact_line_perimeter_A"]),
        }
    return output


def _arc_frame_rows(
    case_id: str,
    branch_id: str,
    branch_direction: str,
    analysis_axis: str,
    case_dir: Path,
    timestep_fs: float,
) -> list[dict[str, object]]:
    arcs = _read_table(case_dir / "tpcl" / "local_arc_metrics.csv.gz")
    first_step = min(int(row["step"]) for row in arcs)
    targets = (
        ("leading_edge", 0.0 if analysis_axis == "x" else 90.0),
        ("trailing_edge", 180.0 if analysis_axis == "x" else 270.0),
    )
    output: list[dict[str, object]] = []
    for region, target in targets:
        selected = [
            row for row in arcs if _angular_distance(_float(row["theta_deg"]), target) < 1.0e-8
        ]
        if not selected:
            raise ValueError(f"{case_id}/{branch_id}: no arc at {target} degrees")
        for row in selected:
            water = _float(row["local_molecular_water_count"])
            hbond_per_water = _float(row["local_surface_water_hbond_per_h2o"])
            if water == 0.0:
                hbond = 0.0
            elif math.isfinite(hbond_per_water):
                hbond = hbond_per_water * water
            else:
                raise ValueError(
                    f"{case_id}/{branch_id}: nonfinite local H-bond density "
                    f"with {water} contact waters at step {row['step']}"
                )
            length = _float(row["local_arc_length_A"])
            sites = _float(row["local_site_count"])
            step = int(row["step"])
            output.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "branch_direction": branch_direction,
                    "analysis_axis": analysis_axis,
                    "region": region,
                    "step": step,
                    "time_ps": (step - first_step) * timestep_fs / 1000.0,
                    "surface_hbond_count": hbond,
                    "contact_water_count": water,
                    "contact_line_length_A": length,
                    "accessible_site_count": sites,
                    "surface_hbond_per_contact_water": _ratio(hbond, water),
                    "surface_hbond_per_contact_line_length_A-1": _ratio(hbond, length),
                    "surface_hbond_per_accessible_site": _ratio(hbond, sites),
                    "ch3_fraction": _float(row["local_ch3_fraction"]),
                    "water_water_hbond_degree": _float(row["local_water_water_hbond_degree"]),
                    "count_definition": "local_arc_proxy",
                }
            )
    return output


def _footprint_frame_rows(
    case_id: str,
    branch_id: str,
    branch_direction: str,
    analysis_axis: str,
    case_dir: Path,
    timestep_fs: float,
) -> list[dict[str, object]]:
    hbonds = _read_table(case_dir / "interfacial_hbond" / "hbond_by_frame.csv")
    site_regions = _surface_site_regions(case_dir)
    first_step = min(int(row["step"]) for row in hbonds)
    output = []
    for row in hbonds:
        step = int(row["step"])
        water = _float(row["footprint_h2o_count"])
        hbond = _float(row["footprint_water_donor_sioh_count"]) + _float(
            row["footprint_sioh_donor_water_count"]
        )
        sites = site_regions[step]["accessible_site_count"]
        perimeter = site_regions[step]["contact_line_perimeter_A"]
        output.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "branch_direction": branch_direction,
                "analysis_axis": analysis_axis,
                "region": "footprint_interior",
                "step": step,
                "time_ps": (step - first_step) * timestep_fs / 1000.0,
                "surface_hbond_count": hbond,
                "contact_water_count": water,
                "contact_line_length_A": perimeter,
                "accessible_site_count": sites,
                "surface_hbond_per_contact_water": _ratio(hbond, water),
                "surface_hbond_per_contact_line_length_A-1": _ratio(hbond, perimeter),
                "surface_hbond_per_accessible_site": _ratio(hbond, sites),
                "ch3_fraction": site_regions[step]["ch3_fraction"],
                "water_water_hbond_degree": _float(row["footprint_water_water_hbond_degree"]),
                "count_definition": "exact_disjoint_footprint_interior",
            }
        )
    return output


def summarize_region_frames(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate counts before normalizing to avoid mean-of-ratios bias."""

    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["branch_id"]),
            str(row["branch_direction"]),
            str(row["analysis_axis"]),
            str(row["region"]),
        )
        grouped[key].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        hbond = sum(_float(row["surface_hbond_count"]) for row in selected)
        water = sum(_float(row["contact_water_count"]) for row in selected)
        length = sum(_float(row["contact_line_length_A"]) for row in selected)
        sites = sum(_float(row["accessible_site_count"]) for row in selected)
        output.append(
            {
                "case_id": key[0],
                "branch_id": key[1],
                "branch_direction": key[2],
                "analysis_axis": key[3],
                "region": key[4],
                "frame_count": len(selected),
                "mean_surface_hbond_count": hbond / len(selected),
                "mean_contact_water_count": water / len(selected),
                "mean_contact_line_length_A": length / len(selected),
                "mean_accessible_site_count": sites / len(selected),
                "surface_hbond_per_contact_water": _ratio(hbond, water),
                "surface_hbond_per_contact_line_length_A-1": _ratio(hbond, length),
                "surface_hbond_per_accessible_site": _ratio(hbond, sites),
                "mean_ch3_fraction": _mean(row["ch3_fraction"] for row in selected),
                "mean_water_water_hbond_degree": _mean(
                    row["water_water_hbond_degree"] for row in selected
                ),
                "count_definition": selected[0]["count_definition"],
            }
        )
    return output


def _anchor_frame_sets(
    anchor_dir: Path,
) -> tuple[dict[int, set[tuple[int, int]]], list[dict[str, str]]]:
    frame_rows = _read_table(anchor_dir / "tpcl_hbond_frames.csv")
    edges: dict[int, set[tuple[int, int]]] = {int(row["step"]): set() for row in frame_rows}
    for row in _read_table(anchor_dir / "tpcl_hbond_edges.csv.gz"):
        if row["edge_scope"] != "surface_anchor_tpcl":
            continue
        if row["donor_species"] == "h2o":
            water_id, site_id = int(row["donor_id"]), int(row["acceptor_id"])
        else:
            water_id, site_id = int(row["acceptor_id"]), int(row["donor_id"])
        edges[int(row["step"])].add((water_id, site_id))
    return edges, frame_rows


def anchor_interval_rows(
    case_id: str,
    branch_id: str,
    direction: str,
    frame_sets: Mapping[int, set[tuple[int, int]]],
    *,
    timestep_fs: float,
) -> list[dict[str, object]]:
    """Measure pair retention between adjacent snapshots, not H-bond lifetimes."""

    steps = sorted(frame_sets)
    if len(steps) < 2:
        raise ValueError(f"{case_id}/{branch_id}: fewer than two anchor frames")
    first_step = steps[0]
    output = []
    for left_step, right_step in zip(steps, steps[1:]):
        left, right = frame_sets[left_step], frame_sets[right_step]
        shared = left.intersection(right)
        union = left.union(right)
        output.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "left_step": left_step,
                "right_step": right_step,
                "left_time_ps": (left_step - first_step) * timestep_fs / 1000.0,
                "right_time_ps": (right_step - first_step) * timestep_fs / 1000.0,
                "left_anchor_pair_count": len(left),
                "right_anchor_pair_count": len(right),
                "shared_anchor_pair_count": len(shared),
                "formed_anchor_pair_count": len(right.difference(left)),
                "lost_anchor_pair_count": len(left.difference(right)),
                "anchor_pair_jaccard": _ratio(len(shared), len(union)),
                "anchor_pair_retained_fraction": _ratio(len(shared), len(left)),
                "sampling_interval_ps": (right_step - left_step) * timestep_fs / 1000.0,
                "evidence_limit": "cross_snapshot_retention_not_hbond_lifetime",
            }
        )
    return output


def _maximum_consecutive_support(
    frame_sets: Mapping[int, set[tuple[int, int]]],
) -> tuple[int, int]:
    active: dict[tuple[int, int], int] = {}
    maximum = 0
    persistent: set[tuple[int, int]] = set()
    for step in sorted(frame_sets):
        present = frame_sets[step]
        next_active: dict[tuple[int, int], int] = {}
        for pair in present:
            support = active.get(pair, 0) + 1
            next_active[pair] = support
            maximum = max(maximum, support)
            if support >= 2:
                persistent.add(pair)
        active = next_active
    return maximum, len(persistent)


def summarize_anchor_branch(
    case_id: str,
    branch_id: str,
    direction: str,
    frame_sets: Mapping[int, set[tuple[int, int]]],
    frame_rows: Sequence[Mapping[str, str]],
    intervals: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    maximum, persistent_count = _maximum_consecutive_support(frame_sets)
    return {
        "case_id": case_id,
        "branch_id": branch_id,
        "direction": direction,
        "frame_count": len(frame_sets),
        "interval_count": len(intervals),
        "mean_snapshot_anchor_pair_count": _mean(len(value) for value in frame_sets.values()),
        "mean_snapshot_anchor_water_count": _mean(
            row["surface_anchor_water_count"] for row in frame_rows
        ),
        "mean_snapshot_anchor_site_count": _mean(
            row["surface_anchor_site_count"] for row in frame_rows
        ),
        "mean_anchor_pair_jaccard": _mean(row["anchor_pair_jaccard"] for row in intervals),
        "mean_anchor_pair_retained_fraction": _mean(
            row["anchor_pair_retained_fraction"] for row in intervals
        ),
        "mean_formed_anchor_pairs_per_interval": _mean(
            row["formed_anchor_pair_count"] for row in intervals
        ),
        "mean_lost_anchor_pairs_per_interval": _mean(
            row["lost_anchor_pair_count"] for row in intervals
        ),
        "fraction_intervals_with_shared_anchor": _mean(
            int(_float(row["shared_anchor_pair_count"]) > 0) for row in intervals
        ),
        "maximum_consecutive_snapshot_support": maximum,
        "anchor_pairs_with_at_least_two_consecutive_snapshots": persistent_count,
        "evidence_limit": "10_ps_snapshot_persistence_not_hbond_lifetime",
    }


def _event_definitions(case_dir: Path) -> list[dict[str, object]]:
    rows = _read_table(case_dir / "derived" / "event_steps.csv")
    events: dict[str, dict[str, object]] = {}
    for row in rows:
        event_id = row["event_id"]
        events[event_id] = {
            "event_id": event_id,
            "event_class": row["event_class"],
            "mechanism_label": EVENT_LABELS[row["event_class"]],
            "event_center_ps": _float(row["event_center_ps"]),
            "event_score": _float(row["event_score"]),
        }
    return [events[key] for key in sorted(events)]


def _window_mean(
    rows: Sequence[Mapping[str, object]],
    field: str,
    center_ps: float,
    half_window_ps: float,
    side: str,
) -> float:
    if side == "pre":
        selected = [
            row for row in rows if center_ps - half_window_ps <= _float(row["time_ps"]) < center_ps
        ]
    else:
        selected = [
            row for row in rows if center_ps < _float(row["time_ps"]) <= center_ps + half_window_ps
        ]
    return _mean(row[field] for row in selected)


def event_region_responses(
    case_id: str,
    branch_id: str,
    direction: str,
    events: Sequence[Mapping[str, object]],
    region_rows: Sequence[Mapping[str, object]],
    *,
    half_window_ps: float,
) -> list[dict[str, object]]:
    output = []
    for event in events:
        center = _float(event["event_center_ps"])
        for region in REGIONS:
            selected = [row for row in region_rows if row["region"] == region]
            item: dict[str, object] = {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                **event,
                "region": region,
                "half_window_ps": half_window_ps,
            }
            for metric in REGION_METRICS:
                pre = _window_mean(selected, metric, center, half_window_ps, "pre")
                post = _window_mean(selected, metric, center, half_window_ps, "post")
                item[f"pre_{metric}"] = pre
                item[f"post_{metric}"] = post
                item[f"delta_{metric}"] = post - pre
            output.append(item)
    return output


def _load_water_network_frames(case_dir: Path, timestep_fs: float) -> list[dict[str, object]]:
    rows = _read_table(case_dir / "local_water_order" / "water_order_by_frame.csv")
    first_step = min(int(row["step"]) for row in rows)
    return [
        {
            **row,
            "time_ps": (int(row["step"]) - first_step) * timestep_fs / 1000.0,
        }
        for row in rows
    ]


def event_network_responses(
    case_id: str,
    branch_id: str,
    direction: str,
    events: Sequence[Mapping[str, object]],
    water_frames: Sequence[Mapping[str, object]],
    anchor_frames: Sequence[Mapping[str, object]],
    anchor_intervals: Sequence[Mapping[str, object]],
    water_intervals: Sequence[Mapping[str, str]],
    *,
    half_window_ps: float,
) -> list[dict[str, object]]:
    output = []
    for event in events:
        center = _float(event["event_center_ps"])
        item: dict[str, object] = {
            "case_id": case_id,
            "branch_id": branch_id,
            "direction": direction,
            **event,
            "half_window_ps": half_window_ps,
        }
        frame_metrics = {
            "water_network_largest_component_fraction": "hbond_largest_component_fraction",
            "water_network_hbond_degree": "mean_hbond_degree",
            "snapshot_anchor_pair_count": "anchor_pair_count",
        }
        for output_name, source_name in frame_metrics.items():
            source = anchor_frames if source_name == "anchor_pair_count" else water_frames
            pre = _window_mean(source, source_name, center, half_window_ps, "pre")
            post = _window_mean(source, source_name, center, half_window_ps, "post")
            item[f"pre_{output_name}"] = pre
            item[f"post_{output_name}"] = post
            item[f"delta_{output_name}"] = post - pre
        for prefix, source, time_field, metrics in (
            (
                "anchor",
                anchor_intervals,
                "mid_time_ps",
                (
                    "anchor_pair_jaccard",
                    "anchor_pair_retained_fraction",
                    "formed_anchor_pair_count",
                    "lost_anchor_pair_count",
                ),
            ),
            (
                "water",
                water_intervals,
                "mid_time_ps",
                (
                    "edge_jaccard",
                    "edge_retained_fraction",
                    "formed_edge_count",
                    "lost_edge_count",
                ),
            ),
        ):
            if prefix == "water":
                source = [row for row in source if row["event_id"] == event["event_id"]]
            normalized = [{**row, "time_ps": row[time_field]} for row in source]
            for metric in metrics:
                pre = _window_mean(normalized, metric, center, half_window_ps, "pre")
                post = _window_mean(normalized, metric, center, half_window_ps, "post")
                item[f"pre_{prefix}_{metric}"] = pre
                item[f"post_{prefix}_{metric}"] = post
                item[f"delta_{prefix}_{metric}"] = post - pre
        output.append(item)
    return output


def _plot(
    region_summary: Sequence[Mapping[str, object]],
    anchor_summary: Sequence[Mapping[str, object]],
    event_network: Sequence[Mapping[str, object]],
    output: Path,
    font_path: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    if font_path is not None:
        font_manager.fontManager.addfont(font_path)
        properties = font_manager.FontProperties(fname=font_path)
        font_manager.findfont(properties, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = properties.get_name()
    driven = [row for row in region_summary if row["branch_direction"] != "none"]
    labels = sorted({f"{row['case_id']}/{str(row['branch_direction']).upper()}" for row in driven})
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    colors = {
        "leading_edge": "#d95f02",
        "trailing_edge": "#1b9e77",
        "footprint_interior": "#7570b3",
    }
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    for offset, region in enumerate(REGIONS):
        selected = {
            f"{row['case_id']}/{str(row['branch_direction']).upper()}": row
            for row in driven
            if row["region"] == region
        }
        axes[0, 0].bar(
            x + (offset - 1) * width,
            [selected[label]["surface_hbond_per_contact_water"] for label in labels],
            width,
            color=colors[region],
            label=region.replace("_", " "),
        )
        axes[0, 1].bar(
            x + (offset - 1) * width,
            [selected[label]["surface_hbond_per_accessible_site"] for label in labels],
            width,
            color=colors[region],
        )
    axes[0, 0].set_ylabel("Surface H-bonds / contact water")
    axes[0, 1].set_ylabel("Surface H-bonds / accessible site")
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=25, ha="right")
    axes[0, 0].legend(frameon=False, fontsize=8)

    anchor_driven = [row for row in anchor_summary if row["direction"] != "none"]
    anchor_labels = [f"{row['case_id']}/{str(row['direction']).upper()}" for row in anchor_driven]
    axes[1, 0].bar(
        np.arange(len(anchor_driven)),
        [row["mean_snapshot_anchor_pair_count"] for row in anchor_driven],
        color="#4c78a8",
    )
    axes[1, 0].set_xticks(np.arange(len(anchor_driven)), anchor_labels, rotation=25, ha="right")
    axes[1, 0].set_ylabel("SiOH-water anchor pairs / snapshot")

    classes = ("DWELL_CANDIDATE", "STARTUP_CANDIDATE", "ADVANCE_CANDIDATE")
    for case_id, color in (("ch3_only", "#999999"), ("mixed291", "#e45756")):
        means = []
        for label in classes:
            selected = [
                row
                for row in event_network
                if row["case_id"] == case_id and row["mechanism_label"] == label
            ]
            means.append(
                _mean(row["delta_water_network_largest_component_fraction"] for row in selected)
            )
        axes[1, 1].plot(classes, means, "o-", label=case_id, color=color)
    axes[1, 1].axhline(0.0, color="black", lw=0.7)
    axes[1, 1].set_ylabel("Post-pre largest-component fraction")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "dynamic_tpcl_anchor_synthesis.png", dpi=240)
    plt.close(figure)


def run_contract(
    contract_path: Path,
    output_path: Path,
    *,
    anchor_root: Path,
) -> dict[str, object]:
    """Run the contract-driven Stage-B3 synthesis."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    anchor_root = Path(anchor_root).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    base = contract_path.parent
    stage_a = resolve_path(contract["stage_a_results"], base)
    mechanism = resolve_path(contract["mechanism_run"], base)
    _verify_hashes(stage_a)
    _verify_hashes(mechanism)
    timestep_fs = _float(contract["timestep_fs"])
    half_window_ps = _float(contract.get("event_half_window_ps", 100.0))
    morphology = morphology_applicability(
        _read_table(stage_a / "cross_interface_stage_a.tsv", delimiter="\t")
    )
    cases = contract.get("cases", [])
    if len(cases) != 6:
        raise ValueError("B3 finite-droplet synthesis requires six branch entries")

    all_region_frames: list[dict[str, object]] = []
    all_anchor_intervals: list[dict[str, object]] = []
    anchor_summaries: list[dict[str, object]] = []
    event_regions: list[dict[str, object]] = []
    event_network: list[dict[str, object]] = []
    input_paths: set[Path] = {
        contract_path,
        stage_a / "OUTPUT-SHA256SUMS",
        stage_a / "cross_interface_stage_a.tsv",
        mechanism / "OUTPUT-SHA256SUMS",
    }
    for entry in cases:
        case_id = str(entry["case_id"])
        branch_id = str(entry["branch_id"])
        direction = str(entry["direction"]).lower()
        if direction not in {"none", "x", "y"}:
            raise ValueError(f"invalid direction: {direction}")
        case_dir = mechanism / "cases" / case_id / branch_id
        axes = ("x", "y") if direction == "none" else (direction,)
        branch_region_rows: list[dict[str, object]] = []
        for axis in axes:
            branch_region_rows.extend(
                _arc_frame_rows(
                    case_id,
                    branch_id,
                    direction,
                    axis,
                    case_dir,
                    timestep_fs,
                )
            )
            branch_region_rows.extend(
                _footprint_frame_rows(
                    case_id,
                    branch_id,
                    direction,
                    axis,
                    case_dir,
                    timestep_fs,
                )
            )
        all_region_frames.extend(branch_region_rows)

        anchor_dir = anchor_root / case_id / branch_id
        frame_sets, anchor_frame_source = _anchor_frame_sets(anchor_dir)
        intervals = anchor_interval_rows(
            case_id,
            branch_id,
            direction,
            frame_sets,
            timestep_fs=timestep_fs,
        )
        for row in intervals:
            row["mid_time_ps"] = 0.5 * (_float(row["left_time_ps"]) + _float(row["right_time_ps"]))
        all_anchor_intervals.extend(intervals)
        anchor_summaries.append(
            summarize_anchor_branch(
                case_id,
                branch_id,
                direction,
                frame_sets,
                anchor_frame_source,
                intervals,
            )
        )
        first_step = min(frame_sets)
        anchor_frames = [
            {
                "step": step,
                "time_ps": (step - first_step) * timestep_fs / 1000.0,
                "anchor_pair_count": len(pairs),
            }
            for step, pairs in sorted(frame_sets.items())
        ]
        if direction != "none":
            events = _event_definitions(case_dir)
            current_region_rows = [
                row for row in branch_region_rows if row["analysis_axis"] == direction
            ]
            event_regions.extend(
                event_region_responses(
                    case_id,
                    branch_id,
                    direction,
                    events,
                    current_region_rows,
                    half_window_ps=half_window_ps,
                )
            )
            water_intervals = _read_table(
                case_dir / "event_hbond_summary" / "edge_turnover_intervals.tsv",
                delimiter="\t",
            )
            for row in water_intervals:
                row["mid_time_ps"] = 0.5 * (
                    _float(row["left_time_ps"]) + _float(row["right_time_ps"])
                )
            event_network.extend(
                event_network_responses(
                    case_id,
                    branch_id,
                    direction,
                    events,
                    _load_water_network_frames(case_dir, timestep_fs),
                    anchor_frames,
                    intervals,
                    water_intervals,
                    half_window_ps=half_window_ps,
                )
            )
        input_paths.update(
            {
                case_dir / "tpcl" / "local_arc_metrics.csv.gz",
                case_dir / "tpcl" / "contour_points.csv.gz",
                case_dir / "tpcl" / "frame_metrics.csv",
                case_dir / "tpcl" / "manifest.json",
                case_dir / "surface_site" / "surface_sites.csv",
                case_dir / "interfacial_hbond" / "hbond_by_frame.csv",
                anchor_dir / "tpcl_hbond_edges.csv.gz",
                anchor_dir / "tpcl_hbond_frames.csv",
                anchor_dir / "summary.json",
                anchor_dir / "manifest.json",
            }
        )
        if direction != "none":
            input_paths.update(
                {
                    case_dir / "derived" / "event_steps.csv",
                    case_dir / "event_hbond_summary" / "edge_turnover_intervals.tsv",
                    case_dir / "local_water_order" / "water_order_by_frame.csv",
                }
            )

    region_summary = summarize_region_frames(all_region_frames)
    output.mkdir(parents=True)
    write_tsv(
        output / "morphology_applicability.tsv",
        morphology,
        tuple(morphology[0]),
    )
    write_tsv(
        output / "region_normalization_by_frame.tsv",
        all_region_frames,
        tuple(all_region_frames[0]),
    )
    write_tsv(
        output / "region_normalization_summary.tsv",
        region_summary,
        tuple(region_summary[0]),
    )
    write_tsv(
        output / "anchor_retention_intervals.tsv",
        all_anchor_intervals,
        tuple(all_anchor_intervals[0]),
    )
    write_tsv(
        output / "anchor_branch_summary.tsv",
        anchor_summaries,
        tuple(anchor_summaries[0]),
    )
    write_tsv(
        output / "event_region_response.tsv",
        event_regions,
        tuple(event_regions[0]),
    )
    write_tsv(
        output / "event_network_response.tsv",
        event_network,
        tuple(event_network[0]),
    )
    input_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(input_paths)
    ]
    write_tsv(output / "input_manifest.tsv", input_rows, tuple(input_rows[0]))
    font_path = contract.get("font_path")
    _plot(
        region_summary,
        anchor_summaries,
        event_network,
        output,
        resolve_path(font_path, base) if font_path else None,
    )
    plan = contract.get("high_density_review_plan", {})
    (output / "HIGH-DENSITY-OUTPUT-REVIEW.md").write_text(
        "# High-density output review plan\n\n"
        "Status: `REVIEW_ONLY_NOT_SUBMITTED`\n\n"
        "This plan is only for a future representative-case diagnostic if the 10 ps "
        "snapshot evidence is insufficient. It does not authorize or submit MD.\n\n"
        f"- Proposed duration: {plan.get('duration_ps', '20-100 ps')}\n"
        f"- Proposed coordinate interval: {plan.get('coordinate_interval_fs', '5-10 fs')}\n"
        "- Required fields: water and contact-region coordinates, atom identity, box, "
        "surface reference, and the existing drive/motion observables.\n"
        "- Intended test: distinguish short-lived surface contacts from cross-snapshot "
        "anchors during independently selected dwell/startup/advance candidates.\n"
        "- Forbidden interpretation: the existing 10 ps data do not resolve H-bond "
        "lifetimes or sub-picosecond dynamics.\n",
        encoding="utf-8",
    )
    summary = {
        "status": "PASS",
        "morphology_rows": len(morphology),
        "finite_branch_entries": len(cases),
        "region_frame_rows": len(all_region_frames),
        "region_summary_rows": len(region_summary),
        "anchor_interval_rows": len(all_anchor_intervals),
        "anchor_branch_rows": len(anchor_summaries),
        "event_region_rows": len(event_regions),
        "event_network_rows": len(event_network),
        "snapshot_interval_ps": _mean(row["sampling_interval_ps"] for row in all_anchor_intervals),
        "surface_anchor_identity_available": True,
        "hbond_lifetime_resolved": False,
        "single_trajectory_descriptive_only": True,
        "new_md_submitted": False,
        "high_density_output_plan_status": "REVIEW_ONLY_NOT_SUBMITTED",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Stage-B3 dynamic TPCL anchor synthesis\n\n"
        "Leading and trailing edges are the single local arc nearest the driven axis. "
        "The footprint interior excludes the two-sided TPCL band. Surface H-bonds are "
        "normalized by contact water, local contact-line length, and accessible surface "
        "sites using count-weighted denominators.\n\n"
        "Slow, acceleration, and fast windows are selected from TPCL kinematics before "
        "network analysis and are reported as dwell, startup, and advance candidates. "
        "The labels are candidates, not causal state assignments.\n\n"
        "Surface-water anchor identity and water-water network identity are separated. "
        "Retention is measured only between 10 ps snapshots and is not an H-bond lifetime, "
        "sub-picosecond rate, friction coefficient, or independent-replica uncertainty.\n\n"
        "The single-TPCL analysis applies to ch3_only and mixed291 finite droplets. "
        "mixed275 has multiple water islands and oh_only is a periodic spread film; their "
        "morphology-specific analyses remain separate.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary


def main() -> int:
    """Command-line entry point for immutable cluster packages."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_contract(
        args.contract,
        args.output,
        anchor_root=args.anchor_root,
    )
    print(args.output.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
