"""Synthesize Stage C replica transport, event, and species-layer diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)

RESPONSE_COMPONENTS = ("Jx_Fx_mps", "Jx_Fy_mps", "Jy_Fx_mps", "Jy_Fy_mps")
EVENT_METRICS = (
    "split_event_count",
    "merge_event_count",
    "persistent_transfer_count",
    "lineage_reassignment_count",
)
TRANSPORT_METRICS = (
    "response_total_velocity_mps",
    "absolute_response_total_velocity_mps",
    "response_persistent_island_transfer_velocity_mps",
    "response_lineage_reassignment_velocity_mps",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _finite(value: object) -> float:
    return float(str(value))


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _pearson(left: Sequence[float], right: Sequence[float]) -> tuple[float, int]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    x = np.asarray([item[0] for item in pairs], dtype=float)
    y = np.asarray([item[1] for item in pairs], dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(x, y)[0, 1]), len(pairs)


def _same_sign(left: float, right: float, tolerance: float = 1.0e-12) -> str:
    if abs(left) <= tolerance or abs(right) <= tolerance:
        return "ZERO_UNRESOLVED"
    return "SAME" if left * right > 0.0 else "OPPOSITE"


def replica_response_comparison(
    stage_b_rows: Sequence[Mapping[str, str]],
    replica_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Compare Stage B and replica02 response matrices without inferential claims."""

    stage_b = {str(row["case_id"]): row for row in stage_b_rows}
    replica = {str(row["case_id"]): row for row in replica_rows}
    output: list[dict[str, object]] = []
    for case_id in sorted(stage_b.keys() & replica.keys()):
        left = np.asarray([_finite(stage_b[case_id][name]) for name in RESPONSE_COMPONENTS])
        right = np.asarray([_finite(replica[case_id][name]) for name in RESPONSE_COMPONENTS])
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        cosine = (
            float(np.dot(left, right) / (left_norm * right_norm))
            if left_norm > 0.0 and right_norm > 0.0
            else float("nan")
        )
        item: dict[str, object] = {
            "case_id": case_id,
            "stage_b_frobenius_norm_mps": left_norm,
            "replica02_frobenius_norm_mps": right_norm,
            "replica_to_stage_b_norm_ratio": (
                right_norm / left_norm if left_norm > 0.0 else float("nan")
            ),
            "matrix_cosine_similarity": cosine,
            "longitudinal_same_sign_count": sum(
                _same_sign(left[index], right[index]) == "SAME" for index in (0, 3)
            ),
            "lateral_same_sign_count": sum(
                _same_sign(left[index], right[index]) == "SAME" for index in (1, 2)
            ),
        }
        for index, name in enumerate(RESPONSE_COMPONENTS):
            label = name.removesuffix("_mps")
            item[f"stage_b_{label}_mps"] = left[index]
            item[f"replica02_{label}_mps"] = right[index]
            item[f"{label}_sign_replication"] = _same_sign(left[index], right[index])
        output.append(item)
    return output


def mixed_event_conditioned_blocks(
    flux_rows: Sequence[Mapping[str, str]],
    lineage_rows: Sequence[Mapping[str, str]],
    exchange_rows: Sequence[Mapping[str, str]],
    *,
    block_ps: float,
    full_window_ps: float,
) -> list[dict[str, object]]:
    """Join topology and membership events to accepted transport blocks."""

    event_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in lineage_rows:
        time_ps = _finite(row["time_ps"])
        if time_ps > full_window_ps + 1.0e-9:
            continue
        key = (str(row["branch_id"]), int(math.floor(time_ps / block_ps + 1.0e-12)))
        event_type = str(row["event_type"]).upper()
        if event_type == "SPLIT":
            event_counts[key]["split_event_count"] += 1
        elif event_type == "MERGE":
            event_counts[key]["merge_event_count"] += 1
    for row in exchange_rows:
        time_ps = _finite(row["time_ps"])
        if time_ps > full_window_ps + 1.0e-9:
            continue
        key = (str(row["branch_id"]), int(math.floor(time_ps / block_ps + 1.0e-12)))
        exchange_class = str(row["exchange_class"])
        if exchange_class == "PERSISTENT_ISLAND_TRANSFER":
            event_counts[key]["persistent_transfer_count"] += 1
        elif "LINEAGE_REASSIGNMENT" in exchange_class:
            event_counts[key]["lineage_reassignment_count"] += 1
    output: list[dict[str, object]] = []
    for row in flux_rows:
        if not math.isclose(_finite(row["window_ps"]), block_ps):
            continue
        start_ps = _finite(row["start_ps"])
        if start_ps >= full_window_ps - 1.0e-9:
            continue
        branch_id = str(row["branch_id"])
        block_index = int(row["window_index"])
        counts = event_counts[(branch_id, block_index)]
        total = _finite(row["response_total_velocity_mps"])
        item: dict[str, object] = {
            "case_id": row["case_id"],
            "branch_id": branch_id,
            "direction": row["direction"],
            "block_index": block_index,
            "start_ps": start_ps,
            "nominal_end_ps": row["end_ps"],
            "actual_end_ps": start_ps + _finite(row["duration_ps"]),
            "duration_ps": row["duration_ps"],
            "response_total_velocity_mps": total,
            "absolute_response_total_velocity_mps": abs(total),
            "response_persistent_island_transfer_velocity_mps": row[
                "response_persistent_island_transfer_velocity_mps"
            ],
            "response_lineage_reassignment_velocity_mps": row[
                "response_lineage_reassignment_velocity_mps"
            ],
        }
        for name in EVENT_METRICS:
            item[name] = counts[name]
        item["has_topology_event"] = int(
            counts["split_event_count"] + counts["merge_event_count"] > 0
        )
        item["has_persistent_transfer"] = int(counts["persistent_transfer_count"] > 0)
        output.append(item)
    return output


def mixed_event_transport_correlations(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for branch_id in sorted({str(row["branch_id"]) for row in rows}):
        selected = [row for row in rows if str(row["branch_id"]) == branch_id]
        for event_metric in EVENT_METRICS:
            x = [_finite(row[event_metric]) for row in selected]
            event_mask = [value > 0.0 for value in x]
            for transport_metric in TRANSPORT_METRICS:
                y = [_finite(row[transport_metric]) for row in selected]
                pearson, support = _pearson(x, y)
                with_event = [value for value, present in zip(y, event_mask) if present]
                without_event = [value for value, present in zip(y, event_mask) if not present]
                mean_with = _mean(with_event)
                mean_without = _mean(without_event)
                output.append(
                    {
                        "branch_id": branch_id,
                        "event_metric": event_metric,
                        "transport_metric": transport_metric,
                        "pearson_r": pearson,
                        "support_blocks": support,
                        "event_blocks": len(with_event),
                        "no_event_blocks": len(without_event),
                        "mean_transport_with_event": mean_with,
                        "mean_transport_without_event": mean_without,
                        "conditioned_difference": mean_with - mean_without,
                        "evidence_limit": "single_trajectory_block_association_not_causality",
                    }
                )
    return output


def _species_block_means(
    rows: Sequence[Mapping[str, str]], block_ps: float, full_window_ps: float
) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        time_ps = _finite(row["time_ps"])
        if time_ps > full_window_ps + 1.0e-9:
            continue
        key = (str(row["branch_id"]), int(math.floor(time_ps / block_ps + 1.0e-12)))
        for name in ("solution_H3O", "framework_OH", "proton_partition_pool"):
            grouped[key][name].append(_finite(row[name]))
    return {
        key: {name: _mean(values) for name, values in columns.items()}
        for key, columns in grouped.items()
    }


def oh_partition_layer_blocks(
    species_rows: Sequence[Mapping[str, str]],
    layer_rows: Sequence[Mapping[str, str]],
    film_rows: Sequence[Mapping[str, str]],
    *,
    selected_layers: Sequence[int],
    block_ps: float,
    full_window_ps: float,
) -> list[dict[str, object]]:
    """Align species excursions with layer-resolved cancellation blocks."""

    species = _species_block_means(species_rows, block_ps, full_window_ps)
    layer_lookup = {
        (str(row["branch_id"]), int(row["block_index"]), int(row["layer_index"])): row
        for row in layer_rows
    }
    film_lookup = {
        (str(row["branch_id"]), int(row["block_index"])): row for row in film_rows
    }
    baseline_branch = next(
        str(row["branch_id"]) for row in film_rows if str(row["direction"]) == "none"
    )
    output: list[dict[str, object]] = []
    driven = sorted(
        {
            (str(row["branch_id"]), str(row["direction"]))
            for row in film_rows
            if str(row["direction"]) != "none"
        }
    )
    for branch_id, direction in driven:
        block_indices = sorted(
            block
            for candidate, block in species
            if candidate == branch_id and block * block_ps < full_window_ps - 1.0e-9
        )
        for block_index in block_indices:
            current = species[(branch_id, block_index)]
            baseline = species.get((baseline_branch, block_index))
            film = film_lookup.get((branch_id, block_index))
            if baseline is None or film is None:
                continue
            weighted_signed = 0.0
            weighted_absolute = 0.0
            total_weight = 0.0
            layer_values: dict[int, float] = {}
            item: dict[str, object] = {
                "case_id": "oh_only",
                "branch_id": branch_id,
                "direction": direction,
                "block_index": block_index,
                "start_ps": block_index * block_ps,
                "actual_end_ps": min((block_index + 1) * block_ps, full_window_ps),
                "mean_solution_H3O": current["solution_H3O"],
                "baseline_solution_H3O": baseline["solution_H3O"],
                "excess_solution_H3O": current["solution_H3O"]
                - baseline["solution_H3O"],
                "mean_framework_OH": current["framework_OH"],
                "baseline_framework_OH": baseline["framework_OH"],
                "excess_framework_OH": current["framework_OH"]
                - baseline["framework_OH"],
                "proton_partition_pool": current["proton_partition_pool"],
                "film_excess_velocity_mps": film["total_excess_velocity_mps"],
            }
            for layer in selected_layers:
                layer_row = layer_lookup.get((branch_id, block_index, int(layer)))
                value = (
                    _finite(layer_row["excess_axis_velocity_mps"])
                    if layer_row is not None
                    else float("nan")
                )
                weight = (
                    _finite(layer_row["mean_count"])
                    if layer_row is not None
                    else 0.0
                )
                item[f"layer_{layer}_excess_velocity_mps"] = value
                layer_values[int(layer)] = value
                if math.isfinite(value) and weight > 0.0:
                    weighted_signed += weight * value
                    weighted_absolute += weight * abs(value)
                    total_weight += weight
            signed_mean = weighted_signed / total_weight if total_weight > 0.0 else float("nan")
            absolute_mean = (
                weighted_absolute / total_weight if total_weight > 0.0 else float("nan")
            )
            item["selected_layer_signed_response_mps"] = signed_mean
            item["selected_layer_absolute_response_mps"] = absolute_mean
            item["selected_layer_cancellation_fraction"] = (
                1.0 - abs(signed_mean) / absolute_mean
                if absolute_mean > 1.0e-12
                else float("nan")
            )
            layer_2 = layer_values.get(2, float("nan"))
            layer_3 = layer_values.get(3, float("nan"))
            item["layer_2_3_opposed"] = int(
                math.isfinite(layer_2) and math.isfinite(layer_3) and layer_2 * layer_3 < 0.0
            )
            output.append(item)
    return output


def oh_partition_layer_coupling(
    rows: Sequence[Mapping[str, object]], selected_layers: Sequence[int]
) -> list[dict[str, object]]:
    y_metrics = [
        "film_excess_velocity_mps",
        "selected_layer_cancellation_fraction",
        *(f"layer_{layer}_excess_velocity_mps" for layer in selected_layers),
    ]
    output: list[dict[str, object]] = []
    for branch_id in sorted({str(row["branch_id"]) for row in rows}):
        selected = [row for row in rows if str(row["branch_id"]) == branch_id]
        x_metrics = ("excess_solution_H3O", "excess_framework_OH")
        for x_metric in x_metrics:
            x = [_finite(row[x_metric]) for row in selected]
            absolute_x = [abs(value) for value in x]
            threshold = float(np.median(absolute_x)) if absolute_x else float("nan")
            high = [value >= threshold for value in absolute_x]
            for y_metric in y_metrics:
                y = [_finite(row[y_metric]) for row in selected]
                pearson, support = _pearson(x, y)
                high_values = [value for value, flag in zip(y, high) if flag]
                low_values = [value for value, flag in zip(y, high) if not flag]
                mean_high = _mean(high_values)
                mean_low = _mean(low_values)
                output.append(
                    {
                        "branch_id": branch_id,
                        "partition_metric": x_metric,
                        "transport_metric": y_metric,
                        "pearson_r": pearson,
                        "support_blocks": support,
                        "median_absolute_excursion": threshold,
                        "high_excursion_blocks": len(high_values),
                        "low_excursion_blocks": len(low_values),
                        "mean_transport_high_excursion": mean_high,
                        "mean_transport_low_excursion": mean_low,
                        "conditioned_difference": mean_high - mean_low,
                        "evidence_limit": "geometric_species_proxy_block_association_not_causality",
                    }
                )
    return output


def _persistent_fraction_lookup(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    return {
        str(row["branch_id"]): abs(_finite(row["absolute_fraction_of_component_l1"]))
        for row in rows
        if str(row["category"]) == "PERSISTENT_ISLAND_TRANSFER"
    }


def _plot(
    response_rows: Sequence[Mapping[str, object]],
    stage_b_categories: Sequence[Mapping[str, str]],
    replica_categories: Sequence[Mapping[str, str]],
    event_correlations: Sequence[Mapping[str, object]],
    layer_correlations: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    labels = [name.removesuffix("_mps") for name in RESPONSE_COMPONENTS]
    x = np.arange(len(labels))
    for axis, case_id in zip(axes[0], ("mixed275", "oh_only")):
        row = next(item for item in response_rows if item["case_id"] == case_id)
        axis.bar(
            x - 0.18,
            [row[f"stage_b_{name.removesuffix('_mps')}_mps"] for name in RESPONSE_COMPONENTS],
            0.36,
            label="Stage B",
        )
        axis.bar(
            x + 0.18,
            [row[f"replica02_{name.removesuffix('_mps')}_mps"] for name in RESPONSE_COMPONENTS],
            0.36,
            label="replica02",
        )
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel("F0-subtracted response (m/s)")
        axis.set_title(case_id)
        axis.axhline(0.0, color="black", lw=0.7)
        axis.legend(frameon=False)

    stage_b_fraction = _persistent_fraction_lookup(stage_b_categories)
    replica_fraction = _persistent_fraction_lookup(replica_categories)
    branches = ["f8e-5_x", "f8e-5_y"]
    axes[1, 0].bar(
        np.arange(2) - 0.18,
        [stage_b_fraction[branch] for branch in branches],
        0.36,
        label="Stage B",
    )
    axes[1, 0].bar(
        np.arange(2) + 0.18,
        [replica_fraction[branch] for branch in branches],
        0.36,
        label="replica02",
    )
    axes[1, 0].set_xticks(np.arange(2), ["X", "Y"])
    axes[1, 0].set_ylabel("Persistent-transfer L1 fraction")
    axes[1, 0].set_title("Exchange contribution")
    axes[1, 0].legend(frameon=False)

    event_values = [
        abs(_finite(row["pearson_r"]))
        for row in event_correlations
        if math.isfinite(_finite(row["pearson_r"]))
    ]
    layer_values = [
        abs(_finite(row["pearson_r"]))
        for row in layer_correlations
        if math.isfinite(_finite(row["pearson_r"]))
    ]
    axes[1, 1].bar(
        ["event/transport", "partition/layer"],
        [max(event_values, default=float("nan")), max(layer_values, default=float("nan"))],
        color=["tab:purple", "tab:green"],
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("Maximum |Pearson r|")
    axes[1, 1].set_title("Exploratory block associations")
    figure.tight_layout()
    figure.savefig(output / "stage_c_mechanism_synthesis.png", dpi=240)
    plt.close(figure)


def _report(
    response_rows: Sequence[Mapping[str, object]],
    stage_b_categories: Sequence[Mapping[str, str]],
    replica_categories: Sequence[Mapping[str, str]],
    event_correlations: Sequence[Mapping[str, object]],
    layer_correlations: Sequence[Mapping[str, object]],
) -> str:
    persistent_b = _persistent_fraction_lookup(stage_b_categories)
    persistent_c = _persistent_fraction_lookup(replica_categories)
    matrix_lines = []
    for row in response_rows:
        matrix_lines.append(
            f"- {row['case_id']}: cosine={_finite(row['matrix_cosine_similarity']):.3f}, "
            f"norm ratio={_finite(row['replica_to_stage_b_norm_ratio']):.3f}, "
            f"longitudinal sign matches={row['longitudinal_same_sign_count']}/2, "
            f"lateral sign matches={row['lateral_same_sign_count']}/2."
        )
    finite_event = [
        row for row in event_correlations if math.isfinite(_finite(row["pearson_r"]))
    ]
    finite_layer = [
        row for row in layer_correlations if math.isfinite(_finite(row["pearson_r"]))
    ]
    strongest_event = max(finite_event, key=lambda row: abs(_finite(row["pearson_r"])))
    strongest_layer = max(finite_layer, key=lambda row: abs(_finite(row["pearson_r"])))
    return (
        "# Stage C replica02 mechanism synthesis\n\n"
        "## Response replication\n\n"
        + "\n".join(matrix_lines)
        + "\n\nThe longitudinal response directions are compared separately from magnitude. "
        "Lateral sign changes or norm changes are treated as non-replication, not averaged away.\n\n"
        "## Persistent-island transfer\n\n"
        f"The L1 fraction is Stage B {persistent_b['f8e-5_x']:.4f}/{persistent_b['f8e-5_y']:.4f} "
        f"and replica02 {persistent_c['f8e-5_x']:.4f}/{persistent_c['f8e-5_y']:.4f} "
        "for X/Y. Exchange remains a small correction rather than the dominant carrier.\n\n"
        "## Exploratory associations\n\n"
        f"The largest finite event/transport block correlation is r={_finite(strongest_event['pearson_r']):.3f} "
        f"for {strongest_event['branch_id']} {strongest_event['event_metric']} versus "
        f"{strongest_event['transport_metric']}. The largest finite species/layer correlation "
        f"is r={_finite(strongest_layer['pearson_r']):.3f} for {strongest_layer['branch_id']} "
        f"{strongest_layer['partition_metric']} versus {strongest_layer['transport_metric']}.\n\n"
        "These are post hoc single-trajectory block associations. Species labels are geometric "
        "nearest-parent proxies, and neither correlation establishes causal proton-mediated "
        "transport, patterned-channel causality, or independent-replica uncertainty.\n"
    )


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    base = contract_path.parent
    full_window_ps = _finite(contract["full_window_ps"])
    block_ps = _finite(contract.get("block_ps", 50.0))
    selected_layers = tuple(int(value) for value in contract.get("selected_layers", [1, 2, 3]))
    paths = {
        name: resolve_path(value, base)
        for name, value in contract["inputs"].items()
        if name != "oh_species_timeseries"
    }
    species_paths = {
        branch: resolve_path(value, base)
        for branch, value in contract["inputs"]["oh_species_timeseries"].items()
    }
    for summary_name in (
        "stage_b_flux_summary",
        "stage_b_layer_summary",
        "stage_b_anisotropy_summary",
        "replica_flux_summary",
        "replica_layer_summary",
        "replica_anisotropy_summary",
    ):
        if json.loads(paths[summary_name].read_text(encoding="utf-8"))["status"] != "PASS":
            raise ValueError(f"Input summary is not PASS: {paths[summary_name]}")

    response_rows = replica_response_comparison(
        _read_tsv(paths["stage_b_response_matrix"]),
        _read_tsv(paths["replica_response_matrix"]),
    )
    event_blocks = mixed_event_conditioned_blocks(
        _read_tsv(paths["replica_time_window_response"]),
        _read_tsv(paths["replica_lineage_events"]),
        _read_tsv(paths["replica_molecule_exchange"]),
        block_ps=block_ps,
        full_window_ps=full_window_ps,
    )
    event_correlations = mixed_event_transport_correlations(event_blocks)
    species_rows = []
    for path in species_paths.values():
        species_rows.extend(_read_tsv(path))
    layer_blocks = oh_partition_layer_blocks(
        species_rows,
        _read_tsv(paths["replica_layer_blocks"]),
        _read_tsv(paths["replica_film_blocks"]),
        selected_layers=selected_layers,
        block_ps=block_ps,
        full_window_ps=full_window_ps,
    )
    layer_correlations = oh_partition_layer_coupling(layer_blocks, selected_layers)
    stage_b_categories = _read_tsv(paths["stage_b_category_response"])
    replica_categories = _read_tsv(paths["replica_category_response"])

    output.mkdir(parents=True)
    write_tsv(output / "replica_response_comparison.tsv", response_rows, tuple(response_rows[0]))
    write_tsv(output / "mixed_event_conditioned_blocks.tsv", event_blocks, tuple(event_blocks[0]))
    write_tsv(
        output / "mixed_event_transport_correlations.tsv",
        event_correlations,
        tuple(event_correlations[0]),
    )
    write_tsv(output / "oh_partition_layer_blocks.tsv", layer_blocks, tuple(layer_blocks[0]))
    write_tsv(
        output / "oh_partition_layer_coupling.tsv",
        layer_correlations,
        tuple(layer_correlations[0]),
    )
    input_paths = {contract_path, *paths.values(), *species_paths.values()}
    input_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(input_paths)
    ]
    write_tsv(output / "input_manifest.tsv", input_rows, ("path", "size_bytes", "sha256"))
    _plot(
        response_rows,
        stage_b_categories,
        replica_categories,
        event_correlations,
        layer_correlations,
        output,
    )
    (output / "STAGE-C-REPORT.md").write_text(
        _report(
            response_rows,
            stage_b_categories,
            replica_categories,
            event_correlations,
            layer_correlations,
        ),
        encoding="utf-8",
    )
    event_values = [
        abs(_finite(row["pearson_r"]))
        for row in event_correlations
        if math.isfinite(_finite(row["pearson_r"]))
    ]
    layer_values = [
        abs(_finite(row["pearson_r"]))
        for row in layer_correlations
        if math.isfinite(_finite(row["pearson_r"]))
    ]
    summary = {
        "status": "PASS",
        "full_window_ps": full_window_ps,
        "response_comparison_rows": len(response_rows),
        "event_conditioned_block_rows": len(event_blocks),
        "event_correlation_rows": len(event_correlations),
        "partition_layer_block_rows": len(layer_blocks),
        "partition_layer_correlation_rows": len(layer_correlations),
        "maximum_absolute_event_transport_pearson_r": max(event_values, default=float("nan")),
        "maximum_absolute_partition_layer_pearson_r": max(layer_values, default=float("nan")),
        "species_definition": "nearest-parent geometric proxy, not formal charge",
        "single_trajectory_descriptive_only": True,
        "causality_established": False,
        "new_md_submitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_output_hashes(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
