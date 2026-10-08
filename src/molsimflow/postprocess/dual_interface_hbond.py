"""Matched-gap water H-bond connectivity across two facing bubble interfaces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records, minimum_image_vectors
from molsimflow.postprocess.dual_interface_water import (
    CaseSpec,
    DualInterfaceConfig,
    _bootstrap_ratio,
    _positions_by_type,
    _trace_rows,
    _write_csv,
    assign_intact_waters,
    gap_window_label,
    local_basis,
    parse_gap_window,
    read_case_manifest,
)


@dataclass(frozen=True)
class HbondConfig:
    gap_windows_A: Tuple[Tuple[float, float], ...] = ((4.0, 6.0), (12.0, 14.0))
    state: str = "separated"
    oxygen_type: int = 2
    hydrogen_type: int = 1
    oh_cutoff_A: float = 1.25
    oo_cutoff_A: float = 3.50
    ha_cutoff_A: float = 2.45
    angle_cutoff_deg: float = 150.0
    rho_max_A: float = 6.0
    side_layer_cap_A: float = 3.0
    block_ns: float = 0.020
    bootstrap_samples: int = 1000
    random_seed: int = 20260821
    max_frames_per_window: int = 0

    def validate(self) -> None:
        if not self.gap_windows_A or any(right <= left for left, right in self.gap_windows_A):
            raise ValueError("Invalid gap windows")
        if min(self.oh_cutoff_A, self.oo_cutoff_A, self.ha_cutoff_A, self.rho_max_A, self.side_layer_cap_A, self.block_ns) <= 0:
            raise ValueError("Distance and block parameters must be positive")
        if not 0 < self.angle_cutoff_deg < 180:
            raise ValueError("H-bond angle cutoff must be between 0 and 180 degrees")


@dataclass(frozen=True)
class HbondCandidate:
    donor_id: int
    hydrogen_id: int
    acceptor_id: int
    donor_acceptor_distance_A: float
    hydrogen_acceptor_distance_A: float
    dha_angle_deg: float


def classify_layer(s_A: float, gap_A: float, side_layer_cap_A: float) -> str:
    """Assign non-overlapping A, central, and B layers inside nominal surfaces."""

    if gap_A <= 0:
        raise ValueError("Layer classification requires a positive nominal gap")
    left, right = -0.5 * gap_A, 0.5 * gap_A
    thickness = min(float(side_layer_cap_A), gap_A / 3.0)
    if s_A <= left + thickness:
        return "A"
    if s_A >= right - thickness:
        return "B"
    return "central"


def hbond_candidates(
    oxygen_ids: Sequence[int],
    oxygen_positions: np.ndarray,
    assignments: Sequence[Tuple[int, int]],
    hydrogen_ids: Sequence[int],
    hydrogen_positions: np.ndarray,
    bounds: np.ndarray,
    max_oo_A: float = 3.7,
    max_ha_A: float = 2.65,
    min_angle_deg: float = 140.0,
) -> List[HbondCandidate]:
    """Enumerate directed water-water H-bond candidates under loose bounds."""

    lengths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    output: List[HbondCandidate] = []
    for donor_index, donor_id in enumerate(oxygen_ids):
        h_indices = assignments[donor_index]
        if h_indices[0] < 0:
            continue
        donor = oxygen_positions[donor_index]
        for acceptor_index, acceptor_id in enumerate(oxygen_ids):
            if donor_index == acceptor_index:
                continue
            acceptor = oxygen_positions[acceptor_index]
            oo_vec = minimum_image_vectors(acceptor - donor, lengths)
            oo = float(np.linalg.norm(oo_vec))
            if oo > max_oo_A:
                continue
            for hydrogen_index in h_indices:
                hydrogen = hydrogen_positions[hydrogen_index]
                hd = minimum_image_vectors(donor - hydrogen, lengths)
                ha_vec = minimum_image_vectors(acceptor - hydrogen, lengths)
                hd_norm, ha = float(np.linalg.norm(hd)), float(np.linalg.norm(ha_vec))
                if hd_norm <= 1e-12 or ha <= 1e-12 or ha > max_ha_A:
                    continue
                cosine = float(np.dot(hd, ha_vec) / (hd_norm * ha))
                angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                if angle < min_angle_deg:
                    continue
                output.append(
                    HbondCandidate(
                        donor_id=int(donor_id), hydrogen_id=int(hydrogen_ids[hydrogen_index]),
                        acceptor_id=int(acceptor_id), donor_acceptor_distance_A=oo,
                        hydrogen_acceptor_distance_A=ha, dha_angle_deg=angle,
                    )
                )
    return output


def select_edges(
    candidates: Iterable[HbondCandidate],
    oo_cutoff_A: float,
    ha_cutoff_A: float,
    angle_cutoff_deg: float,
) -> Dict[Tuple[int, int], HbondCandidate]:
    """Select one auditable directed record for every undirected graph edge."""

    selected: Dict[Tuple[int, int], HbondCandidate] = {}
    for item in candidates:
        if item.donor_acceptor_distance_A > oo_cutoff_A or item.hydrogen_acceptor_distance_A > ha_cutoff_A or item.dha_angle_deg < angle_cutoff_deg:
            continue
        edge = tuple(sorted((item.donor_id, item.acceptor_id)))
        previous = selected.get(edge)
        if previous is None or (item.hydrogen_acceptor_distance_A, -item.dha_angle_deg) < (previous.hydrogen_acceptor_distance_A, -previous.dha_angle_deg):
            selected[edge] = item
    return selected


def graph_metrics(nodes: Set[int], edges: Iterable[Tuple[int, int]], layers: Mapping[int, str]) -> Dict[str, object]:
    adjacency = {node: set() for node in nodes}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    components: List[Set[int]] = []
    seen: Set[int] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: Set[int] = set()
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    largest = max(components, key=len, default=set())
    spanning_components = [component for component in components if any(layers.get(node) == "A" for node in component) and any(layers.get(node) == "B" for node in component)]
    spanning_component = max(spanning_components, key=len, default=set())
    a_nodes = {node for node in nodes if layers.get(node) == "A"}
    b_nodes = {node for node in nodes if layers.get(node) == "B"}
    shortest = math.nan
    if a_nodes and b_nodes:
        queue = deque((node, 0) for node in a_nodes)
        visited = set(a_nodes)
        while queue:
            node, distance = queue.popleft()
            if node in b_nodes:
                shortest = float(distance)
                break
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    central_dual = sum(
        1
        for node in nodes
        if layers.get(node) == "central"
        and any(layers.get(neighbor) == "A" for neighbor in adjacency[node])
        and any(layers.get(neighbor) == "B" for neighbor in adjacency[node])
    )
    return {
        "largest_component_size": len(largest),
        "largest_component_fraction": len(largest) / len(nodes) if nodes else 0.0,
        "spanning_indicator": int(bool(spanning_component)),
        "shortest_A_B_path_edges": shortest,
        "central_dual_connected_count": central_dual,
        "A_participation_fraction": len(a_nodes & spanning_component) / len(a_nodes) if a_nodes else math.nan,
        "B_participation_fraction": len(b_nodes & spanning_component) / len(b_nodes) if b_nodes else math.nan,
        "average_degree": sum(len(value) for value in adjacency.values()) / len(nodes) if nodes else 0.0,
        "component_by_node": {node: index for index, component in enumerate(components) for node in component},
    }


def _cl_access_times(path: Optional[Path], selected_times_ns: Set[float]) -> Set[float]:
    if path is None or not str(path) or not path.exists():
        return set()
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dual-interface-hbond requires pandas for Cl-access classification") from exc
    output: Set[int] = set()
    columns = ["time_ns", "species_canonical", "species_charge_e", "in_bridge_region"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200000):
        mask = (
            chunk["time_ns"].round(9).isin(selected_times_ns)
            & chunk["species_canonical"].astype(str).eq("Cl_minus")
            & (pd.to_numeric(chunk["species_charge_e"], errors="coerce") == -1.0)
            & (pd.to_numeric(chunk["in_bridge_region"], errors="coerce") == 1)
        )
        output.update(round(float(value), 9) for value in chunk.loc[mask, "time_ns"])
    return output


def _summaries(frame_rows: Sequence[Mapping[str, object]], config: HbondConfig) -> List[Dict[str, object]]:
    rng = np.random.default_rng(config.random_seed)
    metrics = [
        "spanning_indicator", "largest_component_fraction", "shortest_A_B_path_edges",
        "central_dual_connected_count", "A_participation_fraction", "B_participation_fraction",
    ]
    groups = []
    cases = sorted({str(row["case_label"]) for row in frame_rows})
    windows = [gap_window_label(window) for window in config.gap_windows_A]
    for case in cases:
        for window in windows:
            groups.append((case, window, "all", [row for row in frame_rows if row["case_label"] == case and row["gap_window"] == window]))
            if case == "TiO2-HCl-S":
                for flag in (0, 1):
                    groups.append((case, window, f"cl_access_{'positive' if flag else 'unflagged'}", [row for row in frame_rows if row["case_label"] == case and row["gap_window"] == window and int(row["cl_access_positive"]) == flag]))
    output = []
    for case, window, subgroup, rows in groups:
        if not rows:
            continue
        blocks = sorted({int(math.floor(float(row["time_ns"]) / config.block_ns + 1e-10)) for row in rows})
        result: Dict[str, object] = {
            "case_label": case, "gap_window": window, "subgroup": subgroup,
            "frame_count": len(rows), "effective_block_count": len(blocks),
            "valid_water_count": sum(int(row["n_valid_intact_water"]) for row in rows),
            "h_assignment_failure_count": sum(int(row["n_mapping_rejected"]) for row in rows),
        }
        for metric in metrics:
            block_sums, block_counts = [], []
            values = []
            for block in blocks:
                block_values = []
                for row in rows:
                    if int(math.floor(float(row["time_ns"]) / config.block_ns + 1e-10)) != block:
                        continue
                    value = float(row[metric])
                    if math.isfinite(value):
                        block_values.append(value)
                        values.append(value)
                block_sums.append(sum(block_values))
                block_counts.append(len(block_values))
            sums = np.asarray(block_sums, dtype=float)
            counts = np.asarray(block_counts, dtype=float)
            mean = float(np.mean(values)) if values else math.nan
            low, high = _bootstrap_ratio(sums, counts, config.bootstrap_samples, rng)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
            result[f"{metric}_support"] = len(values)
        output.append(result)
    return output


def _representatives(frame_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    output = []
    cases = sorted({str(row["case_label"]) for row in frame_rows})
    windows = sorted({str(row["gap_window"]) for row in frame_rows})
    for case in cases:
        for window in windows:
            rows = [row for row in frame_rows if row["case_label"] == case and row["gap_window"] == window]
            if not rows:
                continue
            probability = float(np.mean([int(row["spanning_indicator"]) for row in rows]))
            modal = int(probability >= 0.5)
            candidates = [row for row in rows if int(row["spanning_indicator"]) == modal] or rows
            target = float(np.median([float(row["largest_component_fraction"]) for row in candidates]))
            chosen = min(candidates, key=lambda row: (abs(float(row["largest_component_fraction"]) - target), float(row["time_ns"])))
            output.append(
                {
                    "case_label": case, "gap_window": window, "segment": chosen["segment"],
                    "local_frame": chosen["local_frame"], "time_ns": chosen["time_ns"],
                    "nominal_gap_A": chosen["nominal_gap_A"], "spanning_indicator": chosen["spanning_indicator"],
                    "largest_component_fraction": chosen["largest_component_fraction"],
                    "selection_rule": "modal spanning class; nearest median largest-component fraction; earliest-time tie break",
                }
            )
    return output


def analyze_case(case: CaseSpec, output_dir: Path, config: HbondConfig, ion_position_csv: Optional[Path] = None) -> Dict[str, Path]:
    config.validate()
    water_config = DualInterfaceConfig(
        gap_windows_A=config.gap_windows_A,
        nominal_radius_a_A=case.nominal_radius_a_A,
        nominal_radius_b_A=case.nominal_radius_b_A,
        state=config.state,
        oh_cutoff_A=config.oh_cutoff_A,
        max_frames_per_window=config.max_frames_per_window,
    )
    trace_lookup = _trace_rows(case.trace_csv, water_config)
    selected_times = {round(float(row["time_ns"]), 9) for row in trace_lookup.values()}
    cl_access_times = _cl_access_times(ion_position_csv, selected_times) if case.case_label == "TiO2-HCl-S" else set()
    frame_rows: List[Dict[str, object]] = []
    node_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    sensitivity_rows: List[Dict[str, object]] = []
    seen = set()
    global_counter = -1
    for segment_label, trajectory in case.segments:
        if not trajectory.exists():
            raise FileNotFoundError(trajectory)
        for frame in iter_lammps_dump_records(trajectory):
            global_counter += 1
            key = (segment_label, frame.frame_index)
            trace = trace_lookup.get(key)
            if trace is None:
                continue
            seen.add(key)
            positions = _positions_by_type(frame, {config.oxygen_type, config.hydrogen_type})
            oxygen_ids_all, oxygen_all = positions[config.oxygen_type]
            hydrogen_ids, hydrogen = positions[config.hydrogen_type]
            center_a = np.asarray([float(trace[f"bubble_A_center_{dim}_A"]) for dim in "xyz"])
            center_b = np.asarray([float(trace[f"bubble_B_center_{dim}_A"]) for dim in "xyz"])
            midpoint, e_s, e_u, e_z = local_basis(center_a, center_b, frame.bounds)
            lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
            delta = minimum_image_vectors(oxygen_all - midpoint, lengths)
            s_all, u_all, z_all = delta @ e_s, delta @ e_u, delta @ e_z
            rho_all = np.sqrt(u_all * u_all + z_all * z_all)
            gap = float(trace["nominal_gap_A"])
            in_confined = (rho_all < config.rho_max_A) & (s_all >= -0.5 * gap) & (s_all <= 0.5 * gap)
            assignments_all, status_all = assign_intact_waters(oxygen_all, hydrogen, frame.bounds, config.oh_cutoff_A)
            valid_indices = [int(index) for index in np.where(in_confined)[0] if assignments_all[int(index)][0] >= 0]
            oxygen_ids = oxygen_ids_all[valid_indices]
            oxygen = oxygen_all[valid_indices]
            assignments = [assignments_all[index] for index in valid_indices]
            s = s_all[valid_indices]
            u = u_all[valid_indices]
            z = z_all[valid_indices]
            layers = {int(atom_id): classify_layer(float(value), gap, config.side_layer_cap_A) for atom_id, value in zip(oxygen_ids, s)}
            candidates = hbond_candidates(oxygen_ids, oxygen, assignments, hydrogen_ids, hydrogen, frame.bounds)
            selected = select_edges(candidates, config.oo_cutoff_A, config.ha_cutoff_A, config.angle_cutoff_deg)
            metrics = graph_metrics(set(int(value) for value in oxygen_ids), selected, layers)
            global_frame = int(round(float(trace["time_ns"]) * 1000.0))
            cl_flag: object = int(round(float(trace["time_ns"]), 9) in cl_access_times) if case.case_label == "TiO2-HCl-S" else ""
            base = {
                "case_label": case.case_label, "segment": segment_label, "local_frame": frame.frame_index,
                "global_frame": global_frame, "timestep": frame.timestep, "time_ns": float(trace["time_ns"]),
                "gap_window": trace["gap_window"], "nominal_gap_A": gap, "cl_access_positive": cl_flag,
            }
            frame_row = {
                **base,
                "n_oxygen_candidates": int(np.count_nonzero(in_confined)),
                "n_valid_intact_water": len(valid_indices),
                "n_mapping_rejected": int(np.count_nonzero(in_confined & (status_all != 2))),
                "n_shared_h_rejected": int(np.count_nonzero(in_confined & (status_all == -1))),
                "n_A_water": sum(layer == "A" for layer in layers.values()),
                "n_central_water": sum(layer == "central" for layer in layers.values()),
                "n_B_water": sum(layer == "B" for layer in layers.values()),
                "n_hbond_edges": len(selected),
                **{key: value for key, value in metrics.items() if key != "component_by_node"},
            }
            frame_rows.append(frame_row)
            component_by_node = metrics["component_by_node"]
            for atom_id, sv, uv, zv in zip(oxygen_ids, s, u, z):
                node_rows.append({**base, "oxygen_id": int(atom_id), "s_A": float(sv), "u_A": float(uv), "z_mid_A": float(zv), "layer": layers[int(atom_id)], "component_id": component_by_node[int(atom_id)]})
            for edge, item in selected.items():
                edge_rows.append({**base, "node1_id": edge[0], "node2_id": edge[1], "donor_id": item.donor_id, "hydrogen_id": item.hydrogen_id, "acceptor_id": item.acceptor_id, "donor_acceptor_distance_A": item.donor_acceptor_distance_A, "hydrogen_acceptor_distance_A": item.hydrogen_acceptor_distance_A, "DHA_angle_deg": item.dha_angle_deg})
            for label, oo, ha, angle in (("strict", 3.3, 2.25, 160.0), ("default", 3.5, 2.45, 150.0), ("loose", 3.7, 2.65, 140.0)):
                for layer_cap in (2.5, 3.0, 3.5):
                    trial_layers = {int(atom_id): classify_layer(float(value), gap, layer_cap) for atom_id, value in zip(oxygen_ids, s)}
                    trial_edges = select_edges(candidates, oo, ha, angle)
                    trial = graph_metrics(set(int(value) for value in oxygen_ids), trial_edges, trial_layers)
                    sensitivity_rows.append({**base, "hbond_cutoff_label": label, "oo_cutoff_A": oo, "ha_cutoff_A": ha, "angle_cutoff_deg": angle, "side_layer_cap_A": layer_cap, "spanning_indicator": trial["spanning_indicator"], "largest_component_fraction": trial["largest_component_fraction"]})
    missing = sorted(set(trace_lookup) - seen)
    if missing:
        raise ValueError(f"{len(missing)} selected trace frames were not found; first={missing[0]}")
    summaries = _summaries(frame_rows, config)
    representatives = _representatives(frame_rows)
    sensitivity_summary = []
    keys = sorted({(row["gap_window"], row["hbond_cutoff_label"], row["side_layer_cap_A"]) for row in sensitivity_rows})
    for window, cutoff, layer_cap in keys:
        rows = [row for row in sensitivity_rows if row["gap_window"] == window and row["hbond_cutoff_label"] == cutoff and row["side_layer_cap_A"] == layer_cap]
        sensitivity_summary.append({"case_label": case.case_label, "gap_window": window, "hbond_cutoff_label": cutoff, "side_layer_cap_A": layer_cap, "frame_count": len(rows), "spanning_probability": float(np.mean([row["spanning_indicator"] for row in rows])), "largest_component_fraction_mean": float(np.mean([row["largest_component_fraction"] for row in rows]))})
    output_dir = Path(output_dir)
    outputs = {
        "frames": output_dir / "hbond_frame_summary.csv",
        "nodes": output_dir / "hbond_node_samples.csv.gz",
        "edges": output_dir / "hbond_edge_samples.csv.gz",
        "summary": output_dir / "hbond_matched_gap_summary.csv",
        "sensitivity": output_dir / "hbond_sensitivity_summary.csv",
        "representatives": output_dir / "representative_microstates.csv",
        "statistics": output_dir / "state_statistics.csv",
        "manifest": output_dir / "artifact_manifest.csv",
    }
    _write_csv(outputs["frames"], frame_rows)
    _write_csv(outputs["nodes"], node_rows)
    _write_csv(outputs["edges"], edge_rows)
    _write_csv(outputs["summary"], summaries)
    _write_csv(outputs["sensitivity"], sensitivity_summary)
    _write_csv(outputs["representatives"], representatives)
    _write_csv(outputs["statistics"], [
        {"metric": "case_label", "value": case.case_label},
        {"metric": "selected_frames", "value": len(frame_rows)},
        {"metric": "valid_water_samples", "value": len(node_rows)},
        {"metric": "hbond_geometry", "value": f"DA<={config.oo_cutoff_A} A; HA<={config.ha_cutoff_A} A; DHA>={config.angle_cutoff_deg} deg"},
        {"metric": "water_definition", "value": f"O with exactly two unique H within {config.oh_cutoff_A} A"},
        {"metric": "confined_region", "value": f"between nominal internal surfaces; rho<{config.rho_max_A} A"},
        {"metric": "layer_definition", "value": f"side thickness=min({config.side_layer_cap_A} A,h/3)"},
    ])
    manifest = []
    for name, path in outputs.items():
        if name == "manifest":
            continue
        manifest.append({"artifact": name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_csv(outputs["manifest"], manifest)
    return outputs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_hbond_manifest(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def assemble_figures(case_manifest: Path, output_root: Path, figure_dir: Path, config: HbondConfig) -> Dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    manifest = _read_hbond_manifest(case_manifest)
    cases = [row["case_label"] for row in manifest]
    summaries = {case: _read_csv(Path(output_root) / case / "hbond_matched_gap_summary.csv") for case in cases}
    nodes = {case: _read_csv(Path(output_root) / case / "hbond_node_samples.csv.gz") for case in cases}
    edges = {case: _read_csv(Path(output_root) / case / "hbond_edge_samples.csv.gz") for case in cases}
    representatives = {case: _read_csv(Path(output_root) / case / "representative_microstates.csv") for case in cases}
    sensitivity = {case: _read_csv(Path(output_root) / case / "hbond_sensitivity_summary.csv") for case in cases}
    plot_dir = Path(figure_dir) / "plot_data"
    summary_csv = plot_dir / "multicase_hbond_matched_gap_summary.csv"
    sensitivity_csv = plot_dir / "multicase_hbond_sensitivity_summary.csv"
    representative_csv = plot_dir / "multicase_representative_microstates.csv"
    _write_csv(summary_csv, [{"case_label": case, **row} for case in cases for row in summaries[case]])
    _write_csv(sensitivity_csv, [{"case_label": case, **row} for case in cases for row in sensitivity[case]])
    _write_csv(representative_csv, [{"case_label": case, **row} for case in cases for row in representatives[case]])

    windows = [gap_window_label(window) for window in config.gap_windows_A]
    colors = ["#2166ac", "#b2182b"]
    metrics = [
        ("spanning_indicator", "Spanning probability"),
        ("largest_component_fraction", "Largest-component fraction"),
        ("shortest_A_B_path_edges", "Shortest A-to-B path (edges)"),
        ("central_dual_connected_count", "Central waters linked to A and B"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    x = np.arange(len(cases), dtype=float)
    width = 0.34
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for wi, (window, color) in enumerate(zip(windows, colors)):
            rows = []
            for case in cases:
                matches = [row for row in summaries[case] if row["gap_window"] == window and row["subgroup"] == "all"]
                rows.append(matches[0])
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
            high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
            errors = np.vstack([mean - low, high - mean])
            errors[:, ~np.isfinite(errors).all(axis=0)] = 0.0
            ax.bar(x + (wi - 0.5) * width, mean, width=width, color=color, alpha=0.82, label=f"$h={window[:-1]}$ Å", yerr=errors, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(frameon=False)
    summary_figure = Path(figure_dir) / "candidate_hbond_spanning_p0.png"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(summary_figure, dpi=300)
    plt.close(fig)

    layer_colors = {"A": "#2166ac", "central": "#bdbdbd", "B": "#b2182b"}
    fig, axes = plt.subplots(len(windows), len(cases), figsize=(3.1 * len(cases), 3.0 * len(windows)), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)
    for i, window in enumerate(windows):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            rep = next(row for row in representatives[case] if row["gap_window"] == window)
            key = (rep["segment"], int(rep["local_frame"]))
            nrows = [row for row in nodes[case] if (row["segment"], int(row["local_frame"])) == key]
            erows = [row for row in edges[case] if (row["segment"], int(row["local_frame"])) == key]
            lookup = {int(row["oxygen_id"]): row for row in nrows}
            for edge in erows:
                a, b = int(edge["node1_id"]), int(edge["node2_id"])
                if a in lookup and b in lookup:
                    ax.plot([float(lookup[a]["s_A"]), float(lookup[b]["s_A"])], [float(lookup[a]["z_mid_A"]), float(lookup[b]["z_mid_A"])], color="0.25", linewidth=0.8, alpha=0.75, zorder=1)
            for layer, color in layer_colors.items():
                rows = [row for row in nrows if row["layer"] == layer]
                ax.scatter([float(row["s_A"]) for row in rows], [float(row["z_mid_A"]) for row in rows], s=24, color=color, edgecolor="white", linewidth=0.3, zorder=2)
            gap = float(rep["nominal_gap_A"])
            ax.axvline(-0.5 * gap, color="0.35", linestyle="--", linewidth=0.7)
            ax.axvline(0.5 * gap, color="0.35", linestyle="--", linewidth=0.7)
            ax.set_xlim(-7.5, 7.5)
            ax.set_ylim(-6.5, 6.5)
            ax.text(0.03, 0.96, f"span={int(rep['spanning_indicator'])}; $f_{{max}}$={float(rep['largest_component_fraction']):.2f}", transform=ax.transAxes, va="top", fontsize=7)
            if i == 0:
                ax.set_title(case)
            if j == 0:
                ax.set_ylabel(f"$h={window[:-1]}$ Å\n$z_{{mid}}$ (Å)")
            if i == len(windows) - 1:
                ax.set_xlabel("$s$ (Å)")
    fig.legend(
        handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=layer, markersize=7) for layer, color in layer_colors.items()],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
    )
    microstate_figure = Path(figure_dir) / "candidate_hbond_spanning_microstates.png"
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.92, wspace=0.12, hspace=0.18)
    fig.savefig(microstate_figure, dpi=300)
    plt.close(fig)
    return {
        "summary_figure": summary_figure, "microstate_figure": microstate_figure,
        "summary_plot_data": summary_csv, "sensitivity_plot_data": sensitivity_csv,
        "representative_plot_data": representative_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched-gap dual-interface water H-bond connectivity")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze-case")
    analyze.add_argument("--case-manifest", type=Path, required=True)
    group = analyze.add_mutually_exclusive_group()
    group.add_argument("--case-index", type=int)
    group.add_argument("--case-label")
    analyze.add_argument("--output-root", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--case-manifest", type=Path, required=True)
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--figure-dir", type=Path, required=True)
    for target in (analyze, assemble):
        target.add_argument("--gap-window", action="append", default=[])
        target.add_argument("--block-ns", type=float, default=0.020)
        target.add_argument("--bootstrap-samples", type=int, default=1000)
    analyze.add_argument("--max-frames-per-window", type=int, default=0)
    analyze.add_argument("--oh-cutoff-A", type=float, default=1.25)
    analyze.add_argument("--hbond-oo-cutoff-A", type=float, default=3.50)
    analyze.add_argument("--hbond-ha-cutoff-A", type=float, default=2.45)
    analyze.add_argument("--hbond-angle-cutoff-deg", type=float, default=150.0)
    analyze.add_argument("--rho-max-A", type=float, default=6.0)
    analyze.add_argument("--side-layer-cap-A", type=float, default=3.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    windows = tuple(parse_gap_window(item) for item in args.gap_window) or ((4.0, 6.0), (12.0, 14.0))
    config = HbondConfig(
        gap_windows_A=windows, block_ns=args.block_ns, bootstrap_samples=args.bootstrap_samples,
        max_frames_per_window=getattr(args, "max_frames_per_window", 0),
        oh_cutoff_A=getattr(args, "oh_cutoff_A", 1.25),
        oo_cutoff_A=getattr(args, "hbond_oo_cutoff_A", 3.50),
        ha_cutoff_A=getattr(args, "hbond_ha_cutoff_A", 2.45),
        angle_cutoff_deg=getattr(args, "hbond_angle_cutoff_deg", 150.0),
        rho_max_A=getattr(args, "rho_max_A", 6.0),
        side_layer_cap_A=getattr(args, "side_layer_cap_A", 3.0),
    )
    if args.command == "analyze-case":
        case = read_case_manifest(args.case_manifest, args.case_index, args.case_label)
        rows = _read_hbond_manifest(args.case_manifest)
        row = next(item for item in rows if item["case_label"] == case.case_label)
        ion_path = Path(row["ion_position_csv"]) if row.get("ion_position_csv", "").strip() else None
        outputs = analyze_case(case, Path(args.output_root) / case.case_label, config, ion_path)
    else:
        outputs = assemble_figures(args.case_manifest, args.output_root, args.figure_dir, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
