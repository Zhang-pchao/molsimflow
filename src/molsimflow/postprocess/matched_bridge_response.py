"""Matched HCl/NaCl bridge-response analysis and candidate-figure rendering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records, minimum_image_vectors
from molsimflow.postprocess.dual_interface_water import _positions_by_type, local_basis
from molsimflow.postprocess.hbond_network import classify_hbond_type


DEFAULT_MEDOID_COLUMNS = (
    "delta_n_solution_o_core",
    "delta_ion_hbond_count",
    "delta_surface_h_xy_within_10A",
    "delta_surface_h_min_xy_A",
    "delta_hbond_network_avg_degree",
    "delta_largest_hbond_component_fraction",
)

METRIC_LABELS = {
    "n_solution_o_core": ("Core solution O", "count"),
    "ion_hbond_count": ("Ion-water H bonds", "count"),
    "surface_h_xy_within_10A": ("Surface H within 10 Å", "count"),
    "surface_h_min_xy_A": ("Minimum surface-H distance", "Å"),
    "hbond_network_avg_degree": ("H-bond degree", "dimensionless"),
    "largest_hbond_component_fraction": ("Largest-component fraction", "dimensionless"),
}

PRIMARY_EDGE_TYPES = {
    "water_water",
    "water_h3o",
    "water_oh",
    "h3o_water",
    "oh_water",
}
ION_EDGE_TYPES = PRIMARY_EDGE_TYPES - {"water_water"} | {"water_cl"}


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_medoid(table: "object", columns: Sequence[str] = DEFAULT_MEDOID_COLUMNS):
    """Return a pair table with deterministic distance to its robust standardized center."""

    import pandas as pd

    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Missing medoid columns: {missing}")
    values = table.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    if int(valid.sum()) < 2:
        raise ValueError("At least two complete matched pairs are required")
    usable = values.loc[valid]
    scale = usable.std(axis=0, ddof=1).replace(0.0, np.nan)
    if scale.isna().any():
        bad = list(scale.index[scale.isna()])
        raise ValueError(f"Medoid descriptors have zero or undefined scale: {bad}")
    standardized = (usable - usable.mean(axis=0)) / scale
    center = standardized.median(axis=0)
    distance = np.sqrt(((standardized - center) ** 2).sum(axis=1))
    selected_index = distance.sort_values(kind="stable").index[0]
    audit = table.copy()
    audit["medoid_distance"] = np.nan
    audit.loc[distance.index, "medoid_distance"] = distance
    audit["medoid_selected"] = False
    audit.loc[selected_index, "medoid_selected"] = True
    return audit, selected_index


def block_bootstrap_ci(
    values: Sequence[float],
    blocks: Sequence[str],
    samples: int = 5000,
    seed: int = 20260605,
) -> Tuple[float, float]:
    """Reproduce the source matched-pair block bootstrap."""

    array = np.asarray(values, dtype=float)
    labels = np.asarray(blocks, dtype=str)
    valid = np.isfinite(array)
    array, labels = array[valid], labels[valid]
    if array.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    if unique.size < 2:
        means = [float(np.mean(rng.choice(array, size=array.size, replace=True))) for _ in range(samples)]
    else:
        grouped = [array[labels == label] for label in unique]
        means = []
        for _ in range(samples):
            chosen = rng.integers(0, len(grouped), size=len(grouped))
            means.append(float(np.mean(np.concatenate([grouped[index] for index in chosen]))))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _segment_config(case: Mapping[str, object], segment: str) -> Mapping[str, object]:
    segments = case.get("segments", {})
    if segment not in segments:
        raise KeyError(f"No configured segment {segment!r} for {case.get('case_label')}")
    return segments[segment]


def analyze(config_path: Path, output_dir: Path, manifest_dir: Path, report_dir: Path) -> Dict[str, Path]:
    """Select the descriptor medoid and assemble audited plot data."""

    import pandas as pd

    config = _read_json(config_path)
    paths = config["paths"]
    pair_path = Path(paths["matched_pair_table"])
    frame_path = Path(paths["matched_frame_table"])
    summary_path = Path(paths["matched_summary_table"])
    charge_path = Path(paths["charge_control_table"])
    barrier_path = Path(paths["barrier_table"])
    pair = pd.read_csv(pair_path)
    frame = pd.read_csv(frame_path)
    source_summary = pd.read_csv(summary_path)
    if len(pair) != int(config.get("expected_pair_count", 24)):
        raise ValueError(f"Expected 24 matched pairs, found {len(pair)}")
    if float(pair["abs_gap_delta_A"].max()) > float(config["matching"]["gap_tolerance_A"]) + 1e-12:
        raise ValueError("Matched-pair gap tolerance is violated")
    if float(pair["abs_q_delta_e"].max()) > float(config["matching"]["charge_tolerance_e"]) + 1e-12:
        raise ValueError("Matched-pair charge tolerance is violated")

    medoid_columns = tuple(config.get("medoid_columns", DEFAULT_MEDOID_COLUMNS))
    audit, selected_index = standardized_medoid(pair, medoid_columns)
    selected_pair = audit.loc[selected_index]
    audit_path = output_dir / "matched_pair_medoid_audit.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    blocks = pair["hcl_time_block_ns"].astype(str) + "|" + pair["nacl_time_block_ns"].astype(str)
    metric_rows: List[Dict[str, object]] = []
    primary = source_summary[source_summary["matched_set"].eq("all matched pairs")]
    for delta_column in medoid_columns:
        column = delta_column.removeprefix("delta_")
        label, units = METRIC_LABELS[column]
        source = primary[primary["column"].eq(column)]
        if len(source) != 1:
            raise ValueError(f"Expected one source-summary row for {column}")
        source_row = source.iloc[0]
        values = pd.to_numeric(pair[delta_column], errors="coerce").to_numpy(float)
        low, high = block_bootstrap_ci(
            values,
            blocks,
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        mean = float(np.nanmean(values))
        for value, expected, name in (
            (mean, float(source_row["delta_hcl_minus_nacl_mean"]), "mean"),
            (low, float(source_row["delta_ci95_low"]), "CI low"),
            (high, float(source_row["delta_ci95_high"]), "CI high"),
        ):
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Bootstrap reproduction failed for {column} {name}: {value} != {expected}")
        metric_rows.append(
            {
                "metric": label,
                "column": column,
                "units": units,
                "n_pairs": int(np.isfinite(values).sum()),
                "hcl_mean": float(source_row["hcl_mean"]),
                "nacl_mean": float(source_row["nacl_mean"]),
                "delta_hcl_minus_nacl_mean": mean,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
            }
        )
    metric_path = output_dir / "matched_metric_summary.csv"
    _write_csv(metric_path, metric_rows)

    charge = pd.read_csv(charge_path)
    barrier = pd.read_csv(barrier_path)
    control_rows: List[Dict[str, object]] = []
    for case_key in ("NaCl", "HCl"):
        case = config["cases"][case_key]
        charge_row = charge[charge["case_label"].eq(case["charge_case_label"])]
        barrier_row = barrier[barrier["case_label"].eq(case["barrier_case_label"])]
        if len(charge_row) != 1 or len(barrier_row) != 1:
            raise ValueError(f"Could not resolve unique charge/barrier control for {case_key}")
        control_rows.append(
            {
                "case_key": case_key,
                "case_label": case["case_label"],
                "max_core_abs_charge_e": float(charge_row.iloc[0]["max_mean_Q_core_abs_e"]),
                "max_charge_overlap_fraction": float(charge_row.iloc[0]["max_charge_overlap_fraction"]),
                "delta_F_win_kJ_mol": float(barrier_row.iloc[0]["barrier_kjmol"]),
                "precontact_frame_count": int(charge_row.iloc[0]["n_frames_0_12A"]),
            }
        )
    control_path = output_dir / "charge_only_control.csv"
    _write_csv(control_path, control_rows)

    selected_rows: List[Dict[str, object]] = []
    for case_key, prefix in (("HCl", "hcl"), ("NaCl", "nacl")):
        source_index = int(selected_pair[f"{prefix}_source_index"])
        source = frame.iloc[source_index]
        if str(source["case_key"]) != case_key:
            raise ValueError(f"Source-index mapping failed for {case_key}")
        case = config["cases"][case_key]
        segment = str(source["segment"])
        segment_data = _segment_config(case, segment)
        selected_rows.append(
            {
                "pair_id": int(selected_pair["pair_id"]),
                "medoid_distance": float(selected_pair["medoid_distance"]),
                "selection_rule": "minimum Euclidean distance to componentwise median in sample-SD standardized six-descriptor difference space",
                "case_key": case_key,
                "case_label": case["case_label"],
                "source_frame_table_index": source_index,
                "segment": segment,
                "local_frame": int(source["frame_index"]),
                "time_ns": float(source["time_ns"]),
                "gap_A": float(source["gap_A"]),
                "strict_bridge_abs_charge_e": float(source["strict_bridge_abs_charge"]),
                "source_n_solution_o_core": int(source["n_solution_o_core"]),
                "source_ion_hbond_count": int(source["ion_hbond_count"]),
                "source_hbond_degree": float(source["hbond_network_avg_degree"]),
                "source_lcc_fraction": float(source["largest_hbond_component_fraction"]),
                "source_surface_h_within_10A": int(source["surface_h_xy_within_10A"]),
                "source_surface_h_min_xy_A": float(source["surface_h_min_xy_A"]),
                "trajectory": segment_data["trajectory"],
                "trace_csv": case["trace_csv"],
            }
        )
    selected_path = output_dir / "selected_microstates.csv"
    _write_csv(selected_path, selected_rows)

    source_paths = [pair_path, frame_path, summary_path, charge_path, barrier_path]
    source_paths.extend(Path(row["trajectory"]) for row in selected_rows)
    manifest_rows = []
    for path in source_paths:
        stat = path.stat()
        manifest_rows.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    source_manifest = manifest_dir / "source_manifest.csv"
    _write_csv(source_manifest, manifest_rows)

    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "analysis_report.md"
    report.write_text(
        "# Figure 7 matched bridge-response analysis\n\n"
        f"- Matched pairs: {len(pair)}.\n"
        f"- Mean |Delta gap|: {pair['abs_gap_delta_A'].mean():.5f} A.\n"
        f"- Mean |Delta q|: {pair['abs_q_delta_e'].mean():.5f} e.\n"
        f"- Descriptor medoid: pair {int(selected_pair['pair_id'])}, distance {float(selected_pair['medoid_distance']):.6f}.\n"
        "- Primary statistics use 20 ps paired blocks and 5000 bootstrap resamples.\n"
        "- The charged-core four-pair subset is not used as the primary figure statistic.\n"
    )
    metadata_path = output_dir / "analysis_metadata.json"
    _write_json(
        metadata_path,
        {
            "config": str(Path(config_path).resolve()),
            "matched_pairs": len(pair),
            "selected_pair_id": int(selected_pair["pair_id"]),
            "selected_medoid_distance": float(selected_pair["medoid_distance"]),
            "medoid_columns": list(medoid_columns),
            "bootstrap": config["bootstrap"],
        },
    )
    return {
        "medoid_audit": audit_path,
        "metric_summary": metric_path,
        "charge_control": control_path,
        "selected_microstates": selected_path,
        "source_manifest": source_manifest,
        "analysis_report": report,
        "metadata": metadata_path,
    }


def _target_frame(path: Path, local_frame: int):
    for frame in iter_lammps_dump_records(path):
        if int(frame.frame_index) == int(local_frame):
            return frame
        if int(frame.frame_index) > int(local_frame):
            break
    raise ValueError(f"Frame {local_frame} not found in {path}")


def _trace_row(path: Path, segment: str, local_frame: int, time_ns: float) -> Mapping[str, object]:
    import pandas as pd

    table = pd.read_csv(path)
    rows = table[
        table["segment"].astype(str).eq(segment)
        & pd.to_numeric(table["local_frame"], errors="coerce").eq(local_frame)
    ]
    if len(rows) != 1:
        rows = table[(pd.to_numeric(table["time_ns"], errors="coerce") - time_ns).abs() < 5e-7]
    if len(rows) != 1:
        raise ValueError(f"Could not resolve trace row for {segment} frame {local_frame}")
    return rows.iloc[0]


def _species_from_h_count(count: int) -> str:
    if count == 1:
        return "oh"
    if count == 2:
        return "h2o"
    if count >= 3:
        return "h3o"
    return "other"


def _nearest_oxygen_assignment(
    oxygen_coords: np.ndarray,
    hydrogen_coords: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign H to nearest O with a periodic KD-tree for large reactive systems."""

    from scipy.spatial import cKDTree

    bounds = np.asarray(bounds, dtype=float)
    lengths = bounds[:, 1] - bounds[:, 0]
    oxygen = (np.asarray(oxygen_coords, dtype=float) - bounds[:, 0]) % lengths
    hydrogen = (np.asarray(hydrogen_coords, dtype=float) - bounds[:, 0]) % lengths
    distances, indices = cKDTree(oxygen, boxsize=lengths).query(
        hydrogen,
        k=1,
        distance_upper_bound=float(cutoff_A),
    )
    valid = np.isfinite(distances) & (indices < len(oxygen))
    assigned = np.full(len(hydrogen), -1, dtype=int)
    assigned[valid] = indices[valid].astype(int)
    counts = np.bincount(assigned[valid], minlength=len(oxygen)).astype(int)
    return counts, assigned


def _region_ids(
    ids: np.ndarray,
    coords: np.ndarray,
    center: np.ndarray,
    bounds: np.ndarray,
    radius_A: float,
    lower_A: float,
    upper_A: float,
) -> List[int]:
    lengths = bounds[:, 1] - bounds[:, 0]
    delta = minimum_image_vectors(coords - center, lengths)
    radial = np.sqrt(delta[:, 1] ** 2 + delta[:, 2] ** 2)
    keep = (delta[:, 0] >= lower_A) & (delta[:, 0] <= upper_A) & (radial <= radius_A)
    return [int(atom_id) for atom_id in ids[keep]]


def _enumerate_hbonds(
    donor_ids: Iterable[int],
    acceptor_ids: Iterable[int],
    positions: Mapping[int, np.ndarray],
    donor_hydrogens: Mapping[int, Sequence[int]],
    species: Mapping[int, str],
    bounds: np.ndarray,
    oo_cutoff_A: float,
    ha_cutoff_A: float,
    angle_cutoff_deg: float,
) -> List[Dict[str, object]]:
    lengths = bounds[:, 1] - bounds[:, 0]
    selected: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for donor in donor_ids:
        donor_pos = positions.get(int(donor))
        if donor_pos is None:
            continue
        for hydrogen in donor_hydrogens.get(int(donor), ()):
            h_pos = positions.get(int(hydrogen))
            if h_pos is None:
                continue
            hd = minimum_image_vectors(donor_pos - h_pos, lengths)
            hd_norm = float(np.linalg.norm(hd))
            if hd_norm <= 1e-12:
                continue
            for acceptor in acceptor_ids:
                acceptor = int(acceptor)
                if acceptor == donor:
                    continue
                acceptor_pos = positions.get(acceptor)
                if acceptor_pos is None:
                    continue
                da_vec = minimum_image_vectors(acceptor_pos - donor_pos, lengths)
                da = float(np.linalg.norm(da_vec))
                if da > oo_cutoff_A:
                    continue
                ha_vec = minimum_image_vectors(acceptor_pos - h_pos, lengths)
                ha = float(np.linalg.norm(ha_vec))
                if ha <= 1e-12 or ha > ha_cutoff_A:
                    continue
                angle = float(np.degrees(np.arccos(np.clip(np.dot(hd, ha_vec) / (hd_norm * ha), -1.0, 1.0))))
                if angle < angle_cutoff_deg:
                    continue
                hbond_type = classify_hbond_type(species.get(int(donor), ""), species.get(acceptor, ""))
                if hbond_type is None:
                    continue
                node1, node2 = sorted((int(donor), acceptor))
                key = (hbond_type, node1, node2)
                row = {
                    "node1_id": node1,
                    "node2_id": node2,
                    "donor_id": int(donor),
                    "hydrogen_id": int(hydrogen),
                    "acceptor_id": acceptor,
                    "hbond_type": hbond_type,
                    "donor_acceptor_distance_A": da,
                    "hydrogen_acceptor_distance_A": ha,
                    "DHA_angle_deg": angle,
                }
                previous = selected.get(key)
                if previous is None or (ha, -angle) < (
                    float(previous["hydrogen_acceptor_distance_A"]),
                    -float(previous["DHA_angle_deg"]),
                ):
                    selected[key] = row
    return list(selected.values())


def _component_map(nodes: Set[int], edges: Set[Tuple[int, int]]) -> Tuple[Dict[int, int], int]:
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
    largest_index = max(range(len(components)), key=lambda index: len(components[index]), default=-1)
    return ({node: index for index, component in enumerate(components) for node in component}, largest_index)


def _extract_one(row: Mapping[str, object], config: Mapping[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    trajectory = Path(str(row["trajectory"]))
    frame = _target_frame(trajectory, int(row["local_frame"]))
    types = config["atom_types"]
    wanted = {int(types[name]) for name in ("hydrogen", "oxygen", "sodium", "chloride", "titanium")}
    by_type = _positions_by_type(frame, wanted)
    h_ids, h_coords = by_type[int(types["hydrogen"])]
    o_ids, o_coords = by_type[int(types["oxygen"])]
    na_ids, na_coords = by_type[int(types["sodium"])]
    cl_ids, cl_coords = by_type[int(types["chloride"])]
    ti_ids, _ = by_type[int(types["titanium"])]
    if not len(ti_ids):
        raise ValueError("Matched TiO2 microstate has no titanium atoms")
    max_ti_id = int(np.max(ti_ids))
    solution_mask = o_ids > max_ti_id
    solution_ids = o_ids[solution_mask]
    surface_ids = o_ids[~solution_mask]
    h_count_per_oxygen, hydrogen_to_oxygen_index = _nearest_oxygen_assignment(
        o_coords,
        h_coords,
        frame.bounds,
        float(config["geometry"]["oh_cutoff_A"]),
    )
    oxygen_index = {int(atom_id): index for index, atom_id in enumerate(o_ids)}
    positions: Dict[int, np.ndarray] = {
        int(atom_id): coord for ids, coords in ((o_ids, o_coords), (h_ids, h_coords), (na_ids, na_coords), (cl_ids, cl_coords)) for atom_id, coord in zip(ids, coords)
    }
    donor_hydrogens: Dict[int, List[int]] = defaultdict(list)
    for h_index, o_index in enumerate(hydrogen_to_oxygen_index):
        if int(o_index) >= 0:
            donor_hydrogens[int(o_ids[int(o_index)])].append(int(h_ids[h_index]))
    species: Dict[int, str] = {
        int(atom_id): _species_from_h_count(int(h_count_per_oxygen[oxygen_index[int(atom_id)]]))
        for atom_id in solution_ids
    }
    for atom_id in surface_ids:
        count = int(h_count_per_oxygen[oxygen_index[int(atom_id)]])
        species[int(atom_id)] = "surface_oh" if count >= 1 else "surface_o"
    species.update({int(atom_id): "na" for atom_id in na_ids})
    species.update({int(atom_id): "cl" for atom_id in cl_ids})

    trace = _trace_row(Path(str(row["trace_csv"])), str(row["segment"]), int(row["local_frame"]), float(row["time_ns"]))
    center = np.asarray([float(trace[f"bridge_center_{dim}_A"]) for dim in "xyz"])
    center_a = np.asarray([float(trace[f"bubble_A_center_{dim}_A"]) for dim in "xyz"])
    center_b = np.asarray([float(trace[f"bubble_B_center_{dim}_A"]) for dim in "xyz"])
    _, e_s, e_u, e_z = local_basis(center_a, center_b, frame.bounds)
    geometry = config["geometry"]
    core_ids = set(
        _region_ids(
            solution_ids,
            np.asarray([positions[int(atom_id)] for atom_id in solution_ids]),
            center,
            frame.bounds,
            float(geometry["core_radius_A"]),
            float(geometry["core_lower_A"]),
            float(geometry["core_upper_A"]),
        )
    )
    shell_radius = float(geometry["core_radius_A"]) + float(geometry["shell_thickness_A"])
    shell_lower = float(geometry["core_lower_A"]) - float(geometry["shell_thickness_A"])
    shell_upper = float(geometry["core_upper_A"]) + float(geometry["shell_thickness_A"])
    solution_shell = set(_region_ids(solution_ids, np.asarray([positions[int(i)] for i in solution_ids]), center, frame.bounds, shell_radius, shell_lower, shell_upper))
    surface_shell = set(_region_ids(surface_ids, np.asarray([positions[int(i)] for i in surface_ids]), center, frame.bounds, shell_radius, shell_lower, shell_upper))
    cl_shell = set(_region_ids(cl_ids, cl_coords, center, frame.bounds, shell_radius, shell_lower, shell_upper))
    na_shell = set(_region_ids(na_ids, na_coords, center, frame.bounds, shell_radius, shell_lower, shell_upper))
    donors = [atom_id for atom_id in solution_shell | surface_shell if species.get(atom_id) in {"h2o", "h3o", "oh", "surface_oh"}]
    acceptors = [atom_id for atom_id in solution_shell | surface_shell | cl_shell if species.get(atom_id) in {"h2o", "h3o", "oh", "surface_o", "surface_oh", "cl"}]
    edges = _enumerate_hbonds(
        donors,
        acceptors,
        positions,
        donor_hydrogens,
        species,
        frame.bounds,
        float(geometry["hbond_oo_cutoff_A"]),
        float(geometry["hbond_ha_cutoff_A"]),
        float(geometry["hbond_angle_cutoff_deg"]),
    )
    core_edges = {
        (int(edge["node1_id"]), int(edge["node2_id"]))
        for edge in edges
        if edge["hbond_type"] in PRIMARY_EDGE_TYPES
        and int(edge["node1_id"]) in core_ids
        and int(edge["node2_id"]) in core_ids
    }
    component_by_node, largest_component = _component_map(core_ids, core_edges)
    ion_edges = [edge for edge in edges if edge["hbond_type"] in ION_EDGE_TYPES]
    plotted_ids = set(core_ids) | cl_shell | na_shell
    plotted_ids.update(atom_id for atom_id in solution_shell if species.get(atom_id) in {"h3o", "oh"})
    for edge in ion_edges:
        plotted_ids.update((int(edge["node1_id"]), int(edge["node2_id"])))

    lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
    atom_delta = {atom_id: minimum_image_vectors(positions[atom_id] - center, lengths) for atom_id in plotted_ids}
    node_rows: List[Dict[str, object]] = []
    for atom_id in sorted(plotted_ids):
        delta = atom_delta[atom_id]
        node_rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "case_key": row["case_key"],
                "atom_id": atom_id,
                "species": species.get(atom_id, "unknown"),
                "is_core": int(atom_id in core_ids),
                "is_largest_component": int(component_by_node.get(atom_id, -1) == largest_component),
                "s_A": float(delta @ e_s),
                "u_A": float(delta @ e_u),
                "z_mid_A": float(delta @ e_z),
            }
        )

    surface_h_ids = {
        int(h_ids[h_index])
        for h_index, o_index in enumerate(hydrogen_to_oxygen_index)
        if int(o_index) >= 0 and int(o_ids[int(o_index)]) in set(int(value) for value in surface_ids)
    }
    near_surface_h: List[int] = []
    min_surface_xy = math.inf
    for atom_id in surface_h_ids:
        delta = minimum_image_vectors(positions[atom_id] - center, lengths)
        lateral = float(np.linalg.norm(delta[:2]))
        min_surface_xy = min(min_surface_xy, lateral)
        if lateral <= float(geometry["surface_h_radius_A"]):
            near_surface_h.append(atom_id)
            node_rows.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "case_key": row["case_key"],
                    "atom_id": atom_id,
                    "species": "surface_h",
                    "is_core": 0,
                    "is_largest_component": 0,
                    "s_A": float(delta @ e_s),
                    "u_A": float(delta @ e_u),
                    "z_mid_A": float(delta @ e_z),
                }
            )
    edge_rows = [{"pair_id": int(row["pair_id"]), "case_key": row["case_key"], **edge} for edge in edges]
    lcc_size = sum(component == largest_component for component in component_by_node.values())
    regenerated_degree = 2.0 * len(core_edges) / len(core_ids) if core_ids else 0.0
    regenerated_lcc = lcc_size / len(core_ids) if core_ids else 0.0
    metadata = {
        "pair_id": int(row["pair_id"]),
        "case_key": row["case_key"],
        "case_label": row["case_label"],
        "segment": row["segment"],
        "local_frame": int(row["local_frame"]),
        "time_ns": float(row["time_ns"]),
        "timestep": int(frame.timestep),
        "gap_A": float(row["gap_A"]),
        "strict_bridge_abs_charge_e": float(row["strict_bridge_abs_charge_e"]),
        "source_n_solution_o_core": int(row["source_n_solution_o_core"]),
        "regenerated_n_solution_o_core": len(core_ids),
        "core_count_matches_source": int(len(core_ids) == int(row["source_n_solution_o_core"])),
        "source_ion_hbond_count": int(row["source_ion_hbond_count"]),
        "regenerated_ion_hbond_count": len(ion_edges),
        "source_hbond_degree": float(row["source_hbond_degree"]),
        "regenerated_hbond_degree": regenerated_degree,
        "source_lcc_fraction": float(row["source_lcc_fraction"]),
        "regenerated_lcc_fraction": regenerated_lcc,
        "source_surface_h_within_10A": int(row["source_surface_h_within_10A"]),
        "regenerated_surface_h_within_10A": len(near_surface_h),
        "surface_h_count_matches_source": int(len(near_surface_h) == int(row["source_surface_h_within_10A"])),
        "source_surface_h_min_xy_A": float(row["source_surface_h_min_xy_A"]),
        "regenerated_surface_h_min_xy_A": min_surface_xy,
        "trajectory": str(trajectory),
        "visualization_note": "All quantitative labels use archived source descriptors; nodes and edges are regenerated from the exact raw frame under the recorded geometry.",
    }
    return node_rows, edge_rows, metadata


def extract(config_path: Path, selected_path: Path, output_dir: Path) -> Dict[str, Path]:
    """Extract both exact raw microstates selected by the matched-pair medoid."""

    import pandas as pd

    config = _read_json(config_path)
    selected = pd.read_csv(selected_path)
    if set(selected["case_key"]) != {"HCl", "NaCl"}:
        raise ValueError("Selected microstate table must contain HCl and NaCl")
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    metadata: List[Dict[str, object]] = []
    for row in selected.to_dict("records"):
        case_nodes, case_edges, case_metadata = _extract_one(row, config)
        nodes.extend(case_nodes)
        edges.extend(case_edges)
        metadata.append(case_metadata)
    node_path = output_dir / "microstate_nodes.csv"
    edge_path = output_dir / "microstate_hbond_edges.csv"
    metadata_path = output_dir / "microstate_metadata.csv"
    _write_csv(node_path, nodes)
    _write_csv(edge_path, edges)
    _write_csv(metadata_path, metadata)
    return {"nodes": node_path, "edges": edge_path, "metadata": metadata_path}


def _draw_interval_axis(ax, rows, color: str, xlabel: str) -> None:
    y = np.arange(len(rows))[::-1]
    mean = np.asarray([float(row["delta_hcl_minus_nacl_mean"]) for row in rows])
    low = np.asarray([float(row["delta_ci95_low"]) for row in rows])
    high = np.asarray([float(row["delta_ci95_high"]) for row in rows])
    ax.axvline(0.0, color="0.35", lw=0.8, ls="--")
    for yi, value, left, right in zip(y, mean, low, high):
        face = color if value >= 0 else "white"
        ax.plot([left, right], [yi, yi], color=color, lw=1.8)
        ax.scatter([value], [yi], s=28, facecolor=face, edgecolor=color, linewidth=1.2, zorder=3)
        ax.annotate(f"{value:+.2f}", (value, yi), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=6.8)
    short_labels = {
        "Largest-component fraction": "Largest-component frac.",
        "Minimum surface-H distance": "Min. surface-H distance",
    }
    ax.set_yticks(
        y,
        [short_labels.get(str(row["metric"]), str(row["metric"])) for row in rows],
        fontsize=7,
    )
    ax.set_ylim(-0.55, max(0.55, len(rows) - 0.35))
    ax.set_xlabel(xlabel, fontsize=7.2, labelpad=1)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.margins(x=0.15)


def _draw_microstate(network_ax, surface_ax, case_key: str, nodes, edges, metadata) -> None:
    import pandas as pd

    data = nodes[nodes["case_key"].eq(case_key)].copy()
    edge_data = edges[edges["case_key"].eq(case_key)].copy()
    meta = metadata[metadata["case_key"].eq(case_key)].iloc[0]
    lookup = {int(row.atom_id): row for row in data.itertuples()}
    water_edges = edge_data[
        edge_data["hbond_type"].eq("water_water")
        & edge_data["node1_id"].isin(data.loc[data["is_core"].eq(1), "atom_id"])
        & edge_data["node2_id"].isin(data.loc[data["is_core"].eq(1), "atom_id"])
    ]
    ion_edges = edge_data[edge_data["hbond_type"].isin(ION_EDGE_TYPES)]
    for row in water_edges.itertuples():
        a, b = lookup.get(int(row.node1_id)), lookup.get(int(row.node2_id))
        if a is not None and b is not None:
            network_ax.plot([a.s_A, b.s_A], [a.z_mid_A, b.z_mid_A], color="#7f8fa6", lw=0.55, alpha=0.42, zorder=1)
    for row in ion_edges.itertuples():
        a, b = lookup.get(int(row.node1_id)), lookup.get(int(row.node2_id))
        if a is not None and b is not None:
            network_ax.plot([a.s_A, b.s_A], [a.z_mid_A, b.z_mid_A], color="#c66a1c", lw=1.15, ls="--", alpha=0.9, zorder=2)

    styles = {
        "h2o": ("o", "#7fb3d5", 19, "H$_2$O"),
        "h3o": ("^", "#a23b72", 36, "H$_3$O$^+$"),
        "oh": ("v", "#d89022", 34, "OH$^-$"),
        "cl": ("s", "#327f74", 38, "Cl$^-$"),
        "na": ("D", "#6655a5", 30, "Na$^+$"),
    }
    for species, (marker, color, size, _label) in styles.items():
        subset = data[data["species"].eq(species)]
        if species == "h2o":
            subset = subset[subset["is_core"].eq(1)]
        if subset.empty:
            continue
        sizes = np.maximum(8.0, size - 1.1 * np.abs(subset["u_A"].to_numpy(float)))
        network_ax.scatter(subset["s_A"], subset["z_mid_A"], s=sizes, marker=marker, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.88, zorder=3)
    gap = float(meta["gap_A"])
    network_ax.axvspan(-10.0, -0.5 * gap, color="#f1f3f5", zorder=0)
    network_ax.axvspan(0.5 * gap, 10.0, color="#f1f3f5", zorder=0)
    network_ax.axvline(-0.5 * gap, color="0.35", lw=0.8, ls=":")
    network_ax.axvline(0.5 * gap, color="0.35", lw=0.8, ls=":")
    network_ax.set_xlim(-10.5, 10.5)
    visible = data[~data["species"].eq("surface_h")]
    zlim = max(8.5, float(np.nanmax(np.abs(visible["z_mid_A"]))) + 0.8) if not visible.empty else 8.5
    network_ax.set_ylim(-zlim, zlim)
    network_ax.set_xlabel("bubble axis, $s$ (Å)", fontsize=7.5)
    network_ax.set_ylabel("surface-normal coordinate, $z$ (Å)", fontsize=7.5)
    network_ax.tick_params(labelsize=7)
    network_ax.set_title(
        f"{case_key}: descriptor-medoid frame\n"
        f"gap={gap:.2f} Å; |q$_{{core}}$|={float(meta['strict_bridge_abs_charge_e']):.0f} e; "
        f"core O={int(meta['source_n_solution_o_core'])}",
        loc="left",
        fontsize=8.2,
        pad=3,
    )
    network_ax.text(
        0.99,
        0.02,
        f"ion-water HB={int(meta['source_ion_hbond_count'])}; degree={float(meta['source_hbond_degree']):.2f}",
        transform=network_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="0.25",
    )

    surface = data[data["species"].eq("surface_h")]
    circle = __import__("matplotlib.patches", fromlist=["Circle"]).Circle((0, 0), 10.0, facecolor="#f5f5f5", edgecolor="0.5", lw=0.8)
    surface_ax.add_patch(circle)
    if not surface.empty:
        surface_ax.scatter(surface["s_A"], surface["u_A"], s=38, marker="*", color="#b8860b", edgecolor="white", linewidth=0.4, zorder=3)
    else:
        surface_ax.text(0, 0, "none", ha="center", va="center", fontsize=7, color="0.45")
    surface_ax.set_xlim(-10.8, 10.8)
    surface_ax.set_ylim(-10.8, 10.8)
    surface_ax.set_aspect("equal")
    surface_ax.set_xticks([-10, 0, 10])
    surface_ax.set_yticks([-10, 0, 10])
    surface_ax.tick_params(labelsize=6)
    surface_ax.set_xlabel("$s$ (Å)", fontsize=6.8)
    surface_ax.set_ylabel("$u$ (Å)", fontsize=6.8)
    surface_ax.set_title(f"TiO$_2$ H$^+_{{surf}}$ within 10 Å: {int(meta['source_surface_h_within_10A'])}", fontsize=7.2, loc="left", pad=2)


def plot(output_dir: Path, figure_dir: Path, report_dir: Path) -> Dict[str, Path]:
    """Render the four-panel candidate Figure 7 and its caption."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    control = pd.read_csv(output_dir / "charge_only_control.csv")
    summary = pd.read_csv(output_dir / "matched_metric_summary.csv")
    nodes = pd.read_csv(output_dir / "microstate_nodes.csv")
    edges = pd.read_csv(output_dir / "microstate_hbond_edges.csv")
    metadata = pd.read_csv(output_dir / "microstate_metadata.csv")
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_data = figure_dir / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    for name in ("charge_only_control.csv", "matched_metric_summary.csv", "microstate_nodes.csv", "microstate_hbond_edges.csv", "microstate_metadata.csv", "matched_pair_medoid_audit.csv"):
        source = output_dir / name
        target = plot_data / name
        target.write_bytes(source.read_bytes())

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.25, 7.8), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.75), hspace=0.33)
    top = outer[0, 0].subgridspec(1, 2, width_ratios=(0.86, 1.65), wspace=0.48)
    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.25)
    ax_a = fig.add_subplot(top[0, 0])
    d_grid = top[0, 1].subgridspec(3, 1, height_ratios=(3.0, 2.0, 1.0), hspace=1.0)
    ax_d1, ax_d2, ax_d3 = [fig.add_subplot(d_grid[index, 0]) for index in range(3)]
    b_grid = bottom[0, 0].subgridspec(2, 1, height_ratios=(3.3, 1.35), hspace=0.42)
    c_grid = bottom[0, 1].subgridspec(2, 1, height_ratios=(3.3, 1.35), hspace=0.42)
    ax_b, ax_bs = fig.add_subplot(b_grid[0]), fig.add_subplot(b_grid[1])
    ax_c, ax_cs = fig.add_subplot(c_grid[0]), fig.add_subplot(c_grid[1])

    colors = {"NaCl": "#2f6f9f", "HCl": "#c66a1c"}
    markers = {"NaCl": "o", "HCl": "s"}
    for row in control.itertuples():
        ax_a.scatter(row.max_core_abs_charge_e, row.delta_F_win_kJ_mol, s=55, marker=markers[row.case_key], facecolor=colors[row.case_key], edgecolor="white", linewidth=0.7, zorder=3)
        ax_a.annotate(row.case_key, (row.max_core_abs_charge_e, row.delta_F_win_kJ_mol), xytext=(5, 3 if row.case_key == "HCl" else -11), textcoords="offset points", fontsize=7.5)
    ax_a.plot(control["max_core_abs_charge_e"], control["delta_F_win_kJ_mol"], color="0.55", lw=0.9, ls="--", zorder=1)
    ax_a.set_xlim(0.50, 0.59)
    ax_a.set_ylim(230, 640)
    ax_a.set_xlabel("max bridge-core |charge| (e)", fontsize=7.5)
    ax_a.set_ylabel("$\\Delta F_{win}$ (kJ mol$^{-1}$)", fontsize=7.5)
    ax_a.tick_params(labelsize=7)
    ax_a.grid(color="0.9", lw=0.6)
    charge_ratio = float(control.loc[control.case_key.eq("HCl"), "max_core_abs_charge_e"].iloc[0] / control.loc[control.case_key.eq("NaCl"), "max_core_abs_charge_e"].iloc[0])
    fes_ratio = float(control.loc[control.case_key.eq("HCl"), "delta_F_win_kJ_mol"].iloc[0] / control.loc[control.case_key.eq("NaCl"), "delta_F_win_kJ_mol"].iloc[0])
    ax_a.text(0.03, 0.97, f"charge: {charge_ratio:.2f}×\n$\\Delta F_{{win}}$: {fes_ratio:.2f}×", transform=ax_a.transAxes, va="top", fontsize=7.2)
    ax_a.set_title("Charge-only comparison", loc="left", fontsize=8.5)

    count_rows = summary[summary["column"].isin(["n_solution_o_core", "ion_hbond_count", "surface_h_xy_within_10A"])].to_dict("records")
    topology_rows = summary[summary["column"].isin(["hbond_network_avg_degree", "largest_hbond_component_fraction"])].to_dict("records")
    distance_rows = summary[summary["column"].eq("surface_h_min_xy_A")].to_dict("records")
    _draw_interval_axis(ax_d1, count_rows, colors["HCl"], "HCl − NaCl (count)")
    _draw_interval_axis(ax_d2, topology_rows, colors["NaCl"], "HCl − NaCl")
    _draw_interval_axis(ax_d3, distance_rows, colors["NaCl"], "HCl − NaCl (Å)")
    ax_d1.set_title("24 matched pairs; 95% block-bootstrap CI", fontsize=8.5, pad=4, loc="left")

    _draw_microstate(ax_b, ax_bs, "NaCl", nodes, edges, metadata)
    _draw_microstate(ax_c, ax_cs, "HCl", nodes, edges, metadata)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7fb3d5", label="H$_2$O", markersize=5),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#a23b72", label="H$_3$O$^+$", markersize=5),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#d89022", label="OH$^-$", markersize=5),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#327f74", label="Cl$^-$", markersize=5),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#6655a5", label="Na$^+$", markersize=4.5),
        Line2D([0], [0], color="#7f8fa6", lw=1, label="water-water HB"),
        Line2D([0], [0], color="#c66a1c", lw=1.2, ls="--", label="ion-water HB"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.027), ncol=4, frameon=False, fontsize=6.8, columnspacing=1.2, handletextpad=0.4)
    for label, ax in (("a", ax_a), ("d", ax_d1), ("b", ax_b), ("c", ax_c)):
        ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.text(0.99, 0.008, "All bridge nodes are selected in 3D; panels b,c are orthographic projections (marker size decreases with |u|).", ha="right", fontsize=6.2, color="0.35")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.145)

    stem = figure_dir / "figure7_matched_bridge_response_v1"
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=400 if suffix == "png" else None, facecolor="white")
        outputs[suffix] = path
    plt.close(fig)
    report_dir.mkdir(parents=True, exist_ok=True)
    caption = report_dir / "candidate_caption.md"
    caption.write_text(
        "# Candidate Figure 7 caption\n\n"
        "**Matched microscopic origin of the HCl wet-film response.** "
        "(a) Symmetric NaCl and HCl systems have comparable maximum bridge-core absolute charge but different fixed-window projected free-energy differences. "
        "(b,c) Automatically selected descriptor-medoid matched frames, shown as orthographic projections of the full three-dimensional bridge selection. Solid and dashed edges denote water-water and ion-water hydrogen bonds, respectively; the lower views show surface-bound H within 10 Å laterally of the bridge midpoint. "
        "(d) HCl-minus-NaCl differences across 24 frame pairs matched within 1 Å in surface gap and 0.25 e in strict bridge-core absolute charge. Intervals are 95% confidence intervals from 5000 resamples over 20 ps matched-pair blocks.\n"
    )
    outputs["caption"] = caption
    return outputs


def validate(output_dir: Path, figure_dir: Path, report_dir: Path, manifest_dir: Path) -> Path:
    """Fail closed on statistical, extraction, and rendered-artifact invariants."""

    import pandas as pd

    audit = pd.read_csv(output_dir / "matched_pair_medoid_audit.csv")
    selected = pd.read_csv(output_dir / "selected_microstates.csv")
    metadata = pd.read_csv(output_dir / "microstate_metadata.csv")
    summary = pd.read_csv(output_dir / "matched_metric_summary.csv")
    if len(audit) != 24 or int(audit["medoid_selected"].sum()) != 1:
        raise ValueError("Medoid audit invariant failed")
    if len(selected) != 2 or set(selected["case_key"]) != {"HCl", "NaCl"}:
        raise ValueError("Selected microstate invariant failed")
    if len(summary) != 6 or set(summary["n_pairs"].astype(int)) != {24}:
        raise ValueError("Matched summary invariant failed")
    if not metadata["core_count_matches_source"].astype(bool).all():
        raise ValueError("Regenerated core-water count does not match the archived source")
    if not metadata["surface_h_count_matches_source"].astype(bool).all():
        raise ValueError("Regenerated surface-H count does not match the archived source")
    figure_paths = [figure_dir / f"figure7_matched_bridge_response_v1.{suffix}" for suffix in ("png", "pdf", "svg")]
    if any(not path.exists() or path.stat().st_size < 10_000 for path in figure_paths):
        raise ValueError("Rendered Figure 7 artifact is missing or unexpectedly small")
    artifact_paths = [
        output_dir / "matched_pair_medoid_audit.csv",
        output_dir / "matched_metric_summary.csv",
        output_dir / "charge_only_control.csv",
        output_dir / "selected_microstates.csv",
        output_dir / "microstate_nodes.csv",
        output_dir / "microstate_hbond_edges.csv",
        output_dir / "microstate_metadata.csv",
        *figure_paths,
        report_dir / "candidate_caption.md",
    ]
    manifest_rows = [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in artifact_paths
    ]
    _write_csv(manifest_dir / "artifact_manifest.csv", manifest_rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "validation_report.txt"
    report.write_text(
        "FIG07_VALIDATION_OK\n"
        f"matched_pairs={len(audit)}\n"
        f"selected_pair_id={int(audit.loc[audit.medoid_selected.astype(bool), 'pair_id'].iloc[0])}\n"
        f"core_count_match_cases={int(metadata.core_count_matches_source.sum())}/2\n"
        f"surface_h_count_match_cases={int(metadata.surface_h_count_matches_source.sum())}/2\n"
        "quantitative_statistics=archived matched-pair source table with exact bootstrap reproduction\n"
        "microstate_edges=regenerated from exact raw frames under recorded geometric cutoffs\n"
    )
    print(f"FIG07_VALIDATION_OK pairs={len(audit)} selected_pair={int(audit.loc[audit.medoid_selected.astype(bool), 'pair_id'].iloc[0])}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched HCl/NaCl bridge-response candidate Figure 7")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--config", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--manifest-dir", type=Path, required=True)
    analyze_parser.add_argument("--report-dir", type=Path, required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--config", type=Path, required=True)
    extract_parser.add_argument("--selected", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser.add_argument("--figure-dir", type=Path, required=True)
    plot_parser.add_argument("--report-dir", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--figure-dir", type=Path, required=True)
    validate_parser.add_argument("--report-dir", type=Path, required=True)
    validate_parser.add_argument("--manifest-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        outputs = analyze(args.config, args.output_dir, args.manifest_dir, args.report_dir)
    elif args.command == "extract":
        outputs = extract(args.config, args.selected, args.output_dir)
    elif args.command == "plot":
        outputs = plot(args.output_dir, args.figure_dir, args.report_dir)
    else:
        outputs = {"validation_report": validate(args.output_dir, args.figure_dir, args.report_dir, args.manifest_dir)}
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
