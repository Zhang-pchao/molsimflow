"""Integrate local-flow and fixed-grid ion/water diagnostics without causal overclaim."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from molsimflow.postprocess.constant_force_oxygen import write_output_hashes, write_tsv


LABELS = (
    "hcl_63pairs__waterO_f8e-6_x", "hcl_63pairs__waterO_f8e-6_y",
    "naoh_ph13p4__waterO_f8e-6_x", "naoh_ph13p4__waterO_f8e-6_y",
)
REGIONS = ("front", "rear", "other_wall", "far_wall")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty table: {path}")
    return rows


def finite_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else math.nan


def weighted_mean(rows: list[dict[str, str]], value: str, weight: str) -> tuple[float, int]:
    pairs = [(float(row[value]), int(row[weight])) for row in rows]
    good = [(v, n) for v, n in pairs if n > 0 and math.isfinite(v)]
    total = sum(n for _, n in good)
    return (sum(v * n for v, n in good) / total if total else math.nan, total)


def shift_placebo(predictor: np.ndarray, outcome: np.ndarray) -> tuple[float, float, int]:
    if len(predictor) != len(outcome) or len(predictor) < 10:
        raise ValueError("Paired block series require at least 10 points")
    if not np.isfinite(predictor).all() or not np.isfinite(outcome).all():
        return math.nan, math.nan, len(predictor)
    if np.std(predictor) == 0 or np.std(outcome) == 0:
        return math.nan, math.nan, len(predictor)
    observed = float(spearmanr(predictor, outcome).statistic)
    null = [abs(float(spearmanr(np.roll(predictor, shift), outcome).statistic))
            for shift in range(1, len(predictor))]
    p = (1 + sum(value >= abs(observed) for value in null)) / len(predictor)
    return observed, p, len(predictor)


def bh_qvalues(pvalues: list[float]) -> list[float]:
    finite = [(index, p) for index, p in enumerate(pvalues) if math.isfinite(p)]
    result = [math.nan] * len(pvalues)
    if not finite:
        return result
    ordered = sorted(finite, key=lambda pair: pair[1])
    running = 1.0
    for rank in range(len(ordered), 0, -1):
        index, p = ordered[rank - 1]
        running = min(running, p * len(ordered) / rank)
        result[index] = running
    return result


def integrate(run: Path, output: Path, *, pbc_run: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    pbc_run = run if pbc_run is None else pbc_run
    all_summary = []
    turnover = []
    blocks = []
    for label in LABELS:
        flow = run / "03_local_flow" / label
        grid = run / "05_fixed_grid_water_defect" / label
        for directory in (flow, grid):
            if json.loads((directory / "summary.json").read_text())["status"] != "PASS":
                raise ValueError(f"Unaccepted parent: {directory}")
        frames = read_tsv(flow / "local_flow_geometry.tsv")
        intervals = read_tsv(flow / "edge_flow_intervals.tsv")
        ion_frames = read_tsv(grid / "ion_accessible_volume.tsv")
        grid_intervals = read_tsv(grid / "fixed_grid_renewal.tsv")
        if len(frames) != 381 or len(intervals) != 380:
            raise ValueError(f"Local-flow coverage failure: {label}")
        expected_steps = list(range(30400000, 38000001, 20000))
        if [int(row["step"]) for row in frames] != expected_steps:
            raise ValueError(f"Frame grid mismatch: {label}")
        if [int(row["end_step"]) for row in intervals] != expected_steps[1:]:
            raise ValueError(f"Interval grid mismatch: {label}")
        species = json.loads((grid / "summary.json").read_text())["species"]
        by_region_species = defaultdict(list)
        for row in ion_frames:
            by_region_species[(row["region"], row["species"])].append(row)
        if len(ion_frames) != 381 * 4 * len(species):
            raise ValueError(f"Ion frame coverage failure: {label}")
        if len(grid_intervals) != 380 * 4 * len(species) * 2:
            raise ValueError(f"Grid interval coverage failure: {label}")
        pbc_drive = finite_mean(float(row["pbc_spanning_drive"]) for row in frames)
        pbc_transverse = finite_mean(float(row["pbc_spanning_transverse"]) for row in frames)
        graph_rows = read_tsv(pbc_run / "02_pbc_graph" / label / "PBC-GRAPH-FRAMES.tsv")
        graph_primary = [row for row in graph_rows if float(row["cutoff_A"]) == 5.5]
        if [int(row["step"]) for row in graph_primary] != expected_steps:
            raise ValueError(f"Graph-PBC coverage mismatch: {label}")
        summary = {
            "label": label, "chemistry": frames[0]["case_id"], "direction": frames[0]["direction"],
            "frames": len(frames), "main_bubble_n2_min": min(int(row["main_bubble_n2"]) for row in frames),
            "contact_valid_fraction": finite_mean(float(row["contact_line_valid"]) for row in frames),
            "pbc_spanning_drive_fraction": pbc_drive,
            "pbc_spanning_transverse_fraction": pbc_transverse,
            "graph_winding_x_fraction": finite_mean(float(row["main_winding_x"]) for row in graph_primary),
            "graph_winding_y_fraction": finite_mean(float(row["main_winding_y"]) for row in graph_primary),
            "bubble_velocity_mps": finite_mean(float(row["bubble_velocity_mps"]) for row in intervals),
            "front_water_velocity_mps": finite_mean(float(row["front_water_drive_velocity_mps"]) for row in frames),
            "rear_water_velocity_mps": finite_mean(float(row["rear_water_drive_velocity_mps"]) for row in frames),
            "above_water_velocity_mps": finite_mean(float(row["above_water_drive_velocity_mps"]) for row in frames),
            "far_water_velocity_mps": finite_mean(float(row["bulk_water_drive_velocity_mps"]) for row in frames),
            "bubble_minus_edge_water_mps": finite_mean(float(row["bubble_minus_mean_edge_water_mps"]) for row in intervals),
        }
        all_summary.append(summary)
        for region in REGIONS:
            for sp in species:
                selected = by_region_species[(region, sp)]
                ions = sum(int(row["ion_count"]) for row in selected)
                volume = sum(float(row["accessible_volume_A3"]) for row in selected)
                prior = [row for row in grid_intervals if row["region"] == region and row["species"] == sp]
                exposed = [row for row in prior if row["ion_present_at_start"] == "1"]
                unexposed = [row for row in prior if row["ion_present_at_start"] == "0"]
                exposed_turnover, exposed_cells = weighted_mean(exposed, "mean_oxygen_turnover", "stable_liquid_cells")
                unexposed_turnover, unexposed_cells = weighted_mean(unexposed, "mean_oxygen_turnover", "stable_liquid_cells")
                turnover.append({
                    "label": label, "chemistry": summary["chemistry"], "direction": summary["direction"],
                    "region": region, "species": sp,
                    "ion_count": ions, "accessible_volume_A3": volume,
                    "ion_density_A3_inv": ions / volume if volume else math.nan,
                    "exposed_cell_intervals": exposed_cells,
                    "unexposed_cell_intervals": unexposed_cells,
                    "exposed_oxygen_turnover": exposed_turnover,
                    "unexposed_oxygen_turnover": unexposed_turnover,
                    "exposed_minus_unexposed_turnover": exposed_turnover - unexposed_turnover,
                    "persistent_ion_cell_intervals": sum(int(row["defect_cell_persistent"]) for row in exposed),
                    "persistent_same_carrier_intervals": sum(int(row["same_carrier_persistent"]) for row in exposed),
                })
        frame_by_step = defaultdict(list)
        for row in ion_frames:
            frame_by_step[int(row["step"])].append(row)
        for block in range(38):
            lo = 30400000 + 200000 * block
            hi = lo + 200000
            flow_rows = [row for row in intervals if lo < int(row["end_step"]) <= hi]
            if len(flow_rows) != 10:
                raise ValueError(f"100-ps block incomplete: {label}, {block}")
            for sp in species:
                edge_frames = [row for step in range(lo + 20000, hi + 1, 20000)
                               for row in frame_by_step[step]
                               if row["species"] == sp and row["region"] in ("front", "rear")]
                ion_count = sum(int(row["ion_count"]) for row in edge_frames)
                volume = sum(float(row["accessible_volume_A3"]) for row in edge_frames)
                record = {
                    "label": label, "chemistry": summary["chemistry"], "direction": summary["direction"],
                    "species": sp, "block": block, "start_step": lo, "end_step": hi,
                    "edge_ion_density_A3_inv": ion_count / volume if volume else math.nan,
                    "bubble_velocity_mps": finite_mean(float(row["bubble_velocity_mps"]) for row in flow_rows),
                    "edge_water_velocity_mps": finite_mean(
                        0.5 * (float(row["front_water_velocity_mps"]) + float(row["rear_water_velocity_mps"]))
                        for row in flow_rows),
                    "bubble_minus_edge_water_mps": finite_mean(
                        float(row["bubble_minus_mean_edge_water_mps"]) for row in flow_rows),
                }
                blocks.append(record)
    tests = []
    for label, species, outcome in (
        ("hcl_63pairs__waterO_f8e-6_y", "H3O_plus_candidate", "bubble_minus_edge_water_mps"),
        ("hcl_63pairs__waterO_f8e-6_x", "H3O_plus_candidate", "bubble_minus_edge_water_mps"),
        ("naoh_ph13p4__waterO_f8e-6_y", "OH_minus_candidate", "edge_water_velocity_mps"),
        ("naoh_ph13p4__waterO_f8e-6_x", "OH_minus_candidate", "edge_water_velocity_mps"),
    ):
        rows = sorted([row for row in blocks if row["label"] == label and row["species"] == species],
                      key=lambda row: row["block"])
        predictor = np.asarray([row["edge_ion_density_A3_inv"] for row in rows[:-1]], dtype=float)
        next_outcome = np.asarray([row[outcome] for row in rows[1:]], dtype=float)
        rho, p, n = shift_placebo(predictor, next_outcome)
        tests.append({"label": label, "species": species, "predictor": "edge_ion_density_previous_100ps",
                      "outcome": outcome + "_next_100ps", "paired_blocks": n,
                      "spearman_rho": rho, "circular_shift_p_exploratory": p})
    qvalues = bh_qvalues([row["circular_shift_p_exploratory"] for row in tests])
    for row, q in zip(tests, qvalues):
        row["bh_q_exploratory"] = q
    output.mkdir(parents=True)
    write_tsv(output / "LOCAL-FLOW-AND-TOPOLOGY.tsv", all_summary, tuple(all_summary[0]))
    write_tsv(output / "FIXED-GRID-ION-WATER.tsv", turnover, tuple(turnover[0]))
    write_tsv(output / "100PS-BLOCKS.tsv", blocks, tuple(blocks[0]))
    write_tsv(output / "LAGGED-PLACEBO-EXPLORATORY.tsv", tests, tuple(tests[0]))
    report = {
        "status": "PASS", "branches": len(all_summary), "frames_per_branch": 381,
        "blocks_per_branch": 38, "pure_water_matched_local_observer": False,
        "independent_replicates_per_condition": 1,
        "causal_ion_mechanism_established": False,
        "circular_shift_tests_are_exploratory": True,
        "null_tests_do_not_prove_absence": True,
    }
    (output / "DECISION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_output_hashes(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pbc-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(integrate(args.run, args.output, pbc_run=args.pbc_run), sort_keys=True))
