"""Coarse edge-motion and fixed-cell water/ion joint ledger.

These are 100-ps block modes, not sub-ps depinning or independent events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from molsimflow.postprocess.constant_force_oxygen import write_output_hashes, write_tsv
from molsimflow.postprocess.nanobubble_ion_mediation_review import (
    LABELS,
    finite_mean,
    read_tsv,
    weighted_mean,
)


def motion_mode(front_A: float, rear_A: float) -> str:
    if front_A > 0 and rear_A > 0:
        return "both_forward"
    if front_A < 0 and rear_A < 0:
        return "both_reverse"
    if front_A >= 0 and rear_A <= 0:
        return "width_expanding_or_mixed"
    return "width_contracting_or_mixed"


def analyze(parent: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for label in LABELS:
        flow = parent / "03_local_flow" / label
        grid = parent / "05_fixed_grid_water_defect" / label
        frame = read_tsv(flow / "local_flow_geometry.tsv")
        interval = read_tsv(flow / "edge_flow_intervals.tsv")
        ions = read_tsv(grid / "ion_accessible_volume.tsv")
        renewal = read_tsv(grid / "fixed_grid_renewal.tsv")
        species = json.loads((grid / "summary.json").read_text())["species"]
        if len(frame) != 381 or len(interval) != 380:
            raise ValueError(f"Unaccepted block coverage for {label}")
        frame_by_step = {int(item["step"]): item for item in frame}
        for block in range(38):
            start = 30400000 + block * 200000
            end = start + 200000
            block_intervals = [item for item in interval if start < int(item["end_step"]) <= end]
            if len(block_intervals) != 10:
                raise ValueError(f"Incomplete 100-ps block: {label} {block}")
            front_A = float(frame_by_step[end]["front_edge_unwrapped_A"]) - float(frame_by_step[start]["front_edge_unwrapped_A"])
            rear_A = float(frame_by_step[end]["rear_edge_unwrapped_A"]) - float(frame_by_step[start]["rear_edge_unwrapped_A"])
            bubble_A = float(frame_by_step[end]["bubble_drive_unwrapped_A"]) - float(frame_by_step[start]["bubble_drive_unwrapped_A"])
            edge_avg = finite_mean(
                0.5 * (float(item["front_water_velocity_mps"]) + float(item["rear_water_velocity_mps"]))
                for item in block_intervals
            )
            for region in ("front", "rear"):
                for sp in species:
                    grid_rows = [item for item in renewal
                                 if start < int(item["end_step"]) <= end
                                 and item["region"] == region and item["species"] == sp]
                    if len(grid_rows) != 20:
                        raise ValueError(f"Incomplete renewal join: {label} {block} {region} {sp}")
                    exposed = [item for item in grid_rows if item["ion_present_at_start"] == "1"]
                    unexposed = [item for item in grid_rows if item["ion_present_at_start"] == "0"]
                    exp_mean, exp_cells = weighted_mean(exposed, "mean_oxygen_turnover", "stable_liquid_cells")
                    unexp_mean, unexp_cells = weighted_mean(unexposed, "mean_oxygen_turnover", "stable_liquid_cells")
                    ion_rows = [item for item in ions
                                if start < int(item["step"]) <= end
                                and item["region"] == region and item["species"] == sp]
                    if len(ion_rows) != 10:
                        raise ValueError(f"Incomplete ion join: {label} {block} {region} {sp}")
                    count = sum(int(item["ion_count"]) for item in ion_rows)
                    volume = sum(float(item["accessible_volume_A3"]) for item in ion_rows)
                    rows.append({
                        "label": label, "direction": frame[0]["direction"],
                        "block": block, "start_step": start, "end_step": end,
                        "region": region, "species": sp,
                        "bubble_displacement_A": bubble_A,
                        "front_displacement_A": front_A,
                        "rear_displacement_A": rear_A,
                        "center_mode_displacement_A": 0.5 * (front_A + rear_A),
                        "width_mode_displacement_A": front_A - rear_A,
                        "motion_mode": motion_mode(front_A, rear_A),
                        "edge_water_velocity_mps": edge_avg,
                        "ion_density_A3_inv": count / volume if volume else math.nan,
                        "exposed_cell_intervals": exp_cells,
                        "unexposed_cell_intervals": unexp_cells,
                        "exposed_oxygen_turnover": exp_mean,
                        "unexposed_oxygen_turnover": unexp_mean,
                        "turnover_exposed_minus_unexposed": exp_mean - unexp_mean,
                    })
    output.mkdir(parents=True)
    write_tsv(output / "EDGE-WATER-ION-100PS-BLOCKS.tsv", rows, tuple(rows[0]))
    summary = {
        "status": "PASS", "branches": 4, "blocks_per_branch": 38,
        "block_duration_ps": 100, "region_species_rows": len(rows),
        "coarse_blocks_not_molecular_depinning_events": True,
        "within_trajectory_blocks_not_independent_replicates": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_output_hashes(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.parent, args.output), sort_keys=True))
