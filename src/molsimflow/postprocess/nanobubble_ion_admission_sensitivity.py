"""Summarize liquid-cell admission thresholds without changing hypotheses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from molsimflow.postprocess.constant_force_oxygen import write_output_hashes, write_tsv
from molsimflow.postprocess.nanobubble_ion_mediation_review import read_tsv, weighted_mean


def summarize(baseline: Path, sensitivity: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for branch in ("hcl_63pairs__waterO_f8e-6_y", "naoh_ph13p4__waterO_f8e-6_y"):
        for threshold in (1, 2, 3):
            source = (baseline / branch if threshold == 2 else
                      sensitivity / f"{branch}__min{threshold}")
            status = json.loads((source / "summary.json").read_text())
            if status["status"] != "PASS" or status["frames"] != 381 or status["liquid_cell_criterion"] != threshold:
                raise ValueError(f"Unaccepted admission source: {source}")
            frames = read_tsv(source / "ion_accessible_volume.tsv")
            intervals = read_tsv(source / "fixed_grid_renewal.tsv")
            for species in status["species"]:
                for region in ("front", "rear", "far_wall"):
                    sub = [r for r in frames if r["region"] == region and r["species"] == species]
                    ion_count = sum(int(r["ion_count"]) for r in sub)
                    volume = sum(float(r["accessible_volume_A3"]) for r in sub)
                    near = [r for r in intervals if r["region"] == region and r["species"] == species]
                    exposed, n_exposed = weighted_mean(
                        [r for r in near if r["ion_present_at_start"] == "1"],
                        "mean_oxygen_turnover", "stable_liquid_cells")
                    unexposed, n_unexposed = weighted_mean(
                        [r for r in near if r["ion_present_at_start"] == "0"],
                        "mean_oxygen_turnover", "stable_liquid_cells")
                    rows.append({
                        "branch": branch, "minimum_liquid_oxygen": threshold,
                        "region": region, "species": species,
                        "ion_count": ion_count, "accessible_volume_A3": volume,
                        "ion_density_A3_inv": ion_count / volume if volume else math.nan,
                        "exposed_cell_intervals": n_exposed,
                        "unexposed_cell_intervals": n_unexposed,
                        "turnover_exposed_minus_unexposed": exposed - unexposed,
                    })
    by_key = {(r["branch"], r["minimum_liquid_oxygen"], r["species"], r["region"]): r for r in rows}
    for row in rows:
        far = by_key[(row["branch"], row["minimum_liquid_oxygen"], row["species"], "far_wall")]
        row["ion_density_over_far_wall"] = (
            row["ion_density_A3_inv"] / far["ion_density_A3_inv"]
            if far["ion_density_A3_inv"] > 0 else math.nan
        )
    output.mkdir(parents=True)
    write_tsv(output / "LIQUID-ADMISSION-SENSITIVITY.tsv", rows, tuple(rows[0]))
    summary = {"status": "PASS", "branches": 2, "thresholds": [1, 2, 3],
               "regions": ["front", "rear", "far_wall"], "rows": len(rows),
               "sensitivity_not_independent_replication": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_output_hashes(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.baseline, args.sensitivity, args.output), sort_keys=True))
