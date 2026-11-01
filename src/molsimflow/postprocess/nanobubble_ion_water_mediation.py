"""Fixed-substrate-grid water/ion diagnostics for driven nanobubbles.

This module intentionally keeps oxygen identity, proton-defect occupancy, and
contact-edge geometry separate. It measures sampled (not continuous) renewal.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records_selected
from molsimflow.postprocess.constant_force_oxygen import write_output_hashes, write_tsv


def minimum_image(delta: float, length: float) -> float:
    return delta - length * math.floor(delta / length + 0.5)


def renewal(previous: set[int], current: set[int]) -> tuple[float, float, float]:
    """Sampled oxygen turnover, retained fraction, and net count change."""
    union = len(previous | current)
    shared = len(previous & current)
    return (
        1.0 - shared / union if union else math.nan,
        shared / len(previous) if previous else math.nan,
        float(len(current) - len(previous)),
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _expected_steps(raw: dict) -> list[int]:
    first, last, stride = (int(raw[key]) for key in ("first_step", "last_step", "stride_steps"))
    if first < 0 or last <= first or stride <= 0 or (last - first) % stride:
        raise ValueError("Invalid step grid")
    return list(range(first, last + 1, stride))


def _load_geometry(raw: dict, steps: list[int]) -> dict[int, dict[str, str]]:
    wanted = set(steps)
    result: dict[int, dict[str, str]] = {}
    for row in _read_tsv(Path(raw["geometry_tsv"])):
        step = int(row["step"])
        if step in wanted:
            if step in result:
                raise ValueError(f"Duplicate geometry at {step}")
            result[step] = row
    if set(result) != wanted:
        raise ValueError(f"Geometry coverage mismatch: {len(result)} of {len(steps)}")
    return result


def _iter_coordinates(raw: dict, steps: list[int]):
    wanted = set(steps)
    previous = None
    for source in raw["state_sources"]:
        low, high = int(source["minimum_step"]), int(source["maximum_step"])
        path = Path(source["path"])
        found = 0
        for frame in iter_lammps_dump_records_selected(
            path,
            accept_timestep=lambda step, low=low, high=high: step in wanted and low <= step <= high,
            stop_after_timestep=high,
        ):
            if previous is not None and frame.timestep <= previous:
                raise ValueError(f"Non-monotonic or duplicate state frame: {frame.timestep}")
            previous = frame.timestep
            found += 1
            yield frame
        if found == 0:
            raise ValueError(f"No selected frames in {path}")


def _cell(x: float, y: float, bounds: np.ndarray, nx: int, ny: int) -> tuple[int, int]:
    lx, ly = bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]
    return (
        min(nx - 1, int(((x - bounds[0, 0]) % lx) / lx * nx)),
        min(ny - 1, int(((y - bounds[1, 0]) % ly) / ly * ny)),
    )


def _region(cell: tuple[int, int], nx: int, ny: int, bounds: np.ndarray,
            geometry: dict[str, str], direction: str, edge_half: float,
            transverse_padding: float) -> str:
    lx, ly = bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]
    x = bounds[0, 0] + (cell[0] + 0.5) * lx / nx
    y = bounds[1, 0] + (cell[1] + 0.5) * ly / ny
    drive, transverse = (y, x) if direction == "y" else (x, y)
    ld, lt = (ly, lx) if direction == "y" else (lx, ly)
    transverse_center = float(geometry[f"bubble_center_{'x' if direction == 'y' else 'y'}_A"])
    transverse_half = float(geometry["transverse_width_q05_q95_A"]) / 2 + transverse_padding
    if abs(minimum_image(transverse - transverse_center, lt)) > transverse_half:
        return "far_wall"
    front = float(geometry["advancing_position_A"])
    rear = float(geometry["receding_position_A"])
    if abs(minimum_image(drive - front, ld)) <= edge_half:
        return "front"
    if abs(minimum_image(drive - rear, ld)) <= edge_half:
        return "rear"
    return "other_wall"


def run_case(raw: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    steps = _expected_steps(raw)
    geometry = _load_geometry(raw, steps)
    # Ion candidate samples are only atom IDs and positions, never tracked protons.
    ions: dict[int, dict[str, list[tuple[int, float, float, float]]]] = {
        step: defaultdict(list) for step in steps
    }
    seen: set[tuple[int, str, int]] = set()
    for source in raw["ion_sources"]:
        low, high = int(source["minimum_step"]), int(source["maximum_step"])
        with gzip.open(source["path"], "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = int(row["step"])
                if step not in ions or not low <= step <= high or row["species"] not in raw["species"]:
                    continue
                key = (step, row["species"], int(row["atom_id"]))
                if key in seen:
                    raise ValueError(f"Duplicate ion sample {key}")
                seen.add(key)
                ions[step][row["species"]].append(
                    (int(row["atom_id"]), float(row["x_A"]), float(row["y_A"]), float(row["z_A"]))
                )
    surface_z = float(raw["surface_z_A"])
    layer_height = float(raw["layer_height_A"])
    cell_width = float(raw["cell_width_A"])
    min_oxygen = int(raw["minimum_liquid_oxygen"])
    edge_half = float(raw["edge_halfwidth_A"])
    pad = float(raw["transverse_padding_A"])
    if min(layer_height, cell_width, edge_half) <= 0 or min_oxygen < 1:
        raise ValueError("Grid and liquid-admission parameters must be positive")
    frame_rows: list[dict] = []
    interval_rows: list[dict] = []
    previous = None
    visited = []
    for frame in _iter_coordinates(raw, steps):
        step = frame.timestep
        visited.append(step)
        if step not in geometry:
            raise ValueError(f"Unmatched coordinate step {step}")
        bounds = frame.bounds
        lx, ly = bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]
        nx, ny = max(1, int(lx / cell_width)), max(1, int(ly / cell_width))
        volume = lx * ly * layer_height / (nx * ny)
        ids = np.asarray([int(row[frame.atom_fields.index("id")]) for row in frame.atom_rows], dtype=np.int64)
        types = np.asarray([int(row[frame.atom_fields.index("type")]) for row in frame.atom_rows], dtype=np.int16)
        xyz = np.asarray([[float(row[frame.atom_fields.index(axis)]) for axis in "xyz"] for row in frame.atom_rows], dtype=float)
        if len(ids) != int(raw["expected_atoms"]) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Atom identity/completeness failure at {step}")
        lo, hi = map(int, raw["solution_atom_range"])
        selection = (ids >= lo) & (ids <= hi) & (types == int(raw.get("oxygen_type", 2)))
        selection &= (xyz[:, 2] >= surface_z) & (xyz[:, 2] < surface_z + layer_height)
        water: dict[tuple[int, int], set[int]] = defaultdict(set)
        for atom_id, point in zip(ids[selection], xyz[selection]):
            water[_cell(point[0], point[1], bounds, nx, ny)].add(int(atom_id))
        ion_cells: dict[str, dict[tuple[int, int], set[int]]] = {}
        for species in raw["species"]:
            by_cell: dict[tuple[int, int], set[int]] = defaultdict(set)
            for atom_id, x, y, z in ions[step][species]:
                if surface_z <= z < surface_z + layer_height:
                    by_cell[_cell(x, y, bounds, nx, ny)].add(atom_id)
            ion_cells[species] = by_cell
        regions = {
            (ix, iy): _region((ix, iy), nx, ny, bounds, geometry[step], raw["direction"], edge_half, pad)
            for ix in range(nx) for iy in range(ny)
        }
        for region in ("front", "rear", "other_wall", "far_wall"):
            cells = [cell for cell, value in regions.items() if value == region]
            liquid = [cell for cell in cells if len(water[cell]) >= min_oxygen]
            for species in raw["species"]:
                count = sum(len(ion_cells[species][cell]) for cell in liquid)
                oxygen = sum(len(water[cell]) for cell in liquid)
                frame_rows.append({
                    "case_id": raw["case_id"], "direction": raw["direction"], "step": step,
                    "region": region, "species": species, "region_cells": len(cells),
                    "liquid_cells": len(liquid), "oxygen_count": oxygen,
                    "ion_count": count, "accessible_volume_A3": len(liquid) * volume,
                    "ion_density_A3_inv": count / (len(liquid) * volume) if liquid else math.nan,
                    "ion_per_oxygen": count / oxygen if oxygen else math.nan,
                })
        if previous is not None:
            old_step, old_water, old_ions, old_regions, old_bounds, old_grid = previous
            if (nx, ny) != old_grid or not np.allclose(bounds, old_bounds, atol=1e-8):
                raise ValueError("Fixed substrate grid changed between frames")
            for region in ("front", "rear", "other_wall", "far_wall"):
                stable = [cell for cell, value in regions.items()
                          if value == region and old_regions[cell] == region
                          and len(water[cell]) >= min_oxygen
                          and len(old_water[cell]) >= min_oxygen]
                for species in raw["species"]:
                    for ion_present in (False, True):
                        chosen = [cell for cell in stable if bool(old_ions[species][cell]) == ion_present]
                        measured = [renewal(old_water[cell], water[cell]) for cell in chosen]
                        defect_persistent = sum(bool(old_ions[species][cell]) and bool(ion_cells[species][cell]) for cell in chosen)
                        carrier_persistent = sum(bool(old_ions[species][cell] & ion_cells[species][cell]) for cell in chosen)
                        interval_rows.append({
                            "case_id": raw["case_id"], "direction": raw["direction"],
                            "start_step": old_step, "end_step": step, "region": region,
                            "species": species, "ion_present_at_start": int(ion_present),
                            "stable_liquid_cells": len(chosen),
                            "mean_oxygen_turnover": float(np.mean([m[0] for m in measured])) if measured else math.nan,
                            "mean_oxygen_retained": float(np.mean([m[1] for m in measured])) if measured else math.nan,
                            "mean_oxygen_count_change": float(np.mean([m[2] for m in measured])) if measured else math.nan,
                            "defect_cell_persistent": defect_persistent,
                            "same_carrier_persistent": carrier_persistent,
                        })
        previous = (step, water, ion_cells, regions, bounds.copy(), (nx, ny))
    if visited != steps:
        raise ValueError(f"State coverage mismatch: {len(visited)} of {len(steps)}")
    output.mkdir(parents=True, exist_ok=False)
    write_tsv(output / "ion_accessible_volume.tsv", frame_rows, tuple(frame_rows[0]))
    write_tsv(output / "fixed_grid_renewal.tsv", interval_rows, tuple(interval_rows[0]))
    summary = {
        "status": "PASS", "case_id": raw["case_id"], "direction": raw["direction"],
        "frames": len(visited), "first_step": visited[0], "last_step": visited[-1],
        "sample_interval_ps": int(raw["stride_steps"]) * float(raw["timestep_fs"]) / 1000,
        "fixed_grid": [nx, ny], "cell_volume_A3": volume,
        "species": raw["species"], "liquid_cell_criterion": min_oxygen,
        "ion_is_geometric_candidate": True,
        "turnover_is_sampled_not_continuous": True,
        "blocks_are_not_independent_replicates": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_output_hashes(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_case(json.loads(args.contract.read_text()), args.output), sort_keys=True))


if __name__ == "__main__":
    main()
