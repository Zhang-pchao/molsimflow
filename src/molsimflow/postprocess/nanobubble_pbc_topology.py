"""Detect true periodic winding cycles in an N2 molecular adjacency graph."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from molsimflow.io.lammps_dump import iter_lammps_dump_records_selected
from molsimflow.postprocess.constant_force_oxygen import write_output_hashes, write_tsv
from molsimflow.postprocess.nanobubble_flow_response import _molecule_centers


def winding_components(points: np.ndarray, bounds: np.ndarray, cutoff_A: float) -> list[dict]:
    """Connected components with graph-cycle winding in periodic X/Y.

    A cluster crossing a box face only once can still be unwrapped to a finite
    object; only inconsistent lattice lifts around a graph cycle imply true
    periodic spanning.
    """
    if cutoff_A <= 0:
        raise ValueError("cutoff_A must be positive")
    points = np.asarray(points, dtype=float)
    lengths = bounds[:, 1] - bounds[:, 0]
    pseudo_z = max(1e5, 10 * lengths[2])
    shifted = np.column_stack((
        (points[:, 0] - bounds[0, 0]) % lengths[0],
        (points[:, 1] - bounds[1, 0]) % lengths[1],
        points[:, 2] - bounds[2, 0] + 0.25 * pseudo_z,
    ))
    tree = cKDTree(shifted, boxsize=[lengths[0], lengths[1], pseudo_z])
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in points]
    crossing_edges = 0
    for left, right in tree.query_pairs(cutoff_A):
        displacement = points[right, :2] - points[left, :2]
        shift = -np.rint(displacement / lengths[:2]).astype(int)
        if np.any(shift):
            crossing_edges += 1
        move = int(shift[0]), int(shift[1])
        adjacency[left].append((right, move))
        adjacency[right].append((left, (-move[0], -move[1])))
    visited: dict[int, tuple[int, int]] = {}
    components = []
    for start in range(len(points)):
        if start in visited:
            continue
        visited[start] = (0, 0)
        queue = deque([start])
        members = []
        winding = [False, False]
        while queue:
            current = queue.popleft()
            members.append(current)
            lift = visited[current]
            for neighbor, move in adjacency[current]:
                proposed = (lift[0] + move[0], lift[1] + move[1])
                if neighbor in visited:
                    winding[0] |= proposed[0] != visited[neighbor][0]
                    winding[1] |= proposed[1] != visited[neighbor][1]
                else:
                    visited[neighbor] = proposed
                    queue.append(neighbor)
        components.append({
            "size": len(members), "first_index": min(members),
            "winding_x": int(winding[0]), "winding_y": int(winding[1]),
        })
    return sorted(components, key=lambda item: (-item["size"], item["first_index"]))


def analyze(contract: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    first = int(contract["first_step"])
    last = int(contract["last_step"])
    stride = int(contract["stride_steps"])
    expected = list(range(first, last + 1, stride))
    wanted = set(expected)
    nitrogen = tuple(map(int, contract["nitrogen_atom_range"]))
    excluded_nitrogen_ids = {int(item) for item in contract.get("excluded_nitrogen_ids", [])}
    cutoffs = tuple(map(float, contract["cutoffs_A"]))
    rows = []
    observed = []
    for source in contract["sources"]:
        low, high = int(source["minimum_step"]), int(source["maximum_step"])
        frames = iter_lammps_dump_records_selected(
            Path(source["path"]),
            accept_timestep=lambda step, low=low, high=high: step in wanted and low <= step <= high,
            stop_after_timestep=high,
        )
        count = 0
        for frame in frames:
            count += 1
            observed.append(frame.timestep)
            fields = {name: index for index, name in enumerate(frame.atom_fields)}
            required = {"id", "type", "x", "y", "z"}
            if not required.issubset(fields):
                raise ValueError(f"N2 frame missing {required - fields.keys()}")
            atoms = sorted((int(row[fields["id"]]),
                            [float(row[fields[axis]]) for axis in "xyz"])
                           for row in frame.atom_rows)
            ids = [item[0] for item in atoms]
            if ids != list(range(nitrogen[0], nitrogen[1] + 1)):
                raise ValueError(f"Incomplete N2 identities at step {frame.timestep}")
            points = _molecule_centers(np.asarray([item[1] for item in atoms]), frame.bounds)
            keep = [ids[index] not in excluded_nitrogen_ids
                    and ids[index + 1] not in excluded_nitrogen_ids
                    for index in range(0, len(ids), 2)]
            points = points[np.asarray(keep, dtype=bool)]
            if len(points) < 1:
                raise ValueError("No non-relocated N2 molecules")
            for cutoff in cutoffs:
                components = winding_components(points, frame.bounds, cutoff)
                main = components[0]
                rows.append({
                    "case_id": contract["case_id"], "direction": contract["direction"],
                    "step": frame.timestep, "cutoff_A": cutoff,
                    "main_n2_molecules": main["size"],
                    "excluded_n2_molecules": len(keep) - len(points),
                    "component_count": len(components),
                    "main_winding_x": main["winding_x"],
                    "main_winding_y": main["winding_y"],
                })
        if count == 0:
            raise ValueError(f"No admitted frames in {source['path']}")
    if observed != expected:
        raise ValueError(f"N2 frame coverage mismatch: {len(observed)} of {len(expected)}")
    output.mkdir(parents=True)
    write_tsv(output / "PBC-GRAPH-FRAMES.tsv", rows, tuple(rows[0]))
    summary = {
        "status": "PASS", "case_id": contract["case_id"], "direction": contract["direction"],
        "frames": len(expected), "first_step": first, "last_step": last,
        "cutoffs_A": cutoffs, "graph_winding_not_width_proxy": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_output_hashes(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(json.loads(args.contract.read_text()), args.output), sort_keys=True))
