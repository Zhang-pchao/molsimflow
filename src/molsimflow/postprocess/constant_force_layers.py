"""Resolve height-layer transport, exchange, and density modes for water films."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.postprocess.constant_force_oxygen import (
    iter_oxygen_frames,
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)
from molsimflow.postprocess.surface_site_enrichment import identify_surface_sites


LAYER_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "layer_index",
    "z_low_A", "z_high_A", "count", "number_density_A3", "mean_vx_mps",
    "mean_vy_mps", "surface_flux_x_molecules_per_A_ps",
    "surface_flux_y_molecules_per_A_ps",
)
MODE_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "layer_index",
    "mode_x", "mode_y", "amplitude", "phase_rad",
)
EXCHANGE_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "from_layer",
    "to_layer", "molecule_count", "rate_per_ps",
)
RESIDENCE_FIELDS = (
    "case_id", "branch_id", "direction", "layer_index", "episodes",
    "mean_residence_ps", "median_residence_ps", "p95_residence_ps",
    "right_censored_episodes",
)
SITE_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "eligible_water",
    "assigned_water", "retained_assignments", "site_exchanges", "new_assignments",
    "lost_assignments", "mean_site_distance_A",
)
RESPONSE_FIELDS = (
    "case_id", "branch_id", "direction", "layer_index", "samples",
    "occupied_fraction", "mean_count", "valid_paired_velocity_samples",
    "mean_axis_velocity_mps", "baseline_axis_velocity_mps",
    "mean_excess_axis_velocity_mps", "block_excess_sem_mps",
    "mean_surface_flux_molecules_per_A_ps",
    "mean_excess_surface_flux_molecules_per_A_ps",
)


def _validate_contract(raw: Mapping[str, object]) -> None:
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    if float(raw.get("timestep_fs", 0.0)) <= 0.0:
        raise ValueError("timestep_fs must be positive")
    edges = np.asarray(raw.get("z_edges_A", []), dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("z_edges_A must be strictly increasing")
    modes = raw.get("density_modes", [[1, 0], [0, 1]])
    if not isinstance(modes, list) or any(len(mode) != 2 for mode in modes):
        raise ValueError("density_modes must contain integer [kx, ky] pairs")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    baseline_counts: dict[str, int] = defaultdict(int)
    identities = []
    for entry in cases:
        if not isinstance(entry, dict):
            raise ValueError("Each case entry must be an object")
        for key in ("case_id", "branch_id", "direction", "surface_z_A", "trajectories"):
            if key not in entry:
                raise ValueError(f"Case entry is missing {key}")
        direction = str(entry["direction"]).lower()
        if direction not in {"none", "x", "y"}:
            raise ValueError("direction must be none, x, or y")
        if not isinstance(entry["trajectories"], list) or not entry["trajectories"]:
            raise ValueError("trajectories must be a non-empty list")
        case_id = str(entry["case_id"])
        identities.append((case_id, str(entry["branch_id"])))
        baseline_counts[case_id] += int(direction == "none")
    if len(identities) != len(set(identities)):
        raise ValueError("case_id/branch_id pairs must be unique")
    bad = sorted(case for case, count in baseline_counts.items() if count != 1)
    if bad:
        raise ValueError(f"Each case_id must have exactly one direction=none branch: {bad}")


def density_mode(xy: np.ndarray, lengths_xy: np.ndarray, mode: tuple[int, int]) -> tuple[float, float]:
    if len(xy) == 0:
        return math.nan, math.nan
    phase = 2.0 * np.pi * (
        mode[0] * xy[:, 0] / lengths_xy[0] + mode[1] * xy[:, 1] / lengths_xy[1]
    )
    value = np.mean(np.exp(1j * phase))
    return float(abs(value)), float(np.angle(value))


def nearest_sites(
    xy: np.ndarray,
    sites_xy: np.ndarray,
    lengths_xy: np.ndarray,
    cutoff_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    if len(xy) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    query = np.mod(xy, lengths_xy)
    sites = np.mod(sites_xy, lengths_xy)
    distances, indices = cKDTree(sites, boxsize=lengths_xy).query(query, k=1)
    indices = np.asarray(indices, dtype=int)
    indices[np.asarray(distances) > cutoff_A] = -1
    return indices, np.asarray(distances, dtype=float)


def _site_coordinates(config: Mapping[str, object], base: Path) -> tuple[np.ndarray, list[dict]]:
    xyz = resolve_path(config["initial_xyz"], base)
    elements, coordinates, lengths = read_extxyz_positions(xyz)
    slab_range = tuple(int(value) for value in config["slab_range"])
    sites = identify_surface_sites(
        elements, coordinates, lengths,
        slab_range=slab_range,
        surface_z=float(config["surface_z_A"]),
        surface_depth=float(config.get("surface_depth_A", 3.5)),
        bond_cutoff=float(config.get("bond_cutoff_A", 1.25)),
    )
    requested = str(config.get("site_type", "SiOH"))
    sites = [row for row in sites if row["site_type"] == requested]
    if not sites:
        raise ValueError(f"No {requested} sites identified in {xyz}")
    return np.asarray([[row["x_A"], row["y_A"]] for row in sites]), sites


def _plot(layer_rows: list[dict], response_rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = sorted({(row["case_id"], row["branch_id"]) for row in layer_rows})
    figure, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 6.4), squeeze=False)
    for column, key in enumerate(keys):
        selected = [row for row in layer_rows if (row["case_id"], row["branch_id"]) == key]
        direction = selected[0]["direction"]
        for layer in sorted({int(row["layer_index"]) for row in selected}):
            rows = [row for row in selected if int(row["layer_index"]) == layer]
            axis_key = "mean_vx_mps" if direction == "x" else "mean_vy_mps"
            if direction == "none":
                axis_key = "mean_vx_mps"
            axes[0, column].plot(
                np.asarray([row["time_ps"] for row in rows]) / 1000.0,
                [row[axis_key] for row in rows], label=f"layer {layer}", alpha=0.8,
            )
        response = [row for row in response_rows if (row["case_id"], row["branch_id"]) == key]
        axes[1, column].bar(
            [row["layer_index"] for row in response],
            [row["mean_excess_axis_velocity_mps"] for row in response],
        )
        axes[0, column].set_title(f"{key[0]} / {key[1]}")
        axes[0, column].set_ylabel("Layer velocity (m/s)")
        axes[0, column].legend(frameon=False, fontsize=7)
        axes[1, column].set_xlabel("Layer index")
        axes[1, column].set_ylabel("F - F0 velocity (m/s)")
    figure.tight_layout()
    figure.savefig(output / "layered_transport_overview.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    output.mkdir(parents=True)
    base = contract_path.parent
    timestep_fs = float(raw["timestep_fs"])
    origin = int(raw.get("time_origin_step", 0))
    edges = np.asarray(raw["z_edges_A"], dtype=float)
    modes = [tuple(map(int, mode)) for mode in raw.get("density_modes", [[1, 0], [0, 1]])]
    block_ps = float(raw.get("block_ps", 1000.0))
    site_config = raw.get("surface_sites")
    sites_xy = None
    site_records: list[dict] = []
    if isinstance(site_config, dict):
        sites_xy, site_records = _site_coordinates(site_config, base)
        site_cutoff = float(site_config.get("assignment_cutoff_A", 3.5))
        site_layer = int(site_config.get("layer_index", 0))
    else:
        site_cutoff = math.nan; site_layer = -1

    layer_rows: list[dict] = []
    mode_rows: list[dict] = []
    exchange_rows: list[dict] = []
    site_rows: list[dict] = []
    residence_episodes: dict[tuple[str, str, str, int], list[tuple[float, bool]]] = defaultdict(list)
    input_rows = [{"path": str(contract_path), "size_bytes": contract_path.stat().st_size, "sha256": sha256(contract_path)}]
    if isinstance(site_config, dict):
        xyz_path = resolve_path(site_config["initial_xyz"], base)
        input_rows.append({"path": str(xyz_path), "size_bytes": xyz_path.stat().st_size, "sha256": sha256(xyz_path)})

    for entry in raw["cases"]:
        case_id, branch_id = str(entry["case_id"]), str(entry["branch_id"])
        direction = str(entry["direction"]).lower()
        surface_z = float(entry["surface_z_A"])
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        input_rows.extend({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths)
        previous_layers: np.ndarray | None = None
        previous_time: float | None = None
        previous_sites: dict[int, int] = {}
        residence_state: dict[int, tuple[int, float]] = {}
        final_time = math.nan
        for frame in iter_oxygen_frames(paths):
            if frame.velocities is None:
                raise ValueError(f"Velocities are required at step {frame.timestep}")
            time_ps = (frame.timestep - origin) * timestep_fs / 1000.0
            final_time = time_ps
            lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
            area = lengths[0] * lengths[1]
            dz = frame.coordinates[:, 2] - surface_z
            layer_index = np.digitize(dz, edges, right=False) - 1
            layer_index[(dz < edges[0]) | (dz >= edges[-1])] = -1
            for layer in range(len(edges) - 1):
                mask = layer_index == layer
                count = int(np.count_nonzero(mask))
                thickness = edges[layer + 1] - edges[layer]
                vx_Aps = frame.velocities[mask, 0]
                vy_Aps = frame.velocities[mask, 1]
                layer_rows.append({
                    "case_id": case_id, "branch_id": branch_id, "direction": direction,
                    "step": frame.timestep, "time_ps": time_ps, "layer_index": layer,
                    "z_low_A": edges[layer], "z_high_A": edges[layer + 1], "count": count,
                    "number_density_A3": count / (area * thickness),
                    "mean_vx_mps": 100.0 * float(np.mean(vx_Aps)) if count else math.nan,
                    "mean_vy_mps": 100.0 * float(np.mean(vy_Aps)) if count else math.nan,
                    "surface_flux_x_molecules_per_A_ps": float(np.sum(vx_Aps)) / area,
                    "surface_flux_y_molecules_per_A_ps": float(np.sum(vy_Aps)) / area,
                })
                for mode in modes:
                    amplitude, phase = density_mode(frame.coordinates[mask, :2], lengths[:2], mode)
                    mode_rows.append({
                        "case_id": case_id, "branch_id": branch_id, "direction": direction,
                        "step": frame.timestep, "time_ps": time_ps, "layer_index": layer,
                        "mode_x": mode[0], "mode_y": mode[1], "amplitude": amplitude,
                        "phase_rad": phase,
                    })

            if previous_layers is not None and previous_time is not None:
                dt_ps = time_ps - previous_time
                pairs, counts = np.unique(np.column_stack((previous_layers, layer_index)), axis=0, return_counts=True)
                for pair, count in zip(pairs, counts):
                    if int(pair[0]) == int(pair[1]):
                        continue
                    exchange_rows.append({
                        "case_id": case_id, "branch_id": branch_id, "direction": direction,
                        "step": frame.timestep, "time_ps": time_ps,
                        "from_layer": int(pair[0]), "to_layer": int(pair[1]),
                        "molecule_count": int(count), "rate_per_ps": int(count) / dt_ps,
                    })

            for atom_id, layer in zip(frame.atom_ids, layer_index):
                atom_id, layer = int(atom_id), int(layer)
                old = residence_state.get(atom_id)
                if old is None:
                    residence_state[atom_id] = (layer, time_ps)
                elif old[0] != layer:
                    residence_episodes[(case_id, branch_id, direction, old[0])].append((time_ps - old[1], False))
                    residence_state[atom_id] = (layer, time_ps)

            if sites_xy is not None:
                eligible = layer_index == site_layer
                eligible_ids = frame.atom_ids[eligible]
                assignments, distances = nearest_sites(frame.coordinates[eligible, :2], sites_xy, lengths[:2], site_cutoff)
                current_sites = {int(atom_id): int(site) for atom_id, site in zip(eligible_ids, assignments) if site >= 0}
                shared = set(previous_sites) & set(current_sites)
                retained = sum(previous_sites[atom_id] == current_sites[atom_id] for atom_id in shared)
                exchanges = len(shared) - retained
                site_rows.append({
                    "case_id": case_id, "branch_id": branch_id, "direction": direction,
                    "step": frame.timestep, "time_ps": time_ps,
                    "eligible_water": int(np.count_nonzero(eligible)), "assigned_water": len(current_sites),
                    "retained_assignments": retained, "site_exchanges": exchanges,
                    "new_assignments": len(set(current_sites) - set(previous_sites)),
                    "lost_assignments": len(set(previous_sites) - set(current_sites)),
                    "mean_site_distance_A": float(np.mean(distances[assignments >= 0])) if np.any(assignments >= 0) else math.nan,
                })
                previous_sites = current_sites
            previous_layers = layer_index.copy(); previous_time = time_ps
        if previous_time is None:
            raise ValueError(f"No frames analyzed for {case_id}/{branch_id}")
        for atom_id, (layer, start_time) in residence_state.items():
            residence_episodes[(case_id, branch_id, direction, layer)].append((final_time - start_time, True))

    residence_rows = []
    for (case_id, branch_id, direction, layer), episodes in sorted(residence_episodes.items()):
        durations = np.asarray([value for value, _ in episodes], dtype=float)
        residence_rows.append({
            "case_id": case_id, "branch_id": branch_id, "direction": direction,
            "layer_index": layer, "episodes": len(episodes),
            "mean_residence_ps": float(np.mean(durations)),
            "median_residence_ps": float(np.median(durations)),
            "p95_residence_ps": float(np.quantile(durations, 0.95)),
            "right_censored_episodes": sum(censored for _, censored in episodes),
        })

    baseline = {
        (row["case_id"], row["step"], row["layer_index"]): row
        for row in layer_rows if row["direction"] == "none"
    }
    response_groups: dict[
        tuple[str, str, str, int],
        list[tuple[float, float, float, float, float, float, float]],
    ] = defaultdict(list)
    for row in layer_rows:
        direction = row["direction"]
        if direction == "none":
            axis = float(row["mean_vx_mps"])
            base_axis = axis
            excess = 0.0
            flux = float(row["surface_flux_x_molecules_per_A_ps"])
            excess_flux = 0.0
        else:
            reference = baseline.get((row["case_id"], row["step"], row["layer_index"]))
            if reference is None:
                raise ValueError("Driven and baseline layer grids are not aligned")
            key = "mean_vx_mps" if direction == "x" else "mean_vy_mps"
            flux_key = "surface_flux_x_molecules_per_A_ps" if direction == "x" else "surface_flux_y_molecules_per_A_ps"
            axis = float(row[key]); base_axis = float(reference[key]); excess = axis - base_axis
            flux = float(row[flux_key])
            excess_flux = flux - float(reference[flux_key])
        response_groups[(row["case_id"], row["branch_id"], direction, int(row["layer_index"]))].append(
            (
                float(row["time_ps"]), axis, base_axis, excess, flux,
                float(row["count"]), excess_flux,
            )
        )
    response_rows = []
    for (case_id, branch_id, direction, layer), values in sorted(response_groups.items()):
        array = np.asarray(values, dtype=float)
        block_values = []
        block_ids = np.floor((array[:, 0] - array[0, 0]) / block_ps).astype(int)
        if len(block_ids) > 1 and block_ids[-1] > block_ids[-2]:
            block_ids[-1] = block_ids[-2]
        for block in np.unique(block_ids):
            block_values.append(float(np.nanmean(array[block_ids == block, 3])))
        sem = float(np.nanstd(block_values, ddof=1) / math.sqrt(len(block_values))) if len(block_values) > 1 else math.nan
        response_rows.append({
            "case_id": case_id, "branch_id": branch_id, "direction": direction,
            "layer_index": layer, "samples": len(values),
            "occupied_fraction": float(np.mean(array[:, 5] > 0.0)),
            "mean_count": float(np.mean(array[:, 5])),
            "valid_paired_velocity_samples": int(np.count_nonzero(np.isfinite(array[:, 3]))),
            "mean_axis_velocity_mps": float(np.nanmean(array[:, 1])),
            "baseline_axis_velocity_mps": float(np.nanmean(array[:, 2])),
            "mean_excess_axis_velocity_mps": float(np.nanmean(array[:, 3])),
            "block_excess_sem_mps": sem,
            "mean_surface_flux_molecules_per_A_ps": float(np.nanmean(array[:, 4])),
            "mean_excess_surface_flux_molecules_per_A_ps": float(np.nanmean(array[:, 6])),
        })

    write_tsv(output / "layer_timeseries.tsv", layer_rows, LAYER_FIELDS)
    write_tsv(output / "density_modes.tsv", mode_rows, MODE_FIELDS)
    write_tsv(output / "layer_exchange.tsv", exchange_rows, EXCHANGE_FIELDS)
    write_tsv(output / "residence_summary.tsv", residence_rows, RESIDENCE_FIELDS)
    write_tsv(output / "site_exchange.tsv", site_rows, SITE_FIELDS)
    write_tsv(output / "layer_response_summary.tsv", response_rows, RESPONSE_FIELDS)
    write_tsv(output / "input_manifest.tsv", input_rows, ("path", "size_bytes", "sha256"))
    if site_records:
        write_tsv(output / "surface_sites.tsv", site_records, tuple(site_records[0]))
    if bool(raw.get("write_plots", True)):
        _plot(layer_rows, response_rows, output)
    summary = {
        "status": "PASS", "case_branches": len(raw["cases"]),
        "frames": len({(row["case_id"], row["branch_id"], row["step"]) for row in layer_rows}),
        "layers": len(edges) - 1, "density_modes": [list(mode) for mode in modes],
        "layer_exchange_rows": len(exchange_rows), "residence_groups": len(residence_rows),
        "surface_sites": len(site_records), "site_assignment_rows": len(site_rows),
        "velocity_sem_is_within_trajectory_not_replicate_uncertainty": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Constant-force layered water transport\n\n"
        f"Analyzed {len(raw['cases'])} branches on {len(edges) - 1} height layers. "
        "The response table subtracts the time-aligned zero-force branch.\n\n"
        "Velocity and flux are molecular oxygen observables. Layer residence episodes ending at "
        "the trajectory boundary are right censored, and block SEM is a single-trajectory diagnostic.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
