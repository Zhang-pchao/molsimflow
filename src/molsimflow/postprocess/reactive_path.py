"""Reactive-path geometry, ion-defect, and hydrogen-bond wire descriptors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


BoxLengths = Union[float, Sequence[float]]


@dataclass(frozen=True)
class ReactiveSiteConfig:
    """Explicit one-based reactive-site and molecular-partition indices."""

    organic_oxygen_indices: Tuple[int, ...]
    donor_carbon_index: int
    target_oxygen_index: int
    peroxy_proximal_oxygen_index: int
    peroxy_attachment_carbon_index: int

    def validate(self) -> None:
        if not self.organic_oxygen_indices:
            raise ValueError("organic_oxygen_indices must not be empty")
        if len(set(self.organic_oxygen_indices)) != len(self.organic_oxygen_indices):
            raise ValueError("organic_oxygen_indices must be unique")
        indices = self.organic_oxygen_indices + (
            self.donor_carbon_index,
            self.target_oxygen_index,
            self.peroxy_proximal_oxygen_index,
            self.peroxy_attachment_carbon_index,
        )
        if any(index < 1 for index in indices):
            raise ValueError("Reactive atom indices must be one-based positive integers")
        if self.target_oxygen_index not in self.organic_oxygen_indices:
            raise ValueError("target_oxygen_index must identify an organic oxygen")
        if self.peroxy_proximal_oxygen_index not in self.organic_oxygen_indices:
            raise ValueError("peroxy_proximal_oxygen_index must identify an organic oxygen")
        if self.target_oxygen_index == self.peroxy_proximal_oxygen_index:
            raise ValueError("Target and proximal peroxide oxygen indices must differ")


@dataclass(frozen=True)
class ReactivePathConfig:
    """Geometric thresholds for frame-level reactive-path descriptors."""

    oxygen_h_assignment_A: float = 1.30
    state_bond_A: float = 1.25
    hbond_h_acceptor_A: float = 2.50
    hbond_oxygen_oxygen_A: float = 3.50
    hbond_angle_degree: float = 130.0

    def validate(self) -> None:
        if min(
            self.oxygen_h_assignment_A,
            self.state_bond_A,
            self.hbond_h_acceptor_A,
            self.hbond_oxygen_oxygen_A,
        ) <= 0.0:
            raise ValueError("Distance thresholds must be positive")
        if not 0.0 < self.hbond_angle_degree <= 180.0:
            raise ValueError("hbond_angle_degree must be in (0, 180]")


def _read_table(path: Path) -> List[Dict[str, str]]:
    delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return [dict(row) for row in reader]


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _read_xyz(path: Path) -> List[Dict[str, object]]:
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    atom_count = int(lines[0])
    atoms: List[Dict[str, object]] = []
    for line in lines[2 : atom_count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ atom row in {path}: {line}")
        atoms.append(
            {
                "element": fields[0],
                "xyz": (float(fields[1]), float(fields[2]), float(fields[3])),
            }
        )
    if len(atoms) != atom_count:
        raise ValueError(f"XYZ atom-count mismatch: {path}")
    return atoms


def _box_lengths(box_A: BoxLengths) -> Tuple[float, float, float]:
    if isinstance(box_A, (int, float)):
        value = float(box_A)
        lengths = (value, value, value)
    else:
        lengths = tuple(float(value) for value in box_A)
        if len(lengths) != 3:
            raise ValueError("Periodic box must be a scalar or three lengths")
    if min(lengths) <= 0.0:
        raise ValueError("Periodic box lengths must be positive")
    return lengths


def _vector(origin: Sequence[float], target: Sequence[float], box_A: BoxLengths) -> Tuple[float, float, float]:
    values = []
    for start, end, length in zip(origin, target, _box_lengths(box_A)):
        delta = float(end) - float(start)
        delta -= round(delta / length) * length
        values.append(delta)
    return tuple(values)  # type: ignore[return-value]


def _distance(a: Sequence[float], b: Sequence[float], box_A: BoxLengths) -> float:
    return math.sqrt(sum(value * value for value in _vector(a, b, box_A)))


def _angle_degree(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def _nearest(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    atoms: Sequence[Mapping[str, object]],
    box_A: BoxLengths,
) -> Tuple[float, int, int]:
    return min(
        (
            _distance(atoms[source]["xyz"], atoms[target]["xyz"], box_A),  # type: ignore[arg-type]
            source,
            target,
        )
        for source in source_ids
        for target in target_ids
        if source != target
    )


def _shortest_path(graph: Mapping[int, Sequence[int]], start: Optional[int], target: int) -> Optional[List[int]]:
    if start is None:
        return None
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in sorted(graph.get(node, ())):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _hbond_paths(
    atoms: Sequence[Mapping[str, object]],
    box_A: BoxLengths,
    oxygen_ids: Sequence[int],
    assigned_hydrogens: Mapping[int, Sequence[int]],
    start_oxygen: Optional[int],
    target_oxygen: int,
    config: ReactivePathConfig,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    directed: Dict[int, set[int]] = defaultdict(set)
    undirected: Dict[int, set[int]] = defaultdict(set)
    for donor in oxygen_ids:
        for hydrogen in assigned_hydrogens.get(donor, ()):
            h_to_donor = _vector(atoms[hydrogen]["xyz"], atoms[donor]["xyz"], box_A)  # type: ignore[arg-type]
            for acceptor in oxygen_ids:
                if acceptor == donor:
                    continue
                if _distance(atoms[donor]["xyz"], atoms[acceptor]["xyz"], box_A) > config.hbond_oxygen_oxygen_A:  # type: ignore[arg-type]
                    continue
                if _distance(atoms[hydrogen]["xyz"], atoms[acceptor]["xyz"], box_A) > config.hbond_h_acceptor_A:  # type: ignore[arg-type]
                    continue
                h_to_acceptor = _vector(atoms[hydrogen]["xyz"], atoms[acceptor]["xyz"], box_A)  # type: ignore[arg-type]
                if _angle_degree(h_to_donor, h_to_acceptor) < config.hbond_angle_degree:
                    continue
                directed[donor].add(acceptor)
                undirected[donor].add(acceptor)
                undirected[acceptor].add(donor)
    return (
        _shortest_path(directed, start_oxygen, target_oxygen),
        _shortest_path(undirected, start_oxygen, target_oxygen),
    )


def load_site_configs(path: Path) -> Tuple[Dict[str, ReactiveSiteConfig], ReactivePathConfig]:
    """Load per-system reactive indices and optional threshold overrides."""

    payload = json.loads(Path(path).read_text())
    systems = payload.get("systems", {})
    if not systems:
        raise ValueError("Site config must define a non-empty 'systems' mapping")
    site_configs = {
        str(name): ReactiveSiteConfig(
            organic_oxygen_indices=tuple(int(index) for index in values["organic_oxygen_indices"]),
            donor_carbon_index=int(values["donor_carbon_index"]),
            target_oxygen_index=int(values["target_oxygen_index"]),
            peroxy_proximal_oxygen_index=int(values["peroxy_proximal_oxygen_index"]),
            peroxy_attachment_carbon_index=int(values["peroxy_attachment_carbon_index"]),
        )
        for name, values in systems.items()
    }
    for site_config in site_configs.values():
        site_config.validate()
    threshold_values = payload.get("thresholds", {})
    path_config = ReactivePathConfig(**threshold_values)
    path_config.validate()
    return site_configs, path_config


def describe_reactive_frame(
    atoms: Sequence[Mapping[str, object]],
    box_A: BoxLengths,
    site_config: ReactiveSiteConfig,
    path_config: ReactivePathConfig = ReactivePathConfig(),
) -> Dict[str, object]:
    """Calculate protonation, peroxide, ion-defect, and water-wire descriptors."""

    site_config.validate()
    path_config.validate()
    hydrogen_ids = [index for index, atom in enumerate(atoms) if atom["element"] == "H"]
    oxygen_ids = [index for index, atom in enumerate(atoms) if atom["element"] == "O"]
    organic_oxygen_ids = [index - 1 for index in site_config.organic_oxygen_indices]
    organic_oxygen_set = set(organic_oxygen_ids)
    water_oxygen_ids = [index for index in oxygen_ids if index not in organic_oxygen_set]

    donor_carbon = site_config.donor_carbon_index - 1
    target_oxygen = site_config.target_oxygen_index - 1
    proximal_oxygen = site_config.peroxy_proximal_oxygen_index - 1
    attachment_carbon = site_config.peroxy_attachment_carbon_index - 1
    for index, expected in [
        *((index, "O") for index in organic_oxygen_ids),
        (donor_carbon, "C"),
        (target_oxygen, "O"),
        (proximal_oxygen, "O"),
        (attachment_carbon, "C"),
    ]:
        if index < 0 or index >= len(atoms) or atoms[index]["element"] != expected:
            raise ValueError(f"Configured atom {index + 1} is not element {expected}")
    if not hydrogen_ids:
        raise ValueError("Frame contains no hydrogen atoms")

    assigned_hydrogens: Dict[int, List[int]] = defaultdict(list)
    for hydrogen in hydrogen_ids:
        distance_A, oxygen, _ = _nearest(oxygen_ids, [hydrogen], atoms, box_A)
        if distance_A <= path_config.oxygen_h_assignment_A:
            assigned_hydrogens[oxygen].append(hydrogen)
    hydronium_ids = [oxygen for oxygen in water_oxygen_ids if len(assigned_hydrogens[oxygen]) >= 3]

    min_donor_h_A, _, nearest_donor_h = _nearest([donor_carbon], hydrogen_ids, atoms, box_A)
    min_target_h_A, _, nearest_target_h = _nearest([target_oxygen], hydrogen_ids, atoms, box_A)
    selected_hydronium: Optional[int] = None
    target_hydronium_A: Optional[float] = None
    if hydronium_ids:
        target_hydronium_A, _, selected_hydronium = _nearest(
            [target_oxygen], hydronium_ids, atoms, box_A
        )

    directed, undirected = _hbond_paths(
        atoms,
        box_A,
        water_oxygen_ids + [target_oxygen],
        assigned_hydrogens,
        selected_hydronium,
        target_oxygen,
        path_config,
    )

    if min_donor_h_A <= path_config.state_bond_A and min_target_h_A > path_config.state_bond_A:
        geometry_state = "reactant-like C-H"
    elif min_donor_h_A > path_config.state_bond_A and min_target_h_A <= path_config.state_bond_A:
        geometry_state = "product-like O-H"
    elif min_donor_h_A > path_config.state_bond_A and min_target_h_A > path_config.state_bond_A and hydronium_ids:
        geometry_state = "anion/hydronium-like"
    else:
        geometry_state = "mixed/unclassified"

    return {
        "geometry_state": geometry_state,
        "min_donorC_H_A": min_donor_h_A,
        "nearest_donorC_H_index": nearest_donor_h + 1,
        "min_targetO_H_A": min_target_h_A,
        "nearest_targetO_H_index": nearest_target_h + 1,
        "hydronium_count": len(hydronium_ids),
        "selected_hydronium_O_index": selected_hydronium + 1 if selected_hydronium is not None else "",
        "targetO_hydronium_A": target_hydronium_A if target_hydronium_A is not None else "",
        "directed_wire_bonds": len(directed) - 1 if directed else "",
        "directed_wire_O_indices": ">".join(str(index + 1) for index in directed) if directed else "none",
        "undirected_wire_bonds": len(undirected) - 1 if undirected else "",
        "undirected_wire_O_indices": ">".join(str(index + 1) for index in undirected) if undirected else "none",
        "peroxy_O_O_A": _distance(atoms[proximal_oxygen]["xyz"], atoms[target_oxygen]["xyz"], box_A),  # type: ignore[arg-type]
        "peroxy_attachment_C_O_A": _distance(
            atoms[attachment_carbon]["xyz"],  # type: ignore[arg-type]
            atoms[proximal_oxygen]["xyz"],  # type: ignore[arg-type]
            box_A,
        ),
    }


def analyze_reactive_path(
    manifest_path: Path,
    site_config_path: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """Analyze every manifest-backed XYZ frame and write frame/summary tables."""

    manifest = _read_table(manifest_path)
    if not manifest:
        raise ValueError("Manifest is empty")
    required = ("system", "state", "replicate", "xyz_path")
    missing = [column for column in required if column not in manifest[0]]
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    site_configs, path_config = load_site_configs(site_config_path)
    output_rows: List[Dict[str, object]] = []
    for source_index, row in enumerate(manifest):
        system = str(row.get("system", "")).strip()
        if system not in site_configs:
            raise ValueError(f"Manifest system '{system}' is absent from site config")
        xyz_value = str(row.get("xyz_path", "")).strip()
        if not xyz_value:
            raise ValueError(f"Manifest row {source_index} has no xyz_path")
        xyz_path = Path(xyz_value)
        if not xyz_path.is_absolute():
            xyz_path = Path(manifest_path).parent / xyz_path
        box_fields = ("box_x_A", "box_y_A", "box_z_A")
        if all(str(row.get(field, "")).strip() for field in box_fields):
            box_A: BoxLengths = tuple(float(row[field]) for field in box_fields)
        elif str(row.get("box_A", "")).strip():
            box_A = float(row["box_A"])
        else:
            raise ValueError(
                f"Manifest row {source_index} needs box_A or box_x_A/box_y_A/box_z_A"
            )
        box_x_A, box_y_A, box_z_A = _box_lengths(box_A)
        descriptors = describe_reactive_frame(
            _read_xyz(xyz_path), box_A, site_configs[system], path_config
        )
        output_rows.append(
            {
                "system": system,
                "state": row.get("state", ""),
                "stratum": row.get("stratum", ""),
                "replicate": row.get("replicate", ""),
                "source_row_index": source_index,
                "xyz_path": str(xyz_path),
                "box_A": row.get("box_A", ""),
                "box_x_A": box_x_A,
                "box_y_A": box_y_A,
                "box_z_A": box_z_A,
                **descriptors,
            }
        )

    grouped = Counter(
        (str(row["system"]), str(row["state"]), str(row["geometry_state"]))
        for row in output_rows
    )
    summary_rows = [
        {
            "system": system,
            "state": state,
            "geometry_state": geometry_state,
            "n": count,
            "mean_peroxy_O_O_A": mean(
                float(row["peroxy_O_O_A"])
                for row in output_rows
                if (row["system"], row["state"], row["geometry_state"])
                == (system, state, geometry_state)
            ),
            "mean_peroxy_attachment_C_O_A": mean(
                float(row["peroxy_attachment_C_O_A"])
                for row in output_rows
                if (row["system"], row["state"], row["geometry_state"])
                == (system, state, geometry_state)
            ),
            "directed_wire_connected_n": sum(
                row["directed_wire_bonds"] != ""
                for row in output_rows
                if (row["system"], row["state"], row["geometry_state"])
                == (system, state, geometry_state)
            ),
        }
        for (system, state, geometry_state), count in sorted(grouped.items())
    ]

    output_dir = Path(output_dir)
    outputs = {
        "frames": output_dir / "reactive_path_frames.tsv",
        "summary": output_dir / "reactive_path_summary.tsv",
        "metadata": output_dir / "reactive_path_metadata.json",
    }
    _write_tsv(outputs["frames"], output_rows)
    _write_tsv(outputs["summary"], summary_rows)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "analysis_scope": "Manifest-backed reactive-path geometry and water-wire descriptors; not kinetics.",
                "manifest_path": str(Path(manifest_path).resolve()),
                "site_config_path": str(Path(site_config_path).resolve()),
                "frame_count": len(output_rows),
                "thresholds": path_config.__dict__,
            },
            indent=2,
        )
        + "\n"
    )
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Input frame manifest TSV or CSV"
    )
    parser.add_argument(
        "--site-config", type=Path, required=True, help="Reactive-site JSON configuration"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for output tables"
    )
    args = parser.parse_args(argv)
    outputs = analyze_reactive_path(args.manifest, args.site_config, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
