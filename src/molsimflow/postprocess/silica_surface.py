"""Analyze silica-surface CH3/OH terminations in extended XYZ models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


KNOWN_SIDES = ("lower", "upper", "all")
ELEMENT_COLUMNS = ("Si", "O", "C", "H")
SUMMARY_COLUMNS = (
    "case_label",
    "source_path",
    "side",
    "surface_atom_count",
    "surface_area_A2",
    "side_split_z_A",
    "element_Si",
    "element_O",
    "element_C",
    "element_H",
    "ch3_count",
    "oh_count",
    "terminal_count",
    "ch3_fraction",
    "oh_fraction",
    "ch3_density_per_nm2",
    "oh_density_per_nm2",
    "unassigned_surface_H",
    "z_min_A",
    "z_max_A",
    "z_mean_A",
    "si_z_mean_A",
    "o_z_mean_A",
    "c_z_mean_A",
    "h_z_mean_A",
    "ch3_anchor_z_mean_A",
    "oh_anchor_z_mean_A",
    "ch3_nearest_si_mean_A",
    "oh_nearest_si_mean_A",
    "ch3_h_distance_mean_A",
    "oh_h_distance_mean_A",
)
GROUP_COLUMNS = (
    "case_label",
    "side",
    "group_type",
    "anchor_atom_id",
    "anchor_symbol",
    "x_A",
    "y_A",
    "z_A",
    "h_count",
    "h_atom_ids",
    "mean_h_distance_A",
    "nearest_si_distance_A",
)


@dataclass(frozen=True)
class Atom:
    atom_id: int
    symbol: str
    x: float
    y: float
    z: float

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    xyz_path: Path
    surface_atom_count: Optional[int] = None


@dataclass(frozen=True)
class XyzModel:
    atom_count: int
    comment: str
    lattice: np.ndarray
    atoms: Tuple[Atom, ...]
    requested_counts: Mapping[str, int]


@dataclass(frozen=True)
class AnalysisConfig:
    c_h_cutoff_A: float = 1.25
    o_h_cutoff_A: float = 1.25
    side_split_z_A: Optional[float] = None
    make_plots: bool = True


@dataclass(frozen=True)
class CaseAnalysis:
    case: CaseSpec
    model: XyzModel
    surface_atoms: Tuple[Atom, ...]
    side_split_z_A: float
    surface_area_A2: float
    summary_rows: Tuple[Dict[str, object], ...]
    group_rows: Tuple[Dict[str, object], ...]
    warning_rows: Tuple[Dict[str, object], ...]


def _format_float(value: object, digits: int = 6) -> object:
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.{digits}f}"
        return "nan"
    return value


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else math.nan


def _parse_lattice(comment: str) -> np.ndarray:
    match = re.search(r'Lattice="([^"]+)"', comment)
    if not match:
        return np.full((3, 3), math.nan, dtype=float)
    values = [float(value) for value in match.group(1).split()]
    if len(values) != 9:
        raise ValueError(f"Expected 9 lattice values, found {len(values)}")
    return np.array(values, dtype=float).reshape(3, 3)


def _parse_requested_counts(comment: str) -> Dict[str, int]:
    match = re.search(r'requested_counts="((?:\\.|[^"])*)"', comment)
    if not match:
        return {}
    text = match.group(1).replace('\\"', '"')
    raw = json.loads(text)
    return {str(key): int(value) for key, value in raw.items()}


def read_extxyz(path: Path) -> XyzModel:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is too short: {path}")
    atom_count = int(lines[0].split()[0])
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"XYZ file ended before {atom_count} atom lines: {path}")
    atoms: List[Atom] = []
    for index, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid atom line {index} in {path}: {line!r}")
        atoms.append(
            Atom(
                atom_id=index,
                symbol=parts[0],
                x=float(parts[1]),
                y=float(parts[2]),
                z=float(parts[3]),
            )
        )
    comment = lines[1]
    return XyzModel(
        atom_count=atom_count,
        comment=comment,
        lattice=_parse_lattice(comment),
        atoms=tuple(atoms),
        requested_counts=_parse_requested_counts(comment),
    )


def formula_atom_count(formula: str) -> int:
    clean = re.sub(r"[^A-Za-z0-9]", "", formula)
    if not clean:
        raise ValueError(f"Cannot infer atom count from empty formula: {formula!r}")
    total = 0
    position = 0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", clean):
        if match.start() != position:
            raise ValueError(f"Cannot parse molecular formula: {formula!r}")
        total += int(match.group(2) or "1")
        position = match.end()
    if position != len(clean) or total <= 0:
        raise ValueError(f"Cannot parse molecular formula: {formula!r}")
    return total


def infer_surface_atom_count(model: XyzModel, explicit_count: Optional[int] = None) -> int:
    if explicit_count is not None:
        if explicit_count <= 0 or explicit_count > model.atom_count:
            raise ValueError(f"Invalid explicit surface atom count: {explicit_count}")
        return int(explicit_count)
    if not model.requested_counts:
        raise ValueError(
            "Cannot infer surface atom count because requested_counts metadata is missing; "
            "pass --surface-atom-count or add a manifest surface_atom_count column."
        )
    fluid_atoms = 0
    for name, count in model.requested_counts.items():
        fluid_atoms += int(count) * formula_atom_count(name)
    surface_count = model.atom_count - fluid_atoms
    if surface_count <= 0:
        raise ValueError(
            f"Inferred nonpositive surface atom count ({surface_count}) from requested_counts"
        )
    return surface_count


def _cell_lengths(lattice: np.ndarray) -> Tuple[float, float, float]:
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        return (math.nan, math.nan, math.nan)
    return tuple(float(np.linalg.norm(vector)) for vector in lattice)


def _surface_area_xy(lattice: np.ndarray) -> float:
    if lattice.shape != (3, 3) or not np.isfinite(lattice[:2]).all():
        return math.nan
    return float(np.linalg.norm(np.cross(lattice[0], lattice[1])))


def _coords(atoms: Sequence[Atom]) -> np.ndarray:
    if not atoms:
        return np.empty((0, 3), dtype=float)
    return np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)


def _minimum_image_distances(center: Atom, neighbor_coords: np.ndarray, lengths: Tuple[float, float, float]) -> np.ndarray:
    if len(neighbor_coords) == 0:
        return np.empty((0,), dtype=float)
    deltas = neighbor_coords - center.coord
    for axis in (0, 1):
        length = lengths[axis]
        if math.isfinite(length) and length > 0:
            deltas[:, axis] -= length * np.rint(deltas[:, axis] / length)
    return np.sqrt(np.sum(deltas * deltas, axis=1))


def _hydrogen_neighbors(
    anchors: Sequence[Atom],
    hydrogens: Sequence[Atom],
    cutoff_A: float,
    lengths: Tuple[float, float, float],
) -> Dict[int, Tuple[Tuple[int, ...], Tuple[float, ...]]]:
    h_coords = _coords(hydrogens)
    out: Dict[int, Tuple[Tuple[int, ...], Tuple[float, ...]]] = {}
    for anchor in anchors:
        distances = _minimum_image_distances(anchor, h_coords, lengths)
        neighbor_indices = np.where(distances <= cutoff_A)[0]
        ordered = sorted(neighbor_indices, key=lambda idx: float(distances[idx]))
        out[anchor.atom_id] = (
            tuple(hydrogens[int(idx)].atom_id for idx in ordered),
            tuple(float(distances[int(idx)]) for idx in ordered),
        )
    return out


def _nearest_distance_to_atoms(
    anchors: Sequence[Atom],
    neighbors: Sequence[Atom],
    lengths: Tuple[float, float, float],
) -> Dict[int, float]:
    neighbor_coords = _coords(neighbors)
    out: Dict[int, float] = {}
    for anchor in anchors:
        distances = _minimum_image_distances(anchor, neighbor_coords, lengths)
        out[anchor.atom_id] = float(np.min(distances)) if len(distances) else math.nan
    return out


def _side_for_z(z: float, split_z: float) -> str:
    return "lower" if z < split_z else "upper"


def _choose_side_split(surface_atoms: Sequence[Atom], explicit_split: Optional[float]) -> float:
    if explicit_split is not None:
        return float(explicit_split)
    si_z = [atom.z for atom in surface_atoms if atom.symbol == "Si"]
    reference = si_z or [atom.z for atom in surface_atoms if atom.symbol != "H"] or [atom.z for atom in surface_atoms]
    return 0.5 * (min(reference) + max(reference))


def _row_atom_subset(atoms: Sequence[Atom], side: str, split_z: float) -> List[Atom]:
    if side == "all":
        return list(atoms)
    return [atom for atom in atoms if _side_for_z(atom.z, split_z) == side]


def _group_subset(group_rows: Sequence[Mapping[str, object]], side: str, group_type: str) -> List[Mapping[str, object]]:
    return [row for row in group_rows if row["group_type"] == group_type and (side == "all" or row["side"] == side)]


def _element_count(atoms: Sequence[Atom], symbol: str) -> int:
    return sum(1 for atom in atoms if atom.symbol == symbol)


def _element_z_mean(atoms: Sequence[Atom], symbol: str) -> float:
    return _mean([atom.z for atom in atoms if atom.symbol == symbol])


def _build_summary_row(
    case: CaseSpec,
    source_path: Path,
    side: str,
    atoms: Sequence[Atom],
    all_group_rows: Sequence[Mapping[str, object]],
    surface_area_A2: float,
    split_z: float,
) -> Dict[str, object]:
    ch3_rows = _group_subset(all_group_rows, side, "CH3")
    oh_rows = _group_subset(all_group_rows, side, "OH")
    ch3_count = len(ch3_rows)
    oh_count = len(oh_rows)
    terminal_count = ch3_count + oh_count
    denominator_area = surface_area_A2 * (2.0 if side == "all" else 1.0)
    area_nm2 = denominator_area / 100.0 if math.isfinite(denominator_area) else math.nan
    ch3_h = sum(int(row["h_count"]) for row in ch3_rows)
    oh_h = sum(int(row["h_count"]) for row in oh_rows)
    h_count = _element_count(atoms, "H")
    z_values = [atom.z for atom in atoms]
    row: Dict[str, object] = {
        "case_label": case.label,
        "source_path": str(source_path),
        "side": side,
        "surface_atom_count": len(atoms),
        "surface_area_A2": surface_area_A2,
        "side_split_z_A": split_z,
        "element_Si": _element_count(atoms, "Si"),
        "element_O": _element_count(atoms, "O"),
        "element_C": _element_count(atoms, "C"),
        "element_H": h_count,
        "ch3_count": ch3_count,
        "oh_count": oh_count,
        "terminal_count": terminal_count,
        "ch3_fraction": ch3_count / terminal_count if terminal_count else math.nan,
        "oh_fraction": oh_count / terminal_count if terminal_count else math.nan,
        "ch3_density_per_nm2": ch3_count / area_nm2 if area_nm2 and math.isfinite(area_nm2) else math.nan,
        "oh_density_per_nm2": oh_count / area_nm2 if area_nm2 and math.isfinite(area_nm2) else math.nan,
        "unassigned_surface_H": h_count - ch3_h - oh_h,
        "z_min_A": min(z_values) if z_values else math.nan,
        "z_max_A": max(z_values) if z_values else math.nan,
        "z_mean_A": _mean(z_values),
        "si_z_mean_A": _element_z_mean(atoms, "Si"),
        "o_z_mean_A": _element_z_mean(atoms, "O"),
        "c_z_mean_A": _element_z_mean(atoms, "C"),
        "h_z_mean_A": _element_z_mean(atoms, "H"),
        "ch3_anchor_z_mean_A": _mean([float(row["z_A"]) for row in ch3_rows]),
        "oh_anchor_z_mean_A": _mean([float(row["z_A"]) for row in oh_rows]),
        "ch3_nearest_si_mean_A": _mean([float(row["nearest_si_distance_A"]) for row in ch3_rows]),
        "oh_nearest_si_mean_A": _mean([float(row["nearest_si_distance_A"]) for row in oh_rows]),
        "ch3_h_distance_mean_A": _mean([float(row["mean_h_distance_A"]) for row in ch3_rows]),
        "oh_h_distance_mean_A": _mean([float(row["mean_h_distance_A"]) for row in oh_rows]),
    }
    return row


def analyze_case(case: CaseSpec, config: AnalysisConfig) -> CaseAnalysis:
    model = read_extxyz(case.xyz_path)
    surface_count = infer_surface_atom_count(model, case.surface_atom_count)
    surface_atoms = tuple(model.atoms[:surface_count])
    split_z = _choose_side_split(surface_atoms, config.side_split_z_A)
    lengths = _cell_lengths(model.lattice)
    area_A2 = _surface_area_xy(model.lattice)

    atoms_by_symbol: Dict[str, List[Atom]] = {symbol: [] for symbol in ELEMENT_COLUMNS}
    for atom in surface_atoms:
        atoms_by_symbol.setdefault(atom.symbol, []).append(atom)

    c_h = _hydrogen_neighbors(atoms_by_symbol.get("C", []), atoms_by_symbol.get("H", []), config.c_h_cutoff_A, lengths)
    o_h = _hydrogen_neighbors(atoms_by_symbol.get("O", []), atoms_by_symbol.get("H", []), config.o_h_cutoff_A, lengths)
    nearest_si_for_c = _nearest_distance_to_atoms(atoms_by_symbol.get("C", []), atoms_by_symbol.get("Si", []), lengths)
    nearest_si_for_o = _nearest_distance_to_atoms(atoms_by_symbol.get("O", []), atoms_by_symbol.get("Si", []), lengths)

    group_rows: List[Dict[str, object]] = []
    warning_rows: List[Dict[str, object]] = []
    for carbon in atoms_by_symbol.get("C", []):
        h_ids, h_distances = c_h[carbon.atom_id]
        if len(h_ids) == 3:
            group_rows.append(
                {
                    "case_label": case.label,
                    "side": _side_for_z(carbon.z, split_z),
                    "group_type": "CH3",
                    "anchor_atom_id": carbon.atom_id,
                    "anchor_symbol": carbon.symbol,
                    "x_A": carbon.x,
                    "y_A": carbon.y,
                    "z_A": carbon.z,
                    "h_count": len(h_ids),
                    "h_atom_ids": ";".join(str(value) for value in h_ids),
                    "mean_h_distance_A": _mean(list(h_distances)),
                    "nearest_si_distance_A": nearest_si_for_c.get(carbon.atom_id, math.nan),
                }
            )
        else:
            warning_rows.append(
                {
                    "case_label": case.label,
                    "atom_id": carbon.atom_id,
                    "symbol": "C",
                    "issue": "unexpected_C_H_neighbor_count",
                    "value": len(h_ids),
                }
            )
    for oxygen in atoms_by_symbol.get("O", []):
        h_ids, h_distances = o_h[oxygen.atom_id]
        if len(h_ids) == 1:
            group_rows.append(
                {
                    "case_label": case.label,
                    "side": _side_for_z(oxygen.z, split_z),
                    "group_type": "OH",
                    "anchor_atom_id": oxygen.atom_id,
                    "anchor_symbol": oxygen.symbol,
                    "x_A": oxygen.x,
                    "y_A": oxygen.y,
                    "z_A": oxygen.z,
                    "h_count": len(h_ids),
                    "h_atom_ids": ";".join(str(value) for value in h_ids),
                    "mean_h_distance_A": _mean(list(h_distances)),
                    "nearest_si_distance_A": nearest_si_for_o.get(oxygen.atom_id, math.nan),
                }
            )
        elif len(h_ids) > 1:
            warning_rows.append(
                {
                    "case_label": case.label,
                    "atom_id": oxygen.atom_id,
                    "symbol": "O",
                    "issue": "unexpected_O_H_neighbor_count",
                    "value": len(h_ids),
                }
            )

    summary_rows = []
    for side in KNOWN_SIDES:
        side_atoms = _row_atom_subset(surface_atoms, side, split_z)
        summary_rows.append(_build_summary_row(case, case.xyz_path, side, side_atoms, group_rows, area_A2, split_z))

    return CaseAnalysis(
        case=case,
        model=model,
        surface_atoms=surface_atoms,
        side_split_z_A=split_z,
        surface_area_A2=area_A2,
        summary_rows=tuple(summary_rows),
        group_rows=tuple(group_rows),
        warning_rows=tuple(warning_rows),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(row.get(key, "")) for key in fieldnames})


def _comparison_rows(
    summary_rows: Sequence[Mapping[str, object]],
    reference_case: Optional[str],
    target_case: Optional[str],
    comparison_label: str,
) -> List[Dict[str, object]]:
    labels = []
    for row in summary_rows:
        label = str(row["case_label"])
        if label not in labels:
            labels.append(label)
    if len(labels) < 2:
        return []
    reference = reference_case or labels[1]
    target = target_case or labels[0]
    label = comparison_label or f"{target}_minus_{reference}"
    by_key = {(str(row["case_label"]), str(row["side"])): row for row in summary_rows}
    rows: List[Dict[str, object]] = []
    numeric_columns = [column for column in SUMMARY_COLUMNS if column not in {"case_label", "source_path", "side"}]
    for side in KNOWN_SIDES:
        ref = by_key.get((reference, side))
        tgt = by_key.get((target, side))
        if ref is None or tgt is None:
            continue
        out: Dict[str, object] = {
            "comparison_label": label,
            "reference_case": reference,
            "target_case": target,
            "side": side,
        }
        for column in numeric_columns:
            try:
                out[f"delta_{column}"] = float(tgt[column]) - float(ref[column])
            except (TypeError, ValueError):
                out[f"delta_{column}"] = math.nan
        rows.append(out)
    return rows


def _upper_minus_lower_rows(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    labels = []
    for row in summary_rows:
        label = str(row["case_label"])
        if label not in labels:
            labels.append(label)
    by_key = {(str(row["case_label"]), str(row["side"])): row for row in summary_rows}
    numeric_columns = [column for column in SUMMARY_COLUMNS if column not in {"case_label", "source_path", "side"}]
    rows: List[Dict[str, object]] = []
    for label in labels:
        lower = by_key.get((label, "lower"))
        upper = by_key.get((label, "upper"))
        if lower is None or upper is None:
            continue
        out: Dict[str, object] = {"case_label": label, "delta": "upper_minus_lower"}
        for column in numeric_columns:
            try:
                out[f"delta_{column}"] = float(upper[column]) - float(lower[column])
            except (TypeError, ValueError):
                out[f"delta_{column}"] = math.nan
        rows.append(out)
    return rows


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("matplotlib is required for silica-surface plotting") from exc
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )
    return plt


def _case_labels(summary_rows: Sequence[Mapping[str, object]]) -> List[str]:
    labels: List[str] = []
    for row in summary_rows:
        label = str(row["case_label"])
        if label not in labels:
            labels.append(label)
    return labels


def _summary_by_case_side(summary_rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], Mapping[str, object]]:
    return {(str(row["case_label"]), str(row["side"])): row for row in summary_rows}


def plot_termination_counts(summary_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    plt = _load_pyplot()
    labels = _case_labels(summary_rows)
    by_key = _summary_by_case_side(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    colors = {"CH3": "#4C78A8", "OH": "#F58518"}
    for ax, side in zip(axes, ("lower", "upper")):
        ch3 = [float(by_key[(label, side)]["ch3_count"]) for label in labels]
        oh = [float(by_key[(label, side)]["oh_count"]) for label in labels]
        ax.bar(x - width / 2, ch3, width, label="CH3", color=colors["CH3"])
        ax.bar(x + width / 2, oh, width, label="OH", color=colors["OH"])
        ax.set_title(side)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("termination count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_element_counts(summary_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    plt = _load_pyplot()
    labels = _case_labels(summary_rows)
    by_key = _summary_by_case_side(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    colors = {"Si": "#4C78A8", "O": "#54A24B", "C": "#F58518", "H": "#B279A2"}
    x = np.arange(len(labels), dtype=float)
    for ax, side in zip(axes, ("lower", "upper")):
        bottom = np.zeros(len(labels), dtype=float)
        for element in ELEMENT_COLUMNS:
            values = np.array([float(by_key[(label, side)][f"element_{element}"]) for label in labels])
            ax.bar(x, values, 0.58, bottom=bottom, label=element, color=colors[element])
            bottom += values
        ax.set_title(side)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("surface atom count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right", ncols=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_termination_xy(group_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    plt = _load_pyplot()
    labels: List[str] = []
    for row in group_rows:
        label = str(row["case_label"])
        if label not in labels:
            labels.append(label)
    if not labels:
        return
    fig, axes = plt.subplots(len(labels), 2, figsize=(6.6, 2.7 * len(labels)), squeeze=False)
    colors = {"CH3": "#4C78A8", "OH": "#F58518"}
    for row_idx, label in enumerate(labels):
        for col_idx, side in enumerate(("lower", "upper")):
            ax = axes[row_idx][col_idx]
            for group_type in ("CH3", "OH"):
                rows = [row for row in group_rows if row["case_label"] == label and row["side"] == side and row["group_type"] == group_type]
                ax.scatter(
                    [float(row["x_A"]) for row in rows],
                    [float(row["y_A"]) for row in rows],
                    s=7,
                    alpha=0.78,
                    linewidths=0,
                    label=group_type,
                    color=colors[group_type],
                )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label} {side}")
            ax.set_xlabel("x (A)")
            ax.set_ylabel("y (A)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_surface_z_distributions(analyses: Sequence[CaseAnalysis], output: Path) -> None:
    plt = _load_pyplot()
    fig, axes = plt.subplots(len(analyses), 1, figsize=(6.6, 2.2 * len(analyses)), squeeze=False)
    colors = {"Si": "#4C78A8", "O": "#54A24B", "C": "#F58518", "H": "#B279A2"}
    for ax, analysis in zip(axes[:, 0], analyses):
        for element in ELEMENT_COLUMNS:
            z_values = [atom.z for atom in analysis.surface_atoms if atom.symbol == element]
            if z_values:
                ax.hist(z_values, bins=45, histtype="step", linewidth=1.1, label=element, color=colors[element])
        ax.axvline(analysis.side_split_z_A, color="#6E6E6E", linewidth=0.9, linestyle="--")
        ax.set_title(analysis.case.label)
        ax.set_xlabel("z (A)")
        ax.set_ylabel("count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncols=4, loc="upper center")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> List[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [_format_float(row.get(column, ""), digits=3) for column in columns]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return lines


def write_report(
    path: Path,
    analyses: Sequence[CaseAnalysis],
    summary_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    upper_lower_rows: Sequence[Mapping[str, object]],
    figure_dir: Path,
) -> None:
    lines: List[str] = ["# Silica surface termination analysis", ""]
    lines.append("## Inputs")
    for analysis in analyses:
        lines.append(
            f"- {analysis.case.label}: {analysis.case.xyz_path} "
            f"(surface atoms={len(analysis.surface_atoms)}, split_z={analysis.side_split_z_A:.3f} A)"
        )
    lines.append("")
    lines.append("## Surface termination counts")
    display_rows = [row for row in summary_rows if row["side"] in {"lower", "upper", "all"}]
    lines.extend(
        _markdown_table(
            display_rows,
            [
                "case_label",
                "side",
                "element_Si",
                "element_O",
                "element_C",
                "element_H",
                "ch3_count",
                "oh_count",
                "ch3_fraction",
                "oh_fraction",
                "unassigned_surface_H",
            ],
        )
    )
    lines.append("")
    if comparison_rows:
        lines.append("## Case difference")
        lines.extend(
            _markdown_table(
                comparison_rows,
                [
                    "comparison_label",
                    "side",
                    "delta_element_O",
                    "delta_element_C",
                    "delta_element_H",
                    "delta_ch3_count",
                    "delta_oh_count",
                    "delta_terminal_count",
                    "delta_surface_atom_count",
                ],
            )
        )
        lines.append("")
    if upper_lower_rows:
        lines.append("## Upper-lower difference")
        lines.extend(
            _markdown_table(
                upper_lower_rows,
                [
                    "case_label",
                    "delta_ch3_count",
                    "delta_oh_count",
                    "delta_element_Si",
                    "delta_element_O",
                    "delta_element_C",
                    "delta_element_H",
                    "delta_surface_atom_count",
                ],
            )
        )
        lines.append("")
    lines.append("## Figures")
    for name in [
        "termination_counts_by_side.png",
        "surface_element_counts_by_side.png",
        "termination_xy_map.png",
        "surface_z_distribution.png",
    ]:
        lines.append(f"- {figure_dir / name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    cases: Sequence[CaseSpec],
    output_dir: Path,
    figure_dir: Optional[Path] = None,
    config: Optional[AnalysisConfig] = None,
    reference_case: Optional[str] = None,
    target_case: Optional[str] = None,
    comparison_label: str = "",
) -> Dict[str, Path]:
    if not cases:
        raise ValueError("At least one case is required")
    cfg = config or AnalysisConfig()
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir) if figure_dir is not None else output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    analyses = [analyze_case(case, cfg) for case in cases]
    summary_rows = [row for analysis in analyses for row in analysis.summary_rows]
    group_rows = [row for analysis in analyses for row in analysis.group_rows]
    warning_rows = [row for analysis in analyses for row in analysis.warning_rows]
    case_delta_rows = _comparison_rows(summary_rows, reference_case, target_case, comparison_label)
    upper_lower_rows = _upper_minus_lower_rows(summary_rows)

    paths = {
        "summary_csv": output_dir / "surface_summary.csv",
        "groups_csv": output_dir / "termination_groups.csv",
        "case_delta_csv": output_dir / "case_comparison_delta.csv",
        "upper_lower_csv": output_dir / "upper_minus_lower_delta.csv",
        "warnings_csv": output_dir / "surface_analysis_warnings.csv",
        "json": output_dir / "surface_analysis_summary.json",
        "report": output_dir / "surface_analysis_report.md",
    }
    _write_csv(paths["summary_csv"], summary_rows, SUMMARY_COLUMNS)
    _write_csv(paths["groups_csv"], group_rows, GROUP_COLUMNS)
    delta_columns = list(case_delta_rows[0].keys()) if case_delta_rows else ["comparison_label", "reference_case", "target_case", "side"]
    _write_csv(paths["case_delta_csv"], case_delta_rows, delta_columns)
    upper_lower_columns = list(upper_lower_rows[0].keys()) if upper_lower_rows else ["case_label", "delta"]
    _write_csv(paths["upper_lower_csv"], upper_lower_rows, upper_lower_columns)
    warning_columns = ["case_label", "atom_id", "symbol", "issue", "value"]
    _write_csv(paths["warnings_csv"], warning_rows, warning_columns)

    json_payload = {
        "cases": [
            {
                "case_label": analysis.case.label,
                "source_path": str(analysis.case.xyz_path),
                "atom_count": analysis.model.atom_count,
                "surface_atom_count": len(analysis.surface_atoms),
                "requested_counts": dict(analysis.model.requested_counts),
                "side_split_z_A": analysis.side_split_z_A,
                "surface_area_A2": analysis.surface_area_A2,
            }
            for analysis in analyses
        ],
        "summary_rows": summary_rows,
        "case_comparison_delta": case_delta_rows,
        "upper_minus_lower_delta": upper_lower_rows,
        "warning_count": len(warning_rows),
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    if cfg.make_plots:
        plot_termination_counts(summary_rows, figure_dir / "termination_counts_by_side.png")
        plot_element_counts(summary_rows, figure_dir / "surface_element_counts_by_side.png")
        plot_termination_xy(group_rows, figure_dir / "termination_xy_map.png")
        plot_surface_z_distributions(analyses, figure_dir / "surface_z_distribution.png")
    write_report(paths["report"], analyses, summary_rows, case_delta_rows, upper_lower_rows, figure_dir)
    return paths


def _parse_case_arg(value: str) -> CaseSpec:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("--case must be LABEL:XYZ_PATH or LABEL:XYZ_PATH:SURFACE_ATOM_COUNT")
    surface_count = int(parts[2]) if len(parts) == 3 and parts[2] else None
    return CaseSpec(label=parts[0], xyz_path=Path(parts[1]), surface_atom_count=surface_count)


def _read_manifest(path: Path) -> List[CaseSpec]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = []
        for row in reader:
            label = row.get("case_label") or row.get("label")
            xyz = row.get("xyz_path") or row.get("path") or row.get("model_xyz")
            if not label or not xyz:
                raise ValueError("Manifest rows require case_label and xyz_path columns")
            raw_count = (row.get("surface_atom_count") or "").strip()
            rows.append(CaseSpec(label=label, xyz_path=Path(xyz), surface_atom_count=int(raw_count) if raw_count else None))
        return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="CSV with case_label,xyz_path[,surface_atom_count]")
    input_group.add_argument("--case", action="append", type=_parse_case_arg, help="LABEL:XYZ_PATH[:SURFACE_ATOM_COUNT]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--reference-case", help="Reference case for target-minus-reference deltas")
    parser.add_argument("--target-case", help="Target case for target-minus-reference deltas")
    parser.add_argument("--comparison-label", default="")
    parser.add_argument("--side-split-z-A", type=float)
    parser.add_argument("--c-h-cutoff-A", type=float, default=1.25)
    parser.add_argument("--o-h-cutoff-A", type=float, default=1.25)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases = _read_manifest(args.manifest) if args.manifest is not None else list(args.case or [])
    paths = run_analysis(
        cases=cases,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        config=AnalysisConfig(
            c_h_cutoff_A=args.c_h_cutoff_A,
            o_h_cutoff_A=args.o_h_cutoff_A,
            side_split_z_A=args.side_split_z_A,
            make_plots=not args.no_plots,
        ),
        reference_case=args.reference_case,
        target_case=args.target_case,
        comparison_label=args.comparison_label,
    )
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
