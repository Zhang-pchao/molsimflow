"""Bridge-centered electrostatic proxy descriptors.

This module migrates the reusable parts of the legacy bridge-electrostatics
scripts into a path-explicit API.  Inputs are prepared CSV tables: one long ion
position/species table and, optionally, one frame-level bridge table.  The
calculations are proxy descriptors, not a self-consistent electrostatic solver.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


E_CHARGE_C = 1.602176634e-19
EPS0_F_PER_M = 8.8541878128e-12
KJMOL_PER_A3_TO_GPA = 1.660539067

DEFAULT_SPECIES_ORDER = (
    "H3O_plus",
    "H_plus_surf",
    "Na_plus",
    "Cl_minus",
    "OH_minus_bulk",
    "OH_minus_surf",
)

SPECIES_ALIASES = {
    "h3o": "H3O_plus",
    "h3o_plus": "H3O_plus",
    "h3o+": "H3O_plus",
    "na": "Na_plus",
    "na_plus": "Na_plus",
    "na+": "Na_plus",
    "h_surface": "H_plus_surf",
    "h_plus_surf": "H_plus_surf",
    "h+(surf)": "H_plus_surf",
    "cl": "Cl_minus",
    "cl_minus": "Cl_minus",
    "cl-": "Cl_minus",
    "oh": "OH_minus_bulk",
    "oh_bulk": "OH_minus_bulk",
    "oh_minus_bulk": "OH_minus_bulk",
    "oh-": "OH_minus_bulk",
    "oh_surface": "OH_minus_surf",
    "oh_surf": "OH_minus_surf",
    "oh_minus_surf": "OH_minus_surf",
    "outside_trace_region": "outside_trace_region",
}

SPECIES_CHARGE = {
    "H3O_plus": 1.0,
    "H_plus_surf": 1.0,
    "Na_plus": 1.0,
    "Cl_minus": -1.0,
    "OH_minus_bulk": -1.0,
    "OH_minus_surf": -1.0,
    "outside_trace_region": 0.0,
    "unknown": 0.0,
}


@dataclass(frozen=True)
class BridgeElectrostaticsConfig:
    """Settings for bridge-centered electrostatic proxy calculations."""

    frame_column: str = "global_frame"
    time_column: str = "time_ns"
    species_column: str = "species_canonical"
    charge_column: str = "species_charge_e"
    s_column: str = "bridge_axis_s_A"
    rho_column: str = "bridge_axis_rho_A"
    gap_column: str = "analysis_surface_gap_A"
    fallback_gap_columns: Tuple[str, ...] = ("dynamic_surface_gap_est_A", "surface_gap_A", "surface_gap_estimate_A")
    profile_s_min_A: float = -35.0
    profile_s_max_A: float = 35.0
    profile_bin_width_A: float = 1.0
    profile_rho_max_A: float = 10.0
    core_s_half_width_A: float = 8.0
    core_rho_max_A: float = 6.5
    surface_shell_width_A: float = 4.0
    shell_s_inner_A: float = 8.0
    shell_s_outer_A: float = 22.0
    shell_rho_max_A: float = 10.0
    gap_bin_width_A: float = 2.0
    epsilon_r: float = 78.5
    species_order: Tuple[str, ...] = DEFAULT_SPECIES_ORDER
    box_lengths_A: Optional[Tuple[float, float, float]] = None
    case_label: str = ""

    def validate(self) -> None:
        if self.profile_bin_width_A <= 0:
            raise ValueError("profile_bin_width_A must be positive")
        if self.profile_s_max_A <= self.profile_s_min_A:
            raise ValueError("profile_s_max_A must be greater than profile_s_min_A")
        if self.profile_rho_max_A <= 0 or self.core_rho_max_A <= 0 or self.shell_rho_max_A <= 0:
            raise ValueError("rho cutoffs must be positive")
        if self.core_s_half_width_A < 0 or self.surface_shell_width_A < 0:
            raise ValueError("bridge widths must be non-negative")
        if self.shell_s_outer_A <= self.shell_s_inner_A:
            raise ValueError("shell_s_outer_A must be greater than shell_s_inner_A")
        if self.gap_bin_width_A <= 0:
            raise ValueError("gap_bin_width_A must be positive")
        if self.epsilon_r <= 0:
            raise ValueError("epsilon_r must be positive")


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a CSV table into row dictionaries and field names."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader], [str(field) for field in reader.fieldnames]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    """Write rows to CSV, creating the parent directory if needed."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames) if fieldnames is not None else ordered_fieldnames(rows)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format_csv_value(row.get(name, "")) for name in names})


def ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case_label",
        "global_frame",
        "frame",
        "time_ns",
        "s_center_A",
        "s_left_A",
        "s_right_A",
        "gap_left_A",
        "gap_right_A",
        "gap_center_A",
        "region",
        "species",
        "descriptor",
        "mean",
        "std",
        "n_frames",
    ]
    out = [key for key in preferred if key in keys]
    out.extend(key for key in keys if key not in out)
    return out


def canonical_species(value: object) -> str:
    """Map common legacy species labels onto canonical names."""

    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return "unknown"
    return SPECIES_ALIASES.get(text, SPECIES_ALIASES.get(text.lower(), text))


def charge_for_species(value: object) -> float:
    """Return the default formal charge for a canonical or legacy species name."""

    return float(SPECIES_CHARGE.get(canonical_species(value), 0.0))


def prepare_gap_columns(
    frame_rows: Sequence[Mapping[str, object]],
    *,
    mode: str = "asis",
    dynamic_gap_column: str = "dynamic_surface_gap_est_A",
    d3d_column: str = "d3d_all",
    nominal_radius_a_A: float = 19.0,
    nominal_radius_b_A: float = 19.0,
    output_gap_column: str = "analysis_surface_gap_A",
    strict_precontact_only: bool = False,
    strict_gap_max_A: Optional[float] = None,
    strict_gap_max_tolerance_A: float = 0.5,
) -> List[Dict[str, object]]:
    """Add or update a frame-level analysis surface-gap column."""

    if mode not in {"asis", "dynamic", "nominal"}:
        raise ValueError("mode must be one of asis, dynamic, nominal")
    out: List[Dict[str, object]] = []
    first_nominal_gap = math.nan
    for row in frame_rows:
        new = dict(row)
        dynamic_gap = _as_float(new.get(dynamic_gap_column))
        d3d = _as_float(new.get(d3d_column))
        nominal_gap = d3d - float(nominal_radius_a_A) - float(nominal_radius_b_A) if math.isfinite(d3d) else math.nan
        new["dynamic_surface_gap_est_A_fuzzy_cluster"] = dynamic_gap
        new["nominal_surface_gap_A"] = nominal_gap
        if mode == "nominal":
            new[output_gap_column] = nominal_gap
            new["surface_gap_definition"] = (
                f"nominal:{d3d_column}-{float(nominal_radius_a_A):.3f}-{float(nominal_radius_b_A):.3f}"
            )
        elif mode == "dynamic":
            new[output_gap_column] = dynamic_gap
            new["surface_gap_definition"] = f"dynamic:{dynamic_gap_column}"
        elif output_gap_column not in new or not math.isfinite(_as_float(new.get(output_gap_column))):
            new[output_gap_column] = dynamic_gap if math.isfinite(dynamic_gap) else nominal_gap
            new["surface_gap_definition"] = "asis:fallback"
        else:
            new["surface_gap_definition"] = new.get("surface_gap_definition", "asis")
        gap = _as_float(new.get(output_gap_column))
        if not math.isfinite(first_nominal_gap) and math.isfinite(nominal_gap):
            first_nominal_gap = nominal_gap
        out.append(new)

    max_gap = strict_gap_max_A
    if max_gap is None and mode == "nominal" and math.isfinite(first_nominal_gap):
        max_gap = first_nominal_gap + float(strict_gap_max_tolerance_A)

    filtered: List[Dict[str, object]] = []
    for row in out:
        gap = _as_float(row.get(output_gap_column))
        row["strict_gap_max_A_applied"] = max_gap if strict_precontact_only and max_gap is not None else math.nan
        if strict_precontact_only:
            if not math.isfinite(gap) or gap < 0:
                continue
            if max_gap is not None and gap > float(max_gap):
                continue
        filtered.append(row)
    return filtered


def select_frame_rows(
    frame_rows: Sequence[Mapping[str, object]],
    *,
    frame_column: str = "global_frame",
    time_column: str = "time_ns",
    start_time_ns: Optional[float] = None,
    end_time_ns: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Filter frame rows by optional time and frame-count limits."""

    rows = []
    for row in frame_rows:
        time = _as_float(row.get(time_column))
        if start_time_ns is not None and math.isfinite(time) and time < float(start_time_ns) - 1e-12:
            continue
        if end_time_ns is not None and math.isfinite(time) and time > float(end_time_ns) + 1e-12:
            continue
        rows.append(dict(row))
    rows = sorted(rows, key=lambda row: (_as_float(row.get(time_column)), _as_float(row.get(frame_column))))
    if max_frames is not None and max_frames > 0:
        keep_keys = {_frame_key(row.get(frame_column)) for row in rows[: int(max_frames)]}
        rows = [row for row in rows if _frame_key(row.get(frame_column)) in keep_keys]
    return rows


def add_bridge_coordinates(
    ion_rows: Sequence[Mapping[str, object]],
    frame_rows: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    config: Optional[BridgeElectrostaticsConfig] = None,
    force: bool = False,
) -> List[Dict[str, object]]:
    """Derive bridge-axis ``s`` and ``rho`` coordinates from Cartesian columns."""

    cfg = config or BridgeElectrostaticsConfig()
    frame_lookup = _frame_lookup(frame_rows or (), cfg.frame_column)
    out: List[Dict[str, object]] = []
    for row in ion_rows:
        new = dict(row)
        if not force and math.isfinite(_as_float(new.get(cfg.s_column))) and math.isfinite(_as_float(new.get(cfg.rho_column))):
            out.append(new)
            continue
        frame = frame_lookup.get(_frame_key(new.get(cfg.frame_column)), {})
        pos = _vector_from_row(new, ("x_A", "y_A", "z_A"))
        center = _vector_from_row_with_fallback(new, frame, ("bridge_center_x_A", "bridge_center_y_A", "bridge_center_z_A"))
        bubble_a = _vector_from_row_with_fallback(
            new,
            frame,
            ("bubble_A_center_x_A", "bubble_A_center_y_A", "bubble_A_center_z_A"),
        )
        bubble_b = _vector_from_row_with_fallback(
            new,
            frame,
            ("bubble_B_center_x_A", "bubble_B_center_y_A", "bubble_B_center_z_A"),
        )
        if any(not math.isfinite(value) for value in (*pos, *center, *bubble_a, *bubble_b)):
            new[cfg.s_column] = math.nan
            new[cfg.rho_column] = math.nan
            out.append(new)
            continue
        axis = _minimum_image(np.asarray(bubble_b) - np.asarray(bubble_a), cfg.box_lengths_A)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12 or not math.isfinite(norm):
            unit = np.zeros(3, dtype=float)
            s_value = math.nan
            rho_value = math.nan
        else:
            unit = axis / norm
            disp = _minimum_image(np.asarray(pos) - np.asarray(center), cfg.box_lengths_A)
            s_value = float(np.dot(disp, unit))
            rho2 = max(float(np.dot(disp, disp) - s_value * s_value), 0.0)
            rho_value = math.sqrt(rho2)
        new[cfg.s_column] = s_value
        new[cfg.rho_column] = rho_value
        new["bridge_s_A"] = s_value
        new["bridge_rho_A"] = rho_value
        new["bridge_axis_unit_x"] = float(unit[0])
        new["bridge_axis_unit_y"] = float(unit[1])
        new["bridge_axis_unit_z"] = float(unit[2])
        out.append(new)
    return out


def charge_profile(
    ion_rows: Sequence[Mapping[str, object]],
    frame_rows: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    config: Optional[BridgeElectrostaticsConfig] = None,
) -> List[Dict[str, object]]:
    """Build a bridge-axis charge-density profile."""

    cfg = config or BridgeElectrostaticsConfig()
    cfg.validate()
    bins = _bins(cfg.profile_s_min_A, cfg.profile_s_max_A, cfg.profile_bin_width_A)
    n_frames = max(1, _count_frames(frame_rows, ion_rows, cfg.frame_column))
    species_order = _species_order(ion_rows, cfg)
    total_charge = [0.0 for _ in range(len(bins) - 1)]
    total_count = [0 for _ in range(len(bins) - 1)]
    species_charge = {species: [0.0 for _ in range(len(bins) - 1)] for species in species_order}
    species_count = {species: [0 for _ in range(len(bins) - 1)] for species in species_order}
    for row in ion_rows:
        q = _charge(row, cfg)
        if q == 0.0:
            continue
        s = _as_float(row.get(cfg.s_column, row.get("bridge_s_A")))
        rho = _as_float(row.get(cfg.rho_column, row.get("bridge_rho_A")))
        if not (math.isfinite(s) and math.isfinite(rho)):
            continue
        if rho > cfg.profile_rho_max_A or s < cfg.profile_s_min_A or s >= cfg.profile_s_max_A:
            continue
        index = int(math.floor((s - cfg.profile_s_min_A) / cfg.profile_bin_width_A))
        if index < 0 or index >= len(total_charge):
            continue
        species = canonical_species(row.get(cfg.species_column, row.get("current_trace_species")))
        total_charge[index] += q
        total_count[index] += 1
        if species in species_charge:
            species_charge[species][index] += q
            species_count[species][index] += 1
    volume = math.pi * cfg.profile_rho_max_A * cfg.profile_rho_max_A * cfg.profile_bin_width_A
    rows: List[Dict[str, object]] = []
    cumulative = 0.0
    for index, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        q_per_frame = total_charge[index] / n_frames
        cumulative += q_per_frame
        out: Dict[str, object] = {
            "s_center_A": 0.5 * (left + right),
            "s_left_A": left,
            "s_right_A": right,
            "n_frames": n_frames,
            "bin_volume_A3": volume,
            "net_charge_e_per_frame": q_per_frame,
            "charge_density_e_per_A3": q_per_frame / volume,
            "ion_count_per_frame": total_count[index] / n_frames,
            "cumulative_charge_e_per_frame": cumulative,
        }
        for species in species_order:
            out[f"{species}_charge_e_per_frame"] = species_charge[species][index] / n_frames
            out[f"{species}_count_per_frame"] = species_count[species][index] / n_frames
        rows.append(out)
    return rows


def poisson_proxy(profile_rows: Sequence[Mapping[str, object]], *, epsilon_r: float = 78.5) -> List[Dict[str, object]]:
    """Integrate a 1D unscreened Poisson-like potential proxy from charge density."""

    rows = [dict(row) for row in profile_rows]
    if not rows:
        return []
    if len(rows) < 2:
        for row in rows:
            row["charge_density_C_per_m3"] = _as_float(row.get("charge_density_e_per_A3")) * E_CHARGE_C / 1e-30
            row["electric_field_proxy_V_per_m"] = 0.0
            row["poisson_potential_proxy_V_zero_edge"] = 0.0
            row["poisson_potential_proxy_mV_zero_edge"] = 0.0
        return rows
    centers = np.asarray([_as_float(row.get("s_center_A")) for row in rows], dtype=float)
    densities = np.asarray([_as_float(row.get("charge_density_e_per_A3"), 0.0) for row in rows], dtype=float)
    dx_A = float(np.nanmedian(np.diff(centers)))
    dx_m = dx_A * 1e-10
    rho_c_m3 = densities * E_CHARGE_C / 1e-30
    eps = EPS0_F_PER_M * float(epsilon_r)
    electric = np.cumsum(rho_c_m3) * dx_m / eps
    phi = -np.cumsum(electric) * dx_m
    phi_zero_edge = phi - np.linspace(phi[0], phi[-1], len(phi))
    for index, row in enumerate(rows):
        row["charge_density_C_per_m3"] = float(rho_c_m3[index])
        row["electric_field_proxy_V_per_m"] = float(electric[index])
        row["poisson_potential_proxy_V_zero_edge"] = float(phi_zero_edge[index])
        row["poisson_potential_proxy_mV_zero_edge"] = float(phi_zero_edge[index] * 1000.0)
    return rows


def frame_electrostatics(
    ion_rows: Sequence[Mapping[str, object]],
    frame_rows: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    config: Optional[BridgeElectrostaticsConfig] = None,
) -> List[Dict[str, object]]:
    """Summarize frame-level bridge core/surface-shell charge metrics."""

    cfg = config or BridgeElectrostaticsConfig()
    frames = _base_frame_rows(frame_rows, ion_rows, cfg)
    frame_lookup = {_frame_key(row.get(cfg.frame_column)): row for row in frames}
    species_order = _species_order(ion_rows, cfg)
    out = []
    for frame in frames:
        row = dict(frame)
        for key in [
            "core_net_charge_e",
            "core_abs_charge_e",
            "core_positive_charge_e",
            "core_ion_count",
            "surface_shell_abs_charge_e",
            "trace_cylinder_abs_charge_e",
            "trace_cylinder_net_charge_e",
        ]:
            row[key] = 0.0
        for species in species_order:
            row[f"core_{species}_count"] = 0
            row[f"core_{species}_charge_e"] = 0.0
        out.append(row)
    out_by_frame = {_frame_key(row.get(cfg.frame_column)): row for row in out}
    for ion in ion_rows:
        q = _charge(ion, cfg)
        if q == 0.0:
            continue
        s = _as_float(ion.get(cfg.s_column, ion.get("bridge_s_A")))
        rho = _as_float(ion.get(cfg.rho_column, ion.get("bridge_rho_A")))
        if not (math.isfinite(s) and math.isfinite(rho)):
            continue
        key = _frame_key(ion.get(cfg.frame_column))
        row = out_by_frame.get(key)
        if row is None:
            continue
        frame = frame_lookup.get(key, {})
        gap = _gap_value(ion, frame, cfg)
        in_core = abs(s) <= cfg.core_s_half_width_A and rho <= cfg.core_rho_max_A
        in_surface_shell = (
            math.isfinite(gap)
            and rho <= cfg.profile_rho_max_A
            and abs(abs(s) - max(gap, 0.0) / 2.0) <= cfg.surface_shell_width_A
        )
        in_trace_cylinder = rho <= cfg.profile_rho_max_A and abs(s) <= max(abs(cfg.profile_s_min_A), abs(cfg.profile_s_max_A))
        species = canonical_species(ion.get(cfg.species_column, ion.get("current_trace_species")))
        if in_core:
            row["core_net_charge_e"] += q
            row["core_abs_charge_e"] += abs(q)
            row["core_positive_charge_e"] += q if q > 0 else 0.0
            row["core_ion_count"] += 1
            if species in species_order:
                row[f"core_{species}_count"] += 1
                row[f"core_{species}_charge_e"] += q
        if in_surface_shell:
            row["surface_shell_abs_charge_e"] += abs(q)
        if in_trace_cylinder:
            row["trace_cylinder_abs_charge_e"] += abs(q)
            row["trace_cylinder_net_charge_e"] += q
    for row in out:
        volume = _as_float(row.get("bridge_core_volume_A3"))
        if not math.isfinite(volume) or volume <= 0:
            volume = math.pi * cfg.core_rho_max_A * cfg.core_rho_max_A * (2.0 * cfg.core_s_half_width_A)
            row["bridge_core_volume_A3"] = volume
        row["core_charge_density_e_per_A3"] = row["core_net_charge_e"] / volume if volume > 0 else math.nan
        denom = row["core_abs_charge_e"] + row["surface_shell_abs_charge_e"]
        row["edl_overlap_charge_fraction"] = row["core_abs_charge_e"] / denom if denom > 0 else math.nan
        row["core_charge_neutrality_index"] = (
            abs(row["core_net_charge_e"]) / row["core_abs_charge_e"] if row["core_abs_charge_e"] > 0 else math.nan
        )
        row["trace_to_core_abs_charge_ratio"] = (
            row["trace_cylinder_abs_charge_e"] / row["core_abs_charge_e"] if row["core_abs_charge_e"] > 0 else math.nan
        )
    return out


def electrostatic_coupling_metrics(
    ion_rows: Sequence[Mapping[str, object]],
    frame_rows: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    config: Optional[BridgeElectrostaticsConfig] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Compute region-based charge-separation and species-budget tables."""

    cfg = config or BridgeElectrostaticsConfig()
    frames = _base_frame_rows(frame_rows, ion_rows, cfg)
    ions_by_frame = _group_rows(ion_rows, cfg.frame_column)
    frame_metrics: List[Dict[str, object]] = []
    species_budget: List[Dict[str, object]] = []
    for frame in frames:
        frame_key = _frame_key(frame.get(cfg.frame_column))
        row = dict(frame)
        region_values = {
            "core": [],
            "left_shell": [],
            "right_shell": [],
            "bridge_cylinder": [],
        }
        for ion in ions_by_frame.get(frame_key, []):
            q = _charge(ion, cfg)
            if q == 0.0:
                continue
            s = _as_float(ion.get(cfg.s_column, ion.get("bridge_s_A")))
            rho = _as_float(ion.get(cfg.rho_column, ion.get("bridge_rho_A")))
            if not (math.isfinite(s) and math.isfinite(rho)):
                continue
            species = canonical_species(ion.get(cfg.species_column, ion.get("current_trace_species")))
            memberships = {
                "core": rho <= cfg.core_rho_max_A and abs(s) <= cfg.core_s_half_width_A,
                "left_shell": rho <= cfg.shell_rho_max_A and -cfg.shell_s_outer_A <= s < -cfg.shell_s_inner_A,
                "right_shell": rho <= cfg.shell_rho_max_A and cfg.shell_s_inner_A < s <= cfg.shell_s_outer_A,
                "bridge_cylinder": rho <= cfg.shell_rho_max_A and abs(s) <= cfg.shell_s_outer_A,
            }
            for region, keep in memberships.items():
                if keep:
                    region_values[region].append((q, s, species))
        for region, values in region_values.items():
            _add_region_metrics(row, region, values)
            by_species: Dict[str, List[float]] = {}
            for q, _s, species in values:
                by_species.setdefault(species, []).append(q)
            for species, charges in sorted(by_species.items()):
                species_budget.append(
                    {
                        cfg.frame_column: row.get(cfg.frame_column),
                        "time_ns": row.get(cfg.time_column, math.nan),
                        "surface_gap_A": _gap_value({}, row, cfg),
                        "region": region,
                        "species": species,
                        "count": len(charges),
                        "net_charge_e": float(np.sum(charges)),
                        "abs_charge_e": float(np.sum(np.abs(charges))),
                    }
                )
        _add_coupling_derived_metrics(row)
        frame_metrics.append(row)
    return frame_metrics, species_budget


def gap_bin_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[BridgeElectrostaticsConfig] = None,
    gap_column: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Summarize numeric frame metrics by fixed-width surface-gap bins."""

    cfg = config or BridgeElectrostaticsConfig()
    column = gap_column or cfg.gap_column
    finite_rows = [row for row in rows if math.isfinite(_as_float(row.get(column)))]
    if not finite_rows:
        return []
    values = [_as_float(row.get(column)) for row in finite_rows]
    left0 = math.floor(min(values) / cfg.gap_bin_width_A) * cfg.gap_bin_width_A
    rightN = math.ceil(max(values) / cfg.gap_bin_width_A) * cfg.gap_bin_width_A
    bins = _bins(left0, rightN + cfg.gap_bin_width_A * 0.5, cfg.gap_bin_width_A)
    numeric_columns = _numeric_columns(finite_rows)
    out: List[Dict[str, object]] = []
    for index, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        chunk = [
            row
            for row in finite_rows
            if _as_float(row.get(column)) >= left
            and (_as_float(row.get(column)) < right or index == len(bins) - 2)
        ]
        if not chunk:
            continue
        result: Dict[str, object] = {
            "gap_left_A": left,
            "gap_right_A": right,
            "gap_center_A": float(np.mean([_as_float(row.get(column)) for row in chunk])),
            "n_frames": len({_frame_key(row.get(cfg.frame_column)) for row in chunk}),
        }
        for name in numeric_columns:
            vals = [_as_float(row.get(name)) for row in chunk]
            stats = _summary_stats(vals)
            for suffix, value in stats.items():
                result[f"{name}_{suffix}"] = value
        out.append(result)
    return out


def apparent_disjoining_pressure(
    gap_summary_rows: Sequence[Mapping[str, object]],
    *,
    area_A2: Optional[float],
    energy_column: str = "fes_free_energy_relative_raw_interp_mean",
) -> List[Dict[str, object]]:
    """Compute an apparent disjoining-pressure proxy from binned FES slopes."""

    if area_A2 is None or area_A2 <= 0:
        return []
    rows = [
        dict(row)
        for row in gap_summary_rows
        if math.isfinite(_as_float(row.get("gap_center_A"))) and math.isfinite(_as_float(row.get(energy_column)))
    ]
    rows.sort(key=lambda row: _as_float(row.get("gap_center_A")))
    if len(rows) < 3:
        return []
    h = np.asarray([_as_float(row.get("gap_center_A")) for row in rows], dtype=float)
    f = np.asarray([_as_float(row.get(energy_column)) for row in rows], dtype=float)
    dfdh = np.gradient(f, h)
    out: List[Dict[str, object]] = []
    for index, row in enumerate(rows):
        new = dict(row)
        new["dFdh_raw_per_A"] = float(dfdh[index])
        new["overlap_area_A2"] = float(area_A2)
        new["Pi_raw_per_A3"] = float(-dfdh[index] / area_A2)
        new["Pi_GPa_if_F_is_kJ_per_mol"] = float(new["Pi_raw_per_A3"] * KJMOL_PER_A3_TO_GPA)
        new["pressure_qc_note"] = "apparent proxy from frame-aligned FES; units assume F is kJ/mol for GPa column"
        out.append(new)
    return out


def species_summary(species_budget_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize region/species budgets over all frames."""

    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in species_budget_rows:
        grouped.setdefault((str(row.get("region", "")), str(row.get("species", ""))), []).append(row)
    out: List[Dict[str, object]] = []
    for (region, species), chunk in sorted(grouped.items()):
        counts = [_as_float(row.get("count"), 0.0) for row in chunk]
        net = [_as_float(row.get("net_charge_e"), 0.0) for row in chunk]
        absq = [_as_float(row.get("abs_charge_e"), 0.0) for row in chunk]
        out.append(
            {
                "region": region,
                "species": species,
                "frame_rows": len(chunk),
                "total_count": float(np.sum(counts)),
                "mean_count_per_present_frame": float(np.mean(counts)) if counts else math.nan,
                "total_net_charge_e": float(np.sum(net)),
                "mean_net_charge_e": float(np.mean(net)) if net else math.nan,
                "total_abs_charge_e": float(np.sum(absq)),
            }
        )
    return out


def descriptor_summary(
    frame_metric_rows: Sequence[Mapping[str, object]],
    profile_rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[BridgeElectrostaticsConfig] = None,
) -> List[Dict[str, object]]:
    """Build a compact descriptor/correlation summary from frame and profile tables."""

    cfg = config or BridgeElectrostaticsConfig()
    rows: List[Dict[str, object]] = []
    descriptors = [
        "core_abs_charge_e",
        "shell_abs_charge_e",
        "two_sided_edl_overlap_fraction",
        "abs_bridge_cylinder_axial_dipole_eA",
        "normalized_bridge_dipole_A",
        "left_right_edl_asymmetry_index",
        "core_charge_separation_A",
    ]
    for name in descriptors:
        vals = [_as_float(row.get(name)) for row in frame_metric_rows]
        stats = _summary_stats(vals)
        rows.append({"descriptor": f"mean_{name}", "mean": stats["mean"], "std": stats["std"], "n_frames": stats["count"]})
    if profile_rows:
        xs = np.asarray([_as_float(row.get("s_center_A")) for row in profile_rows], dtype=float)
        ys = np.asarray([abs(_as_float(row.get("charge_density_e_per_A3"), 0.0)) for row in profile_rows], dtype=float)
        finite = np.isfinite(xs) & np.isfinite(ys)
        rows.append(
            {
                "descriptor": "axial_profile_integrated_abs_charge_density_e_A2",
                "mean": float(np.trapz(ys[finite], xs[finite])) if np.count_nonzero(finite) >= 2 else math.nan,
                "std": math.nan,
                "n_frames": _count_frames(frame_metric_rows, (), cfg.frame_column),
            }
        )
    for source in [
        "core_abs_charge_e",
        "two_sided_edl_overlap_fraction",
        "abs_bridge_cylinder_axial_dipole_eA",
        "left_right_edl_asymmetry_index",
    ]:
        for target in [cfg.gap_column, "d3d_all", "Nw_bridge_core", "fes_free_energy_relative_raw_interp"]:
            corr, count = _pearson_for_columns(frame_metric_rows, source, target)
            if count:
                rows.append(
                    {
                        "descriptor": f"pearson_corr_{source}_vs_{target}",
                        "mean": corr,
                        "std": math.nan,
                        "n_frames": count,
                    }
                )
    return rows


def analyze_bridge_electrostatics(
    *,
    ion_table: Path,
    output_dir: Path,
    frame_table: Optional[Path] = None,
    config: Optional[BridgeElectrostaticsConfig] = None,
    derive_coordinates: bool = False,
    force_derive_coordinates: bool = False,
    gap_mode: str = "asis",
    dynamic_gap_column: str = "dynamic_surface_gap_est_A",
    d3d_column: str = "d3d_all",
    nominal_radius_a_A: float = 19.0,
    nominal_radius_b_A: float = 19.0,
    strict_gap_precontact_only: bool = False,
    strict_gap_max_A: Optional[float] = None,
    strict_gap_max_tolerance_A: float = 0.5,
    start_time_ns: Optional[float] = None,
    end_time_ns: Optional[float] = None,
    max_frames: Optional[int] = None,
    disjoining_area_A2: Optional[float] = None,
    run_coupling: bool = True,
) -> Dict[str, Path]:
    """Run the bridge electrostatic proxy workflow and write output tables."""

    cfg = config or BridgeElectrostaticsConfig()
    cfg.validate()
    ions, _ = read_csv_rows(Path(ion_table))
    frames: List[Dict[str, object]]
    if frame_table is not None:
        frames_raw, _ = read_csv_rows(Path(frame_table))
        frames = select_frame_rows(
            frames_raw,
            frame_column=cfg.frame_column,
            time_column=cfg.time_column,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            max_frames=max_frames,
        )
    else:
        frames = _base_frame_rows(None, ions, cfg)
    frames = prepare_gap_columns(
        frames,
        mode=gap_mode,
        dynamic_gap_column=dynamic_gap_column,
        d3d_column=d3d_column,
        nominal_radius_a_A=nominal_radius_a_A,
        nominal_radius_b_A=nominal_radius_b_A,
        output_gap_column=cfg.gap_column,
        strict_precontact_only=strict_gap_precontact_only,
        strict_gap_max_A=strict_gap_max_A,
        strict_gap_max_tolerance_A=strict_gap_max_tolerance_A,
    )
    keep_frames = {_frame_key(row.get(cfg.frame_column)) for row in frames}
    ions = [dict(row) for row in ions if _frame_key(row.get(cfg.frame_column)) in keep_frames]
    if derive_coordinates or force_derive_coordinates:
        ions = add_bridge_coordinates(ions, frames, config=cfg, force=force_derive_coordinates)
    profile = charge_profile(ions, frames, config=cfg)
    potential = poisson_proxy(profile, epsilon_r=cfg.epsilon_r)
    frame_edl = frame_electrostatics(ions, frames, config=cfg)
    gap_summary_rows = gap_bin_summary(frame_edl, config=cfg)
    pi_rows = apparent_disjoining_pressure(gap_summary_rows, area_A2=disjoining_area_A2)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {
        "charge_profile": output / "bridge_charge_profile.csv",
        "poisson_proxy": output / "bridge_poisson_proxy.csv",
        "frame_electrostatics": output / "bridge_electrostatics_frame_table.csv",
        "gap_summary": output / "bridge_electrostatics_gap_summary.csv",
        "apparent_disjoining_pressure": output / "bridge_apparent_disjoining_pressure.csv",
    }
    write_csv_rows(outputs["charge_profile"], profile)
    write_csv_rows(outputs["poisson_proxy"], potential)
    write_csv_rows(outputs["frame_electrostatics"], frame_edl)
    write_csv_rows(outputs["gap_summary"], gap_summary_rows)
    write_csv_rows(outputs["apparent_disjoining_pressure"], pi_rows)
    desc_rows: List[Dict[str, object]] = []
    if run_coupling:
        coupling_rows, species_budget_rows = electrostatic_coupling_metrics(ions, frames, config=cfg)
        species_summary_rows = species_summary(species_budget_rows)
        coupling_gap_rows = gap_bin_summary(coupling_rows, config=cfg)
        desc_rows = descriptor_summary(coupling_rows, profile, config=cfg)
        outputs.update(
            {
                "coupling_frame_table": output / "bridge_charge_coupling_frame_table.csv",
                "region_species_budget": output / "bridge_charge_region_species_budget.csv",
                "region_species_summary": output / "bridge_charge_region_species_summary.csv",
                "coupling_gap_summary": output / "bridge_charge_coupling_gap_summary.csv",
                "descriptor_summary": output / "bridge_electrostatics_descriptor_summary.csv",
            }
        )
        write_csv_rows(outputs["coupling_frame_table"], coupling_rows)
        write_csv_rows(outputs["region_species_budget"], species_budget_rows)
        write_csv_rows(outputs["region_species_summary"], species_summary_rows)
        write_csv_rows(outputs["coupling_gap_summary"], coupling_gap_rows)
        write_csv_rows(outputs["descriptor_summary"], desc_rows)
    manifest = _manifest_rows(
        outputs,
        ion_table=Path(ion_table),
        frame_table=Path(frame_table) if frame_table is not None else None,
        config=cfg,
        n_ion_rows=len(ions),
        n_frame_rows=len(frames),
        gap_mode=gap_mode,
    )
    outputs["manifest"] = output / "bridge_electrostatics_manifest.csv"
    write_csv_rows(outputs["manifest"], manifest)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge-centered electrostatic proxy post-processing")
    parser.add_argument("--ion-table", type=Path, required=True, help="Long CSV with ion positions/species/charge")
    parser.add_argument("--frame-table", type=Path, help="Optional frame-level bridge geometry/FES table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--frame-column", default="global_frame")
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--species-column", default="species_canonical")
    parser.add_argument("--charge-column", default="species_charge_e")
    parser.add_argument("--s-column", default="bridge_axis_s_A")
    parser.add_argument("--rho-column", default="bridge_axis_rho_A")
    parser.add_argument("--gap-column", default="analysis_surface_gap_A")
    parser.add_argument("--species-order", action="append", help="Species order entry; may be repeated or comma separated")
    parser.add_argument("--profile-s-min-A", type=float, default=-35.0)
    parser.add_argument("--profile-s-max-A", type=float, default=35.0)
    parser.add_argument("--profile-bin-width-A", type=float, default=1.0)
    parser.add_argument("--profile-rho-max-A", type=float, default=10.0)
    parser.add_argument("--core-s-half-width-A", type=float, default=8.0)
    parser.add_argument("--core-rho-max-A", type=float, default=6.5)
    parser.add_argument("--surface-shell-width-A", type=float, default=4.0)
    parser.add_argument("--shell-s-inner-A", type=float, default=8.0)
    parser.add_argument("--shell-s-outer-A", type=float, default=22.0)
    parser.add_argument("--shell-rho-max-A", type=float, default=10.0)
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--epsilon-r", type=float, default=78.5)
    parser.add_argument("--box-lengths-A", type=float, nargs=3)
    parser.add_argument("--derive-bridge-coordinates", action="store_true")
    parser.add_argument("--force-derive-bridge-coordinates", action="store_true")
    parser.add_argument("--gap-mode", choices=["asis", "dynamic", "nominal"], default="asis")
    parser.add_argument("--dynamic-gap-column", default="dynamic_surface_gap_est_A")
    parser.add_argument("--d3d-column", default="d3d_all")
    parser.add_argument("--nominal-radius-a-A", type=float, default=19.0)
    parser.add_argument("--nominal-radius-b-A", type=float, default=19.0)
    parser.add_argument("--strict-gap-precontact-only", action="store_true")
    parser.add_argument("--strict-gap-max-A", type=float)
    parser.add_argument("--strict-gap-max-tolerance-A", type=float, default=0.5)
    parser.add_argument("--start-time-ns", type=float)
    parser.add_argument("--end-time-ns", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--disjoining-area-A2", type=float)
    parser.add_argument("--no-coupling", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BridgeElectrostaticsConfig(
        frame_column=args.frame_column,
        time_column=args.time_column,
        species_column=args.species_column,
        charge_column=args.charge_column,
        s_column=args.s_column,
        rho_column=args.rho_column,
        gap_column=args.gap_column,
        profile_s_min_A=args.profile_s_min_A,
        profile_s_max_A=args.profile_s_max_A,
        profile_bin_width_A=args.profile_bin_width_A,
        profile_rho_max_A=args.profile_rho_max_A,
        core_s_half_width_A=args.core_s_half_width_A,
        core_rho_max_A=args.core_rho_max_A,
        surface_shell_width_A=args.surface_shell_width_A,
        shell_s_inner_A=args.shell_s_inner_A,
        shell_s_outer_A=args.shell_s_outer_A,
        shell_rho_max_A=args.shell_rho_max_A,
        gap_bin_width_A=args.gap_bin_width_A,
        epsilon_r=args.epsilon_r,
        species_order=_split_species(args.species_order) or DEFAULT_SPECIES_ORDER,
        box_lengths_A=tuple(args.box_lengths_A) if args.box_lengths_A else None,
        case_label=args.case_label,
    )
    outputs = analyze_bridge_electrostatics(
        ion_table=args.ion_table,
        frame_table=args.frame_table,
        output_dir=args.output_dir,
        config=config,
        derive_coordinates=args.derive_bridge_coordinates or args.force_derive_bridge_coordinates,
        force_derive_coordinates=args.force_derive_bridge_coordinates,
        gap_mode=args.gap_mode,
        dynamic_gap_column=args.dynamic_gap_column,
        d3d_column=args.d3d_column,
        nominal_radius_a_A=args.nominal_radius_a_A,
        nominal_radius_b_A=args.nominal_radius_b_A,
        strict_gap_precontact_only=args.strict_gap_precontact_only,
        strict_gap_max_A=args.strict_gap_max_A,
        strict_gap_max_tolerance_A=args.strict_gap_max_tolerance_A,
        start_time_ns=args.start_time_ns,
        end_time_ns=args.end_time_ns,
        max_frames=args.max_frames,
        disjoining_area_A2=args.disjoining_area_A2,
        run_coupling=not args.no_coupling,
    )
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")
    return 0


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _format_csv_value(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, np.floating):
        out = float(value)
        return "" if math.isnan(out) else f"{out:.12g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def _frame_key(value: object) -> str:
    number = _as_float(value)
    if math.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return str(value)


def _frame_lookup(rows: Sequence[Mapping[str, object]], frame_column: str) -> Dict[str, Mapping[str, object]]:
    return {_frame_key(row.get(frame_column)): row for row in rows}


def _group_rows(rows: Sequence[Mapping[str, object]], frame_column: str) -> Dict[str, List[Mapping[str, object]]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_frame_key(row.get(frame_column)), []).append(row)
    return grouped


def _count_frames(
    frame_rows: Optional[Sequence[Mapping[str, object]]],
    ion_rows: Sequence[Mapping[str, object]],
    frame_column: str,
) -> int:
    rows = frame_rows if frame_rows is not None else ion_rows
    return len({_frame_key(row.get(frame_column)) for row in rows})


def _species_order(rows: Sequence[Mapping[str, object]], config: BridgeElectrostaticsConfig) -> Tuple[str, ...]:
    discovered = {
        canonical_species(row.get(config.species_column, row.get("current_trace_species")))
        for row in rows
        if canonical_species(row.get(config.species_column, row.get("current_trace_species"))) not in {"unknown", ""}
    }
    ordered = [species for species in config.species_order if species in discovered or species in SPECIES_CHARGE]
    ordered.extend(sorted(species for species in discovered if species not in ordered))
    return tuple(ordered)


def _charge(row: Mapping[str, object], config: BridgeElectrostaticsConfig) -> float:
    q = _as_float(row.get(config.charge_column))
    if math.isfinite(q):
        return q
    return charge_for_species(row.get(config.species_column, row.get("current_trace_species")))


def _bins(start: float, stop: float, width: float) -> List[float]:
    count = int(math.ceil((stop - start) / width))
    return [float(start + index * width) for index in range(count + 1)]


def _gap_value(
    ion_row: Mapping[str, object],
    frame_row: Mapping[str, object],
    config: BridgeElectrostaticsConfig,
) -> float:
    for column in (config.gap_column, *config.fallback_gap_columns):
        value = _as_float(ion_row.get(column))
        if math.isfinite(value):
            return value
        value = _as_float(frame_row.get(column))
        if math.isfinite(value):
            return value
    return math.nan


def _base_frame_rows(
    frame_rows: Optional[Sequence[Mapping[str, object]]],
    ion_rows: Sequence[Mapping[str, object]],
    config: BridgeElectrostaticsConfig,
) -> List[Dict[str, object]]:
    if frame_rows is not None:
        seen = set()
        out = []
        for row in frame_rows:
            key = _frame_key(row.get(config.frame_column))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(row))
        return sorted(out, key=lambda row: (_as_float(row.get(config.time_column)), _as_float(row.get(config.frame_column))))
    grouped = _group_rows(ion_rows, config.frame_column)
    out = []
    for key, chunk in grouped.items():
        first = chunk[0]
        row = {
            config.frame_column: first.get(config.frame_column),
            config.time_column: first.get(config.time_column, math.nan),
        }
        for column in [config.gap_column, *config.fallback_gap_columns, "d3d_all", "bridge_core_volume_A3", "Nw_bridge_core"]:
            if column in first:
                row[column] = first.get(column)
        out.append(row)
    return sorted(out, key=lambda row: (_as_float(row.get(config.time_column)), _as_float(row.get(config.frame_column))))


def _vector_from_row(row: Mapping[str, object], columns: Sequence[str]) -> Tuple[float, float, float]:
    return tuple(_as_float(row.get(column)) for column in columns)  # type: ignore[return-value]


def _vector_from_row_with_fallback(
    row: Mapping[str, object],
    fallback: Mapping[str, object],
    columns: Sequence[str],
) -> Tuple[float, float, float]:
    return tuple(_as_float(row.get(column, fallback.get(column))) for column in columns)  # type: ignore[return-value]


def _minimum_image(delta: np.ndarray, box_lengths: Optional[Tuple[float, float, float]]) -> np.ndarray:
    out = np.asarray(delta, dtype=float).copy()
    if box_lengths is None:
        return out
    box = np.asarray(box_lengths, dtype=float)
    valid = np.isfinite(box) & (box > 0)
    out[valid] -= box[valid] * np.round(out[valid] / box[valid])
    return out


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _numeric_columns(rows: Sequence[Mapping[str, object]]) -> List[str]:
    columns = sorted({key for row in rows for key in row.keys()})
    out = []
    for column in columns:
        if any(math.isfinite(_as_float(row.get(column))) for row in rows):
            out.append(column)
    return out


def _add_region_metrics(row: Dict[str, object], region: str, values: Sequence[Tuple[float, float, str]]) -> None:
    charges = np.asarray([item[0] for item in values], dtype=float)
    s_values = np.asarray([item[1] for item in values], dtype=float)
    positive = charges > 0
    negative = charges < 0
    absq = np.abs(charges)
    row[f"{region}_ion_count"] = int(charges.size)
    row[f"{region}_net_charge_e"] = float(np.sum(charges)) if charges.size else 0.0
    row[f"{region}_abs_charge_e"] = float(np.sum(absq)) if charges.size else 0.0
    row[f"{region}_positive_charge_e"] = float(np.sum(charges[positive])) if np.any(positive) else 0.0
    row[f"{region}_negative_abs_charge_e"] = float(np.sum(-charges[negative])) if np.any(negative) else 0.0
    row[f"{region}_axial_dipole_eA"] = float(np.sum(charges * s_values)) if charges.size else 0.0
    row[f"{region}_abs_charge_centroid_s_A"] = _weighted_mean(s_values, absq)
    row[f"{region}_positive_centroid_s_A"] = _weighted_mean(s_values[positive], charges[positive]) if np.any(positive) else math.nan
    row[f"{region}_negative_centroid_s_A"] = _weighted_mean(s_values[negative], -charges[negative]) if np.any(negative) else math.nan
    if math.isfinite(_as_float(row[f"{region}_positive_centroid_s_A"])) and math.isfinite(
        _as_float(row[f"{region}_negative_centroid_s_A"])
    ):
        row[f"{region}_charge_separation_A"] = (
            _as_float(row[f"{region}_positive_centroid_s_A"]) - _as_float(row[f"{region}_negative_centroid_s_A"])
        )
    else:
        row[f"{region}_charge_separation_A"] = math.nan
    row[f"{region}_neutrality_index"] = (
        abs(_as_float(row[f"{region}_net_charge_e"])) / _as_float(row[f"{region}_abs_charge_e"])
        if _as_float(row[f"{region}_abs_charge_e"]) > 0
        else math.nan
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return math.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _add_coupling_derived_metrics(row: Dict[str, object]) -> None:
    left_abs = _as_float(row.get("left_shell_abs_charge_e"), 0.0)
    right_abs = _as_float(row.get("right_shell_abs_charge_e"), 0.0)
    core_abs = _as_float(row.get("core_abs_charge_e"), 0.0)
    shell_abs = left_abs + right_abs
    row["shell_abs_charge_e"] = shell_abs
    row["left_right_edl_asymmetry_index"] = (right_abs - left_abs) / shell_abs if shell_abs > 0 else math.nan
    row["two_sided_edl_overlap_fraction"] = core_abs / (core_abs + shell_abs) if core_abs + shell_abs > 0 else math.nan
    row["core_abs_to_shell_abs_ratio"] = core_abs / shell_abs if shell_abs > 0 else math.nan
    row["abs_core_axial_dipole_eA"] = abs(_as_float(row.get("core_axial_dipole_eA"), 0.0))
    row["abs_bridge_cylinder_axial_dipole_eA"] = abs(_as_float(row.get("bridge_cylinder_axial_dipole_eA"), 0.0))
    bridge_abs = _as_float(row.get("bridge_cylinder_abs_charge_e"), 0.0)
    row["normalized_bridge_dipole_A"] = (
        row["abs_bridge_cylinder_axial_dipole_eA"] / bridge_abs if bridge_abs > 0 else math.nan
    )


def _pearson_for_columns(rows: Sequence[Mapping[str, object]], x_column: str, y_column: str) -> Tuple[float, int]:
    xs = []
    ys = []
    for row in rows:
        x = _as_float(row.get(x_column))
        y = _as_float(row.get(y_column))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        return math.nan, len(xs)
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if float(np.std(x_arr)) <= 0 or float(np.std(y_arr)) <= 0:
        return math.nan, len(xs)
    return float(np.corrcoef(x_arr, y_arr)[0, 1]), len(xs)


def _split_species(raw: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not raw:
        return ()
    out: List[str] = []
    for value in raw:
        for item in str(value).replace(";", ",").replace("|", ",").split(","):
            species = canonical_species(item.strip())
            if species and species != "unknown" and species not in out:
                out.append(species)
    return tuple(out)


def _manifest_rows(
    outputs: Mapping[str, Path],
    *,
    ion_table: Path,
    frame_table: Optional[Path],
    config: BridgeElectrostaticsConfig,
    n_ion_rows: int,
    n_frame_rows: int,
    gap_mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = [
        {"record_type": "input", "name": "ion_table", "path": str(ion_table), "exists": ion_table.exists()},
        {
            "record_type": "input",
            "name": "frame_table",
            "path": str(frame_table) if frame_table is not None else "",
            "exists": frame_table.exists() if frame_table is not None else False,
        },
    ]
    for name, path in sorted(outputs.items()):
        rows.append({"record_type": "output", "name": name, "path": str(path), "exists": path.exists()})
    rows.append(
        {
            "record_type": "config",
            "name": "bridge_electrostatics",
            "case_label": config.case_label,
            "n_ion_rows": n_ion_rows,
            "n_frame_rows": n_frame_rows,
            "gap_mode": gap_mode,
            "profile_s_min_A": config.profile_s_min_A,
            "profile_s_max_A": config.profile_s_max_A,
            "profile_bin_width_A": config.profile_bin_width_A,
            "profile_rho_max_A": config.profile_rho_max_A,
            "core_s_half_width_A": config.core_s_half_width_A,
            "core_rho_max_A": config.core_rho_max_A,
            "surface_shell_width_A": config.surface_shell_width_A,
            "shell_s_inner_A": config.shell_s_inner_A,
            "shell_s_outer_A": config.shell_s_outer_A,
            "shell_rho_max_A": config.shell_rho_max_A,
            "gap_bin_width_A": config.gap_bin_width_A,
            "epsilon_r": config.epsilon_r,
        }
    )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
