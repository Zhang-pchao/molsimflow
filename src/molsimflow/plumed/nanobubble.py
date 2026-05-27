"""Generate PLUMED inputs for N2 nanobubble collective variables.

The module keeps the legacy N2 dimer-centre PLUMED template but exposes the
atom selection as a reusable API.  It can generate the original cluster-size CV
or a surface-distance CV using the centre of mass of a selected top surface
layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from molsimflow.io.lammps_data import read_extxyz_atoms


MASS_ELEMENT_TABLE: Tuple[Tuple[str, float], ...] = (
    ("H", 1.008),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("Na", 22.990),
    ("Si", 28.085),
    ("Cl", 35.450),
    ("Ti", 47.867),
)


@dataclass(frozen=True)
class AtomRecord:
    """One atom with a PLUMED/LAMMPS-compatible 1-based atom id."""

    atom_id: int
    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class SurfaceSelection:
    """Top-layer surface atom selection details."""

    atom_ids: Tuple[int, ...]
    candidate_count: int
    max_z: float
    z_tolerance: float
    stride: int


@dataclass(frozen=True)
class PlumedBiasConfig:
    """Configurable constants for the legacy N2 COM PLUMED template."""

    contact_d0: float = 5.0
    contact_dmax: float = 6.0
    temperature: float = 330.0
    pace: int = 500
    explore_barrier: float = 100.0
    production_barrier: float = 200.0
    state_wstride: int = 5000
    print_stride: int = 1
    secondary_print_stride: int = 10
    flush_stride: int = 100
    restart: bool = False
    hills_explore_file: str = "HILLS_e"
    hills_file: str = "HILLS"
    state_explore_file: str = "STATE_e"
    state_file: str = "STATE"
    colvar_file: str = "COLVAR"
    secondary_colvar_file: str = "COLVAR_step10"


@dataclass(frozen=True)
class N2ComPlumedSummary:
    """Summary returned after writing a PLUMED file."""

    output_file: Path
    dimer_pairs: Tuple[Tuple[int, int], ...]
    surface_selection: Optional[SurfaceSelection]
    mode: str

    @property
    def dimer_count(self) -> int:
        return len(self.dimer_pairs)


def normalize_symbol(symbol: str) -> str:
    """Normalize an element symbol without changing pseudo-type labels."""
    stripped = (symbol or "").strip()
    if not stripped:
        return stripped
    if not stripped[0].isalpha():
        return stripped
    if len(stripped) == 1:
        return stripped.upper()
    return stripped[0].upper() + stripped[1:].lower()


def parse_type_map(entries: Optional[Sequence[str]]) -> Dict[int, str]:
    """Parse CLI type-map entries such as ``8=Si`` or ``3:N``."""
    result: Dict[int, str] = {}
    if not entries:
        return result
    for entry in entries:
        raw = entry.strip()
        if not raw:
            continue
        if "=" in raw:
            lhs, rhs = raw.split("=", 1)
        elif ":" in raw:
            lhs, rhs = raw.split(":", 1)
        else:
            raise ValueError(f"Invalid type-map entry {entry!r}; use TYPE=ELEMENT")
        try:
            atom_type = int(lhs)
        except ValueError as exc:
            raise ValueError(f"Invalid LAMMPS atom type in {entry!r}") from exc
        element = normalize_symbol(rhs)
        if not element:
            raise ValueError(f"Missing element in type-map entry {entry!r}")
        result[atom_type] = element
    return result


def _infer_element_from_mass(mass: float, *, tolerance: float = 0.25) -> Optional[str]:
    best_symbol = None
    best_delta = None
    for symbol, reference in MASS_ELEMENT_TABLE:
        delta = abs(float(mass) - reference)
        if best_delta is None or delta < best_delta:
            best_symbol = symbol
            best_delta = delta
    if best_delta is not None and best_delta <= tolerance:
        return best_symbol
    return None


def _section_lines(lines: Sequence[str], section: str) -> List[str]:
    target = section.lower()
    in_section = False
    body: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        header = stripped.split("#", 1)[0].strip().lower()
        first_token = header.split()[0] if header else ""
        if first_token == target:
            in_section = True
            continue
        if in_section and stripped[0].isalpha():
            break
        if in_section:
            body.append(raw)
    return body


def _type_map_from_lammps_masses(lines: Sequence[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for raw in _section_lines(lines, "Masses"):
        data, _, comment = raw.partition("#")
        parts = data.split()
        if len(parts) < 2:
            continue
        try:
            atom_type = int(parts[0])
            mass = float(parts[1])
        except ValueError:
            continue
        comment_tokens = re.findall(r"[A-Za-z][a-z]?", comment)
        symbol = normalize_symbol(comment_tokens[0]) if comment_tokens else ""
        if not symbol:
            symbol = _infer_element_from_mass(mass) or ""
        if symbol:
            mapping[atom_type] = symbol
    return mapping


def read_lammps_atomic_data_atoms(
    path: Union[str, Path],
    *,
    type_map: Optional[Dict[int, str]] = None,
    atom_style: str = "atomic",
) -> List[AtomRecord]:
    """Read atom ids, elements, and coordinates from a LAMMPS data file."""
    data_path = Path(path)
    lines = data_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    mapping = _type_map_from_lammps_masses(lines)
    if type_map:
        mapping.update({key: normalize_symbol(value) for key, value in type_map.items()})

    if atom_style not in {"atomic", "charge", "full", "molecular"}:
        raise ValueError(f"Unsupported LAMMPS atom style: {atom_style}")
    type_index = 2 if atom_style in {"full", "molecular"} else 1

    atoms: List[AtomRecord] = []
    for raw in _section_lines(lines, "Atoms"):
        parts = raw.split("#", 1)[0].split()
        if not parts:
            continue
        if len(parts) <= max(type_index, 4):
            raise ValueError(f"Invalid atom line in {data_path}: {raw!r}")
        try:
            atom_id = int(parts[0])
            atom_type = int(parts[type_index])
            x, y, z = (float(value) for value in parts[-3:])
        except ValueError as exc:
            raise ValueError(f"Invalid atom line in {data_path}: {raw!r}") from exc
        symbol = mapping.get(atom_type, f"type{atom_type}")
        atoms.append(AtomRecord(atom_id, normalize_symbol(symbol), x, y, z))
    if not atoms:
        raise ValueError(f"No atoms were read from LAMMPS data file: {data_path}")
    return sorted(atoms, key=lambda atom: atom.atom_id)


def read_structure_atoms(
    path: Union[str, Path],
    *,
    structure_format: str = "auto",
    type_map: Optional[Dict[int, str]] = None,
    lammps_atom_style: str = "atomic",
) -> List[AtomRecord]:
    """Read a structure file as 1-based atom records."""
    structure_path = Path(path)
    fmt = structure_format.lower()
    if fmt == "auto":
        suffixes = "".join(structure_path.suffixes).lower()
        if structure_path.suffix.lower() == ".xyz":
            fmt = "extxyz"
        elif suffixes.endswith(".data") or structure_path.suffix.lower() in {".lmp", ".lammps"}:
            fmt = "lammps-data"
        else:
            raise ValueError(f"Cannot infer structure format from {structure_path}")

    if fmt == "extxyz":
        atoms, _ = read_extxyz_atoms(structure_path)
        return [
            AtomRecord(atom_id, normalize_symbol(symbol), x, y, z)
            for atom_id, (symbol, x, y, z) in enumerate(atoms, start=1)
        ]
    if fmt == "lammps-data":
        return read_lammps_atomic_data_atoms(
            structure_path,
            type_map=type_map,
            atom_style=lammps_atom_style,
        )
    raise ValueError(f"Unsupported structure format: {structure_format}")


def pair_consecutive_atoms(atom_ids: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Pair sorted atom ids consecutively as dimers."""
    ids = sorted(int(atom_id) for atom_id in atom_ids)
    if len(ids) % 2:
        raise ValueError("The dimer atom selection must contain an even number of atoms")
    return tuple((ids[i], ids[i + 1]) for i in range(0, len(ids), 2))


def dimer_pairs_from_range(start: int, stop: int) -> Tuple[Tuple[int, int], ...]:
    """Return consecutive dimer pairs from a legacy inclusive atom-id range."""
    if start <= 0 or stop <= 0:
        raise ValueError("Atom ids must be positive 1-based indices")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    return pair_consecutive_atoms(list(range(start, stop + 1)))


def dimer_pairs_from_structure(
    atoms: Sequence[AtomRecord],
    *,
    element: str = "N",
) -> Tuple[Tuple[int, int], ...]:
    """Select all atoms of one element and pair them consecutively."""
    wanted = normalize_symbol(element)
    atom_ids = [atom.atom_id for atom in atoms if atom.symbol == wanted]
    if not atom_ids:
        raise ValueError(f"No atoms with element {wanted!r} were found")
    return pair_consecutive_atoms(atom_ids)


def select_top_layer_atoms(
    atoms: Sequence[AtomRecord],
    *,
    element: str,
    z_tolerance: float = 0.5,
    stride: int = 1,
) -> SurfaceSelection:
    """Select the maximum-z layer of one element and downsample by atom id."""
    if z_tolerance < 0:
        raise ValueError("z_tolerance must be non-negative")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    wanted = normalize_symbol(element)
    candidates = [atom for atom in atoms if atom.symbol == wanted]
    if not candidates:
        raise ValueError(f"No atoms with element {wanted!r} were found")
    max_z = max(atom.z for atom in candidates)
    top_atoms = sorted(
        [atom for atom in candidates if abs(atom.z - max_z) <= z_tolerance],
        key=lambda atom: atom.atom_id,
    )
    if not top_atoms:
        raise ValueError(f"No top-layer atoms found for element {wanted!r}")
    selected = top_atoms[::stride]
    return SurfaceSelection(
        atom_ids=tuple(atom.atom_id for atom in selected),
        candidate_count=len(top_atoms),
        max_z=max_z,
        z_tolerance=z_tolerance,
        stride=stride,
    )


def _format_float(value: float) -> str:
    return f"{value:g}"


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def _format_com_lines(
    dimer_pairs: Sequence[Tuple[int, int]],
    *,
    label_prefix: str,
) -> List[str]:
    return [
        f"{label_prefix}{idx:03d}: COM ATOMS={atom_a},{atom_b}"
        for idx, (atom_a, atom_b) in enumerate(dimer_pairs)
    ]


def _format_group_line(n_pairs: int, *, label_prefix: str, group_label: str) -> str:
    atoms = ",".join(f"{label_prefix}{idx:03d}" for idx in range(n_pairs))
    return f"{group_label}: GROUP ATOMS={atoms}"


def _opes_block(
    action: str,
    *,
    label: str,
    arg: str,
    output_file: str,
    barrier: float,
    state_file: str,
    config: PlumedBiasConfig,
) -> List[str]:
    return [
        f"{action} ...",
        f"  RESTART={_yes_no(config.restart)}",
        f"  LABEL={label}",
        f"  ARG={arg}",
        f"  FILE={output_file}",
        f"  TEMP={_format_float(config.temperature)}",
        f"  PACE={config.pace}",
        f"  BARRIER={_format_float(barrier)}",
        f"  STATE_WFILE={state_file}",
        f"  STATE_WSTRIDE={config.state_wstride}",
        "  STORE_STATES",
        "  WALKERS_MPI",
        f"... {action}",
    ]


def render_n2_com_plumed(
    dimer_pairs: Sequence[Tuple[int, int]],
    *,
    surface_atom_ids: Optional[Sequence[int]] = None,
    config: Optional[PlumedBiasConfig] = None,
    label_prefix: str = "c",
    group_label: str = "reps_center",
) -> str:
    """Render the complete PLUMED text."""
    if not dimer_pairs:
        raise ValueError("At least one dimer pair is required")
    cfg = config or PlumedBiasConfig()
    surface_ids = tuple(surface_atom_ids or ())

    lines: List[str] = [
        "# Auto-generated by molsimflow.plumed.nanobubble",
        "UNITS LENGTH=A",
    ]
    lines.extend(_format_com_lines(dimer_pairs, label_prefix=label_prefix))
    lines.append(_format_group_line(len(dimer_pairs), label_prefix=label_prefix, group_label=group_label))
    lines.extend(
        [
            "",
            (
                f"mat: CONTACT_MATRIX ATOMS={group_label} "
                "SWITCH={CUBIC "
                f"D_0={_format_float(cfg.contact_d0)} D_MAX={_format_float(cfg.contact_dmax)}}}"
            ),
            "dfs: DFSCLUSTERING MATRIX=mat LOWMEM",
            "",
            "n2_num:  CLUSTER_NATOMS CLUSTERS=dfs CLUSTER=1",
            "sum_cn:  CLUSTER_PROPERTIES CLUSTERS=dfs CLUSTER=1 SUM",
            "",
        ]
    )

    if surface_ids:
        lines.extend(
            [
                f"allN2: GROUP ATOMS={group_label}",
                "cb:    COM ATOMS=allN2",
                "",
                f"surf: GROUP ATOMS={','.join(str(atom_id) for atom_id in surface_ids)}",
                "csurf: COM ATOMS=surf",
                "",
                "dz: DISTANCE ATOMS=csurf,cb COMPONENTS",
                "",
            ]
        )
        bias_arg = "dz.z"
        print_args = "n2_num,sum_cn.*,dz.z,opes.bias,opes_e.bias"
    else:
        bias_arg = "sum_cn.sum"
        print_args = "n2_num,sum_cn.*,opes.bias,opes_e.bias"

    lines.extend(
        _opes_block(
            "OPES_METAD_EXPLORE",
            label="opes_e",
            arg=bias_arg,
            output_file=cfg.hills_explore_file,
            barrier=cfg.explore_barrier,
            state_file=cfg.state_explore_file,
            config=cfg,
        )
    )
    lines.append("")
    lines.extend(
        _opes_block(
            "OPES_METAD",
            label="opes",
            arg=bias_arg,
            output_file=cfg.hills_file,
            barrier=cfg.production_barrier,
            state_file=cfg.state_file,
            config=cfg,
        )
    )
    lines.extend(
        [
            "",
            f"PRINT STRIDE={cfg.print_stride}  FILE={cfg.colvar_file}        ARG={print_args}",
            (
                f"PRINT STRIDE={cfg.secondary_print_stride} FILE={cfg.secondary_colvar_file} "
                f"ARG={print_args}"
            ),
            f"FLUSH STRIDE={cfg.flush_stride}",
            "",
        ]
    )
    return "\n".join(lines)


def write_n2_com_plumed(
    output_file: Union[str, Path],
    dimer_pairs: Sequence[Tuple[int, int]],
    *,
    surface_atom_ids: Optional[Sequence[int]] = None,
    config: Optional[PlumedBiasConfig] = None,
    label_prefix: str = "c",
    group_label: str = "reps_center",
) -> Path:
    """Write an N2 COM PLUMED input file."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_n2_com_plumed(
            dimer_pairs,
            surface_atom_ids=surface_atom_ids,
            config=config,
            label_prefix=label_prefix,
            group_label=group_label,
        ),
        encoding="utf-8",
    )
    return output


def _resolve_dimer_pairs(
    *,
    atoms: Optional[Sequence[AtomRecord]],
    start: Optional[int],
    stop: Optional[int],
    dimer_element: str,
) -> Tuple[Tuple[int, int], ...]:
    if start is not None or stop is not None:
        if start is None or stop is None:
            raise ValueError("Provide both --start and --stop, or provide neither")
        return dimer_pairs_from_range(start, stop)
    if atoms is None:
        raise ValueError("Provide --structure when --start/--stop are not set")
    return dimer_pairs_from_structure(atoms, element=dimer_element)


def generate_n2_com_plumed(
    *,
    output_file: Union[str, Path],
    start: Optional[int] = None,
    stop: Optional[int] = None,
    structure_file: Optional[Union[str, Path]] = None,
    structure_format: str = "auto",
    type_map: Optional[Dict[int, str]] = None,
    lammps_atom_style: str = "atomic",
    dimer_element: str = "N",
    with_surface: bool = False,
    surface_element: str = "Si",
    surface_z_tolerance: float = 0.5,
    surface_stride: int = 1,
    config: Optional[PlumedBiasConfig] = None,
    label_prefix: str = "c",
    group_label: str = "reps_center",
) -> N2ComPlumedSummary:
    """Generate a PLUMED file for N2 COM cluster or surface-distance CVs."""
    atoms: Optional[List[AtomRecord]] = None
    if structure_file is not None:
        atoms = read_structure_atoms(
            structure_file,
            structure_format=structure_format,
            type_map=type_map,
            lammps_atom_style=lammps_atom_style,
        )

    dimer_pairs = _resolve_dimer_pairs(
        atoms=atoms,
        start=start,
        stop=stop,
        dimer_element=dimer_element,
    )

    surface_selection = None
    if with_surface:
        if atoms is None:
            raise ValueError("--with-surface requires --structure")
        surface_selection = select_top_layer_atoms(
            atoms,
            element=surface_element,
            z_tolerance=surface_z_tolerance,
            stride=surface_stride,
        )

    write_n2_com_plumed(
        output_file,
        dimer_pairs,
        surface_atom_ids=surface_selection.atom_ids if surface_selection else None,
        config=config,
        label_prefix=label_prefix,
        group_label=group_label,
    )
    mode = "surface-distance" if surface_selection else "cluster-size"
    return N2ComPlumedSummary(
        output_file=Path(output_file),
        dimer_pairs=tuple(dimer_pairs),
        surface_selection=surface_selection,
        mode=mode,
    )


def generate_legacy_n2_com_plumed(
    start: int,
    stop: int,
    output_file: Union[str, Path],
    *,
    config: Optional[PlumedBiasConfig] = None,
) -> N2ComPlumedSummary:
    """Compatibility helper for the old ``gen_plumed.py START STOP`` workflow."""
    return generate_n2_com_plumed(
        output_file=output_file,
        start=start,
        stop=stop,
        config=config,
    )


def iter_atom_ids(pairs: Iterable[Tuple[int, int]]) -> Iterable[int]:
    """Yield atom ids from dimer pairs in pair order."""
    for atom_a, atom_b in pairs:
        yield atom_a
        yield atom_b
