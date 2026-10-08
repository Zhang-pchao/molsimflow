"""Extended XYZ helpers."""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np

from molsimflow.io.poscar import read_poscar_lattice


def _parse_xyz_atom_line(line: str) -> tuple[str, float, float, float]:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"XYZ atom line has fewer than four columns: {line!r}")
    symbol = parts[0]
    x, y, z = (float(value) for value in parts[-3:])
    return symbol, x, y, z


def format_lattice_for_extxyz(lattice: np.ndarray) -> str:
    """Format a 3x3 lattice matrix for an extended XYZ comment line."""
    array = np.asarray(lattice, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 lattice matrix, got shape {array.shape}")
    return " ".join(f"{value:.12g}" for value in array.reshape(-1))


def read_extxyz_positions(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read elements, Cartesian positions, and an orthorhombic cell from extxyz."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is too short: {path}")
    atom_count = int(lines[0].split()[0])
    if len(lines) < atom_count + 2:
        raise ValueError(f"XYZ file ended before {atom_count} atoms were read: {path}")
    lattice = None
    for token in shlex.split(lines[1]):
        if token.lower().startswith("lattice="):
            lattice = np.asarray(token.split("=", 1)[1].split(), dtype=float).reshape(3, 3)
            break
    if lattice is None or not np.allclose(lattice, np.diag(np.diag(lattice))):
        raise ValueError("An orthorhombic extxyz Lattice is required")
    fields = [line.split() for line in lines[2 : atom_count + 2]]
    if any(len(field) < 4 for field in fields):
        raise ValueError(f"XYZ atom line has fewer than four columns: {path}")
    elements = np.asarray([field[0] for field in fields])
    coordinates = np.asarray([[float(value) for value in field[1:4]] for field in fields])
    return elements, coordinates, np.diag(lattice)


def add_pbc_lattice_to_xyz(
    xyz_path: str | Path,
    poscar_path: str | Path,
    output_path: str | Path,
    *,
    z_min_padding: float = 5.0,
) -> Path:
    """Write an extended XYZ file with PBC metadata and shifted z coordinates.

    Coordinates are shifted so the minimum atom z coordinate becomes
    `z_min_padding`.  This preserves the behavior of the legacy helper while
    making all paths explicit.
    """
    input_path = Path(xyz_path)
    output = Path(output_path)
    lattice = read_poscar_lattice(poscar_path)

    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is too short: {input_path}")

    atom_count = int(lines[0].split()[0])
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"XYZ file ended before {atom_count} atoms were read: {input_path}")

    atoms = [_parse_xyz_atom_line(line) for line in atom_lines]
    z_min = min(atom[3] for atom in atoms)
    z_shift = z_min_padding - z_min

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(f"{atom_count}\n")
        handle.write(
            'pbc="T T T" '
            f'lattice="{format_lattice_for_extxyz(lattice)}" '
            "properties=species:S:1:pos:R:3\n"
        )
        for symbol, x, y, z in atoms:
            handle.write(f"{symbol} {x:.6f} {y:.6f} {z + z_shift:.6f}\n")
    return output
