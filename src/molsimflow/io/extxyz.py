"""Extended XYZ helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

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


def add_pbc_lattice_to_xyz(
    xyz_path: Union[str, Path],
    poscar_path: Union[str, Path],
    output_path: Union[str, Path],
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
