"""Minimal POSCAR lattice parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def read_poscar_lattice(path: Union[str, Path]) -> np.ndarray:
    """Read the 3x3 lattice matrix from a VASP POSCAR-style file."""
    poscar_path = Path(path)
    lines = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 5:
        raise ValueError(f"POSCAR file is too short: {poscar_path}")

    scale_values = [float(value) for value in lines[1].split()]
    if len(scale_values) == 1:
        scale = np.array([scale_values[0], scale_values[0], scale_values[0]], dtype=float)
    elif len(scale_values) == 3:
        scale = np.array(scale_values, dtype=float)
    else:
        raise ValueError(f"Unsupported POSCAR scale line in {poscar_path}: {lines[1]!r}")

    lattice = np.array(
        [[float(value) for value in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    )
    return lattice * scale[:, None]
