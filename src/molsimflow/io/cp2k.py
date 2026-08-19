"""Parsers for compact CP2K energy-and-force outputs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_A = 51.422067476325

_ENERGY_RE = re.compile(
    r"ENERGY\|\s+Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s*([-+0-9.eEdD]+)"
)
_FORCE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+[A-Za-z]+\s+"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)"
)


@dataclass(frozen=True)
class Cp2kEnergyForces:
    """The final energy and complete atomic-force block in a CP2K output."""

    energy_hartree: float
    forces_eV_A: np.ndarray

    @property
    def energy_eV(self) -> float:
        return self.energy_hartree * HARTREE_TO_EV


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _force_blocks(lines: Sequence[str]) -> list[tuple[int, np.ndarray]]:
    blocks: list[tuple[int, np.ndarray]] = []
    for marker, line in enumerate(lines):
        if "ATOMIC FORCES in" not in line:
            continue
        rows: list[list[float]] = []
        complete = False
        for candidate in lines[marker + 1 :]:
            if "SUM OF ATOMIC FORCES" in candidate:
                complete = True
                break
            match = _FORCE_ROW_RE.match(candidate)
            if match is None:
                continue
            atom_index = int(match.group(1))
            if atom_index != len(rows) + 1:
                raise ValueError(f"Non-contiguous CP2K force row: got atom {atom_index}")
            rows.append([_as_float(value) for value in match.groups()[1:]])
        if complete and rows:
            blocks.append(
                (
                    marker,
                    np.asarray(rows, dtype=float) * HARTREE_PER_BOHR_TO_EV_PER_A,
                )
            )
    return blocks


def parse_cp2k_energy_forces(
    output_path: Path,
    atom_count: int | None = None,
) -> Cp2kEnergyForces:
    """Read the final energy and final complete force block from CP2K output.

    CP2K may append several geometry or force evaluations to one file. Selecting
    the final complete blocks avoids pairing an early force table with the last
    reported energy. Forces are returned in eV/Angstrom.
    """

    lines = Path(output_path).read_text(encoding="utf-8", errors="replace").splitlines()
    energies = [
        (index, _as_float(match.group(1)))
        for index, line in enumerate(lines)
        if (match := _ENERGY_RE.search(line))
    ]
    if not energies:
        raise ValueError(f"CP2K output has no FORCE_EVAL energy: {output_path}")
    blocks = _force_blocks(lines)
    if not blocks:
        raise ValueError(f"CP2K output has no complete ATOMIC FORCES block: {output_path}")
    force_marker, forces = blocks[-1]
    paired_energies = [value for index, value in energies if index < force_marker]
    if not paired_energies:
        raise ValueError(f"Complete CP2K force block has no preceding energy: {output_path}")
    if atom_count is not None and forces.shape != (int(atom_count), 3):
        raise ValueError(
            f"Expected {atom_count} CP2K force rows, got {forces.shape[0]} in {output_path}"
        )
    return Cp2kEnergyForces(energy_hartree=paired_energies[-1], forces_eV_A=forces)
