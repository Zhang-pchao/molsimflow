"""Explicit optional compatibility check against a supplied legacy driver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from molsimflow.io.hashes import sha256_file as sha256

from .reporting import portable_artifact_path
from .units import EV_TO_KCAL_MOL


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def adapt_reference_source(source: str) -> str:
    """Apply the minimal non-square-grid fix to the legacy reference driver."""
    original = "x,y=np.meshgrid(grid_cv_x,grid_cv_y)"
    replacement = "x,y=np.meshgrid(grid_cv_x,grid_cv_y,indexing='ij')"
    require(source.count(original) == 1, "unexpected reference meshgrid implementation")
    return source.replace(original, replacement)


def run_reference(
    config: Mapping[str, object],
    filtered_colvar: Path,
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centroid_kcal: np.ndarray,
    support: np.ndarray,
    kbt_ev: float,
    bandwidth: Sequence[float],
    cv_columns: Sequence[str] = ("logdistance", "ionization"),
    bias_column: str = "opes.bias",
) -> Dict[str, object]:
    driver = Path(str(config["driver"]))
    require(driver.is_file(), f"reference driver missing: {driver}")
    driver_source = driver.read_text(encoding="utf-8")
    adapted_driver = output / "qc" / "FES_from_Reweighting.non-square-grid-compatible.py"
    adapted_driver.write_text(adapt_reference_source(driver_source), encoding="utf-8")

    def command_for(reference: Path, blocks: int) -> List[str]:
        return [
            sys.executable,
            str(adapted_driver),
            "--outfile",
            str(reference),
            "--colvar",
            str(filtered_colvar),
            "--sigma",
            ",".join(str(float(value)) for value in bandwidth),
            "--kt",
            str(float(kbt_ev)),
            "--cv",
            ",".join(str(value) for value in cv_columns),
            "--bias",
            str(bias_column),
            f"--min={float(x_grid[0])},{float(y_grid[0])}",
            f"--max={float(x_grid[-1])},{float(y_grid[-1])}",
            f"--bin={len(x_grid) - 1},{len(y_grid) - 1}",
            "--fmt=%.16e",
            "--blocks",
            str(int(blocks)),
        ]

    def execute(command: Sequence[str], log_path: Path) -> None:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, f"reference driver failed; see {log_path}")

    def load_surface(reference: Path) -> np.ndarray:
        values = np.loadtxt(reference, comments="#")
        require(values.shape[0] == len(x_grid) * len(y_grid), "reference grid size mismatch")
        surface = values[:, 2].reshape(len(x_grid), len(y_grid)).T * EV_TO_KCAL_MOL
        return surface - np.nanmin(surface)

    exact_reference = output / "qc" / "fes-from-reweighting-centroid-exact-eV.dat"
    exact_command = command_for(exact_reference, 1)
    execute(exact_command, output / "qc" / "fes-from-reweighting-exact.log")
    exact_kcal = load_surface(exact_reference)
    exact_difference = np.abs(exact_kcal - centroid_kcal)

    requested_blocks = int(config.get("blocks", 10))
    block_reference = (
        output / "qc" / f"fes-from-reweighting-centroid-blocks{requested_blocks}-eV.dat"
    )
    block_command = command_for(block_reference, requested_blocks)
    execute(block_command, output / "qc" / "fes-from-reweighting-blocks.log")
    block_kcal = load_surface(block_reference)
    block_difference = np.abs(block_kcal - exact_kcal)
    portable_command = [portable_artifact_path(value, output) for value in exact_command]
    portable_block_command = [portable_artifact_path(value, output) for value in block_command]
    return {
        "original_driver": str(driver),
        "original_driver_sha256": sha256(driver),
        "adapted_driver": portable_artifact_path(adapted_driver, output),
        "adapted_driver_sha256": sha256(adapted_driver),
        "compatibility_patch": "np.meshgrid(..., indexing='ij') for non-square 2D grids",
        "bin_semantics": "contract point counts converted to legacy driver interval counts",
        "command": portable_command,
        "max_abs_difference_kcal_mol": float(np.max(exact_difference[support])),
        "rmse_difference_kcal_mol": float(np.sqrt(np.mean(exact_difference[support] ** 2))),
        "reference_file": portable_artifact_path(exact_reference, output),
        "blocks": requested_blocks,
        "block_command": portable_block_command,
        "block_reference_file": portable_artifact_path(block_reference, output),
        "block_vs_exact_max_abs_kcal_mol": float(np.max(block_difference[support])),
        "block_vs_exact_rmse_kcal_mol": float(np.sqrt(np.mean(block_difference[support] ** 2))),
    }
