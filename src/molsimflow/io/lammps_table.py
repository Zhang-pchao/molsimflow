"""Named scalar tables emitted by LAMMPS fix ave/time."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_fix_ave_time_table(path: Path) -> tuple[list[str], np.ndarray]:
    """Read named scalar columns with the existing two-comment-header policy.

    This returns raw values. Callers own time selection, unit conversion,
    finite-value checks and restart semantics. Chunk/vector output requires
    a different reader because it has multiple data rows per timestep.
    """

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("#")]
    if len(headers) < 2:
        raise ValueError(f"Expected two comment headers in {path}")
    columns = headers[1].lstrip("# ").split()
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(columns):
        raise ValueError(f"Column mismatch in {path}: {data.shape[1]} != {len(columns)}")
    return columns, data
