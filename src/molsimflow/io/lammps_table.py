"""Named scalar tables emitted by LAMMPS fix ave/time."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FixAveChunkBlock:
    """One complete chunk block, preserving named header values and rows."""

    metadata: dict[str, float]
    columns: tuple[str, ...]
    data: np.ndarray


def iter_fix_ave_chunk_blocks(path: Path):
    """Read the three-header scalar-block layout emitted by fix ave/chunk.

    The second header names per-block metadata and its row/chunk count.
    The third names each chunk row. Truncated blocks and schema changes fail.
    Callers interpret timestep units, particle counts and density columns.
    """
    with Path(path).open(encoding="utf-8") as stream:
        headers = []
        for raw in stream:
            if not raw.strip():
                continue
            if not raw.startswith("#"):
                raise ValueError("Expected three fix-ave/chunk comment headers")
            headers.append(raw.lstrip("# ").split())
            if len(headers) == 3:
                break
        if len(headers) != 3:
            raise ValueError("Incomplete fix-ave/chunk header")
        metadata_names, columns = headers[1], headers[2]
        count_fields = [
            name for name in ("Number-of-rows", "Number-of-chunks") if name in metadata_names
        ]
        if len(count_fields) != 1:
            raise ValueError("Exactly one Number-of-rows or Number-of-chunks column is required")
        count_field = count_fields[0]
        if len(set(metadata_names)) != len(metadata_names) or len(set(columns)) != len(columns):
            raise ValueError("Duplicate chunk column names")
        for raw in stream:
            if not raw.strip():
                continue
            if raw.startswith("#"):
                raise ValueError("Repeated chunk headers require explicit segment handling")
            values = np.asarray(raw.split(), dtype=float)
            if len(values) != len(metadata_names) or not np.isfinite(values).all():
                raise ValueError("Invalid chunk block metadata")
            count = values[metadata_names.index(count_field)]
            if count < 1 or count != int(count):
                raise ValueError("Chunk row count must be a positive integer")
            rows = []
            for _ in range(int(count)):
                line = next(stream, None)
                if line is None:
                    raise ValueError("Truncated fix-ave/chunk block")
                row = np.asarray(line.split(), dtype=float)
                if len(row) != len(columns) or not np.isfinite(row).all():
                    raise ValueError("Invalid chunk row")
                rows.append(row)
            yield FixAveChunkBlock(
                dict(zip(metadata_names, values.tolist())), tuple(columns), np.asarray(rows)
            )
