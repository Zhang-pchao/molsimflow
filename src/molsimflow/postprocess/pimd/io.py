"""Strict PIMD input readers and consumed-input verification policies."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from molsimflow.io.hashes import sha256_file as sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def verify_manifest_inputs(
    manifest: Path, run_root: Path, inputs: Sequence[Path],
) -> Dict[str, str]:
    """Verify consumed files against GNU SHA256SUMS, relative to run_root."""
    entries: Dict[Path, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"(\\?)([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, f"invalid SHA256SUMS record at line {line_number}")
        escaped, digest, name = match.groups()
        if escaped:
            # GNU checksum tools escape backslash, newline, and carriage return.
            require(
                re.fullmatch(r"(?:[^\\]|\\[\\nr])*", name) is not None,
                f"invalid SHA256SUMS filename escape at line {line_number}",
            )
            name = re.sub(
                r"\\([\\nr])",
                lambda item: {"\\": "\\", "n": "\n", "r": "\r"}[item.group(1)],
                name,
            )
        path = (run_root / name).resolve()
        require(path not in entries, f"duplicate SHA256SUMS entry: {name}")
        entries[path] = digest.lower()
    require(bool(entries), "empty SHA256SUMS manifest")
    verified: Dict[str, str] = {}
    for path in inputs:
        resolved = path.resolve()
        require(resolved in entries, f"input missing from SHA256SUMS: {path}")
        if str(resolved) in verified:
            continue
        require(path.is_file(), f"input is not a regular file: {path}")
        digest = sha256(path)
        require(digest == entries[resolved], f"input hash mismatch: {path}")
        verified[str(resolved)] = digest
    return verified



def read_plumed(path: Path) -> Tuple[Tuple[str, ...], np.ndarray]:
    fields: Tuple[str, ...] | None = None
    rows: List[List[float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#! FIELDS"):
                candidate = tuple(line.split()[2:])
                if fields is None:
                    fields = candidate
                else:
                    require(fields == candidate, f"FIELDS changed in {path}")
            elif line.startswith("#") or not line.strip():
                continue
            else:
                require(fields is not None, f"missing FIELDS header: {path}")
                values = [float(value) for value in line.split()]
                require(len(values) == len(fields), f"column mismatch: {path}")
                rows.append(values)
    require(fields is not None and rows, f"empty PLUMED table: {path}")
    data = np.asarray(rows, dtype=float)
    require(np.isfinite(data).all(), f"non-finite PLUMED table: {path}")
    return fields, data



def field(data: np.ndarray, fields: Sequence[str], name: str) -> np.ndarray:
    try:
        index = tuple(fields).index(name)
    except ValueError as exc:
        raise ValueError(f"column not found: {name}") from exc
    return data[:, index]



def read_thermo(path: Path) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    header: List[str] | None = None
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if tokens and tokens[0] == "Step" and "Time" in tokens:
                header = tokens
                continue
            if header is None or len(tokens) != len(header):
                continue
            try:
                values = [float(value) for value in tokens]
            except ValueError:
                continue
            row = dict(zip(header, values))
            step = int(round(row["Step"]))
            rows[step] = row
    require(rows, f"no thermo rows: {path}")
    return rows



def dump_frames(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                return
            require(marker.startswith("ITEM: TIMESTEP"), f"bad dump marker: {path}")
            step = int(handle.readline().strip())
            require(handle.readline().startswith("ITEM: NUMBER OF ATOMS"), f"bad atom count: {path}")
            atoms = int(handle.readline().strip())
            require(handle.readline().startswith("ITEM: BOX BOUNDS"), f"bad box header: {path}")
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
            )
            atom_header = handle.readline().split()[2:]
            required = [atom_header.index(name) for name in ("id", "type", "x", "y", "z")]
            values = np.empty((atoms, len(atom_header)), dtype=float)
            for atom in range(atoms):
                tokens = handle.readline().split()
                require(len(tokens) == len(atom_header), f"bad atom row: {path}")
                values[atom] = [float(value) for value in tokens]
            ids = values[:, required[0]].astype(int)
            order = np.argsort(ids)
            types = values[order, required[1]].astype(int)
            positions = values[order][:, required[2:5]]
            yield step, bounds, ids[order], types, positions
