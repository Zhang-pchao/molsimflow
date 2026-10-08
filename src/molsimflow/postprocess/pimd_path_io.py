"""Strict, streaming assembly of complete orthorhombic PIMD dump frames."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records


@dataclass(frozen=True)
class PimdPathFrame:
    """One physical frame; bead order is the explicitly supplied topology."""

    step: int
    bead_ids: tuple[int, ...]
    atom_ids: np.ndarray
    atom_types: np.ndarray
    bounds: np.ndarray
    positions: np.ndarray


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must contain integers")
    if value < minimum:
        raise ValueError(f"{name} entries must be at least {minimum}")
    return int(value)


def iter_pimd_path_frames(
    bead_paths: Mapping[int, Path],
    *,
    bead_order: Sequence[int],
    expected_identity: Mapping[int, int],
    selected_steps: Optional[Sequence[int]] = None,
) -> Iterator[PimdPathFrame]:
    """Read aligned complete paths with fixed tags/types and fully periodic boxes.

    Input paths must be distinct immutable files. Their bead identities come
    from the caller's hash-locked manifest: LAMMPS dump headers cannot prove
    that a file was assigned to the correct bead. Atom rows may be reordered;
    tags/types must match ``expected_identity`` at every frame. Duplicate or
    unmatched steps, missing beads, incomplete tails and non-orthorhombic or
    non-periodic boxes raise ``ValueError``. No frames are silently truncated.

    Coordinates remain wrapped; ``quantum_path.describe_quantum_path`` applies
    its explicit compact-path unwrapping rule. Optional image columns are
    validated as integers but are not used to average independently drifting
    bead images. All input frames, including those outside ``selected_steps``,
    are checked when this iterator is exhausted. Early consumer termination
    does not validate unread tails. This reader does not establish equilibrium,
    bias alignment, a numerical gate or scientific admission.
    """
    order = tuple(_integer(b, "bead_order", 0) for b in bead_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("bead_order must be nonempty and unique")
    path_ids = tuple(_integer(b, "bead_paths keys", 0) for b in bead_paths)
    if set(path_ids) != set(order):
        raise ValueError("bead_paths must match the complete declared bead_order")
    paths = [Path(bead_paths[b]).expanduser().resolve() for b in order]
    if len(set(paths)) != len(paths):
        raise ValueError("each bead must have a distinct input path")
    identity = {
        _integer(tag, "atom IDs", 1): _integer(kind, "atom types", 1)
        for tag, kind in expected_identity.items()
    }
    if not identity:
        raise ValueError("expected_identity must be nonempty")
    atom_ids = np.asarray(sorted(identity), dtype=np.int64)
    atom_types = np.asarray([identity[tag] for tag in atom_ids], dtype=np.int64)
    wanted = None
    if selected_steps is not None:
        selected = tuple(_integer(s, "selected_steps", 0) for s in selected_steps)
        if not selected or selected != tuple(sorted(set(selected))):
            raise ValueError("selected_steps must be nonempty, unique and increasing")
        wanted = set(selected)
    found = set()
    previous_step = None
    generators = [iter_lammps_dump_records(p) for p in paths]
    try:
        for frames in zip_longest(*generators):
            if any(f is None for f in frames):
                raise ValueError("bead trajectories have different frame counts")
            step = frames[0].timestep
            if step < 0 or (previous_step is not None and step <= previous_step):
                raise ValueError("physical steps must be nonnegative and strictly increasing")
            if any(f.timestep != step for f in frames):
                raise ValueError("bead trajectory timesteps do not match")
            previous_step = step
            positions = []
            for frame in frames:
                if frame.box_header.split()[3:] != ["pp", "pp", "pp"]:
                    raise ValueError("only fully periodic orthorhombic boxes are supported")
                if not np.isfinite(frame.bounds).all() or np.any(
                    frame.bounds[:, 1] <= frame.bounds[:, 0]
                ):
                    raise ValueError("invalid box bounds")
                if not np.array_equal(frame.bounds, frames[0].bounds):
                    raise ValueError("bead boxes do not match")
                fields = frame.atom_fields
                if len(set(fields)) != len(fields):
                    raise ValueError("duplicate atom columns")
                required = ("id", "type", "x", "y", "z")
                if not set(required).issubset(fields):
                    raise ValueError("atom columns must include id type x y z")
                image_fields = {"ix", "iy", "iz"}.intersection(fields)
                if image_fields and len(image_fields) != 3:
                    raise ValueError("image columns must include all of ix iy iz")
                rows = {}
                for row in frame.atom_rows:
                    tag = int(row[fields.index("id")])
                    kind = int(row[fields.index("type")])
                    if tag in rows or identity.get(tag) != kind:
                        raise ValueError("duplicate or unexpected atom ID/type")
                    xyz = [float(row[fields.index(axis)]) for axis in ("x", "y", "z")]
                    if not np.isfinite(xyz).all():
                        raise ValueError("non-finite atom coordinates")
                    for name in image_fields:
                        int(row[fields.index(name)])
                    rows[tag] = xyz
                if set(rows) != set(identity):
                    raise ValueError("incomplete atom identity")
                positions.append([rows[tag] for tag in atom_ids])
            if wanted is None or step in wanted:
                found.add(step)
                yield PimdPathFrame(
                    step, order, atom_ids.copy(), atom_types.copy(),
                    frames[0].bounds.copy(), np.asarray(positions, dtype=float),
                )
    finally:
        for generator in generators:
            generator.close()
    if previous_step is None:
        raise ValueError("bead trajectories have no complete frames")
    if wanted is not None and found != wanted:
        raise ValueError("selected physical steps are missing")
