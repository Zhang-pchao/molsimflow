"""Fixed mixed-bead topology, unique-coordinate checks, and native dump admission."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records

_SCHEMA = "molsimflow.mixed-beads.v1"


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must contain positive integers")
    if value <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return int(value)


@dataclass(frozen=True)
class MixedBeadTopology:
    """Fixed atom identities and nested time slices, distinct from CV contraction.

    ``atom_ids`` defines the atom axis. ``partitions`` is the number of complete
    force configurations, not the number of independent beads for every atom.
    """

    atom_ids: tuple[int, ...]
    bead_counts: tuple[int, ...]
    partitions: int

    def __post_init__(self) -> None:
        ids = tuple(_integer(v, "atom_ids") for v in self.atom_ids)
        counts = tuple(_integer(v, "bead_counts") for v in self.bead_counts)
        p = _integer(self.partitions, "partitions")
        if not ids or len(set(ids)) != len(ids) or len(ids) != len(counts):
            raise ValueError("atom_ids must be nonempty, unique and match bead_counts")
        if p & (p - 1) or any(n & (n - 1) or n > p for n in counts):
            raise ValueError("bead counts must be nested powers of two within partitions")
        if max(counts) != p:
            raise ValueError("maximum bead count must equal partitions")
        object.__setattr__(self, "atom_ids", ids)
        object.__setattr__(self, "bead_counts", counts)
        object.__setattr__(self, "partitions", p)

    @property
    def is_mixed(self) -> bool:
        return len(set(self.bead_counts)) > 1

    @property
    def unique_coordinate_count(self) -> int:
        """Number of independent three-component coordinates in one path."""
        return sum(self.bead_counts)

    def bead_indices(self) -> np.ndarray:
        """Zero-based unique-bead index, with shape (partitions, atoms)."""
        return np.arange(self.partitions)[:, None] * np.asarray(self.bead_counts) // self.partitions

    def representatives(self) -> np.ndarray:
        """Boolean ownership mask in complete-configuration order."""
        strides = self.partitions // np.asarray(self.bead_counts)
        return np.arange(self.partitions)[:, None] % strides == 0

    def unique_bead_weights(self) -> tuple[np.ndarray, ...]:
        """Per-atom observable weights; these are not independent frame weights."""
        return tuple(np.full(count, 1.0 / count) for count in self.bead_counts)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "nuclear_ensemble": "fixed_mixed_bead",
            "mapping": "contiguous_nested",
            "partitions": self.partitions,
            "atom_ids": list(self.atom_ids),
            "bead_counts": list(self.bead_counts),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> MixedBeadTopology:
        if not isinstance(metadata, Mapping):
            raise ValueError("mixed-bead metadata must be an object")
        expected = {"schema", "nuclear_ensemble", "mapping", "partitions", "atom_ids", "bead_counts"}
        if set(metadata) != expected:
            raise ValueError("mixed-bead metadata fields do not match the versioned schema")
        if metadata["schema"] != _SCHEMA or metadata["nuclear_ensemble"] != "fixed_mixed_bead":
            raise ValueError("unsupported mixed-bead schema or nuclear ensemble")
        if metadata["mapping"] != "contiguous_nested":
            raise ValueError("unsupported mixed-bead time-slice mapping")
        if not isinstance(metadata["atom_ids"], (list, tuple)) or not isinstance(
            metadata["bead_counts"], (list, tuple)
        ):
            raise ValueError("mixed-bead atom_ids and bead_counts must be arrays")
        return cls(tuple(metadata["atom_ids"]), tuple(metadata["bead_counts"]), metadata["partitions"])

    def validate_coordinates(self, positions: np.ndarray, *, tolerance: float = 1e-10) -> float:
        """Check finite unwrapped coordinates and agreement of repeated copies.

        No periodic-image guessing or post-hoc coordinate averaging is applied.
        Tolerance is an absolute distance in the explicitly selected input units.
        """
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("copy tolerance must be finite and nonnegative")
        values = np.asarray(positions, dtype=float)
        if values.shape != (self.partitions, len(self.atom_ids), 3):
            raise ValueError("coordinates must have shape (partitions, atoms, 3)")
        if not np.isfinite(values).all():
            raise ValueError("mixed-bead coordinates must be finite")
        maximum = 0.0
        for atom, count in enumerate(self.bead_counts):
            stride = self.partitions // count
            owners = np.repeat(values[::stride, atom], stride, axis=0)
            maximum = max(maximum, float(np.max(np.abs(values[:, atom] - owners))))
        if maximum > tolerance:
            raise ValueError("copies of an independent mixed bead disagree")
        return maximum

    def unique_coordinates(self, positions: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one copied array per atom after strict copy validation."""
        self.validate_coordinates(positions)
        values = np.asarray(positions, dtype=float)
        return tuple(values[::self.partitions // count, atom].copy()
                     for atom, count in enumerate(self.bead_counts))

    def physical_force_pullback(self, forces: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return physical -dU/dq with U=sum_b E(R_b)/P, without mass rescaling."""
        values = np.asarray(forces, dtype=float)
        if values.shape != (self.partitions, len(self.atom_ids), 3) or not np.isfinite(values).all():
            raise ValueError("forces must be finite with shape (partitions, atoms, 3)")
        return tuple(values[:, atom].reshape(count, self.partitions // count, 3).sum(axis=1)
                     / self.partitions for atom, count in enumerate(self.bead_counts))


@dataclass(frozen=True)
class MixedPimdPathFrame:
    """One complete path; positions are continuous UNWRAPPED Cartesian values."""

    step: int
    topology: MixedBeadTopology
    atom_types: np.ndarray
    bounds: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None
    maximum_copy_error: float
    maximum_velocity_copy_error: float | None


def iter_mixed_pimd_path_frames(
    partition_paths: Mapping[int, Path],
    *,
    topology: MixedBeadTopology,
    expected_identity: Mapping[int, int],
    metadata_columns: Sequence[str] = ("f_pi[1]", "f_pi[2]", "f_pi[3]"),
    copy_tolerance: float = 1e-10,
    velocity_copy_tolerance: float = 1e-10,
) -> Iterator[MixedPimdPathFrame]:
    """Read native pimd/mixed dumps with all copies, identities and indices checked.

    Partition keys are the zero-based LAMMPS partition indices. Require explicit
    xu/yu/zu columns and the fix's count, one-based bead index, owner flag. Wrapped
    coordinates cannot substitute for continuous paths. All three velocity
    columns are optional together; when present their copies are also checked
    with a separate tolerance in the input velocity units.
    This first reader reuses the strict orthorhombic dump parser, supports fixed
    pp/ff boundaries, and rejects tilted or changing cells. Each yielded object
    is ONE statistical frame. Exhaust the iterator to validate all tails.
    """
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
           for k in partition_paths) or set(partition_paths) != set(range(topology.partitions)):
        raise ValueError("partition paths must contain every zero-based partition exactly once")
    paths = [Path(partition_paths[b]).expanduser().resolve() for b in range(topology.partitions)]
    if len(set(paths)) != len(paths):
        raise ValueError("partition inputs must be distinct immutable files")
    identity = {_integer(tag, "atom IDs"): _integer(kind, "atom types")
                for tag, kind in expected_identity.items()}
    if set(identity) != set(topology.atom_ids):
        raise ValueError("expected_identity must match the fixed mixed-bead atom IDs")
    columns = tuple(metadata_columns)
    if (len(columns) != 3 or not all(isinstance(v, str) and v for v in columns)
            or len(set(columns)) != 3):
        raise ValueError("metadata_columns must name count, bead index and representative")
    reserved = {"id", "type", "x", "y", "z", "xs", "ys", "zs", "xu", "yu", "zu",
                "xsu", "ysu", "zsu", "ix", "iy", "iz", "vx", "vy", "vz"}
    if reserved.intersection(columns):
        raise ValueError("metadata_columns must not alias atom identity, position or velocity fields")
    if not np.isfinite(copy_tolerance) or copy_tolerance < 0:
        raise ValueError("copy tolerance must be finite and nonnegative")
    if not np.isfinite(velocity_copy_tolerance) or velocity_copy_tolerance < 0:
        raise ValueError("velocity copy tolerance must be finite and nonnegative")
    tag_to_index = {tag: index for index, tag in enumerate(topology.atom_ids)}
    indices, owners = topology.bead_indices(), topology.representatives()
    previous, cell, box_header, velocity_presence = None, None, None, None
    generators = [iter_lammps_dump_records(path) for path in paths]
    try:
        for frames in zip_longest(*generators):
            if any(frame is None for frame in frames):
                raise ValueError("mixed-bead trajectories have different frame counts")
            step = frames[0].timestep
            if step < 0 or (previous is not None and step <= previous):
                raise ValueError("mixed-bead steps must be nonnegative and strictly increasing")
            if any(frame.timestep != step for frame in frames):
                raise ValueError("mixed-bead partition timesteps do not match")
            previous = step
            positions, velocities = [], []
            for b, frame in enumerate(frames):
                boundary = frame.box_header.split()[3:]
                if len(boundary) != 3 or any(value not in {"pp", "ff"} for value in boundary):
                    raise ValueError("mixed-bead reader requires fixed orthorhombic pp/ff boundaries")
                if not np.isfinite(frame.bounds).all() or np.any(frame.bounds[:, 1] <= frame.bounds[:, 0]):
                    raise ValueError("invalid mixed-bead box bounds")
                if cell is None:
                    cell, box_header = frame.bounds.copy(), frame.box_header
                if not np.array_equal(frame.bounds, cell) or frame.box_header != box_header:
                    raise ValueError("mixed-bead cell must match across partitions and frames")
                fields = frame.atom_fields
                if len(set(fields)) != len(fields):
                    raise ValueError("duplicate mixed-bead atom columns")
                required = ("id", "type", "xu", "yu", "zu", *columns)
                if not set(required).issubset(fields):
                    raise ValueError("mixed-bead dump requires id type xu yu zu and all metadata columns")
                have_velocity = {"vx", "vy", "vz"}.intersection(fields)
                if have_velocity and len(have_velocity) != 3:
                    raise ValueError("mixed-bead velocities require vx vy vz together")
                if velocity_presence is None:
                    velocity_presence = bool(have_velocity)
                if bool(have_velocity) != velocity_presence:
                    raise ValueError("mixed-bead velocity columns must match in all frames")
                rows = {}
                for row in frame.atom_rows:
                    tag, kind = int(row[fields.index("id")]), int(row[fields.index("type")])
                    if tag in rows or identity.get(tag) != kind:
                        raise ValueError("duplicate or unexpected mixed-bead atom ID/type")
                    atom = tag_to_index[tag]
                    actual = [float(row[fields.index(name)]) for name in columns]
                    expected = [topology.bead_counts[atom], indices[b, atom]+1, int(owners[b, atom])]
                    if actual != expected:
                        raise ValueError("dump bead metadata does not match the declared topology")
                    xyz = [float(row[fields.index(name)]) for name in ("xu", "yu", "zu")]
                    velocity = [float(row[fields.index(name)]) for name in ("vx", "vy", "vz")] if have_velocity else None
                    rows[tag] = (xyz, velocity)
                if set(rows) != set(identity):
                    raise ValueError("incomplete mixed-bead atom identity")
                positions.append([rows[tag][0] for tag in topology.atom_ids])
                if velocity_presence:
                    velocities.append([rows[tag][1] for tag in topology.atom_ids])
            xyz = np.asarray(positions, dtype=float)
            maximum = topology.validate_coordinates(xyz, tolerance=copy_tolerance)
            velocity_array = np.asarray(velocities, dtype=float) if velocity_presence else None
            maximum_velocity = None
            if velocity_array is not None:
                # Exact native output repeats the same Cartesian velocity for copies.
                maximum_velocity = topology.validate_coordinates(
                    velocity_array, tolerance=velocity_copy_tolerance
                )
            yield MixedPimdPathFrame(step, topology,
                                    np.array([identity[tag] for tag in topology.atom_ids]),
                                    cell.copy(), xyz, velocity_array, maximum, maximum_velocity)
    finally:
        for generator in generators:
            generator.close()
    if previous is None:
        raise ValueError("mixed-bead trajectories have no complete frames")
