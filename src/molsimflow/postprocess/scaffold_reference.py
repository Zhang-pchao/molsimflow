"""Immutable Si/bridge-O observer reference with conservative rigid registration.

Only orthorhombic p-p-f boxes and translation are supported. A fixed scaffold
height datum is not a definition of a water layer or of terminal surface height.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors


@dataclass(frozen=True)
class ScaffoldReference:
    """Tuple-only reference returned by :func:`build_scaffold_reference`."""

    atom_ids: tuple[int, ...]
    elements: tuple[str, ...]
    coordinates_A: tuple[tuple[float, float, float], ...]
    bounds_A: tuple[tuple[float, float], ...]
    silicon_ids: tuple[int, ...]
    bridge_oxygen_ids: tuple[int, ...]
    top_ids: tuple[int, ...]
    deformation_tolerance_A: float
    wrapping_tolerance_A: float

    def __post_init__(self) -> None:
        # Defensive tuple conversion also covers direct construction / dataclasses.replace.
        ids, labels, xyz = _atoms(self.atom_ids, self.elements, self.coordinates_A)
        bounds = _box(self.bounds_A, (True, True, False))
        selected = {}
        for name in ("silicon_ids", "bridge_oxygen_ids", "top_ids"):
            selected[name] = tuple(sorted(_ids(getattr(self, name), name,
                                               allow_empty=name == "bridge_oxygen_ids")))
        if (set(selected["silicon_ids"]) & set(selected["bridge_oxygen_ids"])
                or set(selected["silicon_ids"] + selected["bridge_oxygen_ids"]) != set(ids)
                or not set(selected["top_ids"]) <= set(ids)):
            raise ValueError("Reference IDs must exactly partition into Si/bridge O with scaffold top IDs")
        lookup = {atom_id: index for index, atom_id in enumerate(ids)}
        for name, element in (("silicon_ids", "Si"), ("bridge_oxygen_ids", "O")):
            if any(labels[lookup[atom_id]] != element for atom_id in selected[name]):
                raise ValueError("Reference scaffold contains an undeclared element")
        limit = float(np.min(box_lengths(bounds)[:2])) / 4
        if (not np.isfinite(self.deformation_tolerance_A)
                or not 0 <= self.deformation_tolerance_A < limit
                or not np.isfinite(self.wrapping_tolerance_A)
                or not 0 < self.wrapping_tolerance_A < limit):
            raise ValueError("Invalid frozen deformation or wrapping tolerance")
        order = sorted(range(len(ids)), key=lambda index: ids[index])
        canonical = _wrap_xy(xyz[order], bounds)
        object.__setattr__(self, "atom_ids", tuple(ids[index] for index in order))
        object.__setattr__(self, "elements", tuple(labels[index] for index in order))
        object.__setattr__(self, "coordinates_A", tuple(tuple(float(v) for v in row) for row in canonical))
        object.__setattr__(self, "bounds_A", tuple(tuple(float(v) for v in row) for row in bounds))
        for name, value in selected.items():
            object.__setattr__(self, name, value)
        for name in ("deformation_tolerance_A", "wrapping_tolerance_A"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def height_datum_A(self) -> float:
        selected = [xyz[2] for atom_id, xyz in zip(self.atom_ids, self.coordinates_A)
                    if atom_id in self.top_ids]
        return float(np.median(selected))

    @property
    def sha256(self) -> str:
        """Hash sorted identity, canonical geometry and observer conventions."""
        def hex_float(value: float) -> str:
            return float(0.0 if value == 0 else value).hex()

        payload = dict(
            schema="immutable_scaffold_reference_v1",
            atom_ids=self.atom_ids, elements=self.elements,
            coordinates_A=[[hex_float(v) for v in xyz] for xyz in self.coordinates_A],
            bounds_A=[[hex_float(v) for v in row] for row in self.bounds_A],
            silicon_ids=self.silicon_ids, bridge_oxygen_ids=self.bridge_oxygen_ids,
            top_ids=self.top_ids, periodic=[True, True, False],
            deformation_tolerance_A=hex_float(self.deformation_tolerance_A),
            wrapping_tolerance_A=hex_float(self.wrapping_tolerance_A),
            registration="matched_ID_median_translation_minimum_image_XY_reject_half_box_v1",
            height_datum="median_reference_z_of_explicit_top_scaffold_IDs",
            coordinates="angstrom_XY_wrapped_to_reference_bounds_Z_unwrapped",
            layer_offset="external_not_part_of_height_datum",
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ScaffoldRegistration:
    """Frame-minus-reference rigid translation; XY is a minimum-image shift."""

    translation_A: tuple[float, float, float]
    max_residual_A: float
    reference_hash: str

    def __post_init__(self) -> None:
        shift = np.asarray(self.translation_A, dtype=float)
        if (shift.shape != (3,) or not np.isfinite(shift).all()
                or not np.isfinite(self.max_residual_A) or self.max_residual_A < 0
                or not isinstance(self.reference_hash, str) or not self.reference_hash):
            raise ValueError("Invalid immutable registration record")
        object.__setattr__(self, "translation_A", tuple(float(v) for v in shift))
        object.__setattr__(self, "max_residual_A", float(self.max_residual_A))


def _ids(values: Sequence[int], name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    ids = tuple(values)
    if (not ids and not allow_empty) or any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value <= 0
        for value in ids
    ):
        raise ValueError(f"{name} requires positive integer atom IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {name}")
    return tuple(int(value) for value in ids)


def _box(bounds_A: Sequence[Sequence[float]], periodic: Sequence[bool]) -> np.ndarray:
    if tuple(periodic) != (True, True, False) or any(
        not isinstance(flag, (bool, np.bool_)) for flag in periodic
    ):
        raise ValueError("Only the explicit p-p-f periodicity contract is supported")
    bounds = np.asarray(bounds_A, dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("Axis-aligned finite (3, 2) bounds are required; triclinic cells unsupported")
    if np.any(box_lengths(bounds) <= 0):
        raise ValueError("Box lengths must be positive")
    return bounds


def _atoms(atom_ids, elements, coordinates_A):
    ids = _ids(atom_ids, "atom_ids")
    labels = tuple(elements)
    xyz = np.asarray(coordinates_A, dtype=float)
    if len(labels) != len(ids) or xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError("Atom IDs, elements and finite N-by-3 coordinates must agree")
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise ValueError("Elements must be nonempty symbols")
    return ids, labels, xyz


def _wrap_xy(coordinates: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    result = np.array(coordinates, dtype=float, copy=True)
    result[..., :2] = (result[..., :2] - bounds[:2, 0]) % box_lengths(bounds)[:2] + bounds[:2, 0]
    return result


def build_scaffold_reference(
    atom_ids: Sequence[int], elements: Sequence[str], coordinates_A: Sequence[Sequence[float]],
    bounds_A: Sequence[Sequence[float]], *, silicon_ids: Sequence[int],
    bridge_oxygen_ids: Sequence[int], top_ids: Sequence[int], deformation_tolerance_A: float,
    wrapping_tolerance_A: float = 1e-8, periodic: Sequence[bool] = (True, True, False),
) -> ScaffoldReference:
    """Freeze explicitly selected unchanged Si and bridge O, independent of terminals.

    bridge_oxygen_ids must be externally established unchanged bridge atoms;
    this geometry helper does not infer bonding or determine chemical stability.
    top_ids must be a nonempty subset of the selected scaffold. The top median
    is a fixed datum; apply any physical layer offset separately at observation.
    """
    ids, labels, xyz = _atoms(atom_ids, elements, coordinates_A)
    bounds = _box(bounds_A, periodic)
    silicon = tuple(sorted(_ids(silicon_ids, "silicon_ids")))
    bridges = tuple(sorted(_ids(bridge_oxygen_ids, "bridge_oxygen_ids", allow_empty=True)))
    top = tuple(sorted(_ids(top_ids, "top_ids")))
    if set(silicon) & set(bridges):
        raise ValueError("Si and bridge-O ID sets must be disjoint")
    scaffold = tuple(sorted(silicon + bridges))
    if not set(scaffold) <= set(ids) or not set(top) <= set(scaffold):
        raise ValueError("Missing scaffold IDs or top IDs outside the explicit scaffold")
    quarter_box = float(np.min(box_lengths(bounds)[:2])) / 4
    if (not np.isfinite(deformation_tolerance_A) or deformation_tolerance_A < 0
            or deformation_tolerance_A >= quarter_box):
        raise ValueError("Explicit deformation tolerance must be finite, nonnegative and below Lxy_min/4")
    if (not np.isfinite(wrapping_tolerance_A) or wrapping_tolerance_A <= 0
            or wrapping_tolerance_A >= quarter_box):
        raise ValueError("Wrapping ambiguity tolerance must be positive and below Lxy_min/4")
    lookup = {atom_id: index for index, atom_id in enumerate(ids)}
    for selected, expected in ((silicon, "Si"), (bridges, "O")):
        if any(labels[lookup[atom_id]] != expected for atom_id in selected):
            raise ValueError("Selected scaffold must contain only declared Si and bridge O elements")
    indices = [lookup[atom_id] for atom_id in scaffold]
    fixed = _wrap_xy(xyz[indices], bounds)
    return ScaffoldReference(
        scaffold, tuple(labels[i] for i in indices),
        tuple(tuple(float(value) for value in row) for row in fixed),
        tuple(tuple(float(value) for value in row) for row in bounds), silicon, bridges, top,
        float(deformation_tolerance_A), float(wrapping_tolerance_A),
    )


def register_scaffold(
    reference: ScaffoldReference, atom_ids: Sequence[int], elements: Sequence[str],
    coordinates_A: Sequence[Sequence[float]], bounds_A: Sequence[Sequence[float]], *,
    periodic: Sequence[bool] = (True, True, False),
) -> ScaffoldRegistration:
    """Fit only rigid translation to matched immutable IDs, failing on deformation.

    XY displacements are unwrapped around the smallest-ID scaffold atom before
    a componentwise median is taken. Z never uses PBC. A half-box consensus or
    half-box difference to that anchor is ambiguous and rejected. All scaffold
    residual norms must satisfy the reference's explicitly frozen tolerance.
    """
    ids, labels, xyz = _atoms(atom_ids, elements, coordinates_A)
    bounds = _box(bounds_A, periodic)
    if not np.array_equal(bounds, np.asarray(reference.bounds_A)):
        raise ValueError("Reference and frame bounds differ; changing boxes are unsupported")
    lookup = {atom_id: index for index, atom_id in enumerate(ids)}
    if not set(reference.atom_ids) <= set(ids):
        raise ValueError("Missing immutable scaffold atom IDs")
    indices = [lookup[atom_id] for atom_id in reference.atom_ids]
    if tuple(labels[index] for index in indices) != reference.elements:
        raise ValueError("Immutable scaffold element changed")
    displacement = xyz[indices] - np.asarray(reference.coordinates_A)
    lengths = box_lengths(bounds)[:2]
    xy = minimum_image_vectors(displacement[:, :2], lengths)
    relative = minimum_image_vectors(xy - xy[0], lengths)
    if np.any(np.abs(np.abs(relative) - lengths / 2) <= reference.wrapping_tolerance_A):
        raise ValueError("Ambiguous half-box wrapping among scaffold displacements")
    unwrapped = xy[0] + relative
    shift_xy = minimum_image_vectors(np.median(unwrapped, axis=0), lengths)
    if np.any(np.abs(np.abs(shift_xy) - lengths / 2) <= reference.wrapping_tolerance_A):
        raise ValueError("Ambiguous half-box translation consensus")
    shift = np.r_[shift_xy, np.median(displacement[:, 2])]
    residual = displacement - shift
    residual[:, :2] = minimum_image_vectors(residual[:, :2], lengths)
    maximum = float(np.max(np.linalg.norm(residual, axis=1)))
    if maximum > reference.deformation_tolerance_A + 1e-12:
        raise ValueError("Scaffold deformation exceeds frozen tolerance; translation registration rejected")
    return ScaffoldRegistration(tuple(float(value) for value in shift), maximum, reference.sha256)


def _registered_shift(reference: ScaffoldReference, registration: ScaffoldRegistration) -> np.ndarray:
    if registration.reference_hash != reference.sha256:
        raise ValueError("Registration belongs to a different observer reference")
    return np.asarray(registration.translation_A)


def register_coordinates(
    coordinates_A: Sequence[Sequence[float]], reference: ScaffoldReference,
    registration: ScaffoldRegistration,
) -> np.ndarray:
    """Map frame coordinates to reference: subtract XYZ shift, then wrap XY only."""
    coordinates = np.asarray(coordinates_A, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError("Finite N-by-3 coordinates are required")
    shift = _registered_shift(reference, registration)
    return _wrap_xy(coordinates - shift, np.asarray(reference.bounds_A))


def reference_to_frame_sites(
    site_xy_A: Sequence[Sequence[float]], reference: ScaffoldReference,
    registration: ScaffoldRegistration,
) -> np.ndarray:
    """Map fixed reference-frame footprint centers to frame: add shift and wrap XY."""
    sites = np.asarray(site_xy_A, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 2 or not np.isfinite(sites).all():
        raise ValueError("Finite N-by-2 fixed site coordinates are required")
    shift = _registered_shift(reference, registration)
    return _wrap_xy(sites + shift[:2], np.asarray(reference.bounds_A))
