"""Explicit frame joins, structural lineage and crossed-group support contracts.

These helpers validate declared provenance; they cannot establish physical
independence, stationarity, chemical validity or statistical power from metadata.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SourceKey:
    source_id: str
    epoch_id: str
    step: int
    cadence_ps: float
    dt_ps: float


@dataclass(frozen=True)
class ObservationRecord:
    key: SourceKey
    time_ps: float
    origin_step: int
    origin_time_ps: float
    pbc: tuple[bool, bool, bool]
    atom_map_hash: str
    type_map_hash: str
    molecule_map_hash: str
    payload_hash: str
    stream_id: str = "core"
    chemical_mask_valid: bool = True
    missing_reason: str | None = None
    physical_frame_id: str | None = None
    geometry_valid: bool = True
    observer_reference_hash: str | None = None


@dataclass(frozen=True)
class ObservationValidation:
    canonical_indices: tuple[int, ...]
    duplicate_indices: tuple[int, ...]
    missing_indices: tuple[int, ...]


@dataclass(frozen=True)
class ParentKey:
    scaffold_family: str
    structural_parent_id: str


@dataclass(frozen=True)
class SiteKey:
    scaffold_family: str
    overlap_family: str


@dataclass(frozen=True)
class ParentRecord:
    sample_id: str
    scaffold_family: str
    structural_parent_id: str
    construction_id: str
    treatment_arm: str
    velocity_seed: str
    lineage_verified: bool


@dataclass(frozen=True)
class HoldoutMasks:
    train: np.ndarray
    test: np.ndarray
    excluded: np.ndarray


@dataclass(frozen=True)
class FoldSupport:
    status: str
    reasons: tuple[str, ...]
    counts: dict[str, dict[str, int]]

    @property
    def fit_allowed(self) -> bool:
        return self.status == "READY"


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_key(key: SourceKey) -> None:
    _nonempty(key.source_id, "source_id")
    _nonempty(key.epoch_id, "epoch_id")
    if not isinstance(key.step, Integral) or isinstance(key.step, (bool, np.bool_)):
        raise ValueError("step must be an integer")
    for name, value in (("cadence_ps", key.cadence_ps), ("dt_ps", key.dt_ps)):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    ratio = key.cadence_ps / key.dt_ps
    if not isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-7) or ratio < 1:
        raise ValueError("cadence_ps must be an integer multiple of dt_ps")


def validate_observation_contract(
    records: Sequence[ObservationRecord],
    *,
    requested_keys: Sequence[SourceKey] | None = None,
    allow_parity_duplicates: bool = False,
) -> ObservationValidation:
    """Validate joins and return first observations plus explicit missingness.

    Time is origin_time_ps + (step - origin_step) * dt_ps. The same numerical
    step in different sources/epochs is distinct. Physical aliases must be
    explicitly declared and agree in time, integration dt, PBC, atom/type/
    molecule mappings, payload and validity before deduplication. Cadence may
    differ between streams. Epochs can never be aliases across a reset.

    A missing observation is retained, never converted to absence. An invalid
    chemistry/geometry mask requires a reason. Requested keys check selection
    coverage, not only range endpoints; a missing row is a contract failure.
    This validates a collection, not continuity of an ordered trajectory.
    """
    metadata: dict[tuple[str, str], tuple] = {}
    time_offsets: dict[tuple[str, str], float] = {}
    streams: dict[tuple[str, str, str], tuple] = {}
    physical: dict[tuple, tuple[int, ObservationRecord]] = {}
    raw_frames: dict[tuple[str, str, int, str], int] = {}
    raw_physical_ids: dict[tuple[str, str, int], tuple] = {}
    seen_keys: set[SourceKey] = set()
    canonical, duplicates, missing = [], [], []
    for index, record in enumerate(records):
        key = record.key
        _validate_key(key)
        _nonempty(record.stream_id, "stream_id")
        if not isinstance(record.origin_step, Integral) or isinstance(
            record.origin_step, (bool, np.bool_)
        ):
            raise ValueError("origin_step must be an integer")
        if not isfinite(record.time_ps) or not isfinite(record.origin_time_ps):
            raise ValueError("time mapping must be finite")
        expected = record.origin_time_ps + (key.step - record.origin_step) * key.dt_ps
        if not isclose(record.time_ps, expected, rel_tol=0, abs_tol=1e-7):
            raise ValueError("step/time mapping mismatch")
        if len(record.pbc) != 3 or any(type(x) is not bool for x in record.pbc):
            raise ValueError("pbc must contain exactly three booleans")
        for name in ("atom_map_hash", "type_map_hash", "molecule_map_hash"):
            _nonempty(getattr(record, name), name)
        if type(record.chemical_mask_valid) is not bool or type(record.geometry_valid) is not bool:
            raise ValueError("validity masks must be booleans")
        if record.missing_reason is not None:
            _nonempty(record.missing_reason, "missing_reason")
            missing.append(index)
        elif not record.chemical_mask_valid or not record.geometry_valid:
            raise ValueError("invalid chemistry/geometry mask needs an explicit missing_reason")
        else:
            _nonempty(record.payload_hash, "payload_hash")
        if record.observer_reference_hash is not None:
            _nonempty(record.observer_reference_hash, "observer_reference_hash")
        source_epoch = (key.source_id, key.epoch_id)
        identity = (record.pbc, record.atom_map_hash, record.type_map_hash,
                    record.molecule_map_hash, key.dt_ps, record.observer_reference_hash)
        if source_epoch in metadata and metadata[source_epoch] != identity:
            raise ValueError("mixed PBC, identity map, observer reference or integration dt within source/epoch")
        metadata[source_epoch] = identity
        time_offset = record.origin_time_ps - record.origin_step * key.dt_ps
        if source_epoch in time_offsets and not isclose(
            time_offset, time_offsets[source_epoch], rel_tol=0, abs_tol=1e-7
        ):
            raise ValueError("inconsistent step/time mapping within source/epoch")
        time_offsets[source_epoch] = time_offset
        stream = (*source_epoch, record.stream_id)
        mapping = (key.cadence_ps, record.origin_step, record.origin_time_ps)
        if stream in streams and streams[stream] != mapping:
            raise ValueError("mixed cadence or time origin within source/epoch/stream")
        streams[stream] = mapping
        raw_key = (*source_epoch, key.step, record.stream_id)
        if raw_key in raw_frames:
            raise ValueError("duplicate observation in the same source/epoch/stream")
        raw_frames[raw_key] = index
        if record.physical_frame_id is None:
            physical_key = ("source", *source_epoch, key.step)
        else:
            _nonempty(record.physical_frame_id, "physical_frame_id")
            physical_key = ("declared", record.physical_frame_id)
        raw_physical_key = (*source_epoch, key.step)
        if (raw_physical_key in raw_physical_ids
                and raw_physical_ids[raw_physical_key] != physical_key):
            raise ValueError("inconsistent physical identity for the same source/epoch/step")
        raw_physical_ids[raw_physical_key] = physical_key
        if physical_key in physical:
            _, previous = physical[physical_key]
            previous_identity = (previous.pbc, previous.atom_map_hash, previous.type_map_hash,
                                 previous.molecule_map_hash, previous.key.dt_ps,
                                 previous.observer_reference_hash)
            parity = (identity == previous_identity
                      and key.epoch_id == previous.key.epoch_id
                      and isclose(record.time_ps, previous.time_ps, rel_tol=0, abs_tol=1e-7)
                      and record.payload_hash == previous.payload_hash
                      and record.chemical_mask_valid == previous.chemical_mask_valid
                      and record.geometry_valid == previous.geometry_valid
                      and record.missing_reason is None and previous.missing_reason is None)
            if not allow_parity_duplicates or not parity:
                raise ValueError("duplicate physical frame lacks permitted full parity")
            duplicates.append(index)
        else:
            canonical.append(index)
            physical[physical_key] = (index, record)
        seen_keys.add(key)
    if requested_keys is not None:
        for key in requested_keys:
            _validate_key(key)
        if len(set(requested_keys)) != len(requested_keys):
            raise ValueError("requested keys contain duplicates")
        absent = set(requested_keys) - seen_keys
        if absent:
            raise ValueError(f"missing requested frames: {len(absent)}")
    return ObservationValidation(tuple(canonical), tuple(duplicates), tuple(missing))


def validate_parent_lineage(
    records: Sequence[ParentRecord], *, min_independent_parents: int = 1
) -> dict[str, ParentKey]:
    """Resolve declared parents without counting arms or velocity seeds as parents.

    construction_id identifies the independent construction root, not a seed,
    arm, preparation segment or descendant relaxation. lineage_verified means
    the caller inspected supporting provenance, not a qualification inferred here.
    """
    _positive_integer(min_independent_parents, "min_independent_parents")
    samples: dict[str, ParentKey] = {}
    constructions: dict[str, ParentKey] = {}
    parent_construction: dict[ParentKey, str] = {}
    for record in records:
        for name in ("sample_id", "scaffold_family", "structural_parent_id", "construction_id",
                     "treatment_arm", "velocity_seed"):
            _nonempty(getattr(record, name), name)
        if record.lineage_verified is not True:
            raise ValueError("unverified structural construction lineage")
        if record.sample_id in samples:
            raise ValueError("duplicate sample_id")
        parent = ParentKey(record.scaffold_family, record.structural_parent_id)
        if record.construction_id in constructions and constructions[record.construction_id] != parent:
            raise ValueError("shared construction root relabeled as independent parents")
        if parent in parent_construction and parent_construction[parent] != record.construction_id:
            raise ValueError("one parent declares conflicting construction roots")
        samples[record.sample_id] = parent
        constructions[record.construction_id] = parent
        parent_construction[parent] = record.construction_id
    if len(parent_construction) < min_independent_parents:
        raise ValueError("insufficient independent structural parents; arms/seeds are not replicas")
    return samples


def _groups(values: Sequence[Hashable], name: str) -> list[Hashable]:
    result = list(values)
    for value in result:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{name} contains an empty group")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(f"{name} requires hashable group keys") from error
        if isinstance(value, (float, np.floating)) and not isfinite(value):
            raise ValueError(f"{name} contains a nonfinite group")
    return result


def grouped_holdout_masks(
    parent_keys: Sequence[Hashable], site_keys: Sequence[Hashable], *,
    held_out_parents: Sequence[Hashable], held_out_sites: Sequence[Hashable],
) -> HoldoutMasks:
    """Test the parent/site intersection and exclude both crossing strips.

    Train rows share neither a held-out parent nor an overlapping site family.
    Inputs must already collapse shared structural lineage and mapped/overlapping
    sites. Empty training/test cells are retained and fail the support gate.
    """
    parents, sites = _groups(parent_keys, "parents"), _groups(site_keys, "sites")
    if len(parents) != len(sites):
        raise ValueError("parent and site keys must have matching lengths")
    hp = set(_groups(held_out_parents, "held_out_parents"))
    hs = set(_groups(held_out_sites, "held_out_sites"))
    if not hp or not hs or not hp <= set(parents) or not hs <= set(sites):
        raise ValueError("holdouts must name nonempty observed parent and site groups")
    p = np.array([value in hp for value in parents], dtype=bool)
    s = np.array([value in hs for value in sites], dtype=bool)
    return HoldoutMasks(~p & ~s, p & s, p ^ s)


def validate_fold_support(
    labels: Sequence[str], parent_keys: Sequence[Hashable], site_keys: Sequence[Hashable],
    masks: HoldoutMasks, *, classes: Sequence[str] = ("WATER_REFILL", "N2_REPLACEMENT"),
    valid: Sequence[bool] | None = None, strata: Sequence[Hashable] | None = None,
    min_per_class: int = 1, min_parents_per_class: int = 1, min_sites_per_class: int = 1,
) -> FoldSupport:
    """Return NOT_FIT for empty/one-class/unsupported held-out strata or leakage.

    Rows must be deduplicated event representatives, not correlated frames.
    Excluded/invalid/other-class rows never count toward competing support.
    Every declared stratum appearing in test needs each class in both train and
    test. This is a necessary support gate, not a power analysis or fit routine.
    """
    for name, value in (("min_per_class", min_per_class),
                        ("min_parents_per_class", min_parents_per_class),
                        ("min_sites_per_class", min_sites_per_class)):
        _positive_integer(value, name)
    parents, sites = _groups(parent_keys, "parents"), _groups(site_keys, "sites")
    targets = list(classes)
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise ValueError("at least two distinct classes are required")
    for label in targets:
        _nonempty(label, "class label")
    y = np.asarray(labels)
    n = len(parents)
    if y.shape != (n,) or len(sites) != n:
        raise ValueError("labels and group keys must be equal-length one-dimensional sequences")
    selected = []
    for name in ("train", "test", "excluded"):
        mask = np.asarray(getattr(masks, name))
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError("fold masks must be matching boolean vectors")
        selected.append(mask)
    train, test, excluded = selected
    if not np.all(train.astype(int) + test.astype(int) + excluded.astype(int) == 1):
        raise ValueError("fold masks must partition all rows exactly once")
    observed = np.ones(n, dtype=bool) if valid is None else np.asarray(valid)
    if observed.shape != (n,) or observed.dtype != bool:
        raise ValueError("valid must be a matching boolean vector")
    groups = ["all"] * n if strata is None else _groups(strata, "strata")
    if len(groups) != n:
        raise ValueError("strata must match labels")
    reasons: list[str] = []
    if {parents[i] for i in np.flatnonzero(train)} & {parents[i] for i in np.flatnonzero(test)}:
        reasons.append("parent leakage")
    if {sites[i] for i in np.flatnonzero(train)} & {sites[i] for i in np.flatnonzero(test)}:
        reasons.append("site leakage")
    counts: dict[str, dict[str, int]] = {}
    for split, mask in (("train", train), ("test", test)):
        counts[split] = {}
        for label in targets:
            indices = np.flatnonzero(mask & observed & (y == label))
            counts[split][label] = int(len(indices))
            if len(indices) < min_per_class:
                reasons.append(f"{split}: insufficient {label} representatives")
            if len({parents[i] for i in indices}) < min_parents_per_class:
                reasons.append(f"{split}: insufficient {label} parents")
            if len({sites[i] for i in indices}) < min_sites_per_class:
                reasons.append(f"{split}: insufficient {label} sites")
    # Use all declared test rows, including missing outcomes, to define test strata.
    for stratum in sorted({groups[i] for i in np.flatnonzero(test)}, key=repr):
        membership = np.array([group == stratum for group in groups], dtype=bool)
        for split, mask in (("train", train), ("test", test)):
            for label in targets:
                if np.count_nonzero(mask & observed & membership & (y == label)) < min_per_class:
                    reasons.append(f"{split}: unsupported stratum {stratum!r} for {label}")
    return FoldSupport("NOT_FIT" if reasons else "READY", tuple(reasons), counts)
