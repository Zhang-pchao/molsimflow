"""Strict declared inputs for offline quantum-path engineering workflows.

Schema labels and supplied weighting references are declarations, not evidence
of scientific admission. Relative inputs resolve against the contract directory.
No time, length, bias-energy or weighting convention is inferred or converted.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from molsimflow.io.hashes import _sha256
from molsimflow.postprocess.pimd_fes import BIAS_MODES


CONDITIONING_FIELDS = {
    "q_centroid", "distance_centroid", "logdistance_centroid",
    "oo_coordination_centroid_mean",
}
REGION_FIELDS = {"q", "distance", "logdistance"}


def _keys(value: object, name: str, required: set, optional: set = frozenset()) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return value


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonempty_list(value: object, name: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    return value


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _physical_file(path: Path) -> Path:
    # Check before resolving so both file and directory symlinks are rejected.
    absolute = path.absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError(f"symlink input paths are not allowed: {path}")
    try:
        resolved = absolute.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"input must be a regular file: {path}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"unavailable input file: {path}") from exc
    return resolved


def _identity(path: Path) -> tuple:
    info = path.stat()
    return info.st_dev, info.st_ino


def _validate_analysis(analysis: object, frame_count: int) -> None:
    item = _keys(
        analysis, "analysis",
        {"block_frames", "conditioning_fields", "bin_edges", "regions"},
    )
    block = _integer(item["block_frames"], "analysis.block_frames", 1)
    if frame_count % block or frame_count // block < 2:
        raise ValueError("analysis requires equal complete blocks and at least two blocks")
    fields = _nonempty_list(item["conditioning_fields"], "analysis.conditioning_fields")
    for field in fields:
        _string(field, "conditioning field")
        if field not in CONDITIONING_FIELDS:
            raise ValueError(f"unknown conditioning field: {field}")
    if len(set(fields)) != len(fields):
        raise ValueError("conditioning fields must be unique")
    edges = _nonempty_list(item["bin_edges"], "analysis.bin_edges")
    if len(edges) != len(fields):
        raise ValueError("bin_edges must be parallel to conditioning_fields")
    for vector in edges:
        _nonempty_list(vector, "bin edge vector")
        numbers = [_number(v, "bin edge") for v in vector]
        if len(numbers) < 2 or any(a >= b for a, b in zip(numbers, numbers[1:])):
            raise ValueError("bin edges must be finite and strictly increasing")
    names = set()
    for region in _nonempty_list(item["regions"], "analysis.regions"):
        _keys(region, "region", {"name", "bounds"})
        name = _string(region["name"], "region.name")
        if name in names:
            raise ValueError("region names must be unique")
        names.add(name)
        bounds = _keys(region["bounds"], "region.bounds", set(), REGION_FIELDS)
        if not bounds:
            raise ValueError("region bounds must be nonempty")
        for key, vector in bounds.items():
            if not isinstance(vector, list) or len(vector) != 2:
                raise ValueError(f"region bound {key} must contain [lower, upper]")
            low, high = [_number(v, f"region bound {key}") for v in vector]
            if low >= high:
                raise ValueError("region bounds must be finite and increasing")


def load_contract(path: Path) -> dict:
    """Validate a schema-v1 contract and verify every declared input hash.

    The bead list order is the declared bead topology, without ID sorting.
    Returned input paths are absolute Paths. _input_hashes also includes the
    contract; verify_inputs_unchanged rechecks identities and hashes after use.
    Precomputed CSV content/alignment is validated by the workflow reader, not
    this schema loader. Even two complete analysis blocks establish only an
    engineering minimum, never statistical independence or scientific adequacy.
    """
    contract_path = _physical_file(Path(path))
    contract_hash = _sha256(contract_path)
    try:
        data = json.loads(
            contract_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON contract: {contract_path}") from exc
    _keys(
        data, "contract",
        {"schema_version", "run", "atom_identity", "beads", "steps", "geometry", "weights"},
        {"analysis"},
    )
    if _integer(data["schema_version"], "schema_version", 1) != 1:
        raise ValueError("unsupported schema_version")
    run = _keys(
        data["run"], "run",
        {"run_id", "seed_id", "initial_path_id", "parent_restart_id", "bias_mode", "data_role"},
    )
    for key in ("run_id", "seed_id", "initial_path_id"):
        _string(run[key], f"run.{key}")
    if run["parent_restart_id"] is not None:
        _string(run["parent_restart_id"], "run.parent_restart_id")
    if _string(run["bias_mode"], "run.bias_mode") not in BIAS_MODES | {"classical"}:
        raise ValueError("unsupported run.bias_mode")
    if _string(run["data_role"], "run.data_role") not in {"engineering", "discovery", "validation"}:
        raise ValueError("unsupported run.data_role")
    atom_ids = set()
    for atom in _nonempty_list(data["atom_identity"], "atom_identity"):
        _keys(atom, "atom_identity entry", {"id", "type"})
        tag = _integer(atom["id"], "atom id", 1)
        _integer(atom["type"], "atom type", 1)
        if tag in atom_ids:
            raise ValueError("atom IDs must be unique")
        atom_ids.add(tag)
    steps = _keys(data["steps"], "steps", {"first", "last", "stride", "timestep_fs"})
    first = _integer(steps["first"], "steps.first", 0)
    last = _integer(steps["last"], "steps.last", first)
    stride = _integer(steps["stride"], "steps.stride", 1)
    steps["timestep_fs"] = _number(steps["timestep_fs"], "steps.timestep_fs", positive=True)
    if (last - first) % stride:
        raise ValueError("selected steps must include last exactly at the declared stride")
    geometry = _keys(
        data["geometry"], "geometry",
        {"center_type", "assigned_type", "kappa", "distance_kappa", "reference",
         "environment_r0", "length_unit"},
    )
    for key in ("center_type", "assigned_type"):
        _integer(geometry[key], f"geometry.{key}", 1)
    if geometry["center_type"] == geometry["assigned_type"]:
        raise ValueError("center_type and assigned_type must be distinct")
    for key in ("kappa", "distance_kappa", "environment_r0"):
        geometry[key] = _number(geometry[key], f"geometry.{key}", positive=True)
    geometry["reference"] = _number(geometry["reference"], "geometry.reference")
    _string(geometry["length_unit"], "geometry.length_unit")
    if "analysis" in data:
        _validate_analysis(data["analysis"], (last - first) // stride + 1)

    hashes = {str(contract_path): contract_hash}
    identities = {str(contract_path): _identity(contract_path)}
    seen_files = set(identities.values())

    def register(entry: dict, name: str) -> None:
        source = Path(_string(entry["path"], f"{name}.path"))
        if not source.is_absolute():
            source = contract_path.parent / source
        source = _physical_file(source)
        identity = _identity(source)
        if identity in seen_files:
            raise ValueError(f"duplicate input file or hardlink alias: {source}")
        digest = _string(entry["sha256"], f"{name}.sha256")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"invalid SHA256 digest: {name}")
        digest = digest.lower()
        if _sha256(source) != digest:
            raise ValueError(f"SHA256 mismatch: {source}")
        entry["path"], entry["sha256"] = source, digest
        hashes[str(source)] = digest
        identities[str(source)] = identity
        seen_files.add(identity)

    beads = _nonempty_list(data["beads"], "beads")
    if run["bias_mode"] == "classical" and len(beads) != 1:
        raise ValueError("classical mode requires exactly one bead file")
    bead_ids = set()
    for bead in beads:
        _keys(bead, "bead", {"bead_id", "path", "sha256"})
        bead_id = _integer(bead["bead_id"], "bead_id", 0)
        if bead_id in bead_ids:
            raise ValueError("bead IDs must be unique")
        bead_ids.add(bead_id)
        register(bead, "bead")
    weights = _keys(
        data["weights"], "weights", {"kind"},
        {"path", "sha256", "target_id", "admission_reference"},
    )
    if weights["kind"] == "uniform_sampler":
        _keys(weights, "uniform_sampler weights", {"kind"})
    elif weights["kind"] == "precomputed":
        _keys(weights, "precomputed weights", {
            "kind", "path", "sha256", "target_id", "admission_reference",
        })
        _string(weights["target_id"], "weights.target_id")
        _string(weights["admission_reference"], "weights.admission_reference")
        register(weights, "weights")
    else:
        raise ValueError("unsupported weights.kind")
    data["_contract_path"] = contract_path
    data["_input_hashes"] = hashes
    data["_input_identities"] = identities
    verify_inputs_unchanged(data)
    return data


def verify_inputs_unchanged(contract: dict) -> None:
    """Fail if a consumed file disappeared, was aliased/replaced, or changed."""
    for name, digest in contract["_input_hashes"].items():
        path = _physical_file(Path(name))
        if _identity(path) != tuple(contract["_input_identities"][name]):
            raise ValueError(f"input identity changed: {path}")
        if _sha256(path) != digest:
            raise ValueError(f"input SHA256 changed: {path}")
