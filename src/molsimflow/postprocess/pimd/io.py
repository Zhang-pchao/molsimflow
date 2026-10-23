"""Strict PIMD input readers and consumed-input verification policies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from molsimflow.io.hashes import sha256_file as sha256

from .types import (
    PreparedInputs,
    validate_bias_mode,
    validate_primary_estimator,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def verify_manifest_inputs(
    manifest: Path,
    run_root: Path,
    inputs: Sequence[Path],
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
            require(
                handle.readline().startswith("ITEM: NUMBER OF ATOMS"), f"bad atom count: {path}"
            )
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


def prepare_inputs(contract: Mapping[str, object], output: Path, profile: str) -> PreparedInputs:
    from .alignment import time_scale_to_fs

    source = contract["source"]
    selection = contract["selection"]
    reweight = contract["reweight"]
    # Resolve names before mode-specific admission, preserving the input contract
    # and sampling labels verbatim in provenance and output filenames.
    sampling_mode = source.get("sampling_slug", "centroid")
    inferred_bias_mode = {
        "centroid": "centroid_coord",
        "bead_density": "bead_density_shared",
    }.get(sampling_mode, sampling_mode)
    bias_mode = validate_bias_mode(str(reweight.get("bias_mode", inferred_bias_mode)))
    if bias_mode == "contracted_bead_mean":
        from molsimflow.postprocess.path_contraction import validate_contraction_metadata

        metadata = reweight.get("path_contraction")
        validate_contraction_metadata(metadata)
        require(
            reweight.get("weight_kind") == "fixed_bias",
            "contracted bead mean currently requires stationary frozen-bias records",
        )
        require(
            bool(reweight.get("bias_column")), "contracted bead mean requires total bias energy"
        )
        require(
            reweight.get("bias_mode") is not None
            and validate_bias_mode(str(source.get("sampling_slug"))) == bias_mode
            and bool(source.get("sampling_label")),
            "explicit contracted-bead-mean labels are required",
        )
        require(bool(reweight.get("bead_cv_names")), "declare the original real-bead CV columns")
        (output / "qc" / "path-contraction.json").write_text(
            json.dumps({"status": "METADATA_VALIDATED", **metadata}, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    probability_spec = None
    probability_modes = {"bead_probability_mixture", "centroid_probability_mixture"}
    if bias_mode in probability_modes:
        probability_spec = reweight.get("probability_mixture")
        require(
            profile == "core" and len(reweight["cv_names"]) == 1,
            "probability mixtures currently require the one-dimensional core profile",
        )
        require(
            reweight.get("weight_kind") == "fixed_bias",
            "probability mixtures require stationary frozen-bias records",
        )
        require(
            isinstance(probability_spec, dict),
            "probability_mixture manifest specification is required",
        )
        require(
            bool(reweight.get("bias_column")) and not reweight.get("extra_bias_columns"),
            "probability mixtures require one complete-path total-bias column",
        )
        require(
            reweight.get("bias_mode") is not None
            and validate_bias_mode(str(source.get("sampling_slug"))) == bias_mode
            and bool(source.get("sampling_label")),
            "explicit probability-mixture labels are required",
        )
    conditional_spec = None
    if bias_mode == "centroid_conditioned":
        conditional_spec = reweight.get("conditional_path")
        require(
            profile == "core" and len(reweight["cv_names"]) == 1,
            "centroid_conditioned currently requires the one-dimensional core profile",
        )
        require(
            reweight.get("weight_kind") == "fixed_bias",
            "centroid_conditioned requires a stationary frozen-bias record",
        )
        require(
            isinstance(conditional_spec, dict) and conditional_spec.get("frozen") is True,
            "conditional_path must declare a frozen model",
        )
        require(
            bool(reweight.get("bias_column")) and not reweight.get("extra_bias_columns"),
            "centroid_conditioned requires one total-bias column without extra biases",
        )
        require(
            source.get("sampling_slug") == "centroid_conditioned"
            and bool(source.get("sampling_label")),
            "explicit conditional-path labels are required",
        )
    primary_estimator = validate_primary_estimator(
        str(reweight.get("primary_estimator", "probability_mean"))
    )
    require(
        not (
            contract.get("reference") is not None and reweight.get("weight_kind") == "precomputed"
        ),
        "legacy reference cross-check requires bias-energy weights",
    )
    if profile == "water_ionization_opes":
        require(
            bool(reweight.get("bias_column")), "water-ionization diagnostics require bias_column"
        )
    for config, key in ((source, "sampling_observable_id"), (reweight, "target_observable_id")):
        value = config.get(key)
        require(
            value is None or (isinstance(value, str) and bool(value.strip())),
            f"{key} must be a nonempty string",
        )
    sampling_time_scale_to_fs = time_scale_to_fs(
        source.get("sampling_time_scale_to_fs", 1.0),
        "sampling_time_scale_to_fs",
    )
    bead_time_scale_to_fs = time_scale_to_fs(
        source.get("bead_time_scale_to_fs", 1.0),
        "bead_time_scale_to_fs",
    )
    kernel_time_scale_to_fs = time_scale_to_fs(
        source.get("kernel_time_scale_to_fs", sampling_time_scale_to_fs),
        "kernel_time_scale_to_fs",
    )
    bead_offset_fs = float(source.get("bead_time_offset_fs", 0.0))
    require(np.isfinite(bead_offset_fs), "invalid bead_time_offset_fs")
    run_root = Path(source["run_root"])
    require(run_root.is_dir(), f"run root missing: {run_root}")
    raw_manifest = run_root / source["raw_manifest"]
    require(sha256(raw_manifest) == source["raw_manifest_sha256"], "raw manifest hash mismatch")

    sampling_colvar = source.get("sampling_colvar")
    if sampling_colvar is None:
        sampling_colvar = source["centroid_colvar"]
    centroid_path = run_root / str(sampling_colvar)
    bead_paths = [run_root / value for value in source["bead_colvars"]]
    require(bool(bead_paths), "bead_colvars must contain a complete path")
    if "expected_beads" in source:
        expected_beads = source["expected_beads"]
        require(
            isinstance(expected_beads, int)
            and not isinstance(expected_beads, bool)
            and expected_beads > 0,
            "expected_beads must be a positive integer",
        )
        require(len(bead_paths) == expected_beads, "bead input count differs from expected_beads")
    if profile == "water_ionization_opes":
        for key in ("thermo_logs", "trajectories"):
            require(len(source[key]) == len(bead_paths), f"{key} must contain one file per bead")
    # Resolve filesystem identity, including symlinks and hard links, without
    # rejecting distinct files whose bead observations happen to be identical.
    bead_stats = [path.stat() for path in bead_paths]
    require(
        len({(stat.st_dev, stat.st_ino) for stat in bead_stats}) == len(bead_paths),
        "duplicate bead input file",
    )
    consumed_paths = [centroid_path, *bead_paths]
    conditional_model = None
    if conditional_spec is not None:
        from molsimflow.postprocess.conditional_path import load_model

        model_path = run_root / conditional_spec["model_file"]
        conditional_model = load_model(model_path, expected_sha256=conditional_spec["model_sha256"])
        consumed_paths.append(model_path)
    probability_model = None
    if probability_spec is not None:
        from molsimflow.postprocess.probability_mixture import load_manifest

        model_path = run_root / probability_spec["manifest_file"]
        probability_model = load_manifest(
            model_path, expected_sha256=probability_spec["manifest_sha256"]
        )
        require(probability_model["mode"] == bias_mode, "probability mixture mode mismatch")
        require(
            probability_model["expected_beads"] == len(bead_paths),
            "probability mixture bead count mismatch",
        )
        require(probability_model["energy_unit"] == "eV", "probability mixture field must use eV")
        consumed_paths.append(model_path)
        field_keys = ["field"]
        if probability_model["mode"] == "centroid_probability_mixture":
            field_keys = (["field"] if probability_model["coupling"] > 0 else []) + [
                "centroid_field"
            ]
        for key in field_keys:
            input_path = run_root / probability_spec[key + "_file"]
            require(
                sha256(input_path) == probability_model[key + "_sha256"],
                "probability mixture frozen field hash mismatch",
            )
            consumed_paths.append(input_path)
    if profile == "water_ionization_opes":
        consumed_paths.append(run_root / source["kernels"])
        consumed_paths.extend(run_root / name for name in source["thermo_logs"])
        consumed_paths.extend(run_root / name for name in source["trajectories"])
    verified_inputs = verify_manifest_inputs(raw_manifest, run_root, consumed_paths)
    (output / "provenance" / "verified-inputs.json").write_text(
        json.dumps({"scope": "consumed inputs", "sha256": verified_inputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    fields, centroid_data = read_plumed(centroid_path)
    bead_tables = [read_plumed(path) for path in bead_paths]
    require(
        all(
            table_fields == fields[:4] or "time" in table_fields for table_fields, _ in bead_tables
        ),
        "invalid bead table headers",
    )
    return PreparedInputs(
        source=source,
        selection=selection,
        reweight=reweight,
        bias_mode=bias_mode,
        conditional_spec=conditional_spec,
        probability_spec=probability_spec,
        primary_estimator=primary_estimator,
        sampling_time_scale_to_fs=sampling_time_scale_to_fs,
        bead_time_scale_to_fs=bead_time_scale_to_fs,
        kernel_time_scale_to_fs=kernel_time_scale_to_fs,
        bead_offset_fs=bead_offset_fs,
        run_root=run_root,
        conditional_model=conditional_model,
        probability_model=probability_model,
        fields=fields,
        centroid_data=centroid_data,
        bead_tables=bead_tables,
    )
