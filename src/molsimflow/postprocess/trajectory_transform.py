"""Streaming preparation and reference-layer alignment for LAMMPS dumps."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    iter_lammps_dump_records,
    write_lammps_dump_frame,
)


class _ZUnwrapper:
    def __init__(self) -> None:
        self._wrapped: dict[int, float] | None = None
        self._unwrapped: dict[int, float] | None = None

    def update(self, frame: LammpsDumpFrame) -> np.ndarray:
        ids, wrapped = _ids_and_z(frame)
        length = float(frame.bounds[2, 1] - frame.bounds[2, 0])
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"Invalid Z box length at timestep {frame.timestep}: {length}")
        current = dict(zip(ids, wrapped))
        if self._wrapped is None or self._unwrapped is None:
            unwrapped = current.copy()
        else:
            if set(current) != set(self._wrapped):
                raise ValueError(f"Atom ID set changed at timestep {frame.timestep}")
            unwrapped = {}
            for atom_id, value in current.items():
                delta = value - self._wrapped[atom_id]
                delta -= length * math.floor(delta / length + 0.5)
                unwrapped[atom_id] = self._unwrapped[atom_id] + delta
        self._wrapped = current
        self._unwrapped = unwrapped
        return np.asarray([unwrapped[atom_id] for atom_id in ids], dtype=float)


def _column(frame: LammpsDumpFrame, names: Sequence[str]) -> int:
    for name in names:
        if name in frame.atom_fields:
            return frame.atom_fields.index(name)
    raise ValueError(f"LAMMPS dump is missing required column: {'/'.join(names)}")


def _ids_and_z(frame: LammpsDumpFrame) -> tuple[list[int], np.ndarray]:
    id_index = _column(frame, ("id",))
    z_index = _column(frame, ("z", "zu"))
    ids = [int(row[id_index]) for row in frame.atom_rows]
    values = np.asarray([float(row[z_index]) for row in frame.atom_rows], dtype=float)
    if len(set(ids)) != frame.atom_count or any(atom_id < 1 for atom_id in ids):
        raise ValueError(f"Atom IDs must be unique positive integers at timestep {frame.timestep}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Z coordinates must be finite at timestep {frame.timestep}")
    return ids, values


def _replace_z(frame: LammpsDumpFrame, values: Sequence[float]) -> list[list[str]]:
    z_index = _column(frame, ("z", "zu"))
    if len(values) != frame.atom_count:
        raise ValueError("Replacement Z values must preserve the atom count")
    rows = [list(row) for row in frame.atom_rows]
    for row, value in zip(rows, values):
        row[z_index] = f"{float(value):.10f}"
    return rows


def _combined_frames(
    paths: Sequence[Path], drop_first_each_input: bool
) -> Iterator[tuple[int, Path, int, LammpsDumpFrame]]:
    combined_index = 0
    for path in paths:
        for source_index, frame in enumerate(iter_lammps_dump_records(path)):
            if drop_first_each_input and source_index == 0:
                continue
            yield combined_index, path, source_index, frame
            combined_index += 1


def _selected(index: int, start: int, stop: int | None, stride: int) -> bool:
    return index >= start and (stop is None or index < stop) and (index - start) % stride == 0


def _validate_paths(input_paths: Sequence[Path], output_path: Path) -> tuple[list[Path], Path]:
    inputs = [Path(path).expanduser().resolve() for path in input_paths]
    output = Path(output_path).expanduser().resolve()
    if not inputs:
        raise ValueError("At least one input trajectory is required")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path == output:
            raise ValueError("Output trajectory must not overwrite an input trajectory")
    return inputs, output


def prepare_trajectory(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    frame_start: int = 0,
    frame_stop: int | None = None,
    stride: int = 1,
    drop_first_each_input: bool = False,
    unwrap_z: bool = False,
    shift_min_z_A: float | None = None,
    require_within_box: bool = False,
) -> Mapping[str, object]:
    """Concatenate, select, optionally unwrap, and globally shift dump frames."""

    inputs, output = _validate_paths(input_paths, output_path)
    if frame_start < 0 or stride < 1 or (frame_stop is not None and frame_stop <= frame_start):
        raise ValueError("Require start >= 0, stride >= 1, and stop > start when provided")
    if shift_min_z_A is not None and not math.isfinite(shift_min_z_A):
        raise ValueError("shift_min_z_A must be finite")

    unwrapper = _ZUnwrapper()
    selected_min = math.inf
    selected_count = 0
    combined_count = 0
    for index, _, _, frame in _combined_frames(inputs, drop_first_each_input):
        values = unwrapper.update(frame) if unwrap_z else _ids_and_z(frame)[1]
        if _selected(index, frame_start, frame_stop, stride):
            selected_min = min(selected_min, float(np.min(values)))
            selected_count += 1
        combined_count += 1
    if selected_count == 0:
        raise ValueError("Frame selection produced no output frames")
    shift = 0.0 if shift_min_z_A is None else float(shift_min_z_A) - selected_min

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    output_min = math.inf
    output_max = -math.inf
    written = 0
    unwrapper = _ZUnwrapper()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, _, _, frame in _combined_frames(inputs, drop_first_each_input):
                values = unwrapper.update(frame) if unwrap_z else _ids_and_z(frame)[1]
                if not _selected(index, frame_start, frame_stop, stride):
                    continue
                shifted = values + shift
                if require_within_box and (
                    np.min(shifted) < frame.bounds[2, 0] or np.max(shifted) > frame.bounds[2, 1]
                ):
                    raise ValueError(
                        f"Transformed coordinates leave the Z box at timestep {frame.timestep}"
                    )
                write_lammps_dump_frame(handle, frame, _replace_z(frame, shifted))
                output_min = min(output_min, float(np.min(shifted)))
                output_max = max(output_max, float(np.max(shifted)))
                written += 1
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    if written != selected_count:
        raise RuntimeError("Input trajectories changed while they were being processed")

    metadata_path = output.with_name(output.name + ".json")
    metadata = {
        "workflow": "prepare_trajectory",
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "metadata": str(metadata_path),
        "combined_frame_count": combined_count,
        "selected_frame_count": selected_count,
        "selection": {"start": frame_start, "stop": frame_stop, "stride": stride},
        "drop_first_each_input": drop_first_each_input,
        "unwrap_z": unwrap_z,
        "global_z_shift_A": shift,
        "output_z_range_A": [output_min, output_max],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def align_reference_layer(
    input_path: Path,
    output_path: Path,
    *,
    reference_atom_ids: Iterable[int] | None = None,
    reference_atom_type: int | None = None,
    layer_edge: str = "lowest",
    layer_tolerance_A: float = 1.0,
    expected_atom_count: int | None = None,
    reference_z_A: float | None = None,
    require_within_box: bool = False,
) -> Mapping[str, object]:
    """Translate each frame to keep a first-frame reference layer at fixed Z."""

    inputs, output = _validate_paths([input_path], output_path)
    if layer_edge not in {"lowest", "highest"}:
        raise ValueError("layer_edge must be 'lowest' or 'highest'")
    if not math.isfinite(layer_tolerance_A) or layer_tolerance_A < 0.0:
        raise ValueError("layer_tolerance_A must be finite and non-negative")
    if reference_z_A is not None and not math.isfinite(reference_z_A):
        raise ValueError("reference_z_A must be finite")
    requested_ids = {int(value) for value in reference_atom_ids or ()}
    if any(value < 1 for value in requested_ids):
        raise ValueError("reference_atom_ids must be positive LAMMPS atom IDs")
    if expected_atom_count is not None and expected_atom_count < 1:
        raise ValueError("expected_atom_count must be positive")
    if not requested_ids and reference_atom_type is None:
        raise ValueError("Provide reference_atom_ids or reference_atom_type")

    selected_ids: set[int] = set()
    target_z = reference_z_A
    records = iter_lammps_dump_records(inputs[0])
    first = next(records, None)
    if first is None:
        raise ValueError("Input trajectory contains no frames")
    frames = chain((first,), records)

    temporary = output.with_name(output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for frame in frames:
                ids, z_values = _ids_and_z(frame)
                z_by_id = dict(zip(ids, z_values))
                if frame.frame_index == 0:
                    if requested_ids:
                        selected_ids = requested_ids
                    else:
                        type_index = _column(frame, ("type",))
                        candidates = [
                            atom_id
                            for atom_id, row in zip(ids, frame.atom_rows)
                            if int(row[type_index]) == reference_atom_type
                        ]
                        if not candidates:
                            raise ValueError(
                                f"No first-frame atoms have type={reference_atom_type}"
                            )
                        edge = min(z_by_id[value] for value in candidates)
                        if layer_edge == "highest":
                            edge = max(z_by_id[value] for value in candidates)
                        selected_ids = {
                            value
                            for value in candidates
                            if abs(z_by_id[value] - edge) <= layer_tolerance_A
                        }
                    missing = selected_ids.difference(z_by_id)
                    if missing:
                        raise ValueError(f"Reference atom IDs are absent: {sorted(missing)}")
                    if expected_atom_count is not None and len(selected_ids) != expected_atom_count:
                        raise ValueError(
                            f"Selected {len(selected_ids)} reference atoms; expected {expected_atom_count}"
                        )
                    initial_mean = float(np.mean([z_by_id[value] for value in selected_ids]))
                    if target_z is None:
                        target_z = initial_mean
                missing = selected_ids.difference(z_by_id)
                if missing:
                    raise ValueError(
                        f"Reference atom IDs are missing at timestep {frame.timestep}: {sorted(missing)}"
                    )
                if reference_atom_type is not None:
                    type_index = _column(frame, ("type",))
                    type_by_id = {
                        atom_id: int(row[type_index]) for atom_id, row in zip(ids, frame.atom_rows)
                    }
                    wrong_type = [
                        atom_id
                        for atom_id in selected_ids
                        if type_by_id[atom_id] != reference_atom_type
                    ]
                    if wrong_type:
                        raise ValueError(
                            f"Reference atoms changed type at timestep {frame.timestep}: {wrong_type}"
                        )
                assert target_z is not None
                current_mean = float(np.mean([z_by_id[value] for value in selected_ids]))
                shifted = z_values + (float(target_z) - current_mean)
                if require_within_box and (
                    np.min(shifted) < frame.bounds[2, 0] or np.max(shifted) > frame.bounds[2, 1]
                ):
                    raise ValueError(
                        f"Aligned coordinates leave the Z box at timestep {frame.timestep}"
                    )
                write_lammps_dump_frame(handle, frame, _replace_z(frame, shifted))
                frame_count += 1
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    metadata_path = output.with_name(output.name + ".json")
    metadata = {
        "workflow": "align_reference_layer",
        "input": str(inputs[0]),
        "output": str(output),
        "metadata": str(metadata_path),
        "frame_count": frame_count,
        "reference_atom_ids": sorted(selected_ids),
        "reference_atom_type": reference_atom_type,
        "layer_edge": layer_edge,
        "layer_tolerance_A": layer_tolerance_A,
        "reference_mean_z_A": target_z,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Concatenate and select dump frames")
    prepare.add_argument("--input", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--frame-start", type=int, default=0)
    prepare.add_argument("--frame-stop", type=int)
    prepare.add_argument("--stride", type=int, default=1)
    prepare.add_argument("--drop-first-each-input", action="store_true")
    prepare.add_argument("--unwrap-z", action="store_true")
    prepare.add_argument("--shift-min-z-A", type=float)
    prepare.add_argument("--require-within-box", action="store_true")

    align = subparsers.add_parser("align-layer", help="Fix a reference layer's mean Z")
    align.add_argument("--input", type=Path, required=True)
    align.add_argument("--output", type=Path, required=True)
    align.add_argument("--reference-atom-id", type=int, action="append")
    align.add_argument("--reference-atom-type", type=int)
    align.add_argument("--layer-edge", choices=("lowest", "highest"), default="lowest")
    align.add_argument("--layer-tolerance-A", type=float, default=1.0)
    align.add_argument("--expected-atom-count", type=int)
    align.add_argument("--reference-z-A", type=float)
    align.add_argument("--require-within-box", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        metadata = prepare_trajectory(
            args.input,
            args.output,
            frame_start=args.frame_start,
            frame_stop=args.frame_stop,
            stride=args.stride,
            drop_first_each_input=args.drop_first_each_input,
            unwrap_z=args.unwrap_z,
            shift_min_z_A=args.shift_min_z_A,
            require_within_box=args.require_within_box,
        )
    else:
        metadata = align_reference_layer(
            args.input,
            args.output,
            reference_atom_ids=args.reference_atom_id,
            reference_atom_type=args.reference_atom_type,
            layer_edge=args.layer_edge,
            layer_tolerance_A=args.layer_tolerance_A,
            expected_atom_count=args.expected_atom_count,
            reference_z_A=args.reference_z_A,
            require_within_box=args.require_within_box,
        )
    print(metadata["output"])
    print(metadata["metadata"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
