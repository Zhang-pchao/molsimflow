"""Bounded snapshots of growing LAMMPS trajectories.

Only EOF truncation at the tail is tolerated. Interior format corruption and
non-truncation decompression errors fail closed.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
from typing import TextIO


def copy_prefix(source: Path, destination: Path) -> dict:
    source, destination = Path(source), Path(destination)
    before = source.stat()
    h = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        remaining = before.st_size
        while remaining:
            block = reader.read(min(1048576, remaining))
            if not block:
                raise ValueError("Source shrank while copying its fixed byte prefix")
            writer.write(block)
            h.update(block)
            remaining -= len(block)
    after = source.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError("Trajectory was replaced during prefix capture")
    return {
        "source": str(source), "source_prefix_bytes": before.st_size,
        "source_prefix_sha256": h.hexdigest(),
        "source_size_after": after.st_size,
        "source_mtime_ns_before": before.st_mtime_ns,
        "source_mtime_ns_after": after.st_mtime_ns,
        "source_grew_during_copy": after.st_size != before.st_size,
    }


def last_complete_stream(handle: TextIO, expected_atoms: int) -> tuple[str, dict]:
    last = None
    count = 0
    tail = False
    previous_step = -1
    while True:
        header = handle.readline()
        if not header:
            break
        if not header.endswith("\n"):
            tail = True
            break
        if header.strip() != "ITEM: TIMESTEP":
            raise ValueError("Interior corruption: expected timestep header")
        lines = [header]

        def take():
            nonlocal tail
            line = handle.readline()
            if not line or not line.endswith("\n"):
                tail = True
                return None
            lines.append(line)
            return line

        step_line = take()
        if step_line is None:
            break
        step = int(step_line)
        marker = take()
        if marker is None:
            break
        if marker.strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("Missing atom count marker")
        atom_line = take()
        if atom_line is None:
            break
        atoms = int(atom_line)
        if atoms != expected_atoms:
            raise ValueError("Unexpected frame atom count")
        marker = take()
        if marker is None:
            break
        if not marker.startswith("ITEM: BOX BOUNDS") or any(
                x in marker.split() for x in ("xy", "xz", "yz")):
            raise ValueError("Only orthorhombic boxes are supported")
        complete = True
        for _ in range(3):
            line = take()
            if line is None:
                complete = False
                break
            if len(line.split()) != 2:
                raise ValueError("Invalid orthorhombic box row")
        if not complete:
            break
        marker = take()
        if marker is None:
            break
        if not marker.startswith("ITEM: ATOMS"):
            raise ValueError("Missing atom field marker")
        fields = marker.split()[2:]
        if not {"id", "type", "x", "y", "z"} <= set(fields):
            raise ValueError("Required identity/coordinate fields are missing")
        for _ in range(atoms):
            line = take()
            if line is None:
                complete = False
                break
            if len(line.split()) != len(fields):
                raise ValueError("Interior corruption: wrong atom column count")
        if not complete:
            break
        if step <= previous_step:
            raise ValueError("Non-increasing frame timesteps")
        last = "".join(lines)
        previous_step = step
        count += 1
    if last is None:
        raise ValueError("No complete frame in captured trajectory prefix")
    return last, {
        "complete_frames_seen": count, "last_frame_index": count-1,
        "last_frame_timestep": previous_step, "last_frame_atoms": expected_atoms,
        "incomplete_text_tail_ignored": tail,
    }


def extract_prefix_snapshot(prefix: Path, output: Path, expected_atoms: int) -> dict:
    prefix, output = Path(prefix), Path(output)
    if prefix.suffix == ".zst":
        with subprocess.Popen(["zstd", "-q", "-dc", "--", str(prefix)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8") as process:
            try:
                frame, metadata = last_complete_stream(process.stdout, expected_atoms)
            except BaseException:
                process.stdout.close()
                process.wait()
                raise
            process.stdout.close()
            error = process.stderr.read()
            code = process.wait()
        truncation = code != 0 and "premature end" in error.lower()
        if code != 0 and not truncation:
            raise ValueError("Non-tail zstd decode failure: "+error.strip())
        metadata.update({"decoder_exit_code": code, "decoder_stderr": error.strip(),
                         "incomplete_compressed_tail_ignored": truncation})
    else:
        with prefix.open(encoding="utf-8") as handle:
            frame, metadata = last_complete_stream(handle, expected_atoms)
        metadata.update({"decoder_exit_code": 0,
                         "incomplete_compressed_tail_ignored": False})
    temporary = output.with_name(output.name+".tmp")
    temporary.write_text(frame)
    os.replace(temporary, output)
    metadata["snapshot_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata["snapshot_bytes"] = output.stat().st_size
    return metadata
