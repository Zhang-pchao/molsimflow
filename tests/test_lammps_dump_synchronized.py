from __future__ import annotations

from pathlib import Path

import pytest

from molsimflow.io.lammps_dump import iter_synchronized_lammps_dump_records


def _write_dump(path: Path, atom_id: int, steps: tuple[int, ...]) -> None:
    frames = []
    for step in steps:
        frames.append(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type x y z vx vy vz\n"
            f"{atom_id} 1 {atom_id}.0 0 0 0 0 0\n"
        )
    path.write_text("".join(frames), encoding="utf-8")


def test_synchronized_dumps_downsample_faster_stream(tmp_path: Path) -> None:
    fast = tmp_path / "fast.dump"
    slow = tmp_path / "slow.dump"
    _write_dump(fast, 2, (0, 10, 20))
    _write_dump(slow, 1, (0, 20))
    frames = list(iter_synchronized_lammps_dump_records(
        [fast, slow], accept_timestep=lambda step: step % 20 == 0
    ))
    assert [frame.timestep for frame in frames] == [0, 20]
    assert [[int(row[0]) for row in frame.atom_rows] for frame in frames] == [[1, 2], [1, 2]]


def test_synchronized_dump_accepts_one_complete_stream(tmp_path: Path) -> None:
    complete = tmp_path / "complete.dump"
    _write_dump(complete, 1, (0, 10, 20))
    frames = list(iter_synchronized_lammps_dump_records(
        [complete], accept_timestep=lambda step: step % 20 == 0
    ))
    assert [frame.timestep for frame in frames] == [0, 20]
    assert [[int(row[0]) for row in frame.atom_rows] for frame in frames] == [[1], [1]]


def test_synchronized_dumps_reject_overlapping_atom_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.dump"
    second = tmp_path / "second.dump"
    _write_dump(first, 1, (0,))
    _write_dump(second, 1, (0,))
    with pytest.raises(ValueError, match="overlap"):
        list(iter_synchronized_lammps_dump_records([first, second]))


def test_selection_preserves_native_indices_and_default_reader(tmp_path):
    from molsimflow.io.lammps_dump import (
        iter_lammps_dump_records, iter_lammps_dump_records_selected,
    )
    path = tmp_path / "frames.dump"
    _write_dump(path, 1, (0, 10, 20))
    complete = list(iter_lammps_dump_records(path))
    selected = list(iter_lammps_dump_records_selected(path))
    assert [(f.frame_index, f.timestep, f.atom_rows) for f in selected] == [
        (f.frame_index, f.timestep, f.atom_rows) for f in complete]
    sparse = list(iter_lammps_dump_records_selected(path, accept_timestep=lambda t: t % 20 == 0))
    assert [(f.frame_index, f.timestep) for f in sparse] == [(0, 0), (2, 20)]


def test_rejected_truncated_frame_is_not_silently_accepted(tmp_path):
    from molsimflow.io.lammps_dump import iter_lammps_dump_records_selected
    path = tmp_path / "frames.dump"
    _write_dump(path, 1, (0, 10))
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="ended inside"):
        list(iter_lammps_dump_records_selected(path, accept_timestep=lambda t: t == 0))


def test_selected_reader_preserves_safe_endpoint_stopping(tmp_path):
    from molsimflow.io.lammps_dump import (
        iter_lammps_dump_records_selected, iter_lammps_dump_records_until,
    )
    path = tmp_path / "frames.dump"
    _write_dump(path, 1, (0,))
    path.write_text(path.read_text() + "ITEM: TIMESTEP\n10\nBROKEN TAIL\n")
    selected = list(iter_lammps_dump_records_selected(path, stop_after_timestep=0))
    bounded = list(iter_lammps_dump_records_until(path, maximum_timestep=0))
    assert [f.timestep for f in selected] == [f.timestep for f in bounded] == [0]


@pytest.mark.parametrize("change", ["counts", "box", "boundary"])
def test_synchronized_streams_reject_different_physical_frames(tmp_path, change):
    first, second = tmp_path / "first.dump", tmp_path / "second.dump"
    _write_dump(first, 1, (0, 10))
    _write_dump(second, 2, (0,) if change == "counts" else (0, 10))
    if change == "box":
        second.write_text(second.read_text().replace("0 10", "0 10.00001"))
    if change == "boundary":
        second.write_text(second.read_text().replace("pp pp ff", "pp pp pp"))
    with pytest.raises(ValueError):
        list(iter_synchronized_lammps_dump_records([first, second]))
