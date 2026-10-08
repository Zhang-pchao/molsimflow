from pathlib import Path

import numpy as np
import pytest

from molsimflow.postprocess.pimd_path_io import iter_pimd_path_frames


def _write(path: Path, steps=(0, 4), rows=None, bounds="0 10", boundary="pp pp pp"):
    if rows is None:
        rows = ["2 1 1 0 0 0 0 0", "1 2 0 0 0 0 0 0"]
    text = ""
    for step in steps:
        text += (
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n"
            f"ITEM: BOX BOUNDS {boundary}\n{bounds}\n{bounds}\n{bounds}\n"
            "ITEM: ATOMS id type x y z ix iy iz\n" + "\n".join(rows) + "\n"
        )
    path.write_text(text)


def _paths(tmp_path):
    paths = {7: tmp_path / "b7.dump", 3: tmp_path / "b3.dump"}
    for p in paths.values():
        _write(p)
    return paths


def _read(paths, **kwargs):
    return list(iter_pimd_path_frames(
        paths, bead_order=[3, 7], expected_identity={1: 2, 2: 1}, **kwargs
    ))


def test_assemble_by_atom_tag_and_declared_bead_order(tmp_path):
    paths = _paths(tmp_path)
    _write(paths[3], rows=["1 2 0 0 0 0 0 0", "2 1 2 0 0 0 0 0"])
    frames = _read(paths, selected_steps=[4])
    assert len(frames) == 1 and frames[0].step == 4
    assert frames[0].bead_ids == (3, 7)
    np.testing.assert_array_equal(frames[0].atom_ids, [1, 2])
    np.testing.assert_array_equal(frames[0].atom_types, [2, 1])
    np.testing.assert_array_equal(frames[0].positions[:, 1, 0], [2, 1])


@pytest.mark.parametrize("steps,message", [
    ((0,), "frame counts"), ((0, 8), "timesteps"), ((0, 0), "strictly increasing"),
])
def test_reject_misaligned_or_duplicate_frames(tmp_path, steps, message):
    paths = _paths(tmp_path)
    if steps == (0, 0):
        for p in paths.values():
            _write(p, steps)
    else:
        _write(paths[3], steps)
    with pytest.raises(ValueError, match=message):
        _read(paths)


@pytest.mark.parametrize("rows", [
    ["1 2 0 0 0 0 0 0", "1 2 1 0 0 0 0 0"],
    ["1 1 0 0 0 0 0 0", "2 1 1 0 0 0 0 0"],
    ["1 2 0 0 0 0 0 0"],
    ["1 2 nan 0 0 0 0 0", "2 1 1 0 0 0 0 0"],
    ["1.5 2 0 0 0 0 0 0", "2 1 1 0 0 0 0 0"],
    ["1 2 0 0 0 0.5 0 0", "2 1 1 0 0 0 0 0"],
])
def test_reject_invalid_identity_coordinates_or_images(tmp_path, rows):
    paths = _paths(tmp_path)
    _write(paths[3], rows=rows)
    with pytest.raises(ValueError):
        _read(paths)


@pytest.mark.parametrize("bounds,boundary", [
    ("0 11", "pp pp pp"), ("0 10", "pp pp ff"),
    ("0 10 0.1", "xy xz yz pp pp pp"), ("0 nan", "pp pp pp"),
])
def test_reject_inconsistent_or_unsupported_boxes(tmp_path, bounds, boundary):
    paths = _paths(tmp_path)
    _write(paths[3], bounds=bounds, boundary=boundary)
    with pytest.raises(ValueError):
        _read(paths)


def test_selected_prefix_does_not_hide_incomplete_tail(tmp_path):
    paths = _paths(tmp_path)
    with paths[3].open("a") as f:
        f.write("ITEM: TIMESTEP\n8\nITEM: NUMBER OF ATOMS\n2\n")
    with pytest.raises(ValueError):
        _read(paths, selected_steps=[0])


def test_missing_bead_duplicate_path_missing_step_and_empty_input(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="complete declared"):
        _read({3: paths[3]})
    with pytest.raises(ValueError, match="distinct"):
        _read({3: paths[3], 7: paths[3]})
    with pytest.raises(ValueError, match="missing"):
        _read(paths, selected_steps=[12])
    for p in paths.values():
        p.write_text("")
    with pytest.raises(ValueError, match="no complete"):
        _read(paths)


def test_complete_frames_do_not_make_beads_independent_samples(tmp_path):
    from molsimflow.postprocess.pimd_fes import quantum_histogram_masses

    frames = _read(_paths(tmp_path))
    assert len(frames) == 2
    # Duplicating a bead does not change a frame's probability contribution.
    values = np.array([[0.2], [0.8]])
    weights = np.log([0.75, 0.25])
    one = quantum_histogram_masses(values, weights, np.array([0.0, 0.5, 1.0]))
    two = quantum_histogram_masses(
        np.repeat(values, 2, axis=1), weights, np.array([0.0, 0.5, 1.0])
    )
    np.testing.assert_allclose(one["direct"], [0.75, 0.25])
    np.testing.assert_allclose(one["direct"], two["direct"])
