import shutil
import subprocess
from pathlib import Path

import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.nanobubble_ion_distribution import (
    IonFrame,
    Stage,
    _selected_steps,
    build_ion_samples,
    iter_ion_frames,
    parse_stage,
    stage_names,
    top_surface_si_ids,
)


def test_stage_and_top_si_selection_are_explicit():
    stage = parse_stage("pre_attachment:0.01:1.24")
    assert stage == Stage("pre_attachment", 0.01, 1.24)
    assert stage_names([stage], 1.0) == ("pre_attachment",)
    assert stage_names([stage], 2.0) == ()

    elements = np.array(["Si", "Si", "O"])
    coordinates = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    np.testing.assert_array_equal(
        top_surface_si_ids(elements, coordinates, (1, 3), 0.5),
        [2],
    )


def test_drop_first_frame_applies_before_stage_filtering():
    records = {20: ({}, []), 40: ({}, [])}
    assert _selected_steps(records, {0, 20, 40}, True) == [20, 40]
    assert _selected_steps({0: ({}, []), **records}, {0, 20, 40}, True) == [20, 40]


def test_dump_selection_and_bubble_coordinates(tmp_path: Path):
    dump = tmp_path / "frame.dump"
    atom_rows = [
        "1 8 1 1 2",
        "2 2 2 2 2",
        "3 3 9.5 10 10",
        "4 3 9.5 10 10.2",
        "5 3 10.5 10 10",
        "6 3 10.5 10 10.2",
        "7 4 12 10 10",
        "8 5 14 10 10",
        "9 2 15 10 10",
        "10 1 15.8 10 10",
    ]
    dump.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n10\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n0 20\n"
        "ITEM: ATOMS id type x y z\n"
        + "\n".join(atom_rows)
        + "\n"
    )
    selected = next(
        iter_ion_frames(
            dump,
            (1, 2),
            (3, 6),
            (7, 10),
            hydrogen_type=1,
            oxygen_type=2,
            sodium_type=4,
            chloride_type=5,
        )
    )
    np.testing.assert_array_equal(selected.solution_oxygen_ids, [9])
    np.testing.assert_array_equal(selected.sodium_ids, [7])
    np.testing.assert_array_equal(selected.chloride_ids, [8])

    frame = IonFrame(
        dump,
        0,
        0,
        selected.bounds,
        selected.surface,
        selected.nitrogen,
        selected.oxygen_ids,
        selected.oxygen,
        selected.solution_oxygen_ids,
        selected.solution_oxygen,
        selected.hydrogen_ids,
        selected.hydrogen,
        selected.sodium_ids,
        selected.sodium,
        selected.chloride_ids,
        selected.chloride,
    )
    empty = (np.empty(0, dtype=int), np.empty((0, 3), dtype=float))
    rows = build_ion_samples(
        {
            "Na_plus": (np.array([7]), np.array([[12.0, 10.0, 19.0]])),
            "Cl_minus": empty,
            "H3O_plus_candidate": empty,
            "OH_minus_candidate": empty,
        },
        frame=frame,
        time_ns=0.5,
        stages=("pre_attachment",),
        top_si_z_A=2.0,
        terminal_plane_z_A=3.0,
        bubble_center=np.array([10.0, 10.0, 19.0]),
        bubble_R90_A=1.0,
        main_n2_centers=np.array([[9.5, 10.0, 19.0], [10.5, 10.0, 19.0]]),
    )
    assert len(rows) == 1
    assert rows[0]["stage"] == "pre_attachment"
    assert np.isclose(rows[0]["z_from_top_si_A"], 17.0)
    assert np.isclose(rows[0]["z_from_terminal_plane_A"], 16.0)
    assert np.isclose(rows[0]["r_minus_bubble_R90_A"], 1.0)
    assert np.isclose(rows[0]["nearest_main_n2_center_A"], 1.5)


def test_zst_dump_is_streamed_without_materializing_a_copy(tmp_path: Path):
    if shutil.which("zstd") is None:
        raise RuntimeError("test environment must provide zstd for .zst support")
    dump = tmp_path / "frame.dump"
    dump.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n4\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n0 20\n"
        "ITEM: ATOMS id type x y z\n"
        "1 8 1 1 2\n2 2 2 2 2\n3 3 10 10 10\n4 3 10 10 10.2\n"
    )
    subprocess.run(["zstd", "-q", "-f", str(dump)], check=True)
    compressed = tmp_path / "frame.dump.zst"
    dump.unlink()
    frames = list(
        iter_ion_frames(
            compressed,
            (1, 2),
            (3, 4),
            (1, 4),
            hydrogen_type=1,
            oxygen_type=2,
            sodium_type=4,
            chloride_type=5,
        )
    )
    assert len(frames) == 1
    selected = frames[0]
    assert selected.step == 0
    assert not dump.exists()
    assert compressed.exists()


def test_cli_exposes_nanobubble_ion_distribution(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "postprocess",
            "nanobubble-ion-distribution",
            "--trajectory",
            str(tmp_path / "bubble.dump"),
            "--output-dir",
            str(tmp_path / "out"),
            "--reference-structure",
            str(tmp_path / "model.xyz"),
            "--surface-range",
            "1:10",
            "--nitrogen-range",
            "11:20",
            "--solution-range",
            "21:30",
            "--stage",
            "late:0:1",
            "--terminal-surface-z-A",
            "5",
        ]
    )
    assert args.postprocess_command == "nanobubble-ion-distribution"
    assert args.terminal_surface_z_A == 5.0
