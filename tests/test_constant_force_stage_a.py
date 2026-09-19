from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from molsimflow.cli import build_parser
from molsimflow.postprocess.constant_force_stage_a import (
    _block_index,
    _extract_identity_track,
    _read_rows,
    _slope_mps,
    _spearman,
)


def _write_dump(
    path: Path, frames: list[tuple[int, list[tuple[int, int, float, float, float]]]]
) -> None:
    lines = []
    for step, atoms in frames:
        lines.extend(
            [
                "ITEM: TIMESTEP",
                str(step),
                "ITEM: NUMBER OF ATOMS",
                str(len(atoms)),
                "ITEM: BOX BOUNDS pp pp ff",
                "0 10",
                "0 10",
                "0 20",
                "ITEM: ATOMS id type x y z ix iy iz",
            ]
        )
        lines.extend(
            f"{atom_id} {atom_type} {x} {y} {z} 0 0 0" for atom_id, atom_type, x, y, z in atoms
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_identity_track_uses_xy_minimum_image_and_latest_duplicate(tmp_path: Path) -> None:
    first = tmp_path / "first.lammpstrj"
    second = tmp_path / "second.lammpstrj"
    atoms_a = [(10, 1, 9.8, 1.0, 1.0), (20, 7, 0.2, 1.0, 1.0), (30, 2, 8.8, 1.0, 1.0)]
    atoms_b = [(10, 1, 9.7, 1.0, 1.0), (20, 7, 0.3, 1.0, 1.0), (30, 2, 8.7, 1.0, 1.0)]
    atoms_c = [(10, 1, 9.6, 1.0, 1.0), (20, 7, 0.4, 1.0, 1.0), (30, 2, 8.6, 1.0, 1.0)]
    _write_dump(first, [(0, atoms_a), (1000, atoms_b)])
    _write_dump(second, [(1000, atoms_c), (2000, atoms_a)])
    config = {
        "atom_ids": {"hydrogen": 10, "carbon": 20, "framework_oxygen": 30},
        "time_origin_step": 0,
        "timestep_fs": 1.0,
        "ch_cutoff_A": 1.35,
        "oh_cutoff_A": 1.35,
        "cases": [{"branch_id": "fx", "trajectories": [str(first), str(second)]}],
    }
    manifest = []
    rows = _extract_identity_track(config, tmp_path, manifest)

    assert [row["step"] for row in rows] == [0, 1000, 2000]
    assert rows[0]["distance_CH_A"] == pytest.approx(0.4)
    assert rows[1]["distance_CH_A"] == pytest.approx(0.8)
    assert rows[1]["distance_OH_A"] == pytest.approx(1.0)
    assert rows[1]["nearest_owner"] == "carbon"
    assert len(manifest) == 2


def test_time_block_and_velocity_helpers() -> None:
    assert _block_index(0.0, 50.0, 100.0) == 0
    assert _block_index(49.999, 50.0, 100.0) == 0
    assert _block_index(50.0, 50.0, 100.0) == 1
    assert _block_index(100.0, 50.0, 100.0) == 1
    assert _slope_mps([0.0, 10.0, 20.0], [0.0, 0.1, 0.2]) == pytest.approx(1.0)


def test_spearman_and_cli_registration(tmp_path: Path) -> None:
    rho, samples = _spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert rho == pytest.approx(1.0)
    assert samples == 4
    args = build_parser().parse_args(
        [
            "postprocess",
            "constant-force-stage-a",
            "--contract",
            str(tmp_path / "contract.json"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.contract.name == "contract.json"


def test_read_rows_supports_gzip_csv(tmp_path: Path) -> None:
    path = tmp_path / "points.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("step,value\n1,2.5\n")
    assert _read_rows(path) == [{"step": "1", "value": "2.5"}]
