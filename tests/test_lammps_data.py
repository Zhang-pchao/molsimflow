from pathlib import Path

from molsimflow.io.lammps_data import convert_extxyz_to_lammps_atomic_data


def test_extxyz_to_lammps_atomic_data(tmp_path: Path):
    xyz = tmp_path / "model.xyz"
    xyz.write_text(
        "\n".join(
            [
                "2",
                'pbc="T T T" lattice="10 0 0 0 20 0 0 0 30" properties=species:S:1:pos:R:3',
                "H 1 2 3",
                "O 4 5 6",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = convert_extxyz_to_lammps_atomic_data(xyz, tmp_path / "model_atomic.data")
    text = output.read_text(encoding="utf-8")
    assert "2 atoms" in text
    assert "0.00000000  10.00000000 xlo xhi" in text
    assert "#    H" in text
