from pathlib import Path

from molsimflow.io.extxyz import add_pbc_lattice_to_xyz


def test_add_pbc_lattice_to_xyz_shifts_z(tmp_path: Path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "\n".join(
            [
                "test",
                "1.0",
                "10 0 0",
                "0 20 0",
                "0 0 30",
                "H O",
                "1 1",
                "Cartesian",
                "0 0 0",
                "1 1 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    xyz = tmp_path / "packed.xyz"
    xyz.write_text("2\ncomment\nH 0 0 -2\nO 1 1 3\n", encoding="utf-8")
    output = add_pbc_lattice_to_xyz(xyz, poscar, tmp_path / "model.xyz", z_min_padding=5.0)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert 'pbc="T T T"' in lines[1]
    assert lines[2].endswith("5.000000")
    assert lines[3].endswith("10.000000")
