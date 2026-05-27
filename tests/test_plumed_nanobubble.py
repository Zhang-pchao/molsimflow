from pathlib import Path

from molsimflow.plumed.nanobubble import (
    UmbrellaSamplingConfig,
    dimer_pairs_from_range,
    generate_n2_com_plumed,
    parse_type_map,
    read_structure_atoms,
    select_top_layer_atoms,
)


def test_generate_legacy_cluster_size_plumed(tmp_path: Path):
    output = tmp_path / "in.plumed"

    summary = generate_n2_com_plumed(output_file=output, start=5, stop=8)

    text = output.read_text(encoding="utf-8")
    assert summary.mode == "cluster-size"
    assert summary.dimer_count == 2
    assert "c000: COM ATOMS=5,6" in text
    assert "c001: COM ATOMS=7,8" in text
    assert "reps_center: GROUP ATOMS=c000,c001" in text
    assert "ARG=sum_cn.sum" in text
    assert "dz: DISTANCE" not in text


def test_extxyz_surface_distance_uses_top_si_layer_and_stride(tmp_path: Path):
    structure = tmp_path / "model.xyz"
    structure.write_text(
        "\n".join(
            [
                "9",
                'Lattice="10 0 0 0 10 0 0 0 20" Properties=species:S:1:pos:R:3 pbc="T T T"',
                "Si 0 0 1.0",
                "Si 1 0 2.0",
                "Si 2 0 2.0",
                "Si 3 0 2.0",
                "Si 4 0 2.0",
                "N 0 1 4.0",
                "N 0 2 4.1",
                "N 0 3 4.2",
                "N 0 4 4.3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "surface.plumed"

    summary = generate_n2_com_plumed(
        output_file=output,
        structure_file=structure,
        with_surface=True,
        surface_element="Si",
        surface_z_tolerance=0.2,
        surface_stride=2,
    )

    surface = summary.surface_selection
    assert summary.mode == "surface-distance"
    assert summary.bias_mode == "us"
    assert summary.dimer_pairs == ((6, 7), (8, 9))
    assert surface is not None
    assert surface.candidate_count == 4
    assert surface.atom_ids == (2, 4)
    text = output.read_text(encoding="utf-8")
    assert "csurf: COM ATOMS=surf NOPBC" in text
    assert "dz: DISTANCE ATOMS=csurf,cb COMPONENTS NOPBC" in text
    assert "umb: RESTRAINT ARG=dz.z AT=__Z0__ KAPPA=3" in text
    assert "surf: GROUP ATOMS=2,4" in text
    assert "OPES_METAD" not in text


def test_surface_bias_mode_opes_can_be_requested_explicitly(tmp_path: Path):
    structure = tmp_path / "model.xyz"
    structure.write_text(
        "\n".join(
            [
                "6",
                'Lattice="10 0 0 0 10 0 0 0 20" Properties=species:S:1:pos:R:3 pbc="T T T"',
                "Si 0 0 2.0",
                "Si 1 0 2.0",
                "N 0 1 4.0",
                "N 0 2 4.1",
                "N 0 3 4.2",
                "N 0 4 4.3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "surface_opes.plumed"

    summary = generate_n2_com_plumed(
        output_file=output,
        structure_file=structure,
        with_surface=True,
        bias_mode="opes",
    )

    text = output.read_text(encoding="utf-8")
    assert summary.bias_mode == "opes"
    assert "csurf: COM ATOMS=surf" in text
    assert "NOPBC" not in text
    assert "ARG=dz.z" in text
    assert "OPES_METAD" in text
    assert "umb: RESTRAINT" not in text


def test_surface_us_parameters_are_configurable(tmp_path: Path):
    structure = tmp_path / "model.xyz"
    structure.write_text(
        "\n".join(
            [
                "6",
                'Lattice="10 0 0 0 10 0 0 0 20" Properties=species:S:1:pos:R:3 pbc="T T T"',
                "Si 0 0 2.0",
                "Si 1 0 2.0",
                "N 0 1 4.0",
                "N 0 2 4.1",
                "N 0 3 4.2",
                "N 0 4 4.3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "surface_us_custom.plumed"

    summary = generate_n2_com_plumed(
        output_file=output,
        structure_file=structure,
        with_surface=True,
        us_config=UmbrellaSamplingConfig(
            window_center_z="42.5",
            restraint_kappa=5.0,
            enable_upper_wall_sum_cn=True,
            committor_n2_ul1=55.0,
        ),
    )

    text = output.read_text(encoding="utf-8")
    assert summary.bias_mode == "us"
    assert "umb: RESTRAINT ARG=dz.z AT=42.5 KAPPA=5" in text
    assert "UPPER_WALLS ARG=sum_cn.sum" in text
    assert "BASIN_UL1=55" in text


def test_lammps_atomic_data_masses_comments_and_type_map(tmp_path: Path):
    data = tmp_path / "model.atomic.data"
    data.write_text(
        "\n".join(
            [
                "test",
                "",
                "6 atoms",
                "2 atom types",
                "",
                "0 10 xlo xhi",
                "0 10 ylo yhi",
                "0 20 zlo zhi",
                "",
                "Masses",
                "",
                "1 28.085 # Si",
                "2 14.007 # N",
                "",
                "Atoms # atomic",
                "",
                "1 1 0 0 1",
                "2 1 1 0 2",
                "3 2 0 1 4",
                "4 2 0 2 4",
                "5 2 0 3 4",
                "6 2 0 4 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    atoms = read_structure_atoms(data, structure_format="lammps-data")
    surface = select_top_layer_atoms(atoms, element="Si", z_tolerance=0.0, stride=1)
    assert surface.atom_ids == (2,)

    output = tmp_path / "from_data.plumed"
    summary = generate_n2_com_plumed(
        output_file=output,
        structure_file=data,
        structure_format="lammps-data",
        type_map=parse_type_map(["1=Si", "2=N"]),
        with_surface=True,
        surface_z_tolerance=0.0,
    )
    assert summary.bias_mode == "us"
    assert summary.dimer_pairs == ((3, 4), (5, 6))
    assert "surf: GROUP ATOMS=2" in output.read_text(encoding="utf-8")


def test_range_pairing_rejects_odd_count():
    try:
        dimer_pairs_from_range(1, 3)
    except ValueError as exc:
        assert "even number" in str(exc)
    else:
        raise AssertionError("Expected odd atom count to be rejected")
