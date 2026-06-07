from pathlib import Path

import numpy as np

from molsimflow.postprocess.plumed_cv_diagnostics import (
    DiagnosticConfig,
    build_physical_checks,
    drop_duplicate_time_rows,
    geometry_validation,
    infer_cv_kind,
    parse_plumed_definitions,
    read_plumed_table,
)


def test_read_colvar_uses_first_header_and_drops_duplicate_times(tmp_path: Path):
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "\n".join(
            [
                "#! FIELDS time cgs cgs_ch3 opes.bias",
                "0.0 1.0 1.0 -2.0",
                "0.0 1.0 1.0 -2.0",
                "#! FIELDS time cgs cgs_ch3 opes.bias",
                "1.0 2.0 2.0 -1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    table = read_plumed_table(colvar)
    assert table.columns == ("time", "cgs", "cgs_ch3", "opes.bias")
    assert table.header_count == 2
    assert table.row_count == 3

    clean, dropped = drop_duplicate_time_rows(table)
    assert dropped == 1
    assert clean.row_count == 2
    np.testing.assert_allclose(clean.column("time"), [0.0, 1.0])


def test_parse_plumed_definitions_expands_atom_groups(tmp_path: Path):
    plumed = tmp_path / "in.plumed"
    plumed.write_text(
        "\n".join(
            [
                "surf_si: GROUP ATOMS=10-14:2,20",
                "water_o: GROUP ATOMS=100-102",
                "c000: COM ATOMS=1,2",
                "c001: COM ATOMS=3,4",
                "nfilm: INCYLINDER ATOM=surf_ref DATA=water_density DIRECTION=Z "
                "RADIUS={TANH R_0=21.000 D_0=0.0} LOWER=3.5 UPPER=9.5 SIGMA=0.5",
            ]
        ),
        encoding="utf-8",
    )

    definitions = parse_plumed_definitions(plumed)

    assert definitions.groups["surf_si"] == (10, 12, 14, 20)
    assert definitions.groups["water_o"] == (100, 101, 102)
    assert definitions.n2_pairs == ((1, 2), (3, 4))
    assert definitions.nfilm_radius_A == 21.0
    assert definitions.nfilm_lower_A == 3.5
    assert definitions.nfilm_upper_A == 9.5


def test_cgs_geometry_validation_counts_hard_contacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "diag"
    run_dir.mkdir()
    (run_dir / "COLVAR").write_text(
        "#! FIELDS time n2_num cgs cgs_ch3 opes.bias opes_e.bias\n"
        "0.0 1.0 0.8 0.8 -2.0 -1.0\n",
        encoding="utf-8",
    )
    (run_dir / "HILLS").write_text("#! FIELDS time cgs sigma_cgs height logweight\n0.0 0.8 0.1 1.0 0.0\n", encoding="utf-8")
    (run_dir / "in.plumed").write_text(
        "c000: COM ATOMS=1,2\n"
        "n2_centers: GROUP ATOMS=c000\n"
        "surface_terminal_atoms: GROUP ATOMS=3\n"
        "surface_ch3: GROUP ATOMS=3\n",
        encoding="utf-8",
    )
    (run_dir / "bubble_1k.lammpstrj").write_text(
        "ITEM: TIMESTEP\n"
        "0\n"
        "ITEM: NUMBER OF ATOMS\n"
        "3\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0 20\n"
        "0 20\n"
        "0 20\n"
        "ITEM: ATOMS id type x y z\n"
        "1 3 10.0 10.0 10.0\n"
        "2 3 10.0 10.0 11.0\n"
        "3 7 10.0 10.0 15.0\n",
        encoding="utf-8",
    )

    table = read_plumed_table(run_dir / "COLVAR")
    definitions = parse_plumed_definitions(run_dir / "in.plumed")
    rows = geometry_validation(
        DiagnosticConfig(run_dir=run_dir, output_dir=out_dir, cv_kind="cgs"),
        table,
        "cgs",
        definitions,
    )

    assert len(rows) == 1
    assert rows[0]["hard_cgs_total"] == 1
    assert rows[0]["hard_foot_total"] == 1
    assert rows[0]["cgs"] == 0.8


def test_physical_checks_detect_cgs_split_sum(tmp_path: Path):
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time n2_num cgs cgs_ch3 cgs_oh opes.bias opes_e.bias\n"
        "0.0 300 1.5 1.0 0.5 -2 -1\n"
        "1.0 300 2.5 2.0 0.5 -1 -1\n",
        encoding="utf-8",
    )
    table = read_plumed_table(colvar)

    assert infer_cv_kind(table, "auto") == "cgs"
    checks = build_physical_checks(table, None, "cgs", None)
    by_name = {row["check"]: row for row in checks}
    assert by_name["cgs_equals_split_sum"]["status"] == "PASS"
    assert by_name["gas_surface_contact_sampled"]["status"] == "PASS"


def test_nfilm_geometry_validation_handles_periodic_surface_patch(tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "diag"
    run_dir.mkdir()
    (run_dir / "COLVAR").write_text(
        "#! FIELDS time n2_num nfilm opes.bias opes_e.bias\n"
        "0.0 300 1.0 -2.0 -1.0\n",
        encoding="utf-8",
    )
    (run_dir / "in.plumed").write_text(
        "surf_si: GROUP ATOMS=10,11\n"
        "water_o: GROUP ATOMS=20\n"
        "c000: COM ATOMS=1,2\n"
        "nfilm: INCYLINDER ATOM=surf_ref DATA=water_density DIRECTION=Z "
        "RADIUS={TANH R_0=2.000 D_0=0.0} LOWER=1.0 UPPER=3.0 SIGMA=0.5\n",
        encoding="utf-8",
    )
    (run_dir / "bubble_1k.lammpstrj").write_text(
        "ITEM: TIMESTEP\n"
        "0\n"
        "ITEM: NUMBER OF ATOMS\n"
        "5\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0 20\n"
        "0 20\n"
        "0 20\n"
        "ITEM: ATOMS id type x y z\n"
        "1 3 5.0 5.0 5.0\n"
        "2 3 5.0 5.0 6.0\n"
        "10 8 0.5 10.0 5.0\n"
        "11 8 19.5 10.0 5.0\n"
        "20 2 0.0 10.0 7.0\n",
        encoding="utf-8",
    )

    table = read_plumed_table(run_dir / "COLVAR")
    definitions = parse_plumed_definitions(run_dir / "in.plumed")
    rows = geometry_validation(
        DiagnosticConfig(run_dir=run_dir, output_dir=out_dir, cv_kind="nfilm"),
        table,
        "nfilm",
        definitions,
    )

    assert len(rows) == 1
    assert rows[0]["hard_nfilm_count"] == 1


def test_parse_plumed_definitions_accepts_rational_incylinder_radius(tmp_path: Path):
    plumed = tmp_path / "in.plumed"
    plumed.write_text(
        "nfilm: INCYLINDER ATOM=surf_ref DATA=water_density DIRECTION=Z "
        "RADIUS={RATIONAL R_0=21.000 D_MAX=22.000} LOWER=3.786 UPPER=9.786 SIGMA=0.500\n",
        encoding="utf-8",
    )

    definitions = parse_plumed_definitions(plumed)

    assert definitions.nfilm_radius_A == 21.0
    assert definitions.nfilm_lower_A == 3.786
    assert definitions.nfilm_upper_A == 9.786


def test_read_colvar_can_skip_last_valid_data_line(tmp_path: Path):
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time nfilm\n"
        "0.0 1.0\n"
        "1.0 2.0\n",
        encoding="utf-8",
    )

    table = read_plumed_table(colvar, skip_last_data_line=True)

    assert table.row_count == 1
    np.testing.assert_allclose(table.column("time"), [0.0])
    np.testing.assert_allclose(table.column("nfilm"), [1.0])
