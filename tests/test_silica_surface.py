from pathlib import Path

from molsimflow.postprocess.silica_surface import (
    AnalysisConfig,
    CaseSpec,
    analyze_case,
    formula_atom_count,
    infer_surface_atom_count,
    read_extxyz,
)


def test_formula_atom_count_parses_common_species():
    assert formula_atom_count("H2O") == 3
    assert formula_atom_count("CH3") == 4
    assert formula_atom_count("SiO2") == 3


def test_silica_surface_infers_surface_count_and_groups(tmp_path: Path):
    xyz = tmp_path / "model.xyz"
    xyz.write_text(
        "\n".join(
            [
                "11",
                'Lattice="10 0 0 0 10 0 0 0 20" requested_counts="{\\"H2O\\": 1}"',
                "Si 0 0 0",
                "Si 0 0 10",
                "C 1 1 10",
                "H 1.0 1.0 11.0",
                "H 1.9 1.0 10.0",
                "H 1.0 1.9 10.0",
                "O 2 2 0",
                "H 2.0 2.0 1.0",
                "O 5 5 15",
                "H 5 5 16",
                "H 6 5 15",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = read_extxyz(xyz)
    assert infer_surface_atom_count(model) == 8
    result = analyze_case(CaseSpec("case", xyz), AnalysisConfig(make_plots=False))

    group_types = sorted(row["group_type"] for row in result.group_rows)
    assert group_types == ["CH3", "OH"]
    all_row = {row["side"]: row for row in result.summary_rows}["all"]
    assert all_row["ch3_count"] == 1
    assert all_row["oh_count"] == 1
