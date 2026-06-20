from pathlib import Path

from molsimflow.cli import build_parser
from molsimflow.postprocess.sphere_cv_compare import (
    CaseSpec,
    choose_cv_columns,
    parse_case,
    read_case,
    summary_rows,
)


def _write_colvar(path: Path, offset: float) -> None:
    path.write_text(
        "#! FIELDS time n2_num foot_total sum_cn.sum\n"
        f"0.0 {1.0 + offset} {5.0 + offset} {10.0 + offset}\n"
        f"1.0 {2.0 + offset} {6.0 + offset} {12.0 + offset}\n",
        encoding="utf-8",
    )


def test_parse_case_and_read_direct_colvar(tmp_path):
    run_dir = tmp_path / "case_a"
    run_dir.mkdir()
    _write_colvar(run_dir / "COLVAR", 0.0)

    spec = parse_case(f"case A={run_dir}")
    case = read_case(spec, "COLVAR", skip_last_data_line=False)

    assert spec == CaseSpec(label="case A", run_dir=run_dir)
    assert case.table.row_count == 2
    assert case.relative_time.tolist() == [0.0, 1.0]
    assert case.segments[0].time_source == "colvar_time"


def test_choose_cv_columns_and_summary_rows(tmp_path):
    run_a = tmp_path / "case_a"
    run_b = tmp_path / "case_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_colvar(run_a / "COLVAR", 0.0)
    _write_colvar(run_b / "COLVAR", 1.0)
    cases = [
        read_case(CaseSpec("A", run_a), "COLVAR", False),
        read_case(CaseSpec("B", run_b), "COLVAR", False),
    ]

    columns = choose_cv_columns(cases, requested=("foot_total", "missing", "n2_num"))
    rows = summary_rows(cases, columns)

    assert columns == ["foot_total", "n2_num"]
    by_case_cv = {(row["case"], row["cv"]): row for row in rows}
    assert by_case_cv[("A", "foot_total")]["delta"] == 1.0
    assert by_case_cv[("B", "n2_num")]["last"] == 3.0


def test_top_level_parser_exposes_sphere_cv_compare(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "postprocess",
            "sphere-cv-compare",
            "--case",
            f"A={tmp_path / 'a'}",
            "--case",
            f"B={tmp_path / 'b'}",
            "--output-dir",
            str(tmp_path / "out"),
            "--cv",
            "foot_total",
        ]
    )

    assert args.postprocess_command == "sphere-cv-compare"
    assert args.case == [f"A={tmp_path / 'a'}", f"B={tmp_path / 'b'}"]
    assert args.cv == ["foot_total"]
