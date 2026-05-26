import csv
import math

from molsimflow.postprocess.coalescence_state import (
    CoalescenceStateConfig,
    analyze_coalescence_state,
    build_state_table,
    read_bubble_evolution,
    read_plumed_table,
    summarize_by_cv,
    summarize_states,
)


def _write_colvar(path, fields, rows):
    with path.open("w") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for row in rows:
            handle.write(" ".join(str(value) for value in row) + "\n")


def test_read_plumed_table_and_bubble_evolution(tmp_path):
    colvar = tmp_path / "COLVAR"
    _write_colvar(colvar, ["time", "d3d_all", "bridge_cyl_env.sum"], [(0, 42, 5), (1, 39, 7)])

    rows = read_plumed_table(colvar, time_unit="ps")

    assert len(rows) == 2
    assert math.isclose(rows[1]["time_ns"], 0.001)
    assert math.isclose(rows[0]["d3d_all"], 42.0)

    bubbles = tmp_path / "bubble.txt"
    bubbles.write_text(
        "\n".join(
            [
                "# frame time b1 b2 total pct1 pct2 pct_total period",
                "0 0.000 100 98 198 0 0 0 0",
                "1 0.001 190 3 193 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    bubble_rows = read_bubble_evolution(bubbles)
    assert len(bubble_rows) == 2
    assert bubble_rows[1]["Bubble1Size"] == 190.0


def test_build_state_table_assigns_expected_states():
    colvar_rows = [
        {"time_ns": 0.000, "d3d_all": 45.0},
        {"time_ns": 0.001, "d3d_all": 38.0},
        {"time_ns": 0.002, "d3d_all": 37.0},
    ]
    bubble_rows = [
        {"time_ns": 0.000, "Bubble1Size": 100.0, "Bubble2Size": 98.0, "TotalSize": 198.0},
        {"time_ns": 0.001, "Bubble1Size": 95.0, "Bubble2Size": 94.0, "TotalSize": 189.0},
        {"time_ns": 0.002, "Bubble1Size": 188.0, "Bubble2Size": 4.0, "TotalSize": 192.0},
    ]

    table, stats = build_state_table(
        colvar_rows,
        bubble_rows=bubble_rows,
        config=CoalescenceStateConfig(
            sample_interval_ns=0.0,
            nominal_radius_A=19.0,
            close_gap_A=0.0,
            min_persist_samples=1,
            cv_bins=3,
        ),
    )

    assert [row["state"] for row in table] == ["separated", "transition_like", "merged_like"]
    assert math.isclose(stats["surface_contact_distance_A"], 38.0)

    summary = summarize_states(table)
    assert {row["state"] for row in summary} == {"separated", "transition_like", "merged_like"}
    cv_summary = summarize_by_cv(table, bins=3)
    assert cv_summary


def test_analyze_coalescence_state_writes_outputs(tmp_path):
    colvar = tmp_path / "COLVAR"
    post = tmp_path / "COLVAR_POST"
    bubbles = tmp_path / "bubble.txt"
    _write_colvar(colvar, ["time", "d3d_all", "bridge_cyl_env.sum"], [(0, 45, 4), (1, 38, 8), (2, 37, 9)])
    _write_colvar(post, ["time", "n2A_num", "n2B_num"], [(0, 100, 98), (1, 95, 94), (2, 188, 4)])
    bubbles.write_text(
        "\n".join(
            [
                "0 0.000 100 98 198 0 0 0 0",
                "1 0.001 95 94 189 0 0 0 0",
                "2 0.002 188 4 192 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )

    outputs = analyze_coalescence_state(
        colvar=colvar,
        colvar_post=post,
        bubble_evolution=bubbles,
        output_dir=tmp_path / "out",
        config=CoalescenceStateConfig(sample_interval_ns=0.0, min_persist_samples=1, cv_bins=3),
    )

    assert outputs["state_table"].exists()
    assert outputs["state_summary"].exists()
    assert outputs["cv_summary"].exists()
    assert outputs["statistics"].exists()

    with outputs["state_table"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["state"] == "merged_like"
