import csv
import math

from molsimflow.plotting.table_plots import (
    build_heatmap_grid,
    group_rows,
    ordered_unique,
    output_paths,
    read_csv_rows,
)


def test_output_paths_use_suffix_or_requested_formats(tmp_path):
    assert output_paths(tmp_path / "figure.png") == [tmp_path / "figure.png"]
    assert output_paths(tmp_path / "figure", ["png", ".pdf"]) == [
        tmp_path / "figure.png",
        tmp_path / "figure.pdf",
    ]


def test_ordered_unique_and_group_rows():
    rows = [
        {"case": "A", "value": "1"},
        {"case": "B", "value": "2"},
        {"case": "A", "value": "3"},
    ]

    assert ordered_unique(row["case"] for row in rows) == ("A", "B")
    grouped = group_rows(rows, "case")
    assert [row["value"] for row in grouped["A"]] == ["1", "3"]


def test_heatmap_grid_from_long_table():
    rows = [
        {"case": "A", "descriptor": "x", "value": "1.0"},
        {"case": "A", "descriptor": "y", "value": "2.0"},
        {"case": "B", "descriptor": "x", "value": "3.0"},
    ]

    grid = build_heatmap_grid(rows, row_column="case", column_column="descriptor", value_column="value")

    assert grid.row_labels == ("A", "B")
    assert grid.column_labels == ("x", "y")
    assert math.isclose(float(grid.values[0, 1]), 2.0)
    assert math.isnan(float(grid.values[1, 1]))


def test_read_csv_rows(tmp_path):
    table = tmp_path / "table.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerow({"x": "1", "y": "2"})

    rows, fieldnames = read_csv_rows(table)

    assert fieldnames == ["x", "y"]
    assert rows == [{"x": "1", "y": "2"}]
