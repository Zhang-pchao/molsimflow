import csv
import math

import numpy as np

from molsimflow.postprocess.gas_connectivity import (
    GasContactConfig,
    analyze_gas_contact_table,
    gas_contact_metrics,
    summarize_distance_bins,
)


def test_gas_contact_metrics_detects_cross_group_component():
    coms = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [3.3, 0.0, 0.0],
        ]
    )
    groups = ["A", "A", "B", "B"]
    bounds = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])

    metrics = gas_contact_metrics(
        coms,
        groups,
        bounds,
        GasContactConfig(contact_cutoff_A=1.3, connected_lcc_fraction=0.95, connected_group_fraction=0.80),
    )

    assert metrics["cross_contact_flag"] is True
    assert metrics["gas_connected_flag"] is True
    assert metrics["cross_contacts"] == 1
    assert metrics["largest_component_size"] == 4
    assert math.isclose(metrics["min_cross_distance_A"], 1.2)


def test_gas_contact_table_summary_outputs(tmp_path):
    table = tmp_path / "gas_frames.csv"
    rows = [
        {
            "d3d_all": 39.0,
            "cross_contact_flag": "true",
            "gas_connected_flag": "false",
            "cross_contacts": 2,
            "largest_component_fraction": 0.5,
            "largest_component_size": 2,
            "second_component_size": 2,
            "min_cross_distance_A": 1.2,
        },
        {
            "d3d_all": 37.0,
            "cross_contact_flag": "true",
            "gas_connected_flag": "true",
            "cross_contacts": 4,
            "largest_component_fraction": 1.0,
            "largest_component_size": 4,
            "second_component_size": 0,
            "min_cross_distance_A": 0.8,
        },
        {
            "d3d_all": 45.0,
            "cross_contact_flag": "false",
            "gas_connected_flag": "false",
            "cross_contacts": 0,
            "largest_component_fraction": 0.5,
            "largest_component_size": 2,
            "second_component_size": 2,
            "min_cross_distance_A": 8.0,
        },
    ]
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    outputs = analyze_gas_contact_table(
        table,
        tmp_path / "out",
        radius_sum_A=38.0,
        d_range=(36.0, 46.0),
        bin_width_A=2.0,
        min_bin_frames=1,
    )

    assert outputs["d_bin_summary"].exists()
    assert outputs["window_summary"].exists()
    assert outputs["threshold_summary"].exists()
    assert outputs["transition_summary"].exists()

    with outputs["transition_summary"].open() as handle:
        transition = list(csv.DictReader(handle))
    below = next(row for row in transition if row["scope"] == "d_le_radius_sum")
    above = next(row for row in transition if row["scope"] == "d_gt_radius_sum")
    assert math.isclose(float(below["gas_connected_probability"]), 1.0)
    assert math.isclose(float(above["gas_connected_probability"]), 0.0)

    bin_rows = summarize_distance_bins(rows, d_range=(36.0, 46.0), bin_width_A=2.0, radius_sum_A=38.0)
    assert len(bin_rows) == 3
