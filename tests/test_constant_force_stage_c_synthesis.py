import math
from pathlib import Path

from molsimflow.cli import build_parser as build_cli_parser
from molsimflow.postprocess.constant_force_stage_c_synthesis import (
    mixed_event_conditioned_blocks,
    mixed_event_transport_correlations,
    oh_partition_layer_blocks,
    replica_response_comparison,
)


def test_cli_registers_stage_c_synthesis():
    args = build_cli_parser().parse_args(
        [
            "postprocess",
            "constant-force-stage-c-synthesis",
            "--contract",
            "contract.json",
            "--output",
            "results",
        ]
    )
    assert args.contract == Path("contract.json")
    assert args.output == Path("results")
    assert args.func.__name__ == "_cmd_postprocess_constant_force_stage_c_synthesis"


def test_response_comparison_separates_sign_and_intensity():
    names = ("Jx_Fx_mps", "Jx_Fy_mps", "Jy_Fx_mps", "Jy_Fy_mps")
    stage_b = [{"case_id": "case", **dict(zip(names, (1.0, 0.2, 0.1, 2.0)))}]
    replica = [{"case_id": "case", **dict(zip(names, (2.0, -0.1, 0.3, 1.0)))}]
    row = replica_response_comparison(stage_b, replica)[0]
    assert row["longitudinal_same_sign_count"] == 2
    assert row["lateral_same_sign_count"] == 1
    assert row["Jx_Fy_sign_replication"] == "OPPOSITE"
    assert row["replica_to_stage_b_norm_ratio"] > 0.0


def test_event_blocks_keep_partial_final_window_and_event_classes():
    flux = [
        {
            "case_id": "mixed275",
            "branch_id": "x",
            "direction": "x",
            "axis": "x",
            "window_ps": "50",
            "window_index": "1",
            "start_ps": "50",
            "end_ps": "100",
            "duration_ps": "20",
            "response_total_velocity_mps": "2",
            "response_persistent_island_transfer_velocity_mps": "0.2",
            "response_lineage_reassignment_velocity_mps": "-0.1",
        }
    ]
    lineage = [{"branch_id": "x", "time_ps": "60", "event_type": "SPLIT"}]
    exchange = [
        {
            "branch_id": "x",
            "time_ps": "70",
            "exchange_class": "PERSISTENT_ISLAND_TRANSFER",
        },
        {
            "branch_id": "x",
            "time_ps": "80",
            "exchange_class": "MERGE_LINEAGE_REASSIGNMENT",
        },
    ]
    row = mixed_event_conditioned_blocks(
        flux, lineage, exchange, block_ps=50.0, full_window_ps=80.0
    )[0]
    assert row["actual_end_ps"] == 70.0
    assert row["split_event_count"] == 1
    assert row["persistent_transfer_count"] == 1
    assert row["lineage_reassignment_count"] == 1


def test_event_correlations_report_conditioned_support():
    rows = []
    for index in range(4):
        rows.append(
            {
                "branch_id": "x",
                "split_event_count": index,
                "merge_event_count": 0,
                "persistent_transfer_count": 0,
                "lineage_reassignment_count": 0,
                "response_total_velocity_mps": float(index),
                "absolute_response_total_velocity_mps": float(index),
                "response_persistent_island_transfer_velocity_mps": 0.0,
                "response_lineage_reassignment_velocity_mps": 0.0,
            }
        )
    selected = [
        row
        for row in mixed_event_transport_correlations(rows)
        if row["event_metric"] == "split_event_count"
        and row["transport_metric"] == "response_total_velocity_mps"
    ][0]
    assert math.isclose(selected["pearson_r"], 1.0)
    assert selected["event_blocks"] == 3
    assert selected["no_event_blocks"] == 1


def test_partition_layer_blocks_compute_cancellation():
    species = []
    for branch, h3o in (("f0", 8), ("x", 10)):
        for time_ps in (0, 10, 20):
            species.append(
                {
                    "branch_id": branch,
                    "time_ps": str(time_ps),
                    "solution_H3O": str(h3o),
                    "framework_OH": str(20 - h3o),
                    "proton_partition_pool": "20",
                }
            )
    layers = [
        {
            "branch_id": "x",
            "block_index": "0",
            "layer_index": "1",
            "mean_count": "10",
            "excess_axis_velocity_mps": "2",
        },
        {
            "branch_id": "x",
            "block_index": "0",
            "layer_index": "2",
            "mean_count": "10",
            "excess_axis_velocity_mps": "-2",
        },
    ]
    films = [
        {"branch_id": "f0", "direction": "none", "block_index": "0"},
        {
            "branch_id": "x",
            "direction": "x",
            "block_index": "0",
            "total_excess_velocity_mps": "0",
        },
    ]
    row = oh_partition_layer_blocks(
        species,
        layers,
        films,
        selected_layers=(1, 2),
        block_ps=50.0,
        full_window_ps=50.0,
    )[0]
    assert row["excess_solution_H3O"] == 2.0
    assert row["excess_framework_OH"] == -2.0
    assert math.isclose(row["selected_layer_cancellation_fraction"], 1.0)
