import math
from pathlib import Path

import numpy as np

from molsimflow.cli import build_parser as build_cli_parser
from molsimflow.postprocess.constant_force_estimator_consistency import (
    response_block_rows,
    response_comparison_rows,
    time_weighted_mean,
)


def test_cli_registers_constant_force_estimator_consistency():
    args = build_cli_parser().parse_args(
        [
            "postprocess",
            "constant-force-estimator-consistency",
            "--contract",
            "contract.json",
            "--output",
            "results",
        ]
    )
    assert args.contract == Path("contract.json")
    assert args.output == Path("results")
    assert args.func.__name__ == "_cmd_postprocess_constant_force_estimator_consistency"


def test_time_weighted_mean_integrates_linear_series():
    assert math.isclose(time_weighted_mean([0, 10, 20], [0.0, 2.0, 4.0], 0.5), 2.0)


def test_response_comparison_separates_estimator_signs():
    cases = {"case": {"f0": "none", "fx": "x", "fy": "y"}}
    displacement = {
        ("case", "f0"): np.asarray([0.0, 0.0]),
        ("case", "fx"): np.asarray([1.0, 0.2]),
        ("case", "fy"): np.asarray([0.1, 2.0]),
    }
    high = {
        ("case", "f0"): np.asarray([0.0, 0.0]),
        ("case", "fx"): np.asarray([0.9, 0.1]),
        ("case", "fy"): np.asarray([0.2, 1.8]),
    }
    sparse = {
        ("case", "f0"): np.asarray([0.0, 0.0]),
        ("case", "fx"): np.asarray([-0.5, 0.3]),
        ("case", "fy"): np.asarray([0.4, 1.0]),
    }
    rows = response_comparison_rows(cases, displacement, high, sparse)
    row = next(
        item
        for item in rows
        if item["drive_direction"] == "x" and item["response_component"] == "x"
    )
    assert row["high_frequency_displacement_sign_match"] == "SAME"
    assert row["sparse_high_frequency_sign_match"] == "OPPOSITE"


def test_response_blocks_keep_partial_final_window():
    steps = np.arange(0, 21, 2)
    baseline = np.zeros((len(steps), 2))
    driven = np.column_stack((np.ones(len(steps)), np.zeros(len(steps))))
    cases = {"case": {"f0": "none", "fx": "x"}}
    rows = response_block_rows(
        cases,
        {("case", "f0"): (steps, baseline), ("case", "fx"): (steps, driven)},
        start_step=0,
        end_step=20,
        timestep_fs=100.0,
        block_sizes_ps=(1.5,),
    )
    selected = [row for row in rows if row["response_component"] == "x"]
    assert len(selected) == 2
    assert math.isclose(float(selected[-1]["duration_ps"]), 0.5)
    assert math.isclose(float(selected[-1]["response_velocity_mps"]), 1.0)
