import math
from pathlib import Path

import numpy as np

from molsimflow.cli import build_parser as build_cli_parser
from molsimflow.postprocess.constant_force_stage_b_anisotropy import (
    _bin_indices,
    _deposit_path_segments,
    _pearson,
    response_matrix_rows,
)


def test_bin_indices_wrap_periodic_coordinates():
    x, y = _bin_indices(
        np.asarray([[0.0, 0.0], [9.9, 19.9], [10.1, -0.1]]),
        np.asarray([0.0, 0.0]),
        np.asarray([10.0, 20.0]),
        10,
    )
    assert x.tolist() == [0, 9, 0]
    assert y.tolist() == [0, 9, 9]


def test_response_matrix_keeps_longitudinal_and_lateral_components():
    branches = {
        "none": {"velocity_mps": np.asarray([1.0, -2.0])},
        "x": {"velocity_mps": np.asarray([4.0, -1.0])},
        "y": {"velocity_mps": np.asarray([0.5, 3.0])},
    }
    matrix, detail = response_matrix_rows("case", branches)
    assert matrix["Jx_Fx_mps"] == 3.0
    assert matrix["Jy_Fx_mps"] == 1.0
    assert matrix["Jx_Fy_mps"] == -0.5
    assert matrix["Jy_Fy_mps"] == 5.0
    assert detail[0]["longitudinal_mps"] == 3.0
    assert detail[0]["lateral_mps"] == 1.0
    assert detail[1]["longitudinal_mps"] == 5.0
    assert detail[1]["lateral_mps"] == -0.5


def test_path_deposition_preserves_long_periodic_displacement():
    accumulator = np.zeros((2, 4, 4), dtype=float)
    _deposit_path_segments(
        accumulator,
        np.asarray([[1.0, 1.0]]),
        np.asarray([[12.0, -6.0]]),
        np.asarray([0.0, 0.0]),
        np.asarray([10.0, 10.0]),
        4,
    )
    assert np.isclose(np.sum(accumulator[0]), 12.0)
    assert np.isclose(np.sum(accumulator[1]), -6.0)
    assert np.count_nonzero(accumulator[0]) > 1


def test_path_deposition_resolves_grid_crossing_below_half_box():
    accumulator = np.zeros((2, 4, 4), dtype=float)
    _deposit_path_segments(
        accumulator,
        np.asarray([[1.0, 1.0]]),
        np.asarray([[4.0, 0.0]]),
        np.asarray([0.0, 0.0]),
        np.asarray([10.0, 10.0]),
        4,
    )
    assert np.isclose(np.sum(accumulator[0]), 4.0)
    assert np.isclose(np.sum(accumulator[1]), 0.0)
    assert np.count_nonzero(accumulator[0]) >= 2


def test_pearson_excludes_nonfinite_cells_and_reports_support():
    value, count = _pearson(
        np.asarray([0.0, 0.5, 1.0, math.nan]),
        np.asarray([0.0, 1.0, 2.0, 9.0]),
    )
    assert math.isclose(value, 1.0)
    assert count == 3


def test_cli_registers_stage_b_anisotropy():
    args = build_cli_parser().parse_args(
        [
            "postprocess",
            "constant-force-stage-b-anisotropy",
            "--contract",
            "contract.json",
            "--output",
            "results",
        ]
    )
    assert args.contract == Path("contract.json")
    assert args.output == Path("results")
    assert args.func.__name__ == "_cmd_postprocess_constant_force_stage_b_anisotropy"
