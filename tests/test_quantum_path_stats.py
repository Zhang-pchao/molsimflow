"""Count oracles and grouped-sampling limits for conditional path diagnostics."""

import json

import numpy as np
import pytest

from molsimflow.postprocess.quantum_path_stats import conditional_region_statistics


PATH = np.arange(8, dtype=float)
FRACTIONS = np.array([0, 0.25, 0.5, 1, 1, 0.5, 0.25, 0])[:, None]
WEIGHTS = np.tile([1, 2], 4)
BLOCKS = np.repeat([10, 20, 30, 40], 2)


def calculate(**changes):
    arguments = dict(conditioning=np.zeros((8, 1)), path_values=PATH,
                     region_fraction=FRACTIONS, log_weights=np.log(WEIGHTS),
                     bin_edges=[[-0.5, 0.5]], block_ids=BLOCKS)
    arguments.update(changes)
    return conditional_region_statistics(**arguments)


def test_independent_count_and_deleted_block_ratio_oracle():
    result = calculate()
    cell = result["cells"][0]
    assert cell["moments"]["path_mean"] == pytest.approx(11 / 3)
    assert cell["moments"]["path_variance"] == pytest.approx(47 / 9)
    assert cell["moments"]["region_probabilities"] == pytest.approx([7 / 16])
    means = np.array([42 / 9, 36 / 9, 30 / 9, 24 / 9])
    region = np.array([4.75 / 9, 2.75 / 9, 3.25 / 9, 5 / 9])
    for row, mean, probability in zip(cell["leave_one_block_out"], means, region):
        assert row["path_mean"] == pytest.approx(mean)
        assert row["region_probabilities"] == pytest.approx([probability])
    assert cell["path_mean_standard_error"] == pytest.approx(
        np.sqrt(3 / 4 * np.sum((means - means.mean()) ** 2)))
    assert cell["region_standard_errors"][0] == pytest.approx(
        np.sqrt(3 / 4 * np.sum((region - region.mean()) ** 2)))
    assert result["frame_ess"] == pytest.approx(144 / 20)
    assert result["max_frame_weight"] == pytest.approx(1 / 6)
    assert result["block_labels"] == [10, 20, 30, 40]
    json.dumps(result, allow_nan=False)


def test_bin_edges_multidimensional_and_outside_mass_are_explicit():
    coordinates = np.array([[0, 0], [1, 1], [2, 2], [3, 1],
                            [-1, 1], [0.5, 1.5], [1.5, 0.5], [1, 2.1]])
    result = calculate(conditioning=coordinates, log_weights=np.zeros(8),
                       bin_edges=[[0, 1, 2], [0, 1, 2]])
    assert result["bin_shape"] == [2, 2]
    assert result["in_range_frames"] == 5 and result["out_of_range_frames"] == 3
    assert result["in_range_weight_mass"] == pytest.approx(5 / 8)
    assert result["out_of_range_weight_mass"] == pytest.approx(3 / 8)
    assert [cell["frame_count"] for cell in result["cells"]] == [1, 1, 1, 2]
    assert sum(cell["weight_mass"] for cell in result["cells"]) == pytest.approx(5 / 8)


def test_dominant_weight_is_renormalized_after_block_deletion_and_gauge_shift():
    logs = np.repeat([10000.0, 0, 0, 0], 2)
    cell = calculate(log_weights=logs)["cells"][0]
    dropped = cell["leave_one_block_out"][0]
    assert dropped["path_mean"] == pytest.approx(4.5)
    assert dropped["region_probabilities"] == pytest.approx([3.25 / 6])
    assert cell["path_mean_standard_error"] is not None
    shifted = calculate(log_weights=logs + 50000)["cells"][0]
    assert shifted == cell


def test_repeated_beads_do_not_create_more_frames_or_change_statistics():
    beads = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1],
                      [1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    ordinary = calculate(region_fraction=beads.mean(axis=1)[:, None])
    repeated = calculate(region_fraction=np.repeat(beads, 7, axis=1).mean(axis=1)[:, None])
    assert ordinary == repeated
    assert repeated["frame_count"] == 8


def test_correlated_frame_repetition_scales_blocks_not_independent_sample_count():
    repeat = 4
    original = calculate()["cells"][0]
    result = calculate(conditioning=np.zeros((32, 1)), path_values=np.repeat(PATH, repeat),
                       region_fraction=np.repeat(FRACTIONS, repeat, axis=0),
                       log_weights=np.repeat(np.log(WEIGHTS), repeat),
                       block_ids=np.repeat(BLOCKS, repeat))
    cell = result["cells"][0]
    assert result["block_size"] == 8 and result["block_count"] == 4
    assert cell["path_mean_standard_error"] == pytest.approx(original["path_mean_standard_error"])
    assert cell["region_standard_errors"] == pytest.approx(original["region_standard_errors"])
    # Kish ESS counts weight concentration only; it increases without an error reduction.
    assert cell["frame_ess"] == pytest.approx(repeat * original["frame_ess"])


def test_empty_cell_too_few_blocks_and_cell_losing_support_are_not_zero_error():
    cells = calculate(bin_edges=[[-0.5, 0.5, 1.5]])["cells"]
    assert cells[1]["jackknife_status"] == "EMPTY_CELL"
    assert cells[1]["moments"] is None and cells[1]["frame_ess"] is None
    assert cells[1]["region_standard_errors"] == [None]
    one = calculate(block_ids=np.zeros(8, dtype=int))["cells"][0]
    assert one["jackknife_status"] == "TOO_FEW_BLOCKS"
    assert one["path_mean_standard_error"] is None
    coordinates = np.r_[np.zeros(2), np.ones(6)][:, None]
    lost = calculate(conditioning=coordinates, bin_edges=[[-0.5, 0.5, 1.5]])["cells"][0]
    assert lost["jackknife_status"] == "CELL_LOSES_SUPPORT"
    assert lost["leave_one_block_out"][0] is None
    assert lost["region_standard_errors"] == [None]
    json.dumps(cells, allow_nan=False)


@pytest.mark.parametrize("fractions", [np.zeros((8, 1)), np.ones((8, 1)),
                                      np.r_[1, np.zeros(7)][:, None]])
def test_absent_region_or_complement_and_deleted_events_have_unknown_error(fractions):
    cell = calculate(region_fraction=fractions)["cells"][0]
    assert cell["jackknife_status"] == "SUPPORTED"
    assert cell["region_jackknife_support"] == [False]
    assert cell["region_standard_errors"] == [None]


def test_overlapping_regions_are_allowed_without_row_sum_normalization():
    result = calculate(region_fraction=np.tile([0.75, 0.75], (8, 1)))
    assert result["cells"][0]["moments"]["region_probabilities"] == pytest.approx([0.75, 0.75])


@pytest.mark.parametrize("changes", [
    {"conditioning": np.zeros((0, 1))}, {"conditioning": np.zeros(8)},
    {"conditioning": np.zeros((8, 0))}, {"conditioning": np.full((8, 1), np.nan)},
    {"path_values": np.zeros(7)}, {"path_values": np.zeros((8, 1))},
    {"path_values": np.full(8, np.inf)}, {"path_values": np.ones(8, dtype=complex)},
    {"region_fraction": np.zeros(8)}, {"region_fraction": np.zeros((8, 0))},
    {"region_fraction": np.zeros((7, 1))}, {"region_fraction": np.full((8, 1), -0.1)},
    {"region_fraction": np.full((8, 1), 1.1)}, {"region_fraction": np.full((8, 1), np.nan)},
    {"log_weights": np.zeros(7)}, {"log_weights": np.full(8, -np.inf)},
    {"bin_edges": []}, {"bin_edges": [[0]]}, {"bin_edges": [[1, 0]]},
    {"bin_edges": [[0, 1, 1]]}, {"bin_edges": [[0, np.inf]]},
    {"bin_edges": [np.zeros((2, 2))]}, {"bin_edges": [[0, 1], [0, 1]]},
    {"block_ids": np.zeros(8)}, {"block_ids": np.zeros(8, dtype=bool)},
    {"block_ids": np.zeros(7, dtype=int)}, {"block_ids": -np.ones(8, dtype=int)},
    {"block_ids": [0, 0, 1, 1, 0, 0, 2, 2]},
    {"block_ids": [0, 0, 0, 1, 1, 2, 2, 2]},
])
def test_invalid_arrays_edges_or_block_layout_fail_closed(changes):
    with pytest.raises(ValueError):
        calculate(**changes)
