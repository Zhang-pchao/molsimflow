import math

from molsimflow.postprocess.time_alignment import infer_timestep_time_scale, nearest_row_index


def test_nearest_row_index_respects_tolerance():
    rows = [{"time_ns": 0.0}, {"time_ns": 0.001}, {"time_ns": 0.002}]

    assert nearest_row_index(rows, 0.0011, 0.0002) == 1
    assert nearest_row_index(rows, 0.0014, 0.0002) is None


def test_infer_timestep_time_scale_prefers_matching_scale():
    rows = [{"time_ns": 0.0}, {"time_ns": 0.001}, {"time_ns": 0.002}]

    scale = infer_timestep_time_scale([0, 1, 2], rows, tolerance=0.0001)

    assert math.isclose(scale, 0.001)
