"""Frame identity must survive timestamp roundoff without interpolation."""

import numpy as np
import pytest

from molsimflow.postprocess.pimd_reweight import aligned_time_indices


@pytest.mark.parametrize("offset", [-1e-10, 1e-10])
def test_time_alignment_accepts_roundoff_on_either_side_including_endpoints(offset):
    source = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    expected = np.array([0, 2, 4])
    target = source[expected] + offset
    indices = aligned_time_indices(source, target, tolerance=1e-8)
    np.testing.assert_array_equal(indices, expected)
    # Frame data must follow the same mapping as timestamps.
    np.testing.assert_array_equal(np.arange(10).reshape(5, 2)[indices], [[0, 1], [4, 5], [8, 9]])


def test_time_alignment_rejects_two_source_candidates_in_tolerance():
    with pytest.raises(ValueError, match="ambiguous"):
        aligned_time_indices([0.0, 1e-9], [0.5e-9], tolerance=1e-8)


def test_time_alignment_rejects_reusing_one_source_frame():
    with pytest.raises(ValueError, match="same source frame"):
        aligned_time_indices([0.0, 1.0], [-2e-10, -1e-10], tolerance=1e-8)


@pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), -1.0])
def test_time_alignment_rejects_nonfinite_or_negative_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        aligned_time_indices([0.0, 1.0], [0.0], tolerance=tolerance)


@pytest.mark.parametrize("target", [[-0.1], [0.5], [1.1]])
def test_time_alignment_rejects_missing_frames_without_interpolation(target):
    with pytest.raises(ValueError, match="absent|beyond"):
        aligned_time_indices([0.0, 1.0], target, tolerance=1e-8)


def test_time_alignment_zero_tolerance_preserves_exact_frame_identity():
    np.testing.assert_array_equal(
        aligned_time_indices([0.0, 0.5, 1.0], [0.0, 1.0], tolerance=0.0), [0, 2],
    )
