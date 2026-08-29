import numpy as np
import pytest

from molsimflow.plotting.parity import adaptive_histogram_edges


def test_adaptive_histogram_edges_scale_with_sample_count_and_clip_tail():
    small = adaptive_histogram_edges(np.linspace(0.0, 1.0, 100))
    large = adaptive_histogram_edges(np.r_[np.linspace(0.0, 1.0, 40000), 100.0])

    assert len(small) - 1 == 55
    assert len(large) - 1 == 160
    assert 1.0 < large[-1] < 2.0


def test_adaptive_histogram_edges_validate_input():
    with pytest.raises(ValueError):
        adaptive_histogram_edges(np.array([]))
    with pytest.raises(ValueError):
        adaptive_histogram_edges(np.array([1.0]), quantile=0.0)
