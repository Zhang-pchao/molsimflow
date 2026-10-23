"""Independent scientific concepts and compatibility imports."""
import numpy as np
from molsimflow.postprocess import pimd_fes, pimd_reweight
from molsimflow.postprocess.pimd.types import (
    BiasMode, EstimatorKind, ObservableKind, WeightKind, validate_bias_mode,
    validate_primary_estimator,
)
from molsimflow.postprocess.pimd.weights import normalized_log_weights


def test_independent_estimator_and_bias_alias_names():
    assert set(EstimatorKind) == {"probability_mean", "free_energy_mean"}
    assert len(ObservableKind) == 4
    assert validate_primary_estimator(EstimatorKind.PROBABILITY_MEAN) == "probability_mean"
    assert validate_bias_mode("probability_mean") == BiasMode.BEAD_PROBABILITY_MIXTURE
    assert WeightKind.QUASI_STATIC_OPES == "quasi_static_opes"


def test_legacy_normalization_keywords_and_path_count():
    raw = np.array([-1200., 2., 1000.])
    expected = normalized_log_weights(values=raw)
    np.testing.assert_array_equal(pimd_reweight.normalized_log_weights(raw=raw), expected)
    np.testing.assert_array_equal(pimd_fes.normalized_log_weights(values=raw), expected)
    assert expected.shape == raw.shape
