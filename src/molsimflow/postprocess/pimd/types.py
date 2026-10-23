"""Independent estimator, observable, bias and path-weight contracts."""
from __future__ import annotations
from enum import Enum
from typing import Protocol
import numpy as np

class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

class EstimatorKind(_ValueEnum):
    PROBABILITY_MEAN = "probability_mean"
    FREE_ENERGY_MEAN = "free_energy_mean"

class ObservableKind(_ValueEnum):
    REAL_BEAD_CV = "real_bead_cv"
    ARITHMETIC_BEAD_MEAN_CV = "arithmetic_bead_mean_cv"
    CONTRACTED_CV = "contracted_cv"
    CENTROID_CV = "centroid_cv"

class BiasMode(_ValueEnum):
    CENTROID_COORD = "centroid_coord"
    BEAD_MEAN = "bead_mean"
    BEAD_DENSITY_SHARED = "bead_density_shared"
    CENTROID_CONDITIONED = "centroid_conditioned"
    BEAD_PROBABILITY_MIXTURE = "bead_probability_mixture"
    CENTROID_PROBABILITY_MIXTURE = "centroid_probability_mixture"
    CONTRACTED_BEAD_MEAN = "contracted_bead_mean"

class WeightKind(_ValueEnum):
    FIXED_BIAS = "fixed_bias"
    QUASI_STATIC_OPES = "quasi_static_opes"
    PRECOMPUTED = "precomputed"

class WeightProvider(Protocol):
    """Supply one raw log weight per complete frame, in input frame order.

    Providers declare sampling assumptions separately from bias coordinates
    and estimators. Beads are never independent weighted samples.
    """
    def raw_log_weights(self) -> np.ndarray:
        ...

BIAS_MODES = {"centroid_coord", "bead_mean", "bead_density_shared", "centroid_conditioned",
              "bead_probability_mixture", "centroid_probability_mixture", "contracted_bead_mean"}
# Primary input names resolve to the existing output-schema identifiers.
BIAS_MODE_ALIASES = {
    "coordinate_mean": "centroid_coord",
    "cv_mean": "bead_mean",
    "bias_mean": "bead_density_shared",
    "probability_mean": "bead_probability_mixture",
    "contracted_cv_mean": "contracted_bead_mean",
}
BIAS_MODES |= BIAS_MODE_ALIASES.keys()
WEIGHT_KINDS = {"fixed_bias", "quasi_static_opes", "precomputed"}
FES_ESTIMATORS = {"probability_mean", "free_energy_mean"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_bias_mode(value: str) -> str:
    """Resolve a bias name to its stable output-schema identifier.

    Both primary names and original input names are accepted. This does not
    select the FES estimator, which has its own independent contract field.
    """

    mode = str(value)
    _require(mode in BIAS_MODES, f"unsupported PIMD bias mode: {mode}")
    return BIAS_MODE_ALIASES.get(mode, mode)



def validate_primary_estimator(value: str) -> str:
    """Validate the public estimator names accepted as the primary result."""

    estimator = str(value)
    _require(estimator in FES_ESTIMATORS, f"unsupported primary estimator: {estimator}")
    return estimator
