"""Independent estimator, observable, bias and path-weight contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Protocol, Tuple

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

    def raw_log_weights(self) -> np.ndarray: ...


BIAS_MODES = {
    "centroid_coord",
    "bead_mean",
    "bead_density_shared",
    "centroid_conditioned",
    "bead_probability_mixture",
    "centroid_probability_mixture",
    "contracted_bead_mean",
}
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


@dataclass(frozen=True)
class PreparedInputs:
    """Internal stage handoff; arrays preserve the original frame order."""

    source: Mapping[str, object]
    selection: Mapping[str, object]
    reweight: Mapping[str, object]
    bias_mode: str
    conditional_spec: Mapping[str, object] | None
    probability_spec: Mapping[str, object] | None
    primary_estimator: str
    sampling_time_scale_to_fs: float
    bead_time_scale_to_fs: float
    kernel_time_scale_to_fs: float
    bead_offset_fs: float
    run_root: Path
    conditional_model: Mapping[str, object] | None
    probability_model: Mapping[str, object] | None
    fields: Tuple[str, ...]
    centroid_data: np.ndarray
    bead_tables: List[Tuple[Tuple[str, ...], np.ndarray]]


@dataclass(frozen=True)
class AlignedPaths:
    """Internal stage handoff; arrays preserve the original frame order."""

    restart_policy: str
    centroid_restart_duplicates: int
    bead_restart_duplicates: List[int]
    centroid_data: np.ndarray
    time_fs: np.ndarray
    first_time_fs: float
    last_time_fs: float
    selected: np.ndarray
    selected_time_ps: np.ndarray
    selected_steps: np.ndarray
    cv_names: Tuple[str, ...]
    sampling_cv_names: Tuple[str, ...]
    bead_cv_names: Tuple[str, ...]
    sampling_label: str
    sampling_slug: str
    derived_spec: Mapping[str, object] | None
    cv_labels: Tuple[str, ...]
    centroid: np.ndarray
    beads: np.ndarray
    selected_bead_field: Callable[[str], np.ndarray]
    diagnostic_spec: Mapping[str, object] | None
    diagnostic_sampling: np.ndarray | None
    diagnostic_beads: np.ndarray | None
    diagnostic_mean_error: float | None
    derived_centroid: np.ndarray | None
    derived_beads: np.ndarray | None
    derived_validation: Mapping[str, object] | None


@dataclass(frozen=True)
class WeightedPaths:
    """Internal stage handoff; arrays preserve the original frame order."""

    weight_kind: str
    bias_column: str | None
    bias_ev: np.ndarray | None
    opes_bias_ev: np.ndarray | None
    extra_bias_columns: List[str]
    rct_ev: np.ndarray | None
    zed: np.ndarray | None
    neff: np.ndarray | None
    nker: np.ndarray | None
    shared_diagnostic_max_deltas: Dict[str, float]
    kbt_ev: float
    raw_log_weights: np.ndarray
    weights: np.ndarray
    quasi_static_declared: bool


@dataclass(frozen=True)
class JointResults:
    """Internal stage handoff; arrays preserve the original frame order."""

    x_grid: np.ndarray
    y_grid: np.ndarray
    variants: Dict[str, Tuple[float, ...]]
    primary_name: str
    primary_supports: Dict[str, np.ndarray]
    primary_support: np.ndarray
    primary: Dict[str, np.ndarray]
    primary_kcal: Dict[str, np.ndarray]
    marginal_tables: Dict[str, object]
    supports_by_variant: Dict[str, Dict[str, np.ndarray]]
