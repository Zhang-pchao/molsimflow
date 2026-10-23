"""Core estimators and data checks for bead-defined PIMD free energies."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from molsimflow.postprocess.pimd.types import (
    validate_bias_mode,
    validate_primary_estimator,
)
from molsimflow.postprocess.pimd.weights import (
    normalized_log_weights,
    total_bias_energy,
    frame_log_weights,
)
from molsimflow.postprocess.pimd.alignment import (
    restart_unique_indices,
    assemble_bead_frames,
)
from molsimflow.postprocess.pimd.types import BIAS_MODES, BIAS_MODE_ALIASES, WEIGHT_KINDS, FES_ESTIMATORS
from molsimflow.postprocess.pimd.estimators import (
    _histogram_log_mass,
    quantum_histogram_masses,
    bead_density_estimators,
    quantum_fes_1d,
)
from molsimflow.postprocess.pimd.uncertainty import (
    quantum_fes_block_jackknife_1d,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
