"""Namespace for migrated MD post-processing workflows."""

from molsimflow.postprocess.centroids import BubbleCentroidCalculator, UnionFind
from molsimflow.postprocess.bubble_surface_distance import (
    BubbleSurfaceDistanceAnalyzer,
    FrameDistanceResult,
)
from molsimflow.postprocess.bridge_descriptors import BridgeCylinder
from molsimflow.postprocess.bridge_film import BridgeFilmConfig
from molsimflow.postprocess.bridge_water_dynamics import BridgeWaterDynamicsConfig, TraceInputSpec
from molsimflow.postprocess.bridge_water_dewetting import BridgeWaterDewettingConfig
from molsimflow.postprocess.bridge_water_escape import (
    BridgeWaterEscapeConfig,
    SeedPositionRow,
    build_seed_escape_events,
    classify_escape_direction,
)
from molsimflow.postprocess.case_comparison import CasePairSpec, DescriptorTableSpec
from molsimflow.postprocess.coalescence_state import CoalescenceStateConfig
from molsimflow.postprocess.coupling import CouplingConfig
from molsimflow.postprocess.events import TransitionEventConfig
from molsimflow.postprocess.fes_analysis import FesCurve, FesCurveSpec
from molsimflow.postprocess.ion_distribution import AtomRecord, IonZDistribution
from molsimflow.postprocess.ion_species import (
    IonSpeciesConfig,
    IonSpeciesFrameResult,
    MoleculeRecord,
    classify_ion_species,
)
from molsimflow.postprocess.species_assignment import (
    OxygenHydrogenAssignment,
    assign_hydrogen_to_nearest_oxygen,
    classify_oxygen_species_indices,
)
from molsimflow.postprocess.time_alignment import infer_timestep_time_scale, nearest_row_index
from molsimflow.postprocess.transitions import (
    SpeciesStateRow,
    SpeciesTransitionResult,
    build_species_transition_matrix,
    infer_species_order,
)
from molsimflow.postprocess.water_orientation import (
    WaterOrientationSummaryConfig,
    angle_to_axis_deg,
    compute_s_rho,
    compute_water_orientation_sample,
    nematic_order,
)

__all__ = [
    "AtomRecord",
    "BridgeCylinder",
    "BridgeFilmConfig",
    "BridgeWaterDynamicsConfig",
    "BridgeWaterDewettingConfig",
    "BridgeWaterEscapeConfig",
    "BubbleCentroidCalculator",
    "BubbleSurfaceDistanceAnalyzer",
    "CasePairSpec",
    "CoalescenceStateConfig",
    "CouplingConfig",
    "DescriptorTableSpec",
    "FesCurve",
    "FesCurveSpec",
    "FrameDistanceResult",
    "IonSpeciesConfig",
    "IonSpeciesFrameResult",
    "IonZDistribution",
    "MoleculeRecord",
    "OxygenHydrogenAssignment",
    "SeedPositionRow",
    "SpeciesStateRow",
    "SpeciesTransitionResult",
    "TraceInputSpec",
    "TransitionEventConfig",
    "UnionFind",
    "WaterOrientationSummaryConfig",
    "angle_to_axis_deg",
    "assign_hydrogen_to_nearest_oxygen",
    "build_seed_escape_events",
    "build_species_transition_matrix",
    "classify_escape_direction",
    "classify_ion_species",
    "classify_oxygen_species_indices",
    "compute_s_rho",
    "compute_water_orientation_sample",
    "infer_species_order",
    "infer_timestep_time_scale",
    "nematic_order",
    "nearest_row_index",
]
