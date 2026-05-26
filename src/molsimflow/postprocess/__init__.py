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

__all__ = [
    "AtomRecord",
    "BridgeCylinder",
    "BridgeFilmConfig",
    "BridgeWaterDynamicsConfig",
    "BridgeWaterDewettingConfig",
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
    "TraceInputSpec",
    "TransitionEventConfig",
    "UnionFind",
    "assign_hydrogen_to_nearest_oxygen",
    "classify_ion_species",
    "classify_oxygen_species_indices",
    "infer_timestep_time_scale",
    "nearest_row_index",
]
