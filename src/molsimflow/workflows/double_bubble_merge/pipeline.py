"""Recommended stage ordering for double-bubble merging workflows.

This namespace is intentionally for workflow composition, not low-level
algorithms.  Generic utilities should still live under `molsimflow.structure`,
`molsimflow.plumed`, or `molsimflow.postprocess`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class DoubleBubbleMergeStage:
    """One stage in the recommended double-bubble merging workflow."""

    name: str
    command_group: str
    reusable_module: str
    status: str
    notes: str


@dataclass(frozen=True)
class DoubleBubbleResidualAdapter:
    """Optional adapter that is specific to the double-bubble legacy layout."""

    name: str
    legacy_source: str
    target_module: str
    status: str
    expected_output: str
    notes: str


def recommended_postprocess_stages() -> Tuple[DoubleBubbleMergeStage, ...]:
    """Return the current staged workflow plan."""

    stages: List[DoubleBubbleMergeStage] = [
        DoubleBubbleMergeStage(
            name="structure_preparation",
            command_group="structure",
            reusable_module="molsimflow.structure.double_bubble_slab",
            status="migrated",
            notes="Build double-bubble PACKMOL inputs from TiO2 or prebuilt interfaces.",
        ),
        DoubleBubbleMergeStage(
            name="plumed_generation",
            command_group="plumed",
            reusable_module="molsimflow.plumed.double_bubble",
            status="migrated",
            notes="Generate double-bubble PLUMED files from PACKMOL and LAMMPS data.",
        ),
        DoubleBubbleMergeStage(
            name="coalescence_state",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.coalescence_state",
            status="migrated",
            notes="Build the state table used by bridge-water and event descriptors.",
        ),
        DoubleBubbleMergeStage(
            name="bubble_geometry",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.centroids",
            status="migrated",
            notes="Compute centroid and surface-distance descriptors.",
        ),
        DoubleBubbleMergeStage(
            name="ion_species",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.ion_species",
            status="migrated",
            notes="Classify species and build ion distribution inputs.",
        ),
        DoubleBubbleMergeStage(
            name="bridge_descriptors",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.bridge_descriptors",
            status="migrated",
            notes="Water density and strict bridge-ion occupancy table summaries are migrated.",
        ),
        DoubleBubbleMergeStage(
            name="bridge_water_dewetting",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.bridge_water_dewetting",
            status="migrated",
            notes="Compute bridge water count, dewetting fraction, and spanning connectivity.",
        ),
        DoubleBubbleMergeStage(
            name="bridge_water_dynamics",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.bridge_water_dynamics",
            status="migrated",
            notes="Entry/exit flux, turnover, drainage, and seed-survival table summaries are migrated.",
        ),
        DoubleBubbleMergeStage(
            name="bridge_water_escape",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.bridge_water_escape",
            status="migrated",
            notes="Seed-water retained/exited status and escape direction from explicit position tables.",
        ),
        DoubleBubbleMergeStage(
            name="water_orientation",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.water_orientation",
            status="migrated",
            notes="Water-orientation geometry and sample-table summaries.",
        ),
        DoubleBubbleMergeStage(
            name="hbond_network",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.hbond_network",
            status="migrated",
            notes="H-bond edge-table network summaries and lifetimes.",
        ),
        DoubleBubbleMergeStage(
            name="contact_graph",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.contact_graph",
            status="migrated",
            notes="Generic contact graph topology summaries from explicit edge tables.",
        ),
        DoubleBubbleMergeStage(
            name="local_environment",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.local_environment",
            status="migrated",
            notes="Local-environment class summaries and persistent-entity transition matrices.",
        ),
        DoubleBubbleMergeStage(
            name="bridge_microstate",
            command_group="workflow",
            reusable_module="molsimflow.workflows.double_bubble_merge.microstate",
            status="migrated_workflow_adapter",
            notes="Double-bubble-specific frame microstate, species-region, and QC table builder.",
        ),
        DoubleBubbleMergeStage(
            name="species_transitions",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.transitions",
            status="migrated",
            notes="Generic persistent-entity state transition matrices.",
        ),
        DoubleBubbleMergeStage(
            name="transition_events",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.events",
            status="migrated",
            notes="Generic event detection and event-aligned table summaries migrated.",
        ),
        DoubleBubbleMergeStage(
            name="bridge_film",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.bridge_film",
            status="migrated",
            notes="Frame-table film-state, barrier, residence, and coordination summaries migrated.",
        ),
        DoubleBubbleMergeStage(
            name="ion_water_coupling",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.coupling",
            status="migrated",
            notes="Feature-table coupling, lag, state comparison, and event-aligned summaries migrated.",
        ),
        DoubleBubbleMergeStage(
            name="fes_barriers",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.fes_analysis",
            status="migrated",
            notes="Process FES curves and barrier windows.",
        ),
        DoubleBubbleMergeStage(
            name="case_comparison",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.case_comparison",
            status="migrated",
            notes="Join descriptor scorecards and compute deltas/correlations.",
        ),
    ]
    return tuple(stages)


def residual_adapter_plan() -> Tuple[DoubleBubbleResidualAdapter, ...]:
    """Return optional double-bubble adapters that should not be core APIs."""

    adapters: List[DoubleBubbleResidualAdapter] = [
        DoubleBubbleResidualAdapter(
            name="seed_position_table_from_trajectory",
            legacy_source="analysis/bridge_water_escape_direction.py",
            target_module="molsimflow.workflows.double_bubble_merge",
            status="optional_workflow_adapter",
            expected_output="seed_positions.csv for molsimflow.postprocess.bridge_water_escape",
            notes="Legacy case discovery and segment selection are double-bubble-layout specific.",
        ),
        DoubleBubbleResidualAdapter(
            name="water_orientation_samples_from_trajectory",
            legacy_source="analysis/water_orientation_shell.py",
            target_module="molsimflow.workflows.double_bubble_merge",
            status="optional_workflow_adapter",
            expected_output="water_orientation_samples.csv for molsimflow.postprocess.water_orientation",
            notes="Atom selection, bubble-center lookup, and COLVAR alignment are adapter concerns.",
        ),
        DoubleBubbleResidualAdapter(
            name="hbond_edges_from_trajectory",
            legacy_source="analysis/bridge_hbond_network.py",
            target_module="molsimflow.workflows.double_bubble_merge",
            status="optional_workflow_adapter",
            expected_output="hbond_edges.csv for molsimflow.postprocess.hbond_network",
            notes="MDAnalysis-backed H-bond detection should remain optional.",
        ),
        DoubleBubbleResidualAdapter(
            name="contact_edges_and_local_environment_samples",
            legacy_source="analysis/ion_effect_water_topology.py",
            target_module="molsimflow.workflows.double_bubble_merge",
            status="optional_workflow_adapter",
            expected_output="contact_edges.csv and local_environment_samples.csv",
            notes="Membership generation is tied to bridge geometry and project-specific species labels.",
        ),
        DoubleBubbleResidualAdapter(
            name="microstate_and_region_qc_tables",
            legacy_source="analysis/ion_effect_water_topology_stage02.py",
            target_module="molsimflow.workflows.double_bubble_merge",
            status="migrated_workflow_adapter",
            expected_output="bridge_microstate_frame_table.csv and region/QC tables",
            notes="Implemented by molsimflow.workflows.double_bubble_merge.microstate.",
        ),
        DoubleBubbleResidualAdapter(
            name="publication_and_case_synthesis",
            legacy_source="analysis/*_synthesis.py and analysis/case_comparison_*",
            target_module="docs or external notebooks",
            status="do_not_migrate_directly",
            expected_output="publication figures, manuscripts, or reports",
            notes="Use generic scorecard/plotting APIs instead of copying narrative scripts.",
        ),
    ]
    return tuple(adapters)
