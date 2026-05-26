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
            status="partial",
            notes="Water density and ion occupancy migrated; dynamics/H-bonds remain in legacy triage.",
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
            status="partial",
            notes="Table-based entry/exit flux and seed survival migrated; escape direction remains.",
        ),
        DoubleBubbleMergeStage(
            name="transition_events",
            command_group="postprocess",
            reusable_module="molsimflow.postprocess.events",
            status="partial",
            notes="Generic event detection and event-aligned table summaries migrated.",
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
