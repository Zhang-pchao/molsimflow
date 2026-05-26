# Legacy Remaining Triage

## Decision

Do not delete `legacy_sources/` yet.

The directory is still an ignored local migration reference, not publishable
package code.  It contains many private paths and project-specific runners, but
it also still contains reusable algorithms that have not been migrated into
`src/molsimflow`.

Current audit snapshot:

- `legacy_sources/`: about 4.1 MB;
- legacy file count: 420 files;
- legacy Python package modules under `md_postprocess/src/md_postprocess`: 128;
- private path/environment hits in legacy text: 509;
- tracked files under `legacy_sources/`: 0.

## Already Covered By `src/molsimflow`

These legacy areas have first-pass engineered replacements:

- structure preparation:
  - double-bubble slab planning;
  - TiO2 adapter;
  - prebuilt-interface adapter;
  - Packmol input rendering and optional execution;
  - extended XYZ PBC injection;
  - LAMMPS atomic data conversion.
- PLUMED:
  - double-bubble PLUMED generator from PACKMOL and LAMMPS data.
- post-processing:
  - two-bubble centroids;
  - bubble surface distance;
  - coalescence-state assignment;
  - ion species classification;
  - ion z-distribution;
  - first bridge-water and bridge-ion descriptor tables;
  - bridge-water dewetting and spanning-connectivity tables;
  - bridge-water entry/exit flux and seed-survival trace-table summaries;
  - generic transition-event detection and event-aligned table summaries;
  - bridge liquid-film state, barrier, residence, and coordination table
    summaries;
  - ion-water coupling, lag correlation, state-comparison, and event-aligned
    feature-table summaries;
  - 1D FES processing and barrier summaries;
  - case scorecards, case-pair deltas, and descriptor correlations.
- plotting/config:
  - generic CSV-driven line, scatter, and heatmap plots;
  - external workflow config helper;
  - generic scheduler template.

## Remaining Migration Candidates

### P1: Reusable Analysis Modules

These are strong candidates for future engineering into reusable APIs and CLIs.

- `analysis/bridge_water_escape_direction.py`
  - Seed-water escape direction classification from trajectory-backed bridge
    geometry.
  - Suggested target: `molsimflow.postprocess.bridge_water_dynamics`.
- `analysis/bridge_hbond_network.py`
  - Hydrogen-bond graph/network descriptors, lifetimes, motifs, and gap-bin
    summaries.
  - Suggested target: `molsimflow.postprocess.hbond_network`.
  - Keep MDAnalysis as an optional dependency.
- `analysis/water_orientation_shell.py`
  - Water orientation, radial profiles, s-rho maps, CV summaries, and angular
    distributions.
  - Suggested target: `molsimflow.postprocess.water_orientation`.
### P2: Topology And Local Environment

These are useful but large and should be split before migration.

- `analysis/ion_effect_water_topology.py`
  - Contains multiple stages in one large file:
    - bridge atom membership;
    - contact graph topology;
    - cycle analysis;
    - local water environment classification;
    - environment similarity;
    - gap/event-conditioned summaries;
    - topology sensitivity.
  - Suggested targets:
    - `molsimflow.postprocess.bridge_membership`;
    - `molsimflow.postprocess.contact_graph`;
    - `molsimflow.postprocess.local_environment`;
    - `molsimflow.postprocess.sensitivity`.
  - Keep `networkx` optional.
- `analysis/ion_effect_water_topology_stage02.py`
  - Bridge microstate frame table, ion-position table, species-region summaries,
    and water-position QC.
  - Suggested target: split into bridge microstate and region-position helpers.

### P2: Low-Level Reusable Utilities

These should be migrated before more trajectory-heavy modules reuse duplicate
logic.

- `species/hydrogen_oxygen_assignment.py`
  - Periodic O-H assignment and oxygen species classification.
  - Suggested target: `molsimflow.postprocess.species_assignment`.
- `transitions/species_transition_matrix.py`
  - Species transition counting and transition-matrix utilities.
  - Suggested target: `molsimflow.postprocess.transitions`.
- shared LAMMPS dump readers and time-alignment helpers from:
  - `molsimflow.postprocess.bridge_water_dewetting`;
  - `analysis/water_orientation_shell.py`;
  - `analysis/bridge_transition_event_analysis.py`;
  - `analysis/bridge_ion_water_coupling.py`.
  - Suggested target: `molsimflow.io.lammps_dump` and
    `molsimflow.postprocess.time_alignment`.

### P3: Mostly Aggregation Or Publication Layers

These should not be copied directly.  They may provide useful column names or
scorecard ideas, but the generic table APIs already cover much of the reusable
behavior.

- barrier-correlation synthesis modules;
- case-comparison summary/report/JACS/manuscript modules;
- mechanism descriptor synthesis modules;
- `analysis/bridge_ion_species_matrix_synthesis.py`;
- trace montage and publication package scripts.

## Do Not Migrate Directly

These should remain excluded from tracked package code unless rewritten as
small public examples:

- batch setup scripts with scheduler/account/environment assumptions;
- one-off `run_*` wrappers that only hardcode source paths;
- publication narrative/report assembly;
- smoke-test scheduler files with cluster-local assumptions;
- generated analysis outputs, plots, CSVs, logs, or backups.

## Cleanup Policy

Keep `legacy_sources/` local and ignored until the P1/P2 migration candidates
above are either migrated or explicitly rejected.  After that:

1. Confirm `git ls-files legacy_sources` returns no tracked files.
2. Confirm this triage document has no remaining open migration candidates.
3. Archive `legacy_sources/` outside the repository if historical provenance is
   still needed.
4. Remove the local directory with:

```bash
cd /path/to/molsimflow
rm -rf legacy_sources
```

Do not run the deletion step while P1/P2 items remain unresolved.
