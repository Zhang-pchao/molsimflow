# Legacy Remaining Triage

## Decision

Do not upload `legacy_sources/`.

The directory is still an ignored local migration reference, not publishable
package code.  It contains many private paths and project-specific runners.
The reusable first-pass core APIs have now been migrated into `src/molsimflow`;
the remaining items are optional trajectory adapters, project-specific
orchestration layers, or publication/reporting scripts.

The optional double-bubble-specific adapters are tracked in:

```python
from molsimflow.workflows.double_bubble_merge import residual_adapter_plan
```

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
  - bridge-water seed escape-direction analysis from explicit position and
    membership tables;
  - H-bond network summaries from explicit edge tables;
  - generic contact graph topology summaries from explicit edge tables;
  - local-environment class summaries and transition matrices from explicit
    sample tables;
  - generic transition-event detection and event-aligned table summaries;
  - bridge liquid-film state, barrier, residence, and coordination table
    summaries;
  - ion-water coupling, lag correlation, state-comparison, and event-aligned
    feature-table summaries;
  - first shared LAMMPS dump reader, PBC geometry helpers, and nearest-time
    alignment helpers;
  - periodic O-H assignment and oxygen species grouping;
  - species transition matrix counting from explicit long-form state tables;
  - water-orientation geometry and explicit sample-table summaries;
  - 1D FES processing and barrier summaries;
  - case scorecards, case-pair deltas, and descriptor correlations.
- plotting/config:
  - generic CSV-driven line, scatter, and heatmap plots;
  - external workflow config helper;
  - generic scheduler template.

## Residual Legacy Areas

### Optional Trajectory Adapters

These are not blockers for the public package core.  Migrate them later only if
we want direct trajectory-to-table commands instead of explicit table inputs.
They are represented in `residual_adapter_plan()`.

- residual `analysis/bridge_water_escape_direction.py` trajectory adapters
  - Case discovery, trajectory-segment selection, seed-position table
    generation, and publication plots around the now-migrated escape-direction
    table API.
  - Suggested targets: adapter in `molsimflow.postprocess.bridge_water_escape`
    or double-bubble workflow code if the case layout remains project-specific.
- residual `analysis/bridge_hbond_network.py` trajectory adapters
  - H-bond detection from trajectories, hydration motif construction,
    MDAnalysis-backed execution, case discovery, and plotting around the
    now-migrated explicit edge-table network summaries.
  - Suggested targets: optional adapter in
    `molsimflow.postprocess.hbond_network`, plotting through generic
    `molsimflow.plotting`.
  - Keep MDAnalysis as an optional dependency for trajectory-backed detection.
- residual `analysis/water_orientation_shell.py` trajectory adapters
  - Atom selection, bubble-center lookup, hydrogen assignment, COLVAR
    alignment, and optional plotting around the now-migrated orientation
    geometry and sample-table summaries.
  - Suggested targets: trajectory adapter into
    `molsimflow.postprocess.water_orientation`, plotting through generic
    `molsimflow.plotting`.

### Workflow-Specific Topology And Local Environment Adapters

The reusable table cores have been migrated.  The remaining work is mostly
sample/edge generation from project-specific trajectories and optional
sensitivity/reporting adapters.

- `analysis/ion_effect_water_topology.py`
  - Contains multiple stages in one large file:
    - bridge atom membership;
    - residual contact graph topology adapters;
    - cycle analysis;
    - residual local water environment sample generation;
    - environment similarity;
    - gap/event-conditioned summaries;
    - topology sensitivity.
  - Suggested targets:
    - `molsimflow.postprocess.bridge_membership`;
    - residual adapters around `molsimflow.postprocess.contact_graph`;
    - residual adapters around `molsimflow.postprocess.local_environment`;
    - `molsimflow.postprocess.sensitivity`.
  - Keep `networkx` optional.
- `analysis/ion_effect_water_topology_stage02.py`
  - Bridge microstate frame table, ion-position table, species-region summaries,
    and water-position QC.
  - Suggested target: split into bridge microstate and region-position helpers.

### Optional Low-Level Trajectory Utilities

The first shared LAMMPS dump and time-alignment helpers are already migrated.
Extend them only when adding a direct trajectory adapter that needs more raw
atom-record outputs.

- extend shared LAMMPS dump readers and time-alignment helpers for:
  - raw trajectory membership outputs needed by
    `analysis/bridge_water_escape_direction.py`;
  - orientation-specific atom records from `analysis/water_orientation_shell.py`;
  - future H-bond/topology trajectory readers.
  - Current first-pass targets:
    `molsimflow.io.lammps_dump` and `molsimflow.postprocess.time_alignment`.

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

Keep `legacy_sources/` local and ignored only while optional adapters are still
being inspected.  If historical provenance is no longer needed:

1. Confirm `git ls-files legacy_sources` returns no tracked files.
2. Archive `legacy_sources/` outside the repository if historical provenance is
   still needed.
3. Remove the local directory with:

```bash
cd /path/to/molsimflow
rm -rf legacy_sources
```

Do not upload `legacy_sources/` to GitHub.
