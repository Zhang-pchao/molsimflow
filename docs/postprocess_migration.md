# Post-Processing Migration

## Migrated Workflows

The first migrated post-processing family covers two-bubble nitrogen-cluster
analysis and the first ion-analysis core:

- `molsimflow.postprocess.centroids`
  - reads LAMMPS dump trajectories through MDAnalysis;
  - clusters nitrogen atoms under periodic boundary conditions;
  - writes per-frame bubble centroids and optional ion-distance distributions.
- `molsimflow.postprocess.bubble_surface_distance`
  - reuses the centroid clustering helper;
  - computes centroid distance, surface-to-surface minimum distance, and radial
    gap metrics for two-bubble trajectories;
  - can optionally merge the surface-distance time series with a COLVAR file.
- `molsimflow.postprocess.ion_species`
  - classifies TiO2/surface/solution ion species from atomistic frames;
  - writes classified species XYZ files and per-frame statistics.
- `molsimflow.postprocess.ion_distribution`
  - reads classified species XYZ files;
  - computes relative ion z-distribution summaries and density tables.
- `molsimflow.postprocess.bridge_descriptors`
  - computes bridge-water density proxy tables;
  - computes strict bridge-ion occupancy and net-charge descriptor tables;
  - summarizes descriptors by surface-gap bins and named windows.
- `molsimflow.postprocess.fes_analysis`
  - processes 1D FES curves;
  - writes zeroed/smoothed curve tables and barrier summaries.
- `molsimflow.postprocess.case_comparison`
  - joins case-level descriptor tables through explicit CSV manifests;
  - computes target-minus-reference case deltas;
  - ranks descriptor correlations against a selected scorecard target.

The migrated commands intentionally do not provide case-layout defaults such as
relative trajectory or ion-analysis directories.  Pass every trajectory, data,
ion, and COLVAR input explicitly.

## CLI Examples

```bash
molsimflow postprocess centroids \
  --traj_file run.lammpstrj \
  --data system.data \
  --output bubble_centroids.txt \
  --step_interval 10 \
  --disable_ions
```

```bash
molsimflow postprocess bubble-surface-distance \
  --traj_file run.lammpstrj \
  --data system.data \
  --output bubble_surface_distance.txt \
  --step_interval 10 \
  --surface_fraction 0.8
```

```bash
molsimflow postprocess ion-species \
  --traj run.lammpstrj \
  --data system.data \
  --output-dir ion_analysis_results \
  --step-interval 100
```

```bash
molsimflow postprocess ion-z-distribution \
  --species-statistics ion_analysis_results/species_statistics.txt \
  --h3o-file ion_analysis_results/solution_bulk_h3o.xyz \
  --output-dir ion_z_distribution_results
```

```bash
molsimflow postprocess bridge-water-density \
  --input coalescence_state_table.csv \
  --output-dir bridge_water_descriptors
```

```bash
molsimflow postprocess bridge-ion-occupancy \
  --positions tracked_bridge_ion_positions.csv \
  --gap-table coalescence_state_table.csv \
  --output-dir bridge_ion_descriptors
```

```bash
molsimflow postprocess fes-barriers \
  --curve fes-rew.dat "case A" tio2 \
  --output-dir fes_barrier_results \
  --barrier-window contact:0:5
```

```bash
molsimflow postprocess case-scorecard \
  --cases cases.csv \
  --descriptor-manifest descriptor_manifest.csv \
  --output-dir case_comparison_results \
  --pair caseA:caseB:caseB_minus_caseA \
  --target-column barrier__barrier_kjmol \
  --correlate bridge__bridge_waters,bridge__net_charge
```

Ion-distance inputs are optional and must be supplied explicitly:

```bash
molsimflow postprocess centroids \
  --traj_file run.lammpstrj \
  --output bubble_centroids.txt \
  --h3o_file ion_analysis/solution_bulk_h3o.xyz \
  --bulk_oh_file ion_analysis/solution_bulk_oh.xyz \
  --surface_oh_file ion_analysis/solution_surface_oh.xyz \
  --surface_h_file ion_analysis/tio2_surface_h.xyz \
  --na_file ion_analysis/na_ions.xyz \
  --cl_file ion_analysis/cl_ions.xyz \
  --ions_output ions_analysis
```

## Optional Dependencies

These workflows require the analysis extras at runtime:

- `MDAnalysis` for LAMMPS trajectory reading;
- `matplotlib` for generated plots;
- `numpy` for array operations.

The modules import plotting and MDAnalysis dependencies lazily where possible,
so utility functions can still be imported for testing without opening a
trajectory reader.

## Remaining Cleanup

This migration preserves the legacy algorithms and output formats.  Later
cleanup should focus on extracting shared trajectory readers, replacing
legacy-style logging and comments, and adding small trajectory fixtures for
end-to-end tests.
