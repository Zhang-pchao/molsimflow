# Ion Analysis Migration

## Migrated Core

The reusable ion-analysis core is split into two modules:

- `molsimflow.postprocess.ion_species`
  - classifies TiO2 surface H, solution surface OH, solution surface H2O,
    solution bulk OH, solution bulk H3O, Na, and Cl from atomistic frames;
  - keeps atom type mappings and geometric cutoffs configurable;
  - can read LAMMPS dump trajectories through MDAnalysis and write classified
    multi-frame XYZ files plus `species_statistics.txt`.
- `molsimflow.postprocess.ion_distribution`
  - reads the classified multi-frame XYZ files;
  - reads frame-indexed TiO2 surface-z values from `species_statistics.txt`;
  - computes relative ion z-distribution histograms and summary TSV files.

Large legacy plotting and batch-orchestration scripts are not copied directly.
They should be rebuilt on top of these API functions as smaller plotting and
reporting layers.

## CLI Examples

```bash
molsimflow postprocess ion-species \
  --traj run.lammpstrj \
  --data system.data \
  --output-dir ion_analysis_results \
  --step-interval 100 \
  --atom-style "id type x y z"
```

For non-default LAMMPS atom types, pass the type map explicitly:

```bash
molsimflow postprocess ion-species \
  --traj run.lammpstrj \
  --output-dir ion_analysis_results \
  --type-map 1=H 2=O 4=Na 5=Cl 6=Ti
```

Run z-distribution analysis from explicit classified species files:

```bash
molsimflow postprocess ion-z-distribution \
  --species-statistics ion_analysis_results/species_statistics.txt \
  --h3o-file ion_analysis_results/solution_bulk_h3o.xyz \
  --bulk-oh-file ion_analysis_results/solution_bulk_oh.xyz \
  --surface-oh-file ion_analysis_results/solution_surface_oh.xyz \
  --surface-h-file ion_analysis_results/tio2_surface_h.xyz \
  --na-file ion_analysis_results/na_ions.xyz \
  --cl-file ion_analysis_results/cl_ions.xyz \
  --output-dir ion_z_distribution_results \
  --z-min 15.0 \
  --z-bins 100 \
  --z-range 0 30
```

## Current Assumptions

The first migrated classifier preserves the legacy TiO2/solution split:

- TiO2 oxygen atoms are before the Ti block in atom order;
- solution oxygen atoms are after the Ti block;
- top-surface oxygen atoms are identified by coordination to the top Ti layer.

These assumptions are now documented and parameterized at the cutoff level, but
future work should add selector-based alternatives for systems with different
atom ordering.

## Outputs

`ion-species` writes:

- `species_statistics.txt`;
- `solution_bulk_h3o.xyz`;
- `solution_bulk_oh.xyz`;
- `solution_surface_oh.xyz`;
- `solution_surface_h2o.xyz`;
- `tio2_surface_h.xyz`;
- `na_ions.xyz`;
- `cl_ions.xyz`.

`ion-z-distribution` writes:

- `ion_z_distribution_summary.tsv`;
- `ion_z_density.tsv`.
