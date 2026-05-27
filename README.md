# molsimflow

`molsimflow` is a reusable Python package for molecular-simulation workflows:

- structure preparation utilities for slab, bubble, and electrolyte systems;
- file conversion helpers for extended XYZ, LAMMPS data files, and selected
  LAMMPS dump frames;
- PLUMED input generation for double-bubble enhanced-sampling cases;
- a migration target for existing MD post-processing and visualization workflows;
- CSV-driven plotting helpers for reusable analysis figures.

The package is being migrated from project-specific research scripts.  The new
code avoids user-specific absolute paths and exposes paths, environments, and
case settings as command-line arguments or configuration values.

## Current Status

This repository is an engineering scaffold plus the first sanitized utilities.
Legacy sources may be kept locally under `legacy_sources/` for migration
reference, but that directory is intentionally ignored by Git.  Only sanitized
package code, reusable scripts, tests, and necessary documentation should be
tracked.

Implemented first-pass commands:

```bash
molsimflow structure add-extxyz-pbc --poscar POSCAR --xyz packed.xyz --output model.xyz
molsimflow structure extxyz-to-lammps-data --xyz model.xyz --output model_atomic.data
molsimflow structure equal-volume-radius --radii 19 14
molsimflow structure slab-double-bubble --interface-structure interface.xyz --molecule-dir molecules --output-dir case
molsimflow structure slab-double-bubble --interface-structure interface.xyz --molecule-dir molecules --output-dir case --run-packmol
molsimflow structure tio2-double-bubble --bulk-structure bulk.cif --molecule-dir molecules --output-dir case
molsimflow plumed double-bubble --data model_atomic.data --packmol packmol.in --build-py build.py --output in.plumed
molsimflow plumed n2-com --structure model.xyz --with-surface --surface-element Si --surface-stride 10 --output in.plumed
molsimflow plumed n2-com --structure model.xyz --with-surface --surface-element Si --surface-stride 10 --bias-mode opes --output in_surface_opes.plumed
molsimflow postprocess centroids --traj_file run.lammpstrj --output bubble_centroids.txt --disable_ions
molsimflow postprocess bubble-surface-distance --traj_file run.lammpstrj --output bubble_surface_distance.txt
molsimflow postprocess coalescence-state --colvar COLVAR --output-dir coalescence_state
molsimflow postprocess ion-species --traj run.lammpstrj --output-dir ion_analysis_results
molsimflow postprocess ion-z-distribution --species-statistics ion_analysis_results/species_statistics.txt --h3o-file ion_analysis_results/solution_bulk_h3o.xyz
molsimflow postprocess bridge-water-density --input coalescence_state_table.csv --output-dir bridge_water_descriptors
molsimflow postprocess bridge-water-dewetting --dump dump.lammpstrj --water-oxygen-atoms 1201-9000:3 --plumed in.plumed --output-dir bridge_water_dewetting
molsimflow postprocess bridge-water-flux --manifest bridge_water_trace_manifest.csv --output-dir bridge_water_flux
molsimflow postprocess bridge-seed-survival --manifest bridge_water_trace_manifest.csv --output-dir seed_water_survival
molsimflow postprocess transition-events --input bridge_water_dewetting.csv --output-dir transition_events
molsimflow postprocess bridge-film --frame-table bridge_liquid_film_frame_metrics.csv --output-dir bridge_film
molsimflow postprocess ion-water-coupling --feature-table transition_feature_table.csv --output-dir ion_water_coupling
molsimflow postprocess bridge-ion-occupancy --positions tracked_bridge_ion_positions.csv --gap-table coalescence_state_table.csv --output-dir bridge_ion_descriptors
molsimflow postprocess fes-barriers --curve fes-rew.dat "case A" tio2 --output-dir fes_barrier_results
molsimflow postprocess case-scorecard --cases cases.csv --descriptor-manifest descriptor_manifest.csv --output-dir case_comparison_results
molsimflow plot line --input fes_processed_curves.csv --x-column cv --y-column free_energy_smooth_zeroed_kj_mol --group-column label --output fes_curves.png
molsimflow plot scatter --input case_scorecard.csv --x-column bridge__bridge_waters --y-column barrier__barrier_kjmol --label-column case_label --fit-line --output descriptor_vs_barrier.png
molsimflow plot heatmap --input case_descriptor_delta.csv --row-column descriptor --column-column case_pair_label --value-column delta_target_minus_reference --output descriptor_delta_heatmap.png
molsimflow config summary --config workflow.ini
molsimflow config env --config workflow.ini --section scheduler --section structure
```

For direct source-tree execution before installation:

```bash
PYTHONPATH=src python -m molsimflow.cli --help
```

## Installation

From a checkout:

```bash
python -m pip install -e .
```

Install optional runtime groups only when needed:

```bash
python -m pip install -e ".[analysis,structure,dev]"
```

## Verification

Before publishing or committing a migration batch:

```bash
python scripts/audit_public_repo.py
python -m compileall -q src scripts tests
PYTHONPATH=src python -m pytest -q
```

`pytest` is part of the `dev` extra.  If it is not available in a cluster
environment, run the documented smoke commands for the workflow being migrated.

## Layout

```text
src/molsimflow/
  config/        External workflow configuration helpers.
  io/             File readers, writers, converters, and trajectory helpers.
  structure/      Geometry helpers and future structure builders.
  plumed/         PLUMED generators.
  postprocess/    Migrated MD analysis workflows.
  plotting/       CSV-driven plotting helpers.
  workflows/      Project workflow composition namespaces.
docs/             Migration plan and publicization notes.
templates/        Generic scheduler templates.
tests/            Small unit tests for reusable utilities.
legacy_sources/   Local migration snapshot, ignored by Git.
```

## Migration Rule

New package code should be public-facing by default:

- no hardcoded personal absolute input or output paths;
- no mandatory conda/module environment strings inside Python logic;
- no hidden assumptions about one case directory layout unless documented;
- all code comments, docstrings, CLI help, and commit messages in English;
- project-specific examples belong in `examples/` or docs, not in library defaults;
- generated outputs, plots, scheduler logs, and backup files do not belong in Git.

Run the public-readiness audit before committing:

```bash
python scripts/audit_public_repo.py
```
