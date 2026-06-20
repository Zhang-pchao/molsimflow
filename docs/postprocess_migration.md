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
- `molsimflow.postprocess.coalescence_state`
  - reads PLUMED-style COLVAR tables and optional bubble-evolution tables;
  - assigns provisional two-bubble states and surface-gap estimates;
  - writes state tables, state summaries, and CV-binned state probabilities.
- `molsimflow.postprocess.ion_species`
  - classifies TiO2/surface/solution ion species from atomistic frames;
  - writes classified species XYZ files and per-frame statistics.
- `molsimflow.postprocess.species_assignment`
  - assigns hydrogens to nearest oxygens under orthorhombic PBC;
  - groups oxygen indices by assigned hydrogen count for OH/H2O/H3O-style
    species classification.
- `molsimflow.postprocess.transitions`
  - reads explicit long-form species-state tables;
  - counts adjacent-frame transitions for stable entity ids;
  - writes transition-count matrices, row-normalized probabilities, matched
    transition details, species summaries, and run statistics.
- `molsimflow.postprocess.water_orientation`
  - computes water-orientation geometry in a two-center reference frame;
  - summarizes explicit water-orientation sample tables by frame, radial
    distance, bridge-axis/radial bins, CV bins, and angular distributions.
- `molsimflow.postprocess.ion_distribution`
  - reads classified species XYZ files;
  - computes relative ion z-distribution summaries and density tables.
- `molsimflow.postprocess.bridge_descriptors`
  - computes bridge-water density proxy tables;
  - computes strict bridge-ion occupancy and net-charge descriptor tables;
  - summarizes descriptors by surface-gap bins and named windows.
- `molsimflow.postprocess.bridge_water_dewetting`
  - reads selected atoms from LAMMPS dump frames;
  - resolves bubble atom groups from explicit expressions or PLUMED
    `bubA_all`/`bubB_all` labels;
  - computes bridge-cylinder water counts, dewetting fractions, water-cluster
    connectivity, and `d3d_all`-binned summaries.
- `molsimflow.postprocess.bridge_water_dynamics`
  - reads explicit bridge-water trace-metrics CSV files or manifests;
  - computes entry/exit flux, turnover, replacement, and drainage proxies;
  - computes seed-water retention, monotonic survival, and exit-event proxy
    summaries versus surface gap.
- `molsimflow.postprocess.bridge_water_escape`
  - reads explicit seed-water position and bridge-membership tables;
  - classifies retained/exited seed waters and escape directions;
  - writes per-seed escape events, direction summaries, and optional gap-bin
    summaries.
- `molsimflow.postprocess.hbond_network`
  - reads explicit H-bond edge tables;
  - computes per-frame graph connectivity, bridge-spanning flags, H-bond type
    counts, lifetime summaries, and gap-binned network summaries.
- `molsimflow.postprocess.contact_graph`
  - reads explicit contact edge tables;
  - computes generic graph topology metrics, role-mediated edge fractions,
    articulation-node proxies, cycle rank, bridge-spanning flags, and gap-bin
    summaries.
- `molsimflow.postprocess.local_environment`
  - reads explicit local-environment sample tables;
  - computes frame/class summaries for environment labels and numeric features;
  - reuses the transition-matrix core for persistent-entity environment
    transitions.
- `molsimflow.postprocess.events`
  - detects connectivity, water-count drop, and dewetting-jump events from a
    feature CSV;
  - writes event-aligned profiles, aligned summary statistics, lag
    correlations, and pre/post change-point summaries.
- `molsimflow.postprocess.bridge_film`
  - classifies bridge liquid-film states from frame-level composition counts;
  - summarizes barrier-top film states, residence episodes, and ion-water
    coordination samples from explicit tables.
- `molsimflow.postprocess.coupling`
  - computes predictor-target ion/water coupling from feature tables;
  - writes lag correlations, low/high state comparisons, and optional
    event-aligned predictor/target summaries.
- `molsimflow.postprocess.fes_analysis`
  - processes 1D FES curves;
  - writes zeroed/smoothed curve tables and barrier summaries;
  - processes regular 2D FES grids into zeroed/smoothed long-form tables, metadata, and optional contour plots;
  - analyzes final/block/cumulative FES profiles for fixed-window Delta-F convergence from explicit CSV manifests.
- `molsimflow.postprocess.case_comparison`
  - joins case-level descriptor tables through explicit CSV manifests;
  - computes target-minus-reference case deltas;
  - ranks descriptor correlations against a selected scorecard target.
- `molsimflow.postprocess.plumed_cv_diagnostics`
  - reads PLUMED COLVAR/HILLS and generated PLUMED definitions;
  - summarizes CV ranges, duplicate headers, bias columns, and physical checks;
  - optionally validates simple geometry against selected LAMMPS dump frames.
- `molsimflow.postprocess.silica_surface`
  - reads extended XYZ silica-surface models;
  - infers fixed surface atoms from explicit counts or requested-count metadata;
  - summarizes CH3/OH terminations, side splits, densities, warnings, and comparisons.
- `molsimflow.postprocess.particle_flotation`
  - analyzes silica-particle and N2 flotation trajectories from LAMMPS dumps;
  - computes particle lift, slab gap, N2 contact/coverage, radial density, and optional force/velocity summaries.
- `molsimflow.postprocess.gas_connectivity`
  - computes gas COM contact-graph metrics under orthorhombic PBC;
  - summarizes precomputed gas-connectivity frame tables around a radius-sum reference with distance-bin, window, threshold, and transition tables.

The migrated commands intentionally do not provide case-layout defaults such as
relative trajectory or ion-analysis directories.  Pass every trajectory, data,
ion, and COLVAR input explicitly.

## Shared Utilities

The first shared trajectory helpers have been extracted for later
trajectory-heavy migrations:

- `molsimflow.io.lammps_dump` for selected-atom LAMMPS dump frame reading and
  orthorhombic PBC geometry helpers;
- `molsimflow.postprocess.time_alignment` for nearest-time row matching and
  timestep-to-time scale inference.

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
molsimflow postprocess coalescence-state \
  --colvar COLVAR \
  --colvar-post COLVAR_POST \
  --bubble-evolution data_bubble_evolution.txt \
  --output-dir coalescence_state
```

```bash
molsimflow postprocess plumed-cv-diagnostics \
  --run-dir run \
  --output-dir cv_diagnostics \
  --cv-kind auto
```

For coupled CVs, request one or more phase-plane maps. Each pair produces a
total-bias-colored map and a time-colored map:

```bash
molsimflow postprocess plumed-cv-diagnostics \
  --run-dir run --output-dir cv_diagnostics \
  --phase-plane sum_cn.sum nads_total
```

```bash
molsimflow postprocess silica-surface \
  --case model:model.xyz \
  --output-dir silica_surface_summary \
  --no-plots
```

```bash
molsimflow postprocess particle-flotation \
  --trajectory dump.lammpstrj \
  --model-summary model_summary.json \
  --output-dir particle_flotation
```

```bash
molsimflow postprocess gas-contact-summary \
  --input-table gas_connectivity_frame_table.csv \
  --output-dir gas_contact_summary \
  --radius-sum-A 38 \
  --d-range 20 60 \
  --d-bin-width-A 2
```

```bash
molsimflow postprocess fes2d-grid \
  --fes-file fes-rew-2d.dat \
  --output-dir fes2d_grid \
  --x-range 20 52 \
  --y-range 50 380 \
  --prefix d3d_bridge
```

```bash
molsimflow postprocess fes-convergence \
  --manifest fes_convergence_manifest.csv \
  --output-dir fes_convergence \
  --window-low 20 \
  --window-high 52 \
  --infer-blocks
```

`fes-convergence` manifests require `path` and may include `label`,
`group`, `dataset_key`, `series`, `chemistry`, `block_paths`,
`cumulative_paths`, and `cumulative_dir`.  List-valued path columns accept
semicolon, comma, or pipe separators; relative paths resolve against the
manifest file directory.

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
molsimflow postprocess bridge-water-dewetting \
  --dump dump.lammpstrj \
  --plumed in.plumed \
  --water-oxygen-atoms 1201-9000:3 \
  --colvar COLVAR \
  --output-dir bridge_water_dewetting
```

```bash
molsimflow postprocess bridge-water-flux \
  --manifest bridge_water_trace_manifest.csv \
  --output-dir bridge_water_flux
```

```bash
molsimflow postprocess bridge-seed-survival \
  --manifest bridge_water_trace_manifest.csv \
  --output-dir seed_water_survival
```

```bash
molsimflow postprocess bridge-water-escape \
  --input seed_positions.csv \
  --output-dir bridge_water_escape \
  --case-label caseA \
  --in-bridge-column in_bridge_region \
  --gap-column surface_gap_A
```

```bash
molsimflow postprocess hbond-network \
  --input hbond_edges.csv \
  --output-dir hbond_network \
  --case-label caseA \
  --donor-s-column donor_s_A \
  --acceptor-s-column acceptor_s_A \
  --gap-column surface_gap_A
```

```bash
molsimflow postprocess contact-graph \
  --input contact_edges.csv \
  --output-dir contact_graph \
  --case-label caseA \
  --source-s-column source_s_A \
  --target-s-column target_s_A \
  --gap-column surface_gap_A
```

```bash
molsimflow postprocess local-environment \
  --input local_environment_samples.csv \
  --output-dir local_environment \
  --time-column time_ns \
  --class-order tetrahedral,interfacial,distorted \
  --feature-column q,lsi
```

```bash
molsimflow postprocess transition-events \
  --input bridge_water_dewetting.csv \
  --output-dir transition_events
```

```bash
molsimflow postprocess species-transitions \
  --input species_states.csv \
  --output-dir species_transitions \
  --entity-column oxygen_index \
  --time-column time_ns \
  --species-order solution_bulk_oh,solution_surface_oh,solution_surface_h2o
```

```bash
molsimflow postprocess water-orientation-summary \
  --input water_orientation_samples.csv \
  --output-dir water_orientation_summary \
  --rho-bins 30 \
  --rho-max 12 \
  --s-bins 40 \
  --s-min -20 \
  --s-max 20
```

```bash
molsimflow postprocess bridge-film \
  --frame-table bridge_liquid_film_frame_metrics.csv \
  --output-dir bridge_film
```

```bash
molsimflow postprocess ion-water-coupling \
  --feature-table transition_feature_table.csv \
  --output-dir ion_water_coupling
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

Direct OPES-style reweighting can combine restart segments and produce 1D and
2D projections:

```bash
molsimflow postprocess fes-reweight \
  --run-dir run \
  --colvar segment_1/COLVAR \
  --colvar segment_2/COLVAR \
  --hills segment_1/HILLS \
  --hills segment_2/HILLS \
  --cv sum_cn.sum \
  --pair sum_cn.sum foot_total \
  --bias opes.bias \
  --bias opes_e.bias
```

Printed CV traces from multiple cases can be compared with reconstructed
segment times when `in.plumed`, `lmp.out`, and a trajectory dump are available:

```bash
molsimflow postprocess sphere-cv-compare \
  --case sphere15=case15 \
  --case sphere17=case17 \
  --output-dir sphere_cv_compare \
  --cv foot_total \
  --cv sum_cn.sum
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
