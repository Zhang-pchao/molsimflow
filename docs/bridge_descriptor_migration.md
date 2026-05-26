# Bridge Descriptor Migration

## Migrated Core

The first bridge-descriptor migration adds `molsimflow.postprocess.bridge_descriptors`.
It provides reusable, path-explicit APIs for:

- defining a bridge cylinder and testing whether coordinates fall in the bridge
  region;
- computing bridge-water count and density proxies from state/metrics CSV files;
- computing strict bridge-ion occupancy, cation/anion counts, net charge, and
  charge density from tracked ion positions;
- summarizing frame tables by fixed-width surface-gap bins and named gap
  windows.

This replaces the reusable core of several legacy workflows without copying
their plotting, case discovery, or manuscript-report layers.

## CLI Examples

Bridge-water density from a coalescence-state or water-metrics table:

```bash
molsimflow postprocess bridge-water-density \
  --input coalescence_state_table.csv \
  --output-dir bridge_water_descriptors \
  --case-label caseA \
  --bridge-radius-A 8.0 \
  --bridge-length-A 20.0 \
  --gap-column surface_gap_estimate_A \
  --water-count-column bridge_cyl_env.sum
```

Strict bridge-ion occupancy and charge from tracked ion positions:

```bash
molsimflow postprocess bridge-ion-occupancy \
  --positions tracked_bridge_ion_positions.csv \
  --gap-table coalescence_state_table.csv \
  --output-dir bridge_ion_descriptors \
  --case-label caseA \
  --bridge-radius-A 8.0 \
  --bridge-length-A 20.0
```

The ion command uses `in_bridge_region` when present.  If no explicit species
column is given, it tries `current_trace_species`, `current_trace_species_label`,
`species`, and `ion_species` in that order.

## Outputs

`bridge-water-density` writes:

- `bridge_water_density_frame_table.csv`;
- `bridge_water_density_binned.csv`;
- `bridge_water_density_window_summary.csv`.

`bridge-ion-occupancy` writes:

- `bridge_ion_occupancy_frame_table.csv`;
- `bridge_ion_occupancy_binned.csv`;
- `bridge_ion_occupancy_window_summary.csv`.

## Current Scope

The module intentionally produces tables only.  Figure assembly, case-comparison
reports, event-aligned bridge-water drainage, H-bond networks, and ion-water
coupling synthesis should be migrated as separate layers that consume these
tables.
