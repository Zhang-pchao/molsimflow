# Constant-force morphology analysis

`molsimflow` provides two contract-driven workflows for morphologies that are
not represented well by a single droplet center.

## Water islands

```bash
molsimflow postprocess constant-force-islands \
  --contract islands.json \
  --output island-results
```

The input is one or more restart segments containing water-oxygen IDs and
coordinates. Connectivity uses an O--O cutoff with periodic X/Y and
nonperiodic Z. Persistent island IDs are assigned by one-to-one oxygen-ID
overlap. Qualified one-to-many and many-to-one overlaps are also reported as
geometric split and merge events. Each island row contains its PBC-aware
wrapped and unwrapped center, velocity, retained membership, and exchanged
oxygen counts.

`molecule_exchange.tsv` preserves every oxygen identity whose island owner
changes between consecutive frames. The `exchange_class` column distinguishes
transfer between two persistent island tracks, split or merge lineage
reassignment, and entry or exit when the selected oxygen population changes.
`track_exchange_summary.tsv` aggregates each source-target pair, while
`branch_transport_summary.tsv` reports main-island and size-weighted satellite
velocities, gross and net exchange, and transfer rates. Component ranks in the
identity ledger make main-island versus satellite exchange explicit without a
system-specific atom-ID or size cutoff.

Required contract fields are `schema_version`, `timestep_fs`,
`cluster_cutoff_A`, and `cases`. Each case contains `case_id`, `branch_id`,
`direction`, and `trajectories`. Optional lineage thresholds are
`lineage_overlap_fraction`, `event_overlap_fraction`, and
`event_minimum_overlap_count`.

## Layered film transport

```bash
molsimflow postprocess constant-force-layers \
  --contract layers.json \
  --output layer-results
```

The layer workflow requires water-oxygen coordinates and velocities. It
reports height-resolved velocity, surface-integrated molecular flux, exchange
between height layers, residence episodes, and selected XY Fourier density
modes. Every `case_id` must include one `direction=none` branch; driven layer
velocities are time aligned and baseline subtracted.

The optional `surface_sites` object identifies CH3 or SiOH sites from an
extended XYZ reference and records nearest-site retention and exchange for a
selected water layer. This keeps atom ranges and site definitions in the
contract instead of source code.

Residence episodes that reach a trajectory endpoint are marked as right
censored. Block SEM values are single-trajectory diagnostics rather than
independent-replica uncertainty. Island split/merge and site-exchange labels
are geometric observables and do not define chemical reaction rates.
