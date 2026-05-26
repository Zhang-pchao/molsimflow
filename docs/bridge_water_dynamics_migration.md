# Bridge-Water Dynamics Migration

## Scope

`molsimflow.postprocess.bridge_water_dynamics` migrates the reusable table
algorithms from two legacy workflows:

- bridge-water entry/exit flux versus surface gap;
- seed bridge-water retention and survival versus surface gap.

The migrated module does not discover private case directories and does not
write publication figures.  It takes explicit trace-metrics CSV files and
optional coalescence-state tables, then writes reusable CSV summaries.

## Inputs

For one case, pass a trace metrics table directly:

```bash
molsimflow postprocess bridge-water-flux \
  --trace-metrics bridge_water_trace_metrics.csv \
  --state-table coalescence_state_table.csv \
  --case-label caseA \
  --output-dir bridge_water_flux
```

For multiple cases, use a manifest:

```csv
case_label,trace_metrics,state_table
caseA,cases/caseA/bridge_water_trace_metrics.csv,cases/caseA/coalescence_state_table.csv
caseB,cases/caseB/bridge_water_trace_metrics.csv,cases/caseB/coalescence_state_table.csv
```

Relative paths are resolved from the manifest location.

## Flux Command

```bash
molsimflow postprocess bridge-water-flux \
  --manifest bridge_water_trace_manifest.csv \
  --output-dir bridge_water_flux \
  --gap-source coalescence \
  --gap-bin-width-A 2.0 \
  --min-bin-count 5
```

Outputs:

- `bridge_water_flux_frame_table.csv`;
- `bridge_water_flux_binned.csv`;
- `bridge_water_flux_window_summary.csv`;
- `bridge_water_dynamics_inputs.csv`.

The flux command computes first-time entry, inferred exit, turnover,
replacement, drainage, and rate proxies from aggregate trace metrics.

## Seed-Survival Command

```bash
molsimflow postprocess bridge-seed-survival \
  --manifest bridge_water_trace_manifest.csv \
  --output-dir seed_water_survival \
  --gap-source coalescence
```

Outputs:

- `seed_water_survival_frame_table.csv`;
- `seed_water_survival_binned.csv`;
- `seed_water_survival_window_summary.csv`;
- `seed_water_exit_proxy_events.csv`;
- `seed_water_survival_inputs.csv`.

The seed-survival command computes instantaneous seed retention, a monotonic
survival proxy, aggregate seed-loss counts, and gap-conditioned summaries.

## Remaining Legacy Work

The trajectory-backed escape-direction workflow is intentionally not included
in this first migration.  It still depends on raw trajectory segments and legacy
bridge-water trace visualization helpers, so it should be migrated separately
after shared trajectory readers and trace-membership utilities are extracted.
