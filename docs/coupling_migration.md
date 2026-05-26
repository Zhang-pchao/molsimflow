# Ion-Water Coupling Migration

## Scope

`molsimflow.postprocess.coupling` migrates the reusable table core from the
legacy bridge ion-water coupling workflow:

- predictor-target coupling statistics;
- integer-lag correlations;
- low/high state comparisons against a selected target column;
- optional event-aligned predictor and target profiles when transition events
  are supplied.

The migrated module expects a single explicit feature CSV.  Legacy multi-input
auto-merging, plotting, and trajectory fallback modes are intentionally not
copied directly.

## CLI Example

```bash
molsimflow postprocess ion-water-coupling \
  --feature-table transition_feature_table.csv \
  --transition-events transition_events.csv \
  --predictor-column n_bridge_na \
  --predictor-column n_bridge_cl \
  --target-column Nw_bridge \
  --target-column dewet_fraction \
  --output-dir ion_water_coupling
```

If predictor columns are not supplied, numeric columns are inferred after
excluding common frame, time, event, and bridge-water target columns.

## Outputs

The command writes:

- `coupling_feature_table.csv`;
- `ion_water_coupling.csv`;
- `ion_water_lag_correlation.csv`;
- `ion_water_state_comparison.csv`;
- `event_aligned_ion_water_profiles.csv`;
- `event_aligned_ion_water_summary.csv`;
- `state_statistics.csv`.
