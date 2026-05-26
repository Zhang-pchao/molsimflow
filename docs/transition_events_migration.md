# Transition Events Migration

## Scope

`molsimflow.postprocess.events` migrates the reusable core of the legacy
bridge transition-event analysis:

- backward finite-difference derivatives for selected numeric columns;
- connectivity-loss/gain, water-count-drop, and dewetting-jump event detection;
- event-aligned feature profiles;
- event-aligned summary statistics;
- lag correlations between features and target bridge metrics;
- pre/post change-point summaries.

The migrated module intentionally works from one explicit feature CSV.  The
legacy plotting layer and automatic multi-table feature merging are not copied
directly.  They can be rebuilt later on top of shared time-alignment utilities.

## CLI Example

```bash
molsimflow postprocess transition-events \
  --input bridge_water_dewetting.csv \
  --output-dir transition_events \
  --time-column time_ns \
  --event-method hybrid \
  --event-window-before 20 \
  --event-window-after 20 \
  --allow-partial-windows
```

Use explicit thresholds when the percentile defaults are not appropriate:

```bash
molsimflow postprocess transition-events \
  --input transition_feature_table.csv \
  --output-dir transition_events \
  --event-method nw_drop \
  --nw-drop-threshold -1000 \
  --min-event-separation 5
```

## Outputs

The command writes:

- `transition_feature_table.csv`;
- `transition_events.csv`;
- `event_aligned_profiles.csv`;
- `event_aligned_summary.csv`;
- `feature_lag_correlation.csv`;
- `feature_change_point_summary.csv`;
- `state_statistics.csv`.

## Notes

For bridge-water inputs, the command recognizes the common columns written by
`molsimflow.postprocess.bridge_water_dewetting`, including `Nw_bridge`,
`dewet_fraction`, `water_bridge_connected_flag`, and `d3d_all`.
