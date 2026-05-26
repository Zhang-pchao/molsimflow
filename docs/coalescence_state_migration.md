# Coalescence State Migration

## Scope

`molsimflow.postprocess.coalescence_state` assigns provisional two-bubble
coalescence states from lightweight tables:

- a PLUMED-style COLVAR table with `time` and `d3d_all`;
- an optional secondary COLVAR table with cluster counters such as `n2A_num`;
- an optional bubble-evolution table with `Bubble1Size` and `Bubble2Size`.

The migrated implementation is table-oriented and does not import legacy
modules, private case paths, pandas, or plotting code.

## Command

```bash
molsimflow postprocess coalescence-state \
  --colvar COLVAR \
  --colvar-post COLVAR_POST \
  --bubble-evolution data_bubble_evolution.txt \
  --output-dir coalescence_state \
  --colvar-time-unit ps \
  --nominal-radius-A 19 \
  --sample-interval-ns 0.001
```

Outputs:

- `coalescence_state_table.csv`;
- `coalescence_state_summary.csv`;
- `coalescence_state_by_d3d_all.csv`;
- `state_statistics.txt`.

## State Logic

The default surface-gap estimate is:

```text
surface_gap_estimate_A = d3d_all - 2 * nominal_radius_A
```

Rows are assigned to:

- `separated`: both bubble-size estimates remain large and the surface gap is
  above the close-gap threshold;
- `transition_like`: both bubbles remain large but the surface gap is near or
  below the close-gap threshold, or size observables are intermediate;
- `merged_like`: one dominant cluster contains most of the initial total and
  the minor cluster is small;
- `ambiguous`: required observables are missing or too short-lived after the
  persistence filter.

All thresholds are CLI/configurable.

## Workflow Role

This state table is the intended upstream input for later bridge-water dynamics,
H-bond network, event-aligned profile, and ion-water coupling migrations.
