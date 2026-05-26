# Bridge Film Migration

## Scope

`molsimflow.postprocess.bridge_film` migrates the reusable table layer from the
legacy bridge liquid-film stability workflow:

- bridge film-state classification from per-frame composition counts;
- barrier-top frame selection from transition events, CV windows, or dewetting
  quantiles;
- state-distribution summaries for all frames, barrier-top frames, and
  non-barrier frames;
- residence episodes from long-form `species,atom_id,frame/time,in_bridge`
  membership tables;
- coordination summaries from long-form `species,coordination` samples.

The raw trajectory extraction layer is not copied into this module.  It should
be rebuilt later on top of shared LAMMPS dump and species-assignment APIs.

## CLI Example

```bash
molsimflow postprocess bridge-film \
  --frame-table bridge_liquid_film_frame_metrics.csv \
  --transition-events transition_events.csv \
  --residence-membership bridge_membership.csv \
  --coordination-samples ion_water_coordination.csv \
  --output-dir bridge_film
```

The required frame table should contain bridge composition columns such as
`N_oxygen_bridge_total`, `N_water_bridge`, `N_OH_bridge`, `N_H3O_bridge`,
`N_Na_bridge`, and `N_Cl_bridge`.  `Nw_bridge` is accepted as a fallback water
count for frame tables produced by bridge-water dewetting.

## Outputs

The command writes:

- `bridge_film_frame_table.csv`;
- `bridge_film_state_summary.csv`;
- `bridge_film_barrier_top_summary.csv`;
- `bridge_film_residence_events.csv`;
- `bridge_film_residence_summary.csv`;
- `bridge_film_coordination_summary.csv`;
- `state_statistics.csv`.
