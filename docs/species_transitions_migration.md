# Species Transitions Migration

## Scope

`molsimflow.postprocess.transitions` migrates the reusable core of the legacy
species transition matrix utility into a generic long-table API.

The migrated implementation is intentionally input-explicit:

- it reads a CSV with one row per `(frame, entity_id, species)` assignment;
- the persistent entity id is usually an oxygen atom index, but it can be any
  stable molecule or atom identifier;
- it counts adjacent-frame transitions for entities present in both frames;
- it writes long-form transition counts, row-normalized probabilities,
  matched-transition details, species-state summaries, and run statistics.

The package does not scan project-specific directory trees or assume legacy
species names.  Legacy names such as `solution_bulk_oh`,
`solution_surface_oh`, and `solution_surface_h2o` can be passed as an explicit
species order.

## Python API

```python
from molsimflow.postprocess.transitions import (
    build_species_transition_matrix,
    load_species_state_rows,
    parse_species_order,
)

rows = load_species_state_rows(
    "species_states.csv",
    entity_column="oxygen_index",
    time_column="time_ns",
)

result = build_species_transition_matrix(
    rows,
    species_order=parse_species_order([
        "solution_bulk_oh,solution_surface_oh,solution_surface_h2o",
    ]),
)
```

## CLI

```bash
molsimflow postprocess species-transitions \
  --input species_states.csv \
  --output-dir species_transitions \
  --entity-column oxygen_index \
  --time-column time_ns \
  --species-order solution_bulk_oh,solution_surface_oh,solution_surface_h2o
```

Generated files:

- `species_state_table.csv`;
- `species_transition_counts.csv`;
- `species_transition_probabilities.csv`;
- `species_transition_details.csv`;
- `species_state_summary.csv`;
- `state_statistics.csv`.

## Migration Notes

The legacy script mixed three concerns: case-directory discovery, transition
counting, and plotting.  Only the transition-counting algorithm is migrated
here.  Directory discovery and publication plotting should stay outside the
generic package or be rebuilt later as explicit workflow/case layers.
