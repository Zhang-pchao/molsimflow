# Legacy contact-state table inventory

`molsimflow.postprocess.contact_state_inventory` is a table reducer for existing
geometry and contact outputs. It does not decode a trajectory, calculate a new
chemical observable, rerun historical preparation, infer energy, or fit kinetics.

## Public API

```python
from molsimflow.postprocess.contact_state_inventory import (
    ContactStateCase, SourceEpoch, inventory_contact_states,
)

config = ContactStateCase(
    name="late_case_demo", case="surface", tier="E",
    dt_ps=0.0005, origin_step=30_000_000, origin_time_ps=0.0,
    epochs=(SourceEpoch("original", 30_000_000, 36_000_000,
                        "/declared/archive/source.dump"),),
    periodic=(True, True, True),
    geometry_dt_ps=1.0, contact_dt_ps=10.0,
    distance_shells=(3.5, 4.0, 4.5),
)
result = inventory_contact_states(geometry_rows, contact_rows, config=config)
```

`SourceEpoch` pins a segment name, inclusive step range and exact source path.
`dt_ps`, `origin_step` and `origin_time_ps` define the physical clock. Boundary
flags are explicit metadata only: this reducer performs no coordinate operation
and cannot validate those flags from tables. Supply separately verified source
metadata. The example source path above is illustrative, not a runtime default.

## Validation and join

Geometry rows must include material/tier identity, source segment, step, time,
source path and frame index, a canonical frame SHA256, and a historically true
`source_frame_verified` flag. This checks the recorded provenance; it does not
recompute any raw hash. Duplicate physical times or steps fail. The explicit
step-to-time map and sample grid are checked with decimal arithmetic.

The five states `distant`, `unsupported_or_ambiguous`, `water_separated`,
`direct_contact_candidate` and `mixed` each require nonnegative integer cell
counts and corresponding areas. Counts must sum to `cells`; each area must
match its count times `cell_area_A2`, with tolerance only for serialized floating
point area values. Missing fields, nonfinite numbers or closure failures raise
`ValueError`. The entire supplied geometry table is validated.

Contact rows are selected by explicit `name`. Their `(time, segment, shell)`
keys must be unique; each selected time/segment must have all three configured
shells. The 4-A support Boolean must agree with its supported-patch count, and
its molecule count cannot exceed the matched main-gas inventory. Joins require
an exact physical time and epoch; no nearest-time fallback exists. Unselected
contact cases are not claimed as validated.

The legacy contact table has no raw-frame hash. Every joined result therefore
carries `LEGACY_EXACT_TIME_EPOCH_INDEX_NOT_RAW_HASH_VALIDATED`. Copying the
geometry hash into a row must never be interpreted as independently proving
that the historical contact row was calculated from that raw frame.

## Results and interpretation

- `geometry_records`: validated original geometry rows, sorted by physical time.
- `joined`: geometry rows at observed contact times, retaining source identity
  and all original geometry fields. `contact_shells` retains each original
  shell record under keys such as `"3.5"`, `"4"` and `"4.5"`.
- `summary`: separate geometry/contact sample counts, observed 4-A contact
  frames, grid-water-separated-positive frames and their co-occurrence on the
  joined grid. `geometry_water_separated_frames` uses the denser geometry grid.
- `episodes`: contact and absence episodes composed using the existing
  `surface_contact_process.sampled_episodes`; source changes and missing whole
  contact times censor runs. Missing times are never imputed as noncontact.
- `representatives`: one deterministic upper time-median observation for each
  existing joint category, with the complete joined `input_record`. Absent
  categories produce no representative. These are observations, not qualified
  wet/dry basins or unbiased preparation seeds.

`water_separated_cells > 0` and a 4-A molecular contact Boolean are different
observables. Co-occurrence means both were observed in the same frame, not that
they occur in the same patch. `contact_water_O` is the whole-surface 0-3-A oxygen
inventory in these legacy tables; it must not be relabeled bubble-bottom water.
Counts and fractions describe sampled frames. They are not independent event
rates, equilibrium probabilities, free energies or causal mechanism tests.

Sparse contact records can omit whole time points; those omissions remain gaps
and censor episodes. The reducer does not establish that all requested frames
were supplied: the project must compare source registries and declared frame
counts to the result before accepting a complete inventory.
