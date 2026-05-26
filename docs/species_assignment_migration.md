# Species Assignment Migration

## Scope

`molsimflow.postprocess.species_assignment` migrates reusable periodic
oxygen-hydrogen assignment utilities from the legacy species package.

The module provides:

- `assign_hydrogen_to_nearest_oxygen`;
- `classify_oxygen_species_indices`;
- `count_oxygen_species`;
- `OxygenHydrogenAssignment`.

Unlike the legacy helper, the migrated implementation uses NumPy chunking and
does not require SciPy.  It accepts either LAMMPS bounds with shape `(3, 2)` or
box lengths with shape `(3,)`.

## Example

```python
from molsimflow.postprocess.species_assignment import (
    assign_hydrogen_to_nearest_oxygen,
    classify_oxygen_species_indices,
)

assignment = assign_hydrogen_to_nearest_oxygen(
    oxygen_coords=oxygen_xyz,
    hydrogen_coords=hydrogen_xyz,
    bounds_or_lengths=[lx, ly, lz],
    oh_cutoff=1.35,
)

species_indices = classify_oxygen_species_indices(assignment.h_count_per_oxygen)
```

`molsimflow.postprocess.ion_species.find_nearest_oxygen_hydrogens` now delegates
to this shared assignment helper.
