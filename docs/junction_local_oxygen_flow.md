# Junction-local oxygen flow

`molsimflow.postprocess.junction_local_oxygen_flow.oxygen_regions` adapts the existing local-flow region statistics to explicit periodic XYZ geometry and two slab faces. Original terminal-site labels assign each original-solvent oxygen to its nearest material face. Front/rear windows use the main gas cluster q05/q95 X projections; the interior window excludes both edge windows. All regions use a common transverse projection and outward signed height above the nearest site.

Inputs are selected oxygen coordinates/velocities, main gas molecular centers, surface-site coordinates/labels and orthorhombic bounds. Velocities are returned in m/s from A/ps. Small populations return missing velocities with `INSUFFICIENT_OCCUPANCY`, not zero flow. Synthetic tests cover full-box translation, opposite-face exclusion, velocity units and empty-region behavior.

This is an original-oxygen population observable. It does not classify H2O, infer a connected water film, return molecular water temperature or establish detachment. Broad periodic projections are rejected for separate topology review. Regions are geometric sampling windows, not exact continuum bubble edges. Time averaging and relative bubble velocity must preserve trajectory identity epochs and report occupancy; frame standard deviations are not independent-sample errors.
