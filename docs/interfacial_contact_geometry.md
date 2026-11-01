# Site-owned conformal contact neighborhoods

`postprocess.interfacial_contact_geometry` supplies an orthorhombic periodic
surface chart, a contact mask, signed cell distances, local orientation weights,
and exact symmetric endpoint occupation bookkeeping. It requires SciPy.

Supply one ordered face of prepared terminal sites to `surface_grid`. Each
cell retains its prepared nearest-site owner. Supply live sites and selected
main-cluster gas atoms to `contact_geometry`; contact is a 3D periodic distance
threshold, mapped to the fixed material chart. Water is not used to define the
footprint. Empty or full contact has no boundary and cannot be evaluated as a
zero occupation neighborhood. Concavities, holes and disconnected patches are
retained. This discrete contact map is not a continuum liquid-vapor interface.

Signed distance measures opposite-label **cell-center** distance, negative
inside. It has grid-scale geometric error. Normal smoothing changes orientation
only, never contact membership. `orientation_weights` partitions front (+X),
rear (-X), flank and unresolved orientations. `region_weights` evaluates a
half-open signed-distance band and local terminal-relative height band. These
are geometric conventions, not hydration-layer boundaries. Site ownership and
heights must be examined for the intended rough surface; compare grid spacing,
contact thresholds, bands and frozen/live heights before interpretation.

For identical ordered particle populations, evaluate four memberships:
`M0(X0)`, `M0(X1)`, `M1(X0)`, `M1(X1)`. The endpoint budget returns

    particle = [(N01-N00) + (N11-N10)] / 2
    geometry = [(N10-N00) + (N11-N01)] / 2
    delta = N11-N00 = particle + geometry.

Geometry includes changes in footprint, local height and orientation weights.
Counts alone cannot distinguish exchange at constant occupation; retain the
original particle identities to measure saved-endpoint membership renewal.
The decomposition is a symmetric endpoint convention, not a unique continuous
flux, causal drainage mechanism or force balance. Recrossings between saved
states remain unresolved. A moving region is never evidence that its particles
are moving with it. Velocities and independent fixed-region comparisons may
help interpret the result but do not remove temporal aliasing.

Tests cover both periodic seams, common coordinate translation, holes and
disconnected masks, grid refinement, contact loss, local height, weighted
closure and separate stationary-particle/stationary-geometry limits.
