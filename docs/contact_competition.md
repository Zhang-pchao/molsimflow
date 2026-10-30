# Prospective contact replacement and height-resolved velocity

The prescribed_contact_shift helper conserves the area of a periodic Boolean
XY footprint under integer or fractional X shifts. It returns separate gain/loss
fractions without filling holes or removing components. A hypothetical footprint
shift is not an atomic displacement, force, or energy calculation.

site_weighted_load integrates counts under a uniform-within-site assumption and
provides conservative lower/upper allocation bounds for partial ownership areas.
Bounds can be broad; they must not be hidden by an equivalent-count midpoint.

sampled_run_support counts elapsed saved-state spans and resets on gaps. It is
not continuous residence, and changing molecular identity need not change spatial
occupation. Gas and water occupancy are separate caller-provided state fields.

height_velocity_moments preserves particle counts and all three first/second
velocity moments. Bins are half-open, including the last bin; upper-edge samples
are excluded and counted. Empty bins have undefined means. Callers must declare
surface-relative versus laboratory height, selected species, spatial masks,
frame weighting and time-block support. A slow near-wall mean does not prove
adsorption, and thermal second moments are not flow speed or a friction force.
