# Origin-conditioned particle transport

`interfacial_contact_renewal.origin_conditioned_transport` accumulates endpoint
state transitions and displacement moments for identical ordered particle IDs.
Supply already unwrapped coordinates and integer disjoint state labels in
`0 .. state_count-1`; reserve state 0 for outside. No particle is dropped for
leaving a region. IDs must be unique nonnegative integers and match at both
endpoints; automatic identity matching or periodic unwrapping is not performed.

The returned `counts` has shape K by K, with origin then destination axes.
`origin_support` and `destination_support` are the row and column totals.
The four arrays `lab_displacement_sum`, `lab_displacement_squared_sum`,
`relative_displacement_sum`, and `relative_displacement_squared_sum` have
shape K by K by 3. Squares are componentwise second moments, not a covariance
matrix or a squared net displacement. Sum destination axis 1 to retain all
initial members when computing origin-conditioned statistics. Divide only
where the corresponding support is nonzero; zero support means undefined mean.

For lab displacement d and supplied frame displacement b, the relative
displacement is d-b. A stationary particle can have nonzero relative
displacement when the frame moves. Lab and relative results should be
reported together; moving-state membership changes do not alone establish
particle transport. The caller records how its frame origin is defined,
handles native restart joins, and excludes intervals crossing missing frames,
identity discontinuities, velocity reseeding, or incompatible protocols.

In a fixed orthorhombic box, use saved image flags to construct unwrapped
coordinates before calling this function. Inferring a long endpoint
displacement from the minimum image can lose whole-box crossings. A bubble
geometric-center displacement is distinct from integrating an instantaneous
mean velocity of a cluster whose membership changes.

These are sampled-endpoint statistics, not continuous residence, a resolved
crossing flux, or evidence of closed circulation. For sampled multi-frame
return versus survival and gas rotation diagnostics, see
[gas kinematics](interfacial-gas-kinematics.md). Reuse
`coarse_velocity_field` for explicitly defined projections/slices, preserve
lab-frame and reference-relative velocities, and keep block variability
separate from independent-replicate uncertainty.
