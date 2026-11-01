# Whole-gas nearest-solid distribution

`gas_solid_distances` reuses the existing N2 center, largest-component and
partial-periodic nearest-neighbor implementations. It accepts PPP or PPF boxes;
nonperiodic z cannot create molecule bonds, cluster edges or nearest-solid images.
The caller owns atom identity and solid-membership qualification.

For each N2, take the smaller of its two N-to-solid-heavy nearest distances.
`time_mean_cdf` normalizes molecule counts separately in each frame and then
averages frames equally. Do not pool molecules across frames when cluster size
changes. Retain the full distance and membership arrays; a display grid is not
the empirical dataset. Equal frame weighting assumes a regular saved cadence.

The distribution characterizes configurations; it is not a desorption lifetime,
free-energy curve, water-film classifier or causal attribution. Block differences
are descriptive and do not supply independent-replica confidence intervals.
