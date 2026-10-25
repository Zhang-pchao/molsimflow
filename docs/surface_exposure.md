# Surface probe accessibility

`molsimflow.postprocess.surface_exposure` adds three array kernels for an explicitly
selected slab face: `probe_accessibility`, `local_cover_depth`, and
`local_label_fraction`. Reuse existing LAMMPS readers, functional-site identity,
current hydrogen ownership and slab normals in the caller.

The upward probe at a site plus 2.5 A uses heavy-atom 2 A and bound-H 1 A exclusion
spheres. This preserves an existing geometric screening definition while exposing
its parameters and periodic-boundary contract. Additional explicit unit directions
provide shielding sensitivity; fractions over directions are not SASA, insertion
probability, adsorption energy or a wettability rank. Coordinate units are angstroms.

Current solid-bound hydrogen must be supplied by a checked ownership procedure.
Do not include every original surface H after exchange, and do not silently relabel
original CH3/OH identities as current chemistry. Empty H clouds are accepted;
missing sites or solid heavy atoms are rejected. Site/material selection, face,
probe radii, offsets and lateral neighborhoods remain explicit caller choices.

Local cover is a positive height difference within a periodic XY disk. A low site
can remain accessible from oblique directions; height alone does not prove burial.
Local label fractions include the query site and report denominator counts. Pass
one face at a time and preserve original material identity across a seam.

Synthetic tests check shielding, bound-H exclusion, periodic translation,
nonperiodic-Z behavior, directional contrast, neighborhood counts and invalid
contracts. Scientific interpretations require independent structural and hydration
evidence; a migration plateau does not establish reversed wettability.
