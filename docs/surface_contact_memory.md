# Surface contact memory

The surface_contact_memory module supports one observable: endpoint recontact
at the same material locations, averaged over all eligible time origins.

For support sets C(t), the curve is the time-origin mean of
|C(t) intersection C(t+tau)| / |C(t)|. Empty starts are omitted, empty endpoints
contribute zero. Leaving and returning counts as endpoint recontact. This is
not uninterrupted residence or survival. Each row includes its origin count.

supported_material_regions uses a declared periodic orthorhombic geometry,
precomputed nearest-solid distances and local heights. It requires locally
connected molecules from the largest gas cluster. Region radii and contact
shells are explicit. A region may overlap another; region counts are not areas
or independent statistical observations. Frame IDs, surface registration,
boundary policy and source provenance belong to the project adapter.

region_recontact_curve censors every pair crossing a source seam, missing
cadence or invalid observation. An epoch label reused after a break does not
reconnect histories. Temporal origins and blocks do not create independent
physical replicas. No equilibrium or causal interpretation is built in.
