# Surface contact process helpers

`surface_contact_process` provides periodic XY capsule membership, distinct molecule/union-area bookkeeping and source-aware sampled episodes. A finite footprint is a geometric convention, not an interaction-energy or electron-density criterion.

`capsule_membership(points_xy, bonds_xy, box_xy, radius)` returns distinct point/molecule incidences for short bonded molecules. The helper has no Z boundary assumption. Callers must respect their source-specific Z boundary when deriving 3D clusters or distances.

`supported_cell_counts` keeps unique molecular counts, union point samples and normalized molecular weights separate. Each contributing molecule has total weight one over the declared sampling domain, regardless of the number of occupied cells.

`sampled_episodes(states, times, segments, expected_dt)` treats state 1 as present, 0 as observed absence and -1 as unknown. It censors gaps, unknowns, source changes and observation limits. Molecular IDs are deliberately absent: continuous spatial support may coexist with complete identity turnover.

`establishment_outcome` accepts an independently defined supply origin and a separate contact trace. Confirmation occurs only after the requested sampled span; it is never backdated. A 20-unit span at 10-unit cadence requires three observations. Initially occupied states are not at risk for a new establishment, and incomplete or unknown windows do not become negative outcomes.

These helpers do not infer continuous-time bond lifetimes, independent events, chemical causation, or stationary behavior. Physical cutoffs, site chemistry, opportunity cohorts and uncertainty must be supplied and validated by the caller.
